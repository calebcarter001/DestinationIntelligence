#!/usr/bin/env python3

import asyncio
import sys
import os
import json
import tempfile
import shutil
import sqlite3
from datetime import datetime, timedelta
from typing import Dict, Any, List
import pytest
import logging
from unittest.mock import Mock, patch, AsyncMock
import hashlib

# Add project root to path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.caching import read_from_cache, write_to_cache, get_cache_path, CACHE_EXPIRY_DAYS
from src.core.web_discovery_logic import WebDiscoveryLogic
from src.tools.chroma_interaction_tools import ChromaDBManager
from src.core.enhanced_database_manager import EnhancedDatabaseManager
from src.core.enhanced_data_models import Destination, Theme, Evidence
from src.core.evidence_hierarchy import SourceCategory, EvidenceType

class TestFileCaching:
    """Test the basic file-based caching system"""
    
    def setup_method(self):
        """Setup test environment with temporary cache directory"""
        self.temp_dir = tempfile.mkdtemp()
        self.original_cache_dir = os.environ.get('CACHE_DIR')
        os.environ['CACHE_DIR'] = self.temp_dir
        
        # Patch the cache module to use temp directory
        import src.caching
        self.original_cache_dir_var = src.caching.CACHE_DIR
        src.caching.CACHE_DIR = self.temp_dir
    
    def teardown_method(self):
        """Cleanup test environment"""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
        if self.original_cache_dir:
            os.environ['CACHE_DIR'] = self.original_cache_dir
        else:
            os.environ.pop('CACHE_DIR', None)
        
        # Restore original cache directory
        import src.caching
        src.caching.CACHE_DIR = self.original_cache_dir_var

    def test_cache_key_generation(self):
        """Test cache key generation and file path creation"""
        key_parts = ["test", "cache", "key"]
        cache_path = get_cache_path(key_parts)
        
        # Should create MD5 hash of sanitized key
        expected_key = "test_cache_key"
        expected_hash = hashlib.md5(expected_key.encode('utf-8')).hexdigest()
        
        assert cache_path.endswith(f"{expected_hash}.json")
        assert self.temp_dir in cache_path

    def test_cache_write_and_read(self):
        """Test basic cache write and read operations"""
        key_parts = ["test", "data"]
        test_data = {"message": "Hello, Cache!", "numbers": [1, 2, 3]}
        
        # Write to cache
        write_to_cache(key_parts, test_data)
        
        # Verify file exists
        cache_path = get_cache_path(key_parts)
        assert os.path.exists(cache_path)
        
        # Read from cache
        cached_data = read_from_cache(key_parts, expiry_days=1)
        assert cached_data == test_data

    def test_cache_expiry(self):
        """Test cache expiration functionality"""
        key_parts = ["expiry", "test"]
        test_data = {"expired": True}
        
        # Write to cache
        write_to_cache(key_parts, test_data)
        
        # Manually modify timestamp to simulate expired cache
        cache_path = get_cache_path(key_parts)
        with open(cache_path, 'r') as f:
            cache_content = json.load(f)
        
        # Set timestamp to 10 days ago
        old_timestamp = (datetime.now() - timedelta(days=10)).isoformat()
        cache_content["timestamp"] = old_timestamp
        
        with open(cache_path, 'w') as f:
            json.dump(cache_content, f)
        
        # Try to read with 7-day expiry - should return None
        cached_data = read_from_cache(key_parts, expiry_days=7)
        assert cached_data is None

    def test_cache_without_timestamp(self):
        """Test cache behavior with legacy data (no timestamp)"""
        key_parts = ["legacy", "data"]
        test_data = {"legacy": True}
        
        # Write cache file without timestamp (legacy format)
        cache_path = get_cache_path(key_parts)
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        with open(cache_path, 'w') as f:
            json.dump(test_data, f)
        
        # Should return the data directly
        cached_data = read_from_cache(key_parts, expiry_days=7)
        assert cached_data == test_data

    def test_cache_corruption_handling(self):
        """Test handling of corrupted cache files"""
        key_parts = ["corrupted", "cache"]
        
        # Create corrupted cache file
        cache_path = get_cache_path(key_parts)
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        with open(cache_path, 'w') as f:
            f.write("invalid json content {")
        
        # Should handle corruption gracefully
        cached_data = read_from_cache(key_parts, expiry_days=7)
        assert cached_data is None


class TestWebDiscoveryCaching:
    """Test web discovery caching integration"""
    
    def setup_method(self):
        """Setup test environment"""
        self.temp_dir = tempfile.mkdtemp()
        
        # Mock config
        self.config = {
            "web_discovery": {
                "search_results_per_query": 3,
                "min_content_length_chars": 100,
                "max_page_content_bytes": 1024 * 1024,  # 1MB
                "max_sources_for_agent_processing": 5
            },
            "caching": {
                "brave_search_expiry_days": 7,
                "page_content_expiry_days": 30
            },
            "priority_settings": {
                "enable_priority_discovery": False
            }
        }
        
        # Patch cache directory
        import src.caching
        self.original_cache_dir = src.caching.CACHE_DIR
        src.caching.CACHE_DIR = self.temp_dir
        
    def teardown_method(self):
        """Cleanup test environment"""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
        
        # Restore cache directory
        import src.caching
        src.caching.CACHE_DIR = self.original_cache_dir

    @pytest.mark.asyncio
    async def test_brave_search_caching(self):
        """Test Brave Search API result caching"""
        api_key = "test_api_key"
        
        async with WebDiscoveryLogic(api_key, self.config) as discovery:
            # Mock the actual API call with correct Brave API format
            mock_api_results = [
                {"url": "https://example1.com", "title": "Test 1", "description": "Content 1", "age": "1d", "language": "en"},
                {"url": "https://example2.com", "title": "Test 2", "description": "Content 2", "age": "2d", "language": "en"}
            ]
            
            # Expected normalized results
            expected_results = [
                {"url": "https://example1.com", "title": "Test 1", "snippet": "Content 1", "age": "1d", "language": "en"},
                {"url": "https://example2.com", "title": "Test 2", "snippet": "Content 2", "age": "2d", "language": "en"}
            ]
            
            with patch.object(discovery.session, 'get') as mock_get:
                mock_response = AsyncMock()
                mock_response.status = 200
                mock_response.json = AsyncMock(return_value={
                    "web": {"results": mock_api_results}
                })
                mock_get.return_value.__aenter__ = AsyncMock(return_value=mock_response)
                
                query = "test query"
                
                # First call - should hit API and cache
                results1 = await discovery._fetch_brave_search(query)
                assert len(results1) == 2
                assert mock_get.call_count == 1
                
                # Second call - should use cache
                results2 = await discovery._fetch_brave_search(query)
                assert results2 == results1
                assert mock_get.call_count == 1  # No additional API calls
                
                # Verify cache file exists with normalized results
                cache_key = ["brave_search", query]
                cached_data = read_from_cache(cache_key, 7)
                assert cached_data == expected_results

    @pytest.mark.asyncio
    async def test_page_content_caching(self):
        """Test web page content caching"""
        api_key = "test_api_key"
        
        async with WebDiscoveryLogic(api_key, self.config) as discovery:
            url = "https://example.com/test-page"
            mock_html = """
            <html>
                <body>
                    <article>
                        <h1>Test Article</h1>
                        <p>This is test content for caching verification.</p>
                        <p>Multiple paragraphs to ensure sufficient content length.</p>
                    </article>
                </body>
            </html>
            """
            
            # Expected extracted content (BeautifulSoup will extract text content)
            expected_content = "Test Article This is test content for caching verification. Multiple paragraphs to ensure sufficient content length."
            
            with patch.object(discovery.session, 'get') as mock_get:
                mock_response = AsyncMock()
                mock_response.status = 200
                mock_response.headers = {'content-type': 'text/html'}
                mock_response.content.read = AsyncMock(return_value=mock_html.encode('utf-8'))
                mock_get.return_value.__aenter__ = AsyncMock(return_value=mock_response)
                
                # First call - should fetch and cache
                content1 = await discovery._fetch_page_content(url)
                # Check that we get the extracted text content
                assert "test content for caching verification" in content1
                assert "Multiple paragraphs" in content1
                assert mock_get.call_count == 1
                
                # Second call - should use cache
                content2 = await discovery._fetch_page_content(url)
                assert content2 == content1
                assert mock_get.call_count == 1  # No additional fetches
                
                # Verify cache file exists
                cache_key = ["page_content_v2_bytes", url]
                cached_content = read_from_cache(cache_key, 30)
                assert cached_content == content1

    @pytest.mark.asyncio
    async def test_cache_performance_benefits(self):
        """Test that caching provides performance benefits"""
        api_key = "test_api_key"
        
        async with WebDiscoveryLogic(api_key, self.config) as discovery:
            # Pre-populate cache
            query = "performance test"
            cache_key = ["brave_search", query]
            mock_results = [{"url": "https://fast.com", "title": "Fast", "snippet": "Quick"}]
            write_to_cache(cache_key, mock_results)
            
            # Measure cache hit performance
            start_time = datetime.now()
            results = await discovery._fetch_brave_search(query)
            cache_duration = (datetime.now() - start_time).total_seconds()
            
            assert results == mock_results
            assert cache_duration < 0.1  # Should be very fast for cache hit


class TestChromaDBCaching:
    """Test ChromaDB vector database caching"""
    
    def setup_method(self):
        """Setup test ChromaDB environment"""
        self.temp_db_dir = tempfile.mkdtemp()
        self.collection_name = "test_collection"
    
    def teardown_method(self):
        """Cleanup test environment"""
        shutil.rmtree(self.temp_db_dir, ignore_errors=True)

    def test_chromadb_initialization(self):
        """Test ChromaDB manager initialization and persistence"""
        manager = ChromaDBManager(
            db_path=self.temp_db_dir,
            collection_name=self.collection_name
        )
        
        assert manager.client is not None
        assert manager.collection is not None
        assert manager.collection.name == self.collection_name
        
        # Verify persistence directory created
        assert os.path.exists(self.temp_db_dir)

    def test_chromadb_chunk_storage_and_retrieval(self):
        """Test storing and retrieving chunks from ChromaDB"""
        from src.schemas import ProcessedPageChunk
        
        manager = ChromaDBManager(
            db_path=self.temp_db_dir,
            collection_name=self.collection_name
        )
        
        # Create test chunks
        chunks = [
            ProcessedPageChunk(
                chunk_id="chunk_1",
                url="https://example.com/page1",
                title="Test Page 1",
                text_chunk="This is test content about Bali beaches and surfing.",
                chunk_order=0,
                metadata={"category": "travel", "location": "bali"}
            ),
            ProcessedPageChunk(
                chunk_id="chunk_2", 
                url="https://example.com/page2",
                title="Test Page 2",
                text_chunk="Information about Bali culture and traditional ceremonies.",
                chunk_order=0,
                metadata={"category": "culture", "location": "bali"}
            )
        ]
        
        # Add chunks to ChromaDB
        added_count = manager.add_chunks(chunks)
        assert added_count == 2
        
        # Search for similar content
        query_texts = ["Bali tourism activities"]
        search_results = manager.search(query_texts, n_results=2)
        
        assert len(search_results) == 1  # One result set for one query
        assert len(search_results[0]) == 2  # Two chunks returned
        
        # Verify search results contain expected content
        result_texts = [result.document_chunk.text_chunk for result in search_results[0]]
        assert any("beaches" in text for text in result_texts)
        assert any("culture" in text for text in result_texts)

    def test_chromadb_semantic_search_ranking(self):
        """Test semantic search ranking and similarity"""
        from src.schemas import ProcessedPageChunk
        
        manager = ChromaDBManager(
            db_path=self.temp_db_dir,
            collection_name=self.collection_name
        )
        
        # Add chunks with varying relevance
        chunks = [
            ProcessedPageChunk(
                chunk_id="beach_1",
                url="https://example.com/beaches",
                title="Bali Beaches",
                text_chunk="Beautiful beaches in Bali with crystal clear water and white sand. Perfect for surfing and swimming.",
                chunk_order=0,
                metadata={"topic": "beaches"}
            ),
            ProcessedPageChunk(
                chunk_id="temple_1",
                url="https://example.com/temples", 
                title="Bali Temples",
                text_chunk="Ancient temples in Bali showcase traditional Hindu architecture and spiritual practices.",
                chunk_order=0,
                metadata={"topic": "temples"}
            ),
            ProcessedPageChunk(
                chunk_id="food_1",
                url="https://example.com/food",
                title="Bali Cuisine",
                text_chunk="Traditional Balinese food includes nasi goreng, satay, and tropical fruits.",
                chunk_order=0,
                metadata={"topic": "food"}
            )
        ]
        
        manager.add_chunks(chunks)
        
        # Search for beach-related content
        beach_query = ["Bali beach activities swimming"]
        beach_results = manager.search(beach_query, n_results=3)
        
        # Beach content should be ranked highest
        top_result = beach_results[0][0]
        assert "beach" in top_result.document_chunk.text_chunk.lower()
        assert top_result.distance < beach_results[0][1].distance  # Lower distance = higher similarity

    def test_chromadb_persistence_across_sessions(self):
        """Test that ChromaDB data persists across manager instances"""
        from src.schemas import ProcessedPageChunk
        
        # First session - add data
        manager1 = ChromaDBManager(
            db_path=self.temp_db_dir,
            collection_name=self.collection_name
        )
        
        chunk = ProcessedPageChunk(
            chunk_id="persistent_chunk",
            url="https://example.com/persistent",
            title="Persistent Test",
            text_chunk="This chunk should persist across sessions.",
            chunk_order=0,
            metadata={"test": "persistence"}
        )
        
        manager1.add_chunks([chunk])
        
        # Second session - verify data exists
        manager2 = ChromaDBManager(
            db_path=self.temp_db_dir,
            collection_name=self.collection_name
        )
        
        search_results = manager2.search(["persistent chunk"], n_results=1)
        assert len(search_results[0]) == 1
        assert search_results[0][0].document_chunk.chunk_id == "persistent_chunk"


class TestDatabaseCaching:
    """Test enhanced database caching and storage"""
    
    def setup_method(self):
        """Setup test database environment"""
        self.temp_db_path = tempfile.mktemp(suffix='.db')
        self.temp_export_dir = tempfile.mkdtemp()
        
    def teardown_method(self):
        """Cleanup test environment"""
        if os.path.exists(self.temp_db_path):
            os.remove(self.temp_db_path)
        shutil.rmtree(self.temp_export_dir, ignore_errors=True)

    def test_database_storage_and_retrieval(self):
        """Test storing and retrieving destination data from database"""
        db_manager = EnhancedDatabaseManager(
            db_path=self.temp_db_path,
            enable_json_export=True,
            json_export_path=self.temp_export_dir
        )
        
        # Create test destination with themes and evidence
        from datetime import datetime
        
        evidence = Evidence(
            id="test_evidence",
            source_url="https://example.com/test",
            source_category=SourceCategory.GUIDEBOOK,
            evidence_type=EvidenceType.PRIMARY,
            authority_weight=0.8,
            text_snippet="Test evidence about destination attractions.",
            timestamp=datetime.now(),
            confidence=0.9,
            sentiment=0.7,
            cultural_context={"is_local_source": True},
            relationships=[],
            agent_id="test_agent"
        )
        
        theme = Theme(
            theme_id="test_theme",
            macro_category="Nature & Outdoor",
            micro_category="Beaches",
            name="Beautiful Beaches",
            description="Amazing beaches with clear water",
            fit_score=0.9,
            evidence=[evidence]
        )
        
        destination = Destination(
            id="test_destination",
            names=["Test Destination"],
            admin_levels={"country": "Test Country"},
            timezone="UTC",
            country_code="TC",
            themes=[theme]
        )
        
        # Store destination
        results = db_manager.store_destination(destination)
        
        assert results["database_stored"] is True
        assert "storage_summary" in results
        assert results["storage_summary"]["themes"] == 1
        assert results["storage_summary"]["evidence"] == 1

    def test_database_caching_performance(self):
        """Test database query performance with indices"""
        db_manager = EnhancedDatabaseManager(
            db_path=self.temp_db_path,
            enable_json_export=False
        )
        
        # Verify indices were created
        conn = sqlite3.connect(self.temp_db_path)
        cursor = conn.cursor()
        
        # Check for performance indices
        cursor.execute("SELECT name FROM sqlite_master WHERE type='index'")
        indices = [row[0] for row in cursor.fetchall()]
        
        expected_indices = [
            "idx_evidence_destination",
            "idx_evidence_confidence", 
            "idx_themes_destination",
            "idx_themes_category"
        ]
        
        for expected_idx in expected_indices:
            assert expected_idx in indices
        
        conn.close()

    def test_json_export_caching(self):
        """Test JSON export functionality and file caching"""
        db_manager = EnhancedDatabaseManager(
            db_path=self.temp_db_path,
            enable_json_export=True,
            json_export_path=self.temp_export_dir
        )
        
        # Create minimal destination for export test
        destination = Destination(
            id="export_test",
            names=["Export Test Destination"],
            admin_levels={"country": "Test"},
            timezone="UTC",
            country_code="ET"
        )
        
        # Store and export
        results = db_manager.store_destination(destination)
        
        assert results["database_stored"] is True
        assert "json_files_created" in results
        
        # Verify JSON files were created
        json_files = results["json_files_created"]
        for file_type, file_path in json_files.items():
            assert os.path.exists(file_path)
            
            # Verify file contains valid JSON
            with open(file_path, 'r') as f:
                data = json.load(f)
                assert data is not None


class TestCacheIntegration:
    """Test integration between different caching layers"""
    
    def setup_method(self):
        """Setup integrated test environment"""
        self.temp_dir = tempfile.mkdtemp()
        self.temp_db_dir = os.path.join(self.temp_dir, "chroma")
        self.temp_cache_dir = os.path.join(self.temp_dir, "cache")
        self.temp_db_path = os.path.join(self.temp_dir, "test.db")
        
        os.makedirs(self.temp_db_dir, exist_ok=True)
        os.makedirs(self.temp_cache_dir, exist_ok=True)
        
        # Patch cache directory
        import src.caching
        self.original_cache_dir = src.caching.CACHE_DIR
        src.caching.CACHE_DIR = self.temp_cache_dir
        
    def teardown_method(self):
        """Cleanup test environment"""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
        
        # Restore cache directory
        import src.caching
        src.caching.CACHE_DIR = self.original_cache_dir

    def test_cache_layer_coordination(self):
        """Test that different cache layers work together effectively"""
        # Test web cache -> vector cache -> database cache pipeline
        
        # 1. Simulate web content cache
        web_content_key = ["page_content_v2_bytes", "https://example.com/bali"]
        web_content = "Bali is famous for its beautiful beaches, temples, and rice terraces."
        write_to_cache(web_content_key, web_content)
        
        # 2. Process through vector cache
        from src.schemas import ProcessedPageChunk
        chroma_manager = ChromaDBManager(
            db_path=self.temp_db_dir,
            collection_name="integration_test"
        )
        
        chunk = ProcessedPageChunk(
            chunk_id="bali_chunk",
            url="https://example.com/bali",
            title="Bali Guide",
            text_chunk=web_content,
            chunk_order=0,
            metadata={"source": "web_cache"}
        )
        
        chroma_manager.add_chunks([chunk])
        
        # 3. Store in database cache
        db_manager = EnhancedDatabaseManager(
            db_path=self.temp_db_path,
            enable_json_export=False
        )
        
        destination = Destination(
            id="bali_integration",
            names=["Bali"],
            admin_levels={"country": "Indonesia"},
            timezone="Asia/Jakarta", 
            country_code="ID"
        )
        
        db_results = db_manager.store_destination(destination)
        
        # Verify all layers working
        assert read_from_cache(web_content_key, 30) == web_content
        
        search_results = chroma_manager.search(["Bali beaches"], n_results=1)
        assert len(search_results[0]) == 1
        
        assert db_results["database_stored"] is True

    def test_cache_consistency_across_layers(self):
        """Test data consistency across different caching layers"""
        destination_name = "Test Destination"
        content_url = f"https://example.com/{destination_name.lower().replace(' ', '-')}"
        
        # Store in web cache
        web_cache_key = ["page_content_v2_bytes", content_url]
        original_content = f"Information about {destination_name} attractions and culture."
        write_to_cache(web_cache_key, original_content)
        
        # Verify consistency
        cached_content = read_from_cache(web_cache_key, 30)
        assert cached_content == original_content
        
        # Test cache key collision handling
        similar_key = ["page_content_v2_bytes", content_url.upper()]
        different_content = "Different content for similar URL"
        write_to_cache(similar_key, different_content)
        
        # Should maintain separate cache entries
        assert read_from_cache(web_cache_key, 30) == original_content
        assert read_from_cache(similar_key, 30) == different_content


if __name__ == "__main__":
    # Run tests with verbose output
    pytest.main([__file__, "-v", "-s"]) 