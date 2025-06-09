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
            evidence=[evidence],
            confidence_breakdown={
                "overall_confidence": 0.9,
                "evidence_count": 1,
                "source_diversity": 0.8,
                "authority_score": 0.8,
                "confidence_level": "high"
            }
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
        """Test database performance with indexing"""
        db_manager = EnhancedDatabaseManager(
            db_path=self.temp_db_path,
            enable_json_export=False
        )
        
        # Check database indices exist for performance
        conn = sqlite3.connect(self.temp_db_path)
        cursor = conn.cursor()
        
        cursor.execute("SELECT name FROM sqlite_master WHERE type='index';")
        indices = [row[0] for row in cursor.fetchall()]
        
        # Should have performance indices
        expected_indices = [
            "idx_evidence_destination", "idx_themes_destination", 
            "idx_theme_evidence_theme", "idx_authentic_insights_theme"
        ]
        
        for expected_idx in expected_indices:
            assert expected_idx in indices, f"Missing performance index: {expected_idx}"
        
        conn.close()

    def test_json_export_caching(self):
        """Test JSON export file caching"""
        db_manager = EnhancedDatabaseManager(
            db_path=self.temp_db_path,
            enable_json_export=True,
            json_export_path=self.temp_export_dir
        )
        
        # Create minimal destination
        destination = Destination(
            id="export_test",
            names=["Export Test"],
            admin_levels={"country": "Test"},
            timezone="UTC",
            country_code="ET",
            themes=[]
        )
        
        # Store destination - should trigger JSON export
        results = db_manager.store_destination(destination)
        
        assert results["json_exported"] is True
        assert "json_export_path" in results
        
        # Verify JSON file exists
        json_path = results["json_export_path"]
        assert os.path.exists(json_path)
        
        # Verify JSON content
        with open(json_path, 'r') as f:
            exported_data = json.load(f)
        
        assert exported_data["destination"]["id"] == "export_test"


class TestCacheIntegration:
    """Test integration between different cache layers"""
    
    def setup_method(self):
        """Setup integrated test environment"""
        self.temp_cache_dir = tempfile.mkdtemp()
        self.temp_db_dir = tempfile.mkdtemp()
        self.temp_sqlite_path = tempfile.mktemp(suffix='.db')
        
        # Patch cache directory
        import src.caching
        self.original_cache_dir = src.caching.CACHE_DIR
        src.caching.CACHE_DIR = self.temp_cache_dir
        
    def teardown_method(self):
        """Cleanup test environment"""
        shutil.rmtree(self.temp_cache_dir, ignore_errors=True)
        shutil.rmtree(self.temp_db_dir, ignore_errors=True)
        if os.path.exists(self.temp_sqlite_path):
            os.remove(self.temp_sqlite_path)
        
        # Restore cache directory
        import src.caching
        src.caching.CACHE_DIR = self.original_cache_dir

    def test_cache_layer_coordination(self):
        """Test coordination between file cache, ChromaDB, and SQLite"""
        from src.schemas import ProcessedPageChunk
        
        # Initialize all cache layers
        chroma_manager = ChromaDBManager(
            db_path=self.temp_db_dir,
            collection_name="integration_test"
        )
        
        db_manager = EnhancedDatabaseManager(
            db_path=self.temp_sqlite_path,
            enable_json_export=False
        )
        
        # Test data flow through layers
        # 1. Store raw data in file cache
        raw_data = {"source": "web", "content": "Test content about Bali"}
        write_to_cache(["raw", "bali_data"], raw_data)
        
        # 2. Process and store in ChromaDB
        chunk = ProcessedPageChunk(
            chunk_id="integration_chunk",
            url="https://example.com/integration",
            title="Integration Test",
            text_chunk="Test content about Bali for integration testing.",
            chunk_order=0,
            metadata={"test": "integration"}
        )
        chroma_manager.add_chunks([chunk])
        
        # 3. Store structured data in SQLite
        evidence = Evidence(
            id="integration_evidence",
            source_url="https://example.com/integration",
            source_category=SourceCategory.BLOG,
            evidence_type=EvidenceType.SECONDARY,
            authority_weight=0.7,
            text_snippet="Integration test evidence",
            timestamp=datetime.now(),
            confidence=0.8,
            sentiment=0.6,
            cultural_context={"integration": True},
            relationships=[],
            agent_id="integration_agent"
        )
        
        theme = Theme(
            theme_id="integration_theme",
            macro_category="Test",
            micro_category="Integration",
            name="Integration Test Theme",
            description="Theme for testing integration",
            fit_score=0.8,
            evidence=[evidence]
        )
        
        destination = Destination(
            id="integration_destination",
            names=["Integration Test"],
            admin_levels={"country": "Test"},
            timezone="UTC",
            country_code="IT",
            themes=[theme]
        )
        
        db_results = db_manager.store_destination(destination)
        
        # Verify data exists in all layers
        # File cache
        cached_raw = read_from_cache(["raw", "bali_data"], 1)
        assert cached_raw == raw_data
        
        # ChromaDB
        search_results = chroma_manager.search(["Bali integration"], n_results=1)
        assert len(search_results[0]) == 1
        assert search_results[0][0].document_chunk.chunk_id == "integration_chunk"
        
        # SQLite
        assert db_results["database_stored"] is True

    def test_cache_consistency_across_layers(self):
        """Test data consistency across different cache layers"""
        # This test ensures that the same source data produces consistent
        # results across different cache and storage layers
        
        source_url = "https://example.com/consistency"
        source_content = "Consistent test content about destination features."
        
        # Store in file cache with timestamp
        file_cache_data = {
            "url": source_url,
            "content": source_content,
            "processed_at": datetime.now().isoformat()
        }
        write_to_cache(["consistency", "test"], file_cache_data)
        
        # Retrieve and verify consistency
        retrieved_data = read_from_cache(["consistency", "test"], 1)
        assert retrieved_data["url"] == source_url
        assert retrieved_data["content"] == source_content
        
        # Verify timestamp handling
        assert "processed_at" in retrieved_data
        processed_time = datetime.fromisoformat(retrieved_data["processed_at"])
        assert (datetime.now() - processed_time).seconds < 10  # Recent timestamp


if __name__ == "__main__":
    # Run tests with verbose output
    pytest.main([__file__, "-v", "-s"]) 