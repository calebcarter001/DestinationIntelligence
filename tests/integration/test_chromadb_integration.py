"""
ChromaDB Integration Tests
Simple tests for vector storage and retrieval functionality.
"""

import unittest
import sys
import os
import tempfile
import asyncio

# Add the root directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

class TestChromaDBIntegration(unittest.TestCase):
    
    def setUp(self):
        """Set up test environment"""
        from src.tools.enhanced_theme_analysis_tool import EnhancedThemeAnalysisTool
        from src.schemas import EnhancedEvidence, AuthorityType
        from datetime import datetime
        
        self.tool = EnhancedThemeAnalysisTool()
        self.Evidence = EnhancedEvidence
        self.AuthorityType = AuthorityType
        self.datetime = datetime
        
        self.test_content = [
            {
                "url": "https://reddit.com/r/Seattle",
                "content": "Seattle grunge music scene with venues like The Crocodile. Local musicians continue the tradition.",
                "title": "Seattle Grunge Heritage"
            },
            {
                "url": "https://seattle.gov",
                "content": "Seattle public transportation includes buses, light rail, and ferry services for residents and visitors.",
                "title": "Seattle Transportation Guide"
            }
        ]

    def test_content_chunking_for_vectors(self):
        """Test that content is properly chunked for vector storage"""
        # Test content chunking
        for content_item in self.test_content:
            chunks = self.tool._smart_chunk_content(content_item["content"])
            
            # Should return a list of chunks
            self.assertIsInstance(chunks, list)
            
            if chunks:
                # Each chunk should have expected structure
                chunk = chunks[0]
                self.assertIsInstance(chunk, dict)
                self.assertIn("text", chunk)
                self.assertIn("context", chunk)
                self.assertIsInstance(chunk["context"], dict)
                self.assertIn("content_type", chunk["context"])

    def test_vector_storage_simulation(self):
        """Test vector storage simulation (without actual ChromaDB)"""
        # Simulate the vector storage process
        for content_item in self.test_content:
            # Test chunking
            chunks = self.tool._smart_chunk_content(content_item["content"])
            
            # Verify chunks are ready for vector storage
            for chunk in chunks:
                # Should have text content
                self.assertIn("text", chunk)
                self.assertIsInstance(chunk["text"], str)
                self.assertGreater(len(chunk["text"]), 0)
                                     
                # Should have metadata for vector storage
                self.assertIn("context", chunk)
                self.assertIsInstance(chunk["context"], dict)

    def test_theme_search_query_preparation(self):
        """Test preparation of search queries for theme discovery"""
        # Test search queries for different themes
        test_themes = ["grunge music", "coffee culture", "public transportation", "food scene"]
        
        for theme in test_themes:
            # This simulates what would be searched in ChromaDB
            # Test that theme matching works
            test_text = f"Seattle is known for its {theme} and local attractions"
            
            match_result = self.tool._check_theme_match(theme, test_text.lower())
            
            # Should be able to match theme in text
            self.assertIsInstance(match_result, bool)

    def test_evidence_retrieval_from_vectors(self):
        """Test evidence retrieval simulation from vector database"""
        # Create evidence that would be retrieved from vector search
        mock_evidence = [
            self.Evidence(
                id="test_evidence_chromadb_1",
                text_snippet="Seattle grunge music venues",
                source_url="https://reddit.com/r/Seattle",
                source_category=self.AuthorityType.RESIDENT,
                authority_weight=0.7,
                sentiment=0.8,
                confidence=0.8,
                timestamp=self.datetime.now().isoformat(),
                cultural_context={},
                relationships=[],
                agent_id="test_agent"
            ),
            self.Evidence(
                id="test_evidence_chromadb_2",
                text_snippet="Public transportation in Seattle",
                source_url="https://seattle.gov",
                source_category=self.AuthorityType.PROFESSIONAL,
                authority_weight=0.9,
                sentiment=0.6,
                confidence=0.9,
                timestamp=self.datetime.now().isoformat(),
                cultural_context={},
                relationships=[],
                agent_id="test_agent"
            )
        ]
        
        # Test evidence processing (simulates retrieval from vectors)
        for evidence in mock_evidence:
            # Verify evidence structure
            self.assertIsNotNone(evidence.text_snippet)
            self.assertIsNotNone(evidence.source_url)
            self.assertIsInstance(evidence.authority_weight, (int, float))
            
            # Test evidence categorization
            if hasattr(self.tool, '_classify_source_category'):
                category = self.tool._classify_source_category(
                    evidence.source_url, "", evidence.text_snippet
                )
                self.assertIsNotNone(category)

    def test_vector_similarity_scoring_simulation(self):
        """Test simulation of vector similarity scoring"""
        # Test similarity calculation between theme and evidence
        theme_name = "grunge music"
        evidence_texts = [
            "Seattle grunge music scene with local venues",  # High relevance
            "Public transportation and bus routes",          # Low relevance
            "Music venues and live entertainment"            # Medium relevance
        ]
        
        for evidence_text in evidence_texts:
            # Test theme relevance calculation
            relevance = self.tool._calculate_theme_relevance(theme_name, evidence_text)
            
            # Should return a score
            self.assertIsInstance(relevance, (int, float))
            self.assertGreaterEqual(relevance, 0.0)
            self.assertLessEqual(relevance, 1.0)

    def test_search_result_relevance_validation(self):
        """Test that search results would have proper relevance scoring"""
        # Mock search results that would come from ChromaDB
        search_results = [
            {
                "text": "Seattle grunge music started in the 1980s with bands like Soundgarden",
                "source": "music-history.com",
                "relevance": 0.9
            },
            {
                "text": "Coffee culture in Seattle began with local roasters",
                "source": "coffee-guide.com", 
                "relevance": 0.3  # Lower relevance for grunge query
            }
        ]
        
        # Test processing of search results
        for result in search_results:
            # Verify result structure
            self.assertIn("text", result)
            self.assertIn("source", result)
            self.assertIn("relevance", result)
            
            # Verify relevance score
            relevance = result["relevance"]
            self.assertIsInstance(relevance, (int, float))
            self.assertGreaterEqual(relevance, 0.0)
            self.assertLessEqual(relevance, 1.0)

    def test_vector_database_consistency(self):
        """Test that vector operations maintain data consistency"""
        # Test content processing consistency
        test_content = "Seattle grunge music scene and coffee culture"
        
        # Process content multiple times
        chunks1 = self.tool._smart_chunk_content(test_content)
        chunks2 = self.tool._smart_chunk_content(test_content)
        
        # Should produce consistent results
        self.assertEqual(len(chunks1), len(chunks2))
        
        if chunks1 and chunks2:
            # Content should be identical
            self.assertEqual(chunks1[0]["text"], chunks2[0]["text"])

if __name__ == "__main__":
    unittest.main() 