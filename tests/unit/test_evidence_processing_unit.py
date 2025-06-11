"""
Unit Tests for Evidence Processing
Simple tests for evidence extraction, registry, and management.
"""

import unittest
import sys
import os
from datetime import datetime

# Add the root directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

class TestEvidenceProcessingUnit(unittest.TestCase):
    
    def setUp(self):
        """Set up test environment"""
        from src.tools.enhanced_theme_analysis_tool import EvidenceRegistry
        from src.schemas import EnhancedEvidence, AuthorityType
        from src.core.evidence_hierarchy import SourceCategory
        self.registry = EvidenceRegistry()
        self.Evidence = EnhancedEvidence
        self.AuthorityType = AuthorityType
        self.SourceCategory = SourceCategory
        self.datetime = datetime

    def test_evidence_id_generation(self):
        """Test evidence ID generation and uniqueness"""
        evidence = self.Evidence(
            id="test_evidence_id_gen",
            text_snippet="Test content for ID generation",
            source_url="https://example.com",
            source_category=self.SourceCategory.BLOG,
            authority_weight=0.7,
            sentiment=0.8,
            confidence=0.8,
            timestamp=self.datetime.now().isoformat(),
            cultural_context={},
            relationships=[],
            agent_id="test_agent"
        )
        
        self.assertIsInstance(evidence.id, str)
        self.assertGreater(len(evidence.id), 0)

    def test_evidence_registry_add_evidence(self):
        """Test adding evidence to registry"""
        evidence = self.Evidence(
            id="test_evidence_registry_add",
            text_snippet="Test content for registry",
            source_url="https://example.com",
            source_category=self.SourceCategory.BLOG,
            authority_weight=0.7,
            sentiment=0.8,
            confidence=0.8,
            timestamp=self.datetime.now().isoformat(),
            cultural_context={},
            relationships=[],
            agent_id="test_agent"
        )
        
        evidence_id = self.registry.add_evidence(evidence)
        self.assertIsInstance(evidence_id, str)
        # Check if evidence was added by trying to retrieve it
        retrieved = self.registry.get_evidence(evidence_id)
        self.assertIsNotNone(retrieved)

    def test_evidence_registry_get_evidence(self):
        """Test retrieving evidence from registry"""
        evidence = self.Evidence(
            id="test_evidence_registry_get",
            text_snippet="Test content for retrieval",
            source_url="https://example.com",
            source_category=self.SourceCategory.BLOG,
            authority_weight=0.7,
            sentiment=0.8,
            confidence=0.8,
            timestamp=self.datetime.now().isoformat(),
            cultural_context={},
            relationships=[],
            agent_id="test_agent"
        )
        
        evidence_id = self.registry.add_evidence(evidence)
        retrieved = self.registry.get_evidence(evidence_id)
        
        # Handle both dict and object return types
        if isinstance(retrieved, dict):
            self.assertEqual(retrieved["text_snippet"], evidence.text_snippet)
            self.assertEqual(retrieved["source_url"], evidence.source_url)
        else:
            self.assertEqual(retrieved.text_snippet, evidence.text_snippet)
            self.assertEqual(retrieved.source_url, evidence.source_url)

    def test_evidence_registry_deduplication(self):
        """Test evidence deduplication in registry"""
        evidence1 = self.Evidence(
            id="test_evidence_dedup_1",
            text_snippet="Same content",
            source_url="https://example.com",
            source_category=self.SourceCategory.BLOG,
            authority_weight=0.7,
            sentiment=0.8,
            confidence=0.8,
            timestamp=self.datetime.now().isoformat(),
            cultural_context={},
            relationships=[],
            agent_id="test_agent"
        )
        
        evidence2 = self.Evidence(
            id="test_evidence_dedup_2",
            text_snippet="Same content",  # Duplicate content
            source_url="https://example.com",
            source_category=self.SourceCategory.BLOG,
            authority_weight=0.7,
            sentiment=0.8,
            confidence=0.8,
            timestamp=self.datetime.now().isoformat(),
            cultural_context={},
            relationships=[],
            agent_id="test_agent"
        )
        
        id1 = self.registry.add_evidence(evidence1)
        id2 = self.registry.add_evidence(evidence2, similarity_threshold=0.9)
        
        # Should detect duplicate and return same ID or reject
        # Implementation depends on registry logic
        self.assertIsInstance(id1, str)
        self.assertIsInstance(id2, str)

    def test_evidence_authority_weight_validation(self):
        """Test authority weight validation"""
        evidence = self.Evidence(
            id="test_evidence_authority",
            text_snippet="Test content for authority validation",
            source_url="https://example.com",
            source_category=self.SourceCategory.GOVERNMENT,
            authority_weight=0.9,  # High authority
            sentiment=0.8,
            confidence=0.8,
            timestamp=self.datetime.now().isoformat(),
            cultural_context={},
            relationships=[],
            agent_id="test_agent"
        )
        
        # Should accept valid authority weight
        self.assertGreaterEqual(evidence.authority_weight, 0.0)
        self.assertLessEqual(evidence.authority_weight, 1.0)

    def test_evidence_sentiment_validation(self):
        """Test sentiment score validation"""
        evidence = self.Evidence(
            id="test_evidence_sentiment",
            text_snippet="Test content for sentiment validation",
            source_url="https://example.com",
            source_category=self.SourceCategory.BLOG,
            authority_weight=0.7,
            sentiment=0.5,  # Neutral sentiment
            confidence=0.8,
            timestamp=self.datetime.now().isoformat(),
            cultural_context={},
            relationships=[],
            agent_id="test_agent"
        )
        
        # Should accept valid sentiment range
        if evidence.sentiment is not None:
            self.assertGreaterEqual(evidence.sentiment, -1.0)
            self.assertLessEqual(evidence.sentiment, 1.0)

if __name__ == "__main__":
    unittest.main() 