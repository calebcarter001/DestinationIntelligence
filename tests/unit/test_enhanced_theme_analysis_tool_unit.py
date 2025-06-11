"""
Unit Tests for Enhanced Theme Analysis Tool
Simple tests that would have caught the evidence_quality KeyError.
"""

import unittest
import sys
import os
from unittest.mock import Mock
from datetime import datetime
from src.tools.enhanced_theme_analysis_tool import EnhancedThemeAnalysisTool
from src.schemas import EnhancedEvidence, AuthorityType

# Add the root directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

class TestEnhancedThemeAnalysisToolUnit(unittest.TestCase):
    
    def setUp(self):
        """Set up test environment"""
        self.tool = EnhancedThemeAnalysisTool()
        self.Evidence = EnhancedEvidence

    def test_calculate_cultural_enhanced_confidence_returns_required_keys(self):
        """Test that cultural enhanced confidence returns all required keys"""
        mock_evidence = [
            self.Evidence(
                id="test_evidence_cultural_keys",
                source_url="https://example.com",
                source_category=AuthorityType.RESIDENT,
                authority_weight=0.8,
                text_snippet="Test evidence",
                sentiment=0.7,
                confidence=0.8,
                timestamp=datetime.now().isoformat(),
                cultural_context={},
                relationships=[],
                agent_id="test_agent"
            )
        ]
        
        confidence = self.tool._calculate_cultural_enhanced_confidence(
            mock_evidence, {"activity"}, {"test"}, {"summer"}, "Cultural Identity & Atmosphere"
        )
        
        required_keys = ["evidence_quality", "source_diversity", "temporal_coverage", "content_completeness"]
        for key in required_keys:
            self.assertIn(key, confidence, f"Missing key: {key}")

    def test_calculate_enhanced_confidence_fallback_compatibility(self):
        """Test that enhanced confidence fallback maintains compatibility"""
        mock_evidence = [
            self.Evidence(
                id="test_evidence_fallback",
                source_url="https://example.com",
                source_category=AuthorityType.RESIDENT,
                authority_weight=0.8,
                text_snippet="Test evidence",
                sentiment=0.7,
                confidence=0.8,
                timestamp=datetime.now().isoformat(),
                cultural_context={},
                relationships=[],
                agent_id="test_agent"
            )
        ]
        
        confidence = self.tool._calculate_enhanced_confidence(
            mock_evidence, {"activity"}, {"test"}, {"summer"}
        )
        
        # Should have all the same keys as cultural version
        required_keys = ["evidence_quality", "source_diversity", "temporal_coverage", "content_completeness"]
        for key in required_keys:
            self.assertIn(key, confidence, f"Missing key: {key}")

    def test_authenticity_score_calculation(self):
        """Test authenticity score calculation with reddit sources"""
        reddit_evidence = [
            self.Evidence(
                id="test_evidence_reddit",
                source_url="https://reddit.com/r/Seattle",
                source_category=AuthorityType.RESIDENT,
                authority_weight=0.6,
                text_snippet="As a local, I can tell you...",
                sentiment=0.8,
                confidence=0.8,
                timestamp=datetime.now().isoformat(),
                cultural_context={},
                relationships=[],
                agent_id="test_agent"
            )
        ]
        
        score = self.tool._calculate_authenticity_score(reddit_evidence)
        
        self.assertIsInstance(score, (int, float))
        self.assertGreaterEqual(score, 0.0)
        self.assertLessEqual(score, 1.0)

    def test_distinctiveness_score_calculation(self):
        """Test distinctiveness score calculation with unique keywords"""
        distinctive_evidence = [
            self.Evidence(
                id="test_evidence_distinctive",
                source_url="https://example.com",
                source_category=AuthorityType.RESIDENT,
                authority_weight=0.7,
                text_snippet="This unique venue is signature to Seattle only",
                sentiment=0.8,
                confidence=0.8,
                timestamp=datetime.now().isoformat(),
                cultural_context={},
                relationships=[],
                agent_id="test_agent"
            )
        ]
        
        score = self.tool._calculate_distinctiveness_score(distinctive_evidence)
        
        self.assertIsInstance(score, (int, float))
        self.assertGreaterEqual(score, 0.0)
        self.assertLessEqual(score, 1.0)

    def test_confidence_components_structure_validation(self):
        """Test that confidence components have correct structure"""
        mock_evidence = [
            self.Evidence(
                id="test_evidence_structure",
                source_url="https://example.com",
                source_category=AuthorityType.RESIDENT,
                authority_weight=0.8,
                text_snippet="Test evidence",
                sentiment=0.7,
                confidence=0.8,
                timestamp=datetime.now().isoformat(),
                cultural_context={},
                relationships=[],
                agent_id="test_agent"
            )
        ]
        
        confidence = self.tool._calculate_cultural_enhanced_confidence(
            mock_evidence, {"activity"}, {"test"}, {"summer"}, "Cultural Identity & Atmosphere"
        )
        
        # Check component structure
        for component in ["evidence_quality", "source_diversity", "temporal_coverage", "content_completeness"]:
            self.assertIn(component, confidence)
            self.assertIsInstance(confidence[component], (int, float))
            self.assertGreaterEqual(confidence[component], 0.0)
            self.assertLessEqual(confidence[component], 1.0)
        
        # Check total score
        self.assertIn("total_score", confidence)
        self.assertIsInstance(confidence["total_score"], (int, float))
        self.assertGreaterEqual(confidence["total_score"], 0.0)
        self.assertLessEqual(confidence["total_score"], 1.0)

    def test_processing_type_determination(self):
        """Test that macro categories are assigned correct processing types"""
        cultural_category = "Cultural Identity & Atmosphere"
        practical_category = "Transportation & Access"
        hybrid_category = "Food & Dining"
        
        self.assertEqual(self.tool._get_processing_type(cultural_category), "cultural")
        self.assertEqual(self.tool._get_processing_type(practical_category), "practical")
        self.assertEqual(self.tool._get_processing_type(hybrid_category), "hybrid")

if __name__ == "__main__":
    unittest.main() 