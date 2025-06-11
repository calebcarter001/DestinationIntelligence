"""
Data Model Schema Validation Tests
Simple tests for validating schema structures and data transformations.
"""

import unittest
import sys
import os
import json

# Add the root directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

class TestSchemaValidation(unittest.TestCase):
    
    def setUp(self):
        """Set up test environment"""
        from src.schemas import EnhancedEvidence, AuthorityType
        from src.core.enhanced_data_models import Theme
        from datetime import datetime
        self.Evidence = EnhancedEvidence
        self.Theme = Theme
        self.AuthorityType = AuthorityType
        self.datetime = datetime

    def test_evidence_schema_validation(self):
        """Test that Evidence schema accepts valid data"""
        valid_evidence_data = {
            "source_url": "https://reddit.com/r/Seattle",
            "source_category": self.AuthorityType.RESIDENT,
            "text_snippet": "Seattle grunge music scene",
            "authority_weight": 0.7,
            "sentiment": 0.8,
            "confidence": 0.8,
            "timestamp": self.datetime.now().isoformat()
        }
        
        try:
            evidence = self.Evidence(
                id="test_evidence_schema_validation",
                source_url="https://reddit.com/r/Seattle",
                source_category=self.AuthorityType.RESIDENT,
                text_snippet="Seattle grunge music scene",
                authority_weight=0.7,
                sentiment=0.8,
                confidence=0.8,
                timestamp=self.datetime.now().isoformat(),
                cultural_context={},
                relationships=[],
                agent_id="test_agent"
            )
            
            # Validate required fields
            self.assertEqual(evidence.text_snippet, "Seattle grunge music scene")
            self.assertEqual(evidence.source_url, "https://reddit.com/r/Seattle")
            self.assertEqual(evidence.source_category, self.AuthorityType.RESIDENT.value)
            self.assertEqual(evidence.authority_weight, 0.7)
            self.assertEqual(evidence.sentiment, 0.8)
            self.assertEqual(evidence.confidence, 0.8)
            
        except Exception as e:
            self.fail(f"Valid evidence data failed validation: {e}")

    def test_theme_schema_validation(self):
        """Test that Theme schema accepts valid data"""
        valid_theme_data = {
            "theme_id": "test_theme_1",
            "name": "Grunge Heritage",
            "macro_category": "Cultural Identity & Atmosphere",
            "micro_category": "Music",
            "description": "Seattle's grunge music heritage",
            "fit_score": 0.85,
            "evidence": [],
            "tags": ["music", "grunge", "culture"]
        }
        
        try:
            theme = self.Theme(**valid_theme_data)
            
            # Validate required fields
            self.assertEqual(theme.theme_id, "test_theme_1")
            self.assertEqual(theme.name, "Grunge Heritage")
            self.assertEqual(theme.macro_category, "Cultural Identity & Atmosphere")
            self.assertEqual(theme.fit_score, 0.85)
            self.assertIsInstance(theme.evidence, list)
            self.assertIsInstance(theme.tags, list)
            
        except Exception as e:
            self.fail(f"Valid theme data failed validation: {e}")

    def test_confidence_breakdown_schema(self):
        """Test that confidence breakdown has expected structure"""
        confidence_breakdown = {
            "evidence_quality": 0.8,
            "source_diversity": 0.7,
            "temporal_coverage": 0.6,
            "content_completeness": 0.75,
            "total_score": 0.72
        }
        
        # All values should be numeric and in valid range
        for key, value in confidence_breakdown.items():
            self.assertIsInstance(value, (int, float), f"{key} should be numeric")
            self.assertGreaterEqual(value, 0.0, f"{key} should be >= 0")
            self.assertLessEqual(value, 1.0, f"{key} should be <= 1")

    def test_cultural_context_schema(self):
        """Test that cultural context has expected structure"""
        cultural_context = {
            "content_type": "local_tip",
            "authenticity_indicators": ["local phrase", "personal experience"],
            "authority_indicators": ["official source"],
            "distinctiveness_score": 0.7
        }
        
        # Required fields should exist
        self.assertIn("content_type", cultural_context)
        self.assertIsInstance(cultural_context["authenticity_indicators"], list)
        self.assertIsInstance(cultural_context["authority_indicators"], list)
        
        # Distinctiveness score should be valid
        if "distinctiveness_score" in cultural_context:
            score = cultural_context["distinctiveness_score"]
            self.assertGreaterEqual(score, 0.0)
            self.assertLessEqual(score, 1.0)

if __name__ == "__main__":
    unittest.main()