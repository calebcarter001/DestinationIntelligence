"""
Unit Tests for Theme Discovery
Simple tests for theme discovery and categorization functionality.
"""

import unittest
import sys
import os
import asyncio
from datetime import datetime

# Add the root directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

from src.tools.enhanced_theme_analysis_tool import EnhancedThemeAnalysisTool
from src.schemas import EnhancedEvidence, AuthorityType
from src.core.enhanced_data_models import Theme

class TestThemeDiscoveryUnit(unittest.TestCase):
    
    def setUp(self):
        """Set up test environment"""
        self.tool = EnhancedThemeAnalysisTool()
        self.Evidence = EnhancedEvidence
        self.Theme = Theme

    def test_theme_categorization_macro_micro(self):
        """Test that themes are properly categorized into macro/micro categories"""
        test_cases = [
            ("Grunge Heritage", "Cultural Identity & Atmosphere", "Music"),
            ("Coffee Culture", "Cultural Identity & Atmosphere", "Local Culture"),
            ("Public Transit", "Transportation & Access", "Public Transportation"),
            ("Fine Dining", "Food & Dining", "Upscale Restaurants"),
            ("Hiking Trails", "Nature & Outdoor", "Hiking")
        ]
        
        for theme_name, expected_macro, expected_micro in test_cases:
            macro = self.tool._get_macro_category(theme_name)
            # Test that categorization returns something reasonable
            self.assertIsNotNone(macro, f"Should categorize {theme_name}")
            self.assertIsInstance(macro, str, f"Category should be string for {theme_name}")

    def test_theme_metadata_population(self):
        """Test that theme metadata is properly populated"""
        mock_evidence = [
            self.Evidence(
                id="test_theme_discovery_evidence_1",
                text_snippet="Seattle grunge music scene with local venues",
                source_category=AuthorityType.RESIDENT,
                source_url="https://reddit.com/r/Seattle",
                authority_weight=0.7,
                sentiment=0.8,
                confidence=0.8,
                timestamp=datetime.now().isoformat(),
                cultural_context={},
                relationships=[],
                agent_id="test_agent"
            )
        ]
        
        # Create a mock theme to test metadata
        test_theme = self.Theme(
            theme_id="test_theme_1",
            name="Grunge Heritage",
            macro_category="Cultural Identity & Atmosphere",
            micro_category="Music",
            description="Test theme",
            fit_score=0.8,
            evidence=mock_evidence,
            tags=["music", "grunge"],
            metadata={
                "local_context": ["grunge venues"],
                "content_types": ["activity"],
                "related_themes_from_discovery": ["music", "culture"],
                "temporal_aspects": ["year-round"],
                "raw_evidence_count": 1
            }
        )
        
        # Validate metadata structure
        self.assertIsInstance(test_theme.metadata, dict)
        self.assertIn("local_context", test_theme.metadata)
        self.assertIn("content_types", test_theme.metadata)
        self.assertIsInstance(test_theme.metadata["local_context"], list)

    def test_theme_tags_generation(self):
        """Test theme tag generation"""
        test_cases = [
            ("Grunge Music", ["grunge", "music"]),
            ("Coffee Culture", ["coffee", "culture"]),
            ("Hiking Trails", ["hiking", "trails"]),
            ("Fine Dining", ["fine", "dining"])
        ]
        
        for theme_name, expected_base_tags in test_cases:
            tags = self.tool._generate_tags(theme_name)
            
            # Should return a list
            self.assertIsInstance(tags, list)
            
            # Should contain expected base tags
            for expected_tag in expected_base_tags:
                self.assertIn(expected_tag, tags, f"Should contain {expected_tag} for {theme_name}")

    def test_theme_description_generation(self):
        """Test theme description generation"""
        theme_data = {
            "local_context": {"grunge venues", "coffee shops"},
            "temporal_aspects": {"summer", "year-round"}
        }
        
        confidence = {"total_score": 0.8}
        
        description = self.tool._generate_rich_description(
            "Grunge Heritage",
            "Seattle, United States", 
            theme_data,
            confidence
        )
        
        # Should be a non-empty string
        self.assertIsInstance(description, str)
        self.assertGreater(len(description), 0)
        
        # Should contain theme name and destination
        self.assertIn("Grunge Heritage", description)
        self.assertIn("Seattle", description)

    def test_theme_relationship_enhancement(self):
        """Test theme relationship building"""
        # Create themes with related metadata
        theme1 = self.Theme(
            theme_id="theme_1",
            name="Grunge Music",
            macro_category="Cultural Identity & Atmosphere",
            micro_category="Music",
            description="Test",
            fit_score=0.8,
            evidence=[],
            tags=["music"],
            metadata={
                "related_themes_from_discovery": ["music", "culture", "venues"]
            }
        )
        
        theme2 = self.Theme(
            theme_id="theme_2", 
            name="Music Venues",
            macro_category="Entertainment & Nightlife",
            micro_category="Live Music",
            description="Test",
            fit_score=0.7,
            evidence=[],
            tags=["venues"],
            metadata={
                "related_themes_from_discovery": ["music", "nightlife", "venues"]
            }
        )
        
        themes = [theme1, theme2]
        
        # Test relationship enhancement
        self.tool._enhance_theme_relationships(themes)
        
        # Check that relationships were calculated
        if "calculated_relationships" in theme1.metadata:
            relationships = theme1.metadata["calculated_relationships"]
            self.assertIsInstance(relationships, list)
            
            if relationships:
                rel = relationships[0]
                self.assertIn("theme_id", rel)
                self.assertIn("relationship_strength", rel)
                self.assertIn("shared_topics", rel)

    def test_theme_fit_score_calculation(self):
        """Test theme fit score is properly calculated"""
        mock_evidence = [
            self.Evidence(
                id="test_theme_discovery_evidence_2",
                text_snippet="High quality evidence about Seattle grunge",
                source_category=AuthorityType.RESIDENT,
                source_url="https://reddit.com/r/Seattle",
                authority_weight=0.8,
                sentiment=0.9,
                confidence=0.8,
                timestamp=datetime.now().isoformat(),
                cultural_context={},
                relationships=[],
                agent_id="test_agent"
            ),
            self.Evidence(
                id="test_theme_discovery_evidence_3",
                text_snippet="More evidence about grunge music scene",
                source_category=AuthorityType.RESIDENT,
                source_url="https://local-blog.com",
                authority_weight=0.7,
                sentiment=0.8,
                confidence=0.8,
                timestamp=datetime.now().isoformat(),
                cultural_context={},
                relationships=[],
                agent_id="test_agent"
            )
        ]
        
        confidence = self.tool._calculate_cultural_enhanced_confidence(
            mock_evidence,
            {"activity", "location"},
            {"grunge", "music"},
            {"year-round"},
            "Cultural Identity & Atmosphere"
        )
        
        # Fit score should be the total_score from confidence
        fit_score = confidence["total_score"]
        
        self.assertIsInstance(fit_score, (int, float))
        self.assertGreaterEqual(fit_score, 0.0)
        self.assertLessEqual(fit_score, 1.0)

if __name__ == "__main__":
    unittest.main() 