"""
Unit tests for Cultural Intelligence destination comparison functionality.
Tests personality detection, cultural similarity calculations, and enhanced comparison metrics.
"""

import unittest
from unittest.mock import Mock, patch, MagicMock
import sys
import os

# Add the root directory to the path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

# Import comparison functions
from compare_destinations import (
    get_processing_type, 
    load_destination_themes,
    calculate_cultural_intelligence_similarity,
    compare_destinations,
    CATEGORY_PROCESSING_RULES
)

class TestDestinationComparisonCultural(unittest.TestCase):
    """Test cultural intelligence features in destination comparison"""
    
    def setUp(self):
        """Set up test fixtures"""
        from src.schemas import AuthorityType
        from datetime import datetime
        self.sample_dest1_themes = {
            "themes_by_type": {
                "cultural": [
                    {"name": "Grunge Music Heritage", "confidence": 0.85, "macro_category": "Cultural Identity & Atmosphere"},
                    {"name": "Coffee Culture Origins", "confidence": 0.78, "macro_category": "Authentic Experiences"}
                ],
                "practical": [
                    {"name": "Public Transportation", "confidence": 0.90, "macro_category": "Transportation & Access"}
                ],
                "hybrid": [
                    {"name": "Food Scene", "confidence": 0.75, "macro_category": "Food & Dining"}
                ],
                "unknown": []
            },
            "theme_stats": {"cultural": 2, "practical": 1, "hybrid": 1, "unknown": 0},
            "avg_confidence": {"cultural": 0.815, "practical": 0.90, "hybrid": 0.75},
            "high_confidence_themes": {"cultural": 1, "practical": 1, "hybrid": 0},
            "total_themes": 4
        }
        
        self.sample_dest2_themes = {
            "themes_by_type": {
                "cultural": [
                    {"name": "Jazz Heritage", "confidence": 0.82, "macro_category": "Cultural Identity & Atmosphere"},
                ],
                "practical": [
                    {"name": "Public Transportation", "confidence": 0.88, "macro_category": "Transportation & Access"},
                    {"name": "Safety Measures", "confidence": 0.85, "macro_category": "Safety & Security"}
                ],
                "hybrid": [
                    {"name": "Food Scene", "confidence": 0.80, "macro_category": "Food & Dining"}
                ],
                "unknown": []
            },
            "theme_stats": {"cultural": 1, "practical": 2, "hybrid": 1, "unknown": 0},
            "avg_confidence": {"cultural": 0.82, "practical": 0.865, "hybrid": 0.80},
            "high_confidence_themes": {"cultural": 1, "practical": 2, "hybrid": 0},
            "total_themes": 4
        }

    def test_get_processing_type_all_categories(self):
        """Test processing type identification for all category types"""
        test_cases = [
            ("Cultural Identity & Atmosphere", "cultural"),
            ("Safety & Security", "practical"),
            ("Food & Dining", "hybrid"),
            ("Unknown Category", "unknown"),
            (None, "unknown"),
            ("", "unknown")
        ]
        
        for category, expected_type in test_cases:
            result = get_processing_type(category)
            self.assertEqual(result, expected_type, f"Failed for category: {category}")

    def test_category_processing_rules_structure(self):
        """Test that category processing rules are properly structured"""
        self.assertIn("cultural", CATEGORY_PROCESSING_RULES)
        self.assertIn("practical", CATEGORY_PROCESSING_RULES)
        self.assertIn("hybrid", CATEGORY_PROCESSING_RULES)
        
        for proc_type, rules in CATEGORY_PROCESSING_RULES.items():
            self.assertIn("categories", rules)
            self.assertIn("color", rules)
            self.assertIn("icon", rules)
            self.assertIn("weight", rules)
            self.assertIsInstance(rules["categories"], list)
            self.assertGreater(len(rules["categories"]), 0)

    def test_calculate_cultural_intelligence_similarity_theme_distribution(self):
        """Test theme distribution similarity calculation"""
        result = calculate_cultural_intelligence_similarity(self.sample_dest1_themes, self.sample_dest2_themes)
        
        self.assertIn("theme_distribution_similarity", result)
        theme_dist = result["theme_distribution_similarity"]
        
        # Check that all processing types are included
        for proc_type in ["cultural", "practical", "hybrid"]:
            self.assertIn(proc_type, theme_dist)
            self.assertGreaterEqual(theme_dist[proc_type], 0.0)
            self.assertLessEqual(theme_dist[proc_type], 1.0)

    def test_calculate_cultural_intelligence_similarity_confidence(self):
        """Test confidence similarity calculation"""
        result = calculate_cultural_intelligence_similarity(self.sample_dest1_themes, self.sample_dest2_themes)
        
        self.assertIn("confidence_similarity", result)
        conf_sim = result["confidence_similarity"]
        
        # Check confidence similarity for each type
        for proc_type in ["cultural", "practical", "hybrid"]:
            self.assertIn(proc_type, conf_sim)
            self.assertGreaterEqual(conf_sim[proc_type], 0.0)
            self.assertLessEqual(conf_sim[proc_type], 1.0)

    def test_calculate_cultural_intelligence_similarity_content(self):
        """Test content similarity calculation"""
        result = calculate_cultural_intelligence_similarity(self.sample_dest1_themes, self.sample_dest2_themes)
        
        self.assertIn("content_similarity", result)
        content_sim = result["content_similarity"]
        
        # Should find similarity in hybrid themes (both have "Food Scene")
        self.assertGreater(content_sim["hybrid"], 0.0, "Should find similarity in Food Scene")

    def test_calculate_cultural_intelligence_similarity_cultural_character(self):
        """Test cultural character analysis"""
        result = calculate_cultural_intelligence_similarity(self.sample_dest1_themes, self.sample_dest2_themes)
        
        self.assertIn("cultural_character", result)
        char_analysis = result["cultural_character"]
        
        # Check required fields
        required_fields = [
            "cultural_practical_ratio_similarity",
            "dest1_cultural_ratio", 
            "dest2_cultural_ratio",
            "dest1_personality",
            "dest2_personality"
        ]
        
        for field in required_fields:
            self.assertIn(field, char_analysis)

    def test_destination_personality_detection_cultural_focused(self):
        """Test detection of cultural-focused destinations"""
        # Create a destination with high cultural theme ratio
        cultural_focused_themes = {
            "themes_by_type": {
                "cultural": [
                    {"name": "Local Art Scene", "confidence": 0.8, "macro_category": "Cultural Identity & Atmosphere"},
                    {"name": "Music Heritage", "confidence": 0.85, "macro_category": "Authentic Experiences"},
                    {"name": "Historic District", "confidence": 0.78, "macro_category": "Cultural Identity & Atmosphere"},
                    {"name": "Local Festivals", "confidence": 0.82, "macro_category": "Distinctive Features"},
                    {"name": "Community Culture", "confidence": 0.75, "macro_category": "Local Character & Vibe"}
                ], 
                "practical": [{"name": "Transportation", "confidence": 0.85, "macro_category": "Transportation & Access"}], 
                "hybrid": [{"name": "Local Food", "confidence": 0.75, "macro_category": "Food & Dining"}], 
                "unknown": []
            },
            "theme_stats": {"cultural": 5, "practical": 1, "hybrid": 1, "unknown": 0},
            "avg_confidence": {"cultural": 0.8, "practical": 0.85, "hybrid": 0.75},
            "high_confidence_themes": {"cultural": 3, "practical": 1, "hybrid": 0},
            "total_themes": 7
        }
        
        result = calculate_cultural_intelligence_similarity(cultural_focused_themes, self.sample_dest2_themes)
        personality = result["cultural_character"]["dest1_personality"]
        
        self.assertEqual(personality, "cultural", "Should detect cultural-focused personality")

    def test_destination_personality_detection_practical_focused(self):
        """Test detection of practical-focused destinations"""
        # Create a destination with high practical theme ratio
        practical_focused_themes = {
            "themes_by_type": {
                "cultural": [{"name": "Local History", "confidence": 0.7, "macro_category": "Cultural Identity & Atmosphere"}], 
                "practical": [
                    {"name": "Public Transit", "confidence": 0.88, "macro_category": "Transportation & Access"},
                    {"name": "Safety Systems", "confidence": 0.90, "macro_category": "Safety & Security"},
                    {"name": "Healthcare", "confidence": 0.85, "macro_category": "Health & Medical"},
                    {"name": "Budget Options", "confidence": 0.87, "macro_category": "Budget & Costs"},
                    {"name": "Travel Planning", "confidence": 0.89, "macro_category": "Logistics & Planning"}
                ], 
                "hybrid": [{"name": "Restaurant Scene", "confidence": 0.75, "macro_category": "Food & Dining"}], 
                "unknown": []
            },
            "theme_stats": {"cultural": 1, "practical": 5, "hybrid": 1, "unknown": 0},
            "avg_confidence": {"cultural": 0.7, "practical": 0.88, "hybrid": 0.75},
            "high_confidence_themes": {"cultural": 0, "practical": 4, "hybrid": 0},
            "total_themes": 7
        }
        
        result = calculate_cultural_intelligence_similarity(practical_focused_themes, self.sample_dest2_themes)
        personality = result["cultural_character"]["dest1_personality"]
        
        self.assertEqual(personality, "practical", "Should detect practical-focused personality")

    def test_destination_personality_detection_balanced(self):
        """Test detection of balanced destinations"""
        # Create a destination with balanced theme distribution
        balanced_themes = {
            "themes_by_type": {
                "cultural": [
                    {"name": "Cultural Scene", "confidence": 0.75, "macro_category": "Cultural Identity & Atmosphere"},
                    {"name": "Local Art", "confidence": 0.75, "macro_category": "Artistic & Creative Scene"}
                ], 
                "practical": [
                    {"name": "Transportation", "confidence": 0.82, "macro_category": "Transportation & Access"},
                    {"name": "Safety", "confidence": 0.82, "macro_category": "Safety & Security"}
                ], 
                "hybrid": [
                    {"name": "Food Scene", "confidence": 0.78, "macro_category": "Food & Dining"},
                    {"name": "Entertainment", "confidence": 0.78, "macro_category": "Entertainment & Nightlife"},
                    {"name": "Nature Access", "confidence": 0.78, "macro_category": "Nature & Outdoor"}
                ], 
                "unknown": []
            },
            "theme_stats": {"cultural": 2, "practical": 2, "hybrid": 3, "unknown": 0},
            "avg_confidence": {"cultural": 0.75, "practical": 0.82, "hybrid": 0.78},
            "high_confidence_themes": {"cultural": 1, "practical": 1, "hybrid": 2},
            "total_themes": 7
        }
        
        result = calculate_cultural_intelligence_similarity(balanced_themes, self.sample_dest2_themes)
        personality = result["cultural_character"]["dest1_personality"]
        
        self.assertEqual(personality, "balanced", "Should detect balanced personality")

    def test_cultural_similarity_empty_themes(self):
        """Test handling of destinations with no themes"""
        empty_themes = {
            "themes_by_type": {"cultural": [], "practical": [], "hybrid": [], "unknown": []},
            "theme_stats": {"cultural": 0, "practical": 0, "hybrid": 0, "unknown": 0},
            "avg_confidence": {"cultural": 0.0, "practical": 0.0, "hybrid": 0.0},
            "high_confidence_themes": {"cultural": 0, "practical": 0, "hybrid": 0},
            "total_themes": 0
        }
        
        result = calculate_cultural_intelligence_similarity(empty_themes, self.sample_dest2_themes)
        
        # Should handle gracefully without errors
        self.assertIn("theme_distribution_similarity", result)
        self.assertIn("content_similarity", result)
        self.assertIn("cultural_character", result)

    @patch('compare_destinations.EnhancedDatabaseManager')
    def test_compare_destinations_with_cultural_intelligence(self, mock_db_manager):
        """Test full destination comparison with cultural intelligence"""
        # Mock destination objects
        mock_dest1 = Mock()
        mock_dest1.vibe_descriptors = ["urban", "creative"]
        mock_dest1.area_km2 = 369.2
        mock_dest1.primary_language = "English"
        mock_dest1.population = 750000
        mock_dest1.dominant_religions = ["Christianity"]
        mock_dest1.gdp_per_capita_usd = 50000
        mock_dest1.hdi = 0.9
        
        mock_dest2 = Mock()
        mock_dest2.vibe_descriptors = ["urban", "historic"]
        mock_dest2.area_km2 = 400.0
        mock_dest2.primary_language = "English"
        mock_dest2.population = 800000
        mock_dest2.dominant_religions = ["Christianity"]
        mock_dest2.gdp_per_capita_usd = 48000
        mock_dest2.hdi = 0.85
        
        result = compare_destinations(mock_dest1, mock_dest2, self.sample_dest1_themes, self.sample_dest2_themes)
        
        # Check that cultural intelligence is included
        self.assertIn("cultural_intelligence", result)
        self.assertIn("scores", result)
        self.assertIn("cultural_intelligence", result["scores"])
        
        # Check enhanced drivers
        self.assertIn("drivers", result)
        self.assertIn("cultural_insights", result["drivers"])
        
        cultural_insights = result["drivers"]["cultural_insights"]
        self.assertIn("most_similar_category", cultural_insights)
        self.assertIn("most_different_category", cultural_insights)
        self.assertIn("personality_match", cultural_insights)

    def test_cultural_intelligence_score_weights(self):
        """Test that cultural intelligence has proper weight in overall score"""
        # Mock minimal destination objects
        mock_dest1 = Mock()
        mock_dest1.vibe_descriptors = []
        mock_dest1.area_km2 = 100
        mock_dest1.primary_language = "English"
        mock_dest1.population = 100000
        mock_dest1.dominant_religions = []
        mock_dest1.gdp_per_capita_usd = 30000
        mock_dest1.hdi = 0.8
        
        mock_dest2 = Mock()
        mock_dest2.vibe_descriptors = []
        mock_dest2.area_km2 = 200
        mock_dest2.primary_language = "Spanish"
        mock_dest2.population = 200000
        mock_dest2.dominant_religions = []
        mock_dest2.gdp_per_capita_usd = 25000
        mock_dest2.hdi = 0.75
        
        result = compare_destinations(mock_dest1, mock_dest2, self.sample_dest1_themes, self.sample_dest2_themes)
        
        # Cultural intelligence should have significant impact on overall score
        overall_score = result["overall_similarity_score"]
        cultural_score = result["scores"]["cultural_intelligence"]
        
        self.assertGreaterEqual(overall_score, 0.0)
        self.assertLessEqual(overall_score, 1.0)
        self.assertGreaterEqual(cultural_score, 0.0)
        self.assertLessEqual(cultural_score, 1.0)

if __name__ == '__main__':
    unittest.main() 