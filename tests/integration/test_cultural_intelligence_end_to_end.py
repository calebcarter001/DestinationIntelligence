"""
Integration tests for Cultural Intelligence end-to-end functionality.
Tests the complete pipeline from configuration through analysis to comparison and visualization.
"""

import unittest
import tempfile
import os
import sys
import yaml
import json
import sqlite3
from unittest.mock import Mock, patch, MagicMock

# Add the root directory to the path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from src.core.enhanced_database_manager import EnhancedDatabaseManager
from src.core.enhanced_data_models import Destination

# Check if transformers/sentence-transformers are working properly
TRANSFORMERS_WORKING = True
TRANSFORMERS_ERROR = None
try:
    from src.tools.enhanced_theme_analysis_tool import EnhancedThemeAnalysisTool
except Exception as e:
    TRANSFORMERS_WORKING = False
    TRANSFORMERS_ERROR = str(e)

class TestCulturalIntelligenceEndToEnd(unittest.TestCase):
    """Integration tests for the complete cultural intelligence pipeline"""
    
    def setUp(self):
        """Set up test environment with temporary database and configuration"""
        from src.schemas import AuthorityType
        from datetime import datetime
        # Create temporary database
        self.temp_db = tempfile.NamedTemporaryFile(delete=False, suffix='.db')
        self.temp_db_path = self.temp_db.name
        self.temp_db.close()
        
        # Create temporary config file
        self.temp_config = tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.yaml')
        self.config_data = {
            'cultural_intelligence': {
                'enable_cultural_categories': True,
                'enable_authenticity_scoring': True,
                'enable_distinctiveness_filtering': True,
                'authentic_source_indicators': [
                    'reddit.com', 'local', 'community', 'blog', 'forum'
                ],
                'authoritative_source_indicators': [
                    'gov', 'edu', 'official', 'tourism'
                ],
                'distinctiveness_indicators': {
                    'unique_keywords': ['unique', 'distinctive', 'special', 'rare', 'authentic'],
                    'generic_keywords': ['popular', 'common', 'typical', 'standard', 'normal']
                },
                'category_processing_rules': {
                    'cultural': {
                        'confidence_threshold': 0.45,
                        'distinctiveness_threshold': 0.3
                    },
                    'practical': {
                        'confidence_threshold': 0.75,
                        'distinctiveness_threshold': 0.1
                    },
                    'hybrid': {
                        'confidence_threshold': 0.6,
                        'distinctiveness_threshold': 0.2
                    }
                }
            }
        }
        yaml.dump(self.config_data, self.temp_config)
        self.temp_config_path = self.temp_config.name
        self.temp_config.close()
        
        # Set up database manager
        self.db_manager = EnhancedDatabaseManager(db_path=self.temp_db_path)
        self.setup_test_database()
        
    def tearDown(self):
        """Clean up temporary files"""
        self.db_manager.close_db()
        os.unlink(self.temp_db_path)
        os.unlink(self.temp_config_path)
    
    def setup_test_database(self):
        """Set up test database with sample data"""
        # Create destination
        test_destination = Destination(
            id="dest_seattle_united_states",
            names=["Seattle, United States"],
            admin_levels={"country": "United States", "state": "Washington", "city": "Seattle"},
            timezone="America/Los_Angeles",
            population=750000,
            country_code="US",
            area_km2=369.2,
            primary_language="English",
            dominant_religions=["Christianity"],
            gdp_per_capita_usd=50000,
            hdi=0.92,
            vibe_descriptors=["urban", "creative", "tech-savvy", "coffee-culture"]
        )
        
        self.db_manager.store_destination(test_destination)
        
        # Add sample themes with cultural intelligence data
        sample_themes = [
            {
                "theme_id": "theme_grunge_heritage",
                "destination_id": "dest_seattle_united_states",
                "name": "Grunge Music Heritage",
                "macro_category": "Cultural Identity & Atmosphere",
                "micro_category": "Music Scene",
                "description": "Seattle's distinctive grunge music culture and heritage sites",
                "fit_score": 0.92,
                "confidence_level": "HIGH",
                "confidence_breakdown": '{"thematic_relevance": 0.95, "evidence_quality": 0.88, "authority": 0.70, "authenticity": 0.95}',
                "adjusted_overall_confidence": 0.87,
                "traveler_relevance_factor": 0.85,
                "tags": "music,culture,heritage,grunge,distinctive"
            },
            {
                "theme_id": "theme_coffee_culture",
                "destination_id": "dest_seattle_united_states", 
                "name": "Coffee Culture Origins",
                "macro_category": "Authentic Experiences",
                "micro_category": "Local Traditions",
                "description": "Birthplace of modern coffee culture with unique cafe experiences",
                "fit_score": 0.89,
                "confidence_level": "HIGH",
                "confidence_breakdown": '{"thematic_relevance": 0.90, "evidence_quality": 0.85, "authority": 0.75, "authenticity": 0.90}',
                "adjusted_overall_confidence": 0.83,
                "traveler_relevance_factor": 0.88,
                "tags": "coffee,culture,authentic,local,distinctive"
            },
            {
                "theme_id": "theme_public_transport",
                "destination_id": "dest_seattle_united_states",
                "name": "Public Transportation System",
                "macro_category": "Transportation & Access",
                "micro_category": "Public Transit",
                "description": "Comprehensive public transportation network",
                "fit_score": 0.85,
                "confidence_level": "HIGH", 
                "confidence_breakdown": '{"thematic_relevance": 0.88, "evidence_quality": 0.90, "authority": 0.95, "authenticity": 0.60}',
                "adjusted_overall_confidence": 0.86,
                "traveler_relevance_factor": 0.92,
                "tags": "transport,practical,public,system"
            },
            {
                "theme_id": "theme_food_scene",
                "destination_id": "dest_seattle_united_states",
                "name": "Local Food Scene",
                "macro_category": "Food & Dining",
                "micro_category": "Local Cuisine",
                "description": "Diverse local food scene with fresh seafood and farm-to-table",
                "fit_score": 0.81,
                "confidence_level": "MEDIUM-HIGH",
                "confidence_breakdown": '{"thematic_relevance": 0.85, "evidence_quality": 0.78, "authority": 0.80, "authenticity": 0.75}',
                "adjusted_overall_confidence": 0.78,
                "traveler_relevance_factor": 0.83,
                "tags": "food,dining,local,seafood,fresh"
            }
        ]
        
        # Insert themes into database
        cursor = self.db_manager.conn.cursor()
        for theme in sample_themes:
            cursor.execute("""
                INSERT INTO themes (
                    theme_id, destination_id, name, macro_category, micro_category,
                    description, fit_score, confidence_level, confidence_breakdown,
                    adjusted_overall_confidence, traveler_relevance_factor, tags
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                theme["theme_id"], theme["destination_id"], theme["name"],
                theme["macro_category"], theme["micro_category"], theme["description"],
                theme["fit_score"], theme["confidence_level"], theme["confidence_breakdown"],
                theme["adjusted_overall_confidence"], theme["traveler_relevance_factor"], theme["tags"]
            ))
        
        self.db_manager.conn.commit()

    def test_cultural_intelligence_configuration_loading(self):
        """Test that cultural intelligence configuration loads properly"""
        if not TRANSFORMERS_WORKING:
            self.skipTest(f"Transformers libraries not working: {TRANSFORMERS_ERROR}")
            
        # Test configuration loading by passing config directly to tool
        tool = EnhancedThemeAnalysisTool(config=self.config_data)
        
        # Verify configuration was loaded properly
        self.assertIsNotNone(tool.config)
        self.assertIn('cultural_intelligence', tool.config)
        
        # Verify cultural intelligence settings were extracted
        self.assertEqual(tool.cultural_config, self.config_data['cultural_intelligence'])
        self.assertTrue(tool.enable_dual_track)
        self.assertTrue(tool.enable_authenticity_scoring)
        self.assertTrue(tool.enable_distinctiveness)

    def test_theme_categorization_integration(self):
        """Test integration of theme categorization with database"""
        from compare_destinations import load_destination_themes, get_processing_type
        
        # Load themes from database
        themes_data = load_destination_themes(self.temp_db_path, "Seattle, United States")
        
        # Verify themes were loaded and categorized
        self.assertGreater(themes_data["total_themes"], 0)
        self.assertIn("cultural", themes_data["theme_stats"])
        self.assertIn("practical", themes_data["theme_stats"])
        self.assertIn("hybrid", themes_data["theme_stats"])
        
        # Verify specific categorizations
        cultural_themes = themes_data["themes_by_type"]["cultural"]
        practical_themes = themes_data["themes_by_type"]["practical"]
        hybrid_themes = themes_data["themes_by_type"]["hybrid"]
        
        # Should have cultural themes (Grunge Heritage, Coffee Culture)
        self.assertGreaterEqual(len(cultural_themes), 2)
        # Should have practical themes (Transportation)
        self.assertGreaterEqual(len(practical_themes), 1)
        # Should have hybrid themes (Food Scene)
        self.assertGreaterEqual(len(hybrid_themes), 1)

    def test_analyze_themes_script_integration(self):
        """Test integration with analyze_themes.py script functionality"""
        from analyze_themes import get_processing_type, calculate_cultural_intelligence_metrics
        
        # Load themes from database
        cursor = self.db_manager.conn.cursor()
        cursor.execute("""
            SELECT theme_id, name, macro_category, micro_category, description,
                   fit_score, confidence_level, adjusted_overall_confidence
            FROM themes
            WHERE destination_id = ?
        """, ("dest_seattle_united_states",))
        
        themes = cursor.fetchall()
        self.assertGreater(len(themes), 0)
        
        # Test processing type identification for each theme
        cultural_count = 0
        practical_count = 0
        hybrid_count = 0
        
        for theme in themes:
            macro_category = theme[2]
            processing_type = get_processing_type(macro_category)
            
            if processing_type == "cultural":
                cultural_count += 1
            elif processing_type == "practical":
                practical_count += 1
            elif processing_type == "hybrid":
                hybrid_count += 1
        
        # Verify we have themes in each category
        self.assertGreater(cultural_count, 0, "Should have cultural themes")
        self.assertGreater(practical_count, 0, "Should have practical themes")
        self.assertGreater(hybrid_count, 0, "Should have hybrid themes")
        
        # Test cultural intelligence metrics calculation
        themes_data = [
            {"processing_type": "cultural", "confidence": 0.87},
            {"processing_type": "cultural", "confidence": 0.83},
            {"processing_type": "practical", "confidence": 0.86},
            {"processing_type": "hybrid", "confidence": 0.78}
        ]
        
        metrics = calculate_cultural_intelligence_metrics(themes_data)
        
        self.assertIn("theme_distribution", metrics)
        self.assertIn("avg_confidence_by_type", metrics)
        self.assertIn("cultural_practical_ratio", metrics)

    def test_destination_comparison_integration(self):
        """Test integration of destination comparison with cultural intelligence"""
        from compare_destinations import (
            load_destination_themes, 
            calculate_cultural_intelligence_similarity,
            compare_destinations
        )
        
        # Store second test destination for comparison
        test_destination2 = Destination(
            id="test_dest_paris",
            names=["Paris, France"],
            admin_levels={"country": "France", "city": "Paris"},
            timezone="Europe/Paris",
            country_code="FR",
            themes=[]
        )
        self.db_manager.store_destination(test_destination2)
        
        # Load destination data
        seattle = self.db_manager.get_destination_by_name("Seattle, United States")
        paris = self.db_manager.get_destination_by_name("Paris, France")
        
        seattle_themes = load_destination_themes(self.temp_db_path, "Seattle, United States")
        paris_themes = load_destination_themes(self.temp_db_path, "Paris, France")
        
        # Test cultural intelligence similarity calculation
        ci_similarity = calculate_cultural_intelligence_similarity(seattle_themes, paris_themes)
        
        self.assertIn("theme_distribution_similarity", ci_similarity)
        self.assertIn("confidence_similarity", ci_similarity)
        self.assertIn("content_similarity", ci_similarity)
        self.assertIn("cultural_character", ci_similarity)
        
        # Test full destination comparison
        comparison_result = compare_destinations(seattle, paris, seattle_themes, paris_themes)
        
        self.assertIn("cultural_intelligence", comparison_result)
        self.assertIn("scores", comparison_result)
        self.assertIn("cultural_intelligence", comparison_result["scores"])
        self.assertIn("overall_similarity_score", comparison_result)
        
        # Verify cultural insights are included
        self.assertIn("drivers", comparison_result)
        self.assertIn("cultural_insights", comparison_result["drivers"])

    def test_dynamic_viewer_data_integration(self):
        """Test integration with dynamic viewer data generation"""
        from generate_dynamic_viewer import load_and_categorize_themes
        
        # Test loading and categorizing themes for viewer
        themes_data = load_and_categorize_themes(self.temp_db_path, "Seattle, United States")
        
        self.assertIsInstance(themes_data, list)
        self.assertGreater(len(themes_data), 0)
        
        # Verify each theme has required cultural intelligence fields
        for theme in themes_data:
            self.assertIn("processing_type", theme)
            self.assertIn("category_color", theme)
            self.assertIn("category_icon", theme)
            
            # Processing type should be valid
            self.assertIn(theme["processing_type"], ["cultural", "practical", "hybrid", "unknown"])

    def test_full_pipeline_consistency(self):
        """Test consistency across the full cultural intelligence pipeline"""
        from analyze_themes import get_processing_type as analyze_get_type
        from compare_destinations import get_processing_type as compare_get_type
        from generate_dynamic_viewer import get_processing_type as viewer_get_type
        
        # Test that all scripts use consistent categorization
        test_categories = [
            "Cultural Identity & Atmosphere",
            "Safety & Security", 
            "Food & Dining",
            "Unknown Category"
        ]
        
        for category in test_categories:
            analyze_result = analyze_get_type(category)
            compare_result = compare_get_type(category)
            viewer_result = viewer_get_type(category)
            
            # All should return the same processing type
            self.assertEqual(analyze_result, compare_result, f"Inconsistency in {category}")
            self.assertEqual(compare_result, viewer_result, f"Inconsistency in {category}")

    def test_error_handling_integration(self):
        """Test error handling across the cultural intelligence pipeline"""
        from compare_destinations import load_destination_themes
        
        # Test with non-existent destination
        empty_themes = load_destination_themes(self.temp_db_path, "Nonexistent City")
        
        self.assertEqual(empty_themes["total_themes"], 0)
        self.assertIn("cultural", empty_themes["theme_stats"])
        self.assertEqual(empty_themes["theme_stats"]["cultural"], 0)
        
        # Test with corrupted database path
        invalid_themes = load_destination_themes("/invalid/path.db", "Seattle, United States")
        
        self.assertEqual(invalid_themes["total_themes"], 0)

if __name__ == '__main__':
    unittest.main() 