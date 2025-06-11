"""
Integration tests for Cultural Intelligence enhanced scripts.
Tests analyze_themes.py, generate_dynamic_viewer.py, and compare_destinations.py functionality.
"""

import unittest
import tempfile
import os
import sys
import subprocess
import json
import sqlite3
from unittest.mock import Mock, patch

# Add the root directory to the path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from src.core.enhanced_database_manager import EnhancedDatabaseManager
from src.core.enhanced_data_models import Destination

class TestCulturalScriptsIntegration(unittest.TestCase):
    """Integration tests for cultural intelligence enhanced scripts"""
    
    def setUp(self):
        """Set up test environment with sample database"""
        from src.schemas import AuthorityType
        from datetime import datetime
        # Create temporary database
        self.temp_db = tempfile.NamedTemporaryFile(delete=False, suffix='.db')
        self.temp_db_path = self.temp_db.name
        self.temp_db.close()
        
        # Set up database manager and test data
        self.db_manager = EnhancedDatabaseManager(db_path=self.temp_db_path)
        self.setup_test_data()
        
    def tearDown(self):
        """Clean up temporary files"""
        self.db_manager.close_db()
        os.unlink(self.temp_db_path)
    
    def setup_test_data(self):
        """Set up comprehensive test data for script testing"""
        # Create multiple test destinations
        destinations = [
            {
                "id": "dest_seattle_united_states",
                "names": ["Seattle, United States"],
                "admin_levels": {"country": "United States", "state": "Washington", "city": "Seattle"},
                "timezone": "America/Los_Angeles",
                "population": 750000,
                "country_code": "US",
                "area_km2": 369.2,
                "primary_language": "English",
                "dominant_religions": ["Christianity"],
                "gdp_per_capita_usd": 50000,
                "hdi": 0.92,
                "vibe_descriptors": ["urban", "creative", "tech-savvy", "coffee-culture"]
            },
            {
                "id": "dest_tokyo_japan",
                "names": ["Tokyo, Japan"],
                "admin_levels": {"country": "Japan", "prefecture": "Tokyo", "city": "Tokyo"},
                "timezone": "Asia/Tokyo",
                "population": 14000000,
                "country_code": "JP",
                "area_km2": 2194.0,
                "primary_language": "Japanese",
                "dominant_religions": ["Buddhism", "Shintoism"],
                "gdp_per_capita_usd": 40000,
                "hdi": 0.91,
                "vibe_descriptors": ["urban", "traditional", "modern", "efficient"]
            }
        ]
        
        for dest_data in destinations:
            destination = Destination(**dest_data)
            self.db_manager.store_destination(destination)
        
        # Add comprehensive theme data for testing
        sample_themes = [
            # Seattle themes - Cultural focused
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
                "theme_id": "theme_coffee_origins",
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
                "theme_id": "theme_seattle_transport",
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
                "theme_id": "theme_seattle_food",
                "destination_id": "dest_seattle_united_states",
                "name": "Pacific Northwest Cuisine",
                "macro_category": "Food & Dining",
                "micro_category": "Local Cuisine",
                "description": "Fresh seafood and farm-to-table dining culture",
                "fit_score": 0.81,
                "confidence_level": "MEDIUM-HIGH",
                "confidence_breakdown": '{"thematic_relevance": 0.85, "evidence_quality": 0.78, "authority": 0.80, "authenticity": 0.75}',
                "adjusted_overall_confidence": 0.78,
                "traveler_relevance_factor": 0.83,
                "tags": "food,dining,local,seafood,fresh"
            },
            # Tokyo themes - Different cultural profile
            {
                "theme_id": "theme_tokyo_tradition",
                "destination_id": "dest_tokyo_japan",
                "name": "Traditional Temple Culture",
                "macro_category": "Cultural Identity & Atmosphere",
                "micro_category": "Religious Sites",
                "description": "Ancient temple traditions in modern cityscape",
                "fit_score": 0.94,
                "confidence_level": "HIGH",
                "confidence_breakdown": '{"thematic_relevance": 0.96, "evidence_quality": 0.92, "authority": 0.85, "authenticity": 0.88}',
                "adjusted_overall_confidence": 0.90,
                "traveler_relevance_factor": 0.87,
                "tags": "temples,tradition,culture,spiritual,authentic"
            },
            {
                "theme_id": "theme_tokyo_efficiency",
                "destination_id": "dest_tokyo_japan",
                "name": "Transportation Efficiency",
                "macro_category": "Transportation & Access",
                "micro_category": "Public Transit",
                "description": "World-renowned efficient public transportation system",
                "fit_score": 0.96,
                "confidence_level": "HIGH",
                "confidence_breakdown": '{"thematic_relevance": 0.98, "evidence_quality": 0.95, "authority": 0.98, "authenticity": 0.70}',
                "adjusted_overall_confidence": 0.93,
                "traveler_relevance_factor": 0.95,
                "tags": "transport,efficiency,public,system,reliable"
            },
            {
                "theme_id": "theme_tokyo_anime",
                "destination_id": "dest_tokyo_japan",
                "name": "Anime and Manga Culture",
                "macro_category": "Artistic & Creative Scene",
                "micro_category": "Popular Culture",
                "description": "Heart of global anime and manga culture",
                "fit_score": 0.88,
                "confidence_level": "HIGH",
                "confidence_breakdown": '{"thematic_relevance": 0.92, "evidence_quality": 0.85, "authority": 0.75, "authenticity": 0.85}',
                "adjusted_overall_confidence": 0.84,
                "traveler_relevance_factor": 0.82,
                "tags": "anime,manga,culture,creative,distinctive"
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

    def test_analyze_themes_script_functionality(self):
        """Test analyze_themes.py script functionality with cultural intelligence"""
        from analyze_themes import (
            get_processing_type, 
            calculate_cultural_intelligence_metrics,
            load_themes_from_db,
            generate_cultural_report
        )
        
        # Test loading themes from database
        themes = load_themes_from_db(self.temp_db_path, "Seattle, United States")
        self.assertGreater(len(themes), 0, "Should load themes from database")
        
        # Test processing type identification
        cultural_count = 0
        practical_count = 0
        hybrid_count = 0
        
        for theme in themes:
            proc_type = get_processing_type(theme.get("macro_category"))
            if proc_type == "cultural":
                cultural_count += 1
            elif proc_type == "practical":
                practical_count += 1
            elif proc_type == "hybrid":
                hybrid_count += 1
        
        self.assertGreater(cultural_count, 0, "Should identify cultural themes")
        self.assertGreater(practical_count, 0, "Should identify practical themes")
        self.assertGreater(hybrid_count, 0, "Should identify hybrid themes")
        
        # Test cultural intelligence metrics calculation
        themes_with_types = []
        for theme in themes:
            theme_copy = theme.copy()
            theme_copy["processing_type"] = get_processing_type(theme.get("macro_category"))
            theme_copy["confidence"] = theme.get("adjusted_overall_confidence", 0.0)
            themes_with_types.append(theme_copy)
        
        metrics = calculate_cultural_intelligence_metrics(themes_with_types)
        
        # Verify metrics structure
        self.assertIn("theme_distribution", metrics)
        self.assertIn("avg_confidence_by_type", metrics)
        self.assertIn("high_confidence_themes", metrics)
        self.assertIn("cultural_practical_ratio", metrics)
        
        # Test cultural report generation
        report_lines = generate_cultural_report("Seattle, United States", themes_with_types, metrics)
        self.assertIsInstance(report_lines, list)
        self.assertGreater(len(report_lines), 0, "Should generate report content")

    def test_generate_dynamic_viewer_script_functionality(self):
        """Test generate_dynamic_viewer.py script functionality with cultural intelligence"""
        from generate_dynamic_viewer import (
            load_and_categorize_themes,
            get_processing_type,
            apply_category_styling,
            generate_theme_cards_html
        )
        
        # Test loading and categorizing themes
        themes_data = load_and_categorize_themes(self.temp_db_path, "Seattle, United States")
        
        self.assertIsInstance(themes_data, list)
        self.assertGreater(len(themes_data), 0, "Should load themes data")
        
        # Verify each theme has cultural intelligence fields
        for theme in themes_data:
            self.assertIn("processing_type", theme)
            self.assertIn("category_color", theme)
            self.assertIn("category_icon", theme)
            self.assertIn(theme["processing_type"], ["cultural", "practical", "hybrid", "unknown"])
        
        # Test category styling application
        sample_theme = {
            "name": "Test Theme",
            "processing_type": "cultural",
            "macro_category": "Cultural Identity & Atmosphere",
            "confidence": 0.85
        }
        
        styled_theme = apply_category_styling(sample_theme)
        self.assertIn("category_color", styled_theme)
        self.assertIn("category_icon", styled_theme)
        self.assertEqual(styled_theme["category_color"], "#9C27B0")  # Cultural purple
        
        # Test theme cards HTML generation
        html_content = generate_theme_cards_html(themes_data[:2])  # Test with first 2 themes
        self.assertIsInstance(html_content, str)
        self.assertIn("theme-card", html_content)
        
        # Should contain cultural intelligence elements
        cultural_elements = ["category-badge", "processing-type", "confidence"]
        for element in cultural_elements:
            self.assertIn(element, html_content, f"Should contain {element} in HTML")

    def test_compare_destinations_script_functionality(self):
        """Test compare_destinations.py script functionality with cultural intelligence"""
        from compare_destinations import (
            load_destination_themes,
            calculate_cultural_intelligence_similarity,
            compare_destinations,
            get_processing_type
        )
        
        # Load themes for both destinations
        seattle_themes = load_destination_themes(self.temp_db_path, "Seattle, United States")
        tokyo_themes = load_destination_themes(self.temp_db_path, "Tokyo, Japan")
        
        # Verify themes loaded correctly
        self.assertGreater(seattle_themes["total_themes"], 0)
        self.assertGreater(tokyo_themes["total_themes"], 0)
        
        # Test cultural intelligence similarity calculation
        ci_similarity = calculate_cultural_intelligence_similarity(seattle_themes, tokyo_themes)
        
        # Verify similarity structure
        required_fields = [
            "theme_distribution_similarity",
            "confidence_similarity", 
            "content_similarity",
            "cultural_character"
        ]
        
        for field in required_fields:
            self.assertIn(field, ci_similarity)
        
        # Test cultural character analysis
        char_analysis = ci_similarity["cultural_character"]
        self.assertIn("dest1_personality", char_analysis)
        self.assertIn("dest2_personality", char_analysis)
        self.assertIn("cultural_practical_ratio_similarity", char_analysis)
        
        # Load destination objects for comparison
        seattle = self.db_manager.get_destination_by_name("Seattle, United States")
        tokyo = self.db_manager.get_destination_by_name("Tokyo, Japan")
        
        # Test full destination comparison
        comparison_result = compare_destinations(seattle, tokyo, seattle_themes, tokyo_themes)
        
        # Verify comparison result structure
        self.assertIn("overall_similarity_score", comparison_result)
        self.assertIn("scores", comparison_result)
        self.assertIn("cultural_intelligence", comparison_result["scores"])
        self.assertIn("drivers", comparison_result)
        self.assertIn("cultural_insights", comparison_result["drivers"])
        
        # Verify cultural intelligence has significant weight
        ci_score = comparison_result["scores"]["cultural_intelligence"]
        overall_score = comparison_result["overall_similarity_score"]
        
        self.assertGreaterEqual(ci_score, 0.0)
        self.assertLessEqual(ci_score, 1.0)
        self.assertGreaterEqual(overall_score, 0.0)
        self.assertLessEqual(overall_score, 1.0)

    def test_script_consistency_across_processing_types(self):
        """Test that all scripts use consistent processing type identification"""
        from analyze_themes import get_processing_type as analyze_type
        from compare_destinations import get_processing_type as compare_type
        from generate_dynamic_viewer import get_processing_type as viewer_type
        
        test_categories = [
            ("Cultural Identity & Atmosphere", "cultural"),
            ("Safety & Security", "practical"),
            ("Food & Dining", "hybrid"),
            ("Transportation & Access", "practical"),
            ("Authentic Experiences", "cultural"),
            ("Entertainment & Nightlife", "hybrid"),
            ("Unknown Category", "unknown"),
            (None, "unknown")
        ]
        
        for category, expected_type in test_categories:
            analyze_result = analyze_type(category)
            compare_result = compare_type(category)
            viewer_result = viewer_type(category)
            
            # All scripts should return consistent results
            self.assertEqual(analyze_result, expected_type, f"analyze_themes failed for {category}")
            self.assertEqual(compare_result, expected_type, f"compare_destinations failed for {category}")
            self.assertEqual(viewer_result, expected_type, f"generate_dynamic_viewer failed for {category}")
            
            # All scripts should agree with each other
            self.assertEqual(analyze_result, compare_result, f"Inconsistency between analyze and compare for {category}")
            self.assertEqual(compare_result, viewer_result, f"Inconsistency between compare and viewer for {category}")

    def test_script_error_handling(self):
        """Test error handling in cultural intelligence enhanced scripts"""
        from analyze_themes import load_themes_from_db
        from compare_destinations import load_destination_themes
        from generate_dynamic_viewer import load_and_categorize_themes
        
        # Test with non-existent destination
        nonexistent_dest = "Nonexistent City, Country"
        
        # analyze_themes error handling
        analyze_themes = load_themes_from_db(self.temp_db_path, nonexistent_dest)
        self.assertEqual(len(analyze_themes), 0, "Should handle non-existent destination gracefully")
        
        # compare_destinations error handling
        compare_themes = load_destination_themes(self.temp_db_path, nonexistent_dest)
        self.assertEqual(compare_themes["total_themes"], 0, "Should handle non-existent destination gracefully")
        
        # generate_dynamic_viewer error handling
        viewer_themes = load_and_categorize_themes(self.temp_db_path, nonexistent_dest)
        self.assertEqual(len(viewer_themes), 0, "Should handle non-existent destination gracefully")
        
        # Test with invalid database path
        invalid_db_path = "/path/to/nonexistent.db"
        
        analyze_themes_invalid = load_themes_from_db(invalid_db_path, "Seattle, United States")
        self.assertEqual(len(analyze_themes_invalid), 0, "Should handle invalid database path")
        
        compare_themes_invalid = load_destination_themes(invalid_db_path, "Seattle, United States")
        self.assertEqual(compare_themes_invalid["total_themes"], 0, "Should handle invalid database path")

    def test_cultural_intelligence_weighting_consistency(self):
        """Test that cultural intelligence weighting is applied consistently"""
        from compare_destinations import load_destination_themes, compare_destinations
        
        # Load test destinations
        seattle = self.db_manager.get_destination_by_name("Seattle, United States")
        tokyo = self.db_manager.get_destination_by_name("Tokyo, Japan")
        seattle_themes = load_destination_themes(self.temp_db_path, "Seattle, United States")
        tokyo_themes = load_destination_themes(self.temp_db_path, "Tokyo, Japan")
        
        # Perform comparison
        comparison = compare_destinations(seattle, tokyo, seattle_themes, tokyo_themes)
        
        # Cultural intelligence should have highest weight (0.40)
        ci_score = comparison["scores"]["cultural_intelligence"]
        
        # Calculate manual weighted score to verify consistency
        scores = comparison["scores"]
        weights = {
            "cultural_intelligence": 0.40,
            "experiential": 0.20,
            "geography": 0.15,
            "cultural_traditional": 0.15,
            "logistics": 0.10
        }
        
        manual_overall = sum(scores[key] * weights[key] for key in weights if key in scores)
        actual_overall = comparison["overall_similarity_score"]
        
        # Allow small floating point differences
        self.assertAlmostEqual(manual_overall, actual_overall, places=3, 
                              msg="Weighted score calculation should be consistent")

if __name__ == '__main__':
    unittest.main() 