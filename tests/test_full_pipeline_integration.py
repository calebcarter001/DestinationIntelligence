"""
Full Pipeline Integration Test for Cultural Intelligence
Tests the complete pipeline from database to analysis to comparison with real components.
"""

import unittest
import sys
import os
import sqlite3
import tempfile
from unittest.mock import Mock, patch

# Add the root directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

class TestFullPipelineIntegration(unittest.TestCase):
    """Test the complete cultural intelligence pipeline integration"""
    
    def setUp(self):
        """Set up test environment with real database structure"""
        from src.schemas import AuthorityType
        from datetime import datetime
        self.temp_db = tempfile.NamedTemporaryFile(delete=False, suffix='.db')
        self.temp_db_path = self.temp_db.name
        self.temp_db.close()
        
        # Create database with real schema structure
        self.setup_realistic_database()
        
    def tearDown(self):
        """Clean up test environment"""
        try:
            os.unlink(self.temp_db_path)
        except:
            pass
    
    def setup_realistic_database(self):
        """Set up database with realistic schema and test data"""
        conn = sqlite3.connect(self.temp_db_path)
        cursor = conn.cursor()
        
        # Create themes table with full schema
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS themes (
                theme_id TEXT PRIMARY KEY,
                destination_id TEXT,
                name TEXT,
                macro_category TEXT,
                micro_category TEXT,
                description TEXT,
                fit_score REAL,
                confidence_level TEXT,
                confidence_breakdown TEXT,
                adjusted_overall_confidence REAL,
                traveler_relevance_factor REAL,
                tags TEXT,
                created_date TEXT,
                evidence_count INTEGER DEFAULT 0
            )
        """)
        
        # Insert realistic test themes for Seattle
        realistic_themes = [
            # Cultural themes
            ("theme_grunge", "dest_seattle_united_states", "Grunge Music Heritage", 
             "Cultural Identity & Atmosphere", "Music Scene", 
             "Seattle's distinctive grunge music culture and heritage sites including venues where Nirvana and Pearl Jam started",
             0.92, "HIGH", '{"thematic_relevance": 0.95, "evidence_quality": 0.88, "authenticity": 0.95}', 
             0.87, 0.85, "music,culture,grunge,distinctive,heritage", "2024-01-01", 12),
            
            ("theme_coffee", "dest_seattle_united_states", "Coffee Culture Origins",
             "Authentic Experiences", "Local Traditions",
             "Birthplace of modern coffee culture with unique cafe experiences and Starbucks origin story",
             0.89, "HIGH", '{"thematic_relevance": 0.90, "evidence_quality": 0.85, "authenticity": 0.90}',
             0.83, 0.88, "coffee,culture,authentic,local,distinctive", "2024-01-01", 8),
            
            ("theme_pike_place", "dest_seattle_united_states", "Pike Place Fish Throwing",
             "Distinctive Features", "Local Traditions", 
             "Iconic fish throwing tradition at Pike Place Market",
             0.85, "HIGH", '{"thematic_relevance": 0.88, "evidence_quality": 0.82, "authenticity": 0.85}',
             0.81, 0.80, "tradition,unique,market,distinctive", "2024-01-01", 6),
            
            # Practical themes  
            ("theme_transport", "dest_seattle_united_states", "Public Transportation System",
             "Transportation & Access", "Public Transit",
             "Comprehensive light rail, bus, and ferry system connecting the region",
             0.88, "HIGH", '{"thematic_relevance": 0.88, "evidence_quality": 0.90, "authority": 0.95}',
             0.86, 0.92, "transport,practical,public,system,reliable", "2024-01-01", 15),
            
            ("theme_safety", "dest_seattle_united_states", "Urban Safety Considerations", 
             "Safety & Security", "Crime Statistics",
             "Comprehensive safety information for urban travelers including neighborhood guides",
             0.82, "HIGH", '{"thematic_relevance": 0.85, "evidence_quality": 0.88, "authority": 0.90}',
             0.84, 0.89, "safety,crime,security,practical,neighborhoods", "2024-01-01", 10),
            
            # Hybrid themes
            ("theme_food", "dest_seattle_united_states", "Pacific Northwest Cuisine",
             "Food & Dining", "Local Cuisine", 
             "Fresh seafood, farm-to-table dining, and diverse culinary scene",
             0.81, "MEDIUM-HIGH", '{"thematic_relevance": 0.85, "evidence_quality": 0.78, "authenticity": 0.75}',
             0.78, 0.83, "food,dining,local,seafood,fresh,cuisine", "2024-01-01", 9),
            
            ("theme_nightlife", "dest_seattle_united_states", "Capitol Hill Nightlife",
             "Entertainment & Nightlife", "Districts",
             "Vibrant nightlife scene in Capitol Hill with bars, clubs, and live music venues", 
             0.76, "MEDIUM", '{"thematic_relevance": 0.80, "evidence_quality": 0.72, "authenticity": 0.78}',
             0.74, 0.81, "nightlife,entertainment,bars,music,capitol-hill", "2024-01-01", 7)
        ]
        
        for theme in realistic_themes:
            cursor.execute("""
                INSERT INTO themes (
                    theme_id, destination_id, name, macro_category, micro_category,
                    description, fit_score, confidence_level, confidence_breakdown,
                    adjusted_overall_confidence, traveler_relevance_factor, tags, 
                    created_date, evidence_count
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, theme)
        
        conn.commit()
        conn.close()

    def test_theme_loading_and_categorization_pipeline(self):
        """Test the complete theme loading and categorization pipeline"""
        try:
            from compare_destinations import load_destination_themes, get_processing_type
            
            # Load themes using the actual function
            themes_data = load_destination_themes(self.temp_db_path, "Seattle, United States")
            
            # Verify themes were loaded
            self.assertGreater(themes_data["total_themes"], 0, "Should load themes from database")
            
            # Verify categorization worked
            self.assertIn("cultural", themes_data["theme_stats"])
            self.assertIn("practical", themes_data["theme_stats"]) 
            self.assertIn("hybrid", themes_data["theme_stats"])
            
            # Verify we have expected theme types
            self.assertGreater(themes_data["theme_stats"]["cultural"], 0, "Should have cultural themes")
            self.assertGreater(themes_data["theme_stats"]["practical"], 0, "Should have practical themes")
            self.assertGreater(themes_data["theme_stats"]["hybrid"], 0, "Should have hybrid themes")
            
            # Verify confidence calculations
            self.assertIn("avg_confidence", themes_data)
            self.assertGreater(themes_data["avg_confidence"]["cultural"], 0)
            self.assertGreater(themes_data["avg_confidence"]["practical"], 0) 
            self.assertGreater(themes_data["avg_confidence"]["hybrid"], 0)
            
            # Verify themes by type structure
            cultural_themes = themes_data["themes_by_type"]["cultural"]
            practical_themes = themes_data["themes_by_type"]["practical"]
            hybrid_themes = themes_data["themes_by_type"]["hybrid"]
            
            # Check that themes have required fields
            for theme in cultural_themes + practical_themes + hybrid_themes:
                self.assertIn("processing_type", theme)
                self.assertIn("name", theme)
                self.assertIn("macro_category", theme)
                self.assertIn("confidence", theme)
            
            print("✅ Theme loading and categorization pipeline working")
            
        except Exception as e:
            self.fail(f"Theme loading and categorization pipeline failed: {e}")

    def test_cultural_intelligence_similarity_calculation(self):
        """Test the cultural intelligence similarity calculation with real data"""
        try:
            from compare_destinations import (
                load_destination_themes, 
                calculate_cultural_intelligence_similarity
            )
            
            # Load themes for Seattle
            seattle_themes = load_destination_themes(self.temp_db_path, "Seattle, United States")
            
            # Create a mock second destination with different theme distribution
            mock_portland_themes = {
                "themes_by_type": {
                    "cultural": [{"name": "Food Truck Culture"}, {"name": "Craft Beer Scene"}],
                    "practical": [{"name": "MAX Light Rail"}, {"name": "Bike Infrastructure"}],
                    "hybrid": [{"name": "Local Food Scene"}]
                },
                "theme_stats": {"cultural": 2, "practical": 2, "hybrid": 1, "unknown": 0},
                "avg_confidence": {"cultural": 0.78, "practical": 0.85, "hybrid": 0.72},
                "high_confidence_themes": {"cultural": 1, "practical": 2, "hybrid": 0},
                "total_themes": 5
            }
            
            # Calculate similarity
            similarity = calculate_cultural_intelligence_similarity(seattle_themes, mock_portland_themes)
            
            # Verify similarity result structure
            required_fields = [
                "theme_distribution_similarity",
                "confidence_similarity", 
                "content_similarity",
                "cultural_character"
            ]
            
            for field in required_fields:
                self.assertIn(field, similarity, f"Missing field: {field}")
            
            # Verify cultural character analysis
            char_analysis = similarity["cultural_character"]
            self.assertIn("dest1_personality", char_analysis)
            self.assertIn("dest2_personality", char_analysis)
            self.assertIn("cultural_practical_ratio_similarity", char_analysis)
            
            # Verify similarity scores are in valid range
            for proc_type in ["cultural", "practical", "hybrid"]:
                dist_sim = similarity["theme_distribution_similarity"][proc_type]
                conf_sim = similarity["confidence_similarity"][proc_type]
                
                self.assertGreaterEqual(dist_sim, 0.0)
                self.assertLessEqual(dist_sim, 1.0)
                self.assertGreaterEqual(conf_sim, 0.0)
                self.assertLessEqual(conf_sim, 1.0)
            
            print("✅ Cultural intelligence similarity calculation working")
            
        except Exception as e:
            self.fail(f"Cultural intelligence similarity calculation failed: {e}")

    def test_analyze_themes_integration_with_real_data(self):
        """Test analyze_themes.py integration with real database data"""
        try:
            import analyze_themes
            
            # Test loading themes from database  
            themes = analyze_themes.load_themes_from_db(self.temp_db_path, "Seattle, United States")
            
            self.assertGreater(len(themes), 0, "Should load themes from database")
            
            # Test processing type identification on real themes
            cultural_count = 0
            practical_count = 0
            hybrid_count = 0
            
            for theme in themes:
                proc_type = analyze_themes.get_processing_type(theme.get("macro_category"))
                if proc_type == "cultural":
                    cultural_count += 1
                elif proc_type == "practical":
                    practical_count += 1
                elif proc_type == "hybrid":
                    hybrid_count += 1
            
            # Verify we categorized themes correctly
            self.assertGreater(cultural_count, 0, "Should identify cultural themes")
            self.assertGreater(practical_count, 0, "Should identify practical themes")
            self.assertGreater(hybrid_count, 0, "Should identify hybrid themes")
            
            # Test cultural intelligence metrics calculation
            themes_with_types = []
            for theme in themes:
                theme_copy = theme.copy()
                theme_copy["processing_type"] = analyze_themes.get_processing_type(theme.get("macro_category"))
                theme_copy["confidence"] = theme.get("adjusted_overall_confidence", 0.0)
                themes_with_types.append(theme_copy)
            
            metrics = analyze_themes.calculate_cultural_intelligence_metrics(themes_with_types)
            
            # Verify metrics structure
            required_metrics = ["theme_distribution", "avg_confidence_by_type", "cultural_practical_ratio"]
            for metric in required_metrics:
                self.assertIn(metric, metrics, f"Missing metric: {metric}")
            
            print("✅ analyze_themes.py integration with real data working")
            
        except Exception as e:
            self.fail(f"analyze_themes.py integration with real data failed: {e}")

    def test_generate_dynamic_viewer_integration(self):
        """Test generate_dynamic_viewer.py integration with real data"""
        try:
            import generate_dynamic_viewer
            
            # Test loading and categorizing themes
            themes_data = generate_dynamic_viewer.load_and_categorize_themes(
                self.temp_db_path, "Seattle, United States"
            )
            
            self.assertIsInstance(themes_data, list)
            self.assertGreater(len(themes_data), 0, "Should load and categorize themes")
            
            # Verify each theme has cultural intelligence fields
            for theme in themes_data:
                self.assertIn("processing_type", theme)
                self.assertIn("category_color", theme)
                self.assertIn("category_icon", theme)
                
                # Processing type should be valid
                self.assertIn(theme["processing_type"], ["cultural", "practical", "hybrid", "unknown"])
            
            print("✅ generate_dynamic_viewer.py integration working")
            
        except Exception as e:
            self.fail(f"generate_dynamic_viewer.py integration failed: {e}")

    def test_end_to_end_pipeline_consistency(self):
        """Test end-to-end pipeline consistency across all components"""
        try:
            import analyze_themes
            import compare_destinations
            import generate_dynamic_viewer
            
            destination_name = "Seattle, United States"
            
            # Test that all components can load the same data consistently
            analyze_themes_data = analyze_themes.load_themes_from_db(self.temp_db_path, destination_name)
            compare_themes_data = compare_destinations.load_destination_themes(self.temp_db_path, destination_name)
            viewer_themes_data = generate_dynamic_viewer.load_and_categorize_themes(self.temp_db_path, destination_name)
            
            # Verify all loaded data
            self.assertGreater(len(analyze_themes_data), 0)
            self.assertGreater(compare_themes_data["total_themes"], 0)
            self.assertGreater(len(viewer_themes_data), 0)
            
            # Test that categorization is consistent across all components
            test_categories = [
                "Cultural Identity & Atmosphere",
                "Transportation & Access", 
                "Food & Dining"
            ]
            
            for category in test_categories:
                analyze_type = analyze_themes.get_processing_type(category)
                compare_type = compare_destinations.get_processing_type(category)
                viewer_type = generate_dynamic_viewer.get_processing_type(category)
                
                self.assertEqual(analyze_type, compare_type, f"Inconsistent categorization for {category}")
                self.assertEqual(compare_type, viewer_type, f"Inconsistent categorization for {category}")
            
            print("✅ End-to-end pipeline consistency verified")
            
        except Exception as e:
            self.fail(f"End-to-end pipeline consistency test failed: {e}")

def run_full_pipeline_tests():
    """Run the full pipeline integration tests"""
    print("🔗 Full Pipeline Integration Tests")
    print("=" * 50)
    print("Testing complete cultural intelligence pipeline with real data...")
    print()
    
    # Create test suite
    suite = unittest.TestSuite()
    suite.addTest(unittest.makeSuite(TestFullPipelineIntegration))
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Print summary
    print("\n" + "=" * 50)
    print("📊 Full Pipeline Test Results")
    print("=" * 50)
    
    total_tests = result.testsRun
    failures = len(result.failures)
    errors = len(result.errors)
    success = total_tests - failures - errors
    
    print(f"Total Pipeline Tests: {total_tests}")
    print(f"✅ Passed: {success}")
    print(f"❌ Failed: {failures}")
    print(f"🚨 Errors: {errors}")
    
    if failures == 0 and errors == 0:
        print("\n🎉 Full pipeline integration tests passed!")
        print("✅ Cultural Intelligence pipeline works end-to-end with real data")
        return True
    else:
        print("\n⚠️  Some pipeline tests failed. Check output above.")
        return False

if __name__ == "__main__":
    success = run_full_pipeline_tests()
    exit(0 if success else 1) 