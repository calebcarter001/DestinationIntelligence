"""
Integration Verification Tests - Tests the actual working integration with existing codebase
Verifies that cultural intelligence features work with real existing functions and database.
"""

import unittest
import sys
import os
import sqlite3
import tempfile

# Add the root directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

class TestIntegrationVerification(unittest.TestCase):
    """Verify integration with actual existing codebase components"""
    
    def setUp(self):
        """Set up test environment"""
        from src.schemas import AuthorityType
        from datetime import datetime
        self.temp_db = tempfile.NamedTemporaryFile(delete=False, suffix='.db')
        self.temp_db_path = self.temp_db.name
        self.temp_db.close()
        
    def tearDown(self):
        """Clean up test environment"""
        try:
            os.unlink(self.temp_db_path)
        except:
            pass

    def test_actual_script_imports_and_functions(self):
        """Test that we can import scripts and access their actual functions"""
        try:
            # Test analyze_themes.py
            import analyze_themes
            self.assertTrue(hasattr(analyze_themes, 'get_processing_type'))
            self.assertTrue(hasattr(analyze_themes, 'calculate_cultural_intelligence_metrics'))
            self.assertTrue(hasattr(analyze_themes, 'CATEGORY_PROCESSING_RULES'))
            
            # Test compare_destinations.py
            import compare_destinations
            self.assertTrue(hasattr(compare_destinations, 'get_processing_type'))
            self.assertTrue(hasattr(compare_destinations, 'load_destination_themes'))
            self.assertTrue(hasattr(compare_destinations, 'calculate_cultural_intelligence_similarity'))
            
            # Test generate_dynamic_viewer.py
            import generate_dynamic_viewer
            self.assertTrue(hasattr(generate_dynamic_viewer, 'get_processing_type'))
            
            print("✅ All cultural intelligence scripts import successfully")
            
        except ImportError as e:
            self.fail(f"Script import failed: {e}")

    def test_processing_type_consistency_across_scripts(self):
        """Test that processing type identification is consistent across all scripts"""
        try:
            import analyze_themes
            import compare_destinations
            import generate_dynamic_viewer
            
            test_categories = [
                ("Cultural Identity & Atmosphere", "cultural"),
                ("Authentic Experiences", "cultural"),
                ("Safety & Security", "practical"),
                ("Transportation & Access", "practical"),
                ("Food & Dining", "hybrid"),
                ("Entertainment & Nightlife", "hybrid"),
                ("Unknown Category", "unknown"),
                (None, "unknown")
            ]
            
            for category, expected_type in test_categories:
                analyze_result = analyze_themes.get_processing_type(category)
                compare_result = compare_destinations.get_processing_type(category)
                viewer_result = generate_dynamic_viewer.get_processing_type(category)
                
                # All should match expected
                self.assertEqual(analyze_result, expected_type, f"analyze_themes failed for {category}")
                self.assertEqual(compare_result, expected_type, f"compare_destinations failed for {category}")
                self.assertEqual(viewer_result, expected_type, f"generate_dynamic_viewer failed for {category}")
                
                # All should be consistent with each other
                self.assertEqual(analyze_result, compare_result, f"Inconsistency between analyze and compare for {category}")
                self.assertEqual(compare_result, viewer_result, f"Inconsistency between compare and viewer for {category}")
            
            print("✅ Processing type consistency verified across all scripts")
            
        except Exception as e:
            self.fail(f"Processing type consistency test failed: {e}")

    def test_category_processing_rules_consistency(self):
        """Test that category processing rules are consistent across scripts"""
        try:
            import analyze_themes
            import compare_destinations
            
            analyze_rules = analyze_themes.CATEGORY_PROCESSING_RULES
            compare_rules = compare_destinations.CATEGORY_PROCESSING_RULES
            
            # Test that both have the same structure
            for proc_type in ["cultural", "practical", "hybrid"]:
                self.assertIn(proc_type, analyze_rules)
                self.assertIn(proc_type, compare_rules)
                
                # Test that categories match
                analyze_categories = set(analyze_rules[proc_type]["categories"])
                compare_categories = set(compare_rules[proc_type]["categories"])
                self.assertEqual(analyze_categories, compare_categories, f"Category mismatch for {proc_type}")
                
                # Test that colors and icons match
                self.assertEqual(analyze_rules[proc_type]["color"], compare_rules[proc_type]["color"])
                self.assertEqual(analyze_rules[proc_type]["icon"], compare_rules[proc_type]["icon"])
            
            print("✅ Category processing rules consistency verified")
            
        except Exception as e:
            self.fail(f"Category processing rules consistency test failed: {e}")

    def test_database_theme_loading_integration(self):
        """Test that theme loading works with real database structure"""
        try:
            # Create a minimal themes table matching real schema
            conn = sqlite3.connect(self.temp_db_path)
            cursor = conn.cursor()
            
            cursor.execute("""
                CREATE TABLE themes (
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
                    tags TEXT
                )
            """)
            
            # Insert test themes
            test_themes = [
                ("t1", "dest_seattle_united_states", "Grunge Heritage", "Cultural Identity & Atmosphere", "Music", "Test", 0.9, "HIGH", '{"cultural": 0.9}', 0.85, 0.8, "music"),
                ("t2", "dest_seattle_united_states", "Public Transit", "Transportation & Access", "Transit", "Test", 0.8, "HIGH", '{"practical": 0.8}', 0.82, 0.9, "transport"),
                ("t3", "dest_seattle_united_states", "Food Scene", "Food & Dining", "Cuisine", "Test", 0.7, "MEDIUM", '{"hybrid": 0.7}', 0.75, 0.85, "food")
            ]
            
            for theme in test_themes:
                cursor.execute("""
                    INSERT INTO themes (theme_id, destination_id, name, macro_category, micro_category,
                                      description, fit_score, confidence_level, confidence_breakdown, 
                                      adjusted_overall_confidence, traveler_relevance_factor, tags)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, theme)
            
            conn.commit()
            
            # Test theme loading with compare_destinations function
            from compare_destinations import load_destination_themes
            
            themes_data = load_destination_themes(self.temp_db_path, "Seattle, United States")
            
            # Verify structure
            self.assertIn("total_themes", themes_data)
            self.assertIn("theme_stats", themes_data)
            self.assertIn("themes_by_type", themes_data)
            
            # Verify we loaded themes
            self.assertGreater(themes_data["total_themes"], 0)
            self.assertGreater(themes_data["theme_stats"]["cultural"], 0)
            self.assertGreater(themes_data["theme_stats"]["practical"], 0) 
            self.assertGreater(themes_data["theme_stats"]["hybrid"], 0)
            
            conn.close()
            print("✅ Database theme loading integration verified")
            
        except Exception as e:
            self.fail(f"Database theme loading integration failed: {e}")

    def test_cultural_metrics_calculation_integration(self):
        """Test cultural intelligence metrics calculation with realistic data"""
        try:
            from analyze_themes import calculate_cultural_intelligence_metrics
            
            # Create realistic theme data structure (matching what analyze_themes expects)
            sample_themes = [
                {
                    'name': 'Grunge Music Heritage',
                    'overall_confidence': 0.87,
                    'metadata': {'macro_category': 'Cultural Identity & Atmosphere'}
                },
                {
                    'name': 'Coffee Culture Origins', 
                    'overall_confidence': 0.83,
                    'metadata': {'macro_category': 'Authentic Experiences'}
                },
                {
                    'name': 'Public Transportation',
                    'overall_confidence': 0.86,
                    'metadata': {'macro_category': 'Transportation & Access'}
                },
                {
                    'name': 'Local Food Scene',
                    'overall_confidence': 0.78,
                    'metadata': {'macro_category': 'Food & Dining'}
                }
            ]
            
            metrics = calculate_cultural_intelligence_metrics(sample_themes)
            
            # Verify metrics structure
            required_fields = [
                "total_themes", "category_breakdown", "avg_confidence_by_type",
                "high_confidence_themes", "top_cultural_themes", "top_practical_themes"
            ]
            
            for field in required_fields:
                self.assertIn(field, metrics, f"Missing metric field: {field}")
            
            # Verify categorization worked
            self.assertEqual(metrics["total_themes"], 4)
            self.assertGreater(metrics["category_breakdown"]["cultural"], 0)
            self.assertGreater(metrics["category_breakdown"]["practical"], 0)
            self.assertGreater(metrics["category_breakdown"]["hybrid"], 0)
            
            print("✅ Cultural metrics calculation integration verified")
            
        except Exception as e:
            self.fail(f"Cultural metrics calculation integration failed: {e}")

    def test_destination_comparison_integration(self):
        """Test destination comparison cultural intelligence integration"""
        try:
            from compare_destinations import calculate_cultural_intelligence_similarity
            
            # Create mock theme data for two destinations
            dest1_themes = {
                "theme_stats": {"cultural": 3, "practical": 2, "hybrid": 2},
                "avg_confidence": {"cultural": 0.85, "practical": 0.88, "hybrid": 0.75},
                "themes_by_type": {
                    "cultural": [{"name": "Grunge Heritage"}, {"name": "Coffee Culture"}],
                    "practical": [{"name": "Public Transit"}],
                    "hybrid": [{"name": "Food Scene"}]
                },
                "total_themes": 7
            }
            
            dest2_themes = {
                "theme_stats": {"cultural": 2, "practical": 3, "hybrid": 2},
                "avg_confidence": {"cultural": 0.80, "practical": 0.85, "hybrid": 0.78},
                "themes_by_type": {
                    "cultural": [{"name": "Jazz Heritage"}],
                    "practical": [{"name": "Light Rail"}, {"name": "Bike Infrastructure"}],
                    "hybrid": [{"name": "Food Scene"}]
                },
                "total_themes": 7
            }
            
            similarity = calculate_cultural_intelligence_similarity(dest1_themes, dest2_themes)
            
            # Verify similarity structure
            required_fields = [
                "theme_distribution_similarity", "confidence_similarity",
                "content_similarity", "cultural_character"
            ]
            
            for field in required_fields:
                self.assertIn(field, similarity)
            
            # Verify cultural character analysis
            char_analysis = similarity["cultural_character"]
            self.assertIn("dest1_personality", char_analysis)
            self.assertIn("dest2_personality", char_analysis)
            
            print("✅ Destination comparison integration verified")
            
        except Exception as e:
            self.fail(f"Destination comparison integration failed: {e}")

def run_integration_verification():
    """Run integration verification tests"""
    print("🔍 Cultural Intelligence Integration Verification")
    print("=" * 55)
    print("Verifying integration with actual existing codebase...")
    print()
    
    # Create test suite
    suite = unittest.TestSuite()
    suite.addTest(unittest.makeSuite(TestIntegrationVerification))
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Print summary
    print("\n" + "=" * 55)
    print("📊 Integration Verification Results")
    print("=" * 55)
    
    total_tests = result.testsRun
    failures = len(result.failures)
    errors = len(result.errors)
    success = total_tests - failures - errors
    
    print(f"Total Integration Checks: {total_tests}")
    print(f"✅ Verified: {success}")
    print(f"❌ Failed: {failures}")
    print(f"🚨 Errors: {errors}")
    
    if failures == 0 and errors == 0:
        print("\n🎉 All integration verifications passed!")
        print("✅ Cultural Intelligence successfully integrates with existing codebase")
        print("✅ All functions work with real database structure")
        print("✅ Scripts are consistent across the entire pipeline")
        return True
    else:
        print("\n⚠️  Some integration verifications failed. Check output above.")
        return False

if __name__ == "__main__":
    success = run_integration_verification()
    exit(0 if success else 1) 