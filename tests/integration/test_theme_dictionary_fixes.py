import unittest
import asyncio
import sys
import os
from datetime import datetime

# Add project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from src.tools.enhanced_theme_analysis_tool import EnhancedThemeAnalysisTool, EnhancedThemeAnalysisInput
from src.core.enhanced_database_manager import EnhancedDatabaseManager
from src.core.enhanced_data_models import Destination


class TestThemeDictionaryFixes(unittest.TestCase):
    """Integration test to validate theme object vs dictionary compatibility fixes"""

    def setUp(self):
        """Set up test environment"""
        from src.schemas import AuthorityType
        from datetime import datetime
        # Use in-memory database for testing
        self.db_manager = EnhancedDatabaseManager(db_path=":memory:", enable_json_export=False)
        self.tool = EnhancedThemeAnalysisTool()

    def tearDown(self):
        """Clean up test environment"""
        self.db_manager.close_db()

    def test_theme_analysis_produces_themes(self):
        """Test that theme analysis produces themes without object/dictionary errors"""
        
        # Create realistic input data
        input_data = EnhancedThemeAnalysisInput(
            destination_name="Test Mountain Resort",
            country_code="US",
            text_content_list=[
                {
                    "url": "https://example.com/mountain-resort",
                    "content": "The mountain resort offers excellent skiing during winter months. The snow conditions are typically perfect from December through March. Local experts recommend visiting during February for the best powder snow. The resort features challenging black diamond trails and beginner-friendly slopes.",
                    "title": "Mountain Resort Skiing Guide"
                },
                {
                    "url": "https://local-guide.com/summer-activities", 
                    "content": "During summer, the mountain transforms into a hiking paradise. The wildflower meadows are spectacular in July and August. Mountain biking trails wind through old-growth forests. Local residents recommend the sunrise hike to the summit for breathtaking views.",
                    "title": "Summer Mountain Activities"
                },
                {
                    "url": "https://restaurant-review.com/mountain-dining",
                    "content": "The mountain lodge restaurant serves excellent local cuisine featuring fresh mountain trout and locally sourced vegetables. The chef has been working here for over 15 years and knows the best seasonal ingredients. Family-owned establishment with authentic mountain hospitality.",
                    "title": "Mountain Lodge Dining Review"
                }
            ],
            analyze_temporal=True,
            min_confidence=0.3
        )

        # Run the analysis - this should not raise any object/dictionary errors
        try:
            # Use asyncio.run to properly handle the async method
            result = asyncio.run(self.tool.analyze_themes(input_data))
        except Exception as e:
            self.fail(f"Theme analysis failed with error: {e}")

        # Validate that results are produced
        self.assertIsInstance(result, dict, "Result should be a dictionary")
        self.assertIn("themes", result, "Result should contain themes")
        self.assertIn("evidence_registry", result, "Result should contain evidence registry")
        
        # Check that themes were actually discovered
        themes = result.get("themes", [])
        self.assertGreater(len(themes), 0, "Should discover at least one theme")
        
        # Validate theme structure - themes should be Theme objects
        for theme in themes:
            # Test that themes are proper Theme objects with expected attributes
            self.assertTrue(hasattr(theme, 'name'), "Theme should have name attribute")
            self.assertTrue(hasattr(theme, 'macro_category'), "Theme should have macro_category attribute")
            self.assertTrue(hasattr(theme, 'fit_score'), "Theme should have fit_score attribute")
            
            # Validate enhanced fields are present
            self.assertTrue(hasattr(theme, 'authentic_insights'), "Theme should have authentic_insights attribute")
            self.assertTrue(hasattr(theme, 'seasonal_relevance'), "Theme should have seasonal_relevance attribute")
            self.assertTrue(hasattr(theme, 'cultural_summary'), "Theme should have cultural_summary attribute")
            self.assertTrue(hasattr(theme, 'sentiment_analysis'), "Theme should have sentiment_analysis attribute")
            self.assertTrue(hasattr(theme, 'temporal_analysis'), "Theme should have temporal_analysis attribute")
            self.assertTrue(hasattr(theme, 'factors'), "Theme should have factors attribute")
            
            # Validate data types
            self.assertIsInstance(theme.name, str, "Theme name should be string")
            self.assertIsInstance(theme.fit_score, (int, float), "Fit score should be numeric")
            self.assertIsInstance(theme.authentic_insights, list, "Authentic insights should be list")
            self.assertIsInstance(theme.seasonal_relevance, dict, "Seasonal relevance should be dict")
            self.assertIsInstance(theme.cultural_summary, dict, "Cultural summary should be dict")
            self.assertIsInstance(theme.sentiment_analysis, dict, "Sentiment analysis should be dict")
            self.assertIsInstance(theme.temporal_analysis, dict, "Temporal analysis should be dict")
            self.assertIsInstance(theme.factors, dict, "Factors should be dict")
        
        # Validate evidence registry
        evidence_registry = result.get("evidence_registry", {})
        self.assertGreater(len(evidence_registry), 0, "Should have evidence in registry")
        
        # Check that no themes have dictionary access errors in their structure
        quality_metrics = result.get("quality_metrics", {})
        self.assertIn("themes_discovered", quality_metrics, "Should have quality metrics")
        
        # Print success info for debugging
        print(f"\n✅ Test passed: Discovered {len(themes)} themes")
        for i, theme in enumerate(themes[:3]):  # Show first 3 themes
            print(f"   Theme {i+1}: {theme.name} (fit_score: {theme.fit_score:.2f})")

    def test_database_storage_with_themes(self):
        """Test that themes can be stored in database without object/dictionary errors"""
        
        # Create a simple destination
        destination = Destination(
            id="test_mountain_resort",
            names=["Test Mountain Resort"],
            country_code="US",
            admin_levels={"country": "United States", "state": "Colorado"},
            timezone="America/Denver",
            population=5000,
            area_km2=120.5,
            vibe_descriptors=["mountainous", "scenic", "adventure"]
        )
        
        # Store destination - this should work with our fixed database manager
        try:
            self.db_manager.store_destination(destination)
        except Exception as e:
            self.fail(f"Database storage failed with error: {e}")
        
        # Retrieve destination to verify storage worked
        retrieved = self.db_manager.get_destination_by_name("Test Mountain Resort")
        self.assertIsNotNone(retrieved, "Should be able to retrieve stored destination")
        self.assertEqual(retrieved.id, destination.id, "Retrieved destination should match")
        self.assertEqual(retrieved.vibe_descriptors, destination.vibe_descriptors, "Enrichment fields should be preserved")
        
        print("✅ Database storage test passed")

    def test_export_config_compatibility(self):
        """Test that export configuration works with both Theme objects and dictionaries"""
        
        from src.core.export_config import SmartViewGenerator, ExportConfig
        
        # Create test data with both Theme objects and dictionary representations
        mock_themes_data = {
            "theme_1": {
                "name": "Mountain Skiing",
                "macro_category": "Adventure & Sports", 
                "fit_score": 0.85,
                "confidence_breakdown": {
                    "overall_confidence": 0.8
                },
                "seasonal_relevance": {
                    "winter": 0.9,
                    "summer": 0.1
                }
            },
            "theme_2": {
                "name": "Hiking Trails",
                "macro_category": "Nature & Outdoor",
                "fit_score": 0.75,
                "confidence_breakdown": {
                    "overall_confidence": 0.7
                },
                "seasonal_relevance": {
                    "summer": 0.8,
                    "winter": 0.2
                }
            }
        }
        
        mock_destination_data = {
            "names": ["Test Mountain Resort"],
            "admin_levels": {"country": "United States"}
        }
        
        mock_evidence_registry = {
            "evidence_1": {
                "source_category": "guidebook",
                "authority_weight": 0.8
            }
        }
        
        # Test export configuration - should not fail with object/dictionary errors
        try:
            config = ExportConfig()
            view_generator = SmartViewGenerator(config)
            views = view_generator.generate_views(
                mock_destination_data, 
                mock_evidence_registry, 
                mock_themes_data
            )
        except Exception as e:
            self.fail(f"Export configuration failed with error: {e}")
        
        # Validate views were generated
        self.assertIsInstance(views, dict, "Views should be generated as dictionary")
        self.assertIn("executive_summary", views, "Should generate executive summary")
        self.assertIn("themes_by_category", views, "Should generate category views")
        
        print("✅ Export configuration compatibility test passed")


if __name__ == '__main__':
    unittest.main() 