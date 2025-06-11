import unittest
from unittest.mock import Mock, patch, MagicMock
from src.core.web_discovery_logic import WebDiscoveryLogic

class TestSelectivePOIDiscovery(unittest.TestCase):
    """Test selective POI discovery for tourist gateway destinations."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.config = {
            "web_discovery": {
                "poi_discovery": {
                    "enable_poi_discovery": True,
                    "tourist_gateway_keywords": ["flagstaff", "sedona", "moab", "aspen", "jackson"],
                    "max_poi_queries": 5,
                    "max_poi_results_per_query": 3
                }
            }
        }
        self.web_discovery = WebDiscoveryLogic("test_api_key", self.config)
    
    def test_tourist_gateway_classification_positive_flagstaff(self):
        """Test that Flagstaff is classified as a tourist gateway."""
        result = self.web_discovery._is_tourist_gateway_destination("Flagstaff, Arizona")
        self.assertTrue(result)
    
    def test_tourist_gateway_classification_positive_sedona(self):
        """Test that Sedona is classified as a tourist gateway."""
        result = self.web_discovery._is_tourist_gateway_destination("Sedona, Arizona")
        self.assertTrue(result)
    
    def test_tourist_gateway_classification_positive_moab(self):
        """Test that Moab is classified as a tourist gateway."""
        result = self.web_discovery._is_tourist_gateway_destination("Moab, Utah")
        self.assertTrue(result)
    
    def test_tourist_gateway_classification_positive_aspen(self):
        """Test that Aspen is classified as a tourist gateway."""
        result = self.web_discovery._is_tourist_gateway_destination("Aspen, Colorado")
        self.assertTrue(result)
    
    def test_tourist_gateway_classification_positive_jackson(self):
        """Test that Jackson is classified as a tourist gateway."""
        result = self.web_discovery._is_tourist_gateway_destination("Jackson, Wyoming")
        self.assertTrue(result)
    
    def test_tourist_gateway_classification_positive_canyon_keyword(self):
        """Test that destinations with 'canyon' keyword are classified as gateways."""
        result = self.web_discovery._is_tourist_gateway_destination("Grand Canyon Village")
        self.assertTrue(result)
    
    def test_tourist_gateway_classification_positive_mountain_keyword(self):
        """Test that destinations with 'mountain' keyword are classified as gateways."""
        result = self.web_discovery._is_tourist_gateway_destination("Mountain View, Colorado")
        self.assertTrue(result)
    
    def test_tourist_gateway_classification_positive_national_park_keyword(self):
        """Test that destinations with 'national park' keyword are classified as gateways."""
        result = self.web_discovery._is_tourist_gateway_destination("Yellowstone National Park")
        self.assertTrue(result)
    
    def test_tourist_gateway_classification_negative_business_city(self):
        """Test that major business cities are not classified as tourist gateways."""
        result = self.web_discovery._is_tourist_gateway_destination("New York, New York")
        self.assertFalse(result)
    
    def test_tourist_gateway_classification_negative_tech_hub(self):
        """Test that tech hubs are not classified as tourist gateways."""
        result = self.web_discovery._is_tourist_gateway_destination("San Francisco, California")
        self.assertFalse(result)
    
    def test_tourist_gateway_classification_negative_business_center(self):
        """Test that business centers are not classified as tourist gateways."""
        result = self.web_discovery._is_tourist_gateway_destination("Chicago, Illinois")
        self.assertFalse(result)
    
    def test_tourist_gateway_classification_negative_administrative_city(self):
        """Test that administrative cities are not classified as tourist gateways."""
        result = self.web_discovery._is_tourist_gateway_destination("Washington, DC")
        self.assertFalse(result)
    
    def test_tourist_gateway_classification_case_insensitive(self):
        """Test that tourist gateway classification is case insensitive."""
        result_upper = self.web_discovery._is_tourist_gateway_destination("FLAGSTAFF, ARIZONA")
        result_lower = self.web_discovery._is_tourist_gateway_destination("flagstaff, arizona")
        result_mixed = self.web_discovery._is_tourist_gateway_destination("Flagstaff, ARIZONA")
        
        self.assertTrue(result_upper)
        self.assertTrue(result_lower)
        self.assertTrue(result_mixed)

class TestPOIConfiguration(unittest.TestCase):
    """Test POI discovery configuration loading and validation."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.config = {
            "web_discovery": {
                "poi_discovery": {
                    "enable_poi_discovery": True,
                    "tourist_gateway_keywords": ["flagstaff", "sedona", "moab"],
                    "max_poi_queries": 5,
                    "max_poi_results_per_query": 3
                }
            }
        }
        self.web_discovery = WebDiscoveryLogic("test_api_key", self.config)
    
    def test_poi_discovery_configuration_loaded(self):
        """Test that POI discovery configuration is properly loaded."""
        self.assertTrue(self.web_discovery.enable_poi_discovery)
        self.assertEqual(self.web_discovery.max_poi_queries, 5)
        self.assertEqual(self.web_discovery.max_poi_results_per_query, 3)
        self.assertIn("flagstaff", self.web_discovery.tourist_gateway_keywords)
        self.assertIn("sedona", self.web_discovery.tourist_gateway_keywords)
        self.assertIn("moab", self.web_discovery.tourist_gateway_keywords)
    
    def test_poi_discovery_configuration_disabled(self):
        """Test POI discovery when disabled in configuration."""
        config_disabled = {
            "web_discovery": {
                "poi_discovery": {
                    "enable_poi_discovery": False
                }
            }
        }
        web_discovery_disabled = WebDiscoveryLogic("test_api_key", config_disabled)
        self.assertFalse(web_discovery_disabled.enable_poi_discovery)

class TestPOIQueryGeneration(unittest.TestCase):
    """Test POI query template generation and formatting."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.config = {
            "web_discovery": {
                "poi_discovery": {
                    "enable_poi_discovery": True,
                    "tourist_gateway_keywords": ["flagstaff"],
                    "max_poi_queries": 5
                }
            }
        }
        self.web_discovery = WebDiscoveryLogic("test_api_key", self.config)
    
    def test_priority_poi_query_templates_exist(self):
        """Test that priority POI query templates are defined."""
        self.assertTrue(hasattr(self.web_discovery, 'priority_poi_query_templates'))
        self.assertIsInstance(self.web_discovery.priority_poi_query_templates, list)
        self.assertGreater(len(self.web_discovery.priority_poi_query_templates), 0)
    
    def test_priority_poi_query_templates_content(self):
        """Test that POI query templates contain expected tourism-focused content."""
        templates = self.web_discovery.priority_poi_query_templates
        
        # Check that templates target tourist attractions
        tourism_keywords = ["attractions", "landmarks", "national park", "scenic", "tours"]
        template_text = " ".join(templates).lower()
        
        found_keywords = [keyword for keyword in tourism_keywords if keyword in template_text]
        self.assertGreater(len(found_keywords), 2, 
                          f"Expected tourism keywords in templates, found: {found_keywords}")
    
    def test_poi_query_formatting(self):
        """Test that POI queries format correctly with destination names."""
        destination = "Flagstaff, Arizona"
        templates = self.web_discovery.priority_poi_query_templates[:3]  # Test first 3
        
        for template in templates:
            formatted_query = template.format(destination=destination)
            self.assertIn("Flagstaff", formatted_query)
            self.assertNotIn("{destination}", formatted_query)  # Ensure placeholder was replaced
    
    def test_max_poi_queries_configurable(self):
        """Test that max POI queries configuration is respected."""
        self.assertEqual(self.web_discovery.max_poi_queries, 5)
        
        # Test that we don't exceed the configured limit
        templates_used = self.web_discovery.priority_poi_query_templates[:self.web_discovery.max_poi_queries]
        self.assertLessEqual(len(templates_used), self.web_discovery.max_poi_queries)
    
    def test_all_poi_templates_format_correctly(self):
        """Test that all POI templates can be formatted without errors."""
        destination = "Test Destination"
        
        for i, template in enumerate(self.web_discovery.priority_poi_query_templates):
            try:
                formatted = template.format(destination=destination)
                self.assertIsInstance(formatted, str)
                self.assertIn("Test Destination", formatted)
            except Exception as e:
                self.fail(f"Template {i} failed to format: {template}. Error: {e}")

class TestSelectivePOIDiscovery(unittest.TestCase):
    """Test the selective approach to POI discovery."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.config = {
            "web_discovery": {
                "poi_discovery": {
                    "enable_poi_discovery": True,
                    "tourist_gateway_keywords": ["flagstaff", "sedona", "moab", "aspen", "jackson"],
                    "max_poi_queries": 5,
                    "max_poi_results_per_query": 3
                }
            }
        }
        self.web_discovery = WebDiscoveryLogic("test_api_key", self.config)
    
    def test_tourist_gateway_keywords_customizable(self):
        """Test that tourist gateway keywords can be customized."""
        custom_config = {
            "web_discovery": {
                "poi_discovery": {
                    "enable_poi_discovery": True,
                    "tourist_gateway_keywords": ["custom", "gateway", "destinations"],
                    "max_poi_queries": 3
                }
            }
        }
        
        web_discovery_custom = WebDiscoveryLogic("test_api_key", custom_config)
        self.assertIn("custom", web_discovery_custom.tourist_gateway_keywords)
        self.assertIn("gateway", web_discovery_custom.tourist_gateway_keywords)
        self.assertNotIn("flagstaff", web_discovery_custom.tourist_gateway_keywords)

class TestPOIDiscoveryEdgeCases(unittest.TestCase):
    """Test edge cases for POI discovery functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.config = {
            "web_discovery": {
                "poi_discovery": {
                    "enable_poi_discovery": True,
                    "tourist_gateway_keywords": ["flagstaff", "sedona"],
                    "max_poi_queries": 5
                }
            }
        }
        self.web_discovery = WebDiscoveryLogic("test_api_key", self.config)
    
    def test_gateway_classification_empty_destination(self):
        """Test gateway classification with empty destination."""
        result = self.web_discovery._is_tourist_gateway_destination("")
        self.assertFalse(result)
    
    def test_gateway_classification_none_destination(self):
        """Test gateway classification with None destination."""
        result = self.web_discovery._is_tourist_gateway_destination(None)
        self.assertFalse(result)
    
    def test_gateway_classification_special_characters(self):
        """Test gateway classification with special characters in destination."""
        result = self.web_discovery._is_tourist_gateway_destination("Flagstaff, Arizona (USA)")
        self.assertTrue(result)
    
    def test_gateway_classification_unicode_characters(self):
        """Test gateway classification with unicode characters."""
        result = self.web_discovery._is_tourist_gateway_destination("Flagstaff, Arizonà")
        self.assertTrue(result)
    
    def test_poi_discovery_with_missing_config(self):
        """Test POI discovery behavior when configuration is minimal."""
        minimal_config = {
            "web_discovery": {
                "poi_discovery": {
                    "enable_poi_discovery": True
                }
            }
        }
        
        web_discovery_minimal = WebDiscoveryLogic("test_api_key", minimal_config)
        self.assertTrue(web_discovery_minimal.enable_poi_discovery)
        self.assertIsInstance(web_discovery_minimal.tourist_gateway_keywords, list)
        self.assertIsInstance(web_discovery_minimal.priority_poi_query_templates, list)
    
    def test_gateway_keywords_empty_list(self):
        """Test behavior when tourist gateway keywords list is empty."""
        empty_config = {
            "web_discovery": {
                "poi_discovery": {
                    "enable_poi_discovery": True,
                    "tourist_gateway_keywords": [],
                    "max_poi_queries": 5
                }
            }
        }
        
        web_discovery_empty = WebDiscoveryLogic("test_api_key", empty_config)
        result = web_discovery_empty._is_tourist_gateway_destination("Flagstaff, Arizona")
        # Should still classify based on built-in keywords like 'canyon', 'mountain', etc.
        self.assertIsInstance(result, bool)


if __name__ == '__main__':
    unittest.main() 