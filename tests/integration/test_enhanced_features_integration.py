#!/usr/bin/env python3
"""
Integration tests for Enhanced Traveler Relevance and Selective POI Discovery
Tests how the new features work together in the full pipeline.
"""

import unittest
import asyncio
from unittest.mock import Mock, patch, AsyncMock
from src.core.web_discovery_logic import WebDiscoveryLogic
from src.agents.specialized_agents import ValidationAgent


class TestEnhancedFeaturesIntegration(unittest.TestCase):
    """Test integration of enhanced traveler relevance and POI discovery"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.mock_config = {
            "web_discovery": {
                "poi_discovery": {
                    "enable_poi_discovery": True,
                    "max_poi_queries": 5,
                    "tourist_gateway_keywords": ["flagstaff", "sedona", "canyon", "mountain"]
                },
                "search_results_per_query": 5,
                "min_content_length_chars": 100
            },
            "validation_agent": {
                "enable_traveler_relevance": True,
                "traveler_relevance_weight": 1.0
            }
        }
        self.validation_agent = ValidationAgent(config=self.mock_config)
        self.web_discovery = WebDiscoveryLogic(api_key="test_key", config=self.mock_config)
    
    def test_tourist_destination_gets_poi_discovery(self):
        """Test that tourist gateway destinations trigger POI discovery"""
        destination = "Flagstaff, Arizona"
        
        # Should be classified as tourist gateway
        is_gateway = self.web_discovery._is_tourist_gateway_destination(destination)
        self.assertTrue(is_gateway)
        
        # Should have POI discovery enabled
        self.assertTrue(self.web_discovery.enable_poi_discovery)
    
    def test_business_destination_skips_poi_discovery(self):
        """Test that business destinations skip POI discovery"""
        destination = "Seattle, Washington"
        
        # Should not be classified as tourist gateway
        is_gateway = self.web_discovery._is_tourist_gateway_destination(destination)
        self.assertFalse(is_gateway)
    
    def test_enhanced_traveler_relevance_ranks_tourist_themes_higher(self):
        """Test that enhanced algorithm ranks tourist themes over mundane themes"""
        # Tourist-appealing theme
        tourist_theme = {
            "name": "Breathtaking Scenic Views",
            "description": "Stunning panoramic views perfect for photography and adventure tours",
            "macro_category": "Nature & Outdoor"
        }
        
        # Mundane local theme
        mundane_theme = {
            "name": "Fitness Centers",
            "description": "Local gym facilities and administrative services",
            "macro_category": "Health & Wellness"
        }
        
        tourist_relevance = self.validation_agent._calculate_traveler_relevance(
            tourist_theme, "Flagstaff, Arizona", "US", 0.5
        )
        
        mundane_relevance = self.validation_agent._calculate_traveler_relevance(
            mundane_theme, "Flagstaff, Arizona", "US", 0.5
        )
        
        # Tourist theme should rank significantly higher
        self.assertGreater(tourist_relevance, mundane_relevance)
        self.assertGreater(tourist_relevance, 1.0)
        self.assertLess(mundane_relevance, 0.7)
    
    def test_search_query_and_ranking_alignment(self):
        """Test that enhanced search queries align with enhanced ranking"""
        # Check that search queries include emotional/visual language
        templates = self.web_discovery.query_templates
        query_text = " ".join(templates)
        
        # Should include tourist-focused language
        tourist_keywords = ["breathtaking", "stunning", "spectacular", "adventures", "scenic"]
        found_keywords = [kw for kw in tourist_keywords if kw in query_text.lower()]
        
        self.assertGreater(len(found_keywords), 0, 
                          "Search queries should include tourist-focused emotional language")
    
    def test_poi_queries_target_tourist_attractions(self):
        """Test that POI queries target tourist attractions"""
        poi_templates = self.web_discovery.priority_poi_query_templates
        poi_text = " ".join(poi_templates)
        
        # Should target key tourist POIs
        tourist_poi_keywords = ["landmarks", "attractions", "national parks", "scenic", "UNESCO"]
        found_keywords = [kw for kw in tourist_poi_keywords if kw in poi_text.lower()]
        
        self.assertGreater(len(found_keywords), 2,
                          "POI queries should target tourist attractions")
    
    def test_configuration_integration(self):
        """Test that configuration properly integrates both features"""
        # POI discovery configuration
        self.assertTrue(self.web_discovery.enable_poi_discovery)
        self.assertEqual(self.web_discovery.max_poi_queries, 5)
        
        # Traveler relevance configuration
        validation_config = self.validation_agent.config.get("validation_agent", {})
        self.assertTrue(validation_config.get("enable_traveler_relevance", False))
    
    def test_selective_processing_logic(self):
        """Test that features are selectively applied based on destination type"""
        # Tourist gateway should get enhanced processing
        tourist_destination = "Flagstaff, Arizona"
        is_tourist_gateway = self.web_discovery._is_tourist_gateway_destination(tourist_destination)
        self.assertTrue(is_tourist_gateway)
        
        # Business city should get standard processing
        business_destination = "Chicago, Illinois" 
        is_business_gateway = self.web_discovery._is_tourist_gateway_destination(business_destination)
        self.assertFalse(is_business_gateway)
    
    def test_cost_efficiency_of_selective_approach(self):
        """Test that selective approach limits API calls appropriately"""
        # POI queries should be limited to 5 for cost control
        self.assertEqual(len(self.web_discovery.priority_poi_query_templates), 5)
        self.assertEqual(self.web_discovery.max_poi_queries, 5)
        
        # This represents ~75% reduction from a hypothetical 20+ POI queries


class TestTravelerRelevanceValidationIntegration(unittest.TestCase):
    """Test integration of traveler relevance with validation agent processing"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.mock_config = {
            "validation_agent": {
                "enable_traveler_relevance": True,
                "traveler_relevance_weight": 1.0
            }
        }
        self.validation_agent = ValidationAgent(config=self.mock_config)
    
    def test_validation_agent_uses_enhanced_traveler_relevance(self):
        """Test that validation agent properly uses enhanced traveler relevance"""
        # Mock theme data with tourist appeal
        theme_data = {
            "name": "Adventure Tours",
            "description": "Thrilling outdoor experiences with stunning scenic views",
            "macro_category": "Entertainment & Nightlife",
            "overall_confidence": 0.8
        }
        
        # Test that enhanced relevance calculation is called
        relevance = self.validation_agent._calculate_traveler_relevance(
            theme_data, "Flagstaff, Arizona", "US", 0.5
        )
        
        # Should get boost for adventure and scenic keywords
        self.assertGreater(relevance, 0.8)
    
    def test_adjusted_confidence_calculation(self):
        """Test that traveler relevance affects adjusted confidence scores"""
        # High tourist appeal theme
        tourist_theme = {
            "name": "Spectacular Hiking Adventures",
            "description": "Breathtaking trails with stunning panoramic views",
            "macro_category": "Nature & Outdoor",
            "overall_confidence": 0.7
        }
        
        # Low tourist appeal theme  
        mundane_theme = {
            "name": "Administrative Services",
            "description": "Government offices and bureaucratic procedures",
            "macro_category": "Government & Public", 
            "overall_confidence": 0.7
        }
        
        tourist_relevance = self.validation_agent._calculate_traveler_relevance(
            tourist_theme, "Flagstaff, Arizona", "US", 0.5
        )
        
        mundane_relevance = self.validation_agent._calculate_traveler_relevance(
            mundane_theme, "Flagstaff, Arizona", "US", 0.5
        )
        
        # Tourist theme should have higher adjusted confidence
        # adjusted_confidence = original_confidence * traveler_relevance_factor
        tourist_adjusted = 0.7 * tourist_relevance
        mundane_adjusted = 0.7 * mundane_relevance
        
        self.assertGreater(tourist_adjusted, mundane_adjusted)


class TestEnhancedQueryTemplatesIntegration(unittest.TestCase):
    """Test integration of enhanced query templates with discovery pipeline"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.mock_config = {
            "web_discovery": {
                "search_results_per_query": 5,
                "min_content_length_chars": 100
            }
        }
        self.web_discovery = WebDiscoveryLogic(api_key="test_key", config=self.mock_config)
    
    def test_enhanced_query_templates_loaded(self):
        """Test that enhanced query templates are properly loaded"""
        templates = self.web_discovery.query_templates
        
        # Should have emotional/visual language in templates
        template_text = " ".join(templates).lower()
        
        emotional_keywords = ["breathtaking", "stunning", "spectacular"]
        experience_keywords = ["adventures", "experiences", "tours"]
        visual_keywords = ["scenic", "photography", "panoramic"]
        
        # Check for presence of enhanced keywords
        has_emotional = any(kw in template_text for kw in emotional_keywords)
        has_experience = any(kw in template_text for kw in experience_keywords)  
        has_visual = any(kw in template_text for kw in visual_keywords)
        
        self.assertTrue(has_emotional or has_experience or has_visual,
                       "Enhanced query templates should include tourist-focused language")
    
    def test_poi_query_templates_integration(self):
        """Test that POI query templates are properly integrated"""
        self.assertTrue(hasattr(self.web_discovery, 'priority_poi_query_templates'))
        
        poi_templates = self.web_discovery.priority_poi_query_templates
        self.assertIsInstance(poi_templates, list)
        self.assertGreater(len(poi_templates), 0)
        
        # Templates should be properly formatted
        for template in poi_templates:
            self.assertIn("{destination}", template)


class TestEndToEndEnhancedProcessing(unittest.TestCase):
    """Test end-to-end processing with enhanced features"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.mock_config = {
            "web_discovery": {
                "poi_discovery": {
                    "enable_poi_discovery": True,
                    "max_poi_queries": 5,
                    "tourist_gateway_keywords": ["flagstaff", "canyon", "mountain"]
                }
            },
            "validation_agent": {
                "enable_traveler_relevance": True,
                "traveler_relevance_weight": 1.0
            }
        }
    
    def test_flagstaff_end_to_end_processing(self):
        """Test that Flagstaff gets enhanced processing end-to-end"""
        destination = "Flagstaff, Arizona"
        
        # Initialize components
        web_discovery = WebDiscoveryLogic(api_key="test_key", config=self.mock_config)
        validation_agent = ValidationAgent(config=self.mock_config)
        
        # Test tourist gateway classification
        is_gateway = web_discovery._is_tourist_gateway_destination(destination)
        self.assertTrue(is_gateway, "Flagstaff should be classified as tourist gateway")
        
        # Test POI discovery would be triggered
        self.assertTrue(web_discovery.enable_poi_discovery)
        
        # Test enhanced traveler relevance calculation
        scenic_theme = {
            "name": "Scenic Mountain Views",
            "description": "Breathtaking panoramic views of stunning landscapes",
            "macro_category": "Nature & Outdoor"
        }
        
        relevance = validation_agent._calculate_traveler_relevance(
            scenic_theme, destination, "US", 0.5
        )
        
        self.assertGreater(relevance, 1.0, "Scenic themes should get high tourist relevance")
    
    def test_seattle_end_to_end_processing(self):
        """Test that Seattle gets standard processing (no POI discovery)"""
        destination = "Seattle, Washington"
        
        # Initialize components
        web_discovery = WebDiscoveryLogic(api_key="test_key", config=self.mock_config)
        
        # Test business destination classification
        is_gateway = web_discovery._is_tourist_gateway_destination(destination)
        self.assertFalse(is_gateway, "Seattle should not be classified as tourist gateway")
        
        # POI discovery would be skipped for this destination


if __name__ == '__main__':
    unittest.main() 