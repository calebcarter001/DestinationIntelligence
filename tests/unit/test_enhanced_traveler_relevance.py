#!/usr/bin/env python3
"""
Unit tests for Enhanced Traveler Relevance Algorithm
Tests the new heuristics for ranking tourist-relevant themes over local utility themes.
"""

import unittest
from unittest.mock import Mock, patch
from src.agents.specialized_agents import ValidationAgent
import pytest
import asyncio

@pytest.fixture
def validation_agent():
    """Create a ValidationAgent instance for testing."""
    config = {"sentence_transformer_model": "fake-model"}
    agent = ValidationAgent(config=config)
    return agent

class TestEnhancedTravelerRelevance:
    """Test the enhanced traveler relevance calculation."""
    
    @pytest.mark.asyncio
    async def test_emotional_language_detection(self, validation_agent):
        """Test that emotional language keywords increase relevance scores."""
        theme_data = {
            "name": "Breathtaking Mountain Views",
            "description": "Stunning and spectacular scenic overlooks",
            "macro_category": "nature & outdoor"
        }
        
        score = validation_agent._calculate_traveler_relevance(
            theme_data, "Test City", "US", 0.5
        )
        
        # Base: 0.8 + emotional boost (3 words * 0.2 = 0.6) = 1.4
        assert score > 1.2, f"Expected emotional boost, got {score}"
    
    @pytest.mark.asyncio 
    async def test_adventure_persona_targeting(self, validation_agent):
        """Test that adventure-related themes get persona boosts."""
        theme_data = {
            "name": "Extreme Adventure Hiking",
            "description": "Thrilling outdoor climbing experiences", 
            "macro_category": "nature & outdoor"
        }
        
        score = validation_agent._calculate_traveler_relevance(
            theme_data, "Test City", "US", 0.5
        )
        
        # Base: 0.8 + adventure persona: 0.18 + experience boost + emotional = >1.2
        assert score > 1.2, f"Expected adventure boost, got {score}"
    
    @pytest.mark.asyncio
    async def test_romance_persona_targeting(self, validation_agent):
        """Test that romantic themes get appropriate boosts."""
        theme_data = {
            "name": "Romantic Sunset Spots",
            "description": "Intimate and cozy venues for couples",
            "macro_category": "entertainment & nightlife"
        }
        
        score = validation_agent._calculate_traveler_relevance(
            theme_data, "Test City", "US", 0.5
        )
        
        # Base: 0.8 + romantic persona: 0.15 + emotional boost = >1.0
        assert score > 1.0, f"Expected romantic boost, got {score}"
    
    @pytest.mark.asyncio
    async def test_visual_appeal_detection(self, validation_agent):
        """Test that visual appeal keywords increase scores."""
        theme_data = {
            "name": "Scenic Photography Spots",
            "description": "Beautiful panoramic views and landscapes",
            "macro_category": "distinctive features"
        }
        
        score = validation_agent._calculate_traveler_relevance(
            theme_data, "Test City", "US", 0.5
        )
        
        # Base: 0.8 + visual boost (multiple keywords) = >1.0
        assert score > 1.0, f"Expected visual appeal boost, got {score}"
    
    @pytest.mark.asyncio
    async def test_mundane_service_penalties(self, validation_agent):
        """Test that mundane services get penalized."""
        theme_data = {
            "name": "Local Gym and Fitness Center", 
            "description": "Medical center and hospital services",
            "macro_category": "health & medical"
        }
        
        score = validation_agent._calculate_traveler_relevance(
            theme_data, "Test City", "US", 0.5
        )
        
        # Base: 0.3 - service penalties = low score
        assert score < 0.4, f"Expected penalty for mundane services, got {score}"
    
    @pytest.mark.asyncio
    async def test_combined_heuristics_exceptional_theme(self, validation_agent):
        """Test that themes with multiple positive signals get high scores."""
        theme_data = {
            "name": "World-Class Romantic Adventure Tours",
            "description": "Breathtaking scenic hiking experiences with stunning photography opportunities",
            "macro_category": "authentic experiences"
        }
        
        score = validation_agent._calculate_traveler_relevance(
            theme_data, "Test City", "US", 0.5
        )
        
        # Multiple boosts: high category + emotional + adventure + romantic + visual + quality = >1.5
        assert score > 1.4, f"Expected very high score for exceptional theme, got {score}"
    
    @pytest.mark.asyncio
    async def test_geographic_name_penalty(self, validation_agent):
        """Test that generic geographic themes get severely penalized."""
        theme_data = {
            "name": "Arizona General",
            "description": "General Arizona information",
            "macro_category": "general"
        }
        
        # Mock the generic_theme_stop_list to include "general"
        validation_agent.generic_theme_stop_list = ["general", "arizona general"]
        
        score = validation_agent._calculate_traveler_relevance(
            theme_data, "Flagstaff", "US", 0.5
        )
        
        assert score == 0.1, f"Expected severe penalty for geographic name, got {score}"
    
    @pytest.mark.asyncio
    async def test_family_persona_targeting(self, validation_agent):
        """Test that family-friendly themes get appropriate boosts."""
        theme_data = {
            "name": "Family Educational Activities",
            "description": "Safe and fun experiences for kids and children",
            "macro_category": "family & education"
        }
        
        score = validation_agent._calculate_traveler_relevance(
            theme_data, "Test City", "US", 0.5
        )
        
        # Base: 0.6 + family persona: 0.12 = >0.7
        assert score > 0.7, f"Expected family boost, got {score}"
    
    @pytest.mark.asyncio
    async def test_cultural_persona_targeting(self, validation_agent):
        """Test that cultural themes get appropriate boosts."""
        theme_data = {
            "name": "Traditional Cultural Heritage",
            "description": "Authentic historical local experiences",
            "macro_category": "cultural identity & atmosphere"
        }
        
        score = validation_agent._calculate_traveler_relevance(
            theme_data, "Test City", "US", 0.5
        )
        
        # Base: 0.8 + cultural persona: 0.15 + emotional = >1.0
        assert score > 1.0, f"Expected cultural boost, got {score}"
    
    @pytest.mark.asyncio
    async def test_quality_indicators_boost(self, validation_agent):
        """Test that quality indicators increase scores."""
        theme_data = {
            "name": "Must-See Iconic Landmarks",
            "description": "World-class famous attractions",
            "macro_category": "distinctive features"
        }
        
        score = validation_agent._calculate_traveler_relevance(
            theme_data, "Test City", "US", 0.5
        )
        
        # Base: 0.8 + quality boost (multiple indicators) = >1.0
        assert score > 1.0, f"Expected quality boost, got {score}"
    
    @pytest.mark.asyncio
    async def test_experience_vs_service_classification(self, validation_agent):
        """Test experience themes vs service themes."""
        experience_theme = {
            "name": "Adventure Tours and Excursions",
            "description": "Exploration and discovery experiences",
            "macro_category": "authentic experiences"
        }
        
        service_theme = {
            "name": "Banking and Administrative Services",
            "description": "Government office and insurance services",
            "macro_category": "logistics & planning"
        }
        
        exp_score = validation_agent._calculate_traveler_relevance(
            experience_theme, "Test City", "US", 0.5
        )
        
        svc_score = validation_agent._calculate_traveler_relevance(
            service_theme, "Test City", "US", 0.5
        )
        
        assert exp_score > svc_score, f"Experience theme {exp_score} should beat service theme {svc_score}"
        assert exp_score > 1.0, f"Experience theme should be boosted above 1.0, got {exp_score}"
    
    @pytest.mark.asyncio
    async def test_low_tourist_relevance_category(self, validation_agent):
        """Test that low tourist relevance categories get lower base scores."""
        theme_data = {
            "name": "Medical Transportation",
            "description": "Healthcare access and medical logistics",
            "macro_category": "health & medical"
        }
        
        score = validation_agent._calculate_traveler_relevance(
            theme_data, "Test City", "US", 0.5
        )
        
        # Low category base: 0.3, no positive boosts, some penalties = low score
        assert score < 0.5, f"Expected low score for medical theme, got {score}"
    
    @pytest.mark.asyncio
    async def test_none_handling(self, validation_agent):
        """Test that None values are handled gracefully."""
        theme_data = {
            "name": None,
            "description": None,
            "macro_category": None
        }
        
        score = validation_agent._calculate_traveler_relevance(
            theme_data, "Test City", "US", 0.5
        )
        
        # Should not crash and return default
        assert isinstance(score, float), f"Should return float, got {type(score)}"
        assert 0.0 <= score <= 2.0, f"Score should be in valid range, got {score}"
    
    @pytest.mark.asyncio
    async def test_invalid_input_handling(self, validation_agent):
        """Test handling of invalid input types."""
        # Test with non-dict input
        score = validation_agent._calculate_traveler_relevance(
            "invalid", "Test City", "US", 0.5
        )
        
        assert score == 0.5, f"Should return default for invalid input, got {score}"
    
    @pytest.mark.asyncio
    async def test_empty_string_handling(self, validation_agent):
        """Test handling of empty strings."""
        theme_data = {
            "name": "",
            "description": "",
            "macro_category": ""
        }
        
        score = validation_agent._calculate_traveler_relevance(
            theme_data, "Test City", "US", 0.5
        )
        
        # Should use default base relevance
        assert score == 0.5, f"Expected default score for empty theme, got {score}"
    
    @pytest.mark.asyncio
    async def test_upper_bound_enforcement(self, validation_agent):
        """Test that scores don't exceed the upper bound."""
        # Create a theme with maximum possible boosts
        theme_data = {
            "name": "breathtaking stunning spectacular thrilling romantic magical adventure hiking cultural traditional must-see world-class",
            "description": "scenic views beautiful photography tours experiences attractions adventures exploration discovery",
            "macro_category": "authentic experiences"
        }
        
        score = validation_agent._calculate_traveler_relevance(
            theme_data, "Test City", "US", 0.5
        )
        
        assert score <= 2.0, f"Score should not exceed upper bound, got {score}"
        assert score > 1.5, f"Score with all boosts should be high, got {score}"
    
    @pytest.mark.asyncio
    async def test_lower_bound_enforcement(self, validation_agent):
        """Test that scores don't go below the lower bound."""
        theme_data = {
            "name": "terrible awful boring administrative bureaucratic fitness medical gym hospital clinic",
            "description": "boring government office banking insurance legal services repairs",
            "macro_category": "health & medical"
        }
        
        score = validation_agent._calculate_traveler_relevance(
            theme_data, "Test City", "US", 0.5
        )
        
        assert score >= 0.05, f"Score should not go below lower bound, got {score}"


class TestTravelerRelevanceEdgeCases(unittest.TestCase):
    """Test edge cases and error handling in traveler relevance calculation"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.mock_config = {
            "validation_agent": {
                "enable_traveler_relevance": True,
                "traveler_relevance_weight": 1.0
            }
        }
        self.validation_agent = ValidationAgent(config=self.mock_config)
    
    def test_empty_theme_data(self):
        """Test handling of empty theme data"""
        theme_data = {}
        
        relevance = self.validation_agent._calculate_traveler_relevance(
            theme_data, "Flagstaff, Arizona", "US", 0.5
        )
        
        # Should return reasonable default
        self.assertGreater(relevance, 0.1)
        self.assertLess(relevance, 1.0)
    
    def test_missing_fields(self):
        """Test handling of missing fields in theme data"""
        theme_data = {
            "name": "Test Theme"
            # Missing description and macro_category
        }
        
        relevance = self.validation_agent._calculate_traveler_relevance(
            theme_data, "Flagstaff, Arizona", "US", 0.5
        )
        
        # Should handle gracefully
        self.assertIsInstance(relevance, float)
        self.assertGreater(relevance, 0.0)
    
    def test_none_values(self):
        """Test handling of None values"""
        theme_data = {
            "name": None,
            "description": None,
            "macro_category": None
        }
        
        relevance = self.validation_agent._calculate_traveler_relevance(
            theme_data, "Flagstaff, Arizona", "US", 0.5
        )
        
        # Should handle gracefully without crashing
        self.assertIsInstance(relevance, float)
        self.assertGreater(relevance, 0.0)
    
    def test_case_sensitivity(self):
        """Test that keyword matching is case insensitive"""
        theme_data = {
            "name": "BREATHTAKING SCENIC VIEWS",
            "description": "STUNNING PANORAMIC LANDSCAPES",
            "macro_category": "NATURE & OUTDOOR"
        }
        
        relevance = self.validation_agent._calculate_traveler_relevance(
            theme_data, "Flagstaff, Arizona", "US", 0.5
        )
        
        # Should still get emotional language boost despite uppercase
        self.assertGreater(relevance, 1.0)


if __name__ == '__main__':
    unittest.main() 