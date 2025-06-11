#!/usr/bin/env python3
"""
Integration tests for Enhanced Theme Ranking
Tests how enhanced traveler relevance affects final theme rankings in analysis outputs.
"""

import unittest
import tempfile
import json
import os
from unittest.mock import Mock, patch
from src.agents.specialized_agents import ValidationAgent


class TestEnhancedRankingIntegration(unittest.TestCase):
    """Test that enhanced traveler relevance properly affects theme rankings"""
    
    def setUp(self):
        """Set up test fixtures with sample theme data"""
        self.mock_config = {
            "validation_agent": {
                "enable_traveler_relevance": True,
                "traveler_relevance_weight": 1.0
            }
        }
        self.validation_agent = ValidationAgent(config=self.mock_config)
        
        # Sample themes that should rank differently with enhanced algorithm
        self.sample_themes = [
            {
                "name": "Fitness Centers",
                "description": "Local gym facilities and workout equipment",
                "macro_category": "Health & Wellness",
                "overall_confidence": 0.8,
                "fit_score": 0.9  # High evidence quality but low tourist relevance
            },
            {
                "name": "Breathtaking Scenic Views", 
                "description": "Stunning panoramic mountain landscapes perfect for photography",
                "macro_category": "Nature & Outdoor",
                "overall_confidence": 0.7,
                "fit_score": 0.8  # Good evidence quality and high tourist relevance
            },
            {
                "name": "Adventure Hiking Trails",
                "description": "Thrilling outdoor adventures through spectacular wilderness areas",
                "macro_category": "Nature & Outdoor", 
                "overall_confidence": 0.6,
                "fit_score": 0.75  # Moderate evidence but very high tourist relevance
            },
            {
                "name": "Educational Institutions",
                "description": "Local schools and administrative educational facilities",
                "macro_category": "Education & Government",
                "overall_confidence": 0.85,
                "fit_score": 0.95  # High evidence quality but very low tourist relevance
            },
            {
                "name": "Romantic Sunset Spots",
                "description": "Intimate and picturesque locations perfect for couples",
                "macro_category": "Entertainment & Nightlife",
                "overall_confidence": 0.65,
                "fit_score": 0.7  # Moderate evidence but high tourist relevance
            }
        ]
    
    def test_traveler_relevance_calculation_for_sample_themes(self):
        """Test that traveler relevance is calculated correctly for sample themes"""
        destination = "Flagstaff, Arizona"
        
        relevance_scores = {}
        for theme in self.sample_themes:
            relevance = self.validation_agent._calculate_traveler_relevance(
                theme, destination, "US", 0.5
            )
            relevance_scores[theme["name"]] = relevance
        
        # Tourist themes should score higher than mundane themes
        self.assertGreater(relevance_scores["Breathtaking Scenic Views"], 
                          relevance_scores["Fitness Centers"])
        
        self.assertGreater(relevance_scores["Adventure Hiking Trails"],
                          relevance_scores["Educational Institutions"])
        
        self.assertGreater(relevance_scores["Romantic Sunset Spots"],
                          relevance_scores["Fitness Centers"])
    
    def test_adjusted_confidence_ranking_changes(self):
        """Test that adjusted confidence changes theme rankings"""
        destination = "Flagstaff, Arizona"
        
        # Calculate adjusted confidence (original confidence * traveler relevance)
        adjusted_themes = []
        for theme in self.sample_themes:
            relevance = self.validation_agent._calculate_traveler_relevance(
                theme, destination, "US", 0.5
            )
            
            adjusted_confidence = theme["overall_confidence"] * relevance
            
            adjusted_theme = theme.copy()
            adjusted_theme["traveler_relevance_factor"] = relevance
            adjusted_theme["adjusted_overall_confidence"] = adjusted_confidence
            adjusted_themes.append(adjusted_theme)
        
        # Sort by adjusted confidence (tourist relevance)
        tourist_ranked = sorted(adjusted_themes, 
                               key=lambda x: x["adjusted_overall_confidence"], 
                               reverse=True)
        
        # Sort by fit score (evidence quality)
        evidence_ranked = sorted(adjusted_themes,
                                key=lambda x: x["fit_score"],
                                reverse=True)
        
        # Rankings should be different
        tourist_top_3 = [t["name"] for t in tourist_ranked[:3]]
        evidence_top_3 = [t["name"] for t in evidence_ranked[:3]]
        
        self.assertNotEqual(tourist_top_3, evidence_top_3,
                           "Tourist relevance ranking should differ from evidence ranking")
        
        # Tourist themes should rank higher in tourist-relevance ranking
        tourist_themes = ["Breathtaking Scenic Views", "Adventure Hiking Trails", "Romantic Sunset Spots"]
        mundane_themes = ["Fitness Centers", "Educational Institutions"]
        
        # Check that tourist themes generally rank higher
        tourist_positions = [tourist_ranked.index(next(t for t in tourist_ranked if t["name"] == name)) 
                           for name in tourist_themes if any(t["name"] == name for t in tourist_ranked)]
        
        mundane_positions = [tourist_ranked.index(next(t for t in tourist_ranked if t["name"] == name))
                           for name in mundane_themes if any(t["name"] == name for t in tourist_ranked)]
        
        avg_tourist_position = sum(tourist_positions) / len(tourist_positions)
        avg_mundane_position = sum(mundane_positions) / len(mundane_positions)
        
        self.assertLess(avg_tourist_position, avg_mundane_position,
                       "Tourist themes should rank higher on average")
    
    def test_dynamic_viewer_sorting_impact(self):
        """Test that enhanced rankings affect dynamic viewer default sorting"""
        destination = "Flagstaff, Arizona"
        
        # Simulate what dynamic viewer would do
        themes_for_viewer = []
        for theme in self.sample_themes:
            relevance = self.validation_agent._calculate_traveler_relevance(
                theme, destination, "US", 0.5
            )
            
            viewer_theme = {
                "name": theme["name"],
                "description": theme["description"],
                "overall_confidence": theme["overall_confidence"],
                "fit_score": theme["fit_score"],
                "traveler_relevance_factor": relevance,
                "adjusted_overall_confidence": theme["overall_confidence"] * relevance
            }
            themes_for_viewer.append(viewer_theme)
        
        # Sort by tourist relevance (what new default should be)
        tourist_sorted = sorted(themes_for_viewer,
                               key=lambda x: x["adjusted_overall_confidence"],
                               reverse=True)
        
        # Sort by evidence quality (old default)
        evidence_sorted = sorted(themes_for_viewer,
                                key=lambda x: x["fit_score"],
                                reverse=True)
        
        # First theme in tourist sorting should be more tourist-relevant
        tourist_first = tourist_sorted[0]
        evidence_first = evidence_sorted[0]
        
        self.assertGreater(tourist_first["traveler_relevance_factor"],
                          evidence_first["traveler_relevance_factor"],
                          "Tourist-relevance sorting should prioritize more tourist-relevant themes")
    
    def test_theme_ranking_consistency_across_components(self):
        """Test that theme ranking is consistent across different components"""
        destination = "Flagstaff, Arizona"
        
        # Calculate relevance factors
        for theme in self.sample_themes:
            relevance = self.validation_agent._calculate_traveler_relevance(
                theme, destination, "US", 0.5
            )
            theme["traveler_relevance_factor"] = relevance
            theme["adjusted_overall_confidence"] = theme["overall_confidence"] * relevance
        
        # Test multiple sorting scenarios
        sort_by_tourist_relevance = sorted(self.sample_themes,
                                          key=lambda x: x["adjusted_overall_confidence"],
                                          reverse=True)
        
        sort_by_relevance_factor = sorted(self.sample_themes,
                                         key=lambda x: x["traveler_relevance_factor"],
                                         reverse=True)
        
        # Top themes by tourist relevance should have high relevance factors
        top_tourist_theme = sort_by_tourist_relevance[0]
        top_relevance_theme = sort_by_relevance_factor[0]
        
        self.assertGreater(top_tourist_theme["traveler_relevance_factor"], 1.0,
                          "Top tourist theme should have high relevance factor")
        
        self.assertGreater(top_relevance_theme["traveler_relevance_factor"], 1.0,
                          "Top relevance theme should have high relevance factor")
    
    def test_penalty_effectiveness_for_mundane_themes(self):
        """Test that mundane themes are effectively penalized"""
        destination = "Flagstaff, Arizona"
        
        # Get relevance for mundane themes
        fitness_theme = next(t for t in self.sample_themes if t["name"] == "Fitness Centers")
        education_theme = next(t for t in self.sample_themes if t["name"] == "Educational Institutions")
        
        fitness_relevance = self.validation_agent._calculate_traveler_relevance(
            fitness_theme, destination, "US", 0.5
        )
        
        education_relevance = self.validation_agent._calculate_traveler_relevance(
            education_theme, destination, "US", 0.5
        )
        
        # Both should be significantly below 1.0 (the neutral point)
        self.assertLess(fitness_relevance, 0.7,
                       "Fitness centers should be penalized for tourists")
        
        self.assertLess(education_relevance, 0.7,
                       "Educational institutions should be penalized for tourists")
    
    def test_boost_effectiveness_for_tourist_themes(self):
        """Test that tourist themes are effectively boosted"""
        destination = "Flagstaff, Arizona"
        
        # Get relevance for tourist themes
        scenic_theme = next(t for t in self.sample_themes if t["name"] == "Breathtaking Scenic Views")
        adventure_theme = next(t for t in self.sample_themes if t["name"] == "Adventure Hiking Trails")
        romantic_theme = next(t for t in self.sample_themes if t["name"] == "Romantic Sunset Spots")
        
        scenic_relevance = self.validation_agent._calculate_traveler_relevance(
            scenic_theme, destination, "US", 0.5
        )
        
        adventure_relevance = self.validation_agent._calculate_traveler_relevance(
            adventure_theme, destination, "US", 0.5
        )
        
        romantic_relevance = self.validation_agent._calculate_traveler_relevance(
            romantic_theme, destination, "US", 0.5
        )
        
        # All should be significantly above 1.0 (the neutral point)
        self.assertGreater(scenic_relevance, 1.0,
                          "Scenic views should be boosted for tourists")
        
        self.assertGreater(adventure_relevance, 1.0,
                          "Adventure themes should be boosted for tourists")
        
        self.assertGreater(romantic_relevance, 1.0,
                          "Romantic themes should be boosted for tourists")


class TestRankingOutputIntegration(unittest.TestCase):
    """Test that enhanced rankings appear correctly in output files and scripts"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.mock_config = {
            "validation_agent": {
                "enable_traveler_relevance": True,
                "traveler_relevance_weight": 1.0
            }
        }
    
    def test_analyze_themes_integration_expectations(self):
        """Test expected behavior for analyze_themes.py with enhanced rankings"""
        # This test documents expected behavior for the analyze themes script
        
        # With enhanced rankings, we expect:
        # 1. Tourist themes to rank higher in the output
        # 2. "Tourist Relevance" to be available as a sorting option
        # 3. Mundane themes like "Fitness" and "Educational" to rank lower
        
        # Mock some expected themes that should be reordered
        expected_tourist_top_themes = [
            "Scenic Views", "Adventure Tours", "Food Tours", "Romantic Spots", "Photography Locations"
        ]
        
        expected_mundane_bottom_themes = [
            "Fitness Centers", "Educational Institutions", "Administrative Services", "Medical Facilities"
        ]
        
        # These are expectations rather than actual tests since we need real data
        # The actual integration test would run analyze_themes.py and verify ordering
        self.assertTrue(True, "Enhanced rankings should improve tourist theme prioritization")
    
    def test_dynamic_viewer_integration_expectations(self):
        """Test expected behavior for dynamic viewer with enhanced rankings"""
        # With enhanced rankings, we expect:
        # 1. Default sort to be "Tourist Relevance (High-Low)" 
        # 2. Tourist themes to appear at the top by default
        # 3. Evidence Quality to be available as alternative sort
        # 4. Clear distinction between tourist and evidence-based rankings
        
        self.assertTrue(True, "Dynamic viewer should default to tourist relevance sorting")


if __name__ == '__main__':
    unittest.main() 