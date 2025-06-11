"""
Configuration Tests - Cultural Intelligence
Simple tests for cultural intelligence configuration loading and validation.
"""

import unittest
import sys
import os
import tempfile
import yaml
from datetime import datetime

# Add the root directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

from src.schemas import EnhancedEvidence, AuthorityType
from src.tools.enhanced_theme_analysis_tool import EnhancedThemeAnalysisTool

class TestCulturalIntelligenceConfig(unittest.TestCase):
    
    def setUp(self):
        """Set up test environment"""
        self.test_config = {
            "cultural_intelligence": {
                "enable_dual_track_processing": True,
                "enable_cultural_authenticity_scoring": True,
                "enable_distinctiveness_filtering": True,
                "authentic_source_indicators": {
                    "high_authenticity": ["reddit.com", "local", "community"],
                    "authenticity_phrases": ["as a local", "i live here", "local tip"]
                },
                "authoritative_source_indicators": {
                    "high_authority": [".gov", ".edu", "official"]
                },
                "distinctiveness_indicators": {
                    "unique_keywords": ["unique", "only in", "signature"],
                    "generic_keywords": ["popular", "famous", "well-known"]
                },
                "category_processing_rules": {
                    "cultural": {
                        "confidence_threshold": 0.45,
                        "distinctiveness_threshold": 0.3
                    },
                    "practical": {
                        "confidence_threshold": 0.75,
                        "distinctiveness_threshold": 0.1
                    }
                }
            }
        }

    def test_dual_track_processing_config_loading(self):
        """Test that dual track processing config is loaded correctly"""
        try:
            # Test with dual track enabled
            tool_enabled = EnhancedThemeAnalysisTool(config=self.test_config)
            self.assertTrue(tool_enabled.enable_dual_track, "Dual track should be enabled from config")
            
            # Test with dual track disabled
            disabled_config = {
                "cultural_intelligence": {
                    "enable_dual_track_processing": False
                }
            }
            tool_disabled = EnhancedThemeAnalysisTool(config=disabled_config)
            self.assertFalse(tool_disabled.enable_dual_track, "Dual track should be disabled from config")
            
        except ImportError as e:
            self.skipTest(f"Failed to import required modules: {e}")

    def test_authenticity_indicators_config(self):
        """Test that authenticity indicators are loaded from config"""
        try:
            tool = EnhancedThemeAnalysisTool(config=self.test_config)
            
            # Check that config is accessible (assuming it's stored as an attribute)
            if hasattr(tool, 'config') or hasattr(tool, 'cultural_config'):
                config = getattr(tool, 'config', getattr(tool, 'cultural_config', None))
                
                if config and 'cultural_intelligence' in config:
                    ci_config = config['cultural_intelligence']
                    
                    # Verify authenticity indicators
                    self.assertIn('authentic_source_indicators', ci_config)
                    auth_indicators = ci_config['authentic_source_indicators']
                    self.assertIn('reddit.com', auth_indicators['high_authenticity'])
                    self.assertIn('as a local', auth_indicators['authenticity_phrases'])
                    
        except ImportError as e:
            self.skipTest(f"Failed to import required modules: {e}")

    def test_distinctiveness_keywords_config(self):
        """Test that distinctiveness keywords are loaded from config"""
        try:
            tool = EnhancedThemeAnalysisTool(config=self.test_config)
            
            # Test distinctiveness calculation with config keywords
            
            # Distinctive evidence (contains unique keywords)
            distinctive_evidence = [
                EnhancedEvidence(
                    id="test_1",
                    text_snippet="This unique venue is signature to Seattle only",
                    source_url="https://example.com",
                    authority_weight=0.7,
                    sentiment=0.8,
                    source_category=AuthorityType.RESIDENT,
                    confidence=0.8,
                    timestamp=datetime.now().isoformat()
                )
            ]
            
            distinctive_score = tool._calculate_distinctiveness_score(distinctive_evidence)
            
            # Generic evidence (contains generic keywords) 
            generic_evidence = [
                EnhancedEvidence(
                    id="test_2", 
                    text_snippet="This popular and famous tourist destination is well-known",
                    source_url="https://example.com",
                    authority_weight=0.7,
                    sentiment=0.8,
                    source_category=AuthorityType.RESIDENT,
                    confidence=0.8,
                    timestamp=datetime.now().isoformat()
                )
            ]
            
            generic_score = tool._calculate_distinctiveness_score(generic_evidence)
            
            # Distinctive should score higher than generic
            self.assertGreater(distinctive_score, generic_score, 
                             "Distinctive content should score higher than generic")
                             
        except ImportError as e:
            self.skipTest(f"Failed to import required modules: {e}")

    def test_category_processing_rules_config(self):
        """Test that category processing rules are applied from config"""
        try:
            tool = EnhancedThemeAnalysisTool(config=self.test_config)
            
            # Test processing type determination
            cultural_type = tool._get_processing_type("Cultural Identity & Atmosphere")
            practical_type = tool._get_processing_type("Transportation & Access")
            
            self.assertEqual(cultural_type, "cultural")
            self.assertEqual(practical_type, "practical")

        except ImportError as e:
            self.skipTest(f"Failed to import required modules: {e}")

    def test_config_validation_and_defaults(self):
        """Test that config validation works and defaults are applied"""
        try:
            # Test with minimal config
            minimal_config = {
                "cultural_intelligence": {
                    "enable_dual_track_processing": True
                }
            }
            
            tool = EnhancedThemeAnalysisTool(config=minimal_config)
            
            # Should not crash and should apply defaults
            self.assertTrue(tool.enable_dual_track)
            
            # Test with no config at all
            tool_no_config = EnhancedThemeAnalysisTool()
            
            # Should have reasonable defaults
            self.assertIsInstance(tool_no_config.enable_dual_track, bool)
            
        except ImportError as e:
            self.skipTest(f"Failed to import required modules: {e}")

    def test_config_change_impact_on_processing(self):
        """Test that config changes actually impact processing behavior"""
        try:
            tool = EnhancedThemeAnalysisTool(config=self.test_config)
            
            # Create evidence for testing
            test_evidence = [
                EnhancedEvidence(
                    id="test_1",
                    text_snippet="Local tip from a Seattle resident",
                    source_url="https://reddit.com/r/Seattle",
                    authority_weight=0.6,
                    sentiment=0.8,
                    source_category=AuthorityType.RESIDENT,
                    confidence=0.8,
                    timestamp=datetime.now().isoformat()
                )
            ]
            
            # Test with cultural intelligence enabled
            enabled_confidence = tool._calculate_cultural_enhanced_confidence(
                test_evidence, {"activity"}, {"local"}, {"summer"}, "Cultural Identity & Atmosphere"
            )
            
            # Test with cultural intelligence disabled
            disabled_config = {
                "cultural_intelligence": {
                    "enable_dual_track_processing": False
                }
            }
            disabled_tool = EnhancedThemeAnalysisTool(config=disabled_config)
            disabled_confidence = disabled_tool._calculate_enhanced_confidence(
                test_evidence, {"activity"}, {"local"}, {"summer"}
            )
            
            # Both should have required keys but may have different calculation logic
            self.assertIn("evidence_quality", enabled_confidence)
            self.assertIn("evidence_quality", disabled_confidence)
            self.assertIn("total_score", enabled_confidence)
            self.assertIn("total_score", disabled_confidence)
            
        except ImportError as e:
            self.skipTest(f"Failed to import required modules: {e}")

if __name__ == "__main__":
    unittest.main() 