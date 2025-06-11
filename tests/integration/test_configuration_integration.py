"""
Configuration Integration Tests
Simple tests for configuration loading and integration across components.
"""

import unittest
import sys
import os
import tempfile
import yaml
from datetime import datetime
from src.schemas import EnhancedEvidence, AuthorityType

# Add the root directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

class TestConfigurationIntegration(unittest.TestCase):
    
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
                    },
                    "hybrid": {
                        "confidence_threshold": 0.6,
                        "distinctiveness_threshold": 0.2
                    }
                }
            }
        }

    def test_config_loading_integration(self):
        """Test that configuration is properly loaded and applied"""
        from src.tools.enhanced_theme_analysis_tool import EnhancedThemeAnalysisTool
        
        # Test with full config
        tool = EnhancedThemeAnalysisTool(config=self.test_config)
        
        # Verify dual track is enabled
        self.assertTrue(tool.enable_dual_track, 
                      "Dual track should be enabled from config")
        
        # Test configuration propagation through the system
        self.assertIsNotNone(tool, "Tool should initialize with config")

    def test_confidence_thresholds_from_config(self):
        """Test that confidence thresholds from config are applied"""
        try:
            from src.tools.enhanced_theme_analysis_tool import EnhancedThemeAnalysisTool
            
            tool = EnhancedThemeAnalysisTool(config=self.test_config)
            
            # Test different category processing
            mock_evidence = [
                EnhancedEvidence(
                    id="test_evidence_config_1",
                    text_snippet="Test content",
                    source_url="https://example.com",
                    authority_weight=0.7,
                    sentiment=0.8,
                    source_category=AuthorityType.OFFICIAL,
                    confidence=0.8,
                    timestamp="2024-01-01T00:00:00",
                    cultural_context={},
                    relationships=[],
                    agent_id="test_agent"
                )
            ]
            
            # Test cultural processing (should use 0.45 threshold)
            cultural_confidence = tool._calculate_cultural_enhanced_confidence(
                mock_evidence,
                {"activity"},
                {"test"},
                {"summer"},
                "Cultural Identity & Atmosphere"
            )
            
            # Verify confidence structure
            self.assertIn("total_score", cultural_confidence)
            self.assertIsInstance(cultural_confidence["total_score"], (int, float))
            
        except ImportError as e:
            self.skipTest("Required modules not available")

    def test_source_indicators_from_config(self):
        """Test that source indicators from config are used"""
        from src.tools.enhanced_theme_analysis_tool import EnhancedThemeAnalysisTool
        
        tool = EnhancedThemeAnalysisTool(config=self.test_config)
        
        # Test authentic source detection
        authentic_evidence = [
            EnhancedEvidence(
                id="test_evidence_config_2",
                text_snippet="As a local, I recommend this hidden gem",
                source_url="https://reddit.com/r/Seattle",
                authority_weight=0.6,
                sentiment=0.8,
                source_category=AuthorityType.COMMUNITY,
                confidence=0.8,
                timestamp="2024-01-01T00:00:00",
                cultural_context={},
                relationships=[],
                agent_id="test_agent"
            )
        ]
        
        auth_score = tool._calculate_authenticity_score(authentic_evidence)
        
        # Should detect high authenticity based on config indicators
        self.assertGreater(auth_score, 0.5, 
                         "Should detect high authenticity from config indicators")
        
        # Test authoritative source detection
        official_evidence = [
            EnhancedEvidence(
                id="test_evidence_config_3",
                text_snippet="Official city information",
                source_url="https://seattle.gov/tourism",
                authority_weight=0.9,
                sentiment=0.7,
                source_category=AuthorityType.OFFICIAL,
                confidence=0.9,
                timestamp="2024-01-01T00:00:00",
                cultural_context={},
                relationships=[],
                agent_id="test_agent"
            )
        ]
        
        authority_score = tool._calculate_authority_score(official_evidence)
        
        # Should detect high authority based on config indicators
        self.assertGreater(authority_score, 0.7,
                         "Should detect high authority from config indicators")

    def test_category_rules_from_config(self):
        """Test that category processing rules from config are applied"""
        from src.tools.enhanced_theme_analysis_tool import EnhancedThemeAnalysisTool
        
        tool = EnhancedThemeAnalysisTool(config=self.test_config)
        
        # Test category processing type determination
        cultural_type = tool._get_processing_type("Cultural Identity & Atmosphere")
        practical_type = tool._get_processing_type("Transportation & Access")
        hybrid_type = tool._get_processing_type("Food & Dining")
        
        # Should match expected processing types
        self.assertEqual(cultural_type, "cultural")
        self.assertEqual(practical_type, "practical")
        self.assertEqual(hybrid_type, "hybrid")

    def test_config_validation_and_defaults(self):
        """Test config validation and default value application"""
        try:
            from src.tools.enhanced_theme_analysis_tool import EnhancedThemeAnalysisTool
            
            # Test with minimal config
            minimal_config = {
                "cultural_intelligence": {
                    "enable_dual_track_processing": True
                }
            }
            
            tool = EnhancedThemeAnalysisTool(config=minimal_config)
            
            # Should not crash and should have defaults
            self.assertTrue(tool.enable_dual_track)
            
            # Test with no config
            tool_no_config = EnhancedThemeAnalysisTool()
            
            # Should have reasonable defaults
            self.assertIsInstance(tool_no_config.enable_dual_track, bool)
            
        except ImportError as e:
            self.skipTest("Required modules not available")

    def test_nested_config_access(self):
        """Test accessing nested configuration values"""
        from src.tools.enhanced_theme_analysis_tool import EnhancedThemeAnalysisTool
        
        tool = EnhancedThemeAnalysisTool(config=self.test_config)
        
        # Test that nested config values are accessible
        # This verifies the config is properly structured and accessible
        
        # Test distinctiveness calculation with config keywords
        distinctive_evidence = [
            EnhancedEvidence(
                id="test_evidence_config_4",
                text_snippet="This unique venue is signature to Seattle only",
                source_url="https://example.com",
                authority_weight=0.7,
                sentiment=0.8,
                source_category=AuthorityType.OFFICIAL,
                confidence=0.8,
                timestamp="2024-01-01T00:00:00",
                cultural_context={},
                relationships=[],
                agent_id="test_agent"
            )
        ]
        
        distinctive_score = tool._calculate_distinctiveness_score(distinctive_evidence)
        
        # Should work with config-defined keywords
        self.assertIsInstance(distinctive_score, (int, float))
        self.assertGreaterEqual(distinctive_score, 0.0)
        self.assertLessEqual(distinctive_score, 1.0)

    def test_config_change_impact_on_processing(self):
        """Test that config changes actually impact processing results"""
        try:
            from src.tools.enhanced_theme_analysis_tool import EnhancedThemeAnalysisTool
            
            test_evidence = [
                EnhancedEvidence(
                    id="test_evidence_config_5",
                    text_snippet="Local tip from Seattle resident",
                    source_url="https://reddit.com/r/Seattle",
                    authority_weight=0.6,
                    sentiment=0.8,
                    source_category=AuthorityType.COMMUNITY,
                    confidence=0.8,
                    timestamp="2024-01-01T00:00:00",
                    cultural_context={},
                    relationships=[],
                    agent_id="test_agent"
                )
            ]
            
            # Test with cultural intelligence enabled
            enabled_tool = EnhancedThemeAnalysisTool(config=self.test_config)
            enabled_result = enabled_tool._calculate_cultural_enhanced_confidence(
                test_evidence, {"activity"}, {"local"}, {"summer"}, "Cultural Identity & Atmosphere"
            )
            
            # Test with cultural intelligence disabled
            disabled_config = {
                "cultural_intelligence": {
                    "enable_dual_track_processing": False
                }
            }
            disabled_tool = EnhancedThemeAnalysisTool(config=disabled_config)
            disabled_result = disabled_tool._calculate_enhanced_confidence(
                test_evidence, {"activity"}, {"local"}, {"summer"}
            )
            
            # Both should work but may have different logic
            self.assertIn("total_score", enabled_result)
            self.assertIn("total_score", disabled_result)
            
        except ImportError as e:
            self.skipTest("Required modules not available")

    def test_config_environment_variables(self):
        """Test configuration loading from environment variables"""
        try:
            from src.tools.enhanced_theme_analysis_tool import EnhancedThemeAnalysisTool
            
            # Set test environment variables
            os.environ["CULTURAL_INTELLIGENCE_ENABLED"] = "true"
            os.environ["CULTURAL_CONFIDENCE_THRESHOLD"] = "0.55"
            
            # Create tool with env vars
            tool = EnhancedThemeAnalysisTool()
            
            # Should respect environment variables
            self.assertTrue(tool.enable_dual_track)
            
            # Clean up
            del os.environ["CULTURAL_INTELLIGENCE_ENABLED"]
            del os.environ["CULTURAL_CONFIDENCE_THRESHOLD"]
            
        except ImportError as e:
            self.skipTest("Required modules not available")

if __name__ == "__main__":
    unittest.main() 