"""
Theme Generation Integration Test - Tests the actual theme generation process
This test would have caught the evidence_quality KeyError that the other tests missed.
"""

import unittest
import sys
import os
import asyncio
from unittest.mock import Mock, patch

# Add the root directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

class TestThemeGenerationIntegration(unittest.TestCase):
    """Test the actual theme generation process that runs in the main app"""
    
    def setUp(self):
        """Set up test environment"""
        from src.schemas import AuthorityType, EnhancedEvidence
        from datetime import datetime
        self.AuthorityType = AuthorityType
        self.Evidence = EnhancedEvidence
        self.datetime = datetime
        # Create a realistic config that matches what the app uses
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
                }
            }
        }
        
    def test_enhanced_theme_analysis_tool_theme_generation(self):
        """Test the actual theme generation process that would catch the KeyError"""
        try:
            from src.tools.enhanced_theme_analysis_tool import EnhancedThemeAnalysisTool, EnhancedThemeAnalysisInput
            
            # Create tool with our test config
            tool = EnhancedThemeAnalysisTool(config=self.test_config)
            
            # Verify cultural intelligence is enabled
            self.assertTrue(tool.enable_dual_track, "Cultural intelligence should be enabled")
            
            # Create realistic test content that would generate themes
            test_content = [
                {
                    "url": "https://reddit.com/r/Seattle/comments/local_tips",
                    "content": "As a local who has lived in Seattle for 10 years, I can tell you that the grunge music scene is still alive and well. You should check out the authentic venues where Nirvana and Pearl Jam started. The coffee culture here is unique - we have the original Starbucks and amazing local roasters that you won't find anywhere else.",
                    "title": "Local's Guide to Seattle"
                },
                {
                    "url": "https://visitseattle.org/things-to-do",
                    "content": "Seattle offers world-class museums, beautiful parks, and excellent dining. The Space Needle provides panoramic views of the city. Pike Place Market is a popular destination for tourists with fresh seafood and local vendors.",
                    "title": "Official Seattle Tourism Guide"
                }
            ]
            
            # Create input data
            input_data = EnhancedThemeAnalysisInput(
                destination_name="Seattle, United States",
                country_code="US",
                text_content_list=test_content,
                analyze_temporal=True,
                min_confidence=0.5,
                config=self.test_config
            )
            
            # Run the actual theme analysis process (this would fail with KeyError before the fix)
            async def run_test():
                result = await tool.analyze_themes(input_data)
                return result
            
            # Execute the async test
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                result = loop.run_until_complete(run_test())
                
                # Verify that themes were generated without errors
                self.assertIsInstance(result, dict, "Should return a result dictionary")
                
                # Check that we have the expected structure (this would fail if KeyError occurred)
                if "discovered_themes" in result:
                    themes = result["discovered_themes"]
                    print(f"✅ Successfully generated {len(themes)} themes without KeyError")
                    
                    # Test that cultural intelligence categorization worked
                    if themes:
                        for theme in themes[:3]:  # Check first 3 themes
                            self.assertIn("macro_category", theme.__dict__ if hasattr(theme, '__dict__') else theme)
                            print(f"✅ Theme '{theme.name if hasattr(theme, 'name') else 'Unknown'}' categorized successfully")
                
                print("✅ Theme generation process completed successfully")
                
            finally:
                loop.close()
                
        except Exception as e:
            self.fail(f"Theme generation failed with error: {e}")

    def test_confidence_calculation_compatibility(self):
        """Test that both confidence calculation methods return compatible structures"""
        try:
            from src.tools.enhanced_theme_analysis_tool import EnhancedThemeAnalysisTool
            from src.schemas import EnhancedEvidence
            
            # Create tool
            tool = EnhancedThemeAnalysisTool(config=self.test_config)
            
            # Create mock evidence
            mock_evidence = [
                self.Evidence(
                    id="test_evidence_theme_gen_1",
                    text_snippet="Local tip: Seattle's grunge music scene is unique",
                    source_url="https://reddit.com/r/Seattle",
                    source_category=self.AuthorityType.RESIDENT,
                    authority_weight=0.7,
                    sentiment=0.8,
                    confidence=0.8,
                    timestamp=self.datetime.now().isoformat(),
                    cultural_context={},
                    relationships=[],
                    agent_id="test_agent"
                ),
                self.Evidence(
                    id="test_evidence_theme_gen_2",
                    text_snippet="Official tourism information about Seattle attractions",
                    source_url="https://visitseattle.org",
                    source_category=self.AuthorityType.PROFESSIONAL,
                    authority_weight=0.9,
                    sentiment=0.6,
                    confidence=0.9,
                    timestamp=self.datetime.now().isoformat(),
                    cultural_context={},
                    relationships=[],
                    agent_id="test_agent"
                )
            ]
            
            # Test cultural enhanced confidence (new method)
            cultural_components = tool._calculate_cultural_enhanced_confidence(
                mock_evidence,
                {"activity", "location"},
                {"grunge music", "coffee culture"},
                {"summer", "fall"},
                "Cultural Identity & Atmosphere"
            )
            
            # Test regular enhanced confidence (fallback method)
            regular_components = tool._calculate_enhanced_confidence(
                mock_evidence,
                {"activity", "location"},
                {"grunge music", "coffee culture"},
                {"summer", "fall"}
            )
            
            # Both should have the required keys for compatibility
            required_keys = ["evidence_quality", "source_diversity", "temporal_coverage", "content_completeness", "total_score"]
            
            for key in required_keys:
                self.assertIn(key, cultural_components, f"Cultural confidence missing key: {key}")
                self.assertIn(key, regular_components, f"Regular confidence missing key: {key}")
                
            print("✅ Both confidence calculation methods have compatible keys")
            
        except Exception as e:
            self.fail(f"Confidence calculation compatibility test failed: {e}")

    def test_dual_track_processing_activation(self):
        """Test that dual-track processing is properly activated based on config"""
        try:
            from src.tools.enhanced_theme_analysis_tool import EnhancedThemeAnalysisTool
            
            # Test with dual-track enabled
            config_enabled = {"cultural_intelligence": {"enable_dual_track_processing": True}}
            tool_enabled = EnhancedThemeAnalysisTool(config=config_enabled)
            self.assertTrue(tool_enabled.enable_dual_track, "Dual-track should be enabled")
            
            # Test with dual-track disabled
            config_disabled = {"cultural_intelligence": {"enable_dual_track_processing": False}}
            tool_disabled = EnhancedThemeAnalysisTool(config=config_disabled)
            self.assertFalse(tool_disabled.enable_dual_track, "Dual-track should be disabled")
            
            # Test with no config (should default)
            tool_default = EnhancedThemeAnalysisTool()
            self.assertIsInstance(tool_default.enable_dual_track, bool, "Should have boolean default")
            
            print("✅ Dual-track processing activation works correctly")
            
        except Exception as e:
            self.fail(f"Dual-track processing test failed: {e}")

def run_theme_generation_tests():
    """Run the theme generation integration tests"""
    print("🔧 Theme Generation Integration Tests")
    print("=" * 50)
    print("Testing the actual theme generation process that the main app uses...")
    print()
    
    # Create test suite
    suite = unittest.TestSuite()
    suite.addTest(unittest.makeSuite(TestThemeGenerationIntegration))
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Print summary
    print("\n" + "=" * 50)
    print("📊 Theme Generation Test Results")
    print("=" * 50)
    
    total_tests = result.testsRun
    failures = len(result.failures)
    errors = len(result.errors)
    success = total_tests - failures - errors
    
    print(f"Total Theme Generation Tests: {total_tests}")
    print(f"✅ Passed: {success}")
    print(f"❌ Failed: {failures}")
    print(f"🚨 Errors: {errors}")
    
    if failures == 0 and errors == 0:
        print("\n🎉 All theme generation tests passed!")
        print("✅ The actual theme generation process works correctly")
        print("✅ This test would have caught the evidence_quality KeyError")
        return True
    else:
        print("\n⚠️  Some theme generation tests failed.")
        print("💡 This demonstrates the importance of testing the right layer!")
        return False

if __name__ == "__main__":
    success = run_theme_generation_tests()
    exit(0 if success else 1) 