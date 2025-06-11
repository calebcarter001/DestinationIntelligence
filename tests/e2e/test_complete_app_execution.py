"""
End-to-End Tests - Complete App Execution
Simple tests for testing the complete application execution flow.
"""

import unittest
import sys
import os
import asyncio
import tempfile

# Add the root directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

from src.schemas import AuthorityType
from src.tools.enhanced_theme_analysis_tool import EnhancedThemeAnalysisTool, EnhancedThemeAnalysisInput
from datetime import datetime

class TestCompleteAppExecution(unittest.TestCase):
    
    def setUp(self):
        """Set up test environment"""
        self.tool = EnhancedThemeAnalysisTool()
        self.input_class = EnhancedThemeAnalysisInput

    def test_full_destination_analysis_pipeline(self):
        """Test the complete pipeline from content input to final themes"""
        # Sample realistic content
        test_content = [
            {
                "url": "https://reddit.com/r/Seattle",
                "content": "As a Seattle local, I highly recommend visiting the original Starbucks at Pike Place Market. The grunge music scene here is incredible - check out venues like The Crocodile where famous bands got their start. The coffee culture is authentic and deep-rooted.",
                "title": "Local's Seattle Recommendations"
            },
            {
                "url": "https://seattle.gov/tourism",
                "content": "Seattle offers excellent public transportation including buses, light rail, and ferries. The city is known for its museums, parks, and waterfront attractions. Safety information and accessibility resources are available for visitors.",
                "title": "Official Seattle Tourism Information"
            },
            {
                "url": "https://visitseattle.org/food",
                "content": "Seattle's culinary scene features fresh seafood, international cuisine, and innovative restaurants. The Pike Place Market offers local produce and specialty foods. Food tours showcase the city's diverse dining options.",
                "title": "Seattle Food Scene Guide"
            }
        ]
        
        # Create input for analysis
        input_data = self.input_class(
            destination_name="Seattle, United States",
            country_code="US",
            text_content_list=test_content,
            analyze_temporal=True,
            min_confidence=0.3  # Lower threshold for testing
        )
        
        async def run_analysis():
            return await self.tool.analyze_themes(input_data)
        
        # Execute the complete pipeline
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            result = loop.run_until_complete(run_analysis())
            
            # Verify pipeline completed successfully
            self.assertIsInstance(result, dict, "Should return analysis result")
            
            # Check for evidence extraction
            if "extracted_evidence" in result:
                evidence = result["extracted_evidence"]
                self.assertGreater(len(evidence), 0, "Should extract evidence from content")
                print(f"✅ Extracted {len(evidence)} pieces of evidence")
            
            # Check for theme discovery (critical test - would have failed with KeyError)
            if "discovered_themes" in result:
                themes = result["discovered_themes"]
                print(f"✅ Generated {len(themes)} themes without KeyError")
                
                # Verify theme structure
                for theme in themes[:2]:  # Check first 2 themes
                    if hasattr(theme, 'name'):
                        self.assertIsNotNone(theme.name, "Theme should have a name")
                        self.assertIsNotNone(theme.macro_category, "Theme should have macro category")
                        print(f"✅ Theme '{theme.name}' in category '{theme.macro_category}'")
            
            print("✅ Complete pipeline executed successfully")
            
        except Exception as e:
            self.fail(f"Complete pipeline execution failed: {e}")
        finally:
            loop.close()

    def test_cultural_intelligence_theme_categorization(self):
        """Test that cultural intelligence properly categorizes themes"""
        # Content designed to generate different theme types
        cultural_content = [
            {
                "url": "https://reddit.com/r/Seattle",
                "content": "As a local musician, the grunge scene here is authentic and unique to Seattle. Underground venues and local bands carry on the tradition started by Nirvana and Soundgarden.",
                "title": "Local Music Scene Insights"
            }
        ]
        
        practical_content = [
            {
                "url": "https://seattle.gov/transportation",
                "content": "Public transportation in Seattle includes bus routes, light rail connections, and ferry services. Parking information and accessibility options are available for all visitors.",
                "title": "Seattle Transportation Guide"
            }
        ]
        
        hybrid_content = [
            {
                "url": "https://seattlefoodie.com",
                "content": "Seattle's food scene combines fresh Pacific Northwest ingredients with international influences. Local restaurants offer both fine dining and casual neighborhood spots.",
                "title": "Seattle Dining Scene"
            }
        ]
        
        # Test each content type
        for content_type, content in [("cultural", cultural_content), ("practical", practical_content), ("hybrid", hybrid_content)]:
            input_data = self.input_class(
                destination_name="Seattle, United States",
                country_code="US", 
                text_content_list=content,
                analyze_temporal=False,
                min_confidence=0.2
            )
            
            async def run_test():
                return await self.tool.analyze_themes(input_data)
            
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                result = loop.run_until_complete(run_test())
                
                # Should complete without errors regardless of content type
                self.assertIsInstance(result, dict)
                print(f"✅ {content_type.title()} content processed successfully")
                
            except Exception as e:
                self.fail(f"{content_type.title()} content processing failed: {e}")
            finally:
                loop.close()

    def test_output_file_structure_validation(self):
        """Test that output has valid structure for downstream processing"""
        test_content = [
            {
                "url": "https://example.com",
                "content": "Seattle coffee culture and music scene information",
                "title": "Seattle Guide"
            }
        ]
        
        input_data = self.input_class(
            destination_name="Seattle, United States",
            country_code="US",
            text_content_list=test_content,
            analyze_temporal=True,
            min_confidence=0.3
        )
        
        async def run_test():
            return await self.tool.analyze_themes(input_data)
        
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            result = loop.run_until_complete(run_test())
            
            # Validate output structure for downstream scripts
            required_sections = ["extracted_evidence", "discovered_themes", "validated_themes"]
            
            for section in required_sections:
                if section in result:
                    self.assertIsInstance(result[section], list, f"{section} should be a list")
                    
            # If themes were generated, validate their structure
            if "discovered_themes" in result and result["discovered_themes"]:
                theme = result["discovered_themes"][0]
                
                # Check that themes have required attributes for database storage
                required_attributes = ["name", "macro_category", "fit_score"]
                for attr in required_attributes:
                    if hasattr(theme, attr):
                        self.assertIsNotNone(getattr(theme, attr), f"Theme should have {attr}")
                        
            print("✅ Output structure validation passed")
            
        except Exception as e:
            self.fail(f"Output structure validation failed: {e}")
        finally:
            loop.close()

    def test_error_recovery_in_pipeline(self):
        """Test that pipeline handles edge cases gracefully"""
        # Test with minimal content
        minimal_content = [
            {
                "url": "https://example.com",
                "content": "Short",
                "title": "Minimal"
            }
        ]
        
        input_data = self.input_class(
            destination_name="Test Destination",
            country_code="US",
            text_content_list=minimal_content,
            analyze_temporal=False,
            min_confidence=0.1  # Very low threshold
        )
        
        async def run_test():
            return await self.tool.analyze_themes(input_data)
        
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            result = loop.run_until_complete(run_test())
            
            # Should not crash even with minimal content
            self.assertIsInstance(result, dict)
            print("✅ Pipeline handled minimal content gracefully")
            
        except Exception as e:
            self.fail(f"Pipeline should handle minimal content gracefully: {e}")
        finally:
            loop.close()

if __name__ == "__main__":
    unittest.main() 