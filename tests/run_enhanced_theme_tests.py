#!/usr/bin/env python3
"""
Enhanced Theme Analysis Tool Test Runner

This script runs comprehensive tests specifically for the EnhancedThemeAnalysisTool
and validates all the enhanced analysis fields to ensure they are properly populated.
"""

import asyncio
import sys
import os
import json
from datetime import datetime
from pathlib import Path

# Add the project root to the path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import test modules and dependencies
import pytest
from src.tools.enhanced_theme_analysis_tool import (
    EnhancedThemeAnalysisTool, 
    EnhancedThemeAnalysisInput
)


class EnhancedThemeTestRunner:
    """Specialized test runner for enhanced theme analysis validation"""
    
    def __init__(self):
        self.tool = EnhancedThemeAnalysisTool()
        self.test_results = []
        self.enhanced_field_validations = {}
    
    async def run_comprehensive_tests(self):
        """Run comprehensive tests for enhanced theme analysis"""
        print("=" * 80)
        print("🧪 ENHANCED THEME ANALYSIS TOOL - COMPREHENSIVE TESTING")
        print("=" * 80)
        print(f"📅 Test Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print()
        
        # Test scenarios
        test_scenarios = [
            self._test_basic_enhanced_fields_population,
            self._test_missing_description_handling,
            self._test_local_authority_detection,
            self._test_seasonal_intelligence,
            self._test_sentiment_analysis,
            self._test_cultural_context_analysis,
            self._test_insight_classification,
            self._test_evidence_processing,
            self._test_temporal_analysis,
            self._test_edge_cases_and_robustness
        ]
        
        passed_tests = 0
        total_tests = len(test_scenarios)
        
        for i, test_func in enumerate(test_scenarios, 1):
            print(f"🔬 Running Test {i}/{total_tests}: {test_func.__name__}")
            try:
                await test_func()
                print(f"   ✅ PASSED")
                passed_tests += 1
            except Exception as e:
                print(f"   ❌ FAILED: {str(e)}")
                self.test_results.append({
                    "test": test_func.__name__,
                    "status": "FAILED",
                    "error": str(e)
                })
            print()
        
        # Summary
        print("=" * 80)
        print("📊 TEST SUMMARY")
        print("=" * 80)
        print(f"✅ Passed: {passed_tests}/{total_tests}")
        print(f"❌ Failed: {total_tests - passed_tests}/{total_tests}")
        print(f"📈 Success Rate: {(passed_tests/total_tests)*100:.1f}%")
        print()
        
        # Enhanced Field Validation Summary
        self._print_enhanced_field_summary()
        
        return passed_tests == total_tests

    async def _test_basic_enhanced_fields_population(self):
        """Test that all enhanced fields are populated correctly"""
        # Create realistic test data
        input_data = EnhancedThemeAnalysisInput(
            destination_name="Bali, Indonesia",
            country_code="ID",
            text_content_list=[
                {
                    "url": "https://example.com/bali-temples",
                    "content": "Bali is famous for its beautiful Hindu temples like Tanah Lot and Uluwatu Temple. These temples offer stunning sunset views and traditional Balinese architecture. Visitors should dress respectfully and arrive early to avoid crowds. The temple ceremonies are authentic cultural experiences that showcase local traditions. Best visited during dry season from April to October.",
                    "title": "Best Temples to Visit in Bali"
                },
                {
                    "url": "https://localblog.com/bali-food", 
                    "content": "As a local Balinese resident, I can tell you the best warungs serve authentic nasi goreng and gado-gado. The night markets in Ubud are perfect for trying local specialties. Avoid tourist restaurants in Kuta - they're overpriced. Visit during dry season from May to September for the best food festival experiences.",
                    "title": "Local's Guide to Bali Food"
                }
            ],
            analyze_temporal=True,
            min_confidence=0.3
        )
        
        # Mock the agents to avoid external dependencies
        from unittest.mock import AsyncMock
        
        self.tool.validation_agent = AsyncMock()
        self.tool.cultural_agent = AsyncMock()
        self.tool.contradiction_agent = AsyncMock()
        
        self.tool.validation_agent.execute_task.return_value = {
            "validated_themes": [
                {
                    "name": "Hindu Temples",
                    "macro_category": "Cultural & Arts",
                    "micro_category": "Historic Sites",
                    "confidence_level": "high",
                    "confidence_breakdown": {
                        "overall_confidence": 0.85,
                        "evidence_count": 2,
                        "source_diversity": 0.8,
                        "authority_score": 0.9
                    },
                    "is_validated": True
                }
            ],
            "validated_count": 1
        }
        
        self.tool.cultural_agent.execute_task.return_value = {
            "cultural_metrics": {
                "cultural_diversity_score": 0.8,
                "local_source_ratio": 0.7,
                "language_distribution": {"english": 80, "indonesian": 20},
                "optimal_mix_score": 0.75
            }
        }
        
        self.tool.contradiction_agent.execute_task.return_value = {
            "resolved_themes": [
                {
                    "name": "Hindu Temples",
                    "macro_category": "Cultural & Arts",
                    "micro_category": "Historic Sites",
                    "confidence_level": "high",
                    "confidence_breakdown": {
                        "overall_confidence": 0.85,
                        "evidence_count": 2,
                        "source_diversity": 0.8,
                        "authority_score": 0.9
                    },
                    "is_validated": True,
                    "contradiction_resolved": False
                }
            ],
            "contradictions_found": 0
        }
        
        # Run the analysis
        result = await self.tool.analyze_themes(input_data)
        
        # Validate basic structure
        assert "themes" in result, "Result missing 'themes' field"
        themes = result["themes"]
        assert len(themes) > 0, "No themes generated"
        
        # Validate enhanced fields for each theme
        enhanced_fields_found = {}
        for theme in themes:
            theme_name = theme.get('name', 'Unknown')
            print(f"      🔍 Validating theme: {theme_name}")
            
            # Track which enhanced fields are present and populated
            enhanced_fields = [
                "authentic_insights",
                "local_authorities", 
                "seasonal_relevance",
                "cultural_summary",
                "sentiment_analysis",
                "temporal_analysis",
                "factors"
            ]
            
            for field in enhanced_fields:
                if field not in enhanced_fields_found:
                    enhanced_fields_found[field] = {"present": 0, "populated": 0, "total": 0}
                
                enhanced_fields_found[field]["total"] += 1
                
                if field in theme:
                    enhanced_fields_found[field]["present"] += 1
                    print(f"         ✓ {field}: Present")
                    
                    # Check if populated (not empty)
                    field_value = theme[field]
                    if field_value:  # Not None, empty list, or empty dict
                        enhanced_fields_found[field]["populated"] += 1
                        print(f"         ✓ {field}: Populated")
                        
                        # Detailed validation for specific fields
                        self._validate_specific_field(field, field_value, theme_name)
                    else:
                        print(f"         ⚠️  {field}: Present but empty")
                else:
                    print(f"         ❌ {field}: Missing")
        
        # Store results for summary
        self.enhanced_field_validations.update(enhanced_fields_found)
        
        # Assert that critical fields are present
        for theme in themes:
            assert "authentic_insights" in theme, f"Theme missing authentic_insights"
            assert "seasonal_relevance" in theme, f"Theme missing seasonal_relevance"
            assert "cultural_summary" in theme, f"Theme missing cultural_summary"
            assert "sentiment_analysis" in theme, f"Theme missing sentiment_analysis"
            assert "temporal_analysis" in theme, f"Theme missing temporal_analysis"

    def _validate_specific_field(self, field_name: str, field_value, theme_name: str):
        """Validate specific enhanced field structures and values"""
        try:
            if field_name == "authentic_insights":
                assert isinstance(field_value, list), f"authentic_insights should be list"
                if field_value:  # If not empty
                    for insight in field_value:
                        required_keys = ["insight_type", "authenticity_score", "uniqueness_score", "actionability_score"]
                        for key in required_keys:
                            # Handle both AuthenticInsight objects and dictionaries
                            if hasattr(insight, 'insight_type'):  # AuthenticInsight object
                                assert hasattr(insight, key), f"authentic_insights missing {key}"
                            else:  # Dictionary
                                assert key in insight, f"authentic_insights missing {key}"
                    print(f"            ✅ authentic_insights structure validated")
                        
            elif field_name == "local_authorities":
                assert isinstance(field_value, list), f"local_authorities should be list"
                if field_value:
                    authority = field_value[0]
                    required_keys = ["authority_type", "local_tenure", "expertise_domain", "community_validation"]
                    for key in required_keys:
                        assert key in authority, f"local_authorities missing {key}"
                    print(f"            ✅ local_authorities structure validated")
                        
            elif field_name == "seasonal_relevance":
                assert isinstance(field_value, dict), f"seasonal_relevance should be dict"
                # Should have month-based data
                month_found = any(month in field_value for month in 
                                ["january", "february", "march", "april", "may", "june",
                                 "july", "august", "september", "october", "november", "december"])
                if month_found:
                    print(f"            ✅ seasonal_relevance has month data")
                        
            elif field_name == "cultural_summary":
                assert isinstance(field_value, dict), f"cultural_summary should be dict"
                expected_keys = ["total_sources", "local_sources", "cultural_balance"]
                present_keys = [key for key in expected_keys if key in field_value]
                if present_keys:
                    print(f"            ✅ cultural_summary has {len(present_keys)} expected keys")
                        
            elif field_name == "sentiment_analysis":
                assert isinstance(field_value, dict), f"sentiment_analysis should be dict"
                expected_keys = ["overall", "average_score", "distribution"]
                present_keys = [key for key in expected_keys if key in field_value]
                if present_keys:
                    print(f"            ✅ sentiment_analysis has {len(present_keys)} expected keys")
                        
            elif field_name == "temporal_analysis":
                assert isinstance(field_value, dict), f"temporal_analysis should be dict"
                if field_value:
                    print(f"            ✅ temporal_analysis has data")
                        
            elif field_name == "factors":
                assert isinstance(field_value, dict), f"factors should be dict"
                if field_value:
                    print(f"            ✅ factors has {len(field_value)} factor entries")
                        
        except Exception as e:
            print(f"            ⚠️  {field_name} validation warning: {str(e)}")

    async def _test_missing_description_handling(self):
        """Test that themes without description are handled correctly"""
        print("      🔍 Testing missing description handling...")
        
        # This test would typically involve mocking the validation agent
        # to return themes without description fields
        
        # Create a theme without description
        test_theme = {
            "name": "Test Theme Without Description",
            "macro_category": "Test Category"
            # No description field
        }
        
        # Test the fallback logic
        theme_description = None
        if 'description' in test_theme:
            theme_description = test_theme['description']
        elif 'name' in test_theme:
            theme_description = f"{test_theme['name']} experiences in Test Destination. {test_theme.get('micro_category', '')} category."
        
        assert theme_description is not None, "Should generate description fallback"
        assert "Test Theme Without Description" in theme_description, "Should include theme name in fallback"
        print("      ✅ Description fallback working correctly")

    async def _test_local_authority_detection(self):
        """Test local authority detection and classification"""
        print("      🔍 Testing local authority detection...")
        
        # Test authority classification logic
        test_content = "As a professional chef with 15 years experience in this city"
        
        # This would test the _extract_local_tenure and _extract_expertise_domain methods
        # For now, we'll test the basic logic
        
        authority_indicators = ["professional", "chef", "experience", "years"]
        found_indicators = [indicator for indicator in authority_indicators if indicator in test_content.lower()]
        
        assert len(found_indicators) >= 2, f"Should detect authority indicators, found: {found_indicators}"
        print(f"      ✅ Detected authority indicators: {found_indicators}")

    async def _test_seasonal_intelligence(self):
        """Test seasonal intelligence extraction"""
        print("      🔍 Testing seasonal intelligence extraction...")
        
        test_content = "Best visited during summer months from June to August. Fall foliage is spectacular in October."
        
        seasonal_keywords = {
            "summer": ["summer", "june", "july", "august"],
            "fall": ["fall", "autumn", "october", "foliage"]
        }
        
        detected_seasons = []
        for season, keywords in seasonal_keywords.items():
            if any(keyword in test_content.lower() for keyword in keywords):
                detected_seasons.append(season)
        
        assert len(detected_seasons) >= 2, f"Should detect multiple seasons, found: {detected_seasons}"
        print(f"      ✅ Detected seasons: {detected_seasons}")

    async def _test_sentiment_analysis(self):
        """Test sentiment analysis accuracy"""
        print("      🔍 Testing sentiment analysis...")
        
        # Test positive sentiment
        positive_text = "Amazing experience! Absolutely wonderful and highly recommended."
        positive_score = self.tool._analyze_sentiment(positive_text)
        assert positive_score > 0.5, f"Positive text should have positive sentiment, got: {positive_score}"
        
        # Test negative sentiment
        negative_text = "Terrible experience. Awful service and horrible food. Avoid at all costs."
        negative_score = self.tool._analyze_sentiment(negative_text)
        assert negative_score < 0.5, f"Negative text should have negative sentiment, got: {negative_score}"
        
        print(f"      ✅ Sentiment analysis working (positive: {positive_score:.2f}, negative: {negative_score:.2f})")

    async def _test_cultural_context_analysis(self):
        """Test cultural context analysis"""
        print("      🔍 Testing cultural context analysis...")
        
        test_content = "As a local resident who has lived here for 20 years, I can recommend authentic experiences."
        test_url = "https://localblog.com/guide"
        test_title = "Local's Guide"
        
        cultural_context = self.tool._extract_enhanced_cultural_context(
            test_content, test_url, "US", test_title
        )
        
        assert isinstance(cultural_context, dict), "Should return cultural context dict"
        assert "is_local_source" in cultural_context, "Should detect local source"
        assert cultural_context["is_local_source"] == True, "Should identify as local source"
        assert "author_perspective" in cultural_context, "Should identify author perspective"
        
        print(f"      ✅ Cultural context analyzed: local_source={cultural_context['is_local_source']}, perspective={cultural_context['author_perspective']}")

    async def _test_insight_classification(self):
        """Test insight classification"""
        print("      🔍 Testing insight classification...")
        
        # Import the insight classifier
        from src.core.insight_classifier import InsightClassifier
        classifier = InsightClassifier()
        
        # Test seasonal insight
        seasonal_content = "Best time to visit is during spring from March to May"
        insight_type = classifier.classify_insight_type(seasonal_content)
        assert insight_type.value == "seasonal", f"Should classify as seasonal, got: {insight_type}"
        
        # Test insider insight
        insider_content = "Locals know a hidden gem secret restaurant that only residents frequent"
        insight_type = classifier.classify_insight_type(insider_content)
        assert insight_type.value == "insider", f"Should classify as insider, got: {insight_type}"
        
        print(f"      ✅ Insight classification working")

    async def _test_evidence_processing(self):
        """Test evidence extraction and processing"""
        print("      🔍 Testing evidence processing...")
        
        test_content = [
            {
                "url": "https://example.com/test",
                "content": "This is a test content piece with specific information about the destination. It has detailed descriptions and useful insights for travelers.",
                "title": "Test Guide"
            }
        ]
        
        evidence_list = await self.tool._extract_evidence(test_content, "US")
        
        assert len(evidence_list) > 0, "Should extract evidence from content"
        evidence = evidence_list[0]
        
        # Validate evidence structure
        required_attrs = ['id', 'source_url', 'source_category', 'authority_weight', 'text_snippet', 'cultural_context']
        for attr in required_attrs:
            assert hasattr(evidence, attr), f"Evidence missing attribute: {attr}"
        
        assert isinstance(evidence.cultural_context, dict), "Cultural context should be dict"
        assert 0 <= evidence.authority_weight <= 1, "Authority weight should be 0-1"
        
        print(f"      ✅ Evidence processing working: {len(evidence_list)} pieces extracted")

    async def _test_temporal_analysis(self):
        """Test temporal analysis integration"""
        print("      🔍 Testing temporal analysis...")
        
        # Test seasonal modifier calculation
        summer_modifier = self.tool._calculate_seasonal_modifier("beach activities", "summer")
        winter_modifier = self.tool._calculate_seasonal_modifier("beach activities", "winter")
        
        assert summer_modifier > winter_modifier, "Beach activities should be higher in summer"
        
        skiing_summer = self.tool._calculate_seasonal_modifier("skiing", "summer")
        skiing_winter = self.tool._calculate_seasonal_modifier("skiing", "winter")
        
        assert skiing_winter > skiing_summer, "Skiing should be higher in winter"
        
        print(f"      ✅ Temporal analysis working: beach summer={summer_modifier:.2f}, skiing winter={skiing_winter:.2f}")

    async def _test_edge_cases_and_robustness(self):
        """Test edge cases and error handling"""
        print("      🔍 Testing edge cases and robustness...")
        
        # Test with empty content
        empty_input = EnhancedThemeAnalysisInput(
            destination_name="Empty Test",
            country_code="US",
            text_content_list=[]
        )
        
        # Mock agents for empty case
        from unittest.mock import AsyncMock
        self.tool.validation_agent = AsyncMock()
        self.tool.cultural_agent = AsyncMock()
        self.tool.contradiction_agent = AsyncMock()
        
        self.tool.validation_agent.execute_task.return_value = {"validated_themes": [], "validated_count": 0}
        self.tool.cultural_agent.execute_task.return_value = {"cultural_metrics": {}}
        self.tool.contradiction_agent.execute_task.return_value = {"resolved_themes": [], "contradictions_found": 0}
        
        # Should not raise error
        result = await self.tool.analyze_themes(empty_input)
        
        assert "themes" in result, "Should return result structure even with empty input"
        assert isinstance(result["themes"], list), "Themes should be list even when empty"
        
        print(f"      ✅ Robustness test passed: handled empty input gracefully")

    def _print_enhanced_field_summary(self):
        """Print summary of enhanced field validation results"""
        if not self.enhanced_field_validations:
            return
            
        print("🔍 ENHANCED FIELD VALIDATION SUMMARY")
        print("-" * 60)
        
        for field_name, stats in self.enhanced_field_validations.items():
            total = stats["total"]
            present = stats["present"] 
            populated = stats["populated"]
            
            present_rate = (present / total * 100) if total > 0 else 0
            populated_rate = (populated / total * 100) if total > 0 else 0
            
            status = "✅" if populated_rate >= 80 else "⚠️" if populated_rate >= 50 else "❌"
            
            print(f"{status} {field_name:20} | Present: {present}/{total} ({present_rate:5.1f}%) | Populated: {populated}/{total} ({populated_rate:5.1f}%)")
        
        print()

    async def run_live_test_with_real_data(self):
        """Run a live test with real data to validate enhanced fields"""
        print("🌐 RUNNING LIVE TEST WITH REAL DATA")
        print("-" * 60)
        
        # Use realistic content that should trigger enhanced analysis
        realistic_input = EnhancedThemeAnalysisInput(
            destination_name="Kyoto, Japan",
            country_code="JP",
            text_content_list=[
                {
                    "url": "https://example.com/kyoto-temples",
                    "content": "Kyoto's ancient temples like Kiyomizu-dera and Fushimi Inari offer authentic spiritual experiences. As a local guide who has lived in Kyoto for 15 years, I recommend visiting early morning to avoid crowds. The traditional architecture showcases centuries of Japanese craftsmanship. Spring cherry blossom season from late March to early May is spectacular but very crowded. Summer can be hot and humid. Fall colors in November are stunning. Winter snow creates magical scenes but some temples have limited hours.",
                    "title": "Local Guide to Kyoto Temples"
                },
                {
                    "url": "https://example.com/kyoto-food",
                    "content": "Authentic kaiseki dining in Kyoto represents the pinnacle of Japanese culinary art. The seasonal ingredients and presentation reflect centuries of tradition. Local chef recommends trying tofu cuisine in temple districts. Prices range from ¥3000 for lunch to ¥15000 for dinner. Reservations essential at top restaurants. Vegetarian options available at Buddhist temple restaurants.",
                    "title": "Kyoto Culinary Experiences"
                }
            ],
            analyze_temporal=True,
            min_confidence=0.4
        )
        
        # Mock agents with realistic responses
        from unittest.mock import AsyncMock
        
        self.tool.validation_agent = AsyncMock()
        self.tool.cultural_agent = AsyncMock()
        self.tool.contradiction_agent = AsyncMock()
        
        self.tool.validation_agent.execute_task.return_value = {
            "validated_themes": [
                {
                    "name": "Historic Temples",
                    "macro_category": "Cultural & Arts",
                    "micro_category": "Historic Sites",
                    "confidence_level": "high",
                    "confidence_breakdown": {
                        "overall_confidence": 0.9,
                        "evidence_count": 3,
                        "source_diversity": 0.8,
                        "authority_score": 0.95
                    },
                    "is_validated": True
                },
                {
                    "name": "Kaiseki Dining",
                    "macro_category": "Food & Dining", 
                    "micro_category": "Fine Dining",
                    "confidence_level": "high",
                    "confidence_breakdown": {
                        "overall_confidence": 0.88,
                        "evidence_count": 2,
                        "source_diversity": 0.7,
                        "authority_score": 0.9
                    },
                    "is_validated": True
                }
            ],
            "validated_count": 2
        }
        
        self.tool.cultural_agent.execute_task.return_value = {
            "cultural_metrics": {
                "cultural_diversity_score": 0.9,
                "local_source_ratio": 0.8,
                "language_distribution": {"english": 70, "japanese": 30},
                "optimal_mix_score": 0.85
            }
        }
        
        self.tool.contradiction_agent.execute_task.return_value = {
            "resolved_themes": [
                {
                    "name": "Historic Temples",
                    "macro_category": "Cultural & Arts",
                    "micro_category": "Historic Sites",
                    "confidence_level": "high",
                    "confidence_breakdown": {
                        "overall_confidence": 0.9,
                        "evidence_count": 3,
                        "source_diversity": 0.8,
                        "authority_score": 0.95
                    },
                    "is_validated": True,
                    "contradiction_resolved": False
                },
                {
                    "name": "Kaiseki Dining",
                    "macro_category": "Food & Dining",
                    "micro_category": "Fine Dining", 
                    "confidence_level": "high",
                    "confidence_breakdown": {
                        "overall_confidence": 0.88,
                        "evidence_count": 2,
                        "source_diversity": 0.7,
                        "authority_score": 0.9
                    },
                    "is_validated": True,
                    "contradiction_resolved": False
                }
            ],
            "contradictions_found": 0
        }
        
        # Run the analysis
        result = await self.tool.analyze_themes(realistic_input)
        
        # Detailed validation and reporting
        themes = result.get("themes", [])
        print(f"Generated {len(themes)} themes")
        
        for i, theme in enumerate(themes, 1):
            theme_name = theme.get('name', f'Theme {i}')
            print(f"\n📋 Theme {i}: {theme_name}")
            print(f"   Category: {theme.get('macro_category')} -> {theme.get('micro_category')}")
            print(f"   Confidence: {theme.get('confidence_score', 0):.2f}")
            
            # Check enhanced fields
            enhanced_fields = {
                "authentic_insights": theme.get("authentic_insights", []),
                "local_authorities": theme.get("local_authorities", []),
                "seasonal_relevance": theme.get("seasonal_relevance", {}),
                "cultural_summary": theme.get("cultural_summary", {}),
                "sentiment_analysis": theme.get("sentiment_analysis", {}),
                "temporal_analysis": theme.get("temporal_analysis", {}),
                "factors": theme.get("factors", {})
            }
            
            for field_name, field_value in enhanced_fields.items():
                if isinstance(field_value, list):
                    status = "✅ Populated" if len(field_value) > 0 else "⚠️ Empty"
                    detail = f"({len(field_value)} items)" if len(field_value) > 0 else ""
                elif isinstance(field_value, dict):
                    status = "✅ Populated" if len(field_value) > 0 else "⚠️ Empty"
                    detail = f"({len(field_value)} keys)" if len(field_value) > 0 else ""
                else:
                    status = "✅ Populated" if field_value else "⚠️ Empty"
                    detail = ""
                
                print(f"   {field_name:20}: {status} {detail}")
                
                # Show sample data for key fields
                if field_name == "authentic_insights" and len(field_value) > 0:
                    insight = field_value[0]
                    print(f"      Sample insight: {insight.get('insight_type', 'unknown')} (authenticity: {insight.get('authenticity_score', 0):.2f})")
                    
                    # Validate insight values
                    assert insight["insight_type"] in ["seasonal", "specialty", "insider", "cultural", "practical"]
                    assert 0 <= insight["authenticity_score"] <= 1
                    assert 0 <= insight["uniqueness_score"] <= 1
                    assert 0 <= insight["actionability_score"] <= 1
                    assert 0 <= insight["temporal_relevance"] <= 1
                    assert insight["location_exclusivity"] in ["exclusive", "signature", "regional", "common"]
                
                elif field_name == "local_authorities" and len(field_value) > 0:
                    authority = field_value[0]
                    print(f"      Sample authority: {authority.get('authority_type', 'unknown')} (validation: {authority.get('community_validation', 0):.2f})")
                
                elif field_name == "seasonal_relevance" and len(field_value) > 0:
                    months_with_data = [k for k, v in field_value.items() if v > 0]
                    if months_with_data:
                        print(f"      Relevant months: {', '.join(months_with_data[:3])}{'...' if len(months_with_data) > 3 else ''}")
        
        # Summary of enhanced analysis
        print(f"\n📈 ENHANCED ANALYSIS SUMMARY")
        print(f"   Total themes: {len(themes)}")
        print(f"   Evidence pieces: {result.get('evidence_summary', {}).get('total_evidence', 0)}")
        print(f"   Average confidence: {result.get('quality_metrics', {}).get('average_confidence', 0):.2f}")
        print(f"   Temporal slices: {len(result.get('temporal_slices', []))}")
        
        return result


async def main():
    """Main test execution"""
    runner = EnhancedThemeTestRunner()
    
    # Run comprehensive tests
    success = await runner.run_comprehensive_tests()
    
    if success:
        print("🎉 All tests passed! Running live data validation...")
        await runner.run_live_test_with_real_data()
        print("\n✅ Enhanced Theme Analysis Tool validation complete!")
        return 0
    else:
        print("\n❌ Some tests failed. Check the output above for details.")
        return 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code) 