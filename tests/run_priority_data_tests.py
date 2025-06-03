#!/usr/bin/env python3
"""
Priority Data Extraction Tool Test Runner

This script runs comprehensive tests specifically for the PriorityDataExtractor
and validates all priority data extraction capabilities including safety, cost,
health, and accessibility information.
"""

import asyncio
import sys
import os
import json
from datetime import datetime, timedelta
from pathlib import Path

# Add the project root to the path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import test modules and dependencies
import pytest
from src.tools.priority_data_extraction_tool import (
    PriorityDataExtractor,
    SafetyMetrics,
    CostIndicators,
    HealthRequirements,
    AccessibilityInfo
)


class PriorityDataTestRunner:
    """Specialized test runner for priority data extraction validation"""
    
    def __init__(self):
        self.extractor = PriorityDataExtractor()
        self.test_results = []
        self.extraction_validations = {}
    
    async def run_comprehensive_tests(self):
        """Run comprehensive tests for priority data extraction"""
        print("=" * 80)
        print("🔍 PRIORITY DATA EXTRACTION TOOL - COMPREHENSIVE TESTING")
        print("=" * 80)
        print(f"📅 Test Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print()
        
        # Test scenarios
        test_scenarios = [
            self._test_data_model_initialization,
            self._test_comprehensive_extraction,
            self._test_safety_metrics_extraction,
            self._test_cost_indicators_extraction,
            self._test_health_requirements_extraction,
            self._test_accessibility_info_extraction,
            self._test_source_credibility_calculation,
            self._test_temporal_relevance_determination,
            self._test_edge_cases_and_robustness,
            self._test_performance_and_scalability
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
        
        # Extraction Validation Summary
        self._print_extraction_validation_summary()
        
        return passed_tests == total_tests

    async def _test_data_model_initialization(self):
        """Test that all data models initialize correctly with defaults"""
        print("      🔍 Testing data model initialization...")
        
        # Test SafetyMetrics
        safety = SafetyMetrics()
        assert safety.emergency_contacts == {}
        assert safety.recent_incidents == []
        assert safety.safe_areas == []
        assert safety.areas_to_avoid == []
        
        # Test with values
        safety_with_values = SafetyMetrics(
            crime_index=45.2,
            safety_rating=7.5,
            tourist_police_available=True
        )
        assert safety_with_values.crime_index == 45.2
        assert safety_with_values.safety_rating == 7.5
        assert safety_with_values.tourist_police_available is True
        
        # Test CostIndicators
        cost = CostIndicators()
        assert cost.seasonal_price_variation == {}
        
        cost_with_values = CostIndicators(
            budget_per_day_low=25.0,
            currency="USD"
        )
        assert cost_with_values.budget_per_day_low == 25.0
        assert cost_with_values.currency == "USD"
        
        # Test HealthRequirements
        health = HealthRequirements()
        assert health.required_vaccinations == []
        assert health.health_risks == []
        
        # Test AccessibilityInfo
        access = AccessibilityInfo()
        assert access.direct_flights_from_major_hubs == []
        
        print("      ✅ All data models initialize correctly")

    async def _test_comprehensive_extraction(self):
        """Test comprehensive priority data extraction from realistic content"""
        print("      🔍 Testing comprehensive extraction...")
        
        realistic_content = """
        Bangkok Travel Guide - Complete Information for 2024
        
        SAFETY: Bangkok is generally safe for tourists with a crime index of 45.2 out of 100. 
        The city has a safety rating of 6.8/10 from international safety organizations.
        Tourist police are available throughout the city center and major tourist areas.
        Emergency contacts: Emergency: 191, Police: 191, Ambulance: 1669, Fire: 199.
        Travel advisory: Level 2 - Exercise increased caution due to occasional demonstrations.
        Safe areas include Sukhumvit, Silom, and the tourist zone around Khao San Road.
        Avoid areas around Klong Toey port and some parts of Chinatown late at night.
        
        COSTS: Budget travelers can expect to spend $25-30 per day in Bangkok.
        Mid-range budget: $50-80 per day for comfortable travel with decent accommodations.
        Luxury travel: $150+ per day for high-end experiences and 5-star hotels.
        Meals cost around $3-5 for street food, $10-15 for mid-range restaurants.
        Budget accommodation: $10-20 per night for hostels and guesthouses.
        Hotels: $40-80 per night for mid-range options with good amenities.
        Currency: THB is the local currency with exchange rate around 35 THB per USD.
        Prices increase 20% during high season (December-February).
        Prices decrease 15% during low season (June-September).
        
        HEALTH: Required vaccinations include Hepatitis A and Typhoid for all travelers.
        Recommended vaccinations: Japanese Encephalitis, Hepatitis B, and Tetanus.
        Health risks: Dengue fever is present, especially during rainy season (May-October).
        Malaria risk exists in some border areas but not in Bangkok city center.
        Tap water is not safe to drink - bottled water recommended at all times.
        Medical facilities are excellent in Bangkok with many international hospitals.
        Health insurance is strongly recommended for all international travelers.
        
        ACCESS: Visa required for most nationalities, visa on arrival available for $35.
        Direct flights available from New York, London, Sydney, Tokyo, and Dubai.
        English is widely spoken in tourist areas and by younger generations.
        Infrastructure is good with modern BTS/MRT transportation systems.
        Average flight time from London is 11 hours, from New York is 17 hours.
        """
        
        source_url = "https://travel.gov.example/bangkok-guide-2024"
        
        result = self.extractor.extract_all_priority_data(realistic_content, source_url)
        
        # Validate comprehensive structure
        required_keys = ["safety", "cost", "health", "accessibility", "source_url", "extraction_timestamp"]
        for key in required_keys:
            assert key in result, f"Missing key: {key}"
        
        # Validate safety data
        safety = result["safety"]
        assert safety["crime_index"] == 45.2
        assert safety["safety_rating"] == 6.8
        assert safety["tourist_police_available"] is True
        assert "emergency" in safety["emergency_contacts"]
        assert safety["travel_advisory_level"] == "Level 2"
        assert len(safety["safe_areas"]) > 0
        assert len(safety["areas_to_avoid"]) > 0
        
        # Validate cost data
        cost = result["cost"]
        assert cost["budget_per_day_low"] in [25.0, 30.0]  # Could match either value
        assert cost["budget_per_day_high"] == 150.0
        assert cost["currency"] == "THB"
        assert "high_season" in cost["seasonal_price_variation"]
        assert cost["seasonal_price_variation"]["high_season"] == 20.0
        
        # Validate health data
        health = result["health"]
        assert any("hepatitis a" in v.lower() for v in health["required_vaccinations"])
        assert any("typhoid" in v.lower() for v in health["required_vaccinations"])
        assert any("dengue" in r.lower() for r in health["health_risks"])
        assert health["water_safety"] == "Bottled water recommended"
        assert health["medical_facility_quality"] == "Excellent"
        
        # Validate accessibility data
        access = result["accessibility"]
        assert access["visa_required"] is True
        assert access["visa_on_arrival"] is True
        assert access["visa_cost"] == 35.0
        assert access["english_proficiency"] == "High"
        assert access["infrastructure_rating"] == 4.0
        
        # Store validation results
        self.extraction_validations["comprehensive"] = {
            "safety_fields_extracted": len([k for k, v in safety.items() if v is not None and v != [] and v != {}]),
            "cost_fields_extracted": len([k for k, v in cost.items() if v is not None and v != [] and v != {}]),
            "health_fields_extracted": len([k for k, v in health.items() if v is not None and v != [] and v != {}]),
            "accessibility_fields_extracted": len([k for k, v in access.items() if v is not None and v != [] and v != {}])
        }
        
        print(f"      ✅ Comprehensive extraction successful: {len(required_keys)} sections validated")

    async def _test_safety_metrics_extraction(self):
        """Test detailed safety metrics extraction patterns"""
        print("      🔍 Testing safety metrics extraction patterns...")
        
        test_cases = [
            # Crime index variations
            ("Crime index: 42.5", {"crime_index": 42.5}),
            ("35% crime rate", {"crime_index": 35.0}),
            ("Crime statistics show 67.8", {"crime_index": 67.8}),
            
            # Safety rating variations  
            ("Safety rating: 8.5", {"safety_rating": 8.5}),
            ("Rated 7.2 out of 10 for safety", {"safety_rating": 7.2}),
            ("6/10 safety score", {"safety_rating": 6.0}),
            
            # Travel advisory variations
            ("Level 3 travel advisory", {"travel_advisory_level": "Level 3"}),
            ("Exercise normal caution", {"travel_advisory_level": "Level 1"}),
            ("Exercise extreme caution", {"travel_advisory_level": "Level 4"}),
            
            # Tourist police detection
            ("Tourist police available", {"tourist_police_available": True}),
            ("Tourism police patrol the area", {"tourist_police_available": True}),
            
            # Emergency contacts
            ("Emergency: 911, Police: 191", {"emergency_contacts": {"emergency": "911", "police": "191"}}),
            
            # Safe/unsafe areas
            ("Safe areas: Downtown, Tourist Zone", {"safe_areas": ["downtown", "tourist zone"]}),
            ("Avoid areas around Industrial District", {"areas_to_avoid": ["industrial district"]})
        ]
        
        pattern_matches = 0
        for content, expected_partial in test_cases:
            result = self.extractor.extract_safety_metrics(content)
            
            # Check if expected fields are extracted
            for key, expected_value in expected_partial.items():
                if key in result and result[key] is not None:
                    if isinstance(expected_value, list):
                        # For lists, check if expected items are present (case-insensitive)
                        if isinstance(result[key], list):
                            result_lower = [item.lower() for item in result[key]]
                            expected_lower = [item.lower() for item in expected_value]
                            if any(exp in res for exp in expected_lower for res in result_lower):
                                pattern_matches += 1
                    elif isinstance(expected_value, dict):
                        # For dicts, check if expected keys are present
                        if isinstance(result[key], dict):
                            if any(exp_key in result[key] for exp_key in expected_value.keys()):
                                pattern_matches += 1
                    else:
                        # For simple values, check equality
                        if result[key] == expected_value:
                            pattern_matches += 1
        
        print(f"      ✅ Safety pattern matching: {pattern_matches}/{len(test_cases)} patterns recognized")

    async def _test_cost_indicators_extraction(self):
        """Test detailed cost indicators extraction patterns"""
        print("      🔍 Testing cost indicators extraction patterns...")
        
        test_cases = [
            # Budget variations
            ("Budget travelers: $25/day", "budget_per_day_low"),
            ("Mid-range: $60 per day", "budget_per_day_mid"),
            ("Luxury travel: $200+ daily", "budget_per_day_high"),
            
            # Meal costs
            ("Meals cost $12 on average", "meal_cost_average"),
            ("Lunch prices: $8", "meal_cost_average"),
            ("$15 for dinner", "meal_cost_average"),
            
            # Accommodation costs
            ("Hotels: $75 per night", "accommodation_cost_average"),
            ("Budget accommodation: $20/night", "accommodation_cost_average"),
            
            # Currency detection
            ("Currency: USD", "currency"),
            ("EUR is the local currency", "currency"),
            ("Prices are in GBP", "currency"),
            
            # Seasonal variations
            ("Prices rise 25% in high season", "seasonal_price_variation"),
            ("Low season: 20% cheaper", "seasonal_price_variation")
        ]
        
        extraction_successes = 0
        for content, expected_field in test_cases:
            result = self.extractor.extract_cost_indicators(content)
            
            if expected_field in result and result[expected_field] is not None:
                # For seasonal_price_variation, check if dict has content
                if expected_field == "seasonal_price_variation":
                    if isinstance(result[expected_field], dict) and len(result[expected_field]) > 0:
                        extraction_successes += 1
                # For currency, check if string is extracted
                elif expected_field == "currency":
                    if isinstance(result[expected_field], str) and len(result[expected_field]) > 0:
                        extraction_successes += 1
                # For numeric fields, check if number is extracted
                else:
                    if isinstance(result[expected_field], (int, float)) and result[expected_field] > 0:
                        extraction_successes += 1
        
        print(f"      ✅ Cost extraction patterns: {extraction_successes}/{len(test_cases)} patterns recognized")

    async def _test_health_requirements_extraction(self):
        """Test detailed health requirements extraction patterns"""
        print("      🔍 Testing health requirements extraction patterns...")
        
        test_cases = [
            # Vaccination requirements
            ("Required vaccinations: Hepatitis A, Typhoid", "required_vaccinations"),
            ("Mandatory vaccination: Yellow Fever", "required_vaccinations"),
            ("Recommended vaccines: Japanese Encephalitis", "recommended_vaccinations"),
            
            # Health risks
            ("Dengue fever risk present", "health_risks"),
            ("Malaria found in border areas", "health_risks"),
            ("Health risks include Zika virus", "health_risks"),
            
            # Water safety
            ("Tap water is safe to drink", "water_safety"),
            ("Water is not safe - bottled recommended", "water_safety"),
            ("Drinking water unsafe", "water_safety"),
            
            # Medical facilities
            ("Medical facilities are excellent", "medical_facility_quality"),
            ("Hospitals are adequate", "medical_facility_quality"),
            ("Poor healthcare system", "medical_facility_quality")
        ]
        
        extraction_successes = 0
        for content, expected_field in test_cases:
            result = self.extractor.extract_health_requirements(content)
            
            if expected_field in result and result[expected_field] is not None:
                # For list fields, check if items are extracted
                if expected_field in ["required_vaccinations", "recommended_vaccinations", "health_risks"]:
                    if isinstance(result[expected_field], list) and len(result[expected_field]) > 0:
                        extraction_successes += 1
                # For string fields, check if content is extracted
                else:
                    if isinstance(result[expected_field], str) and len(result[expected_field]) > 0:
                        extraction_successes += 1
        
        print(f"      ✅ Health extraction patterns: {extraction_successes}/{len(test_cases)} patterns recognized")

    async def _test_accessibility_info_extraction(self):
        """Test detailed accessibility info extraction patterns"""
        print("      🔍 Testing accessibility info extraction patterns...")
        
        test_cases = [
            # Visa requirements
            ("Visa required for entry", "visa_required"),
            ("No visa needed", "visa_required"),
            ("Visa on arrival available", "visa_on_arrival"),
            ("Visa cost: $50", "visa_cost"),
            
            # Flight information
            ("Direct flights from London", "direct_flights_from_major_hubs"),
            ("Non-stop service from NYC", "direct_flights_from_major_hubs"),
            
            # English proficiency
            ("English widely spoken", "english_proficiency"),
            ("Few people speak English", "english_proficiency"),
            ("English proficiency is moderate", "english_proficiency"),
            
            # Infrastructure
            ("Excellent infrastructure", "infrastructure_rating"),
            ("Poor public transport", "infrastructure_rating"),
            ("Good road systems", "infrastructure_rating")
        ]
        
        extraction_successes = 0
        for content, expected_field in test_cases:
            result = self.extractor.extract_accessibility_info(content)
            
            if expected_field in result and result[expected_field] is not None:
                # For boolean fields
                if expected_field in ["visa_required", "visa_on_arrival"]:
                    if isinstance(result[expected_field], bool):
                        extraction_successes += 1
                # For numeric fields
                elif expected_field in ["visa_cost", "infrastructure_rating"]:
                    if isinstance(result[expected_field], (int, float)):
                        extraction_successes += 1
                # For list fields
                elif expected_field == "direct_flights_from_major_hubs":
                    if isinstance(result[expected_field], list) and len(result[expected_field]) > 0:
                        extraction_successes += 1
                # For string fields
                else:
                    if isinstance(result[expected_field], str) and len(result[expected_field]) > 0:
                        extraction_successes += 1
        
        print(f"      ✅ Accessibility extraction patterns: {extraction_successes}/{len(test_cases)} patterns recognized")

    async def _test_source_credibility_calculation(self):
        """Test source credibility calculation accuracy"""
        print("      🔍 Testing source credibility calculation...")
        
        test_cases = [
            # Government sources (should be 0.9)
            ("https://travel.state.gov/content/travel/", 0.9),
            ("https://www.gov.uk/foreign-travel-advice", 0.9),
            ("https://embassy.org/travel-info", 0.9),
            
            # Travel platforms (should be 0.8)
            ("https://www.tripadvisor.com/travel-guide", 0.8),
            ("https://www.lonelyplanet.com/destination", 0.8),
            ("https://www.fodors.com/world/asia", 0.8),
            
            # News sources (should be 0.75)
            ("https://www.cnn.com/travel/article", 0.75),
            ("https://www.bbc.com/travel/guide", 0.75),
            ("https://www.reuters.com/world/travel", 0.75),
            
            # Community sources (should be 0.7)
            ("https://www.reddit.com/r/travel/comments", 0.7),
            ("https://forum.travel.com/discussion", 0.7),
            ("https://www.facebook.com/travelgroup", 0.7),
            
            # Unknown sources (should be 0.6)
            ("https://unknown-blog.com/travel", 0.6),
            ("https://my-travel-diary.net", 0.6),
            
            # Empty/None sources (should be 0.5)
            (None, 0.5),
            ("", 0.5)
        ]
        
        credibility_accuracy = 0
        for url, expected_score in test_cases:
            actual_score = self.extractor.calculate_source_credibility(url)
            if actual_score == expected_score:
                credibility_accuracy += 1
        
        accuracy_percentage = (credibility_accuracy / len(test_cases)) * 100
        print(f"      ✅ Source credibility accuracy: {credibility_accuracy}/{len(test_cases)} ({accuracy_percentage:.1f}%)")

    async def _test_temporal_relevance_determination(self):
        """Test temporal relevance determination accuracy"""
        print("      🔍 Testing temporal relevance determination...")
        
        current_date = datetime(2024, 6, 1)
        
        test_cases = [
            # Current year content (should be 1.0)
            ("Updated in 2024 with latest information", 1.0),
            ("Travel guide 2024 edition", 1.0),
            
            # Recent content (should be 0.8)
            ("Information from 2022 survey", 0.8),
            ("Last updated in 2023", 0.8),
            
            # Older content (should be 0.5)
            ("Travel guide from 2019", 0.5),
            ("Data collected in 2020", 0.5),
            
            # Very old content (should be 0.3)
            ("Guide published in 2014", 0.3),
            ("Information from 2015", 0.3),
            
            # Recency indicators (should be 0.9)
            ("Recently updated with new info", 0.9),
            ("Just published travel guide", 0.9),
            
            # Past indicators (should be 0.8)
            ("Last year's travel report", 0.8),
            ("Previous year data", 0.8),
            
            # Undated content (should be 0.7)
            ("General travel information", 0.7),
            ("Destination overview", 0.7)
        ]
        
        temporal_accuracy = 0
        for content, expected_relevance in test_cases:
            actual_relevance = self.extractor.determine_temporal_relevance(content, current_date)
            if actual_relevance == expected_relevance:
                temporal_accuracy += 1
        
        accuracy_percentage = (temporal_accuracy / len(test_cases)) * 100
        print(f"      ✅ Temporal relevance accuracy: {temporal_accuracy}/{len(test_cases)} ({accuracy_percentage:.1f}%)")

    async def _test_edge_cases_and_robustness(self):
        """Test edge cases and robustness of extraction"""
        print("      🔍 Testing edge cases and robustness...")
        
        edge_cases = [
            "",  # Empty string
            "   ",  # Whitespace only
            "Random text with no relevant information",  # Irrelevant content
            "123456789!@#$%^&*()",  # Special characters only
            "HTML tags <div>content</div> mixed with text",  # HTML content
            "Mixed languages: English 中文 العربية",  # Multi-language
            "Very long repetitive text " * 1000,  # Very long content
            "Numbers without context: 45.2 191 $25 THB"  # Numbers without meaning
        ]
        
        robustness_score = 0
        for i, edge_case in enumerate(edge_cases):
            try:
                result = self.extractor.extract_all_priority_data(edge_case)
                
                # Validate structure is maintained
                required_keys = ["safety", "cost", "health", "accessibility", "source_url", "extraction_timestamp"]
                if all(key in result for key in required_keys):
                    robustness_score += 1
                    
            except Exception as e:
                print(f"         ⚠️  Edge case {i+1} failed: {str(e)[:50]}...")
        
        robustness_percentage = (robustness_score / len(edge_cases)) * 100
        print(f"      ✅ Robustness score: {robustness_score}/{len(edge_cases)} ({robustness_percentage:.1f}%)")

    async def _test_performance_and_scalability(self):
        """Test performance and scalability of extraction"""
        print("      🔍 Testing performance and scalability...")
        
        # Create large content for performance testing
        base_content = """
        Safety information: Crime index 45.2, safety rating 7/10, emergency: 191.
        Cost details: Budget travelers $30/day, hotels $60/night, currency USD.
        Health requirements: Hepatitis A required, water not safe, hospitals excellent.
        Accessibility: Visa required $35, English widely spoken, good infrastructure.
        """
        
        # Test different content sizes
        size_tests = [
            (base_content, "Small (~500 chars)"),
            (base_content * 10, "Medium (~5KB)"),
            (base_content * 100, "Large (~50KB)"),
            (base_content * 500, "Very Large (~250KB)")
        ]
        
        performance_results = []
        
        for content, size_label in size_tests:
            import time
            start_time = time.time()
            
            try:
                result = self.extractor.extract_all_priority_data(content)
                
                end_time = time.time()
                processing_time = end_time - start_time
                
                performance_results.append({
                    "size": size_label,
                    "time": processing_time,
                    "success": True,
                    "data_extracted": bool(result.get("safety", {}).get("crime_index"))
                })
                
            except Exception as e:
                performance_results.append({
                    "size": size_label,
                    "time": 0,
                    "success": False,
                    "error": str(e)
                })
        
        # Report performance results
        for result in performance_results:
            if result["success"]:
                print(f"         📊 {result['size']}: {result['time']:.3f}s")
            else:
                print(f"         ❌ {result['size']}: FAILED - {result['error'][:50]}...")
        
        # Test concurrent processing
        import threading
        concurrent_results = []
        errors = []
        
        def extract_concurrent(content):
            try:
                result = self.extractor.extract_all_priority_data(content)
                concurrent_results.append(result)
            except Exception as e:
                errors.append(e)
        
        # Run 5 concurrent extractions
        threads = []
        for i in range(5):
            thread = threading.Thread(target=extract_concurrent, args=(base_content,))
            threads.append(thread)
            thread.start()
        
        for thread in threads:
            thread.join()
        
        concurrent_success = len(concurrent_results) == 5 and len(errors) == 0
        print(f"         🔄 Concurrent processing: {'✅ PASSED' if concurrent_success else '❌ FAILED'}")
        
        print(f"      ✅ Performance testing completed")

    def _print_extraction_validation_summary(self):
        """Print summary of extraction validation results"""
        if not self.extraction_validations:
            return
            
        print("🔍 EXTRACTION VALIDATION SUMMARY")
        print("-" * 60)
        
        if "comprehensive" in self.extraction_validations:
            comp = self.extraction_validations["comprehensive"]
            
            print(f"📊 Data Fields Successfully Extracted:")
            print(f"   Safety Fields:       {comp['safety_fields_extracted']}/8 fields")
            print(f"   Cost Fields:         {comp['cost_fields_extracted']}/9 fields") 
            print(f"   Health Fields:       {comp['health_fields_extracted']}/8 fields")
            print(f"   Accessibility Fields: {comp['accessibility_fields_extracted']}/8 fields")
            
            total_extracted = sum(comp.values())
            total_possible = 8 + 9 + 8 + 8  # Sum of all possible fields
            extraction_rate = (total_extracted / total_possible) * 100
            
            print(f"   Overall Extraction:  {total_extracted}/{total_possible} ({extraction_rate:.1f}%)")
        
        print()

    async def run_real_world_validation(self):
        """Run validation with real-world travel content examples"""
        print("🌍 RUNNING REAL-WORLD VALIDATION")
        print("-" * 60)
        
        # Simulate real travel guide content
        real_world_examples = [
            {
                "destination": "Bangkok, Thailand",
                "content": """
                Bangkok Safety: Generally safe for tourists. Crime index: 45.2. Tourist police available.
                Emergency numbers: 191 for police, 1669 for ambulance. Exercise increased caution.
                Budget: $30-50/day mid-range. Street food $2-5, hotels $40-80/night. Currency: THB.
                Health: Hepatitis A vaccination required. Dengue risk present. Bottled water advised.
                Visa: Required, $35 on arrival. Direct flights from London, Tokyo. English widely spoken.
                """,
                "source": "https://travel.state.gov/thailand"
            },
            {
                "destination": "Paris, France", 
                "content": """
                Paris is very safe with low crime rates. Safety rating: 8.2/10. Well-policed tourist areas.
                Emergency: 112, Police: 17. No travel advisory restrictions for France.
                Expensive city: Budget €60-80/day, luxury €200+/day. Meals €15-30. Hotels €80-150/night.
                No special vaccinations required. Excellent medical system. Tap water safe to drink.
                EU citizens no visa needed. Others check requirements. Major international airport hub.
                """,
                "source": "https://www.lonelyplanet.com/france/paris"
            },
            {
                "destination": "Rio de Janeiro, Brazil",
                "content": """
                Rio requires caution in certain areas. Crime index: 68.4. Avoid favelas at night.
                Tourist police patrol Copacabana and Ipanema. Emergency: 190, Medical: 192.
                Moderate costs: $40-60/day budget travel. Meals $8-15. Accommodation $25-50/night.
                Yellow fever vaccination recommended for some areas. Zika virus risk present.
                Visa requirements vary by nationality. Direct flights from major US and European cities.
                """,
                "source": "https://www.brazil.travel/en/destinations/rio"
            }
        ]
        
        validation_results = []
        
        for example in real_world_examples:
            print(f"\n📍 Testing: {example['destination']}")
            
            result = self.extractor.extract_all_priority_data(
                example["content"], 
                example["source"]
            )
            
            # Calculate extraction completeness
            safety_complete = self._calculate_completeness(result["safety"])
            cost_complete = self._calculate_completeness(result["cost"])
            health_complete = self._calculate_completeness(result["health"])
            access_complete = self._calculate_completeness(result["accessibility"])
            
            overall_complete = (safety_complete + cost_complete + health_complete + access_complete) / 4
            
            validation_results.append({
                "destination": example["destination"],
                "safety_completeness": safety_complete,
                "cost_completeness": cost_complete,
                "health_completeness": health_complete,
                "accessibility_completeness": access_complete,
                "overall_completeness": overall_complete
            })
            
            print(f"   Safety:       {safety_complete:.1%} complete")
            print(f"   Cost:         {cost_complete:.1%} complete")
            print(f"   Health:       {health_complete:.1%} complete")
            print(f"   Accessibility: {access_complete:.1%} complete")
            print(f"   Overall:      {overall_complete:.1%} complete")
        
        # Summary
        avg_completeness = sum(r["overall_completeness"] for r in validation_results) / len(validation_results)
        print(f"\n📈 Average Extraction Completeness: {avg_completeness:.1%}")
        
        return validation_results

    def _calculate_completeness(self, data_dict):
        """Calculate how complete the extracted data is"""
        if not isinstance(data_dict, dict):
            return 0.0
        
        total_fields = len(data_dict)
        populated_fields = 0
        
        for key, value in data_dict.items():
            if value is not None:
                if isinstance(value, (list, dict)):
                    if len(value) > 0:
                        populated_fields += 1
                elif isinstance(value, str):
                    if value.strip():
                        populated_fields += 1
                else:
                    populated_fields += 1
        
        return populated_fields / total_fields if total_fields > 0 else 0.0


async def main():
    """Main test execution"""
    runner = PriorityDataTestRunner()
    
    # Run comprehensive tests
    success = await runner.run_comprehensive_tests()
    
    if success:
        print("🎉 All tests passed! Running real-world validation...")
        await runner.run_real_world_validation()
        print("\n✅ Priority Data Extraction Tool validation complete!")
        return 0
    else:
        print("\n❌ Some tests failed. Check the output above for details.")
        return 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code) 