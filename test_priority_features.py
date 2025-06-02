#!/usr/bin/env python3
"""
Unit tests for priority-focused traveler concern features
"""
import unittest
from datetime import datetime
from unittest.mock import Mock, patch, MagicMock

from src.tools.priority_data_extraction_tool import (
    PriorityDataExtractor,
    SafetyMetrics,
    CostIndicators,
    HealthRequirements,
    AccessibilityInfo
)
from src.tools.priority_aggregation_tool import PriorityAggregationTool
from src.schemas import PageContent, PriorityMetrics


class TestPriorityDataExtraction(unittest.TestCase):
    """Test priority data extraction functionality"""
    
    def setUp(self):
        self.extractor = PriorityDataExtractor()
    
    def test_extract_safety_metrics(self):
        """Test safety metrics extraction"""
        content = """
        The city has a crime index of 35.2, which is relatively low. 
        Tourist police are available 24/7. Emergency number: 911.
        Travel advisory: Level 2 - Exercise increased caution.
        Avoid areas near the old port at night.
        Safe neighborhoods include Downtown and Riverside.
        """
        
        result = self.extractor.extract_safety_metrics(content)
        
        self.assertEqual(result['crime_index'], 35.2)
        self.assertTrue(result['tourist_police_available'])
        self.assertEqual(result['emergency_contacts']['emergency'], '911')
        self.assertEqual(result['travel_advisory_level'], 'Level 2')
        self.assertIn('old port', ' '.join(result['areas_to_avoid']))
        self.assertIn('Downtown', result['safe_areas'])
    
    def test_extract_cost_indicators(self):
        """Test cost indicators extraction"""
        content = """
        Budget travelers can expect to spend $40-50 per day.
        Mid-range travel costs around $100 per day.
        Luxury travelers should budget $250+ per day.
        Meals cost $5-15, accommodation from $20 per night.
        Currency: USD. High season prices increase by 30%.
        """
        
        result = self.extractor.extract_cost_indicators(content)
        
        self.assertEqual(result['budget_per_day_low'], 40.0)
        self.assertEqual(result['budget_per_day_mid'], 100.0)
        self.assertEqual(result['budget_per_day_high'], 250.0)
        self.assertEqual(result['currency'], 'USD')
        self.assertEqual(result['seasonal_price_variation']['high_season'], 30.0)
    
    def test_extract_health_requirements(self):
        """Test health requirements extraction"""
        content = """
        Required vaccinations: Yellow Fever, Hepatitis A.
        Recommended vaccinations: Typhoid, Malaria prophylaxis.
        Tap water is not safe to drink, bottled water recommended.
        Medical facilities are adequate. Health risks include dengue.
        """
        
        result = self.extractor.extract_health_requirements(content)
        
        self.assertIn('yellow fever', result['required_vaccinations'])
        self.assertIn('typhoid', result['recommended_vaccinations'])
        self.assertEqual(result['water_safety'], 'Bottled water recommended')
        self.assertEqual(result['medical_facility_quality'], 'Adequate')
        self.assertIn('Dengue', result['health_risks'])
    
    def test_extract_accessibility_info(self):
        """Test accessibility information extraction"""
        content = """
        Visa required for US citizens. Visa on arrival available.
        Visa costs $50. Direct flights from New York and Miami.
        English is widely spoken. Infrastructure rating: good.
        Public transport is reliable and affordable.
        """
        
        result = self.extractor.extract_accessibility_info(content)
        
        self.assertTrue(result['visa_required'])
        self.assertTrue(result['visa_on_arrival'])
        self.assertEqual(result['visa_cost'], 50.0)
        self.assertIn('new york', result['direct_flights_from_major_hubs'])
        self.assertEqual(result['english_proficiency'], 'High')
        self.assertEqual(result['infrastructure_rating'], 4.0)
    
    def test_source_credibility_calculation(self):
        """Test source credibility scoring"""
        # Government source
        self.assertEqual(
            self.extractor.calculate_source_credibility('https://travel.state.gov'),
            0.9
        )
        
        # Major travel platform
        self.assertEqual(
            self.extractor.calculate_source_credibility('https://tripadvisor.com'),
            0.8
        )
        
        # News source
        self.assertEqual(
            self.extractor.calculate_source_credibility('https://bbc.com'),
            0.75
        )
        
        # Community source
        self.assertEqual(
            self.extractor.calculate_source_credibility('https://reddit.com'),
            0.7
        )
        
        # Unknown source
        self.assertEqual(
            self.extractor.calculate_source_credibility('https://unknown-site.com'),
            0.6
        )
    
    def test_temporal_relevance(self):
        """Test temporal relevance calculation"""
        # Current year content
        current_year = datetime.now().year
        content = f"Updated in {current_year} with latest information."
        self.assertEqual(self.extractor.determine_temporal_relevance(content), 1.0)
        
        # Two years old
        old_content = f"This information from {current_year - 2}."
        self.assertEqual(self.extractor.determine_temporal_relevance(old_content), 0.8)
        
        # Recent indicator
        recent_content = "Recently updated travel information."
        self.assertEqual(self.extractor.determine_temporal_relevance(recent_content), 0.9)


class TestPriorityAggregation(unittest.TestCase):
    """Test priority data aggregation functionality"""
    
    def setUp(self):
        self.aggregator = PriorityAggregationTool()
    
    def test_aggregate_safety_data(self):
        """Test safety data aggregation"""
        page_contents = [
            self._create_page_content({
                "safety": {
                    "crime_index": 30.0,
                    "tourist_police_available": True,
                    "areas_to_avoid": ["downtown at night"]
                }
            }, credibility=0.9),
            self._create_page_content({
                "safety": {
                    "crime_index": 35.0,
                    "tourist_police_available": True,
                    "areas_to_avoid": ["old port"]
                }
            }, credibility=0.8)
        ]
        
        result = self.aggregator._run(
            destination_name="Test City",
            page_contents=page_contents,
            confidence_threshold=0.6
        )
        
        metrics = result['priority_metrics']
        self.assertIsNotNone(metrics.crime_index)
        self.assertTrue(metrics.tourist_police_available)
        self.assertGreater(result['aggregation_confidence'], 0.7)
    
    def test_aggregate_cost_data(self):
        """Test cost data aggregation"""
        page_contents = [
            self._create_page_content({
                "cost": {
                    "budget_per_day_low": 40.0,
                    "budget_per_day_mid": 100.0,
                    "currency": "USD"
                }
            }),
            self._create_page_content({
                "cost": {
                    "budget_per_day_low": 50.0,
                    "budget_per_day_mid": 120.0,
                    "currency": "USD"
                }
            })
        ]
        
        result = self.aggregator._run(
            destination_name="Test City",
            page_contents=page_contents
        )
        
        metrics = result['priority_metrics']
        # Should be median of [40, 50] = 45
        self.assertEqual(metrics.budget_per_day_low, 45.0)
        # Should be median of [100, 120] = 110
        self.assertEqual(metrics.budget_per_day_mid, 110.0)
        self.assertEqual(metrics.currency, "USD")
    
    def test_priority_insights_generation(self):
        """Test priority insights are generated correctly"""
        page_contents = [
            self._create_page_content({
                "safety": {"areas_to_avoid": ["downtown", "port area"]},
                "health": {"required_vaccinations": ["Yellow Fever", "Hepatitis A"]},
                "accessibility": {"visa_required": True, "visa_cost": 50.0}
            })
        ]
        
        result = self.aggregator._run(
            destination_name="Test City",
            page_contents=page_contents
        )
        
        insights = result['priority_insights']
        self.assertGreater(len(insights), 0)
        
        # Check for specific insight types
        insight_names = [i.insight_name for i in insights]
        self.assertIn("Areas to Avoid", insight_names)
        self.assertIn("Required Vaccinations", insight_names)
        self.assertIn("Visa Requirements", insight_names)
    
    def test_confidence_calculation(self):
        """Test aggregation confidence calculation"""
        # High credibility sources
        high_cred_pages = [
            self._create_page_content({"safety": {"crime_index": 30}}, credibility=0.9)
            for _ in range(3)
        ]
        
        # Low credibility sources
        low_cred_pages = [
            self._create_page_content({"safety": {"crime_index": 30}}, credibility=0.5)
            for _ in range(3)
        ]
        
        high_result = self.aggregator._run("Test", high_cred_pages)
        low_result = self.aggregator._run("Test", low_cred_pages)
        
        self.assertGreater(
            high_result['aggregation_confidence'],
            low_result['aggregation_confidence']
        )
    
    def _create_page_content(self, priority_data, credibility=0.7):
        """Helper to create PageContent with priority data"""
        page = PageContent(
            url="https://test.com",
            title="Test",
            content="Test content",
            content_length=100
        )
        page.priority_data = {
            **priority_data,
            "source_credibility": credibility,
            "temporal_relevance": 0.8,
            "source_url": "https://test.com"
        }
        return page


class TestWebDiscoveryIntegration(unittest.TestCase):
    """Test web discovery priority integration"""
    
    @patch('src.core.web_discovery_logic.WebDiscoveryLogic._fetch_brave_search')
    @patch('src.core.web_discovery_logic.WebDiscoveryLogic._fetch_page_content')
    async def test_priority_discovery_enabled(self, mock_fetch_page, mock_brave_search):
        """Test that priority discovery generates additional queries"""
        from src.core.web_discovery_logic import WebDiscoveryLogic
        
        # Mock responses
        mock_brave_search.return_value = [
            {"url": "https://test1.com", "title": "Test 1", "snippet": "Safety info"},
            {"url": "https://test2.com", "title": "Test 2", "snippet": "Cost info"}
        ]
        mock_fetch_page.return_value = "Sample content about the destination"
        
        config = {
            "priority_settings": {"enable_priority_discovery": True},
            "web_discovery": {"search_results_per_query": 2}
        }
        
        logic = WebDiscoveryLogic("fake_api_key", config)
        
        # Generate priority queries
        queries = logic.generate_priority_focused_queries("Paris, France")
        
        # Should have queries for all priority categories
        self.assertIn("safety", queries)
        self.assertIn("cost", queries)
        self.assertIn("health", queries)
        self.assertIn("weather", queries)
        self.assertIn("accessibility", queries)
        
        # Each category should have multiple queries
        for category, query_list in queries.items():
            self.assertGreater(len(query_list), 0)
            # Check queries are properly formatted
            for query in query_list:
                self.assertIn("Paris, France", query)


class TestEnhancedThemeAnalysisIntegration(unittest.TestCase):
    """Test enhanced theme analysis with priority integration"""
    
    @patch('src.tools.enhanced_theme_analysis_tool.PriorityAggregationTool._arun')
    async def test_priority_integration_in_theme_analysis(self, mock_aggregator):
        """Test that theme analysis integrates priority data"""
        from src.tools.enhanced_theme_analysis_tool import EnhancedAnalyzeThemesFromEvidenceTool
        
        # Mock priority aggregation result
        mock_aggregator.return_value = {
            "priority_metrics": PriorityMetrics(
                safety_score=8.5,
                crime_index=25.0,
                budget_per_day_low=50.0
            ),
            "priority_insights": [
                MagicMock(
                    insight_name="Low Crime Rate",
                    priority_category="safety",
                    confidence_score=0.85
                )
            ]
        }
        
        tool = EnhancedAnalyzeThemesFromEvidenceTool()
        
        # Create test content with priority data
        page_content = PageContent(
            url="https://test.com",
            title="Test",
            content="Test content about hiking and restaurants",
            content_length=100
        )
        page_content.priority_data = {"safety": {"crime_index": 25.0}}
        
        # Run analysis
        result = await tool._create_analysis_function(None, None)(
            destination_name="Test City",
            text_content_list=[page_content]
        )
        
        # Check that priority data is included
        self.assertIsNotNone(result.priority_metrics)
        self.assertEqual(len(result.priority_insights), 1)


if __name__ == "__main__":
    unittest.main() 