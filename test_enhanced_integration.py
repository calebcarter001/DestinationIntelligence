#!/usr/bin/env python3
"""
Integration tests for enhanced destination intelligence system with priority features
"""
import asyncio
import unittest
import logging
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO)

from src.schemas import PageContent, ThemeInsightOutput, PriorityMetrics, DestinationInsight
from src.agents.enhanced_crewai_destination_analyst import EnhancedCrewAIDestinationAnalyst
from src.tools.web_discovery_tools import DiscoverAndFetchContentTool
from src.tools.enhanced_theme_analysis_tool import EnhancedAnalyzeThemesFromEvidenceTool
from src.tools.priority_data_extraction_tool import PriorityDataExtractor
from src.tools.priority_aggregation_tool import PriorityAggregationTool
from src.core.web_discovery_logic import WebDiscoveryLogic


class TestFullIntegration(unittest.TestCase):
    """Test full integration of enhanced system with priority features"""
    
    def setUp(self):
        """Set up test environment"""
        self.config = {
            "priority_settings": {
                "enable_priority_discovery": True,
                "priority_weights": {
                    "safety": 1.5,
                    "cost": 1.3,
                    "health": 1.2,
                    "accessibility": 1.1,
                    "weather": 1.0
                }
            },
            "web_discovery": {
                "search_results_per_query": 3,
                "max_urls_to_fetch_content_for": 5,
                "min_content_length_chars": 100
            },
            "processing_settings": {
                "content_intelligence": {
                    "min_validated_theme_confidence": 0.5
                }
            }
        }
        
        # Sample content with themes and priority data
        self.sample_content_1 = """
        Bend, Oregon is a paradise for outdoor enthusiasts. The city offers incredible hiking trails 
        through the Cascade Mountains, with popular spots like Pilot Butte and Tumalo Falls. 
        
        Safety Information: Bend has a low crime index of 28.5, making it one of the safer cities 
        in Oregon. Tourist police are available, and the emergency number is 911. The downtown 
        area is very safe, even at night. 
        
        Cost Guide: Budget travelers can manage on $60-80 per day, while mid-range visitors should 
        budget around $150 per day. Luxury accommodations and dining can push costs to $300+ daily.
        
        The city is famous for its craft brewery scene, with over 30 breweries offering tours and 
        tastings. Deschutes Brewery is the most famous, but don't miss Crux Fermentation Project.
        
        Health & Medical: No special vaccinations required. Tap water is safe to drink. Good medical 
        facilities available at St. Charles Medical Center. Main health concern is altitude sickness.
        
        Museums and cultural attractions include the High Desert Museum, showcasing regional wildlife 
        and Native American culture. The downtown area features local art galleries and studios.
        """
        
        self.sample_content_2 = """
        Outdoor Adventures in Bend: The city is a year-round playground for adventure seekers.
        
        Winter Sports: Mt. Bachelor offers world-class skiing and snowboarding just 30 minutes away.
        Cross-country skiing and snowshoeing are popular in the Cascade Mountains.
        
        Visa & Access: US destination, no visa required for domestic travelers. International visitors 
        need standard US visa. Direct flights available from Seattle, San Francisco, and Denver.
        English is the primary language. Infrastructure is excellent with rating of 4.5/5.
        
        Summer Activities: Rock climbing at Smith Rock State Park, mountain biking on numerous trails,
        and rafting on the Deschutes River. The weather is perfect from June to September.
        
        Family-Friendly: Bend is very family-oriented with many kid-friendly activities including
        the Sun Mountain Fun Center and numerous easy hiking trails suitable for children.
        
        Dining Scene: Farm-to-table restaurants dominate, with fresh local ingredients. Must-try
        restaurants include Ariana and Jackalope Grill. Food costs average $15-25 per meal.
        """
        
        self.sample_content_3 = """
        Practical Travel Guide for Bend, Oregon
        
        Weather Patterns: Best time to visit is June-September for warm, dry weather. Winter brings
        snow and cold temperatures, perfect for winter sports. Spring and fall are mild but variable.
        
        Transportation: Bend is car-dependent with limited public transit. Rental cars recommended.
        Parking is generally free and abundant. The city is becoming more bike-friendly.
        
        Neighborhoods: The Old Mill District offers shopping and dining along the Deschutes River.
        Downtown Bend features historic buildings, boutique shops, and the majority of restaurants.
        The Westside is more residential but has some hidden gem eateries.
        
        Hotels range from budget motels ($80/night) to luxury resorts ($400+/night). Popular options
        include The Oxford Hotel (boutique), McMenamins Old St. Francis School (unique), and 
        Tetherow Resort (luxury golf resort).
        
        Seasonal Considerations: High season (summer) sees 25% higher prices. Book accommodations
        early. Winter is busy at ski resorts but quieter in town, offering better deals.
        """
    
    @patch('src.tools.web_discovery_tools.JinaReaderTool._arun')
    @patch('src.core.web_discovery_logic.WebDiscoveryLogic._fetch_brave_search')
    async def test_full_workflow_with_priority_data(self, mock_brave_search, mock_jina):
        """Test the complete workflow including priority data extraction"""
        
        # Mock search results
        mock_brave_search.return_value = [
            {
                "url": "https://visitbend.com/guide",
                "title": "Complete Guide to Bend Oregon",
                "snippet": "Comprehensive travel guide with safety and cost information"
            },
            {
                "url": "https://traveloregon.com/bend",
                "title": "Bend Oregon Outdoor Adventures",
                "snippet": "Outdoor activities and adventure sports in Bend"
            },
            {
                "url": "https://bendoregon.gov/visit",
                "title": "Official Bend Travel Information",
                "snippet": "Official visitor information including practical details"
            }
        ]
        
        # Mock content fetching - return our sample content
        mock_jina.side_effect = [
            self.sample_content_1,
            self.sample_content_2,
            self.sample_content_3
        ]
        
        # Initialize the web discovery tool
        discovery_tool = DiscoverAndFetchContentTool(
            brave_api_key="test_key",
            config=self.config
        )
        
        # Step 1: Discover and fetch content
        page_contents = await discovery_tool._arun("Bend, Oregon")
        
        # Verify we got content
        self.assertEqual(len(page_contents), 3)
        self.assertIsInstance(page_contents[0], PageContent)
        
        # Step 2: Extract priority data from content
        extractor = PriorityDataExtractor()
        for page in page_contents:
            priority_data = extractor.extract_all_priority_data(
                page.content,
                page.url
            )
            page.priority_data = priority_data
        
        # Verify priority data was extracted
        self.assertIsNotNone(page_contents[0].priority_data)
        self.assertIn('safety', page_contents[0].priority_data)
        self.assertIn('cost', page_contents[0].priority_data)
        
        # Step 3: Run theme analysis with priority aggregation
        theme_tool = EnhancedAnalyzeThemesFromEvidenceTool()
        
        theme_result = await theme_tool._create_analysis_function(None, None)(
            destination_name="Bend, Oregon",
            country_code="US",
            text_content_list=page_contents,
            config=self.config
        )
        
        # Verify theme extraction worked
        self.assertIsInstance(theme_result, ThemeInsightOutput)
        all_themes = theme_result.validated_themes + theme_result.discovered_themes
        self.assertGreater(len(all_themes), 0)
        
        # Check for expected themes
        theme_names = [theme.insight_name for theme in all_themes]
        logging.info(f"Extracted themes: {theme_names}")
        
        # Should find outdoor/hiking themes
        outdoor_themes = [t for t in theme_names if any(
            keyword in t.lower() for keyword in ['hiking', 'outdoor', 'trail', 'mountain']
        )]
        self.assertGreater(len(outdoor_themes), 0, "Should find outdoor-related themes")
        
        # Should find brewery/dining themes  
        dining_themes = [t for t in theme_names if any(
            keyword in t.lower() for keyword in ['brewery', 'restaurant', 'dining', 'food']
        )]
        self.assertGreater(len(dining_themes), 0, "Should find dining-related themes")
        
        # Should find cultural themes
        cultural_themes = [t for t in theme_names if any(
            keyword in t.lower() for keyword in ['museum', 'culture', 'art', 'gallery']
        )]
        self.assertGreater(len(cultural_themes), 0, "Should find cultural themes")
        
        # Verify priority data was aggregated
        self.assertIsNotNone(theme_result.priority_metrics)
        self.assertIsInstance(theme_result.priority_metrics, PriorityMetrics)
        
        # Check specific priority metrics
        priority_metrics = theme_result.priority_metrics
        self.assertIsNotNone(priority_metrics.crime_index)
        self.assertLess(priority_metrics.crime_index, 35)  # Should extract ~28.5
        
        self.assertIsNotNone(priority_metrics.budget_per_day_low)
        self.assertGreater(priority_metrics.budget_per_day_low, 50)  # Should extract ~60-80
        
        # Verify priority insights were generated
        self.assertIsInstance(theme_result.priority_insights, list)
        if theme_result.priority_insights:
            insight_names = [i.insight_name for i in theme_result.priority_insights]
            logging.info(f"Priority insights: {insight_names}")
    
    async def test_priority_aggregation_multiple_sources(self):
        """Test priority data aggregation from multiple sources"""
        
        # Create page contents with different priority data
        pages = []
        for i in range(3):
            page = PageContent(
                url=f"https://source{i}.com",
                title=f"Source {i}",
                content="Content",
                content_length=100
            )
            page.priority_data = {
                "safety": {
                    "crime_index": 25.0 + i * 5,  # 25, 30, 35
                    "tourist_police_available": True,
                    "emergency_contacts": {"emergency": "911"}
                },
                "cost": {
                    "budget_per_day_low": 60.0 + i * 10,  # 60, 70, 80
                    "currency": "USD"
                },
                "source_credibility": 0.8,
                "temporal_relevance": 0.9,
                "source_url": page.url
            }
            pages.append(page)
        
        # Run aggregation
        aggregator = PriorityAggregationTool()
        result = aggregator._run(
            destination_name="Test Destination",
            page_contents=pages
        )
        
        # Verify aggregation
        metrics = result['priority_metrics']
        self.assertIsNotNone(metrics.crime_index)
        # Should be median of [25, 30, 35] = 30
        self.assertEqual(metrics.crime_index, 30.0)
        
        # Should be median of [60, 70, 80] = 70
        self.assertEqual(metrics.budget_per_day_low, 70.0)
        
        # Should have high confidence due to multiple sources
        self.assertGreater(result['aggregation_confidence'], 0.7)
    
    async def test_theme_confidence_with_priority_weight(self):
        """Test that priority sources affect theme confidence"""
        
        # Create content from high-priority source (government)
        gov_page = PageContent(
            url="https://state.gov/travel/bend",
            title="Government Travel Advisory",
            content="""
            Bend Oregon Travel Information
            
            Outdoor activities are the main attraction, with excellent hiking trails.
            The city has numerous craft breweries and restaurants.
            Cultural attractions include museums and art galleries.
            
            Safety: Very safe destination with low crime rates.
            Cost: Moderate costs, budget $100-200 per day.
            """,
            content_length=500
        )
        
        # Create content from lower-priority source
        blog_page = PageContent(
            url="https://travelblog.com/bend",
            title="Travel Blog",
            content="""
            My trip to Bend was amazing!
            
            Great hiking and outdoor adventures everywhere.
            The brewery scene is fantastic.
            Museums are worth visiting.
            """,
            content_length=200
        )
        
        # Add priority data
        extractor = PriorityDataExtractor()
        for page in [gov_page, blog_page]:
            page.priority_data = extractor.extract_all_priority_data(
                page.content, page.url
            )
        
        # Run theme analysis
        theme_tool = EnhancedAnalyzeThemesFromEvidenceTool()
        result = await theme_tool._create_analysis_function(None, None)(
            destination_name="Bend, Oregon",
            text_content_list=[gov_page, blog_page]
        )
        
        # Themes from government source should have higher confidence
        all_themes = result.validated_themes + result.discovered_themes
        for theme in all_themes:
            # Check if evidence is primarily from government source
            if theme.source_urls and 'state.gov' in theme.source_urls[0]:
                self.assertGreater(theme.confidence_score, 0.7)
    
    async def test_priority_query_generation(self):
        """Test that priority queries are generated correctly"""
        
        logic = WebDiscoveryLogic("test_key", self.config)
        
        # Generate priority queries
        queries = logic.generate_priority_focused_queries("Paris, France")
        
        # Verify all priority categories have queries
        self.assertIn("safety", queries)
        self.assertIn("cost", queries)
        self.assertIn("health", queries)
        self.assertIn("weather", queries)
        self.assertIn("accessibility", queries)
        
        # Verify queries are destination-specific
        for category, query_list in queries.items():
            for query in query_list:
                self.assertIn("Paris, France", query)
        
        # Check specific query patterns
        safety_queries = queries["safety"]
        self.assertTrue(any("crime" in q for q in safety_queries))
        self.assertTrue(any("tourist police" in q for q in safety_queries))
        
        cost_queries = queries["cost"]
        self.assertTrue(any("budget" in q for q in cost_queries))
        self.assertTrue(any("daily" in q for q in cost_queries))
    
    @patch('src.core.enhanced_database_manager.EnhancedDatabaseManager')
    async def test_enhanced_analyst_with_priorities(self, mock_db):
        """Test the enhanced CrewAI analyst with priority features"""
        
        # Mock tools
        mock_tools = []
        
        # Mock discovery tool
        mock_discovery = Mock()
        mock_discovery.name = "discover_and_fetch_web_content_for_destination"
        mock_discovery._arun = Mock(return_value=[
            PageContent(
                url="https://test.com",
                title="Test",
                content=self.sample_content_1,
                content_length=len(self.sample_content_1),
                priority_data={
                    "safety": {"crime_index": 28.5},
                    "cost": {"budget_per_day_low": 60.0}
                }
            )
        ])
        mock_tools.append(mock_discovery)
        
        # Mock other required tools
        for tool_name in [
            "process_content_with_vectorize",
            "add_processed_chunks_to_chromadb",
            "semantic_search_chromadb",
            "enhanced_destination_analysis",
            "analyze_themes_from_evidence",
            "store_destination_insights"
        ]:
            mock_tool = Mock()
            mock_tool.name = tool_name
            mock_tool._arun = Mock(return_value={
                "status": "success",
                "chunks": [],
                "validated_themes": [],
                "discovered_themes": [],
                "priority_metrics": PriorityMetrics(),
                "priority_insights": []
            })
            mock_tools.append(mock_tool)
        
        # Create analyst
        analyst = EnhancedCrewAIDestinationAnalyst(
            llm=Mock(),
            tools=mock_tools
        )
        
        # Run analysis
        result = await analyst.analyze_destination(
            "Bend, Oregon",
            self.config["processing_settings"]
        )
        
        # Verify result includes priority data
        self.assertIn("priority_insights", result)
        self.assertIn("priority_summary", result)
        
        # Verify workflow completed
        self.assertEqual(result["status"], "Success")


class TestPriorityDataQuality(unittest.TestCase):
    """Test data quality and edge cases for priority features"""
    
    def test_empty_content_handling(self):
        """Test handling of empty or invalid content"""
        extractor = PriorityDataExtractor()
        
        # Empty content
        result = extractor.extract_all_priority_data("", "https://test.com")
        self.assertIsNotNone(result)
        self.assertIn("safety", result)
        
        # No relevant data
        result = extractor.extract_all_priority_data(
            "This content has no travel information at all.",
            "https://test.com"
        )
        safety_data = result["safety"]
        self.assertIsNone(safety_data["crime_index"])
        self.assertIsNone(safety_data["safety_rating"])
    
    def test_conflicting_data_resolution(self):
        """Test resolution of conflicting priority data"""
        pages = [
            self._create_page_with_data({
                "safety": {"crime_index": 20.0},
                "cost": {"budget_per_day_low": 50.0}
            }, url="https://official.gov"),
            self._create_page_with_data({
                "safety": {"crime_index": 40.0},
                "cost": {"budget_per_day_low": 80.0}
            }, url="https://blog.com")
        ]
        
        aggregator = PriorityAggregationTool()
        # Lower threshold to include both sources (blog credibility 0.6 * temporal 0.8 = 0.48)
        result = aggregator._run("Test", pages, confidence_threshold=0.4)
        
        # Should use median for aggregation
        metrics = result['priority_metrics']
        self.assertEqual(metrics.crime_index, 30.0)  # Median of [20, 40]
        self.assertEqual(metrics.budget_per_day_low, 65.0)  # Median of [50, 80]
    
    def test_temporal_relevance_impact(self):
        """Test that old data is weighted less"""
        extractor = PriorityDataExtractor()
        
        # Current year content
        current_content = f"Updated in {datetime.now().year}: Crime rate is 25%"
        current_relevance = extractor.determine_temporal_relevance(current_content)
        
        # Old content
        old_content = "Updated in 2015: Crime rate is 25%"
        old_relevance = extractor.determine_temporal_relevance(old_content)
        
        self.assertGreater(current_relevance, old_relevance)
    
    def _create_page_with_data(self, priority_data, url="https://test.com"):
        """Helper to create PageContent with priority data"""
        page = PageContent(
            url=url,
            title="Test",
            content="Test content",
            content_length=100
        )
        extractor = PriorityDataExtractor()
        page.priority_data = {
            **priority_data,
            "source_credibility": extractor.calculate_source_credibility(url),
            "temporal_relevance": 0.8,
            "source_url": url
        }
        return page


if __name__ == "__main__":
    # Run async tests
    unittest.main() 