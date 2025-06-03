import unittest
from datetime import datetime
from src.core.seasonal_intelligence import SeasonalIntelligence
from src.core.enhanced_data_models import SeasonalWindow


class TestSeasonalIntelligence(unittest.TestCase):
    
    def setUp(self):
        self.seasonal_intel = SeasonalIntelligence()
    
    def test_extract_seasonal_patterns_maple_season(self):
        content = [
            "Maple syrup season runs from late February through early April",
            "Peak sap flow occurs during March when temperatures fluctuate above and below freezing",
            "Best time to visit sugar houses is weekends in March"
        ]
        
        patterns = self.seasonal_intel.extract_seasonal_patterns(content)
        
        self.assertIsInstance(patterns, list)
        self.assertGreater(len(patterns), 0)
        
        # Should identify seasonal windows
        for pattern in patterns:
            self.assertIsInstance(pattern, dict)
            if "seasonal_window" in pattern:
                window = pattern["seasonal_window"]
                self.assertIsInstance(window, SeasonalWindow)
    
    def test_extract_seasonal_patterns_fall_foliage(self):
        content = [
            "Fall foliage season typically runs from late September to mid-October",
            "Peak colors usually occur during the first two weeks of October",
            "Book accommodations 2-3 months in advance for foliage season"
        ]
        
        patterns = self.seasonal_intel.extract_seasonal_patterns(content)
        
        self.assertIsInstance(patterns, list)
        self.assertGreater(len(patterns), 0)
        
        # Should detect fall/autumn patterns
        pattern_text = str(patterns).lower()
        self.assertTrue(any(keyword in pattern_text for keyword in ["fall", "autumn", "foliage", "october"]))
    
    def test_extract_seasonal_patterns_no_seasonal_content(self):
        content = [
            "The restaurant serves excellent food year-round",
            "Great destination for any time of year",
            "Always open with consistent quality"
        ]
        
        patterns = self.seasonal_intel.extract_seasonal_patterns(content)
        
        # Should return empty list or minimal patterns for non-seasonal content
        self.assertIsInstance(patterns, list)
        # May return empty or very few patterns
    
    def test_calculate_current_relevance_in_season(self):
        # Create a seasonal window for current month
        current_month = datetime.now().month
        
        window = SeasonalWindow(
            start_month=current_month,
            end_month=(current_month + 1) % 12 if current_month != 12 else 1,
            peak_weeks=[],
            booking_lead_time=None,
            specific_dates=None
        )
        
        relevance = self.seasonal_intel.calculate_current_relevance(window)
        
        self.assertIsInstance(relevance, float)
        self.assertGreaterEqual(relevance, 0.0)
        self.assertLessEqual(relevance, 1.0)
        self.assertGreater(relevance, 0.5)  # Should be high since we're in season
    
    def test_calculate_current_relevance_out_of_season(self):
        # Create a seasonal window for a different month
        current_month = datetime.now().month
        out_of_season_month = (current_month + 6) % 12
        if out_of_season_month == 0:
            out_of_season_month = 12
        
        window = SeasonalWindow(
            start_month=out_of_season_month,
            end_month=out_of_season_month,
            peak_weeks=[],
            booking_lead_time=None,
            specific_dates=None
        )
        
        relevance = self.seasonal_intel.calculate_current_relevance(window)
        
        self.assertIsInstance(relevance, float)
        self.assertGreaterEqual(relevance, 0.0)
        self.assertLessEqual(relevance, 1.0)
        self.assertLess(relevance, 0.5)  # Should be low since we're out of season
    
    def test_calculate_current_relevance_none_window(self):
        relevance = self.seasonal_intel.calculate_current_relevance(None)
        
        # Should return some default value for no seasonal window
        self.assertIsInstance(relevance, float)
        self.assertGreaterEqual(relevance, 0.0)
        self.assertLessEqual(relevance, 1.0)
    
    def test_generate_timing_recommendations_in_season(self):
        current_month = datetime.now().month
        
        window = SeasonalWindow(
            start_month=current_month,
            end_month=(current_month + 2) % 12 if current_month <= 10 else (current_month + 2) - 12,
            peak_weeks=[],
            booking_lead_time="2 weeks",
            specific_dates=None
        )
        
        recommendations = self.seasonal_intel.generate_timing_recommendations(window, "maple syrup tours")
        
        self.assertIsInstance(recommendations, list)
        self.assertGreater(len(recommendations), 0)
        
        # Should include current timing advice
        rec_text = " ".join(recommendations).lower()
        self.assertTrue(any(keyword in rec_text for keyword in ["now", "current", "season", "visit"]))
    
    def test_generate_timing_recommendations_upcoming_season(self):
        current_month = datetime.now().month
        future_month = (current_month + 3) % 12
        if future_month == 0:
            future_month = 12
        
        window = SeasonalWindow(
            start_month=future_month,
            end_month=(future_month + 1) % 12 if future_month != 12 else 1,
            peak_weeks=[],
            booking_lead_time="1 month",
            specific_dates=None
        )
        
        recommendations = self.seasonal_intel.generate_timing_recommendations(window, "festival")
        
        self.assertIsInstance(recommendations, list)
        self.assertGreater(len(recommendations), 0)
        
        # Should include future planning advice
        rec_text = " ".join(recommendations).lower()
        self.assertTrue(any(keyword in rec_text for keyword in ["plan", "advance", "book", "upcoming"]))
    
    def test_generate_timing_recommendations_no_window(self):
        recommendations = self.seasonal_intel.generate_timing_recommendations(None, "general activity")
        
        self.assertIsInstance(recommendations, list)
        # Should provide general timing advice even without seasonal window
        if recommendations:
            rec_text = " ".join(recommendations).lower()
            self.assertTrue(any(keyword in rec_text for keyword in ["visit", "time", "available"]))
    
    def test_month_name_helper_function(self):
        # Test the month_name helper function if accessible
        test_months = [
            ("january", 1), ("february", 2), ("march", 3), ("april", 4),
            ("may", 5), ("june", 6), ("july", 7), ("august", 8),
            ("september", 9), ("october", 10), ("november", 11), ("december", 12)
        ]
        
        for month_name, expected_num in test_months:
            # Test through seasonal pattern extraction since month_name might be internal
            content = [f"Season runs from {month_name} to {month_name}"]
            patterns = self.seasonal_intel.extract_seasonal_patterns(content)
            
            # If patterns are found, check if month conversion worked
            if patterns and any("seasonal_window" in p for p in patterns):
                for pattern in patterns:
                    if "seasonal_window" in pattern:
                        window = pattern["seasonal_window"]
                        if window.start_month:
                            self.assertGreaterEqual(window.start_month, 1)
                            self.assertLessEqual(window.start_month, 12)
    
    def test_seasonal_intelligence_comprehensive_workflow(self):
        # Test a complete workflow
        content = [
            "Vermont maple syrup season runs from late February through April",
            "Peak production occurs in March when temperatures fluctuate",
            "Sugar houses offer tours on weekends during maple season",
            "Book tours 2-3 weeks in advance during peak season"
        ]
        
        # Extract patterns
        patterns = self.seasonal_intel.extract_seasonal_patterns(content)
        self.assertGreater(len(patterns), 0)
        
        # For each pattern with a seasonal window
        for pattern in patterns:
            if "seasonal_window" in pattern:
                window = pattern["seasonal_window"]
                
                # Calculate current relevance
                relevance = self.seasonal_intel.calculate_current_relevance(window)
                self.assertIsInstance(relevance, float)
                
                # Generate recommendations
                recommendations = self.seasonal_intel.generate_timing_recommendations(window, "maple syrup tours")
                self.assertIsInstance(recommendations, list)
    
    def test_edge_cases_and_error_handling(self):
        # Empty content
        patterns = self.seasonal_intel.extract_seasonal_patterns([])
        self.assertIsInstance(patterns, list)
        
        # None content
        patterns = self.seasonal_intel.extract_seasonal_patterns(None)
        self.assertIsInstance(patterns, list)
        
        # Invalid seasonal window
        invalid_window = SeasonalWindow(
            start_month=15,  # Invalid month
            end_month=25,    # Invalid month
            peak_weeks=[],
            booking_lead_time=None,
            specific_dates=None
        )
        
        # Should handle gracefully
        relevance = self.seasonal_intel.calculate_current_relevance(invalid_window)
        self.assertIsInstance(relevance, float)
        self.assertGreaterEqual(relevance, 0.0)
        self.assertLessEqual(relevance, 1.0)


if __name__ == '__main__':
    unittest.main() 