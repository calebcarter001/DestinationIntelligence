import unittest
from src.core.insight_classifier import InsightClassifier
from src.schemas import InsightType, LocationExclusivity
from src.core.enhanced_data_models import SeasonalWindow


class TestInsightClassifier(unittest.TestCase):
    
    def setUp(self):
        self.classifier = InsightClassifier()
    
    def test_classify_insight_type_seasonal(self):
        seasonal_content = [
            "Maple syrup season runs from February to April",
            "Best time to visit is during fall foliage season",
            "Peak season for whale watching is June through September",
            "Winter hours are reduced from December to March"
        ]
        
        for content in seasonal_content:
            result = self.classifier.classify_insight_type(content)
            self.assertEqual(result, InsightType.SEASONAL)
    
    def test_classify_insight_type_specialty(self):
        specialty_content = [
            "Famous for their craft beer and local brewery scene",
            "Known for artisan cheese production and local dairy farms",
            "Signature dish includes fresh caught salmon",
            "Local specialty coffee roasted in-house"
        ]
        
        for content in specialty_content:
            result = self.classifier.classify_insight_type(content)
            self.assertEqual(result, InsightType.SPECIALTY)
    
    def test_classify_insight_type_insider(self):
        insider_content = [
            "Locals know to avoid the main tourist area",
            "Hidden gem that only residents frequent",
            "Secret spot known only to longtime locals",
            "Local tip: visit early morning to avoid crowds"
        ]
        
        for content in insider_content:
            result = self.classifier.classify_insight_type(content)
            self.assertEqual(result, InsightType.INSIDER)
    
    def test_classify_insight_type_cultural(self):
        cultural_content = [
            "Traditional festival celebrating local heritage",
            "Historic district with indigenous cultural sites",
            "Museum showcasing regional art and culture",
            "Cultural center preserving traditional practices"
        ]
        
        for content in cultural_content:
            result = self.classifier.classify_insight_type(content)
            self.assertEqual(result, InsightType.CULTURAL)
    
    def test_classify_insight_type_practical(self):
        practical_content = [
            "Located at 123 Main Street, open daily 9-5",
            "Free parking available behind the building",
            "Reservations required, call (555) 123-4567",
            "How to get there: take the bus line 42"
        ]
        
        for content in practical_content:
            result = self.classifier.classify_insight_type(content)
            self.assertEqual(result, InsightType.PRACTICAL)
    
    def test_classify_insight_type_default(self):
        generic_content = "This is a nice place to visit."
        result = self.classifier.classify_insight_type(generic_content)
        self.assertEqual(result, InsightType.PRACTICAL)  # Default
    
    def test_determine_location_exclusivity_exclusive(self):
        exclusive_content = [
            "The only place in the world where you can see this",
            "Unique to this region, found nowhere else",
            "Exclusively available in this location",
            "Cannot be found anywhere else on earth"
        ]
        
        for content in exclusive_content:
            result = self.classifier.determine_location_exclusivity(content)
            self.assertEqual(result, LocationExclusivity.EXCLUSIVE)
    
    def test_determine_location_exclusivity_signature(self):
        signature_content = [
            "Famous for being the best in the region",
            "Known as the premier destination for this activity",
            "Renowned worldwide for this specialty",
            "Best known attraction in the area"
        ]
        
        for content in signature_content:
            result = self.classifier.determine_location_exclusivity(content)
            self.assertEqual(result, LocationExclusivity.SIGNATURE)
    
    def test_determine_location_exclusivity_regional(self):
        regional_content = [
            "Common throughout the Pacific Northwest",
            "Found across the entire region",
            "Popular throughout the state",
            "Regional specialty available in several cities"
        ]
        
        for content in regional_content:
            result = self.classifier.determine_location_exclusivity(content)
            self.assertEqual(result, LocationExclusivity.REGIONAL)
    
    def test_determine_location_exclusivity_common(self):
        common_content = [
            "Available everywhere and commonly found",
            "Standard tourist attraction found in most cities",
            "Common activity available worldwide",
            "Typical restaurant chain with multiple locations"
        ]
        
        for content in common_content:
            result = self.classifier.determine_location_exclusivity(content)
            self.assertEqual(result, LocationExclusivity.COMMON)
    
    def test_determine_location_exclusivity_default(self):
        generic_content = "This is a place."
        result = self.classifier.determine_location_exclusivity(generic_content)
        self.assertEqual(result, LocationExclusivity.REGIONAL)  # Default
    
    def test_extract_seasonal_window_basic(self):
        content = "Open from March to November, peak season July-August"
        window = self.classifier.extract_seasonal_window(content)
        
        self.assertIsInstance(window, SeasonalWindow)
        self.assertEqual(window.start_month, 3)  # March
        self.assertEqual(window.end_month, 11)   # November
    
    def test_extract_seasonal_window_with_lead_time(self):
        content = "Maple season February to April, book 2 weeks in advance"
        window = self.classifier.extract_seasonal_window(content)
        
        self.assertIsInstance(window, SeasonalWindow)
        self.assertEqual(window.start_month, 2)  # February
        self.assertEqual(window.end_month, 4)    # April
        self.assertEqual(window.booking_lead_time, "2 weeks")
    
    def test_extract_seasonal_window_no_match(self):
        content = "Great place to visit anytime"
        window = self.classifier.extract_seasonal_window(content)
        
        self.assertIsNone(window)
    
    def test_extract_actionable_details_complete(self):
        content = """
        Located at 456 Oak Street in downtown. 
        Open Tuesday through Sunday from 10am to 6pm.
        Call (555) 987-6543 for reservations.
        Best time to visit is early morning or late afternoon.
        """
        
        details = self.classifier.extract_actionable_details(content)
        
        self.assertIsInstance(details, dict)
        self.assertIn("where", details)
        self.assertIn("when", details)
        self.assertIn("how", details)
        
        # Check specific extracted information
        self.assertIn("456 oak street", details["where"].lower())
        self.assertIn("10am to 6pm", details["when"].lower())
    
    def test_extract_actionable_details_partial(self):
        content = "Great restaurant, call ahead for reservations."
        
        details = self.classifier.extract_actionable_details(content)
        
        self.assertIsInstance(details, dict)
        # Should have some details but not all
        self.assertIn("how", details)
        self.assertIn("call", details["how"].lower())
    
    def test_extract_actionable_details_empty(self):
        content = "Nice place."
        
        details = self.classifier.extract_actionable_details(content)
        
        self.assertIsInstance(details, dict)
        # Should be mostly empty
        self.assertTrue(len(details) == 0 or all(not v for v in details.values()))
    
    def test_month_name_to_number(self):
        # Test the helper function if it's accessible
        # (This might need to be adjusted based on implementation)
        month_mapping = {
            "january": 1, "february": 2, "march": 3,
            "april": 4, "may": 5, "june": 6,
            "july": 7, "august": 8, "september": 9,
            "october": 10, "november": 11, "december": 12
        }
        
        for month_name, expected_num in month_mapping.items():
            # Test case variations
            for variation in [month_name, month_name.capitalize(), month_name[:3]]:
                # This test assumes the classifier has some way to convert months
                # If not directly accessible, we can test through extract_seasonal_window
                content = f"Open from {variation} to December"
                window = self.classifier.extract_seasonal_window(content)
                if window:
                    self.assertEqual(window.start_month, expected_num)


if __name__ == '__main__':
    unittest.main() 