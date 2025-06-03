import unittest
from src.core.destination_classifier import DestinationClassifier, DestinationType, SourceStrategy


class TestDestinationClassifier(unittest.TestCase):
    
    def setUp(self):
        self.classifier = DestinationClassifier()
    
    def test_classify_destination_type_global_hub(self):
        global_hubs = [
            {"names": ["Paris"], "admin_levels": {"country": "France"}, "population": 2200000},
            {"names": ["Tokyo"], "admin_levels": {"country": "Japan"}, "population": 14000000},
            {"names": ["New York City"], "admin_levels": {"country": "United States"}, "population": 8400000},
            {"names": ["London"], "admin_levels": {"country": "United Kingdom"}, "population": 9000000}
        ]
        
        for dest_data in global_hubs:
            result = self.classifier.classify_destination_type(dest_data)
            self.assertEqual(result, DestinationType.GLOBAL_HUB)
    
    def test_classify_destination_type_business_hub(self):
        business_hubs = [
            {"names": ["Frankfurt"], "admin_levels": {"country": "Germany"}, "population": 750000},
            {"names": ["Gurgaon"], "admin_levels": {"country": "India"}, "population": 1200000},
            {"names": ["Charlotte"], "admin_levels": {"country": "United States"}, "population": 900000}
        ]
        
        for dest_data in business_hubs:
            result = self.classifier.classify_destination_type(dest_data)
            # Business hubs might be classified as BUSINESS_HUB or REGIONAL depending on implementation
            self.assertIn(result, [DestinationType.BUSINESS_HUB, DestinationType.REGIONAL])
    
    def test_classify_destination_type_regional(self):
        regional_destinations = [
            {"names": ["Bend"], "admin_levels": {"country": "United States", "state": "Oregon"}, "population": 100000},
            {"names": ["Kyoto"], "admin_levels": {"country": "Japan"}, "population": 1500000},
            {"names": ["Charleston"], "admin_levels": {"country": "United States", "state": "South Carolina"}, "population": 150000}
        ]
        
        for dest_data in regional_destinations:
            result = self.classifier.classify_destination_type(dest_data)
            self.assertEqual(result, DestinationType.REGIONAL)
    
    def test_classify_destination_type_remote_getaway(self):
        remote_destinations = [
            {"names": ["Bhutan"], "admin_levels": {"country": "Bhutan"}, "population": 50000},
            {"names": ["Patagonia"], "admin_levels": {"country": "Chile"}, "population": 10000},
            {"names": ["Faroe Islands"], "admin_levels": {"country": "Denmark"}, "population": 30000}
        ]
        
        for dest_data in remote_destinations:
            result = self.classifier.classify_destination_type(dest_data)
            self.assertEqual(result, DestinationType.REMOTE_GETAWAY)
    
    def test_get_scoring_weights_global_hub(self):
        weights = self.classifier.get_scoring_weights(DestinationType.GLOBAL_HUB)
        
        self.assertIsInstance(weights, dict)
        self.assertIn("authenticity", weights)
        self.assertIn("uniqueness", weights)
        self.assertIn("actionability", weights)
        self.assertIn("temporal_relevance", weights)
        
        # For global hubs, authenticity should be weighted higher
        self.assertGreaterEqual(weights["authenticity"], 0.3)
        
        # All weights should sum to 1.0
        total_weight = sum(weights.values())
        self.assertAlmostEqual(total_weight, 1.0, places=2)
    
    def test_get_scoring_weights_regional(self):
        weights = self.classifier.get_scoring_weights(DestinationType.REGIONAL)
        
        self.assertIsInstance(weights, dict)
        
        # For regional destinations, uniqueness might be weighted higher
        self.assertGreaterEqual(weights["uniqueness"], 0.2)
        
        # All weights should sum to 1.0
        total_weight = sum(weights.values())
        self.assertAlmostEqual(total_weight, 1.0, places=2)
    
    def test_get_scoring_weights_remote_getaway(self):
        weights = self.classifier.get_scoring_weights(DestinationType.REMOTE_GETAWAY)
        
        self.assertIsInstance(weights, dict)
        
        # For remote getaways, uniqueness and authenticity should be emphasized
        self.assertGreaterEqual(weights["uniqueness"], 0.25)
        self.assertGreaterEqual(weights["authenticity"], 0.25)
        
        # All weights should sum to 1.0
        total_weight = sum(weights.values())
        self.assertAlmostEqual(total_weight, 1.0, places=2)
    
    def test_get_scoring_weights_business_hub(self):
        weights = self.classifier.get_scoring_weights(DestinationType.BUSINESS_HUB)
        
        self.assertIsInstance(weights, dict)
        
        # For business hubs, actionability should be weighted higher
        self.assertGreaterEqual(weights["actionability"], 0.3)
        
        # All weights should sum to 1.0
        total_weight = sum(weights.values())
        self.assertAlmostEqual(total_weight, 1.0, places=2)
    
    def test_get_source_strategy_global_hub(self):
        strategy = self.classifier.get_source_strategy(DestinationType.GLOBAL_HUB)
        self.assertEqual(strategy, SourceStrategy.FILTER_QUALITY_FROM_ABUNDANCE)
    
    def test_get_source_strategy_regional(self):
        strategy = self.classifier.get_source_strategy(DestinationType.REGIONAL)
        self.assertEqual(strategy, SourceStrategy.COMPREHENSIVE_LOCAL_SELECTIVE_NATIONAL)
    
    def test_get_source_strategy_business_hub(self):
        strategy = self.classifier.get_source_strategy(DestinationType.BUSINESS_HUB)
        self.assertEqual(strategy, SourceStrategy.BUSINESS_FOCUSED_PRACTICAL)
    
    def test_get_source_strategy_remote_getaway(self):
        strategy = self.classifier.get_source_strategy(DestinationType.REMOTE_GETAWAY)
        self.assertEqual(strategy, SourceStrategy.ULTRA_LOCAL_NICHE_EXPERT)
    
    def test_destination_type_enum_values(self):
        # Test that all enum values are properly defined
        self.assertEqual(DestinationType.GLOBAL_HUB.value, "global_hub")
        self.assertEqual(DestinationType.REGIONAL.value, "regional")
        self.assertEqual(DestinationType.BUSINESS_HUB.value, "business_hub")
        self.assertEqual(DestinationType.REMOTE_GETAWAY.value, "remote_getaway")
    
    def test_source_strategy_enum_values(self):
        # Test that all source strategy enum values are properly defined
        self.assertEqual(SourceStrategy.FILTER_QUALITY_FROM_ABUNDANCE.value, "filter_quality_from_abundance")
        self.assertEqual(SourceStrategy.COMPREHENSIVE_LOCAL_SELECTIVE_NATIONAL.value, "comprehensive_local_selective_national")
        self.assertEqual(SourceStrategy.BUSINESS_FOCUSED_PRACTICAL.value, "business_focused_practical")
        self.assertEqual(SourceStrategy.ULTRA_LOCAL_NICHE_EXPERT.value, "ultra_local_niche_expert")
    
    def test_classify_destination_edge_cases(self):
        # Test edge cases and invalid inputs
        
        # Empty destination data
        empty_data = {"names": [], "admin_levels": {}}
        result = self.classifier.classify_destination_type(empty_data)
        self.assertIn(result, list(DestinationType))  # Should return some valid type
        
        # Missing population data
        no_pop_data = {"names": ["Test City"], "admin_levels": {"country": "Test Country"}}
        result = self.classifier.classify_destination_type(no_pop_data)
        self.assertIn(result, list(DestinationType))  # Should handle gracefully
    
    def test_all_destination_types_have_weights(self):
        # Ensure all destination types have scoring weight configurations
        for dest_type in DestinationType:
            weights = self.classifier.get_scoring_weights(dest_type)
            self.assertIsInstance(weights, dict)
            self.assertGreater(len(weights), 0)
    
    def test_all_destination_types_have_strategies(self):
        # Ensure all destination types have source strategies
        for dest_type in DestinationType:
            strategy = self.classifier.get_source_strategy(dest_type)
            self.assertIsInstance(strategy, SourceStrategy)


if __name__ == '__main__':
    unittest.main() 