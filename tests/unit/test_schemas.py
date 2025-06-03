import unittest
from src.schemas import InsightType, AuthorityType, LocationExclusivity


class TestInsightType(unittest.TestCase):
    
    def test_insight_type_values(self):
        self.assertEqual(InsightType.SEASONAL.value, "seasonal")
        self.assertEqual(InsightType.SPECIALTY.value, "specialty")
        self.assertEqual(InsightType.INSIDER.value, "insider")
        self.assertEqual(InsightType.CULTURAL.value, "cultural")
        self.assertEqual(InsightType.PRACTICAL.value, "practical")
    
    def test_insight_type_from_value(self):
        self.assertEqual(InsightType("seasonal"), InsightType.SEASONAL)
        self.assertEqual(InsightType("specialty"), InsightType.SPECIALTY)
        self.assertEqual(InsightType("insider"), InsightType.INSIDER)
        self.assertEqual(InsightType("cultural"), InsightType.CULTURAL)
        self.assertEqual(InsightType("practical"), InsightType.PRACTICAL)
    
    def test_insight_type_count(self):
        # Ensure we have exactly 5 insight types
        insight_types = list(InsightType)
        self.assertEqual(len(insight_types), 5)


class TestAuthorityType(unittest.TestCase):
    
    def test_authority_type_values(self):
        self.assertEqual(AuthorityType.PRODUCER.value, "producer")
        self.assertEqual(AuthorityType.RESIDENT.value, "long_term_resident")
        self.assertEqual(AuthorityType.PROFESSIONAL.value, "industry_professional")
        self.assertEqual(AuthorityType.CULTURAL.value, "cultural_institution")
        self.assertEqual(AuthorityType.SEASONAL_WORKER.value, "seasonal_worker")
    
    def test_authority_type_from_value(self):
        self.assertEqual(AuthorityType("producer"), AuthorityType.PRODUCER)
        self.assertEqual(AuthorityType("long_term_resident"), AuthorityType.RESIDENT)
        self.assertEqual(AuthorityType("industry_professional"), AuthorityType.PROFESSIONAL)
        self.assertEqual(AuthorityType("cultural_institution"), AuthorityType.CULTURAL)
        self.assertEqual(AuthorityType("seasonal_worker"), AuthorityType.SEASONAL_WORKER)
    
    def test_authority_type_count(self):
        # Ensure we have exactly 5 authority types
        authority_types = list(AuthorityType)
        self.assertEqual(len(authority_types), 5)


class TestLocationExclusivity(unittest.TestCase):
    
    def test_location_exclusivity_values(self):
        self.assertEqual(LocationExclusivity.EXCLUSIVE.value, "exclusive")
        self.assertEqual(LocationExclusivity.SIGNATURE.value, "signature")
        self.assertEqual(LocationExclusivity.REGIONAL.value, "regional")
        self.assertEqual(LocationExclusivity.COMMON.value, "common")
    
    def test_location_exclusivity_from_value(self):
        self.assertEqual(LocationExclusivity("exclusive"), LocationExclusivity.EXCLUSIVE)
        self.assertEqual(LocationExclusivity("signature"), LocationExclusivity.SIGNATURE)
        self.assertEqual(LocationExclusivity("regional"), LocationExclusivity.REGIONAL)
        self.assertEqual(LocationExclusivity("common"), LocationExclusivity.COMMON)
    
    def test_location_exclusivity_count(self):
        # Ensure we have exactly 4 exclusivity levels
        exclusivity_levels = list(LocationExclusivity)
        self.assertEqual(len(exclusivity_levels), 4)
    
    def test_exclusivity_hierarchy(self):
        # Test that we can compare exclusivity levels conceptually
        # (though not implemented as comparable, we test their existence)
        levels = [LocationExclusivity.EXCLUSIVE, LocationExclusivity.SIGNATURE, 
                 LocationExclusivity.REGIONAL, LocationExclusivity.COMMON]
        self.assertEqual(len(levels), 4)
        
        # Test that all expected values are unique
        values = [level.value for level in levels]
        self.assertEqual(len(values), len(set(values)))


if __name__ == '__main__':
    unittest.main() 