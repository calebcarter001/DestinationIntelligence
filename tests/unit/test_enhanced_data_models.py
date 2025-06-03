import unittest
from datetime import datetime, date
from src.core.enhanced_data_models import (
    AuthenticInsight, SeasonalWindow, LocalAuthority, Theme, Evidence, 
    TemporalSlice, DimensionValue, Destination, PointOfInterest
)
from src.schemas import InsightType, AuthorityType, LocationExclusivity
from src.core.evidence_hierarchy import SourceCategory, EvidenceType


class TestAuthenticInsight(unittest.TestCase):
    
    def setUp(self):
        self.seasonal_window = SeasonalWindow(
            start_month=6, end_month=8, peak_weeks=[26, 27], 
            booking_lead_time="2 weeks", specific_dates=["07/04"]
        )
        
        self.authentic_insight = AuthenticInsight(
            insight_type=InsightType.SEASONAL,
            authenticity_score=0.9,
            uniqueness_score=0.8,
            actionability_score=0.7,
            temporal_relevance=0.85,
            location_exclusivity=LocationExclusivity.SIGNATURE,
            seasonal_window=self.seasonal_window,
            local_validation_count=5
        )
    
    def test_authentic_insight_creation(self):
        self.assertEqual(self.authentic_insight.insight_type, InsightType.SEASONAL)
        self.assertEqual(self.authentic_insight.authenticity_score, 0.9)
        self.assertEqual(self.authentic_insight.uniqueness_score, 0.8)
        self.assertEqual(self.authentic_insight.actionability_score, 0.7)
        self.assertEqual(self.authentic_insight.temporal_relevance, 0.85)
        self.assertEqual(self.authentic_insight.location_exclusivity, LocationExclusivity.SIGNATURE)
        self.assertEqual(self.authentic_insight.local_validation_count, 5)
        self.assertIsInstance(self.authentic_insight.seasonal_window, SeasonalWindow)
    
    def test_authentic_insight_to_dict(self):
        result = self.authentic_insight.to_dict()
        
        self.assertEqual(result["insight_type"], "seasonal")
        self.assertEqual(result["authenticity_score"], 0.9)
        self.assertEqual(result["uniqueness_score"], 0.8)
        self.assertEqual(result["actionability_score"], 0.7)
        self.assertEqual(result["temporal_relevance"], 0.85)
        self.assertEqual(result["location_exclusivity"], "signature")
        self.assertEqual(result["local_validation_count"], 5)
        
        # Check seasonal window is properly serialized
        self.assertIsInstance(result["seasonal_window"], dict)
        self.assertEqual(result["seasonal_window"]["start_month"], 6)
        self.assertEqual(result["seasonal_window"]["end_month"], 8)


class TestSeasonalWindow(unittest.TestCase):
    
    def test_seasonal_window_creation(self):
        window = SeasonalWindow(
            start_month=3, end_month=5, peak_weeks=[12, 13, 14],
            booking_lead_time="1 month", specific_dates=["03/21", "04/15"]
        )
        
        self.assertEqual(window.start_month, 3)
        self.assertEqual(window.end_month, 5)
        self.assertEqual(window.peak_weeks, [12, 13, 14])
        self.assertEqual(window.booking_lead_time, "1 month")
        self.assertEqual(window.specific_dates, ["03/21", "04/15"])
    
    def test_seasonal_window_to_dict(self):
        window = SeasonalWindow(
            start_month=12, end_month=2, peak_weeks=[52, 1],
            booking_lead_time=None, specific_dates=None
        )
        
        result = window.to_dict()
        self.assertEqual(result["start_month"], 12)
        self.assertEqual(result["end_month"], 2)
        self.assertEqual(result["peak_weeks"], [52, 1])
        self.assertIsNone(result["booking_lead_time"])
        self.assertIsNone(result["specific_dates"])


class TestLocalAuthority(unittest.TestCase):
    
    def test_local_authority_creation(self):
        authority = LocalAuthority(
            authority_type=AuthorityType.PRODUCER,
            local_tenure=10,
            expertise_domain="Maple syrup production",
            community_validation=0.95
        )
        
        self.assertEqual(authority.authority_type, AuthorityType.PRODUCER)
        self.assertEqual(authority.local_tenure, 10)
        self.assertEqual(authority.expertise_domain, "Maple syrup production")
        self.assertEqual(authority.community_validation, 0.95)
    
    def test_local_authority_to_dict(self):
        authority = LocalAuthority(
            authority_type=AuthorityType.RESIDENT,
            local_tenure=None,
            expertise_domain="Local events",
            community_validation=0.7
        )
        
        result = authority.to_dict()
        self.assertEqual(result["authority_type"], "long_term_resident")
        self.assertIsNone(result["local_tenure"])
        self.assertEqual(result["expertise_domain"], "Local events")
        self.assertEqual(result["community_validation"], 0.7)


class TestEnhancedTheme(unittest.TestCase):
    
    def setUp(self):
        self.evidence = Evidence(
            id="test_evidence_1",
            source_url="https://example.com",
            source_category=SourceCategory.BLOG,
            evidence_type=EvidenceType.TERTIARY,
            authority_weight=0.5,
            text_snippet="Test evidence snippet",
            timestamp=datetime.now(),
            confidence=0.7
        )
        
        self.authentic_insight = AuthenticInsight(
            insight_type=InsightType.CULTURAL,
            authenticity_score=0.8,
            uniqueness_score=0.6,
            actionability_score=0.7,
            temporal_relevance=0.5,
            location_exclusivity=LocationExclusivity.REGIONAL,
            seasonal_window=None,
            local_validation_count=3
        )
        
        self.local_authority = LocalAuthority(
            authority_type=AuthorityType.PROFESSIONAL,
            local_tenure=5,
            expertise_domain="Tour guiding",
            community_validation=0.8
        )
    
    def test_enhanced_theme_creation(self):
        theme = Theme(
            theme_id="test_theme_1",
            macro_category="Cultural & Arts",
            micro_category="Museums & Galleries",
            name="Local Art Scene",
            description="Vibrant local art galleries and studios",
            fit_score=0.9,
            evidence=[self.evidence],
            authentic_insights=[self.authentic_insight],
            local_authorities=[self.local_authority],
            seasonal_relevance={"summer": 0.8, "winter": 0.6},
            regional_uniqueness=0.7,
            insider_tips=["Visit during first Friday art walks"]
        )
        
        self.assertEqual(theme.theme_id, "test_theme_1")
        self.assertEqual(theme.macro_category, "Cultural & Arts")
        self.assertEqual(theme.micro_category, "Museums & Galleries")
        self.assertEqual(theme.name, "Local Art Scene")
        self.assertEqual(len(theme.authentic_insights), 1)
        self.assertEqual(len(theme.local_authorities), 1)
        self.assertEqual(theme.seasonal_relevance["summer"], 0.8)
        self.assertEqual(theme.regional_uniqueness, 0.7)
        self.assertEqual(len(theme.insider_tips), 1)
    
    def test_enhanced_theme_to_dict(self):
        theme = Theme(
            theme_id="test_theme_2",
            macro_category="Food & Dining",
            micro_category="Local Specialties",
            name="Craft Beer Scene",
            description="Local breweries and craft beer culture",
            fit_score=0.85,
            evidence=[],
            authentic_insights=[self.authentic_insight],
            local_authorities=[self.local_authority],
            seasonal_relevance={"fall": 0.9},
            regional_uniqueness=0.8,
            insider_tips=["Try the seasonal harvest ales"]
        )
        
        result = theme.to_dict()
        
        self.assertEqual(result["theme_id"], "test_theme_2")
        self.assertEqual(result["macro_category"], "Food & Dining")
        self.assertEqual(result["micro_category"], "Local Specialties")
        self.assertEqual(result["name"], "Craft Beer Scene")
        self.assertEqual(result["fit_score"], 0.85)
        self.assertEqual(len(result["authentic_insights"]), 1)
        self.assertEqual(len(result["local_authorities"]), 1)
        self.assertEqual(result["seasonal_relevance"]["fall"], 0.9)
        self.assertEqual(result["regional_uniqueness"], 0.8)
        self.assertEqual(len(result["insider_tips"]), 1)


class TestEnhancedDestination(unittest.TestCase):
    
    def setUp(self):
        self.authentic_insight = AuthenticInsight(
            insight_type=InsightType.INSIDER,
            authenticity_score=0.9,
            uniqueness_score=0.8,
            actionability_score=0.7,
            temporal_relevance=0.6,
            location_exclusivity=LocationExclusivity.EXCLUSIVE,
            seasonal_window=None,
            local_validation_count=2
        )
        
        self.local_authority = LocalAuthority(
            authority_type=AuthorityType.RESIDENT,
            local_tenure=15,
            expertise_domain="Local history",
            community_validation=0.9
        )
    
    def test_enhanced_destination_creation(self):
        destination = Destination(
            id="test_dest_1",
            names=["Test City"],
            admin_levels={"country": "US", "state": "OR"},
            timezone="America/Los_Angeles",
            country_code="US",
            authentic_insights=[self.authentic_insight],
            local_authorities=[self.local_authority]
        )
        
        self.assertEqual(destination.id, "test_dest_1")
        self.assertEqual(len(destination.authentic_insights), 1)
        self.assertEqual(len(destination.local_authorities), 1)
        self.assertIsInstance(destination.authentic_insights[0], AuthenticInsight)
        self.assertIsInstance(destination.local_authorities[0], LocalAuthority)
    
    def test_enhanced_destination_to_dict(self):
        destination = Destination(
            id="test_dest_2",
            names=["Another City"],
            admin_levels={"country": "CA"},
            timezone="America/Toronto",
            country_code="CA",
            authentic_insights=[self.authentic_insight],
            local_authorities=[self.local_authority]
        )
        
        result = destination.to_dict()
        
        self.assertEqual(result["id"], "test_dest_2")
        self.assertEqual(len(result["authentic_insights"]), 1)
        self.assertEqual(len(result["local_authorities"]), 1)
        self.assertIsInstance(result["authentic_insights"][0], dict)
        self.assertIsInstance(result["local_authorities"][0], dict)


if __name__ == '__main__':
    unittest.main() 