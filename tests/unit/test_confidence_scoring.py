import unittest
from datetime import datetime
from src.core.confidence_scoring import (
    ConfidenceScorer, AuthenticityScorer, UniquenessScorer, 
    ActionabilityScorer, MultiDimensionalScore, ConfidenceBreakdown, ConfidenceLevel
)
from src.core.enhanced_data_models import Evidence, LocalAuthority, AuthenticInsight
from src.core.evidence_hierarchy import SourceCategory, EvidenceType
from src.schemas import AuthorityType, InsightType, LocationExclusivity


class TestMultiDimensionalScore(unittest.TestCase):
    
    def test_multi_dimensional_score_creation(self):
        score = MultiDimensionalScore(
            authenticity=0.8,
            uniqueness=0.7,
            actionability=0.9,
            temporal_relevance=0.6
        )
        
        self.assertEqual(score.authenticity, 0.8)
        self.assertEqual(score.uniqueness, 0.7)
        self.assertEqual(score.actionability, 0.9)
        self.assertEqual(score.temporal_relevance, 0.6)
    
    def test_multi_dimensional_score_weighted_average(self):
        score = MultiDimensionalScore(
            authenticity=0.8,
            uniqueness=0.6,
            actionability=1.0,
            temporal_relevance=0.4
        )
        
        # Test default weights (equal)
        avg = score.weighted_average()
        expected = (0.8 + 0.6 + 1.0 + 0.4) / 4
        self.assertAlmostEqual(avg, expected, places=3)
        
        # Test custom weights
        weights = {"authenticity": 0.4, "uniqueness": 0.3, "actionability": 0.2, "temporal_relevance": 0.1}
        weighted_avg = score.weighted_average(weights)
        expected_weighted = 0.8*0.4 + 0.6*0.3 + 1.0*0.2 + 0.4*0.1
        self.assertAlmostEqual(weighted_avg, expected_weighted, places=3)


class TestAuthenticityScorer(unittest.TestCase):
    
    def setUp(self):
        self.scorer = AuthenticityScorer()
        
        self.local_authority = LocalAuthority(
            authority_type=AuthorityType.PRODUCER,
            local_tenure=10,
            expertise_domain="Local brewing",
            community_validation=0.9
        )
        
        self.evidence = Evidence(
            id="test_evidence",
            source_url="https://localbrewery.com",
            source_category=SourceCategory.BLOG,
            evidence_type=EvidenceType.PRIMARY,
            authority_weight=0.8,
            text_snippet="Authentic local brewery content",
            timestamp=datetime.now(),
            confidence=0.7
        )
    
    def test_calculate_local_authority_score(self):
        authorities = [self.local_authority]
        score = self.scorer.calculate_local_authority_score(authorities)
        
        self.assertGreater(score, 0.0)
        self.assertLessEqual(score, 1.0)
    
    def test_calculate_source_diversity_score(self):
        evidence_list = [self.evidence]
        score = self.scorer.calculate_source_diversity_score(evidence_list)
        
        # Single source should have 0 diversity
        self.assertGreaterEqual(score, 0.0)
        self.assertLessEqual(score, 1.0)
    
    def test_calculate_authenticity_complete(self):
        authorities = [self.local_authority]
        evidence_list = [self.evidence]
        content = "Local brewery with authentic craft beer traditions"
        
        score = self.scorer.calculate_authenticity(authorities, evidence_list, content)
        
        self.assertGreater(score, 0.0)
        self.assertLessEqual(score, 1.0)
    
    def test_calculate_authenticity_empty_inputs(self):
        score = self.scorer.calculate_authenticity([], [], "")
        self.assertEqual(score, 0.0)


class TestUniquenessScorer(unittest.TestCase):
    
    def setUp(self):
        self.scorer = UniquenessScorer()
    
    def test_calculate_uniqueness_with_exclusive(self):
        insight = AuthenticInsight(
            insight_type=InsightType.SPECIALTY,
            authenticity_score=0.8,
            uniqueness_score=0.0,  # Will be calculated
            actionability_score=0.7,
            temporal_relevance=0.6,
            location_exclusivity=LocationExclusivity.EXCLUSIVE,
            seasonal_window=None,
            local_validation_count=5
        )
        
        score = self.scorer.calculate_uniqueness([insight], "unique local specialty")
        self.assertGreater(score, 0.5)  # Should be high for exclusive
    
    def test_calculate_uniqueness_with_common(self):
        insight = AuthenticInsight(
            insight_type=InsightType.PRACTICAL,
            authenticity_score=0.6,
            uniqueness_score=0.0,
            actionability_score=0.8,
            temporal_relevance=0.5,
            location_exclusivity=LocationExclusivity.COMMON,
            seasonal_window=None,
            local_validation_count=2
        )
        
        score = self.scorer.calculate_uniqueness([insight], "common tourist attraction")
        self.assertLess(score, 0.8)  # Should be lower for common
    
    def test_calculate_uniqueness_empty(self):
        score = self.scorer.calculate_uniqueness([], "")
        self.assertEqual(score, 0.0)


class TestActionabilityScorer(unittest.TestCase):
    
    def setUp(self):
        self.scorer = ActionabilityScorer()
    
    def test_calculate_actionability_with_details(self):
        content = """
        Visit the local brewery at 123 Main Street. 
        Open Monday-Friday 2-8pm, Saturday-Sunday 12-10pm.
        Call (555) 123-4567 to make a reservation.
        Best to visit during their happy hour 4-6pm.
        """
        
        score = self.scorer.calculate_actionability(content)
        self.assertGreater(score, 0.5)  # Should be high with specific details
    
    def test_calculate_actionability_minimal_details(self):
        content = "There's a nice brewery somewhere in town."
        
        score = self.scorer.calculate_actionability(content)
        self.assertLess(score, 0.5)  # Should be low with vague details
    
    def test_calculate_actionability_empty(self):
        score = self.scorer.calculate_actionability("")
        self.assertEqual(score, 0.0)
    
    def test_extract_actionable_elements(self):
        content = """
        Located at 456 Oak Street, open daily 9am-5pm.
        Book online at website.com or call (555) 987-6543.
        Best time to visit is early morning or late afternoon.
        """
        
        elements = self.scorer.extract_actionable_elements(content)
        
        # Should find address, hours, contact, timing
        self.assertGreater(len(elements), 0)
        element_text = " ".join(elements).lower()
        self.assertIn("456 oak street", element_text)
        self.assertIn("9am-5pm", element_text)
        self.assertIn("555", element_text)


class TestConfidenceScorer(unittest.TestCase):
    
    def setUp(self):
        self.scorer = ConfidenceScorer()
    
    def test_calculate_confidence_with_evidence(self):
        evidence_list = [
            Evidence(
                id="evidence1",
                source_url="https://government.org",
                source_category=SourceCategory.GOVERNMENT,
                evidence_type=EvidenceType.PRIMARY,
                authority_weight=0.9,
                text_snippet="Official tourism information",
                timestamp=datetime.now(),
                confidence=0.8
            )
        ]
        
        breakdown = self.scorer.calculate_confidence(evidence_list)
        
        self.assertIsInstance(breakdown, ConfidenceBreakdown)
        self.assertGreater(breakdown.overall_confidence, 0.0)
        self.assertLessEqual(breakdown.overall_confidence, 1.0)
        self.assertIn(breakdown.confidence_level, list(ConfidenceLevel))
    
    def test_calculate_confidence_empty_evidence(self):
        breakdown = self.scorer.calculate_confidence([])
        
        self.assertEqual(breakdown.overall_confidence, 0.0)
        self.assertEqual(breakdown.confidence_level, ConfidenceLevel.INSUFFICIENT)
    
    def test_evidence_quality_score(self):
        high_quality = Evidence(
            id="hq_evidence",
            source_url="https://academic.edu",
            source_category=SourceCategory.ACADEMIC,
            evidence_type=EvidenceType.PRIMARY,
            authority_weight=0.95,
            text_snippet="Peer-reviewed research",
            timestamp=datetime.now(),
            confidence=0.9
        )
        
        low_quality = Evidence(
            id="lq_evidence",
            source_url="https://unknown-blog.com",
            source_category=SourceCategory.UNKNOWN,
            evidence_type=EvidenceType.TERTIARY,
            authority_weight=0.2,
            text_snippet="Unverified information",
            timestamp=datetime(2020, 1, 1),
            confidence=0.3
        )
        
        hq_score = self.scorer._evidence_quality_score(high_quality)
        lq_score = self.scorer._evidence_quality_score(low_quality)
        
        self.assertGreater(hq_score, lq_score)


if __name__ == '__main__':
    unittest.main() 