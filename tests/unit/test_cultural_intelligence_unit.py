"""
Unit Tests for Cultural Intelligence Components
Simple tests for cultural intelligence scoring and filtering functionality.
"""

import unittest
import sys
import os

# Add the root directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

class TestCulturalIntelligenceUnit(unittest.TestCase):
    
    def setUp(self):
        """Set up test environment"""
        from src.tools.enhanced_theme_analysis_tool import EnhancedThemeAnalysisTool
        from src.schemas import EnhancedEvidence, AuthorityType
        from datetime import datetime
        self.tool = EnhancedThemeAnalysisTool()
        self.Evidence = EnhancedEvidence
        self.AuthorityType = AuthorityType
        self.datetime = datetime

    def test_authentic_source_detection(self):
        """Test detection of authentic sources"""
        reddit_evidence = [
            self.Evidence(
                id="test_evidence_reddit",
                text_snippet="As a local who's lived here for 10 years...",
                source_url="https://reddit.com/r/Seattle",
                source_category=self.AuthorityType.RESIDENT,
                authority_weight=0.6,
                sentiment=0.8,
                confidence=0.8,
                timestamp=self.datetime.now().isoformat(),
                cultural_context={},
                relationships=[],
                agent_id="test_agent"
            )
        ]
        
        # Should detect reddit as authentic source
        has_authentic = self.tool._has_authentic_indicators(reddit_evidence)
        self.assertTrue(has_authentic, "Reddit should be considered authentic source")

    def test_distinctiveness_keyword_matching(self):
        """Test distinctiveness keyword detection"""
        distinctive_evidence = [
            self.Evidence(
                id="test_evidence_distinctive",
                text_snippet="This unique venue is signature to Seattle only",
                source_url="https://example.com",
                source_category=self.AuthorityType.RESIDENT,
                authority_weight=0.7,
                sentiment=0.8,
                confidence=0.8,
                timestamp=self.datetime.now().isoformat(),
                cultural_context={},
                relationships=[],
                agent_id="test_agent"
            )
        ]
        
        # Should detect distinctiveness keywords
        score = self.tool._calculate_distinctiveness_score(distinctive_evidence)
        self.assertGreaterEqual(score, 0.5, "Should detect distinctiveness keywords")

    def test_authenticity_phrase_detection(self):
        """Test detection of authenticity phrases"""
        authentic_evidence = [
            self.Evidence(
                id="test_evidence_phrases",
                text_snippet="As a local, I can tell you this is a hidden gem",
                source_url="https://example.com",
                source_category=self.AuthorityType.RESIDENT,
                authority_weight=0.7,
                sentiment=0.8,
                confidence=0.8,
                timestamp=self.datetime.now().isoformat(),
                cultural_context={},
                relationships=[],
                agent_id="test_agent"
            )
        ]
        
        # Should detect authenticity phrases
        score = self.tool._calculate_authenticity_score(authentic_evidence)
        self.assertGreaterEqual(score, 0.5, "Should detect authenticity phrases")

    def test_cultural_context_scoring(self):
        """Test cultural context scoring"""
        cultural_evidence = [
            self.Evidence(
                id="test_evidence_cultural",
                text_snippet="Seattle's grunge music scene is deeply rooted in local culture",
                source_url="https://example.com",
                source_category=self.AuthorityType.RESIDENT,
                authority_weight=0.7,
                sentiment=0.8,
                confidence=0.8,
                timestamp=self.datetime.now().isoformat(),
                cultural_context={"is_cultural": True},
                relationships=[],
                agent_id="test_agent"
            )
        ]
        
        # Should score higher for cultural themes
        confidence = self.tool._calculate_cultural_enhanced_confidence(
            cultural_evidence, {"music"}, {"grunge"}, {"historical"}, "Cultural Identity & Atmosphere"
        )
        
        self.assertIn("total_score", confidence)
        self.assertGreaterEqual(confidence["total_score"], 0.4, "Cultural themes should have reasonable scores")

    def test_authority_vs_authenticity_balance(self):
        """Test balance between authority and authenticity"""
        official_evidence = [
            self.Evidence(
                id="test_evidence_official",
                text_snippet="Official tourism information about Seattle attractions",
                source_url="https://visitseattle.org",
                source_category=self.AuthorityType.OFFICIAL,
                authority_weight=0.9,
                sentiment=0.7,
                confidence=0.9,
                timestamp=self.datetime.now().isoformat(),
                cultural_context={"is_official": True},
                relationships=[],
                agent_id="test_agent"
            )
        ]
        
        resident_evidence = [
            self.Evidence(
                id="test_evidence_resident",
                text_snippet="As a local, here's what tourists don't know about Seattle",
                source_url="https://reddit.com/r/Seattle",
                source_category=self.AuthorityType.RESIDENT,
                authority_weight=0.6,
                sentiment=0.8,
                confidence=0.7,
                timestamp=self.datetime.now().isoformat(),
                cultural_context={"is_local": True},
                relationships=[],
                agent_id="test_agent"
            )
        ]
        
        # Both should have valid scores but different characteristics
        official_score = self.tool._calculate_cultural_enhanced_confidence(
            official_evidence, {"tourism"}, {"official"}, {"current"}, "Practical Information"
        )
        
        resident_score = self.tool._calculate_cultural_enhanced_confidence(
            resident_evidence, {"local"}, {"insider"}, {"current"}, "Cultural Identity & Atmosphere"
        )
        
        self.assertIsInstance(official_score["total_score"], (int, float))
        self.assertIsInstance(resident_score["total_score"], (int, float))

    def test_has_authentic_indicators(self):
        """Test has_authentic_indicators method"""
        evidence_with_indicators = [
            self.Evidence(
                id="test_evidence_indicators",
                text_snippet="As a local resident, I live here and know the hidden spots",
                source_url="https://reddit.com/r/Seattle",
                source_category=self.AuthorityType.RESIDENT,
                authority_weight=0.7,
                sentiment=0.8,
                confidence=0.8,
                timestamp=self.datetime.now().isoformat(),
                cultural_context={},
                relationships=[],
                agent_id="test_agent"
            )
        ]
        
        # Should detect multiple authentic indicators
        has_indicators = self.tool._has_authentic_indicators(evidence_with_indicators)
        self.assertTrue(has_indicators, "Should detect authentic indicators")

    def test_cultural_track_processing(self):
        """Test cultural track processing type assignment"""
        cultural_categories = [
            "Cultural Identity & Atmosphere",
            "Cultural & Arts", 
            "Heritage & History"
        ]
        
        for category in cultural_categories:
            processing_type = self.tool._get_processing_type(category)
            self.assertEqual(processing_type, "cultural", 
                           f"{category} should be processed as cultural")

    def test_practical_track_processing(self):
        """Test practical track processing type assignment"""
        practical_categories = [
            "Transportation & Access",
            "Safety & Security",
            "Budget & Costs"
        ]
        
        for category in practical_categories:
            processing_type = self.tool._get_processing_type(category)
            self.assertEqual(processing_type, "practical",
                           f"{category} should be processed as practical")

    def test_hybrid_track_processing(self):
        """Test hybrid track processing type assignment"""
        hybrid_categories = [
            "Food & Dining",
            "Entertainment & Nightlife", 
            "Nature & Outdoor"
        ]
        
        for category in hybrid_categories:
            processing_type = self.tool._get_processing_type(category)
            self.assertEqual(processing_type, "hybrid",
                           f"{category} should be processed as hybrid")

if __name__ == "__main__":
    unittest.main() 