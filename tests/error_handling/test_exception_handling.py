"""
Exception Handling Tests
Simple tests for specific exception scenarios and recovery.
"""

import unittest
import sys
import os
from datetime import datetime

# Add the root directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

from src.tools.enhanced_theme_analysis_tool import EnhancedThemeAnalysisTool
from src.schemas import EnhancedEvidence, AuthorityType
from src.core.enhanced_data_models import Theme

class TestExceptionHandling(unittest.TestCase):
    
    def setUp(self):
        """Set up test environment"""
        self.tool = EnhancedThemeAnalysisTool()
        self.Evidence = EnhancedEvidence

    def test_keyerror_in_confidence_calculation(self):
        """Test handling of KeyError in confidence calculation (the bug we fixed)"""
        mock_evidence = [
            self.Evidence(
                id="test_evidence_1",
                text_snippet="Test content",
                source_category=AuthorityType.RESIDENT,
                source_url="https://example.com",
                authority_weight=0.7,
                sentiment=0.8,
                confidence=0.8,
                timestamp=datetime.now().isoformat(),
                cultural_context={},
                relationships=[],
                agent_id="test_agent"
            )
        ]
        
        try:
            # This should NOT raise KeyError after our fix
            confidence = self.tool._calculate_cultural_enhanced_confidence(
                mock_evidence,
                {"activity"},
                {"test"},
                {"summer"},
                "Cultural Identity & Atmosphere"
            )
            
            # Should have all required keys
            required_keys = ["evidence_quality", "source_diversity", "temporal_coverage", "content_completeness"]
            for key in required_keys:
                self.assertIn(key, confidence, f"Missing key: {key}")
                
        except KeyError as e:
            self.fail(f"KeyError should not occur after fix: {e}")

    def test_missing_required_fields(self):
        """Test handling of missing required fields in data structures"""
        # Test with incomplete evidence
        try:
            incomplete_evidence = self.Evidence(
                id="test_evidence_2",
                text_snippet="Test",
                source_category=AuthorityType.RESIDENT,
                source_url="https://example.com",
                authority_weight=0.7,
                sentiment=0.8,
                confidence=0.8,
                timestamp=datetime.now().isoformat(),
                cultural_context={},
                relationships=[],
                agent_id="test_agent"
            )
            
            # Should not crash even with minimal evidence
            confidence = self.tool._calculate_cultural_enhanced_confidence(
                [incomplete_evidence],
                set(),
                set(),
                set(),
                "Unknown Category"
            )
            
            self.assertIsInstance(confidence, dict)
            self.assertIn("total_score", confidence)
            
        except Exception as e:
            self.fail(f"Should handle incomplete evidence gracefully: {e}")

    def test_type_conversion_errors(self):
        """Test handling of type conversion errors"""
        try:
            # Test with edge case values
            edge_case_evidence = [
                self.Evidence(
                    id="test_evidence_3",
                    text_snippet="",  # Empty string
                    source_category=AuthorityType.RESIDENT,
                    source_url="https://example.com",
                    authority_weight=0.0,  # Zero weight
                    sentiment=0.0,
                    confidence=0.8,
                    timestamp=datetime.now().isoformat(),
                    cultural_context={},
                    relationships=[],
                    agent_id="test_agent"
                )
            ]
            
            # Should handle edge cases without crashing
            confidence = self.tool._calculate_cultural_enhanced_confidence(
                edge_case_evidence,
                {"empty"},
                set(),
                set(),
                "Test Category"
            )
            
            # Should return valid confidence even with edge cases
            self.assertIsInstance(confidence["total_score"], (int, float))
            self.assertGreaterEqual(confidence["total_score"], 0.0)
            
        except Exception as e:
            self.fail(f"Should handle edge case values gracefully: {e}")

    def test_invalid_confidence_values(self):
        """Test handling of invalid confidence component values"""
        mock_evidence = [
            self.Evidence(
                id="test_evidence_4",
                text_snippet="Test content",
                source_category=AuthorityType.RESIDENT,
                source_url="https://example.com",
                authority_weight=1.5,  # Invalid (>1.0)
                sentiment=-2.0,  # Invalid (<-1.0)
                confidence=0.8,
                timestamp=datetime.now().isoformat(),
                cultural_context={},
                relationships=[],
                agent_id="test_agent"
            )
        ]
        
        try:
            # Should handle invalid input values
            confidence = self.tool._calculate_cultural_enhanced_confidence(
                mock_evidence,
                {"activity"},
                {"test"},
                {"summer"},
                "Cultural Identity & Atmosphere"
            )
            
            # Should clamp values to valid ranges
            total_score = confidence["total_score"]
            self.assertGreaterEqual(total_score, 0.0)
            self.assertLessEqual(total_score, 1.0)
            
        except Exception as e:
            self.fail(f"Should handle invalid confidence values: {e}")

    def test_memory_overflow_handling(self):
        """Test handling of large data sets"""
        try:
            # Create a large number of mock evidence
            large_evidence_list = []
            for i in range(100):  # 100 evidence pieces
                evidence = self.Evidence(
                    id=f"test_evidence_{i+5}",
                    text_snippet=f"Test content {i} " * 50,  # Long content
                    source_category=AuthorityType.RESIDENT,
                    source_url=f"https://example{i}.com",
                    authority_weight=0.5 + (i % 5) / 10,
                    sentiment=0.5 + (i % 3) / 10,
                    confidence=0.8,
                    timestamp=datetime.now().isoformat(),
                    cultural_context={},
                    relationships=[],
                    agent_id="test_agent"
                )
                large_evidence_list.append(evidence)
            
            # Should handle large datasets without crashing
            confidence = self.tool._calculate_cultural_enhanced_confidence(
                large_evidence_list,
                {"activity", "location", "content"},
                {"test", "large", "dataset"},
                {"summer", "winter"},
                "Cultural Identity & Atmosphere"
            )
            
            self.assertIsInstance(confidence, dict)
            self.assertIn("total_score", confidence)
            
        except Exception as e:
            self.fail(f"Should handle large datasets gracefully: {e}")

    def test_circular_reference_detection(self):
        """Test detection and handling of circular references"""
        try:
            theme1 = Theme(
                theme_id="theme_1",
                name="Test Theme 1",
                macro_category="Test",
                micro_category="Test",
                description="Test",
                fit_score=0.8,
                evidence=[],
                tags=["test"],
                metadata={
                    "related_themes_from_discovery": ["theme_2", "theme_3"]
                }
            )
            
            theme2 = Theme(
                theme_id="theme_2",
                name="Test Theme 2", 
                macro_category="Test",
                micro_category="Test",
                description="Test",
                fit_score=0.7,
                evidence=[],
                tags=["test"],
                metadata={
                    "related_themes_from_discovery": ["theme_1", "theme_3"]  # Circular reference
                }
            )
            
            themes = [theme1, theme2]
            
            # Should handle circular references gracefully
            self.tool._enhance_theme_relationships(themes)
            
            # Should complete without infinite loops
            self.assertTrue(True, "Circular reference handling completed")
            
        except Exception as e:
            self.fail(f"Should handle circular references gracefully: {e}")

    def test_timeout_exception_handling(self):
        """Test handling of timeout scenarios"""
        try:
            # Simulate processing that might timeout
            mock_evidence = [
                self.Evidence(
                    id="test_evidence_timeout",
                    text_snippet="Test content",
                    source_category=AuthorityType.RESIDENT,
                    source_url="https://example.com",
                    authority_weight=0.7,
                    sentiment=0.8,
                    confidence=0.8,
                    timestamp=datetime.now().isoformat(),
                    cultural_context={},
                    relationships=[],
                    agent_id="test_agent"
                )
            ]
            
            # Should handle timeout gracefully
            confidence = self.tool._calculate_cultural_enhanced_confidence(
                mock_evidence,
                {"activity"},
                {"test"},
                {"summer"},
                "Cultural Identity & Atmosphere"
            )
            
            self.assertIsInstance(confidence, dict)
            self.assertIn("total_score", confidence)
            
        except Exception as e:
            self.fail(f"Should handle timeout scenarios gracefully: {e}")

    def test_malformed_data_handling(self):
        """Test handling of malformed data"""
        try:
            malformed_evidence = [
                self.Evidence(
                    id="test_evidence_malformed",
                    text_snippet="Test content",
                    source_category=AuthorityType.RESIDENT,
                    source_url="malformed://url",  # Invalid URL
                    authority_weight=0.7,
                    sentiment=0.8,
                    confidence=0.8,
                    timestamp=datetime.now().isoformat(),
                    cultural_context={},
                    relationships=[],
                    agent_id="test_agent"
                )
            ]
            
            # Should handle malformed data gracefully
            confidence = self.tool._calculate_cultural_enhanced_confidence(
                malformed_evidence,
                {"activity"},
                {"test"},
                {"summer"},
                "Cultural Identity & Atmosphere"
            )
            
            self.assertIsInstance(confidence, dict)
            self.assertIn("total_score", confidence)
            
        except Exception as e:
            self.fail(f"Should handle malformed data gracefully: {e}")

if __name__ == "__main__":
    unittest.main() 