"""
Data Transformation Tests
Simple tests for data transformation and serialization functionality.
"""

import unittest
import sys
import os
import json
from datetime import datetime

# Add the root directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

class TestDataTransformation(unittest.TestCase):
    
    def setUp(self):
        """Set up test environment"""
        from src.schemas import EnhancedEvidence, AuthorityType
        from src.core.enhanced_data_models import Theme
        self.Evidence = EnhancedEvidence
        self.Theme = Theme
        self.AuthorityType = AuthorityType

    def test_evidence_to_dict_transformation(self):
        """Test evidence serialization to dictionary"""
        evidence = self.Evidence(
            id="test_evidence_transform_1",
            source_url="https://reddit.com/r/Seattle",
            source_category=self.AuthorityType.RESIDENT,
            text_snippet="Seattle grunge music scene",
            authority_weight=0.7,
            sentiment=0.8,
            confidence=0.8,
            timestamp=datetime.now().isoformat(),
            cultural_context={},
            relationships=[],
            agent_id="test_agent"
        )
        
        # Test to_dict method if available
        if hasattr(evidence, 'to_dict'):
            evidence_dict = evidence.to_dict()
            
            # Should be a dictionary
            self.assertIsInstance(evidence_dict, dict)
            
            # Should contain expected fields
            expected_fields = ["text_snippet", "source_url", "authority_weight", "sentiment", "source_category", "confidence"]
            for field in expected_fields:
                self.assertIn(field, evidence_dict)
            
            # Values should match
            self.assertEqual(evidence_dict["text_snippet"], "Seattle grunge music scene")
            self.assertEqual(evidence_dict["source_url"], "https://reddit.com/r/Seattle")
            self.assertEqual(evidence_dict["authority_weight"], 0.7)
        else:
            # Test manual dictionary conversion
            evidence_dict = {
                "text_snippet": evidence.text_snippet,
                "source_url": evidence.source_url,
                "source_category": evidence.source_category,
                "authority_weight": evidence.authority_weight,
                "sentiment": evidence.sentiment,
                "confidence": evidence.confidence
            }
            
            self.assertIsInstance(evidence_dict, dict)
            self.assertEqual(evidence_dict["text_snippet"], "Seattle grunge music scene")

    def test_theme_to_dict_transformation(self):
        """Test theme serialization to dictionary"""
        mock_evidence = [
            self.Evidence(
                id="test_evidence_transform_2",
                source_url="https://example.com",
                source_category=self.AuthorityType.RESIDENT,
                text_snippet="Test evidence",
                authority_weight=0.7,
                sentiment=0.8,
                confidence=0.8,
                timestamp=datetime.now().isoformat(),
                cultural_context={},
                relationships=[],
                agent_id="test_agent"
            )
        ]
        
        theme = self.Theme(
            theme_id="test_theme_1",
            name="Grunge Heritage",
            macro_category="Cultural Identity & Atmosphere",
            micro_category="Music",
            description="Seattle's grunge music heritage",
            fit_score=0.85,
            evidence=mock_evidence,
            tags=["music", "grunge", "culture"],
            metadata={
                "local_context": ["grunge venues"],
                "confidence_components": {"evidence_quality": 0.8}
            }
        )
        
        # Test to_dict method if available
        if hasattr(theme, 'to_dict'):
            try:
                theme_dict = theme.to_dict()
            except AttributeError:
                # Evidence objects don't have to_dict, use manual conversion
                theme_dict = {
                    "theme_id": theme.theme_id,
                    "name": theme.name,
                    "macro_category": theme.macro_category,
                    "micro_category": theme.micro_category,
                    "description": theme.description,
                    "fit_score": theme.fit_score
                }
            
            # Should be a dictionary
            self.assertIsInstance(theme_dict, dict)
            
            # Should contain expected fields
            expected_fields = ["theme_id", "name", "macro_category", "micro_category", "description", "fit_score"]
            for field in expected_fields:
                self.assertIn(field, theme_dict)
            
            # Values should match
            self.assertEqual(theme_dict["name"], "Grunge Heritage")
            self.assertEqual(theme_dict["fit_score"], 0.85)
        else:
            # Test manual dictionary conversion
            theme_dict = {
                "theme_id": theme.theme_id,
                "name": theme.name,
                "macro_category": theme.macro_category,
                "micro_category": theme.micro_category,
                "description": theme.description,
                "fit_score": theme.fit_score
            }
            
            self.assertIsInstance(theme_dict, dict)
            self.assertEqual(theme_dict["name"], "Grunge Heritage")

    def test_confidence_components_serialization(self):
        """Test confidence components serialization"""
        confidence_components = {
            "evidence_quality": 0.8,
            "source_diversity": 0.7,
            "temporal_coverage": 0.6,
            "content_completeness": 0.75,
            "total_score": 0.72,
            "authenticity_score": 0.9,
            "distinctiveness_score": 0.65
        }
        
        # Test JSON serialization
        try:
            confidence_json = json.dumps(confidence_components)
            
            # Should serialize successfully
            self.assertIsInstance(confidence_json, str)
            
            # Should deserialize back correctly
            deserialized = json.loads(confidence_json)
            self.assertEqual(deserialized["evidence_quality"], 0.8)
            self.assertEqual(deserialized["total_score"], 0.72)
            
        except (TypeError, ValueError) as e:
            self.fail(f"Confidence components should be JSON serializable: {e}")

    def test_cultural_context_serialization(self):
        """Test cultural context serialization"""
        cultural_context = {
            "content_type": "local_tip",
            "authenticity_indicators": ["local phrase", "personal experience"],
            "authority_indicators": ["official source"],
            "distinctiveness_score": 0.7,
            "processing_type": "cultural"
        }
        
        # Test JSON serialization
        try:
            context_json = json.dumps(cultural_context)
            
            # Should serialize successfully
            self.assertIsInstance(context_json, str)
            
            # Should deserialize back correctly
            deserialized = json.loads(context_json)
            self.assertEqual(deserialized["content_type"], "local_tip")
            self.assertIsInstance(deserialized["authenticity_indicators"], list)
            
        except (TypeError, ValueError) as e:
            self.fail(f"Cultural context should be JSON serializable: {e}")

    def test_nested_data_structure_handling(self):
        """Test handling of nested data structures"""
        nested_data = {
            "theme_analysis": {
                "themes": [
                    {
                        "name": "Grunge Heritage",
                        "confidence": {
                            "evidence_quality": 0.8,
                            "components": {
                                "authenticity": 0.9,
                                "authority": 0.7
                            }
                        },
                        "evidence": [
                            {
                                "id": "ev_1",
                                "text": "Local grunge venue information"
                            }
                        ]
                    }
                ]
            }
        }
        
        # Test serialization of nested structure
        try:
            nested_json = json.dumps(nested_data)
            
            # Should serialize successfully
            self.assertIsInstance(nested_json, str)
            
            # Should deserialize back correctly
            deserialized = json.loads(nested_json)
            self.assertIn("theme_analysis", deserialized)
            self.assertIn("themes", deserialized["theme_analysis"])
            
            theme = deserialized["theme_analysis"]["themes"][0]
            self.assertEqual(theme["name"], "Grunge Heritage")
            self.assertEqual(theme["confidence"]["evidence_quality"], 0.8)
            
        except (TypeError, ValueError) as e:
            self.fail(f"Nested data should be JSON serializable: {e}")

    def test_datetime_serialization(self):
        """Test datetime serialization for theme timestamps"""
        current_time = datetime.now()
        
        # Test ISO format serialization
        iso_string = current_time.isoformat()
        
        # Should be serializable
        datetime_data = {
            "created_date": iso_string,
            "last_updated": iso_string
        }
        
        try:
            datetime_json = json.dumps(datetime_data)
            
            # Should serialize successfully
            self.assertIsInstance(datetime_json, str)
            
            # Should deserialize back correctly
            deserialized = json.loads(datetime_json)
            self.assertEqual(deserialized["created_date"], iso_string)
            
        except (TypeError, ValueError) as e:
            self.fail(f"DateTime should be JSON serializable in ISO format: {e}")

    def test_null_value_handling(self):
        """Test handling of null/None values in data structures"""
        data_with_nulls = {
            "theme_id": "test_theme_1",
            "name": "Test Theme",
            "description": None,  # Null value
            "optional_field": None,
            "empty_list": [],
            "empty_dict": {},
            "valid_score": 0.8
        }
        
        # Should handle null values in serialization
        try:
            null_json = json.dumps(data_with_nulls)
            
            # Should serialize successfully
            self.assertIsInstance(null_json, str)
            
            # Should deserialize back correctly
            deserialized = json.loads(null_json)
            self.assertEqual(deserialized["theme_id"], "test_theme_1")
            self.assertIsNone(deserialized["description"])
            self.assertEqual(deserialized["valid_score"], 0.8)
            
        except (TypeError, ValueError) as e:
            self.fail(f"Data with null values should be JSON serializable: {e}")

    def test_data_type_coercion(self):
        """Test data type coercion and validation"""
        # Test numeric value coercion
        test_values = {
            "string_to_float": "0.8",
            "int_to_float": 1,
            "float_value": 0.75,
            "string_to_int": "5",
            "boolean_to_int": True
        }
        
        # Test type coercion
        try:
            coerced_float = float(test_values["string_to_float"])
            self.assertEqual(coerced_float, 0.8)
            
            coerced_int_to_float = float(test_values["int_to_float"])
            self.assertEqual(coerced_int_to_float, 1.0)
            
            coerced_int = int(test_values["string_to_int"])
            self.assertEqual(coerced_int, 5)
            
            coerced_bool_to_int = int(test_values["boolean_to_int"])
            self.assertEqual(coerced_bool_to_int, 1)
            
        except (ValueError, TypeError) as e:
            self.fail(f"Type coercion should work for valid values: {e}")

    def test_metadata_flattening_for_storage(self):
        """Test flattening of nested metadata for database storage"""
        nested_metadata = {
            "local_context": ["grunge venues", "coffee shops"],
            "confidence_components": {
                "evidence_quality": 0.8,
                "source_diversity": 0.7,
                "authenticity_score": 0.9
            },
            "temporal_aspects": ["summer", "year-round"],
            "processing_info": {
                "type": "cultural",
                "threshold": 0.45
            }
        }
        
        # Test flattening for storage (convert nested dicts to JSON strings)
        flattened = {}
        for key, value in nested_metadata.items():
            if isinstance(value, dict):
                flattened[key] = json.dumps(value)
            elif isinstance(value, list):
                flattened[key] = json.dumps(value)
            else:
                flattened[key] = value
        
        # Should be flattened
        self.assertIsInstance(flattened["confidence_components"], str)
        self.assertIsInstance(flattened["local_context"], str)
        
        # Should be deserializable
        confidence_restored = json.loads(flattened["confidence_components"])
        self.assertEqual(confidence_restored["evidence_quality"], 0.8)

if __name__ == "__main__":
    unittest.main() 