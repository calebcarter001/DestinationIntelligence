"""
Database Integration Tests
Simple tests for theme and evidence storage in the database.
"""

import unittest
import sys
import os
import tempfile
import sqlite3
import json
from datetime import datetime
from src.schemas import EnhancedEvidence, AuthorityType
from src.core.enhanced_data_models import Theme

# Add the root directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

class TestDatabaseIntegration(unittest.TestCase):
    
    def setUp(self):
        """Set up test environment with temporary database"""
        self.temp_db = tempfile.NamedTemporaryFile(delete=False, suffix='.db')
        self.temp_db_path = self.temp_db.name
        self.temp_db.close()
        
        self.Evidence = EnhancedEvidence
        self.Theme = Theme

    def tearDown(self):
        """Clean up temporary database"""
        try:
            os.unlink(self.temp_db_path)
        except:
            pass

    def test_theme_storage_schema_compatibility(self):
        """Test that theme data matches database schema"""
        # Create test database with themes table
        conn = sqlite3.connect(self.temp_db_path)
        cursor = conn.cursor()
        
        # Create themes table with expected schema
        cursor.execute("""
            CREATE TABLE themes (
                theme_id TEXT PRIMARY KEY,
                destination_id TEXT,
                name TEXT,
                macro_category TEXT,
                micro_category TEXT,
                description TEXT,
                fit_score REAL,
                confidence_level TEXT,
                adjusted_overall_confidence REAL,
                traveler_relevance_factor REAL,
                confidence_breakdown TEXT,
                tags TEXT
            )
        """)
        
        # Test data that should be storable
        test_theme_data = {
            "theme_id": "test_theme_1",
            "destination_id": "dest_seattle_us",
            "name": "Grunge Heritage",
            "macro_category": "Cultural Identity & Atmosphere",
            "micro_category": "Music",
            "description": "Test theme",
            "fit_score": 0.85,
            "confidence_level": "HIGH",
            "adjusted_overall_confidence": 0.82,
            "traveler_relevance_factor": 0.8,
            "confidence_breakdown": '{"evidence_quality": 0.8, "source_diversity": 0.7}',
            "tags": "music,grunge,culture"
        }
        
        try:
            cursor.execute("""
                INSERT INTO themes (theme_id, destination_id, name, macro_category, micro_category,
                                  description, fit_score, confidence_level, adjusted_overall_confidence,
                                  traveler_relevance_factor, confidence_breakdown, tags)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, tuple(test_theme_data.values()))
            
            conn.commit()
            
            # Verify storage
            cursor.execute("SELECT COUNT(*) FROM themes")
            count = cursor.fetchone()[0]
            self.assertEqual(count, 1, "Theme should be stored successfully")
            
        except sqlite3.Error as e:
            self.fail(f"Database storage failed: {e}")
        finally:
            conn.close()

    def test_evidence_storage_schema_compatibility(self):
        """Test that evidence data matches database schema"""
        conn = sqlite3.connect(self.temp_db_path)
        cursor = conn.cursor()
        
        # Create evidence table
        cursor.execute("""
            CREATE TABLE evidence (
                evidence_id TEXT PRIMARY KEY,
                destination_id TEXT,
                text_snippet TEXT,
                source_url TEXT,
                authority_weight REAL,
                sentiment REAL,
                cultural_context TEXT,
                source_category TEXT,
                confidence REAL,
                timestamp TEXT
            )
        """)
        
        test_evidence_data = {
            "evidence_id": "ev_test_1",
            "destination_id": "dest_seattle_us", 
            "text_snippet": "Local tip about Seattle grunge music",
            "source_url": "https://reddit.com/r/Seattle",
            "authority_weight": 0.7,
            "sentiment": 0.8,
            "cultural_context": '{"content_type": "local_tip"}',
            "source_category": AuthorityType.RESIDENT.value,
            "confidence": 0.8,
            "timestamp": datetime.now().isoformat()
        }
        
        try:
            cursor.execute("""
                INSERT INTO evidence (evidence_id, destination_id, text_snippet, source_url,
                                    authority_weight, sentiment, cultural_context, source_category,
                                    confidence, timestamp)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, tuple(test_evidence_data.values()))
            
            conn.commit()
            
            # Verify storage
            cursor.execute("SELECT COUNT(*) FROM evidence")
            count = cursor.fetchone()[0]
            self.assertEqual(count, 1, "Evidence should be stored successfully")
            
        except sqlite3.Error as e:
            self.fail(f"Evidence storage failed: {e}")
        finally:
            conn.close()

    def test_theme_evidence_relationship_storage(self):
        """Test that theme-evidence relationships can be stored"""
        conn = sqlite3.connect(self.temp_db_path)
        cursor = conn.cursor()
        
        # Create relationship table
        cursor.execute("""
            CREATE TABLE theme_evidence_relationships (
                theme_id TEXT,
                evidence_id TEXT,
                relevance_score REAL,
                PRIMARY KEY (theme_id, evidence_id)
            )
        """)
        
        test_relationship = {
            "theme_id": "theme_test_1",
            "evidence_id": "ev_test_1",
            "relevance_score": 0.85
        }
        
        try:
            cursor.execute("""
                INSERT INTO theme_evidence_relationships (theme_id, evidence_id, relevance_score)
                VALUES (?, ?, ?)
            """, tuple(test_relationship.values()))
            
            conn.commit()
            
            # Verify relationship storage
            cursor.execute("SELECT COUNT(*) FROM theme_evidence_relationships")
            count = cursor.fetchone()[0]
            self.assertEqual(count, 1, "Relationship should be stored successfully")
            
        except sqlite3.Error as e:
            self.fail(f"Relationship storage failed: {e}")
        finally:
            conn.close()

    def test_confidence_breakdown_json_storage(self):
        """Test that confidence breakdown JSON can be stored and retrieved"""
        conn = sqlite3.connect(self.temp_db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            CREATE TABLE themes (
                theme_id TEXT PRIMARY KEY,
                confidence_breakdown TEXT
            )
        """)
        
        # Test confidence breakdown that would be generated by cultural intelligence
        confidence_breakdown = {
            "evidence_quality": 0.8,
            "source_diversity": 0.7,
            "temporal_coverage": 0.6,
            "content_completeness": 0.75,
            "total_score": 0.72,
            "authenticity_score": 0.9,
            "distinctiveness_score": 0.65
        }
        
        confidence_json = json.dumps(confidence_breakdown)
        
        try:
            cursor.execute("INSERT INTO themes (theme_id, confidence_breakdown) VALUES (?, ?)",
                         ("test_theme", confidence_json))
            conn.commit()
            
            # Retrieve and verify
            cursor.execute("SELECT confidence_breakdown FROM themes WHERE theme_id = ?", ("test_theme",))
            result = cursor.fetchone()[0]
            
            retrieved_data = json.loads(result)
            self.assertEqual(retrieved_data["evidence_quality"], 0.8)
            self.assertIn("authenticity_score", retrieved_data)
            
        except (sqlite3.Error, json.JSONDecodeError) as e:
            self.fail(f"Confidence breakdown storage/retrieval failed: {e}")
        finally:
            conn.close()

if __name__ == "__main__":
    unittest.main() 