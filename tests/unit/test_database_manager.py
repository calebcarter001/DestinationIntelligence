import unittest
import os
import sys
import json
from datetime import datetime

# Add project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from src.core.enhanced_database_manager import EnhancedDatabaseManager
from src.core.enhanced_data_models import Destination, Theme, Evidence
from src.core.evidence_hierarchy import SourceCategory, EvidenceType
from src.core.confidence_scoring import ConfidenceScorer, ConfidenceLevel

class TestDatabaseManager(unittest.TestCase):

    def setUp(self):
        """Set up a fresh in-memory database for each test."""
        self.db_path = ":memory:"
        self.db_manager = EnhancedDatabaseManager(db_path=self.db_path, enable_json_export=False)
        
        # Create a sample evidence piece
        self.sample_evidence = Evidence(
            id="test_evidence_1",
            source_url="https://test.gov/source",
            source_category=SourceCategory.GOVERNMENT,
            evidence_type=EvidenceType.PRIMARY,
            authority_weight=0.9,
            text_snippet="Test evidence snippet for theme analysis",
            timestamp=datetime.now(),
            confidence=0.85
        )
        
        # Create a confidence scorer and calculate confidence
        confidence_scorer = ConfidenceScorer()
        confidence_breakdown = confidence_scorer.calculate_confidence([self.sample_evidence])
        
        # Create a sample theme with all required fields
        self.theme = Theme(
            theme_id="theme_crystal_canyons",
            macro_category="Nature & Outdoor",
            micro_category="Geological",
            name="Crystal Canyons",
            description="Glistening canyons with unique geological formations.",
            fit_score=0.9,
            evidence=[self.sample_evidence],
            confidence_breakdown=confidence_breakdown,
            tags=["nature", "geology", "scenic"],
            metadata={"source": "test"},
            factors={"theme_strength": 0.9},
            cultural_summary={"local_heavy": True},
            sentiment_analysis={"overall": "positive"},
            temporal_analysis={"seasonal_relevance": 0.8}
        )
        
        # Create destination with current schema
        self.destination = Destination(
            id="test_dest_testopolis",
            names=["Testopolis"],
            country_code="FA",
            timezone="UTC",
            population=123456,
            area_km2=500.5,
            primary_language="English",
            hdi=0.9,
            gdp_per_capita_usd=55000,
            vibe_descriptors=["magical", "ancient"],
            historical_summary="A city of wonders.",
            unesco_sites=["The Crystal Spire"],
            annual_tourist_arrivals=2000000,
            popularity_stage="mature",
            visa_info_url="http://fantasia.gov/visa",
            admin_levels={"country": "Fantasia", "city": "Testopolis"},
            core_geo={"elevation": 1200, "climate": "temperate"}
        )

    def tearDown(self):
        """Close the database connection."""
        self.db_manager.close_db()

    def test_database_schema_creation(self):
        """Test that all required tables are created."""
        connection = self.db_manager.get_connection()
        cursor = connection.cursor()
        
        # Check that all expected tables exist
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        tables = [row[0] for row in cursor.fetchall()]
        
        expected_tables = [
            'destinations', 'themes', 'evidence', 'theme_evidence', 
            'dimensions', 'temporal_slices', 'pois', 'seasonal_windows', 
            'authentic_insights', 'local_authorities'
        ]
        
        for table in expected_tables:
            self.assertIn(table, tables, f"Table {table} should exist")

    def test_store_and_get_destination(self):
        """Tests the full cycle of storing and retrieving a complete Destination object."""
        # 1. Store the destination
        self.db_manager.store_destination(self.destination)

        # 2. Retrieve the destination by name
        loaded_destination = self.db_manager.get_destination_by_name("Testopolis")
        
        # 3. Assertions
        self.assertIsNotNone(loaded_destination)
        self.assertEqual(loaded_destination.id, self.destination.id)
        self.assertEqual(loaded_destination.population, self.destination.population)
        self.assertEqual(loaded_destination.area_km2, self.destination.area_km2)
        self.assertEqual(loaded_destination.vibe_descriptors, self.destination.vibe_descriptors)
        self.assertEqual(loaded_destination.historical_summary, self.destination.historical_summary)

    def test_store_theme_separately(self):
        """Test storing theme independently with proper evidence handling."""
        # Store destination first
        self.db_manager.store_destination(self.destination)
        
        # Store theme separately (as themes are now stored independently)
        connection = self.db_manager.get_connection()
        self.db_manager._store_theme(connection.cursor(), self.destination.id, self.theme)
        connection.commit()
        
        # Verify theme was stored
        cursor = connection.cursor()
        cursor.execute("SELECT COUNT(*) FROM themes WHERE destination_id = ?", (self.destination.id,))
        theme_count = cursor.fetchone()[0]
        self.assertEqual(theme_count, 1)
        
        # Verify evidence was stored
        cursor.execute("SELECT COUNT(*) FROM evidence WHERE destination_id = ?", (self.destination.id,))
        evidence_count = cursor.fetchone()[0]
        self.assertEqual(evidence_count, 1)

    def test_confidence_system_integration(self):
        """Test that confidence system works with database storage."""
        # Store destination and theme
        self.db_manager.store_destination(self.destination)
        connection = self.db_manager.get_connection()
        self.db_manager._store_theme(connection.cursor(), self.destination.id, self.theme)
        connection.commit()
        
        # Verify confidence data was stored
        cursor = connection.cursor()
        cursor.execute("""
            SELECT confidence_level, confidence_breakdown 
            FROM themes 
            WHERE destination_id = ? AND name = ?
        """, (self.destination.id, self.theme.name))
        
        result = cursor.fetchone()
        self.assertIsNotNone(result)
        confidence_level, confidence_breakdown = result
        
        # Should have valid confidence level (case insensitive)
        self.assertIn(confidence_level.upper(), ['HIGH', 'MEDIUM', 'LOW', 'INSUFFICIENT'])
        
        # Should have confidence breakdown JSON
        self.assertIsNotNone(confidence_breakdown)
        breakdown_data = json.loads(confidence_breakdown)
        self.assertIn('overall_confidence', breakdown_data)

    def test_new_enrichment_fields(self):
        """Test that new enrichment fields are properly stored and retrieved."""
        # Add some enrichment data
        self.destination.dominant_religions = ["Christianity", "Islam"]
        self.destination.unesco_sites = ["The Crystal Spire", "Ancient Library"]
        
        # Store and retrieve
        self.db_manager.store_destination(self.destination)
        loaded_destination = self.db_manager.get_destination_by_name("Testopolis")
        
        # Verify enrichment fields
        self.assertEqual(loaded_destination.dominant_religions, self.destination.dominant_religions)
        self.assertEqual(loaded_destination.unesco_sites, self.destination.unesco_sites)
        self.assertEqual(loaded_destination.annual_tourist_arrivals, self.destination.annual_tourist_arrivals)
        self.assertEqual(loaded_destination.popularity_stage, self.destination.popularity_stage)

if __name__ == '__main__':
    unittest.main() 