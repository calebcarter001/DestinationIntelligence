#!/usr/bin/env python3

import asyncio
import sys
import os
import json
from datetime import datetime
from typing import Dict, List, Any
import time
import sqlite3
import logging
import pytest
import shutil

# Add project root to path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.tools.enhanced_theme_analysis_tool import EnhancedThemeAnalysisTool, EnhancedThemeAnalysisInput
from src.core.json_export_manager import JsonExportManager
from src.core.enhanced_database_manager import EnhancedDatabaseManager
from src.core.enhanced_data_models import Destination, Theme, Evidence
from src.core.evidence_hierarchy import SourceCategory, EvidenceType

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Test database path
TEST_DB_PATH = "test_enhanced_destination_intelligence.db"
# Test output directory
TEST_OUTPUT_DIR = "test_destination_insights"

class EnhancedFieldsValidator:
    """Comprehensive validator for enhanced fields throughout the pipeline"""
    
    def __init__(self):
        self.test_results = {
            "analysis_fields": {"passed": 0, "failed": 0, "details": []},
            "json_export_fields": {"passed": 0, "failed": 0, "details": []},
            "database_fields": {"passed": 0, "failed": 0, "details": []},
            "output_files": {"passed": 0, "failed": 0, "details": []}
        }
        
    def log_result(self, category: str, test_name: str, passed: bool, details: str = ""):
        """Log test result"""
        if passed:
            self.test_results[category]["passed"] += 1
            status = "✅ PASS"
        else:
            self.test_results[category]["failed"] += 1
            status = "❌ FAIL"
            
        self.test_results[category]["details"].append({
            "test": test_name,
            "status": status,
            "details": details
        })
        print(f"{status}: {test_name} - {details}")
    
    def print_summary(self):
        """Print comprehensive test summary"""
        print("\n" + "="*80)
        print("🧪 COMPREHENSIVE ENHANCED FIELDS TEST SUMMARY")
        print("="*80)
        
        total_passed = 0
        total_failed = 0
        
        for category, results in self.test_results.items():
            passed = results["passed"]
            failed = results["failed"]
            total = passed + failed
            
            total_passed += passed
            total_failed += failed
            
            print(f"\n📊 {category.upper().replace('_', ' ')}:")
            print(f"   ✅ Passed: {passed}/{total}")
            print(f"   ❌ Failed: {failed}/{total}")
            
            if failed > 0:
                print(f"   🔍 Failed Tests:")
                for detail in results["details"]:
                    if "❌" in detail["status"]:
                        print(f"      - {detail['test']}: {detail['details']}")
        
        print(f"\n🎯 OVERALL RESULTS:")
        print(f"   ✅ Total Passed: {total_passed}")
        print(f"   ❌ Total Failed: {total_failed}")
        print(f"   📈 Success Rate: {(total_passed/(total_passed+total_failed)*100):.1f}%")
        
        if total_failed == 0:
            print(f"\n🎉 ALL TESTS PASSED! Enhanced fields are working correctly.")
        else:
            print(f"\n⚠️  {total_failed} tests failed. Enhanced fields need attention.")
        
        print("="*80)

    def test_database_storage_fields(self, destination_name: str, country_code: str) -> bool:
        """Test database storage of enhanced fields"""
        try:
            db_manager = EnhancedDatabaseManager()
            
            # Check if the database manager has enhanced storage capabilities
            has_enhanced_storage = hasattr(db_manager, 'store_destination')
            self.log_result(
                "database_fields",
                "Enhanced Storage Methods",
                has_enhanced_storage,
                f"Database manager has enhanced storage capability: {has_enhanced_storage}"
            )
            
            # Check if enhanced theme storage exists  
            has_enhanced_theme_storage = hasattr(db_manager, '_store_theme')
            self.log_result(
                "database_fields", 
                "Enhanced Theme Storage",
                has_enhanced_theme_storage,
                f"Database manager has enhanced theme storage: {has_enhanced_theme_storage}"
            )
            
            return has_enhanced_storage and has_enhanced_theme_storage
            
        except Exception as e:
            self.log_result(
                "database_fields",
                "Database Storage Test",
                False,
                f"Database storage test failed: {str(e)}"
            )
            return False

    def test_output_files_validation(self, destination_name: str, country_code: str) -> bool:
        """Test that output files exist and have enhanced content"""
        try:
            # Look for recent output files
            insights_dir = "destination_insights"
            recent_files = []
            
            if os.path.exists(insights_dir):
                for root, dirs, files in os.walk(insights_dir):
                    for file in files:
                        if file.endswith('.json') and 'bali_indonesia' in file:
                            file_path = os.path.join(root, file)
                            # Check if file was created recently (last hour)
                            file_time = os.path.getmtime(file_path)
                            if time.time() - file_time < 3600:  # 1 hour
                                recent_files.append(file)
            
            has_recent_files = len(recent_files) > 0
            self.log_result(
                "output_files",
                "Recent Output Files Found",
                has_recent_files,
                f"Found {len(recent_files)} recent files: {recent_files[:3]}"  # Show first 3
            )
            
            return has_recent_files
            
        except Exception as e:
            self.log_result(
                "output_files",
                "Output Files Validation",
                False,
                f"Output files validation failed: {str(e)}"
            )
            return False

class EnhancedFieldsDebugger:
    def __init__(self, db_path: str = "enhanced_destination_intelligence.db"):
        self.db_path = db_path
        self.conn = None
        
    def connect(self):
        """Connect to the database"""
        try:
            self.conn = sqlite3.connect(self.db_path)
            logger.info(f"Connected to database: {self.db_path}")
        except sqlite3.Error as e:
            logger.error(f"Failed to connect to database: {e}")
            raise
            
    def check_evidence_fields(self) -> Dict[str, Any]:
        """Check enhanced fields in evidence table"""
        if not self.conn:
            self.connect()
            
        cursor = self.conn.cursor()
        results = {
            "total_evidence": 0,
            "fields": {
                "sentiment": {"null": 0, "not_null": 0},
                "cultural_context": {"empty": 0, "populated": 0},
                "relationships": {"empty": 0, "populated": 0},
                "agent_id": {"null": 0, "not_null": 0},
                "published_date": {"null": 0, "not_null": 0}
            },
            "sample_populated": {}
        }
        
        # Get total count
        cursor.execute("SELECT COUNT(*) FROM evidence")
        results["total_evidence"] = cursor.fetchone()[0]
        
        # Check each field
        cursor.execute("""
            SELECT 
                sentiment,
                cultural_context,
                relationships,
                agent_id,
                published_date,
                source_url,
                text_snippet
            FROM evidence
        """)
        
        for row in cursor:
            sentiment, cultural_context, relationships, agent_id, published_date, source_url, text_snippet = row
            
            # Check sentiment
            if sentiment is None:
                results["fields"]["sentiment"]["null"] += 1
            else:
                results["fields"]["sentiment"]["not_null"] += 1
                if not results["sample_populated"].get("sentiment"):
                    results["sample_populated"]["sentiment"] = {
                        "value": sentiment,
                        "source_url": source_url,
                        "snippet": text_snippet[:100]
                    }
            
            # Check cultural_context
            if not cultural_context or cultural_context == '{}':
                results["fields"]["cultural_context"]["empty"] += 1
            else:
                results["fields"]["cultural_context"]["populated"] += 1
                if not results["sample_populated"].get("cultural_context"):
                    results["sample_populated"]["cultural_context"] = {
                        "value": json.loads(cultural_context),
                        "source_url": source_url,
                        "snippet": text_snippet[:100]
                    }
            
            # Check relationships
            if not relationships or relationships == '[]':
                results["fields"]["relationships"]["empty"] += 1
            else:
                results["fields"]["relationships"]["populated"] += 1
                if not results["sample_populated"].get("relationships"):
                    results["sample_populated"]["relationships"] = {
                        "value": json.loads(relationships),
                        "source_url": source_url,
                        "snippet": text_snippet[:100]
                    }
            
            # Check agent_id
            if agent_id is None:
                results["fields"]["agent_id"]["null"] += 1
            else:
                results["fields"]["agent_id"]["not_null"] += 1
                if not results["sample_populated"].get("agent_id"):
                    results["sample_populated"]["agent_id"] = {
                        "value": agent_id,
                        "source_url": source_url,
                        "snippet": text_snippet[:100]
                    }
            
            # Check published_date
            if published_date is None:
                results["fields"]["published_date"]["null"] += 1
            else:
                results["fields"]["published_date"]["not_null"] += 1
                if not results["sample_populated"].get("published_date"):
                    results["sample_populated"]["published_date"] = {
                        "value": published_date,
                        "source_url": source_url,
                        "snippet": text_snippet[:100]
                    }
        
        return results

    def check_theme_evidence_links(self) -> Dict[str, Any]:
        """Check theme-evidence relationships"""
        if not self.conn:
            self.connect()
            
        cursor = self.conn.cursor()
        results = {
            "total_themes": 0,
            "total_theme_evidence_links": 0,
            "themes_without_evidence": 0,
            "evidence_without_theme": 0,
            "sample_theme": None
        }
        
        # Get total themes
        cursor.execute("SELECT COUNT(*) FROM themes")
        results["total_themes"] = cursor.fetchone()[0]
        
        # Get total theme-evidence links
        cursor.execute("SELECT COUNT(*) FROM theme_evidence")
        results["total_theme_evidence_links"] = cursor.fetchone()[0]
        
        # Check themes without evidence
        cursor.execute("""
            SELECT t.theme_id, t.name, t.macro_category
            FROM themes t
            LEFT JOIN theme_evidence te ON t.theme_id = te.theme_id
            WHERE te.evidence_id IS NULL
        """)
        themes_without_evidence = cursor.fetchall()
        results["themes_without_evidence"] = len(themes_without_evidence)
        if themes_without_evidence:
            results["sample_theme_without_evidence"] = {
                "theme_id": themes_without_evidence[0][0],
                "name": themes_without_evidence[0][1],
                "macro_category": themes_without_evidence[0][2]
            }
        
        # Check evidence without theme
        cursor.execute("""
            SELECT e.id, e.source_url
            FROM evidence e
            LEFT JOIN theme_evidence te ON e.id = te.evidence_id
            WHERE te.theme_id IS NULL
        """)
        evidence_without_theme = cursor.fetchall()
        results["evidence_without_theme"] = len(evidence_without_theme)
        if evidence_without_theme:
            results["sample_evidence_without_theme"] = {
                "evidence_id": evidence_without_theme[0][0],
                "source_url": evidence_without_theme[0][1]
            }
        
        return results

    def close(self):
        """Close database connection"""
        if self.conn:
            self.conn.close()
            logger.info("Closed database connection")

def setup_test_environment():
    """Set up test environment with clean database and output directory"""
    # Remove test database if it exists
    if os.path.exists(TEST_DB_PATH):
        os.remove(TEST_DB_PATH)
    
    # Remove and recreate test output directory
    if os.path.exists(TEST_OUTPUT_DIR):
        shutil.rmtree(TEST_OUTPUT_DIR)
    os.makedirs(TEST_OUTPUT_DIR)
    
    # Create subdirectories
    for subdir in ['evidence', 'themes', 'full_insights']:
        os.makedirs(os.path.join(TEST_OUTPUT_DIR, subdir))

def create_test_data():
    """Create test destination with themes and evidence"""
    # Create test evidence
    evidence1 = Evidence(
        id="test-evidence-1",
        source_url="https://example.com/test1",
        source_category=SourceCategory.BLOG,
        evidence_type=EvidenceType.TERTIARY,
        authority_weight=0.8,
        text_snippet="Beautiful beaches and vibrant culture",
        timestamp=datetime.now(),
        confidence=0.9,
        sentiment=0.8,
        cultural_context={
            "is_local_source": True,
            "local_entities": ["Kuta Beach", "Tanah Lot"],
            "content_type": "experience"
        },
        relationships=[{
            "target_id": "test-evidence-2",
            "relationship_type": "thematic_similarity",
            "strength": "high"
        }],
        agent_id="test_agent_123",
        published_date=datetime.now()
    )
    
    evidence2 = Evidence(
        id="test-evidence-2",
        source_url="https://example.com/test2",
        source_category=SourceCategory.GUIDEBOOK,
        evidence_type=EvidenceType.SECONDARY,
        authority_weight=0.9,
        text_snippet="Rich cultural heritage and traditional ceremonies",
        timestamp=datetime.now(),
        confidence=0.85,
        sentiment=0.7,
        cultural_context={
            "is_local_source": True,
            "local_entities": ["Ubud", "Uluwatu Temple"],
            "content_type": "guide"
        },
        relationships=[{
            "target_id": "test-evidence-1",
            "relationship_type": "thematic_similarity",
            "strength": "high"
        }],
        agent_id="test_agent_123",
        published_date=datetime.now()
    )
    
    # Create test theme
    theme = Theme(
        theme_id="test-theme-1",
        macro_category="Culture & Heritage",
        micro_category="Local Traditions",
        name="Cultural Experiences",
        description="Rich cultural experiences and traditional ceremonies",
        fit_score=0.9,
        evidence=[evidence1, evidence2],
        tags=["culture", "tradition", "ceremony"],
        created_date=datetime.now(),
        factors={
            "source_diversity": 2,
            "authority_distribution": {"high_authority_ratio": 0.85},
            "temporal_relevance": 0.9
        },
        cultural_summary={
            "total_sources": 2,
            "local_sources": 2,
            "cultural_balance": "local-heavy",
            "key_cultural_elements": ["ceremonies", "temples", "traditions"]
        },
        sentiment_analysis={
            "overall": "very positive",
            "average_score": 0.75,
            "confidence": 0.87,
            "distribution": {"positive": 0.8, "neutral": 0.2, "negative": 0}
        },
        temporal_analysis={
            "evidence_span_days": 30,
            "seasonal_indicators": ["year-round"],
            "temporal_confidence": 0.8
        }
    )
    
    # Create test destination
    destination = Destination(
        id="bali-indonesia",
        names=["Bali"],
        admin_levels={"country": "Indonesia", "state": "Bali"},
        timezone="Asia/Makassar",
        country_code="ID"
    )
    destination.add_theme(theme)
    
    return destination

@pytest.mark.asyncio
async def test_database_persistence_and_export():
    """Test database persistence and JSON export with actual data"""
    print("\n🗄️ Testing Database Persistence and JSON Export")
    print("-" * 60)
    
    # Set up clean test environment
    setup_test_environment()
    
    # Create test data
    test_destination = create_test_data()
    
    try:
        # Initialize database manager with test database
        db_manager = EnhancedDatabaseManager(db_path=TEST_DB_PATH)
        
        # Store destination data
        print("📥 Storing destination data in database...")
        results = db_manager.store_destination(test_destination)
        
        # Verify database storage
        print("🔍 Verifying database storage...")
        conn = sqlite3.connect(TEST_DB_PATH)
        cursor = conn.cursor()
        
        # Check destination table
        cursor.execute("SELECT * FROM destinations WHERE id = ?", (test_destination.id,))
        dest_row = cursor.fetchone()
        assert dest_row is not None, "Destination not found in database"
        print("✅ Destination stored successfully")
        
        # Check themes table
        cursor.execute("SELECT * FROM themes")
        theme_rows = cursor.fetchall()
        assert len(theme_rows) > 0, "No themes found in database"
        print(f"✅ Found {len(theme_rows)} theme(s) in database")
        
        # Check evidence table
        cursor.execute("SELECT * FROM evidence")
        evidence_rows = cursor.fetchall()
        assert len(evidence_rows) > 0, "No evidence found in database"
        print(f"✅ Found {len(evidence_rows)} evidence entries in database")
        
        # Verify enhanced fields in evidence
        cursor.execute("""
            SELECT sentiment, cultural_context, relationships, agent_id, published_date 
            FROM evidence LIMIT 1
        """)
        enhanced_fields = cursor.fetchone()
        assert all(field is not None for field in enhanced_fields), "Enhanced fields missing in evidence"
        print("✅ Enhanced fields present in evidence")
        
        # Close database connection
        conn.close()
        
        # Export to JSON
        print("\n📤 Testing JSON export...")
        export_manager = JsonExportManager(export_base_path=TEST_OUTPUT_DIR)
        
        # Export destination data
        json_files = export_manager.export_destination_insights(test_destination)
        
        # Verify JSON files
        expected_files = [
            os.path.join(TEST_OUTPUT_DIR, 'evidence', f'{test_destination.id}_latest.json'),
            os.path.join(TEST_OUTPUT_DIR, 'themes', f'{test_destination.id}_latest.json'),
            os.path.join(TEST_OUTPUT_DIR, 'full_insights', f'{test_destination.id}_latest.json')
        ]
        
        for file_path in expected_files:
            assert os.path.exists(file_path), f"Expected export file not found: {file_path}"
            
            with open(file_path, 'r') as f:
                exported_data = json.load(f)
                assert exported_data is not None, f"Empty export file: {file_path}"
            
            print(f"✅ Verified export file: {os.path.basename(file_path)}")
        
        # Verify enhanced fields in exported JSON
        themes_file = os.path.join(TEST_OUTPUT_DIR, 'themes', f'{test_destination.id}_latest.json')
        with open(themes_file, 'r') as f:
            themes_data = json.load(f)
            
            # Find first theme
            first_theme = None
            for category in themes_data["themes_by_category"].values():
                if isinstance(category, list) and category:
                    first_theme = category[0]
                    break
        
        assert first_theme is not None, "No themes found in export"
        enhanced_theme_fields = ["factors", "cultural_summary", "sentiment_analysis", "temporal_analysis"]
        for field in enhanced_theme_fields:
            assert field in first_theme, f"Enhanced field {field} missing in exported theme"
            assert first_theme[field], f"Enhanced field {field} is empty in exported theme"
        
        print("✅ Enhanced fields present in exported themes")
        
        print("\n🎉 Database persistence and JSON export tests passed!")
        return True
        
    except Exception as e:
        print(f"❌ Error during database/export test: {str(e)}")
        raise

@pytest.mark.asyncio
async def test_comprehensive_enhanced_fields():
    """Run comprehensive test of all enhanced field fixes"""
    
    validator = EnhancedFieldsValidator()
    
    print("🚀 Starting Comprehensive Enhanced Fields Test")
    print("="*60)
    
    # Run database persistence and export test first
    print("\n📊 PHASE 0: Database Persistence and Export Test")
    print("-" * 40)
    
    try:
        db_export_success = await test_database_persistence_and_export()
        validator.log_result(
            "database_fields",
            "Database Persistence and Export",
            db_export_success,
            "Successfully stored and exported test data"
        )
    except Exception as e:
        validator.log_result(
            "database_fields",
            "Database Persistence and Export",
            False,
            f"Failed to store or export test data: {str(e)}"
        )

    # Test destination
    destination_name = "Test City, Test State"
    country_code = "US"
    
    # Sample test content with rich data for analysis
    test_content = [
        {
            "url": "https://example.com/test-guide",
            "content": """
            Test City offers amazing hiking trails and beautiful mountain views. The local culture is vibrant 
            and welcoming to tourists. Many visitors enjoy the affordable cost of living and temperate weather.
            Swimming and water sports are popular in summer, while winter brings excellent skiing opportunities.
            
            Local tip: Visit the downtown area early in the morning to avoid crowds. The coffee shops open at 6am
            and serve the best local roasted coffee. Don't miss the weekly farmers market on Saturdays.
            
            The historic district features beautiful architecture from the 1800s. Several museums showcase
            the region's mining heritage. Outdoor enthusiasts will love the network of bike trails.
            """,
            "title": "Complete Guide to Test City"
        },
        {
            "url": "https://localblog.com/test-city-secrets",
            "content": """
            As a long-time resident, I can tell you the best kept secrets of Test City. The sunset views from
            Eagle Peak are absolutely stunning - perfect for photography. The local brewery district has
            expanded significantly with 5 new craft breweries opening this year.
            
            For families, the Science Discovery Center is a must-visit, especially with young children.
            The interactive exhibits are both fun and educational. Pro tip: Get there right when they open
            to beat the school groups.
            
            Warning: Parking downtown can be challenging during summer festivals. Use the park-and-ride
            system instead. The public transportation is excellent and runs every 15 minutes.
            """,
            "title": "Local Secrets: Test City Insider Tips"
        }
    ]
    
    print(f"📝 Testing with {len(test_content)} content pieces for {destination_name}")
    
    # ====================
    # TEST 1: Enhanced Analysis Fields
    # ====================
    print(f"\n🔬 PHASE 1: Enhanced Analysis Fields")
    print("-" * 40)
    
    try:
        analyzer = EnhancedThemeAnalysisTool()
        
        input_data = EnhancedThemeAnalysisInput(
            destination_name=destination_name,
            country_code=country_code,
            text_content_list=test_content,
            analyze_temporal=True,
            min_confidence=0.3  # Lower threshold for testing
        )
        
        result = await analyzer.analyze_themes(input_data)
        
        # Test 1.1: Basic result structure
        validator.log_result(
            "analysis_fields", 
            "Basic Result Structure",
            all(key in result for key in ["themes", "evidence_summary", "quality_metrics"]),
            f"Result keys: {list(result.keys())}"
        )
        
        # Test 1.2: Themes have enhanced fields
        themes = result.get("themes", [])
        validator.log_result(
            "analysis_fields",
            "Themes Generated",
            len(themes) > 0,
            f"Generated {len(themes)} themes"
        )
        
        if themes:
            first_theme = themes[0]
            
            # Test enhanced theme fields
            enhanced_theme_fields = ["factors", "cultural_summary", "sentiment_analysis", "temporal_analysis"]
            for field in enhanced_theme_fields:
                has_field = field in first_theme
                is_populated = has_field and first_theme[field] is not None and first_theme[field] != {}
                validator.log_result(
                    "analysis_fields",
                    f"Theme.{field} populated",
                    is_populated,
                    f"Value: {type(first_theme.get(field))} - {str(first_theme.get(field))[:100]}..."
                )
            
            # Test evidence has enhanced fields
            evidence_list = first_theme.get("evidence_summary", [])
            validator.log_result(
                "analysis_fields",
                "Evidence Generated",
                len(evidence_list) > 0,
                f"Generated {len(evidence_list)} evidence pieces"
            )
            
            if evidence_list:
                first_evidence = evidence_list[0]
                enhanced_evidence_fields = ["sentiment", "published_date", "relationships", "agent_id", "cultural_context"]
                for field in enhanced_evidence_fields:
                    has_field = field in first_evidence
                    is_populated = has_field and first_evidence[field] is not None
                    validator.log_result(
                        "analysis_fields",
                        f"Evidence.{field} populated",
                        is_populated,
                        f"Value: {type(first_evidence.get(field))} - {str(first_evidence.get(field))[:100]}..."
                    )
        
    except Exception as e:
        validator.log_result("analysis_fields", "Analysis Execution", False, f"Error: {str(e)}")
    
    # ====================
    # TEST 2: JSON Export Fields
    # ====================
    print(f"\n📄 PHASE 2: JSON Export Fields")
    print("-" * 40)
    
    try:
        # Create test evidence with enhanced fields first
        test_evidence = Evidence(
            id="test-evidence-1",
            source_url="https://example.com/test",
            source_category=SourceCategory.BLOG,
            evidence_type=EvidenceType.TERTIARY,
            authority_weight=0.8,
            text_snippet="Test evidence content with enhanced fields",
            timestamp=datetime.now(),
            confidence=0.8,
            sentiment=0.6,  # Populated sentiment
            cultural_context={  # Populated cultural context
                "is_local_source": True,
                "local_entities": ["Test Attraction"],
                "content_type": "experience"
            },
            relationships=[{  # Populated relationships
                "target_id": "test-evidence-2",
                "relationship_type": "thematic_similarity",
                "strength": "medium"
            }],
            agent_id="test_agent_123",  # Populated agent ID
            published_date=datetime.now()  # Populated published date
        )
        
        # Create test theme with enhanced fields
        test_theme = Theme(
            theme_id="test-theme-1",
            macro_category="Test Category",
            micro_category="Test Subcategory",
            name="Test Theme",
            description="Test theme with enhanced fields",
            fit_score=0.8,
            evidence=[test_evidence],
            tags=["test", "enhanced"],
            created_date=datetime.now(),
            factors={  # Populated factors
                "source_diversity": 3,
                "authority_distribution": {"high_authority_ratio": 0.8}
            },
            cultural_summary={  # Populated cultural summary
                "total_sources": 1,
                "local_sources": 1,
                "cultural_balance": "local-heavy"
            },
            sentiment_analysis={  # Populated sentiment analysis
                "overall": "positive",
                "average_score": 0.6,
                "confidence": 0.8
            },
            temporal_analysis={  # Populated temporal analysis
                "evidence_span_days": 30,
                "seasonal_indicators": ["summer"]
            }
        )
        
        # Create a test destination object for export
        destination = Destination(
            id="test-destination-id",
            names=["Test City"],
            admin_levels={"country": "Test Country", "state": "Test State"},
            timezone="UTC",
            country_code="TC"
        )
        
        # Add test theme to destination
        destination.add_theme(test_theme)
        
        # Test JSON export manager
        export_manager = JsonExportManager()
        
        # Test evidence export
        evidence_export = export_manager._create_evidence_export(destination)
        validator.log_result(
            "json_export_fields",
            "Evidence Export Structure",
            "all_evidence" in evidence_export and len(evidence_export["all_evidence"]) > 0,
            f"Evidence export keys: {list(evidence_export.keys())}"
        )
        
        if evidence_export.get("all_evidence"):
            exported_evidence = evidence_export["all_evidence"][0]
            enhanced_evidence_fields = ["sentiment", "published_date", "relationships", "agent_id", "cultural_context"]
            for field in enhanced_evidence_fields:
                has_field = field in exported_evidence
                is_populated = has_field and exported_evidence[field] is not None
                validator.log_result(
                    "json_export_fields",
                    f"Exported Evidence.{field}",
                    is_populated,
                    f"Value: {exported_evidence.get(field)}"
                )
        
        # Test themes export
        themes_export = export_manager._create_themes_export(destination)
        validator.log_result(
            "json_export_fields",
            "Themes Export Structure",
            "themes_by_category" in themes_export and len(themes_export["themes_by_category"]) > 0,
            f"Themes export keys: {list(themes_export.keys())}"
        )
        
        if themes_export.get("themes_by_category"):
            # Find first theme in any category
            first_theme = None
            for category, themes_list in themes_export["themes_by_category"].items():
                if isinstance(themes_list, list) and themes_list:
                    first_theme = themes_list[0]
                    break
            
            if first_theme:
                enhanced_theme_fields = ["factors", "cultural_summary", "sentiment_analysis", "temporal_analysis"]
                for field in enhanced_theme_fields:
                    has_field = field in first_theme
                    is_populated = has_field and first_theme[field] is not None and first_theme[field] != {}
                    validator.log_result(
                        "json_export_fields",
                        f"Exported Theme.{field}",
                        is_populated,
                        f"Value: {type(first_theme.get(field))} - {str(first_theme.get(field))[:100]}..."
                    )
        
    except Exception as e:
        validator.log_result("json_export_fields", "JSON Export Execution", False, f"Error: {str(e)}")
    
    # ====================
    # TEST 3: Database Storage Fields
    # ====================
    print(f"\n🗄️ PHASE 3: Database Storage Fields")
    print("-" * 40)
    
    try:
        db_manager = EnhancedDatabaseManager()
        
        # Check if the database manager has enhanced storage capabilities
        has_enhanced_storage = hasattr(db_manager, 'store_destination')
        validator.log_result(
            "database_fields",
            "Enhanced Storage Methods",
            has_enhanced_storage,
            f"Database manager has enhanced storage capability: {has_enhanced_storage}"
        )
        
        # Check if enhanced theme storage exists  
        has_enhanced_theme_storage = hasattr(db_manager, '_store_theme')
        validator.log_result(
            "database_fields", 
            "Enhanced Theme Storage",
            has_enhanced_theme_storage,
            f"Database manager has enhanced theme storage: {has_enhanced_theme_storage}"
        )
        
    except Exception as e:
        validator.log_result("database_fields", "Database Storage Check", False, f"Error: {str(e)}")
    
    # ====================
    # TEST 4: Output Files Validation
    # ====================
    print(f"\n📁 PHASE 4: Output Files Validation")
    print("-" * 40)
    
    try:
        # Look for recent output files
        output_dir = "destination_insights"
        subdirs = ["evidence", "themes", "full_insights"]
        
        recent_files = []
        for subdir in subdirs:
            subdir_path = os.path.join(output_dir, subdir)
            if os.path.exists(subdir_path):
                files = [f for f in os.listdir(subdir_path) if f.endswith('.json')]
                # Get most recent file
                if files:
                    files.sort(reverse=True)  # Assuming filename has timestamp
                    recent_files.append(os.path.join(subdir_path, files[0]))
        
        validator.log_result(
            "output_files",
            "Recent Output Files Found",
            len(recent_files) > 0,
            f"Found {len(recent_files)} recent files: {[os.path.basename(f) for f in recent_files]}"
        )
        
        # Test each output file for enhanced fields
        for file_path in recent_files:
            try:
                with open(file_path, 'r') as f:
                    data = json.load(f)
                
                file_name = os.path.basename(file_path)
                
                if "evidence" in file_name:
                    # Test evidence file
                    evidence_items = data.get("evidence", [])
                    if evidence_items:
                        first_evidence = evidence_items[0]
                        enhanced_fields = ["sentiment", "published_date", "relationships", "agent_id"]
                        populated_fields = [f for f in enhanced_fields if f in first_evidence and first_evidence[f] is not None]
                        validator.log_result(
                            "output_files",
                            f"Evidence File Enhanced Fields ({file_name})",
                            len(populated_fields) >= 2,  # At least 2 enhanced fields should be populated
                            f"Populated fields: {populated_fields}"
                        )
                
                elif "themes" in file_name:
                    # Test themes file
                    themes_data = data
                    if themes_data:
                        # Find first theme in any category
                        first_theme = None
                        for category, themes_list in themes_data.items():
                            if isinstance(themes_list, list) and themes_list:
                                first_theme = themes_list[0]
                                break
                        
                        if first_theme:
                            enhanced_fields = ["factors", "cultural_summary", "sentiment_analysis", "temporal_analysis"]
                            populated_fields = [f for f in enhanced_fields if f in first_theme and first_theme[f] is not None and first_theme[f] != {}]
                            validator.log_result(
                                "output_files",
                                f"Themes File Enhanced Fields ({file_name})",
                                len(populated_fields) >= 2,  # At least 2 enhanced fields should be populated
                                f"Populated fields: {populated_fields}"
                            )
                
                elif "full" in file_name:
                    # Test full insights file
                    if "themes" in data:
                        themes_list = data["themes"]
                        if themes_list:
                            first_theme = themes_list[0]
                            enhanced_fields = ["factors", "cultural_summary", "sentiment_analysis", "temporal_analysis"]
                            populated_fields = [f for f in enhanced_fields if f in first_theme and first_theme[f] is not None and first_theme[f] != {}]
                            validator.log_result(
                                "output_files",
                                f"Full Insights Enhanced Fields ({file_name})",
                                len(populated_fields) >= 2,
                                f"Populated fields: {populated_fields}"
                            )
                
            except Exception as e:
                validator.log_result("output_files", f"File Analysis ({file_name})", False, f"Error reading file: {str(e)}")
    
    except Exception as e:
        validator.log_result("output_files", "Output Files Check", False, f"Error: {str(e)}")
    
    # Print comprehensive summary
    validator.print_summary()
    
    return validator.test_results

def main():
    debugger = EnhancedFieldsDebugger()
    try:
        # Check evidence fields
        evidence_results = debugger.check_evidence_fields()
        logger.info("\n=== Evidence Fields Analysis ===")
        logger.info(f"Total evidence entries: {evidence_results['total_evidence']}")
        for field, counts in evidence_results["fields"].items():
            logger.info(f"\n{field.upper()} Analysis:")
            for status, count in counts.items():
                percentage = (count / evidence_results["total_evidence"]) * 100
                logger.info(f"  {status}: {count} ({percentage:.2f}%)")
            if field in evidence_results["sample_populated"] and evidence_results["sample_populated"][field]:
                sample = evidence_results["sample_populated"][field]
                logger.info(f"  Sample populated {field}:")
                logger.info(f"    Value: {sample['value']}")
                logger.info(f"    Source: {sample['source_url']}")
                logger.info(f"    Snippet: {sample['snippet']}")
        
        # Check theme-evidence relationships
        relationship_results = debugger.check_theme_evidence_links()
        logger.info("\n=== Theme-Evidence Relationship Analysis ===")
        logger.info(f"Total themes: {relationship_results['total_themes']}")
        logger.info(f"Total theme-evidence links: {relationship_results['total_theme_evidence_links']}")
        logger.info(f"Themes without evidence: {relationship_results['themes_without_evidence']}")
        logger.info(f"Evidence without theme: {relationship_results['evidence_without_theme']}")
        
        if relationship_results.get("sample_theme_without_evidence"):
            sample = relationship_results["sample_theme_without_evidence"]
            logger.info("\nSample theme without evidence:")
            logger.info(f"  Theme ID: {sample['theme_id']}")
            logger.info(f"  Name: {sample['name']}")
            logger.info(f"  Category: {sample['macro_category']}")
            
        if relationship_results.get("sample_evidence_without_theme"):
            sample = relationship_results["sample_evidence_without_theme"]
            logger.info("\nSample evidence without theme:")
            logger.info(f"  Evidence ID: {sample['evidence_id']}")
            logger.info(f"  Source: {sample['source_url']}")
            
    finally:
        debugger.close()

if __name__ == "__main__":
    print("🧪 Enhanced Fields Comprehensive Test Suite")
    print("=" * 60)
    print("This test validates all enhanced field fixes:")
    print("• Enhanced analysis populates all fields correctly")
    print("• JSON export manager includes all enhanced fields") 
    print("• Database storage preserves enhanced data")
    print("• Output files contain non-null enhanced values")
    print("=" * 60)
    
    try:
        results = asyncio.run(test_comprehensive_enhanced_fields())
        
        # Return appropriate exit code
        total_failed = sum(category["failed"] for category in results.values())
        if total_failed == 0:
            print("\n🎉 All tests passed! Enhanced fields are working correctly.")
            sys.exit(0)
        else:
            print(f"\n⚠️ {total_failed} tests failed. Enhanced fields need attention.")
            sys.exit(1)
            
    except Exception as e:
        print(f"\n❌ Test suite failed with error: {e}")
        sys.exit(1)

    main() 