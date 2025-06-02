import logging
import sqlite3
from typing import List, Type, Dict, Any
from datetime import datetime
from langchain.tools import StructuredTool
from pydantic import BaseModel, Field

from src.core.database_manager import DatabaseManager
from src.tools.enhanced_content_analysis_tool import EnhancedDestinationAnalysis, StructuredDestinationInsight

logger = logging.getLogger(__name__)

class EnhancedStorageInput(BaseModel):
    destination_name: str = Field(description="Name of the destination")
    enhanced_analysis: EnhancedDestinationAnalysis = Field(description="Enhanced destination analysis results")

class EnhancedDatabaseStorageTool(StructuredTool):
    name: str = "store_enhanced_destination_insights"
    description: str = (
        "Stores enhanced destination insights including detailed attractions, hotels, restaurants, "
        "activities, neighborhoods, and practical information in the database."
    )
    args_schema: Type[BaseModel] = EnhancedStorageInput
    
    def __init__(self, db_manager: DatabaseManager, **kwargs):
        super().__init__(**kwargs)
        # Store DB manager as private attribute to avoid Pydantic field conflicts
        self._db_manager = db_manager
        self._ensure_enhanced_tables_exist()
    
    def _ensure_enhanced_tables_exist(self):
        """Create enhanced tables for storing detailed destination insights"""
        
        create_statements = [
            """
            CREATE TABLE IF NOT EXISTS enhanced_destination_analysis (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                destination_name TEXT NOT NULL,
                summary TEXT,
                analysis_timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                total_attractions INTEGER DEFAULT 0,
                total_hotels INTEGER DEFAULT 0,
                total_restaurants INTEGER DEFAULT 0,
                total_activities INTEGER DEFAULT 0,
                total_neighborhoods INTEGER DEFAULT 0,
                total_practical_info INTEGER DEFAULT 0,
                UNIQUE(destination_name, analysis_timestamp)
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS destination_insights (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                destination_name TEXT NOT NULL,
                category TEXT NOT NULL,
                insight_name TEXT NOT NULL,
                description TEXT,
                highlights TEXT, -- JSON string
                practical_info TEXT, -- JSON string
                source_evidence TEXT,
                confidence_score REAL,
                analysis_timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (destination_name) REFERENCES enhanced_destination_analysis(destination_name)
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS priority_metrics (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                destination_name TEXT NOT NULL,
                safety_score REAL,
                crime_index REAL,
                tourist_police_available BOOLEAN,
                emergency_contacts TEXT, -- JSON string
                travel_advisory_level TEXT,
                budget_per_day_low REAL,
                budget_per_day_mid REAL,
                budget_per_day_high REAL,
                currency TEXT,
                required_vaccinations TEXT, -- JSON string
                health_risks TEXT, -- JSON string
                water_safety TEXT,
                medical_facility_quality TEXT,
                visa_required BOOLEAN,
                visa_on_arrival BOOLEAN,
                visa_cost REAL,
                english_proficiency TEXT,
                infrastructure_rating REAL,
                analysis_timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (destination_name) REFERENCES enhanced_destination_analysis(destination_name)
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS priority_insights (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                destination_name TEXT NOT NULL,
                insight_type TEXT NOT NULL,
                insight_name TEXT NOT NULL,
                description TEXT,
                evidence TEXT, -- JSON string
                confidence_score REAL,
                priority_category TEXT,
                priority_impact TEXT,
                temporal_relevance REAL,
                source_urls TEXT, -- JSON string
                analysis_timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (destination_name) REFERENCES enhanced_destination_analysis(destination_name)
            )
            """,
            """
            CREATE INDEX IF NOT EXISTS idx_insights_destination_category 
            ON destination_insights(destination_name, category)
            """,
            """
            CREATE INDEX IF NOT EXISTS idx_insights_confidence 
            ON destination_insights(confidence_score DESC)
            """,
            """
            CREATE INDEX IF NOT EXISTS idx_priority_destination 
            ON priority_metrics(destination_name)
            """,
            """
            CREATE INDEX IF NOT EXISTS idx_priority_insights_destination 
            ON priority_insights(destination_name, priority_category)
            """
        ]
        
        try:
            # Use the connection directly from DatabaseManager
            if not self._db_manager.conn:
                logger.error("[EnhancedStorage] DatabaseManager connection is None")
                return
                
            cursor = self._db_manager.conn.cursor()
            
            for statement in create_statements:
                cursor.execute(statement)
            
            self._db_manager.conn.commit()
            logger.info("[EnhancedStorage] Enhanced database tables ensured")
            
        except sqlite3.Error as e:
            logger.error(f"[EnhancedStorage] Error creating enhanced tables: {e}")
        except Exception as e:
            logger.error(f"[EnhancedStorage] Unexpected error creating tables: {e}")
    
    async def _arun(self, destination_name: str, enhanced_analysis: EnhancedDestinationAnalysis = None, config = None) -> str:
        """Store enhanced destination analysis in database"""
        
        # Handle backward compatibility with destination_data parameter
        if enhanced_analysis is None and hasattr(self, '_last_destination_data'):
            # Try to use cached data from previous call
            enhanced_analysis = self._last_destination_data
            destination_name = getattr(enhanced_analysis, 'destination_name', destination_name)
        
        logger.info(f"[EnhancedStorage] Storing enhanced analysis for {destination_name}")
        
        try:
            # Use the connection directly from DatabaseManager
            if not self._db_manager.conn:
                return f"Error: DatabaseManager connection is None for {destination_name}"
            
            cursor = self._db_manager.conn.cursor()
            
            analysis_timestamp = datetime.now()
            
            # Count insights by category
            insights_counts = {
                'attractions': len(enhanced_analysis.attractions) if hasattr(enhanced_analysis, 'attractions') else 0,
                'hotels': len(enhanced_analysis.hotels) if hasattr(enhanced_analysis, 'hotels') else 0,
                'restaurants': len(enhanced_analysis.restaurants) if hasattr(enhanced_analysis, 'restaurants') else 0,
                'activities': len(enhanced_analysis.activities) if hasattr(enhanced_analysis, 'activities') else 0,
                'neighborhoods': len(enhanced_analysis.neighborhoods) if hasattr(enhanced_analysis, 'neighborhoods') else 0,
                'practical_info': len(enhanced_analysis.practical_info) if hasattr(enhanced_analysis, 'practical_info') else 0
            }
            
            # Insert main analysis record
            cursor.execute("""
                INSERT OR REPLACE INTO enhanced_destination_analysis 
                (destination_name, summary, analysis_timestamp, total_attractions, total_hotels, 
                 total_restaurants, total_activities, total_neighborhoods, total_practical_info)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                destination_name,
                enhanced_analysis.summary if hasattr(enhanced_analysis, 'summary') else '',
                analysis_timestamp,
                insights_counts['attractions'],
                insights_counts['hotels'],
                insights_counts['restaurants'],
                insights_counts['activities'],
                insights_counts['neighborhoods'],
                insights_counts['practical_info']
            ))
            
            # Insert all insights by category
            all_insights = []
            if hasattr(enhanced_analysis, 'attractions'):
                all_insights.extend(enhanced_analysis.attractions)
            if hasattr(enhanced_analysis, 'hotels'):
                all_insights.extend(enhanced_analysis.hotels)
            if hasattr(enhanced_analysis, 'restaurants'):
                all_insights.extend(enhanced_analysis.restaurants)
            if hasattr(enhanced_analysis, 'activities'):
                all_insights.extend(enhanced_analysis.activities)
            if hasattr(enhanced_analysis, 'neighborhoods'):
                all_insights.extend(enhanced_analysis.neighborhoods)
            if hasattr(enhanced_analysis, 'practical_info'):
                all_insights.extend(enhanced_analysis.practical_info)
            
            insights_inserted = 0
            for insight in all_insights:
                # Convert lists and dicts to JSON strings for storage
                import json
                highlights_json = json.dumps(insight.highlights) if hasattr(insight, 'highlights') and insight.highlights else "[]"
                practical_info_json = json.dumps(insight.practical_info) if hasattr(insight, 'practical_info') and insight.practical_info else "{}"
                
                cursor.execute("""
                    INSERT INTO destination_insights 
                    (destination_name, category, insight_name, description, highlights, 
                     practical_info, source_evidence, confidence_score, analysis_timestamp)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    destination_name,
                    insight.category if hasattr(insight, 'category') else 'unknown',
                    insight.name if hasattr(insight, 'name') else 'Unknown',
                    insight.description if hasattr(insight, 'description') else '',
                    highlights_json,
                    practical_info_json,
                    insight.source_evidence if hasattr(insight, 'source_evidence') else '',
                    insight.confidence_score if hasattr(insight, 'confidence_score') else 0.0,
                    analysis_timestamp
                ))
                insights_inserted += 1
            
            # Store priority metrics if available
            priority_metrics_stored = False
            if hasattr(enhanced_analysis, 'priority_metrics') and enhanced_analysis.priority_metrics:
                priority_data = enhanced_analysis.priority_metrics
                import json
                
                cursor.execute("""
                    INSERT OR REPLACE INTO priority_metrics 
                    (destination_name, safety_score, crime_index, tourist_police_available,
                     emergency_contacts, travel_advisory_level, budget_per_day_low,
                     budget_per_day_mid, budget_per_day_high, currency,
                     required_vaccinations, health_risks, water_safety,
                     medical_facility_quality, visa_required, visa_on_arrival,
                     visa_cost, english_proficiency, infrastructure_rating, analysis_timestamp)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    destination_name,
                    priority_data.safety_score if hasattr(priority_data, 'safety_score') else None,
                    priority_data.crime_index if hasattr(priority_data, 'crime_index') else None,
                    priority_data.tourist_police_available if hasattr(priority_data, 'tourist_police_available') else None,
                    json.dumps(priority_data.emergency_contacts) if hasattr(priority_data, 'emergency_contacts') and priority_data.emergency_contacts else "{}",
                    priority_data.travel_advisory_level if hasattr(priority_data, 'travel_advisory_level') else None,
                    priority_data.budget_per_day_low if hasattr(priority_data, 'budget_per_day_low') else None,
                    priority_data.budget_per_day_mid if hasattr(priority_data, 'budget_per_day_mid') else None,
                    priority_data.budget_per_day_high if hasattr(priority_data, 'budget_per_day_high') else None,
                    priority_data.currency if hasattr(priority_data, 'currency') else None,
                    json.dumps(priority_data.required_vaccinations) if hasattr(priority_data, 'required_vaccinations') and priority_data.required_vaccinations else "[]",
                    json.dumps(priority_data.health_risks) if hasattr(priority_data, 'health_risks') and priority_data.health_risks else "[]",
                    priority_data.water_safety if hasattr(priority_data, 'water_safety') else None,
                    priority_data.medical_facility_quality if hasattr(priority_data, 'medical_facility_quality') else None,
                    priority_data.visa_required if hasattr(priority_data, 'visa_required') else None,
                    priority_data.visa_on_arrival if hasattr(priority_data, 'visa_on_arrival') else None,
                    priority_data.visa_cost if hasattr(priority_data, 'visa_cost') else None,
                    priority_data.english_proficiency if hasattr(priority_data, 'english_proficiency') else None,
                    priority_data.infrastructure_rating if hasattr(priority_data, 'infrastructure_rating') else None,
                    analysis_timestamp
                ))
                priority_metrics_stored = True
            
            # Store priority insights if available
            priority_insights_count = 0
            if hasattr(enhanced_analysis, 'priority_insights') and enhanced_analysis.priority_insights:
                for p_insight in enhanced_analysis.priority_insights:
                    import json
                    
                    cursor.execute("""
                        INSERT INTO priority_insights 
                        (destination_name, insight_type, insight_name, description,
                         evidence, confidence_score, priority_category, priority_impact,
                         temporal_relevance, source_urls, analysis_timestamp)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """, (
                        destination_name,
                        p_insight.insight_type if hasattr(p_insight, 'insight_type') else 'Priority Concern',
                        p_insight.insight_name if hasattr(p_insight, 'insight_name') else 'Unknown',
                        p_insight.description if hasattr(p_insight, 'description') else '',
                        json.dumps(p_insight.evidence) if hasattr(p_insight, 'evidence') and p_insight.evidence else "[]",
                        p_insight.confidence_score if hasattr(p_insight, 'confidence_score') else 0.0,
                        p_insight.priority_category if hasattr(p_insight, 'priority_category') else None,
                        p_insight.priority_impact if hasattr(p_insight, 'priority_impact') else None,
                        p_insight.temporal_relevance if hasattr(p_insight, 'temporal_relevance') else None,
                        json.dumps(p_insight.source_urls) if hasattr(p_insight, 'source_urls') and p_insight.source_urls else "[]",
                        analysis_timestamp
                    ))
                    priority_insights_count += 1
            
            self._db_manager.conn.commit()
            
            success_message = (
                f"Enhanced analysis stored successfully for {destination_name}. "
                f"Stored {insights_inserted} detailed insights across {len(insights_counts)} categories: "
                f"{insights_counts['attractions']} attractions, {insights_counts['hotels']} hotels, "
                f"{insights_counts['restaurants']} restaurants, {insights_counts['activities']} activities, "
                f"{insights_counts['neighborhoods']} neighborhoods, {insights_counts['practical_info']} practical info items."
            )
            
            if priority_metrics_stored:
                success_message += f" Priority metrics stored."
            if priority_insights_count > 0:
                success_message += f" {priority_insights_count} priority insights stored."
            
            logger.info(f"[EnhancedStorage] {success_message}")
            return success_message
            
        except sqlite3.Error as e:
            error_msg = f"Database error storing enhanced analysis for {destination_name}: {e}"
            logger.error(f"[EnhancedStorage] {error_msg}")
            return f"Error: {error_msg}"
        except Exception as e:
            error_msg = f"Unexpected error storing enhanced analysis for {destination_name}: {e}"
            logger.error(f"[EnhancedStorage] {error_msg}")
            return f"Error: {error_msg}"
    
    def _run(self, destination_name: str, enhanced_analysis: EnhancedDestinationAnalysis = None, config = None) -> str:
        import asyncio
        return asyncio.run(self._arun(destination_name, enhanced_analysis, config))
    
    def get_stored_insights(self, destination_name: str, category: str = None, min_confidence: float = 0.0) -> List[Dict[str, Any]]:
        """Retrieve stored insights for a destination"""
        
        try:
            if not self._db_manager.conn:
                logger.error("[EnhancedStorage] DatabaseManager connection is None")
                return []
                
            cursor = self._db_manager.conn.cursor()
            
            base_query = """
                SELECT category, insight_name, description, highlights, practical_info, 
                       source_evidence, confidence_score, analysis_timestamp
                FROM destination_insights 
                WHERE destination_name = ? AND confidence_score >= ?
            """
            params = [destination_name, min_confidence]
            
            if category:
                base_query += " AND category = ?"
                params.append(category)
            
            base_query += " ORDER BY confidence_score DESC, category, insight_name"
            
            cursor.execute(base_query, params)
            rows = cursor.fetchall()
            
            insights = []
            for row in rows:
                import json
                insight = {
                    'category': row[0],
                    'name': row[1],
                    'description': row[2],
                    'highlights': json.loads(row[3]) if row[3] else [],
                    'practical_info': json.loads(row[4]) if row[4] else {},
                    'source_evidence': row[5],
                    'confidence_score': row[6],
                    'analysis_timestamp': row[7]
                }
                insights.append(insight)
            
            return insights
            
        except Exception as e:
            logger.error(f"[EnhancedStorage] Error retrieving insights: {e}")
            return [] 