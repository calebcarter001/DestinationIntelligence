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
            CREATE INDEX IF NOT EXISTS idx_insights_destination_category 
            ON destination_insights(destination_name, category)
            """,
            """
            CREATE INDEX IF NOT EXISTS idx_insights_confidence 
            ON destination_insights(confidence_score DESC)
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
    
    async def _arun(self, destination_name: str, enhanced_analysis: EnhancedDestinationAnalysis) -> str:
        """Store enhanced destination analysis in database"""
        
        logger.info(f"[EnhancedStorage] Storing enhanced analysis for {destination_name}")
        
        try:
            # Use the connection directly from DatabaseManager
            if not self._db_manager.conn:
                return f"Error: DatabaseManager connection is None for {destination_name}"
            
            cursor = self._db_manager.conn.cursor()
            
            analysis_timestamp = datetime.now()
            
            # Count insights by category
            insights_counts = {
                'attractions': len(enhanced_analysis.attractions),
                'hotels': len(enhanced_analysis.hotels),
                'restaurants': len(enhanced_analysis.restaurants),
                'activities': len(enhanced_analysis.activities),
                'neighborhoods': len(enhanced_analysis.neighborhoods),
                'practical_info': len(enhanced_analysis.practical_info)
            }
            
            # Insert main analysis record
            cursor.execute("""
                INSERT OR REPLACE INTO enhanced_destination_analysis 
                (destination_name, summary, analysis_timestamp, total_attractions, total_hotels, 
                 total_restaurants, total_activities, total_neighborhoods, total_practical_info)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                destination_name,
                enhanced_analysis.summary,
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
            all_insights.extend(enhanced_analysis.attractions)
            all_insights.extend(enhanced_analysis.hotels)
            all_insights.extend(enhanced_analysis.restaurants)
            all_insights.extend(enhanced_analysis.activities)
            all_insights.extend(enhanced_analysis.neighborhoods)
            all_insights.extend(enhanced_analysis.practical_info)
            
            insights_inserted = 0
            for insight in all_insights:
                # Convert lists and dicts to JSON strings for storage
                import json
                highlights_json = json.dumps(insight.highlights) if insight.highlights else "[]"
                practical_info_json = json.dumps(insight.practical_info) if insight.practical_info else "{}"
                
                cursor.execute("""
                    INSERT INTO destination_insights 
                    (destination_name, category, insight_name, description, highlights, 
                     practical_info, source_evidence, confidence_score, analysis_timestamp)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    destination_name,
                    insight.category,
                    insight.name,
                    insight.description,
                    highlights_json,
                    practical_info_json,
                    insight.source_evidence,
                    insight.confidence_score,
                    analysis_timestamp
                ))
                insights_inserted += 1
            
            self._db_manager.conn.commit()
            
            success_message = (
                f"Enhanced analysis stored successfully for {destination_name}. "
                f"Stored {insights_inserted} detailed insights across {len(insights_counts)} categories: "
                f"{insights_counts['attractions']} attractions, {insights_counts['hotels']} hotels, "
                f"{insights_counts['restaurants']} restaurants, {insights_counts['activities']} activities, "
                f"{insights_counts['neighborhoods']} neighborhoods, {insights_counts['practical_info']} practical info items."
            )
            
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
    
    def _run(self, destination_name: str, enhanced_analysis: EnhancedDestinationAnalysis) -> str:
        import asyncio
        return asyncio.run(self._arun(destination_name, enhanced_analysis))
    
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