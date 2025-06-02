import sqlite3
import logging
import json
from datetime import datetime
from typing import Optional
import os

# Assuming DestinationInsight will be imported from data_models.py if needed
# from ..data_models import DestinationInsight # Corrected relative import for core module

class DatabaseManager:
    """Store real insights and evidence"""
    
    def __init__(self, db_path: str = "real_destination_intelligence.db"):
        # Ensure db_path is in the project root, not src/core
        project_root = os.path.join(os.path.dirname(__file__), '..', '..') # Go up two levels from src/core
        self.db_path = os.path.join(project_root, db_path)
        self.conn = None
        self.logger = logging.getLogger(__name__ + '.DatabaseManager')
        self.init_database()
        
    def init_database(self):
        """Initialize database for real data"""
        try:
            self.conn = sqlite3.connect(self.db_path) 
            cursor = self.conn.cursor()
            
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS real_insights (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    destination_name TEXT NOT NULL,
                    insight_type TEXT NOT NULL,
                    insight_name TEXT NOT NULL,
                    description TEXT,
                    confidence_score REAL,
                    evidence_urls TEXT,
                    content_snippets TEXT,
                    is_discovered_theme BOOLEAN,
                    created_date TIMESTAMP,
                    UNIQUE(destination_name, insight_type, insight_name)
                )
            """)
            
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS source_analysis_log (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    destination_name TEXT NOT NULL,
                    source_url TEXT NOT NULL,
                    source_title TEXT,
                    content_length INTEGER,
                    processed_date TIMESTAMP,
                    status TEXT
                )
            """)
            
            self.conn.commit()
            self.logger.info(f"Real data database initialized/connected: {self.db_path}")
        except sqlite3.Error as e:
            self.logger.error(f"Database initialization error: {e}")
            if self.conn:
                self.conn.close()
            raise 

    def log_source_processing(self, destination: str, source_url: str, source_title: Optional[str], content_length: Optional[int], status: str):
        if not self.conn: return
        try:
            cursor = self.conn.cursor()
            cursor.execute("""
                INSERT INTO source_analysis_log (destination_name, source_url, source_title, content_length, processed_date, status)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (destination, source_url, source_title, content_length, datetime.now().isoformat(), status))
            self.conn.commit()
        except sqlite3.Error as e:
            self.logger.error(f"Error logging source processing for {source_url}: {e}")

    # Type hint for insight would be DestinationInsight if imported from ..data_models
    def store_real_insight(self, destination: str, insight):
        """Store insight with real evidence"""
        if not self.conn:
            self.logger.error("Database connection not available. Cannot store insight.")
            return
        try:
            cursor = self.conn.cursor()
            cursor.execute("""
                INSERT OR REPLACE INTO real_insights 
                (destination_name, insight_type, insight_name, description, 
                 confidence_score, evidence_urls, content_snippets, is_discovered_theme, created_date)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                destination,
                insight.insight_type,
                insight.insight_name,
                insight.description,
                insight.confidence_score,
                json.dumps(insight.evidence_sources),
                json.dumps(insight.content_snippets),
                insight.is_discovered_theme,
                insight.created_date
            ))
            self.conn.commit()
        except sqlite3.Error as e:
            self.logger.error(f"Database error storing insight for {destination} - {insight.insight_name}: {e}")

    def close_db(self):
        if self.conn:
            self.conn.close()
            self.logger.info("Database connection closed.") 