import sqlite3
import logging
import json
from datetime import datetime
from typing import Optional, List, Dict, Any
import os

from .enhanced_data_models import Destination, Theme, Evidence, TemporalSlice, DimensionValue
from .confidence_scoring import ConfidenceBreakdown
from .json_export_manager import JsonExportManager

class EnhancedDatabaseManager:
    """Enhanced database manager supporting comprehensive destination intelligence"""
    
    def __init__(self, db_path: str = "enhanced_destination_intelligence.db", 
                 enable_json_export: bool = True,
                 json_export_path: Optional[str] = None):
        project_root = os.path.join(os.path.dirname(__file__), '..', '..')
        self.db_path = os.path.join(project_root, db_path)
        self.conn = None
        self.logger = logging.getLogger(__name__)
        
        # Initialize JSON export manager if enabled
        self.enable_json_export = enable_json_export
        self.json_export_manager = None
        
        if enable_json_export:
            if json_export_path is None:
                json_export_path = os.path.join(project_root, "destination_insights")
            self.json_export_manager = JsonExportManager(json_export_path)
            self.logger.info(f"JSON export enabled. Files will be saved to: {json_export_path}")
        
        self.init_database()
        
    def init_database(self):
        """Initialize enhanced database schema"""
        try:
            self.conn = sqlite3.connect(self.db_path)
            cursor = self.conn.cursor()
            
            # Destinations table with enhanced fields
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS destinations (
                    id TEXT PRIMARY KEY,
                    names TEXT NOT NULL,  -- JSON array
                    admin_levels TEXT,    -- JSON object
                    timezone TEXT,
                    population INTEGER,
                    country_code TEXT,
                    core_geo TEXT,        -- JSON object
                    lineage TEXT,         -- JSON object
                    meta TEXT,            -- JSON object
                    last_updated TIMESTAMP,
                    destination_revision INTEGER DEFAULT 1
                )
            """)
            
            # Evidence table with full tracking
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS evidence (
                    id TEXT PRIMARY KEY,
                    destination_id TEXT,
                    source_url TEXT NOT NULL,
                    source_category TEXT,
                    evidence_type TEXT,
                    authority_weight REAL,
                    text_snippet TEXT,
                    timestamp TIMESTAMP,
                    confidence REAL,
                    sentiment REAL,
                    cultural_context TEXT,  -- JSON
                    relationships TEXT,     -- JSON array
                    agent_id TEXT,
                    published_date TIMESTAMP,
                    FOREIGN KEY (destination_id) REFERENCES destinations(id)
                )
            """)
            
            # Themes table with confidence breakdown
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS themes (
                    theme_id TEXT PRIMARY KEY,
                    destination_id TEXT,
                    macro_category TEXT,
                    micro_category TEXT,
                    name TEXT NOT NULL,
                    description TEXT,
                    fit_score REAL,
                    confidence_breakdown TEXT,  -- JSON
                    confidence_level TEXT,
                    tags TEXT,                  -- JSON array
                    created_date TIMESTAMP,
                    last_validated TIMESTAMP,
                    FOREIGN KEY (destination_id) REFERENCES destinations(id)
                )
            """)
            
            # Theme-Evidence relationship table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS theme_evidence (
                    theme_id TEXT,
                    evidence_id TEXT,
                    PRIMARY KEY (theme_id, evidence_id),
                    FOREIGN KEY (theme_id) REFERENCES themes(theme_id),
                    FOREIGN KEY (evidence_id) REFERENCES evidence(id)
                )
            """)
            
            # Dimensions table (60-dimension matrix)
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS dimensions (
                    destination_id TEXT,
                    dimension_name TEXT,
                    value REAL,
                    unit TEXT,
                    sample_period TEXT,
                    confidence REAL,
                    source_evidence_ids TEXT,  -- JSON array
                    last_updated TIMESTAMP,
                    PRIMARY KEY (destination_id, dimension_name),
                    FOREIGN KEY (destination_id) REFERENCES destinations(id)
                )
            """)
            
            # Temporal slices table (SCD2)
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS temporal_slices (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    destination_id TEXT,
                    valid_from TIMESTAMP,
                    valid_to TIMESTAMP,
                    season TEXT,
                    seasonal_highlights TEXT,   -- JSON
                    special_events TEXT,        -- JSON array
                    weather_patterns TEXT,      -- JSON
                    visitor_patterns TEXT,      -- JSON
                    FOREIGN KEY (destination_id) REFERENCES destinations(id)
                )
            """)
            
            # POIs table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS pois (
                    poi_id TEXT PRIMARY KEY,
                    destination_id TEXT,
                    name TEXT NOT NULL,
                    description TEXT,
                    location TEXT,              -- JSON {lat, lng}
                    address TEXT,
                    poi_type TEXT,
                    theme_tags TEXT,            -- JSON array
                    ada_accessible BOOLEAN,
                    ada_features TEXT,          -- JSON array
                    media_urls TEXT,            -- JSON array
                    operating_hours TEXT,       -- JSON
                    price_range TEXT,
                    rating REAL,
                    review_count INTEGER,
                    FOREIGN KEY (destination_id) REFERENCES destinations(id)
                )
            """)
            
            # Create indexes for performance
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_evidence_destination ON evidence(destination_id)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_evidence_source ON evidence(source_url)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_themes_destination ON themes(destination_id)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_themes_confidence ON themes(confidence_level)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_dimensions_destination ON dimensions(destination_id)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_temporal_destination ON temporal_slices(destination_id)")
            
            self.conn.commit()
            self.logger.info(f"Enhanced database initialized: {self.db_path}")
            
        except sqlite3.Error as e:
            self.logger.error(f"Database initialization error: {e}")
            if self.conn:
                self.conn.close()
            raise

    def store_destination(self, destination: Destination, 
                         analysis_metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Store or update a complete destination
        
        Args:
            destination: Destination object to store
            analysis_metadata: Optional metadata about the analysis
            
        Returns:
            Dictionary with storage results including DB status and JSON file paths
        """
        results = {
            "database_stored": False,
            "json_files_created": {},
            "errors": []
        }
        
        if not self.conn:
            results["errors"].append("No database connection")
            return results
            
        try:
            cursor = self.conn.cursor()
            
            # Store main destination
            cursor.execute("""
                INSERT OR REPLACE INTO destinations 
                (id, names, admin_levels, timezone, population, country_code, 
                 core_geo, lineage, meta, last_updated, destination_revision)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                destination.id,
                json.dumps(destination.names),
                json.dumps(destination.admin_levels),
                destination.timezone,
                destination.population,
                destination.country_code,
                json.dumps(destination.core_geo),
                json.dumps(destination.lineage),
                json.dumps(destination.meta),
                destination.last_updated.isoformat(),
                destination.destination_revision
            ))
            
            # Store themes and evidence
            for theme in destination.themes:
                self._store_theme(cursor, destination.id, theme)
                
            # Store dimensions
            for dim_name, dim_value in destination.dimensions.items():
                if dim_value.value is not None:
                    self._store_dimension(cursor, destination.id, dim_name, dim_value)
                    
            # Store temporal slices
            for temporal_slice in destination.temporal_slices:
                self._store_temporal_slice(cursor, destination.id, temporal_slice)
                
            # Store POIs
            for poi in destination.pois:
                self._store_poi(cursor, destination.id, poi)
                
            self.conn.commit()
            self.logger.info(f"Stored destination in database: {destination.id}")
            results["database_stored"] = True
            
            # Export to JSON if enabled
            if self.enable_json_export and self.json_export_manager:
                try:
                    json_files = self.json_export_manager.export_destination_insights(
                        destination, 
                        analysis_metadata
                    )
                    results["json_files_created"] = json_files
                    self.logger.info(f"Exported {len(json_files)} JSON files for {destination.names[0] if destination.names else destination.id}")
                except Exception as json_error:
                    error_msg = f"JSON export error: {str(json_error)}"
                    self.logger.error(error_msg)
                    results["errors"].append(error_msg)
            
        except sqlite3.Error as e:
            error_msg = f"Database error storing destination {destination.id}: {e}"
            self.logger.error(error_msg)
            results["errors"].append(error_msg)
            self.conn.rollback()
            
        return results

    def _store_theme(self, cursor, destination_id: str, theme: Theme):
        """Store a theme with its evidence"""
        # Store theme
        confidence_breakdown_json = None
        confidence_level = "insufficient"
        
        if theme.confidence_breakdown:
            confidence_breakdown_json = json.dumps(theme.confidence_breakdown.to_dict())
            confidence_level = theme.confidence_breakdown.confidence_level.value
            
        cursor.execute("""
            INSERT OR REPLACE INTO themes
            (theme_id, destination_id, macro_category, micro_category, name,
             description, fit_score, confidence_breakdown, confidence_level,
             tags, created_date, last_validated)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            theme.theme_id,
            destination_id,
            theme.macro_category,
            theme.micro_category,
            theme.name,
            theme.description,
            theme.fit_score,
            confidence_breakdown_json,
            confidence_level,
            json.dumps(theme.tags),
            theme.created_date.isoformat(),
            theme.last_validated.isoformat() if theme.last_validated else None
        ))
        
        # Store evidence
        for evidence in theme.evidence:
            self._store_evidence(cursor, destination_id, evidence)
            
            # Link theme to evidence
            cursor.execute("""
                INSERT OR IGNORE INTO theme_evidence (theme_id, evidence_id)
                VALUES (?, ?)
            """, (theme.theme_id, evidence.id))

    def _store_evidence(self, cursor, destination_id: str, evidence: Evidence):
        """Store a piece of evidence"""
        cursor.execute("""
            INSERT OR REPLACE INTO evidence
            (id, destination_id, source_url, source_category, evidence_type,
             authority_weight, text_snippet, timestamp, confidence, sentiment,
             cultural_context, relationships, agent_id, published_date)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            evidence.id,
            destination_id,
            evidence.source_url,
            evidence.source_category.value,
            evidence.evidence_type.value,
            evidence.authority_weight,
            evidence.text_snippet,
            evidence.timestamp.isoformat(),
            evidence.confidence,
            evidence.sentiment,
            json.dumps(evidence.cultural_context) if evidence.cultural_context else None,
            json.dumps(evidence.relationships),
            evidence.agent_id,
            evidence.published_date.isoformat() if evidence.published_date else None
        ))

    def _store_dimension(self, cursor, destination_id: str, name: str, value: DimensionValue):
        """Store a dimension value"""
        cursor.execute("""
            INSERT OR REPLACE INTO dimensions
            (destination_id, dimension_name, value, unit, sample_period,
             confidence, source_evidence_ids, last_updated)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            destination_id,
            name,
            value.value,
            value.unit,
            value.sample_period,
            value.confidence,
            json.dumps(value.source_evidence_ids),
            value.last_updated.isoformat()
        ))

    def _store_temporal_slice(self, cursor, destination_id: str, temporal_slice: TemporalSlice):
        """Store a temporal slice"""
        cursor.execute("""
            INSERT INTO temporal_slices
            (destination_id, valid_from, valid_to, season, seasonal_highlights,
             special_events, weather_patterns, visitor_patterns)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            destination_id,
            temporal_slice.valid_from.isoformat(),
            temporal_slice.valid_to.isoformat() if temporal_slice.valid_to else None,
            temporal_slice.season,
            json.dumps(temporal_slice.seasonal_highlights),
            json.dumps(temporal_slice.special_events),
            json.dumps(temporal_slice.weather_patterns) if temporal_slice.weather_patterns else None,
            json.dumps(temporal_slice.visitor_patterns) if temporal_slice.visitor_patterns else None
        ))

    def _store_poi(self, cursor, destination_id: str, poi):
        """Store a POI"""
        cursor.execute("""
            INSERT OR REPLACE INTO pois
            (poi_id, destination_id, name, description, location, address,
             poi_type, theme_tags, ada_accessible, ada_features, media_urls,
             operating_hours, price_range, rating, review_count)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            poi.poi_id,
            destination_id,
            poi.name,
            poi.description,
            json.dumps(poi.location),
            poi.address,
            poi.poi_type,
            json.dumps(poi.theme_tags),
            poi.ada_accessible,
            json.dumps(poi.ada_features),
            json.dumps(poi.media_urls),
            json.dumps(poi.operating_hours) if poi.operating_hours else None,
            poi.price_range,
            poi.rating,
            poi.review_count
        ))

    def get_destination_themes(
        self, 
        destination_id: str, 
        min_confidence: float = 0.5,
        macro_category: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Get themes for a destination with optional filtering"""
        if not self.conn:
            return []
            
        try:
            cursor = self.conn.cursor()
            
            query = """
                SELECT t.*, COUNT(te.evidence_id) as evidence_count
                FROM themes t
                LEFT JOIN theme_evidence te ON t.theme_id = te.theme_id
                WHERE t.destination_id = ?
            """
            params = [destination_id]
            
            if macro_category:
                query += " AND t.macro_category = ?"
                params.append(macro_category)
                
            query += " GROUP BY t.theme_id"
            
            cursor.execute(query, params)
            themes = []
            
            for row in cursor.fetchall():
                confidence_breakdown = json.loads(row[7]) if row[7] else {}
                total_confidence = confidence_breakdown.get('total_confidence', 0)
                
                if total_confidence >= min_confidence:
                    themes.append({
                        'theme_id': row[0],
                        'macro_category': row[2],
                        'micro_category': row[3],
                        'name': row[4],
                        'description': row[5],
                        'fit_score': row[6],
                        'confidence_breakdown': confidence_breakdown,
                        'confidence_level': row[8],
                        'tags': json.loads(row[9]) if row[9] else [],
                        'evidence_count': row[12]
                    })
                    
            return themes
            
        except sqlite3.Error as e:
            self.logger.error(f"Error retrieving themes: {e}")
            return []

    def get_high_confidence_themes(
        self, 
        min_confidence: float = 0.8,
        limit: int = 100
    ) -> List[Dict[str, Any]]:
        """Get high-confidence themes across all destinations"""
        if not self.conn:
            return []
            
        try:
            cursor = self.conn.cursor()
            
            cursor.execute("""
                SELECT d.names, t.*
                FROM themes t
                JOIN destinations d ON t.destination_id = d.id
                WHERE t.confidence_level IN ('verified', 'strongly_supported')
                ORDER BY json_extract(t.confidence_breakdown, '$.total_confidence') DESC
                LIMIT ?
            """, (limit,))
            
            themes = []
            for row in cursor.fetchall():
                themes.append({
                    'destination_names': json.loads(row[0]),
                    'theme_name': row[5],
                    'macro_category': row[3],
                    'confidence_level': row[9],
                    'confidence_score': json.loads(row[8])['total_confidence']
                })
                
            return themes
            
        except sqlite3.Error as e:
            self.logger.error(f"Error retrieving high confidence themes: {e}")
            return []

    def search_by_dimension(
        self,
        dimension_name: str,
        min_value: float,
        max_value: Optional[float] = None
    ) -> List[Dict[str, Any]]:
        """Search destinations by dimension value"""
        if not self.conn:
            return []
            
        try:
            cursor = self.conn.cursor()
            
            query = """
                SELECT d.id, d.names, dim.value, dim.confidence
                FROM dimensions dim
                JOIN destinations d ON dim.destination_id = d.id
                WHERE dim.dimension_name = ? AND dim.value >= ?
            """
            params = [dimension_name, min_value]
            
            if max_value is not None:
                query += " AND dim.value <= ?"
                params.append(max_value)
                
            query += " ORDER BY dim.value DESC"
            
            cursor.execute(query, params)
            
            results = []
            for row in cursor.fetchall():
                results.append({
                    'destination_id': row[0],
                    'destination_names': json.loads(row[1]),
                    'value': row[2],
                    'confidence': row[3]
                })
                
            return results
            
        except sqlite3.Error as e:
            self.logger.error(f"Error searching by dimension: {e}")
            return []

    def get_temporal_data(
        self,
        destination_id: str,
        target_date: Optional[datetime] = None
    ) -> Optional[Dict[str, Any]]:
        """Get temporal slice for a specific date"""
        if not self.conn:
            return None
            
        if target_date is None:
            target_date = datetime.now()
            
        try:
            cursor = self.conn.cursor()
            
            cursor.execute("""
                SELECT * FROM temporal_slices
                WHERE destination_id = ?
                AND valid_from <= ?
                AND (valid_to IS NULL OR valid_to > ?)
                ORDER BY valid_from DESC
                LIMIT 1
            """, (destination_id, target_date.isoformat(), target_date.isoformat()))
            
            row = cursor.fetchone()
            if row:
                return {
                    'valid_from': row[2],
                    'valid_to': row[3],
                    'season': row[4],
                    'seasonal_highlights': json.loads(row[5]) if row[5] else {},
                    'special_events': json.loads(row[6]) if row[6] else [],
                    'weather_patterns': json.loads(row[7]) if row[7] else None,
                    'visitor_patterns': json.loads(row[8]) if row[8] else None
                }
                
            return None
            
        except sqlite3.Error as e:
            self.logger.error(f"Error retrieving temporal data: {e}")
            return None

    def archive_old_json_exports(self, days_to_keep: int = 30):
        """Archive old JSON exports if JSON export is enabled"""
        if self.json_export_manager:
            self.json_export_manager.archive_old_exports(days_to_keep)
            self.logger.info(f"Archived JSON exports older than {days_to_keep} days")

    def close_db(self):
        """Close database connection"""
        if self.conn:
            self.conn.close()
            self.logger.info("Database connection closed.") 