import sqlite3
import logging
import json
from datetime import datetime
from typing import Optional, List, Dict, Any
import os
import uuid

from .enhanced_data_models import Destination, Theme, Evidence, TemporalSlice, DimensionValue, AuthenticInsight, SeasonalWindow, LocalAuthority, PointOfInterest, SpecialEvent
from .confidence_scoring import ConfidenceBreakdown, ConfidenceLevel
from .consolidated_json_export_manager import ConsolidatedJsonExportManager
from .safe_dict_utils import safe_get_nested
from src.schemas import InsightType, LocationExclusivity, AuthorityType



class EnhancedDatabaseManager:
    """A rewritten, robust database manager with explicit connection management."""
    
    def __init__(self, db_path: str = "enhanced_destination_intelligence.db", 
                 enable_json_export: bool = True,
                 json_export_path: Optional[str] = None,
                 config: Optional[Dict[str, Any]] = None):
        """
        Initialize database manager with adaptive configuration support
        
        Args:
            db_path: Path to the SQLite database file
            enable_json_export: Whether to enable JSON export
            json_export_path: Path for JSON exports
            config: Application configuration dictionary for adaptive processing
        """
        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
        self.db_path = os.path.join(project_root, db_path)
        self.logger = logging.getLogger(__name__)
        self.conn = None
        
        # Store configuration for adaptive processing
        self.config = config or {}
        
        # Initialize consolidated JSON export manager if enabled
        self.enable_json_export = enable_json_export
        self.json_export_manager = None
        
        if enable_json_export:
            effective_export_path = json_export_path # Capture the path being used
            if effective_export_path is None:
                effective_export_path = os.path.join(project_root, "destination_insights")
            
            # ADDED DEBUG LOG
            self.logger.info(f"[EnhancedDBManager INIT] Received json_export_path: {json_export_path}")
            self.logger.info(f"[EnhancedDBManager INIT] Effective export_base_path for ConsolidatedManager: {effective_export_path}")
            
            # Pass configuration to export manager for adaptive processing
            self.json_export_manager = ConsolidatedJsonExportManager(
                export_base_path=effective_export_path,
                config=self.config
            )
            # Original log message, now uses effective_export_path for clarity
            self.logger.info(f"Consolidated JSON export enabled with adaptive intelligence. Files will be saved to: {self.json_export_manager.consolidated_path}")
        
        self.connect()
        self.init_database()
        
    def connect(self):
        """Establish database connection"""
        if not os.path.exists(os.path.dirname(self.db_path)):
            os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        
        self.conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row  # Enable column access by name
        
        # Initialize database schema if needed
        self.init_database()
    
    def get_connection(self):
        """Get the database connection object"""
        if not self.conn:
            self.connect()
        return self.conn

    def close_db(self):
        """Closes the database connection."""
        if self.conn:
            self.conn.close()
            self.conn = None
            self.logger.info("Database connection closed.")

    def _execute_sql(self, cursor, sql: str, params: tuple = ()):
        """Executes a SQL command on a given cursor."""
        try:
            cursor.execute(sql, params)
        except sqlite3.Error as e:
            self.logger.error(f"Failed to execute SQL: {sql} with params {params}. Error: {e}", exc_info=True)
            raise

    def _add_column_if_not_exists(self, cursor, table_name: str, column_name: str, column_definition: str):
        """Safely add a column to a table if it doesn't exist."""
        try:
            cursor.execute(f"ALTER TABLE {table_name} ADD COLUMN {column_name} {column_definition}")
            self.logger.info(f"Added column {column_name} to table {table_name}")
        except sqlite3.OperationalError as e:
            if "duplicate column name" in str(e).lower():
                self.logger.debug(f"Column {column_name} already exists in table {table_name}")
            else:
                self.logger.error(f"Error adding column {column_name} to table {table_name}: {e}")
                raise

    def init_database(self):
        """Initializes the complete database schema with all required tables."""
        if not self.conn: 
            self.connect()
        
        cursor = self.conn.cursor()
        
        try:
            # Create destinations table with all required columns
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS destinations (
                    id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    names TEXT,
                    country_code TEXT,
                    population INTEGER,
                    area_km2 REAL,
                    primary_language TEXT,
                    hdi REAL,
                    gdp_per_capita_usd REAL,
                    vibe_descriptors TEXT,
                    historical_summary TEXT,
                    unesco_sites TEXT,
                    annual_tourist_arrivals INTEGER,
                    popularity_stage TEXT,
                    visa_info_url TEXT,
                    last_updated TEXT,
                    admin_levels TEXT,
                    core_geo TEXT,
                    meta TEXT,
                    destination_revision INTEGER DEFAULT 1,
                    dominant_religions TEXT,
                    timezone TEXT
                )
            """)

            # Create themes table with all required columns
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS themes (
                    theme_id TEXT PRIMARY KEY,
                    destination_id TEXT NOT NULL,
                    name TEXT NOT NULL,
                    description TEXT,
                    macro_category TEXT,
                    micro_category TEXT,
                    fit_score REAL DEFAULT 0.0,
                    tags TEXT,
                    sentiment_analysis TEXT,
                    temporal_analysis TEXT,
                    confidence_level TEXT,
                    confidence_breakdown TEXT,
                    authentic_insights TEXT,
                    local_authorities TEXT,
                    source_evidence_ids TEXT,
                    traveler_relevance_factor REAL DEFAULT 0.5,
                    adjusted_overall_confidence REAL DEFAULT 0.0,
                    FOREIGN KEY(destination_id) REFERENCES destinations(id)
                )
            """)

            # Create evidence table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS evidence (
                    id TEXT PRIMARY KEY,
                    destination_id TEXT NOT NULL,
                    content TEXT,
                    source_url TEXT,
                    source_type TEXT,
                    publication_date TEXT,
                    scrape_timestamp TEXT,
                    relevance_score REAL DEFAULT 0.0,
                    confidence REAL DEFAULT 0.0,
                    author TEXT,
                    title TEXT,
                    language TEXT,
                    word_count INTEGER DEFAULT 0,
                    sentiment REAL DEFAULT 0.0,
                    cultural_context TEXT,
                    relationships TEXT,
                    agent_id TEXT,
                    published_date TEXT,
                    FOREIGN KEY(destination_id) REFERENCES destinations(id)
                )
            """)

            # Create theme_evidence junction table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS theme_evidence (
                    theme_id TEXT,
                    evidence_id TEXT,
                    PRIMARY KEY (theme_id, evidence_id),
                    FOREIGN KEY (theme_id) REFERENCES themes(theme_id),
                    FOREIGN KEY (evidence_id) REFERENCES evidence(id)
                )
            """)

            # Create dimensions table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS dimensions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    destination_id TEXT NOT NULL,
                    name TEXT NOT NULL,
                    value REAL,
                    confidence REAL DEFAULT 0.0,
                    source TEXT,
                    last_updated TEXT,
                    FOREIGN KEY(destination_id) REFERENCES destinations(id)
                )
            """)

            # Create temporal_slices table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS temporal_slices (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    destination_id TEXT NOT NULL,
                    start_date TEXT,
                    end_date TEXT,
                    slice_type TEXT,
                    average_price_level REAL,
                    weather_data TEXT,
                    crowd_level TEXT,
                    special_events TEXT,
                    local_insights TEXT,
                    FOREIGN KEY(destination_id) REFERENCES destinations(id)
                )
            """)

            # Create pois table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS pois (
                    poi_id TEXT PRIMARY KEY,
                    destination_id TEXT NOT NULL,
                    name TEXT NOT NULL,
                    poi_type TEXT,
                    coordinates TEXT,
                    description TEXT,
                    rating REAL,
                    price_level INTEGER,
                    operating_hours TEXT,
                    contact_info TEXT,
                    FOREIGN KEY(destination_id) REFERENCES destinations(id)
                )
            """)

            # Create seasonal_windows table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS seasonal_windows (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    start_month INTEGER,
                    end_month INTEGER,
                    peak_weeks TEXT,
                    booking_lead_time INTEGER,
                    specific_dates TEXT
                )
            """)

            # Create authentic_insights table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS authentic_insights (
                    id TEXT PRIMARY KEY,
                    destination_id TEXT NOT NULL,
                    insight_type TEXT,
                    authenticity_score REAL DEFAULT 0.0,
                    uniqueness_score REAL DEFAULT 0.0,
                    actionability_score REAL DEFAULT 0.0,
                    temporal_relevance REAL DEFAULT 0.0,
                    location_exclusivity TEXT,
                    seasonal_window_id INTEGER,
                    local_validation_count INTEGER DEFAULT 0,
                    FOREIGN KEY(destination_id) REFERENCES destinations(id),
                    FOREIGN KEY(seasonal_window_id) REFERENCES seasonal_windows(id)
                )
            """)

            # Create local_authorities table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS local_authorities (
                    id TEXT PRIMARY KEY,
                    destination_id TEXT NOT NULL,
                    authority_type TEXT,
                    local_tenure INTEGER DEFAULT 0,
                    expertise_domain TEXT,
                    community_validation REAL DEFAULT 0.0,
                    FOREIGN KEY(destination_id) REFERENCES destinations(id)
                )
            """)

            # Add missing columns to existing tables if they don't exist
            self._add_column_if_not_exists(cursor, "destinations", "names", "TEXT")
            self._add_column_if_not_exists(cursor, "destinations", "meta", "TEXT")
            self._add_column_if_not_exists(cursor, "destinations", "destination_revision", "INTEGER DEFAULT 1")
            self._add_column_if_not_exists(cursor, "destinations", "dominant_religions", "TEXT")
            self._add_column_if_not_exists(cursor, "destinations", "timezone", "TEXT")
            self._add_column_if_not_exists(cursor, "themes", "traveler_relevance_factor", "REAL DEFAULT 0.5")
            self._add_column_if_not_exists(cursor, "themes", "adjusted_overall_confidence", "REAL DEFAULT 0.0")
            
            # Add missing columns to evidence table
            self._add_column_if_not_exists(cursor, "evidence", "sentiment", "REAL DEFAULT 0.0")
            self._add_column_if_not_exists(cursor, "evidence", "cultural_context", "TEXT")
            self._add_column_if_not_exists(cursor, "evidence", "relationships", "TEXT")
            self._add_column_if_not_exists(cursor, "evidence", "agent_id", "TEXT")
            self._add_column_if_not_exists(cursor, "evidence", "published_date", "TEXT")

            self.conn.commit()
            
            # Create performance indices
            self._create_performance_indices(cursor)
            
            self.logger.info("Complete database schema initialized successfully.")
            
        except sqlite3.Error as e:
            self.logger.error(f"Error initializing database schema: {e}", exc_info=True)
            raise

    def _create_performance_indices(self, cursor):
        """Create performance indices for better query performance"""
        indices = [
            "CREATE INDEX IF NOT EXISTS idx_evidence_destination ON evidence(destination_id)",
            "CREATE INDEX IF NOT EXISTS idx_themes_destination ON themes(destination_id)",
            "CREATE INDEX IF NOT EXISTS idx_theme_evidence_theme ON theme_evidence(theme_id)",
            "CREATE INDEX IF NOT EXISTS idx_authentic_insights_theme ON authentic_insights(destination_id)"
        ]
        
        for index_sql in indices:
            try:
                cursor.execute(index_sql)
                self.logger.debug(f"Created index: {index_sql}")
            except sqlite3.Error as e:
                self.logger.warning(f"Failed to create index {index_sql}: {e}")
        
        self.conn.commit()

    def store_destination(self, destination: Destination, 
                         analysis_metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Store or update a destination with all its related data"""
        
        # Safe access to destination ID
        if hasattr(destination, 'id'):
            destination_id = destination.id
        elif isinstance(destination, dict):
            destination_id = destination.get('id', 'unknown_destination')
        else:
            destination_id = 'unknown_destination'
        
        # Safe access to destination names
        if hasattr(destination, 'names'):
            names = destination.names
        elif isinstance(destination, dict):
            names = destination.get('names', [])
        else:
            names = []
        
        # Safe access to other destination attributes
        def safe_get_dest_attr(obj, attr, default=None):
            if hasattr(obj, attr):
                return getattr(obj, attr, default)
            elif isinstance(obj, dict):
                return obj.get(attr, default)
            else:
                return default
        
        results = {
            "database_stored": False,
            "json_files_created": {},
            "errors": [],
            "warnings": [],
            "validation_errors": [],
            "storage_summary": {
                "themes": 0,
                "evidence": 0,
                "dimensions": 0,
                "temporal_slices": 0,
                "pois": 0,
                "insights": 0,
                "authorities": 0
            },
            "json_exported": False
        }
        
        try:
            # Validate destination
            validation_errors = self._validate_destination(destination)
            if validation_errors:
                self.logger.warning(f"Validation warnings for destination {destination_id}: {validation_errors}")
                results["validation_errors"] = validation_errors
            
            cursor = self.conn.cursor()
            
            # Store core destination data
            self.logger.debug(f"Executing INSERT or REPLACE for destination ID: {destination_id}")
            
            # Safe access to all destination attributes
            country_code = safe_get_dest_attr(destination, 'country_code', 'US')
            population = safe_get_dest_attr(destination, 'population', None)
            timezone = safe_get_dest_attr(destination, 'timezone', 'UTC')
            core_geo = safe_get_dest_attr(destination, 'core_geo', {})
            meta = safe_get_dest_attr(destination, 'meta', {})
            last_updated = safe_get_dest_attr(destination, 'last_updated', datetime.now())
            destination_revision = safe_get_dest_attr(destination, 'destination_revision', 1)
            
            cursor.execute("""
                INSERT OR REPLACE INTO destinations 
                (id, name, names, country_code, population, timezone, core_geo, meta, last_updated, destination_revision)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                destination_id, 
                names[0] if names else destination_id,  # name (singular) - first name or fallback to ID
                json.dumps(names),  # names (plural) - full list as JSON
                country_code, 
                population,
                timezone, 
                json.dumps(core_geo), 
                json.dumps(meta),
                last_updated.isoformat() if hasattr(last_updated, 'isoformat') else str(last_updated),
                destination_revision
            ))
            
            self.logger.debug(f"Successfully stored core destination data for {destination_id}")
            
            # Clear existing related data for this destination
            cursor.execute("DELETE FROM themes WHERE destination_id = ?", (destination_id,))
            
            # Store themes
            themes = safe_get_dest_attr(destination, 'themes', [])
            for theme in themes:
                try:
                    self._store_theme(cursor, destination_id, theme)
                    results["storage_summary"]["themes"] += 1
                except Exception as e:
                    # Safe access to theme_id for error logging
                    if hasattr(theme, 'theme_id'):
                        theme_id = theme.theme_id
                    elif isinstance(theme, dict):
                        theme_id = theme.get('theme_id', 'unknown_theme')
                    else:
                        theme_id = 'unknown_theme'
                    results["warnings"].append(f"Unexpected error storing theme {theme_id}: {e}")
            
            # Store authentic insights
            insights = safe_get_dest_attr(destination, 'authentic_insights', [])
            for insight in insights:
                seasonal_window_id = self._store_seasonal_window(cursor, getattr(insight, 'seasonal_window', None))
                self._store_authentic_insight(cursor, destination_id, insight, seasonal_window_id)
                results["storage_summary"]["insights"] += 1
            
            # Store local authorities
            authorities = safe_get_dest_attr(destination, 'local_authorities', [])
            for authority in authorities:
                self._store_local_authority(cursor, destination_id, authority)
                results["storage_summary"]["authorities"] += 1
            
            # Store theme-level insights and authorities
            for theme in themes:
                theme_insights = safe_get_dest_attr(theme, 'authentic_insights', [])
                for insight in theme_insights:
                    seasonal_window_id = self._store_seasonal_window(cursor, getattr(insight, 'seasonal_window', None))
                    self._store_authentic_insight(cursor, destination_id, insight, seasonal_window_id)
                    results["storage_summary"]["insights"] += 1
                
                theme_authorities = safe_get_dest_attr(theme, 'local_authorities', [])
                for authority in theme_authorities:
                    self._store_local_authority(cursor, destination_id, authority)
                    results["storage_summary"]["authorities"] += 1
            
            # Store dimensions
            dimensions = safe_get_dest_attr(destination, 'dimensions', {})
            for dim_name, dim_value in dimensions.items():
                self._store_dimension(cursor, destination_id, dim_name, dim_value)
                results["storage_summary"]["dimensions"] += 1
            
            # Store temporal slices
            temporal_slices = safe_get_dest_attr(destination, 'temporal_slices', [])
            for temporal_slice in temporal_slices:
                self._store_temporal_slice(cursor, destination_id, temporal_slice)
                results["storage_summary"]["temporal_slices"] += 1
            
            # Store POIs
            pois = safe_get_dest_attr(destination, 'pois', [])
            for poi in pois:
                self._store_poi(cursor, destination_id, poi)
                results["storage_summary"]["pois"] += 1
            
            # Commit transaction
            self.conn.commit()
            results["database_stored"] = True
            
            self.logger.info(f"Successfully stored destination {destination_id}: "
                           f"{results['storage_summary']['themes']} themes, "
                           f"{results['storage_summary']['evidence']} evidence, "
                           f"{results['storage_summary']['dimensions']} dimensions, "
                           f"{results['storage_summary']['temporal_slices']} temporal slices, "
                           f"{results['storage_summary']['pois']} POIs, "
                           f"{results['storage_summary']['insights']} insights, "
                           f"{results['storage_summary']['authorities']} authorities")
            
            # Export to JSON if enabled
            if self.json_export_manager:
                try:
                    json_file_path = self.json_export_manager.export_destination_insights(
                        destination, analysis_metadata
                    )
                    results["json_files_created"]["comprehensive"] = json_file_path
                    results["json_exported"] = True
                    results["json_export_path"] = json_file_path
                    self.logger.info(f"Exported consolidated JSON file: {os.path.basename(json_file_path)}")
                except Exception as e:
                    error_msg = f"JSON export error: {e}"
                    self.logger.error(error_msg)
                    results["errors"].append(error_msg)
            
        except sqlite3.Error as e:
            error_msg = f"Database error storing destination {destination_id}: {e}"
            self.logger.error(error_msg)
            results["errors"].append(error_msg)
            try:
                self.conn.rollback()
            except:
                pass
            
        except Exception as e:
            error_msg = f"Unexpected error storing destination {destination_id}: {e}"
            self.logger.error(error_msg)
            results["errors"].append(error_msg)
            try:
                self.conn.rollback()
            except:
                pass
        
        return results

    def _validate_destination(self, destination: Destination) -> List[str]:
        """Validate destination data before storage"""
        errors = []
        
        # Safe access to destination attributes
        def safe_get_dest_attr(obj, attr, default=None):
            if hasattr(obj, attr):
                return getattr(obj, attr, default)
            elif isinstance(obj, dict):
                return obj.get(attr, default)
            else:
                return default
        
        # Check required fields
        dest_id = safe_get_dest_attr(destination, 'id', None)
        dest_names = safe_get_dest_attr(destination, 'names', [])
        dest_country_code = safe_get_dest_attr(destination, 'country_code', None)
        dest_timezone = safe_get_dest_attr(destination, 'timezone', None)
        dest_themes = safe_get_dest_attr(destination, 'themes', [])
        dest_dimensions = safe_get_dest_attr(destination, 'dimensions', {})
        
        if not dest_id:
            errors.append("Destination ID is required")
        if not dest_names or not dest_names[0]:
            errors.append("Destination must have at least one name")
        if not dest_country_code:
            errors.append("Country code is required")
        if not dest_timezone:
            errors.append("Timezone is required")
            
        # Validate themes
        for theme in dest_themes:
            # Handle both Theme objects and dictionaries
            if hasattr(theme, 'theme_id'):  # Theme object
                theme_id = theme.theme_id
                theme_name = theme.name
                theme_fit_score = theme.fit_score
                evidence_list = theme.evidence
            else:  # Dictionary
                theme_id = theme.get('theme_id', '') if isinstance(theme, dict) else ''
                theme_name = theme.get('name', '') if isinstance(theme, dict) else ''
                theme_fit_score = theme.get('fit_score', 0.0) if isinstance(theme, dict) else 0.0
                evidence_list = theme.get('evidence', []) if isinstance(theme, dict) else []
            
            if not theme_id:
                errors.append(f"Theme missing ID: {theme_name}")
            if not theme_name:
                errors.append(f"Theme {theme_id} missing name")
            
            if theme_fit_score < 0 or theme_fit_score > 1:
                errors.append(f"Theme {theme_name} has invalid fit_score: {theme_fit_score}")
            
            for evidence in evidence_list:
                # SAFETY: Handle both Evidence objects and dictionaries
                evidence_id = getattr(evidence, 'id', None) or evidence.get('id', '') if isinstance(evidence, dict) else evidence.id if hasattr(evidence, 'id') else None
                evidence_source_url = getattr(evidence, 'source_url', None) or evidence.get('source_url', '') if isinstance(evidence, dict) else evidence.source_url if hasattr(evidence, 'source_url') else None
                evidence_confidence = getattr(evidence, 'confidence', 0) or evidence.get('confidence', 0) if isinstance(evidence, dict) else evidence.confidence if hasattr(evidence, 'confidence') else 0
                
                if not evidence_id:
                    errors.append(f"Evidence missing ID in theme {theme_name}")
                if not evidence_source_url:
                    errors.append(f"Evidence missing source_url in theme {theme_name}")
                if evidence_confidence < 0 or evidence_confidence > 1:
                    errors.append(f"Evidence has invalid confidence in theme {theme_name}: {evidence_confidence}")
                    
        # Validate dimensions
        for dim_name, dim_value in dest_dimensions.items():
            if hasattr(dim_value, 'confidence'):
                if dim_value.confidence < 0 or dim_value.confidence > 1:
                    errors.append(f"Dimension {dim_name} has invalid confidence: {dim_value.confidence}")
                    
        return errors

    def _store_theme(self, cursor, destination_id: str, theme: Theme):
        """Store or update a theme"""
        
        # Handle Theme objects, ThemeWrapper objects, and dictionaries
        def safe_get_theme_attr(obj, attr, default=None):
            """Safely get attribute from Theme, ThemeWrapper, or dict"""
            # First try direct attribute access
            if hasattr(obj, attr):
                return getattr(obj, attr, default)
            # Then try .get() method (for ThemeWrapper and dicts)
            elif hasattr(obj, 'get') and callable(getattr(obj, 'get')):
                return obj.get(attr, default)
            # Then try dictionary access
            elif isinstance(obj, dict):
                return obj.get(attr, default)
            # Finally try _data attribute (for ThemeWrapper internal data)
            elif hasattr(obj, '_data') and isinstance(obj._data, dict):
                return obj._data.get(attr, default)
            else:
                return default
        
        theme_id = safe_get_theme_attr(theme, 'theme_id', '')
        theme_name = safe_get_theme_attr(theme, 'name', '')
        
        level_obj = theme.get_confidence_level() if hasattr(theme, 'get_confidence_level') else safe_get_theme_attr(theme, 'confidence_level', 'unknown')
        confidence_level_value_to_store = None

        if isinstance(level_obj, ConfidenceLevel):
            confidence_level_value_to_store = level_obj.value
        elif isinstance(level_obj, str):
            self.logger.warning(f"Theme {theme_id}: get_confidence_level() returned a string: '{level_obj}'. Using it directly for DB.")
            confidence_level_value_to_store = level_obj
        else:
            self.logger.error(f"Theme {theme_id}: get_confidence_level() returned unexpected type: {type(level_obj)} with value '{level_obj}'. Defaulting confidence to 'unknown'.")
            confidence_level_value_to_store = "unknown"

        # Ensure new fields have default FLOAT values if not present or None on the theme object.
        raw_relevance_factor = safe_get_theme_attr(theme, 'traveler_relevance_factor', None)
        traveler_relevance_factor = raw_relevance_factor if isinstance(raw_relevance_factor, float) else 0.5 # Default to neutral float

        raw_adjusted_confidence = safe_get_theme_attr(theme, 'adjusted_overall_confidence', None)
        adjusted_overall_confidence = raw_adjusted_confidence # Keep it as is, could be None initially

        # If adjusted_overall_confidence was not directly on the theme object OR was None,
        # try to calculate it now using the (now guaranteed float) traveler_relevance_factor.
        if adjusted_overall_confidence is None: 
            confidence_breakdown = safe_get_theme_attr(theme, 'confidence_breakdown', None)
            if confidence_breakdown:
                # Handle both ConfidenceBreakdown objects and dictionaries
                if hasattr(confidence_breakdown, 'overall_confidence'):
                    # ConfidenceBreakdown object
                    original_confidence = confidence_breakdown.overall_confidence
                elif isinstance(confidence_breakdown, dict) and 'overall_confidence' in confidence_breakdown:
                    # Dictionary format
                    original_confidence = confidence_breakdown['overall_confidence']
                else:
                    original_confidence = None
                
                if original_confidence is not None and isinstance(original_confidence, (int, float)):
                    adjusted_overall_confidence = float(original_confidence) * traveler_relevance_factor
                else:
                    self.logger.warning(f"Theme {theme_id}: original_confidence in breakdown is not a valid number ('{type(original_confidence)}'). Setting adjusted_confidence to 0.")
                    adjusted_overall_confidence = 0.0
            else:
                # No breakdown or no original confidence in breakdown, cannot calculate adjusted from it.
                # If traveler_relevance_factor itself was also defaulted (e.g. to 0.5), this would be 0.
                # It implies this theme might be problematic or missing crucial data upstream.
                self.logger.warning(f"Theme {theme_id}: Missing confidence_breakdown or overall_confidence in it. Setting adjusted_confidence to 0.")
                adjusted_overall_confidence = 0.0 

        # Final check to ensure adjusted_overall_confidence is a float for DB storage
        if not isinstance(adjusted_overall_confidence, float):
            self.logger.error(f"Theme {theme_id}: adjusted_overall_confidence ended up as non-float ('{type(adjusted_overall_confidence)}'). Defaulting to 0.0 for DB.")
            adjusted_overall_confidence = 0.0

        
        values = (
            theme_id,
            destination_id,
            safe_get_theme_attr(theme, 'name', ''),
            safe_get_theme_attr(theme, 'description', ''),
            safe_get_theme_attr(theme, 'macro_category', ''),
            safe_get_theme_attr(theme, 'micro_category', ''),
            safe_get_theme_attr(theme, 'fit_score', 0.0),
            json.dumps(safe_get_theme_attr(theme, 'tags', [])),
            json.dumps(safe_get_theme_attr(theme, 'sentiment_analysis', None)) if safe_get_theme_attr(theme, 'sentiment_analysis', None) else None,
            json.dumps(safe_get_theme_attr(theme, 'temporal_analysis', None)) if safe_get_theme_attr(theme, 'temporal_analysis', None) else None,
            confidence_level_value_to_store,
            # Proper type checking for confidence_breakdown
            json.dumps(
                confidence_breakdown if isinstance(confidence_breakdown, dict)
                else confidence_breakdown.to_dict() if hasattr(confidence_breakdown, 'to_dict') and callable(getattr(confidence_breakdown, 'to_dict'))
                else {"error": "unexpected_confidence_breakdown_type", "type": str(type(confidence_breakdown))}
            ) if confidence_breakdown else None,
            # Proper type checking for authentic_insights
            json.dumps([
                ai if isinstance(ai, dict)
                else ai.to_dict() if hasattr(ai, 'to_dict') and callable(getattr(ai, 'to_dict'))
                else {"error": "unexpected_insight_type", "type": str(type(ai))}
                for ai in safe_get_theme_attr(theme, 'authentic_insights', [])
            ]),
            # Proper type checking for local_authorities
            json.dumps([
                la if isinstance(la, dict)
                else la.to_dict() if hasattr(la, 'to_dict') and callable(getattr(la, 'to_dict'))
                else {"error": "unexpected_authority_type", "type": str(type(la))}
                for la in safe_get_theme_attr(theme, 'local_authorities', [])
            ]),
            json.dumps([
                evidence.id if hasattr(evidence, 'id') 
                else evidence.get('id', '') if isinstance(evidence, dict)
                else ''
                for evidence in safe_get_theme_attr(theme, 'evidence', [])
            ]),
            traveler_relevance_factor,
            adjusted_overall_confidence
        )
        
        # DEBUG: Keep value count for debugging
        if len(values) != 17:
            print(f"ERROR: Expected 17 values, got {len(values)}: {values}")
        
        cursor.execute("""
            INSERT OR REPLACE INTO themes 
            (theme_id, destination_id, name, description, macro_category, micro_category,
             fit_score, tags, sentiment_analysis, temporal_analysis, confidence_level,
             confidence_breakdown, authentic_insights, local_authorities, source_evidence_ids,
             traveler_relevance_factor, adjusted_overall_confidence)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, values)
        
        # Store evidence
        evidence_stored_count = 0
        theme_id = safe_get_theme_attr(theme, 'theme_id', '')
        
        for evidence in safe_get_theme_attr(theme, 'evidence', []):
            try:
                self._store_evidence(cursor, destination_id, evidence)
                evidence_stored_count += 1
                
                # Get evidence ID - handle both objects and dictionaries
                evidence_id = (
                    evidence.id if hasattr(evidence, 'id')
                    else evidence['id'] if isinstance(evidence, dict) and 'id' in evidence
                    else ''
                )
                
                # Link theme to evidence if table exists
                try:
                    cursor.execute("""
                        INSERT OR IGNORE INTO theme_evidence (theme_id, evidence_id)
                        VALUES (?, ?)
                    """, (theme_id, evidence_id))
                except sqlite3.OperationalError:
                    # theme_evidence table might not exist, create it
                    cursor.execute("""
                        CREATE TABLE IF NOT EXISTS theme_evidence (
                            theme_id TEXT,
                            evidence_id TEXT,
                            PRIMARY KEY (theme_id, evidence_id),
                            FOREIGN KEY (theme_id) REFERENCES themes(theme_id),
                            FOREIGN KEY (evidence_id) REFERENCES evidence(id)
                        )
                    """)
                    cursor.execute("""
                        INSERT OR IGNORE INTO theme_evidence (theme_id, evidence_id)
                        VALUES (?, ?)
                    """, (theme_id, evidence_id))
            except Exception as e:
                evidence_id_for_log = (
                    evidence.id if hasattr(evidence, 'id')
                    else evidence['id'] if isinstance(evidence, dict) and 'id' in evidence
                    else 'unknown'
                )
                self.logger.warning(f"Failed to store evidence {evidence_id_for_log}: {e}")
        
        if evidence_stored_count == 0:
            self.logger.warning(f"No evidence stored for theme {theme_id}")

    def _store_authentic_insight(self, cursor, destination_id: str, insight: AuthenticInsight, seasonal_window_id: Optional[int]):
        """Store an authentic insight"""
        cursor.execute("""
            INSERT OR REPLACE INTO authentic_insights
            (id, destination_id, insight_type, authenticity_score, uniqueness_score,
             actionability_score, temporal_relevance, location_exclusivity, seasonal_window_id,
             local_validation_count)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            str(uuid.uuid4()), # Generate a new UUID for the insight ID
            destination_id,
            # Safe attribute access for both dict and object formats
            safe_get_nested(insight, ['insight_type', 'value'], getattr(insight, 'insight_type', 'unknown')) if isinstance(insight, dict) else (insight.insight_type.value if hasattr(insight.insight_type, 'value') else str(insight.insight_type)),
            getattr(insight, 'authenticity_score', 0.0),
            getattr(insight, 'uniqueness_score', 0.0),
            getattr(insight, 'actionability_score', 0.0),
            getattr(insight, 'temporal_relevance', 0.0),
            safe_get_nested(insight, ['location_exclusivity', 'value'], getattr(insight, 'location_exclusivity', 'unknown')) if isinstance(insight, dict) else (insight.location_exclusivity.value if hasattr(insight.location_exclusivity, 'value') else str(insight.location_exclusivity)),
            seasonal_window_id,
            getattr(insight, 'local_validation_count', 0)
        ))

    def _store_seasonal_window(self, cursor, seasonal_window: Optional[SeasonalWindow]) -> Optional[int]:
        """Store a seasonal window and return its ID"""
        if not seasonal_window:
            return None
        
        cursor.execute("""
            INSERT INTO seasonal_windows
            (start_month, end_month, peak_weeks, booking_lead_time, specific_dates)
            VALUES (?, ?, ?, ?, ?)
        """, (
            getattr(seasonal_window, 'start_month', None),
            getattr(seasonal_window, 'end_month', None),
            json.dumps(getattr(seasonal_window, 'peak_weeks', [])),
            getattr(seasonal_window, 'booking_lead_time', None),
            json.dumps(getattr(seasonal_window, 'specific_dates', []))
        ))
        return cursor.lastrowid

    def _store_local_authority(self, cursor, destination_id: str, authority: LocalAuthority):
        """Store a local authority"""
        cursor.execute("""
            INSERT OR REPLACE INTO local_authorities
            (id, destination_id, authority_type, local_tenure, expertise_domain, community_validation)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (
            str(uuid.uuid4()), # Generate a new UUID for the local authority ID
            destination_id,
            safe_get_nested(authority, ['authority_type', 'value'], getattr(authority, 'authority_type', 'unknown')) if isinstance(authority, dict) else (authority.authority_type.value if hasattr(authority.authority_type, 'value') else str(authority.authority_type)),
            getattr(authority, 'local_tenure', None),
            getattr(authority, 'expertise_domain', ''),
            getattr(authority, 'community_validation', 0.0)
        ))

    def _store_evidence(self, cursor, destination_id: str, evidence: Evidence):
        """Store or update evidence"""
        # Helper function to handle timestamp formatting
        def format_timestamp(timestamp):
            if timestamp is None:
                return None
            elif isinstance(timestamp, str):
                return timestamp  # Already formatted
            elif hasattr(timestamp, 'isoformat'):
                return timestamp.isoformat()  # datetime object
            else:
                return str(timestamp)  # fallback to string
        
        # SAFETY: Handle both Evidence objects and dictionaries
        evidence_id = getattr(evidence, 'id', None) or evidence.get('id', '') if isinstance(evidence, dict) else evidence.id if hasattr(evidence, 'id') else None
        text_snippet = getattr(evidence, 'text_snippet', None) or evidence.get('text_snippet', '') if isinstance(evidence, dict) else evidence.text_snippet if hasattr(evidence, 'text_snippet') else None
        source_url = getattr(evidence, 'source_url', None) or evidence.get('source_url', '') if isinstance(evidence, dict) else evidence.source_url if hasattr(evidence, 'source_url') else None
        timestamp = getattr(evidence, 'timestamp', None) or evidence.get('timestamp', None) if isinstance(evidence, dict) else evidence.timestamp if hasattr(evidence, 'timestamp') else None
        confidence = getattr(evidence, 'confidence', 0.0) or evidence.get('confidence', 0.0) if isinstance(evidence, dict) else evidence.confidence if hasattr(evidence, 'confidence') else 0.0
        
        # Handle source_category - could be enum or string
        source_category = None
        if isinstance(evidence, dict):
            source_category = evidence.get('source_category', None)
        elif hasattr(evidence, 'source_category'):
            source_category = getattr(evidence.source_category, 'value', str(evidence.source_category)) if hasattr(evidence, 'source_category') else None
        
        cursor.execute("""
            INSERT OR REPLACE INTO evidence 
            (id, destination_id, content, source_url, source_type, publication_date,
             scrape_timestamp, relevance_score, confidence, author, title, language, word_count,
             sentiment, cultural_context, relationships, agent_id, published_date)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            evidence_id,
            destination_id,
            text_snippet,  # Map text_snippet to content
            source_url,
            source_category,
            format_timestamp(getattr(evidence, 'published_date', None) or evidence.get('published_date', None) if isinstance(evidence, dict) else None),  # publication_date
            format_timestamp(timestamp),
            getattr(evidence, 'relevance_score', 0.0) or evidence.get('relevance_score', 0.0) if isinstance(evidence, dict) else 0.0,
            confidence,
            getattr(evidence, 'author', None) or evidence.get('author', None) if isinstance(evidence, dict) else None,
            getattr(evidence, 'title', None) or evidence.get('title', None) if isinstance(evidence, dict) else None,
            getattr(evidence, 'language', 'en') or evidence.get('language', 'en') if isinstance(evidence, dict) else 'en',
            len(text_snippet.split()) if text_snippet else 0,
            getattr(evidence, 'sentiment', 0.0) or evidence.get('sentiment', 0.0) if isinstance(evidence, dict) else 0.0,
            json.dumps(getattr(evidence, 'cultural_context', {}) or evidence.get('cultural_context', {}) if isinstance(evidence, dict) else {}),
            json.dumps(getattr(evidence, 'relationships', []) or evidence.get('relationships', []) if isinstance(evidence, dict) else []),
            getattr(evidence, 'agent_id', None) if hasattr(evidence, 'agent_id') else (evidence.get('agent_id', None) if isinstance(evidence, dict) else None),
            format_timestamp(getattr(evidence, 'published_date', None) or evidence.get('published_date', None) if isinstance(evidence, dict) else None)  # published_date (duplicate)
        ))

    def _store_dimension(self, cursor, destination_id: str, name: str, value: DimensionValue):
        """Store a dimension value"""
        cursor.execute("""
            INSERT OR REPLACE INTO dimensions
            (destination_id, name, value, confidence, source, last_updated)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (
            destination_id,
            name,
            value.value,
            value.confidence,
            getattr(value, 'source', None),
            value.last_updated.isoformat() if value.last_updated else None
        ))

    def _store_temporal_slice(self, cursor, destination_id: str, temporal_slice: TemporalSlice):
        """Store a single temporal slice."""
        self.logger.debug(f"Storing temporal slice for destination {destination_id}")
        cursor.execute("""
            INSERT INTO temporal_slices (
                destination_id, start_date, end_date, slice_type, average_price_level,
                weather_data, crowd_level, special_events, local_insights
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            destination_id,
            getattr(temporal_slice, 'valid_from', None),
            getattr(temporal_slice, 'valid_to', None),
            getattr(temporal_slice, 'season', 'unknown'),
            temporal_slice.average_price_level,
            json.dumps(getattr(temporal_slice, 'weather_patterns', {})),
            getattr(temporal_slice, 'crowd_level', 'unknown'),
            # Proper type checking for special_events
            json.dumps([
                event if isinstance(event, dict)
                else event.to_dict() if hasattr(event, 'to_dict') and callable(getattr(event, 'to_dict'))
                else str(event) if event is not None
                else {"error": "null_event"}
                for event in temporal_slice.special_events
            ]),
            json.dumps(getattr(temporal_slice, 'seasonal_highlights', {}))
        ))
        self.logger.debug(f"Temporal slice for {destination_id} stored successfully.")

    def _store_poi(self, cursor, destination_id: str, poi):
        """Store a single point of interest."""
        cursor.execute("""
            INSERT OR REPLACE INTO pois
            (poi_id, destination_id, name, poi_type, coordinates, description,
             rating, price_level, operating_hours, contact_info)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            poi.poi_id,
            destination_id,
            poi.name,
            poi.poi_type,
            json.dumps(getattr(poi, 'location', {})),
            poi.description,
            poi.rating,
            getattr(poi, 'price_range', None),
            json.dumps(poi.operating_hours) if poi.operating_hours else None,
            getattr(poi, 'contact_info', None)
        ))

    def get_destination_themes(
        self, 
        destination_id: str, 
        min_confidence: float = 0.5,
        macro_category: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Retrieve themes for a destination, optionally filtered by confidence and category"""
        themes_data = []
        query = """
            SELECT
                theme_id, macro_category, micro_category, name, description, fit_score,
                confidence_breakdown, confidence_level, tags, created_date, last_validated,
                authentic_insights, local_authorities, seasonal_relevance, regional_uniqueness, insider_tips, factors, cultural_summary, sentiment_analysis, temporal_analysis
            FROM themes
            WHERE destination_id = ?
        """
        params = [destination_id]

        if macro_category:
            query += " AND macro_category = ?"
            params.append(macro_category)

        cursor = self.conn.cursor()
        cursor.execute(query, tuple(params))
        rows = cursor.fetchall()

        for row in rows:
            (theme_id, macro_category, micro_category, name, description, fit_score,
             confidence_breakdown_json, confidence_level_str, tags_json, created_date_str,
             last_validated_str, authentic_insights_json, local_authorities_json, 
             seasonal_relevance_json, regional_uniqueness, insider_tips_json, factors_json, cultural_summary_json, sentiment_analysis_json, temporal_analysis_json) = row

            confidence_breakdown = ConfidenceBreakdown(**json.loads(confidence_breakdown_json)) if confidence_breakdown_json else None
            confidence_level = ConfidenceLevel(confidence_level_str) if confidence_level_str else None
            tags = json.loads(tags_json) if tags_json else []
            created_date = datetime.fromisoformat(created_date_str) if created_date_str else None
            last_validated = datetime.fromisoformat(last_validated_str) if last_validated_str else None
            authentic_insights = []
            if authentic_insights_json:
                try:
                    ai_data_list = json.loads(authentic_insights_json)
                    for ai_data in ai_data_list:
                        # Ensure all required fields are present with defaults
                        ai_dict = {
                            'insight_type': InsightType(ai_data.get('insight_type', 'practical')),
                            'authenticity_score': ai_data.get('authenticity_score', 0.0),
                            'uniqueness_score': ai_data.get('uniqueness_score', 0.0),
                            'actionability_score': ai_data.get('actionability_score', 0.0),
                            'temporal_relevance': ai_data.get('temporal_relevance', 0.0),
                            'location_exclusivity': LocationExclusivity(ai_data.get('location_exclusivity', 'common')),
                            'seasonal_window': SeasonalWindow(**ai_data['seasonal_window']) if ai_data.get('seasonal_window') else None,
                            'local_validation_count': ai_data.get('local_validation_count', 0)
                        }
                        authentic_insights.append(AuthenticInsight(**ai_dict))
                except Exception as e:
                    self.logger.warning(f"Error deserializing authentic_insights: {e}")
                    
            local_authorities = []
            if local_authorities_json:
                try:
                    la_data_list = json.loads(local_authorities_json)
                    for la_data in la_data_list:
                        # Ensure all required fields are present with defaults
                        la_dict = {
                            'authority_type': AuthorityType(la_data.get('authority_type', 'resident')),
                            'local_tenure': la_data.get('local_tenure'),
                            'expertise_domain': la_data.get('expertise_domain', ''),
                            'community_validation': la_data.get('community_validation', 0.0)
                        }
                        local_authorities.append(LocalAuthority(**la_dict))
                except Exception as e:
                    self.logger.warning(f"Error deserializing local_authorities: {e}")

            theme_obj = Theme(
                theme_id=theme_id,
                macro_category=macro_category,
                micro_category=micro_category,
                name=name,
                description=description,
                fit_score=fit_score,
                confidence_breakdown=confidence_breakdown,
                tags=tags,
                created_date=created_date,
                last_validated=last_validated,
                authentic_insights=authentic_insights, # New field
                local_authorities=local_authorities, # New field
                seasonal_relevance=json.loads(seasonal_relevance_json) if seasonal_relevance_json else {}, # New field
                regional_uniqueness=regional_uniqueness, # New field
                insider_tips=json.loads(insider_tips_json) if insider_tips_json else [], # New field
                factors=json.loads(factors_json) if factors_json else {}, # New field
                cultural_summary=json.loads(cultural_summary_json) if cultural_summary_json else {}, # New field
                sentiment_analysis=json.loads(sentiment_analysis_json) if sentiment_analysis_json else {}, # New field
                temporal_analysis=json.loads(temporal_analysis_json) if temporal_analysis_json else {} # New field
            )
            
            if theme_obj.confidence_breakdown and theme_obj.confidence_breakdown.overall_confidence >= min_confidence:
                # Defensive type checking for theme_obj
                if hasattr(theme_obj, 'to_dict') and callable(getattr(theme_obj, 'to_dict')):
                    themes_data.append(theme_obj.to_dict())
                else:
                    self.logger.warning(f"Theme object {theme_obj.theme_id} missing to_dict method, creating fallback dict")
                    themes_data.append({"error": "missing_to_dict_method", "theme_id": getattr(theme_obj, 'theme_id', 'unknown')})

        return themes_data

    def get_high_confidence_themes(
        self, 
        min_confidence: float = 0.8,
        limit: int = 100
    ) -> List[Dict[str, Any]]:
        """Retrieve high confidence themes from the database"""
        themes_data = []
        cursor = self.conn.cursor()
        # Retrieve all fields, then filter by confidence_level in Python
        # because SQL doesn't directly understand the Enum values.
        cursor.execute("""
            SELECT
                theme_id, macro_category, micro_category, name, description, fit_score,
                confidence_breakdown, confidence_level, tags, created_date, last_validated,
                authentic_insights, local_authorities, seasonal_relevance, regional_uniqueness, insider_tips, factors, cultural_summary, sentiment_analysis, temporal_analysis
            FROM themes
            LIMIT ?
        """, (limit,))
        rows = cursor.fetchall()

        for row in rows:
            (theme_id, macro_category, micro_category, name, description, fit_score,
             confidence_breakdown_json, confidence_level_str, tags_json, created_date_str,
             last_validated_str, authentic_insights_json, local_authorities_json,
             seasonal_relevance_json, regional_uniqueness, insider_tips_json, factors_json, cultural_summary_json, sentiment_analysis_json, temporal_analysis_json) = row

            confidence_breakdown = ConfidenceBreakdown(**json.loads(confidence_breakdown_json)) if confidence_breakdown_json else None
            confidence_level = ConfidenceLevel(confidence_level_str) if confidence_level_str else None
            tags = json.loads(tags_json) if tags_json else []
            created_date = datetime.fromisoformat(created_date_str) if created_date_str else None
            last_validated = datetime.fromisoformat(last_validated_str) if last_validated_str else None
            authentic_insights = []
            if authentic_insights_json:
                try:
                    ai_data_list = json.loads(authentic_insights_json)
                    for ai_data in ai_data_list:
                        # Ensure all required fields are present with defaults
                        ai_dict = {
                            'insight_type': InsightType(ai_data.get('insight_type', 'practical')),
                            'authenticity_score': ai_data.get('authenticity_score', 0.0),
                            'uniqueness_score': ai_data.get('uniqueness_score', 0.0),
                            'actionability_score': ai_data.get('actionability_score', 0.0),
                            'temporal_relevance': ai_data.get('temporal_relevance', 0.0),
                            'location_exclusivity': LocationExclusivity(ai_data.get('location_exclusivity', 'common')),
                            'seasonal_window': SeasonalWindow(**ai_data['seasonal_window']) if ai_data.get('seasonal_window') else None,
                            'local_validation_count': ai_data.get('local_validation_count', 0)
                        }
                        authentic_insights.append(AuthenticInsight(**ai_dict))
                except Exception as e:
                    self.logger.warning(f"Error deserializing authentic_insights: {e}")
                    
            local_authorities = []
            if local_authorities_json:
                try:
                    la_data_list = json.loads(local_authorities_json)
                    for la_data in la_data_list:
                        # Ensure all required fields are present with defaults
                        la_dict = {
                            'authority_type': AuthorityType(la_data.get('authority_type', 'resident')),
                            'local_tenure': la_data.get('local_tenure'),
                            'expertise_domain': la_data.get('expertise_domain', ''),
                            'community_validation': la_data.get('community_validation', 0.0)
                        }
                        local_authorities.append(LocalAuthority(**la_dict))
                except Exception as e:
                    self.logger.warning(f"Error deserializing local_authorities: {e}")

            theme_obj = Theme(
                theme_id=theme_id,
                macro_category=macro_category,
                micro_category=micro_category,
                name=name,
                description=description,
                fit_score=fit_score,
                confidence_breakdown=confidence_breakdown,
                tags=tags,
                created_date=created_date,
                last_validated=last_validated,
                authentic_insights=authentic_insights, # New field
                local_authorities=local_authorities, # New field
                seasonal_relevance=json.loads(seasonal_relevance_json) if seasonal_relevance_json else {}, # New field
                regional_uniqueness=regional_uniqueness, # New field
                insider_tips=json.loads(insider_tips_json) if insider_tips_json else [], # New field
                factors=json.loads(factors_json) if factors_json else {}, # New field
                cultural_summary=json.loads(cultural_summary_json) if cultural_summary_json else {}, # New field
                sentiment_analysis=json.loads(sentiment_analysis_json) if sentiment_analysis_json else {}, # New field
                temporal_analysis=json.loads(temporal_analysis_json) if temporal_analysis_json else {} # New field
            )
            
            if theme_obj.confidence_breakdown and theme_obj.confidence_breakdown.overall_confidence >= min_confidence:
                # Defensive type checking for theme_obj
                if hasattr(theme_obj, 'to_dict') and callable(getattr(theme_obj, 'to_dict')):
                    themes_data.append(theme_obj.to_dict())
                else:
                    self.logger.warning(f"Theme object {theme_obj.theme_id} missing to_dict method, creating fallback dict")
                    themes_data.append({"error": "missing_to_dict_method", "theme_id": getattr(theme_obj, 'theme_id', 'unknown')})

        return themes_data

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

    def export_from_normalized_schema(self, destination_id: str) -> Dict[str, Any]:
        """
        Export destination data directly from normalized database relationships
        This bypasses object loading and provides maximum efficiency
        """
        if not self.conn:
            raise ValueError("No database connection")
            
        cursor = self.conn.cursor()
        
        try:
            # Get destination basic info
            cursor.execute("""
                SELECT id, names, admin_levels, timezone, population, country_code,
                       core_geo, lineage, meta, last_updated, destination_revision
                FROM destinations WHERE id = ?
            """, (destination_id,))
            
            dest_row = cursor.fetchone()
            if not dest_row:
                raise ValueError(f"Destination {destination_id} not found")
            
            destination_data = {
                "id": dest_row[0],
                "names": json.loads(dest_row[1]),
                "admin_levels": json.loads(dest_row[2]),
                "timezone": dest_row[3],
                "population": dest_row[4],
                "country_code": dest_row[5],
                "core_geo": json.loads(dest_row[6]) if dest_row[6] else {},
                "lineage": json.loads(dest_row[7]) if dest_row[7] else {},
                "meta": json.loads(dest_row[8]) if dest_row[8] else {},
                "last_updated": dest_row[9],
                "destination_revision": dest_row[10]
            }
            
            # Get all evidence (normalized source of truth)
            cursor.execute("""
                SELECT id, source_url, source_category, evidence_type, authority_weight,
                       text_snippet, timestamp, confidence, sentiment, cultural_context,
                       relationships, agent_id, published_date, factors
                FROM evidence WHERE destination_id = ?
            """, (destination_id,))
            
            evidence_registry = {}
            for ev_row in cursor.fetchall():
                evidence_registry[ev_row[0]] = {
                    "id": ev_row[0],
                    "source_url": ev_row[1],
                    "source_category": ev_row[2],
                    "evidence_type": ev_row[3],
                    "authority_weight": ev_row[4],
                    "text_snippet": ev_row[5],
                    "timestamp": ev_row[6],
                    "confidence": ev_row[7],
                    "sentiment": ev_row[8],
                    "cultural_context": json.loads(ev_row[9]) if ev_row[9] else {},
                    "relationships": json.loads(ev_row[10]) if ev_row[10] else [],
                    "agent_id": ev_row[11],
                    "published_date": ev_row[12],
                    "factors": json.loads(ev_row[13]) if ev_row[13] else {}
                }
            
            # Get themes with evidence references
            cursor.execute("""
                SELECT t.theme_id, t.macro_category, t.micro_category, t.name, t.description,
                       t.fit_score, t.tags, t.sentiment_analysis, t.temporal_analysis,
                       t.confidence_level, t.confidence_breakdown, t.authentic_insights,
                       t.local_authorities, t.source_evidence_ids
                FROM themes t WHERE t.destination_id = ?
            """, (destination_id,))
            
            themes_data = {}
            theme_evidence_relationships = {}
            
            for theme_row in cursor.fetchall():
                theme_id = theme_row[0]
                themes_data[theme_id] = {
                    "theme_id": theme_id,
                    "macro_category": theme_row[1],
                    "micro_category": theme_row[2],
                    "name": theme_row[3],
                    "description": theme_row[4],
                    "fit_score": theme_row[5],
                    "tags": json.loads(theme_row[6]) if theme_row[6] else [],
                    "sentiment_analysis": json.loads(theme_row[7]) if theme_row[7] else {},
                    "temporal_analysis": json.loads(theme_row[8]) if theme_row[8] else {},
                    "confidence_level": theme_row[9],
                    "confidence_breakdown": json.loads(theme_row[10]) if theme_row[10] else {},
                    "authentic_insights": json.loads(theme_row[11]) if theme_row[11] else [],
                    "local_authorities": json.loads(theme_row[12]) if theme_row[12] else [],
                    "evidence_references": json.loads(theme_row[13]) if theme_row[13] else []
                }
                
                # Get evidence relationships from normalized table
                cursor.execute("""
                    SELECT evidence_id FROM theme_evidence WHERE theme_id = ?
                """, (theme_id,))
                evidence_ids = [row[0] for row in cursor.fetchall()]
                theme_evidence_relationships[theme_id] = evidence_ids
            
            # Get relationships from normalized tables
            cursor.execute("""
                SELECT theme_id, insight_id, relationship_strength
                FROM theme_insight_relationships WHERE theme_id IN ({})
            """.format(','.join('?' for _ in themes_data.keys())), list(themes_data.keys()))
            
            theme_insight_relationships = {}
            for rel_row in cursor.fetchall():
                theme_id = rel_row[0]
                if theme_id not in theme_insight_relationships:
                    theme_insight_relationships[theme_id] = []
                theme_insight_relationships[theme_id].append({
                    "insight_id": rel_row[1],
                    "strength": rel_row[2]
                })
            
            return {
                "export_metadata": {
                    "version": "3.1",
                    "export_method": "normalized_schema_direct",
                    "export_timestamp": datetime.now().isoformat(),
                    "format": "reference_based_optimized"
                },
                "destination": destination_data,
                "data": {
                    "evidence": evidence_registry,
                    "themes": themes_data,
                },
                "relationships": {
                    "theme_evidence": theme_evidence_relationships,
                    "theme_insights": theme_insight_relationships,
                },
                "optimization_stats": {
                    "evidence_count": len(evidence_registry),
                    "themes_count": len(themes_data),
                    "relationships_count": len(theme_evidence_relationships) + len(theme_insight_relationships)
                }
            }
            
        except sqlite3.Error as e:
            self.logger.error(f"Error exporting from normalized schema: {e}")
            raise 

    def get_destination_by_name(self, name: str) -> Optional[Destination]:
        """Retrieves and reconstructs a full Destination object by its primary name."""
        if not self.conn:
            self.logger.error("Database connection is not open.")
            return None
            
        cursor = self.conn.cursor()
        cursor.execute("SELECT * FROM destinations WHERE name = ?;", (name,))
        dest_row = cursor.fetchone()
        
        if not dest_row:
            self.logger.warning(f"No destination found with name: {name}")
            return None

        # Fetch column names from cursor description
        columns = [description[0] for description in cursor.description]
        dest_data = dict(zip(columns, dest_row))
        
        # Reconstruct Destination object with correct type handling
        dest = Destination(
            id=dest_data.get('id'),
            names=[dest_data.get('name')],  # Treat 'name' as a plain string
            country_code=dest_data.get('country_code'),
            population=dest_data.get('population'),
            area_km2=dest_data.get('area_km2'),
            primary_language=dest_data.get('primary_language'),
            hdi=dest_data.get('hdi'),
            gdp_per_capita_usd=dest_data.get('gdp_per_capita_usd'),
            vibe_descriptors=json.loads(dest_data.get('vibe_descriptors') or '[]'),
            historical_summary=dest_data.get('historical_summary'),
            unesco_sites=json.loads(dest_data.get('unesco_sites') or '[]'),
            annual_tourist_arrivals=dest_data.get('annual_tourist_arrivals'),
            popularity_stage=dest_data.get('popularity_stage'),
            visa_info_url=dest_data.get('visa_info_url'),
            last_updated=datetime.fromisoformat(dest_data.get("last_updated")) if dest_data.get("last_updated") else datetime.now(),
            admin_levels=json.loads(dest_data.get('admin_levels') or '{}'),
            core_geo=json.loads(dest_data.get('core_geo') or '{}'),
            timezone=dest_data.get('timezone')
        )
        
        # Add the enrichment fields that might not be in constructor
        dest.dominant_religions = json.loads(dest_data.get('dominant_religions') or '[]')

        cursor.execute("SELECT * FROM themes WHERE destination_id = ?;", (dest.id,))
        theme_rows = cursor.fetchall()
        theme_columns = [desc[0] for desc in cursor.description]

        for row in theme_rows:
            theme_data = dict(zip(theme_columns, row))
            dest.themes.append(Theme(
                theme_id=theme_data.get('theme_id'), 
                name=theme_data.get('name'), 
                description=theme_data.get('description'), 
                macro_category=theme_data.get('macro_category'), 
                micro_category=theme_data.get('micro_category'), 
                fit_score=theme_data.get('fit_score', 0.0)
            ))
        
        self.logger.info(f"Successfully loaded destination '{name}' with {len(dest.themes)} themes.")
        return dest 

    def __del__(self):
        self.close_db() 