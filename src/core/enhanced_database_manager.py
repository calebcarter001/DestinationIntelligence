import sqlite3
import logging
import json
from datetime import datetime
from typing import Optional, List, Dict, Any
import os
import uuid

from .enhanced_data_models import Destination, Theme, Evidence, TemporalSlice, DimensionValue, AuthenticInsight, SeasonalWindow, LocalAuthority
from .confidence_scoring import ConfidenceBreakdown, ConfidenceLevel
from .json_export_manager import JsonExportManager
from src.schemas import InsightType, LocationExclusivity, AuthorityType

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
                    factors TEXT,           -- JSON object for additional factors
                    FOREIGN KEY (destination_id) REFERENCES destinations(id)
                )
            """)
            
            # Themes table with enhanced metadata
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS themes (
                    theme_id TEXT PRIMARY KEY,
                    destination_id TEXT,
                    macro_category TEXT,
                    micro_category TEXT,
                    name TEXT NOT NULL,
                    description TEXT,
                    fit_score REAL,
                    tags TEXT,  -- JSON array
                    sentiment_analysis TEXT,  -- JSON object
                    temporal_analysis TEXT,   -- JSON object
                    confidence_level TEXT,
                    confidence_breakdown TEXT,  -- JSON object
                    authentic_insights TEXT,    -- JSON array
                    local_authorities TEXT,     -- JSON array
                    source_evidence_ids TEXT,   -- JSON array
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
            
            # Dimensions table for quantified aspects
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
            
            # Temporal slices for time-based insights
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS temporal_slices (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    destination_id TEXT,
                    valid_from TIMESTAMP,
                    valid_to TIMESTAMP,
                    season TEXT,
                    seasonal_highlights TEXT,  -- JSON array
                    special_events TEXT,       -- JSON array
                    weather_patterns TEXT,     -- JSON object
                    visitor_patterns TEXT,     -- JSON object
                    FOREIGN KEY (destination_id) REFERENCES destinations(id)
                )
            """)
            
            # POIs table for points of interest
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS pois (
                    poi_id TEXT PRIMARY KEY,
                    destination_id TEXT,
                    name TEXT NOT NULL,
                    description TEXT,
                    location TEXT,  -- JSON object with lat/lng
                    address TEXT,
                    poi_type TEXT,
                    theme_tags TEXT,  -- JSON array
                    ada_accessible BOOLEAN,
                    ada_features TEXT,  -- JSON array
                    media_urls TEXT,   -- JSON array
                    operating_hours TEXT,  -- JSON object
                    price_range TEXT,
                    rating REAL,
                    review_count INTEGER,
                    FOREIGN KEY (destination_id) REFERENCES destinations(id)
                )
            """)
            
            # Authentic insights table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS authentic_insights (
                    id TEXT PRIMARY KEY,
                    destination_id TEXT,
                    insight_type TEXT,
                    authenticity_score REAL,
                    uniqueness_score REAL,
                    actionability_score REAL,
                    temporal_relevance REAL,
                    location_exclusivity TEXT,
                    seasonal_window_id INTEGER,
                    local_validation_count INTEGER,
                    FOREIGN KEY (destination_id) REFERENCES destinations(id),
                    FOREIGN KEY (seasonal_window_id) REFERENCES seasonal_windows(id)
                )
            """)
            
            # Seasonal windows table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS seasonal_windows (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    start_month INTEGER,
                    end_month INTEGER,
                    peak_weeks TEXT,  -- JSON array
                    booking_lead_time INTEGER,
                    specific_dates TEXT  -- JSON array
                )
            """)
            
            # Local authorities table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS local_authorities (
                    id TEXT PRIMARY KEY,
                    destination_id TEXT,
                    authority_type TEXT,
                    local_tenure INTEGER,
                    expertise_domain TEXT,
                    community_validation INTEGER,
                    FOREIGN KEY (destination_id) REFERENCES destinations(id)
                )
            """)
            
            # Create performance indices
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_evidence_destination ON evidence(destination_id)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_evidence_confidence ON evidence(confidence)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_evidence_source_category ON evidence(source_category)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_evidence_timestamp ON evidence(timestamp)")
            
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_themes_destination ON themes(destination_id)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_themes_category ON themes(macro_category)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_themes_confidence ON themes(confidence_level)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_themes_fit_score ON themes(fit_score)")
            
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_temporal_destination ON temporal_slices(destination_id)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_temporal_dates ON temporal_slices(valid_from, valid_to)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_temporal_season ON temporal_slices(season)")
            
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_pois_destination ON pois(destination_id)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_pois_type ON pois(poi_type)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_pois_rating ON pois(rating)")
            
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_dimensions_destination ON dimensions(destination_id)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_dimensions_name ON dimensions(dimension_name)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_dimensions_confidence ON dimensions(confidence)")
            
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_authentic_destination ON authentic_insights(destination_id)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_authentic_type ON authentic_insights(insight_type)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_authentic_scores ON authentic_insights(authenticity_score, uniqueness_score)")
            
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_authorities_destination ON local_authorities(destination_id)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_authorities_type ON local_authorities(authority_type)")
            
            self.conn.commit()
            self.logger.info("Enhanced database schema initialized with performance indices")
            
        except sqlite3.Error as e:
            self.logger.error(f"Failed to initialize database: {e}")
            raise

    def store_destination(self, destination: Destination, 
                         analysis_metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Store or update a complete destination with enhanced validation and error handling
        
        Args:
            destination: Destination object to store
            analysis_metadata: Optional metadata about the analysis
            
        Returns:
            Dictionary with storage results including DB status and JSON file paths
        """
        results = {
            "database_stored": False,
            "json_files_created": {},
            "errors": [],
            "warnings": [],
            "validation_errors": []
        }
        
        if not self.conn:
            results["errors"].append("No database connection")
            return results
        
        # Validate destination data before storage
        validation_errors = self._validate_destination(destination)
        if validation_errors:
            results["validation_errors"] = validation_errors
            self.logger.warning(f"Validation warnings for destination {destination.id}: {validation_errors}")
            
        try:
            # Begin transaction for atomic operations
            cursor = self.conn.cursor()
            cursor.execute("BEGIN TRANSACTION")
            
            # Store main destination with validation
            try:
                cursor.execute("""
                    INSERT OR REPLACE INTO destinations 
                    (id, names, admin_levels, timezone, population, country_code, 
                     core_geo, lineage, meta, last_updated, destination_revision)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    str(destination.id),
                    json.dumps(destination.names),
                    json.dumps(destination.admin_levels),
                    str(destination.timezone),
                    destination.population,
                    str(destination.country_code),
                    json.dumps(destination.core_geo),
                    json.dumps(destination.lineage),
                    json.dumps(destination.meta),
                    destination.last_updated.isoformat(),
                    destination.destination_revision
                ))
                self.logger.info(f"Stored destination record: {destination.id}")
            except sqlite3.Error as e:
                results["errors"].append(f"Failed to store destination record: {e}")
                cursor.execute("ROLLBACK")
                return results
            
            # Store themes and evidence with validation
            themes_stored = 0
            evidence_stored = 0
            insights_stored = 0
            authorities_stored = 0
            
            for theme in destination.themes:
                try:
                    self._store_theme(cursor, destination.id, theme)
                    themes_stored += 1
                    
                    # Store AuthenticInsights associated with the theme
                    for insight in theme.authentic_insights:
                        try:
                            seasonal_window_id = self._store_seasonal_window(cursor, insight.seasonal_window)
                            self._store_authentic_insight(cursor, destination.id, insight, seasonal_window_id)
                            insights_stored += 1
                        except sqlite3.Error as e:
                            results["warnings"].append(f"Failed to store insight for theme {theme.theme_id}: {e}")

                    # Store LocalAuthorities associated with the theme
                    for authority in theme.local_authorities:
                        try:
                            self._store_local_authority(cursor, destination.id, authority)
                            authorities_stored += 1
                        except sqlite3.Error as e:
                            results["warnings"].append(f"Failed to store authority for theme {theme.theme_id}: {e}")
                    
                    # Count evidence stored
                    evidence_stored += len(theme.evidence)
                    
                except sqlite3.Error as e:
                    results["warnings"].append(f"Failed to store theme {theme.theme_id}: {e}")
                    
            # Store dimensions with validation
            dimensions_stored = 0
            for dim_name, dim_value in destination.dimensions.items():
                if dim_value.value is not None:
                    try:
                        self._store_dimension(cursor, destination.id, dim_name, dim_value)
                        dimensions_stored += 1
                    except sqlite3.Error as e:
                        results["warnings"].append(f"Failed to store dimension {dim_name}: {e}")
                        
            # Store temporal slices with validation
            temporal_stored = 0
            for temporal_slice in destination.temporal_slices:
                try:
                    self._store_temporal_slice(cursor, destination.id, temporal_slice)
                    temporal_stored += 1
                except sqlite3.Error as e:
                    results["warnings"].append(f"Failed to store temporal slice: {e}")
                    
            # Store POIs with validation
            pois_stored = 0
            for poi in destination.pois:
                try:
                    self._store_poi(cursor, destination.id, poi)
                    pois_stored += 1
                except sqlite3.Error as e:
                    results["warnings"].append(f"Failed to store POI {poi.poi_id}: {e}")
                    
            # Commit transaction
            self.conn.commit()
            
            # Log storage summary
            self.logger.info(f"Successfully stored destination {destination.id}: "
                           f"{themes_stored} themes, {evidence_stored} evidence, "
                           f"{dimensions_stored} dimensions, {temporal_stored} temporal slices, "
                           f"{pois_stored} POIs, {insights_stored} insights, {authorities_stored} authorities")
            
            results["database_stored"] = True
            results["storage_summary"] = {
                "themes": themes_stored,
                "evidence": evidence_stored,
                "dimensions": dimensions_stored,
                "temporal_slices": temporal_stored,
                "pois": pois_stored,
                "insights": insights_stored,
                "authorities": authorities_stored
            }
            
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
            try:
                self.conn.rollback()
            except:
                pass
            
        except Exception as e:
            error_msg = f"Unexpected error storing destination {destination.id}: {e}"
            self.logger.error(error_msg, exc_info=True)
            results["errors"].append(error_msg)
            try:
                self.conn.rollback()
            except:
                pass
            
        return results

    def _validate_destination(self, destination: Destination) -> List[str]:
        """Validate destination data before storage"""
        errors = []
        
        # Check required fields
        if not destination.id:
            errors.append("Destination ID is required")
        if not destination.names or not destination.names[0]:
            errors.append("Destination must have at least one name")
        if not destination.country_code:
            errors.append("Country code is required")
        if not destination.timezone:
            errors.append("Timezone is required")
            
        # Validate themes
        for theme in destination.themes:
            if not theme.theme_id:
                errors.append(f"Theme missing ID: {theme.name}")
            if not theme.name:
                errors.append(f"Theme missing name for ID: {theme.theme_id}")
            if theme.fit_score < 0 or theme.fit_score > 1:
                errors.append(f"Theme {theme.name} has invalid fit_score: {theme.fit_score}")
                
            # Validate evidence
            for evidence in theme.evidence:
                if not evidence.id:
                    errors.append(f"Evidence missing ID in theme {theme.name}")
                if not evidence.source_url:
                    errors.append(f"Evidence missing source_url in theme {theme.name}")
                if evidence.confidence < 0 or evidence.confidence > 1:
                    errors.append(f"Evidence has invalid confidence in theme {theme.name}: {evidence.confidence}")
                    
        # Validate dimensions
        for dim_name, dim_value in destination.dimensions.items():
            if dim_value.confidence < 0 or dim_value.confidence > 1:
                errors.append(f"Dimension {dim_name} has invalid confidence: {dim_value.confidence}")
                
        return errors

    def _store_theme(self, cursor, destination_id: str, theme: Theme):
        """Store or update a theme"""
        cursor.execute("""
            INSERT OR REPLACE INTO themes 
            (theme_id, destination_id, macro_category, micro_category, name, description,
             fit_score, tags, sentiment_analysis, temporal_analysis, confidence_level,
             confidence_breakdown, authentic_insights, local_authorities, source_evidence_ids)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            theme.theme_id,
            destination_id,
            theme.macro_category,
            theme.micro_category,
            theme.name,
            theme.description,
            theme.fit_score,
            json.dumps(theme.tags),
            json.dumps(theme.sentiment_analysis) if theme.sentiment_analysis else None,
            json.dumps(theme.temporal_analysis) if theme.temporal_analysis else None,
            theme.get_confidence_level().value,
            json.dumps(theme.confidence_breakdown.to_dict()) if theme.confidence_breakdown else None,
            json.dumps([ai.to_dict() for ai in theme.authentic_insights]),
            json.dumps([la.to_dict() for la in theme.local_authorities]),
            json.dumps([evidence.id for evidence in theme.evidence])  # Store evidence IDs
        ))
        
        # Store evidence
        for evidence in theme.evidence:
            self._store_evidence(cursor, destination_id, evidence)
            
            # Link theme to evidence if table exists
            try:
                cursor.execute("""
                    INSERT OR IGNORE INTO theme_evidence (theme_id, evidence_id)
                    VALUES (?, ?)
                """, (theme.theme_id, evidence.id))
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
                """, (theme.theme_id, evidence.id))

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
            insight.insight_type.value,
            insight.authenticity_score,
            insight.uniqueness_score,
            insight.actionability_score,
            insight.temporal_relevance,
            insight.location_exclusivity.value,
            seasonal_window_id,
            insight.local_validation_count
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
            seasonal_window.start_month,
            seasonal_window.end_month,
            json.dumps(seasonal_window.peak_weeks),
            seasonal_window.booking_lead_time,
            json.dumps(seasonal_window.specific_dates)
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
            authority.authority_type.value,
            authority.local_tenure,
            authority.expertise_domain,
            authority.community_validation
        ))

    def _store_evidence(self, cursor, destination_id: str, evidence: Evidence):
        """Store or update evidence"""
        cursor.execute("""
            INSERT OR REPLACE INTO evidence 
            (id, destination_id, source_url, source_category, evidence_type,
             authority_weight, text_snippet, timestamp, confidence, sentiment,
             cultural_context, relationships, agent_id, published_date, factors)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
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
            evidence.published_date.isoformat() if evidence.published_date else None,
            json.dumps(evidence.factors) if evidence.factors else None
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
                themes_data.append(theme_obj.to_dict()) # Assuming to_dict is updated in Theme

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
                themes_data.append(theme_obj.to_dict())

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

    def close_db(self):
        """Close database connection"""
        if self.conn:
            self.conn.close()
            self.logger.info("Database connection closed.") 