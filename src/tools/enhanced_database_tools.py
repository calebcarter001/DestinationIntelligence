"""
Enhanced database tools for storing destination insights with full enhanced data model support
"""

import logging
from typing import Dict, Any, Optional, List
from langchain.tools import Tool
import hashlib
import os
from datetime import datetime

from ..core.enhanced_database_manager import EnhancedDatabaseManager
from ..core.enhanced_data_models import (
    Destination, Theme, Evidence,
    AuthenticInsight, LocalAuthority, SeasonalWindow
)
from ..core.confidence_scoring import ConfidenceBreakdown, ConfidenceLevel
from ..schemas import InsightType, LocationExclusivity, AuthorityType

class StoreEnhancedDestinationInsightsTool(Tool):
    """Enhanced tool for storing destination insights with JSON export"""
    
    def __init__(self, db_manager: EnhancedDatabaseManager):
        """
        Initialize the enhanced store destination insights tool
        
        Args:
            db_manager: Enhanced database manager instance with JSON export
        """
        logger = logging.getLogger(__name__)
        
        # Create wrapper functions that capture the db_manager
        def create_tool_func(db_mgr):
            def _store_insights(destination_data: Dict[str, Any] = None,
                              config: Any = None, run_manager: Any = None, **kwargs) -> str:
                """
                Store enhanced destination insights with all new features
                
                Args:
                    destination_data: Dictionary containing destination object and metadata
                    config: LangChain RunnableConfig (not used)
                    run_manager: LangChain callback manager (not used)
                    
                Returns:
                    Status message with storage results and JSON export paths
                """
                try:
                    if destination_data is None:
                        return "Error: No destination data provided"
                    
                    # Handle Destination object and dict input
                    if isinstance(destination_data, Destination):
                        logger.info("DEBUG: Path 1 - destination_data is already a Destination object")
                        destination = destination_data
                        analysis_metadata = {}
                    elif isinstance(destination_data, dict):
                        logger.info("DEBUG: Path 2 - destination_data is a dictionary")
                        # Check if it's wrapped with metadata
                        if "destination" in destination_data and isinstance(destination_data["destination"], Destination):
                            logger.info("DEBUG: Path 2a - destination key contains Destination object")
                            destination = destination_data["destination"]
                            analysis_metadata = destination_data.get("analysis_metadata", {})
                        elif "destination" in destination_data and isinstance(destination_data["destination"], dict):
                            logger.info("DEBUG: Path 2b - destination key contains dictionary")
                            # Handle case where analysis result contains evidence_registry
                            destination_dict = destination_data["destination"]
                            logger.info(f"DEBUG: destination_dict keys: {list(destination_dict.keys())}")
                            logger.info(f"DEBUG: themes in destination_dict: {len(destination_dict.get('themes', []))}")
                            
                            destination = Destination(
                                names=[destination_dict.get("destination_name", "Unknown")],
                                id=f"dest_{destination_dict.get('destination_name', 'unknown').lower().replace(' ', '_')}",
                                country_code=destination_dict.get("country_code", "US"),
                                timezone="UTC",
                                admin_levels={}
                            )
                            
                            # CRITICAL FIX: Add ALL attributes to the destination object!
                            
                            # 1. Add themes (already working)
                            themes_data = destination_dict.get("themes", [])
                            logger.info(f"DEBUG: Found {len(themes_data)} themes to add to destination")
                            if themes_data:
                                # Convert theme dictionaries to Theme objects if needed
                                from ..core.enhanced_data_models import Theme
                                from ..core.evidence_hierarchy import SourceCategory, EvidenceType
                                themes_added = 0
                                for i, theme_data in enumerate(themes_data):
                                    try:
                                        if isinstance(theme_data, dict):
                                            logger.info(f"DEBUG: Converting theme {i}: {theme_data.get('name', 'Unknown')}")
                                            
                                            # Reconstruct ConfidenceBreakdown object
                                            cb_obj = None
                                            confidence_breakdown_dict = theme_data.get("confidence_breakdown")
                                            if confidence_breakdown_dict and isinstance(confidence_breakdown_dict, dict):
                                                try:
                                                    # Make a copy to modify for initialization
                                                    cb_dict_for_init = confidence_breakdown_dict.copy()
                                                    
                                                    # Convert confidence_level string to Enum member
                                                    level_str = cb_dict_for_init.get("confidence_level")
                                                    if isinstance(level_str, str):
                                                        try:
                                                            cb_dict_for_init["confidence_level"] = ConfidenceLevel(level_str)
                                                        except ValueError as e_val:
                                                            logger.error(f"DEBUG: Invalid string for ConfidenceLevel enum '{level_str}': {e_val}. Defaulting in ConfidenceBreakdown.")
                                                    
                                                    cb_obj = ConfidenceBreakdown(**cb_dict_for_init)
                                                except Exception as e_cb:
                                                    logger.error(f"DEBUG: Error reconstructing ConfidenceBreakdown for theme {theme_data.get('name')}: {e_cb}")

                                            # Reconstruct Evidence objects
                                            reconstructed_evidence_list = []
                                            evidence_data_list = theme_data.get("evidence", [])
                                            if isinstance(evidence_data_list, list):
                                                for ev_data in evidence_data_list:
                                                    if isinstance(ev_data, dict):
                                                        try:
                                                            # Attempt to parse datetime if it's a string
                                                            ts_val = ev_data.get("timestamp")
                                                            timestamp_obj = datetime.fromisoformat(ts_val) if isinstance(ts_val, str) else datetime.now()
                                                            
                                                            pub_date_val = ev_data.get("published_date")
                                                            published_date_obj = None
                                                            if pub_date_val:
                                                                try:
                                                                    published_date_obj = datetime.fromisoformat(pub_date_val) if isinstance(pub_date_val, str) else None
                                                                except ValueError: # handle cases where it might not be a valid ISO format
                                                                    logger.warning(f"Could not parse published_date: {pub_date_val}")


                                                            reconstructed_evidence_list.append(
                                                                Evidence(
                                                                    id=ev_data.get("id", ""),
                                                                    source_url=ev_data.get("source_url", ""),
                                                                    source_category=SourceCategory(ev_data.get("source_category", "unknown")),
                                                                    evidence_type=EvidenceType(ev_data.get("evidence_type", "text")),
                                                                    authority_weight=float(ev_data.get("authority_weight", 0.0)),
                                                                    text_snippet=ev_data.get("text_snippet", ""),
                                                                    timestamp=timestamp_obj,
                                                                    confidence=float(ev_data.get("confidence", 0.0)),
                                                                    sentiment=float(ev_data.get("sentiment", 0.0)) if ev_data.get("sentiment") is not None else None,
                                                                    cultural_context=ev_data.get("cultural_context"),
                                                                    relationships=ev_data.get("relationships", []),
                                                                    agent_id=ev_data.get("agent_id"),
                                                                    published_date=published_date_obj,
                                                                    factors=ev_data.get("factors")
                                                                )
                                                            )
                                                        except Exception as e_ev:
                                                            logger.error(f"DEBUG: Error reconstructing Evidence for theme {theme_data.get('name')}: {e_ev}")
                                                    elif isinstance(ev_data, Evidence): # if it's already an Evidence object
                                                        reconstructed_evidence_list.append(ev_data)


                                            # Reconstruct AuthenticInsight objects
                                            reconstructed_authentic_insights = []
                                            ai_list_data = theme_data.get("authentic_insights", [])
                                            if isinstance(ai_list_data, list):
                                                for ai_data in ai_list_data:
                                                    if isinstance(ai_data, dict):
                                                        try:
                                                            ai_dict_for_init = ai_data.copy()
                                                            # Convert insight_type string to Enum
                                                            it_str = ai_dict_for_init.get("insight_type")
                                                            if isinstance(it_str, str):
                                                                try:
                                                                    ai_dict_for_init["insight_type"] = InsightType(it_str)
                                                                except ValueError:
                                                                    logger.warning(f"Invalid insight_type string '{it_str}', defaulting.")
                                                                    ai_dict_for_init["insight_type"] = InsightType.UNKNOWN # Or your default
                                                            
                                                            # Convert location_exclusivity string to Enum
                                                            le_str = ai_dict_for_init.get("location_exclusivity")
                                                            if isinstance(le_str, str):
                                                                try:
                                                                    ai_dict_for_init["location_exclusivity"] = LocationExclusivity(le_str)
                                                                except ValueError:
                                                                    logger.warning(f"Invalid location_exclusivity string '{le_str}', defaulting.")
                                                                    ai_dict_for_init["location_exclusivity"] = LocationExclusivity.COMMON # Or your default

                                                            # Handle nested SeasonalWindow if it's also a dict
                                                            sw_data = ai_dict_for_init.get("seasonal_window")
                                                            if isinstance(sw_data, dict):
                                                                ai_dict_for_init["seasonal_window"] = SeasonalWindow(**sw_data)
                                                            elif not (sw_data is None or isinstance(sw_data, SeasonalWindow)):
                                                                logger.warning(f"Unexpected type for seasonal_window in AuthenticInsight: {type(sw_data)}")
                                                                ai_dict_for_init["seasonal_window"] = None
                                                                
                                                            reconstructed_authentic_insights.append(AuthenticInsight(**ai_dict_for_init))
                                                        except Exception as e_ai:
                                                            logger.error(f"DEBUG: Error reconstructing AuthenticInsight for theme {theme_data.get('name')}: {e_ai}")
                                                    elif isinstance(ai_data, AuthenticInsight):
                                                         reconstructed_authentic_insights.append(ai_data)


                                            # Reconstruct LocalAuthority objects
                                            reconstructed_local_authorities = []
                                            la_list_data = theme_data.get("local_authorities", [])
                                            if isinstance(la_list_data, list):
                                                for la_data in la_list_data:
                                                    if isinstance(la_data, dict):
                                                        try:
                                                            la_dict_for_init = la_data.copy()
                                                            # Convert authority_type string to Enum
                                                            at_str = la_dict_for_init.get("authority_type")
                                                            if isinstance(at_str, str):
                                                                try:
                                                                    la_dict_for_init["authority_type"] = AuthorityType(at_str)
                                                                except ValueError:
                                                                    logger.warning(f"Invalid authority_type string '{at_str}', defaulting.")
                                                                    la_dict_for_init["authority_type"] = AuthorityType.OTHER # Or your default
                                                            reconstructed_local_authorities.append(LocalAuthority(**la_dict_for_init))
                                                        except Exception as e_la:
                                                            logger.error(f"DEBUG: Error reconstructing LocalAuthority for theme {theme_data.get('name')}: {e_la}")
                                                    elif isinstance(la_data, LocalAuthority):
                                                        reconstructed_local_authorities.append(la_data)
                                            
                                            theme = Theme(
                                                theme_id=theme_data.get("theme_id", f"theme_{i}_{theme_data.get('name', 'unknown').lower().replace(' ', '_')}") ,
                                                name=theme_data.get("name", "Unknown Theme"),
                                                macro_category=theme_data.get("macro_category", "General"),
                                                micro_category=theme_data.get("micro_category", theme_data.get("category", "General")),
                                                description=theme_data.get("description", f"Theme about {theme_data.get('name', 'unknown topic')}" ),
                                                fit_score=float(theme_data.get("fit_score", 0.5)),
                                                confidence_breakdown=cb_obj,
                                                evidence=reconstructed_evidence_list,
                                                tags=theme_data.get("tags", []),
                                                created_date=datetime.fromisoformat(theme_data.get("created_date")) if theme_data.get("created_date") and isinstance(theme_data.get("created_date"), str) else datetime.now(),
                                                last_validated=datetime.fromisoformat(theme_data.get("last_validated")) if theme_data.get("last_validated") and isinstance(theme_data.get("last_validated"), str) else None,
                                                metadata=theme_data.get("metadata", {}),
                                                authentic_insights=reconstructed_authentic_insights,
                                                local_authorities=reconstructed_local_authorities,
                                                seasonal_relevance=theme_data.get("seasonal_relevance", {}),
                                                regional_uniqueness=float(theme_data.get("regional_uniqueness", 0.0)),
                                                insider_tips=theme_data.get("insider_tips", []),
                                                factors=theme_data.get("factors", {}),
                                                cultural_summary=theme_data.get("cultural_summary", {}),
                                                sentiment_analysis=theme_data.get("sentiment_analysis", {}),
                                                temporal_analysis=theme_data.get("temporal_analysis", {})
                                            )
                                            destination.themes.append(theme)
                                            themes_added += 1
                                            logger.info(f"DEBUG: Successfully added theme {i}: {theme.name}")
                                        elif hasattr(theme_data, 'insight_name'):  # DestinationInsight object
                                            logger.info(f"DEBUG: Converting DestinationInsight {i}: {theme_data.insight_name}")
                                            # Create Theme object from DestinationInsight
                                            theme = Theme(
                                                theme_id=f"theme_{i}_{theme_data.insight_name.lower().replace(' ', '_').replace(',', '')}",
                                                name=theme_data.insight_name,
                                                macro_category=getattr(theme_data, 'insight_type', 'General'),
                                                micro_category=getattr(theme_data, 'priority_category', 'General'),
                                                description=getattr(theme_data, 'description', f"Theme about {theme_data.insight_name}"),
                                                fit_score=getattr(theme_data, 'confidence_score', 0.5) or 0.5
                                            )
                                            # Add evidence if present
                                            if hasattr(theme_data, 'evidence') and theme_data.evidence:
                                                theme.evidence = theme_data.evidence
                                            destination.themes.append(theme)
                                            themes_added += 1
                                            logger.info(f"DEBUG: Successfully added DestinationInsight as theme {i}: {theme.name}")
                                        elif hasattr(theme_data, 'name'):  # Already a Theme object
                                            logger.info(f"DEBUG: Adding existing Theme object {i}: {theme_data.name}")
                                            destination.themes.append(theme_data)
                                            themes_added += 1
                                        else:
                                            logger.warning(f"DEBUG: Skipping invalid theme data {i}: {type(theme_data)}")
                                    except Exception as e:
                                        logger.error(f"DEBUG: Error converting theme {i}: {e}", exc_info=True)
                                        logger.error(f"DEBUG: Theme data was: {theme_data}")
                                
                                logger.info(f"DEBUG: Successfully added {themes_added} out of {len(themes_data)} themes to destination")
                                logger.info(f"DEBUG: Destination now has {len(destination.themes)} themes total")
                            
                            # 2. Add dimensions
                            dimensions_data = destination_dict.get("dimensions", {})
                            logger.info(f"DEBUG: Found {len(dimensions_data)} dimensions to add")
                            for dim_name, dim_value in dimensions_data.items():
                                if isinstance(dim_value, (int, float)):
                                    destination.update_dimension(dim_name, dim_value, confidence=0.8)
                                elif isinstance(dim_value, dict):
                                    # Handle more complex dimension format
                                    value = dim_value.get("value", dim_value.get("score", 0))
                                    unit = dim_value.get("unit", "score")
                                    confidence = dim_value.get("confidence", 0.8)
                                    destination.update_dimension(dim_name, value, unit, confidence)
                            
                            # 3. Add POIs
                            pois_data = destination_dict.get("pois", [])
                            logger.info(f"DEBUG: Found {len(pois_data)} POIs to add")
                            from ..core.enhanced_data_models import PointOfInterest
                            for poi_data in pois_data:
                                if isinstance(poi_data, dict):
                                    poi = PointOfInterest(
                                        poi_id=poi_data.get("poi_id", f"poi_{len(destination.pois)}"),
                                        name=poi_data.get("name", "Unknown POI"),
                                        description=poi_data.get("description", ""),
                                        location=poi_data.get("location", {}),
                                        address=poi_data.get("address"),
                                        poi_type=poi_data.get("poi_type", "attraction"),
                                        theme_tags=poi_data.get("theme_tags", []),
                                        ada_accessible=poi_data.get("ada_accessible"),
                                        ada_features=poi_data.get("ada_features", []),
                                        media_urls=poi_data.get("media_urls", []),
                                        operating_hours=poi_data.get("operating_hours"),
                                        price_range=poi_data.get("price_range"),
                                        rating=poi_data.get("rating"),
                                        review_count=poi_data.get("review_count")
                                    )
                                    destination.pois.append(poi)
                            
                            # 4. Add temporal slices
                            temporal_data = destination_dict.get("temporal_slices", [])
                            logger.info(f"DEBUG: Found {len(temporal_data)} temporal slices to add")
                            from ..core.enhanced_data_models import TemporalSlice
                            for temporal_item in temporal_data:
                                if isinstance(temporal_item, dict):
                                    # Parse date strings if needed
                                    valid_from = datetime.now()
                                    valid_to = None
                                    if "valid_from" in temporal_item:
                                        try:
                                            valid_from = datetime.fromisoformat(temporal_item["valid_from"])
                                        except:
                                            valid_from = datetime.now()
                                    if "valid_to" in temporal_item and temporal_item["valid_to"]:
                                        try:
                                            valid_to = datetime.fromisoformat(temporal_item["valid_to"])
                                        except:
                                            valid_to = None
                                    
                                    temporal_slice = TemporalSlice(
                                        valid_from=valid_from,
                                        valid_to=valid_to,
                                        season=temporal_item.get("season"),
                                        seasonal_highlights=temporal_item.get("seasonal_highlights", {}),
                                        special_events=temporal_item.get("special_events", []),
                                        weather_patterns=temporal_item.get("weather_patterns"),
                                        visitor_patterns=temporal_item.get("visitor_patterns")
                                    )
                                    destination.temporal_slices.append(temporal_slice)
                            
                            # 5. Add authentic insights
                            insights_data = destination_dict.get("authentic_insights", [])
                            logger.info(f"DEBUG: Found {len(insights_data)} authentic insights to add")
                            # Insights are already AuthenticInsight objects from the analyst
                            for insight in insights_data:
                                if hasattr(insight, 'insight_type'):  # Already an AuthenticInsight object
                                    destination.authentic_insights.append(insight)
                            
                            # 6. Add local authorities  
                            authorities_data = destination_dict.get("local_authorities", [])
                            logger.info(f"DEBUG: Found {len(authorities_data)} local authorities to add")
                            # Authorities are already LocalAuthority objects from the analyst
                            for authority in authorities_data:
                                if hasattr(authority, 'authority_type'):  # Already a LocalAuthority object
                                    destination.local_authorities.append(authority)
                            
                            logger.info(f"DEBUG: Final destination has {len(destination.themes)} themes, {len(destination.dimensions)} dimensions, {len(destination.pois)} POIs, {len(destination.temporal_slices)} temporal slices, {len(destination.authentic_insights)} insights, {len(destination.local_authorities)} authorities")
                            
                            analysis_metadata = {
                                "evidence_registry": destination_data.get("evidence_registry", {}),
                                "themes": destination_data.get("themes", []),
                                "quality_metrics": destination_data.get("quality_metrics", {}),
                                "evidence_summary": destination_data.get("evidence_summary", {})
                            }
                        else:
                            logger.error(f"DEBUG: Path 2c - Invalid destination data format. Keys: {list(destination_data.keys())}")
                            return f"Error: Invalid destination data format"
                    else:
                        logger.error(f"DEBUG: Path 3 - Invalid input type {type(destination_data)}")
                        return f"Error: Invalid input type {type(destination_data)}"
                    
                    # Validate destination before storage
                    logger.info(f"Storing enhanced destination: {destination.names[0] if destination.names else destination.id}")
                    logger.info(f"  - Themes: {len(destination.themes)}")
                    logger.info(f"  - Evidence total: {sum(len(t.evidence) for t in destination.themes)}")
                    logger.info(f"  - Dimensions: {len([d for d in destination.dimensions.values() if d.value is not None])}")
                    
                    # Store in enhanced database with JSON export
                    results = db_mgr.store_destination(destination, analysis_metadata)
                    
                    # Build comprehensive response message
                    if results["database_stored"]:
                        message_parts = [f"✅ Successfully stored enhanced insights for {destination.names[0] if destination.names else 'destination'}"]
                        
                        # Add detailed storage summary
                        if "storage_summary" in results:
                            summary = results["storage_summary"]
                            message_parts.append(f"📊 Stored: {summary['themes']} themes, {summary['evidence']} evidence pieces")
                            message_parts.append(f"📏 Dimensions: {summary['dimensions']} populated")
                            
                            if summary['temporal_slices'] > 0:
                                message_parts.append(f"🕐 Temporal slices: {summary['temporal_slices']}")
                            if summary['pois'] > 0:
                                message_parts.append(f"📍 POIs: {summary['pois']}")
                            if summary['insights'] > 0:
                                message_parts.append(f"💡 Authentic insights: {summary['insights']}")
                            if summary['authorities'] > 0:
                                message_parts.append(f"👥 Local authorities: {summary['authorities']}")
                        
                        # Add validation warnings if any
                        if results.get("validation_errors"):
                            message_parts.append(f"\n⚠️  Validation warnings ({len(results['validation_errors'])}):")
                            for error in results["validation_errors"][:3]:  # Show first 3
                                message_parts.append(f"  • {error}")
                            if len(results["validation_errors"]) > 3:
                                message_parts.append(f"  • ... and {len(results['validation_errors']) - 3} more")
                        
                        # Add storage warnings if any
                        if results.get("warnings"):
                            message_parts.append(f"\n⚠️  Storage warnings ({len(results['warnings'])}):")
                            for warning in results["warnings"][:3]:  # Show first 3
                                message_parts.append(f"  • {warning}")
                            if len(results["warnings"]) > 3:
                                message_parts.append(f"  • ... and {len(results['warnings']) - 3} more")
                        
                        # Add JSON export info
                        if results["json_files_created"]:
                            message_parts.append(f"\n📁 JSON files exported ({len(results['json_files_created'])}):")
                            for file_type, path in results["json_files_created"].items():
                                filename = os.path.basename(path)
                                message_parts.append(f"  • {file_type}: {filename}")
                        
                        # Add quality metrics
                        theme_confidence_levels = {}
                        for theme in destination.themes:
                            level = theme.get_confidence_level().value
                            theme_confidence_levels[level] = theme_confidence_levels.get(level, 0) + 1
                        
                        if theme_confidence_levels:
                            message_parts.append("\n📈 Quality metrics:")
                            for level, count in sorted(theme_confidence_levels.items()):
                                message_parts.append(f"  • {level} confidence: {count} themes")
                        
                        return "\n".join(message_parts)
                    else:
                        error_details = []
                        if results.get("errors"):
                            error_details.extend(results["errors"])
                        if results.get("validation_errors"):
                            error_details.extend([f"Validation: {err}" for err in results["validation_errors"]])
                        
                        error_msg = " | ".join(error_details) if error_details else "Unknown error"
                        return f"❌ Failed to store destination insights: {error_msg}"
                        
                except Exception as e:
                    logger.error(f"Error storing enhanced destination insights: {e}", exc_info=True)
                    return f"❌ Error storing enhanced destination insights: {str(e)}"
            
            async def _astore_insights(destination_data: Dict[str, Any] = None, 
                                     config: Any = None, run_manager: Any = None, **kwargs) -> str:
                """Async wrapper for LangChain compatibility"""
                
                logger.info(f"_astore_insights called with destination_data type: {type(destination_data)}")
                
                if destination_data is not None:
                    return _store_insights(destination_data=destination_data, config=config, run_manager=run_manager)
                else:
                    logger.error(f"No valid destination data found. destination_data={destination_data}, kwargs={kwargs}")
                    return "Error: No valid destination data provided"
            
            return _store_insights, _astore_insights
        
        # Create the functions with captured db_manager
        func, coroutine = create_tool_func(db_manager)
        
        super().__init__(
            name="store_destination_insights",
            description=(
                "Store enhanced destination insights in the database with evidence hierarchy, "
                "confidence scoring, temporal data, and automatic consolidated JSON export. "
                "Input should be a destination object with all enhanced data."
            ),
            func=func,
            coroutine=coroutine
        ) 