"""
Enhanced database tools for storing destination insights with full enhanced data model support
"""

import logging
from typing import Dict, Any, Optional, List
from langchain.tools import Tool
import hashlib
import os

from ..core.enhanced_database_manager import EnhancedDatabaseManager
from ..core.enhanced_data_models import Destination, Theme

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
            def _store_insights(destination_name: str = None, insights: List[Any] = None,
                              destination_data: Dict[str, Any] = None,
                              config: Any = None, run_manager: Any = None, **kwargs) -> str:
                """
                Store enhanced destination insights with all new features
                
                Supports both legacy (destination_name + insights) and new (destination_data) formats.
                The config and run_manager parameters are required by LangChain but not used.
                
                Args:
                    destination_name: Legacy format - name of destination
                    insights: Legacy format - list of insights
                    destination_data: New format - Dictionary containing destination object or data
                    config: LangChain RunnableConfig (not used)
                    run_manager: LangChain callback manager (not used)
                    
                Returns:
                    Status message with storage results and JSON export paths
                """
                try:
                    # Handle backward compatibility first
                    if destination_name is not None and insights is not None:
                        # Convert basic format to legacy format
                        legacy_data = {
                            "name": destination_name,
                            "insights": insights
                        }
                        return _store_legacy_insights(legacy_data, db_mgr)
                    
                    # Handle new format
                    if destination_data is None:
                        # Try to extract from kwargs
                        if "destination_name" in kwargs and "insights" in kwargs:
                            legacy_data = {
                                "name": kwargs["destination_name"],
                                "insights": kwargs["insights"]
                            }
                            return _store_legacy_insights(legacy_data, db_mgr)
                        return "Error: No valid destination data provided"
                    
                    # Handle both Destination object and dict input
                    if isinstance(destination_data, Destination):
                        destination = destination_data
                        analysis_metadata = {}
                    elif isinstance(destination_data, dict):
                        # Check if it's wrapped with metadata
                        if "destination" in destination_data and isinstance(destination_data["destination"], Destination):
                            destination = destination_data["destination"]
                            analysis_metadata = destination_data.get("analysis_metadata", {})
                        else:
                            # Try to create Destination from dict (backward compatibility)
                            logger.warning("Received dictionary instead of Destination object - limited functionality")
                            return _store_legacy_insights(destination_data, db_mgr)
                    else:
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
            
            async def _astore_insights(destination_name: str = None, insights: List[Any] = None, 
                                     destination_data: Dict[str, Any] = None, 
                                     config: Any = None, run_manager: Any = None, **kwargs) -> str:
                """Async wrapper with backward compatibility and LangChain parameters"""
                # Note: config and run_manager are required by LangChain but we don't use them
                
                logger.info(f"_astore_insights called with: destination_name={destination_name}, "
                           f"insights={type(insights) if insights else None}, destination_data={destination_data}, "
                           f"kwargs={kwargs}")
                
                # Handle backward compatibility with basic tool interface
                if destination_name is not None and insights is not None:
                    # Convert basic format to enhanced format
                    legacy_data = {
                        "name": destination_name,
                        "insights": insights
                    }
                    logger.info(f"Using legacy format with destination_name={destination_name}")
                    # Pass as destination_data parameter
                    return _store_insights(destination_data=legacy_data, config=config, run_manager=run_manager)
                elif destination_data is not None:
                    # New enhanced format
                    logger.info("Using enhanced format with destination_data")
                    return _store_insights(destination_data=destination_data, config=config, run_manager=run_manager)
                else:
                    # Try to extract from kwargs
                    if "destination_name" in kwargs and "insights" in kwargs:
                        legacy_data = {
                            "name": kwargs["destination_name"],
                            "insights": kwargs["insights"]
                        }
                        logger.info("Using legacy format from kwargs")
                        return _store_insights(destination_data=legacy_data, config=config, run_manager=run_manager)
                    logger.error(f"No valid destination data found. Args: destination_name={destination_name}, "
                               f"insights={insights}, destination_data={destination_data}, kwargs={kwargs}")
                    return "Error: No valid destination data provided"
            
            return _store_insights, _astore_insights
        
        def _store_legacy_insights(destination_data: Dict[str, Any], db_mgr: EnhancedDatabaseManager) -> str:
            """
            Handle legacy format for backward compatibility
            """
            try:
                # Extract basic info
                name = destination_data.get("name", destination_data.get("destination_name", "Unknown"))
                insights = destination_data.get("insights", destination_data.get("themes", []))
                
                logger.info(f"_store_legacy_insights called with name={name}, insights type={type(insights)}, count={len(insights) if isinstance(insights, list) else 1}")
                
                # Create a minimal Destination object
                from datetime import datetime
                from ..core.enhanced_data_models import Destination, Theme
                
                destination = Destination(
                    names=[name],  # Fixed: use 'names' as a list
                    id=f"legacy_{name.lower().replace(' ', '_').replace(',', '')}",
                    country_code="US",  # Default, should be extracted
                    timezone="UTC",  # Default, should be extracted
                    admin_levels={"city": name}
                )
                
                # Process insights - handle both dict and object formats
                themes_added = 0
                if isinstance(insights, list):
                    for insight_item in insights:
                        if hasattr(insight_item, 'validated_themes'):
                            # It's a ThemeInsightOutput object
                            logger.info(f"Processing ThemeInsightOutput with {len(insight_item.validated_themes)} validated themes and {len(insight_item.discovered_themes)} discovered themes")
                            
                            # Process validated themes
                            for theme_insight in insight_item.validated_themes:
                                # Import required models
                                from ..core.enhanced_data_models import Evidence
                                from ..core.evidence_hierarchy import SourceCategory, EvidenceType
                                from ..core.confidence_scoring import ConfidenceBreakdown, ConfidenceLevel
                                
                                # Extract evidence data from the DestinationInsight
                                evidence_list = []
                                if hasattr(theme_insight, 'evidence') and theme_insight.evidence:
                                    for idx, evidence_text in enumerate(theme_insight.evidence):
                                        # Create Evidence objects from the evidence strings
                                        source_url = theme_insight.source_urls[idx] if hasattr(theme_insight, 'source_urls') and idx < len(theme_insight.source_urls) else ""
                                        evidence = Evidence(
                                            id=f"evidence_{idx}_{datetime.now().timestamp()}",
                                            source_url=source_url,
                                            source_category=SourceCategory.BLOG,  # Default to BLOG instead of NEWS
                                            evidence_type=EvidenceType.PRIMARY,
                                            authority_weight=0.7,  # Default authority
                                            text_snippet=evidence_text,
                                            timestamp=datetime.now(),
                                            confidence=0.7,
                                            cultural_context={}
                                        )
                                        evidence_list.append(evidence)
                                
                                # Create confidence breakdown from the single confidence score
                                confidence_breakdown = None
                                if hasattr(theme_insight, 'confidence_score') and theme_insight.confidence_score is not None:
                                    # Create a simplified confidence breakdown
                                    conf_score = theme_insight.confidence_score
                                    confidence_breakdown = ConfidenceBreakdown(
                                        overall_confidence=conf_score,
                                        confidence_level=ConfidenceLevel.VERY_HIGH if conf_score > 0.85 else 
                                                        ConfidenceLevel.HIGH if conf_score > 0.7 else
                                                        ConfidenceLevel.MODERATE if conf_score > 0.5 else
                                                        ConfidenceLevel.LOW if conf_score > 0.3 else
                                                        ConfidenceLevel.INSUFFICIENT,
                                        evidence_count=len(evidence_list),
                                        source_diversity=conf_score,
                                        authority_score=conf_score,
                                        recency_score=conf_score,
                                        consistency_score=conf_score,
                                        factors={}
                                    )
                                
                                theme = Theme(
                                    theme_id=f"{theme_insight.insight_name.lower().replace(' ', '_')}_{datetime.now().timestamp()}",
                                    macro_category=theme_insight.insight_type,
                                    micro_category=theme_insight.insight_name,  # Use insight_name as micro_category for now
                                    name=theme_insight.insight_name,
                                    description=theme_insight.description or "",
                                    fit_score=theme_insight.confidence_score or 0.5,
                                    evidence=evidence_list,  # Now properly populated
                                    confidence_breakdown=confidence_breakdown,  # Now properly populated
                                    tags=getattr(theme_insight, 'tags', [])
                                )
                                destination.add_theme(theme)
                                themes_added += 1
                            
                            # Process discovered themes
                            for theme_insight in insight_item.discovered_themes:
                                # Import required models (already imported above)
                                
                                # Extract evidence data from the DestinationInsight
                                evidence_list = []
                                if hasattr(theme_insight, 'evidence') and theme_insight.evidence:
                                    for idx, evidence_text in enumerate(theme_insight.evidence):
                                        # Create Evidence objects from the evidence strings
                                        source_url = theme_insight.source_urls[idx] if hasattr(theme_insight, 'source_urls') and idx < len(theme_insight.source_urls) else ""
                                        evidence = Evidence(
                                            id=f"evidence_{idx}_{datetime.now().timestamp()}",
                                            source_url=source_url,
                                            source_category=SourceCategory.BLOG,  # Default to BLOG instead of NEWS
                                            evidence_type=EvidenceType.PRIMARY,
                                            authority_weight=0.7,  # Default authority
                                            text_snippet=evidence_text,
                                            timestamp=datetime.now(),
                                            confidence=0.7,
                                            cultural_context={}
                                        )
                                        evidence_list.append(evidence)
                                
                                # Create confidence breakdown from the single confidence score
                                confidence_breakdown = None
                                if hasattr(theme_insight, 'confidence_score') and theme_insight.confidence_score is not None:
                                    # Create a simplified confidence breakdown
                                    conf_score = theme_insight.confidence_score
                                    confidence_breakdown = ConfidenceBreakdown(
                                        overall_confidence=conf_score,
                                        confidence_level=ConfidenceLevel.VERY_HIGH if conf_score > 0.85 else 
                                                        ConfidenceLevel.HIGH if conf_score > 0.7 else
                                                        ConfidenceLevel.MODERATE if conf_score > 0.5 else
                                                        ConfidenceLevel.LOW if conf_score > 0.3 else
                                                        ConfidenceLevel.INSUFFICIENT,
                                        evidence_count=len(evidence_list),
                                        source_diversity=conf_score,
                                        authority_score=conf_score,
                                        recency_score=conf_score,
                                        consistency_score=conf_score,
                                        factors={}
                                    )
                                
                                theme = Theme(
                                    theme_id=f"{theme_insight.insight_name.lower().replace(' ', '_')}_{datetime.now().timestamp()}",
                                    macro_category=theme_insight.insight_type,
                                    micro_category=theme_insight.insight_name,  # Use insight_name as micro_category for now
                                    name=theme_insight.insight_name,
                                    description=theme_insight.description or "",
                                    fit_score=theme_insight.confidence_score or 0.5,
                                    evidence=evidence_list,  # Now properly populated
                                    confidence_breakdown=confidence_breakdown,  # Now properly populated
                                    tags=getattr(theme_insight, 'tags', [])
                                )
                                destination.add_theme(theme)
                                themes_added += 1
                        elif hasattr(insight_item, 'themes'):
                            # It's the result from enhanced theme analysis with 'themes' attribute
                            logger.info(f"Processing enhanced theme analysis result with {len(insight_item.themes)} themes")
                            
                            for theme_dict in insight_item.themes:
                                # Import required models
                                from ..core.enhanced_data_models import Evidence
                                from ..core.evidence_hierarchy import SourceCategory, EvidenceType
                                from ..core.confidence_scoring import ConfidenceBreakdown, ConfidenceLevel
                                
                                # Extract evidence data
                                evidence_list = []
                                evidence_summary = theme_dict.get('evidence_summary', [])
                                
                                # Convert evidence summary to Evidence objects
                                for idx, ev_data in enumerate(evidence_summary):
                                    evidence = Evidence(
                                        id=ev_data.get('id', f"ev_{idx}_{datetime.now().timestamp()}"),
                                        source_url=ev_data.get('source_url', ''),
                                        source_category=SourceCategory[ev_data.get('source_category', 'GENERAL_MEDIA')],
                                        evidence_type=EvidenceType.PRIMARY,  # Default
                                        authority_weight=ev_data.get('authority_weight', 0.5),
                                        text_snippet=ev_data.get('text_snippet', ''),
                                        timestamp=datetime.now(),
                                        confidence=ev_data.get('authority_weight', 0.5),
                                        cultural_context=ev_data.get('cultural_context'),
                                        sentiment=ev_data.get('sentiment'),
                                        relationships=ev_data.get('relationships', {}),
                                        agent_id=ev_data.get('agent_id'),
                                        published_date=datetime.fromisoformat(ev_data.get('published_date').replace('Z', '+00:00')) if ev_data.get('published_date') else None
                                    )
                                    evidence_list.append(evidence)
                                
                                # Create confidence breakdown from theme data
                                confidence_breakdown_data = theme_dict.get('confidence_breakdown', {})
                                if confidence_breakdown_data:
                                    confidence_breakdown = ConfidenceBreakdown(
                                        overall_confidence=confidence_breakdown_data.get('overall_confidence', 0.5),
                                        confidence_level=ConfidenceLevel[confidence_breakdown_data.get('confidence_level', 'MODERATE')],
                                        evidence_count=len(evidence_list),
                                        source_diversity=confidence_breakdown_data.get('source_diversity', 0.5),
                                        authority_score=confidence_breakdown_data.get('authority_score', 0.5),
                                        recency_score=confidence_breakdown_data.get('recency_score', 0.5),
                                        consistency_score=confidence_breakdown_data.get('consistency_score', 0.5),
                                        factors=confidence_breakdown_data.get('factors', {})
                                    )
                                else:
                                    # Fallback confidence breakdown
                                    confidence_breakdown = ConfidenceBreakdown(
                                        overall_confidence=theme_dict.get('confidence_score', 0.5),
                                        confidence_level=ConfidenceLevel.MODERATE,
                                        evidence_count=len(evidence_list),
                                        source_diversity=0.5,
                                        authority_score=0.5,
                                        recency_score=0.5,
                                        consistency_score=0.5,
                                        factors={}
                                    )
                                
                                # Create Theme object with all enhanced fields including analytical data
                                theme = Theme(
                                    theme_id=theme_dict.get('theme_id', f"theme_{datetime.now().timestamp()}"),
                                    macro_category=theme_dict.get('macro_category', 'Other'),
                                    micro_category=theme_dict.get('micro_category', theme_dict.get('name', 'Unknown')),
                                    name=theme_dict.get('name', 'Unknown Theme'),
                                    description=theme_dict.get('description', ''),
                                    fit_score=theme_dict.get('fit_score', 0.5),
                                    evidence=evidence_list,
                                    confidence_breakdown=confidence_breakdown,
                                    tags=theme_dict.get('tags', []),
                                    created_date=datetime.now(),
                                    last_validated=datetime.now(),
                                    metadata=theme_dict.get('metadata', {}),
                                    # Enhanced fields
                                    authentic_insights=[],  # These would be populated by specialized tools
                                    local_authorities=[],   # These would be populated by specialized tools
                                    seasonal_relevance=theme_dict.get('seasonal_relevance', {}),
                                    regional_uniqueness=theme_dict.get('regional_uniqueness', 0.0),
                                    insider_tips=theme_dict.get('insider_tips', []),
                                    # Analytical data fields - this is the key fix!
                                    factors=theme_dict.get('factors', {}),
                                    cultural_summary=theme_dict.get('cultural_summary', {}),
                                    sentiment_analysis=theme_dict.get('sentiment_analysis', {}),
                                    temporal_analysis=theme_dict.get('temporal_analysis', {})
                                )
                                
                                destination.add_theme(theme)
                                themes_added += 1
                        elif isinstance(insight_item, dict):
                            # Handle plain dict format (backward compatibility)
                            theme = Theme(
                                theme_id=f"{insight_item.get('name', 'unknown').lower().replace(' ', '_')}_{datetime.now().timestamp()}",
                                macro_category=insight_item.get('category', 'Other'),
                                micro_category=insight_item.get('subcategory', insight_item.get('name', 'Other')),
                                name=insight_item.get('name', 'Unknown'),
                                description=insight_item.get('description', ''),
                                fit_score=insight_item.get('confidence', 0.5),
                                tags=insight_item.get('tags', [])
                            )
                            destination.add_theme(theme)
                            themes_added += 1
                
                logger.info(f"Creating destination object for {name} with {themes_added} themes")
                
                # Store using enhanced database manager
                results = db_mgr.store_destination(destination)
                
                # Fixed: Handle dict return value properly
                if isinstance(results, dict) and results.get("database_stored"):
                    result_parts = [f"Enhanced storage: Success - stored {len(destination.themes)} themes"]
                    if results.get("json_files_created"):
                        result_parts.append(f"JSON files: {len(results['json_files_created'])} exported")
                    return " | ".join(result_parts)
                else:
                    error_msg = results.get("errors", ["Unknown error"]) if isinstance(results, dict) else str(results)
                    return f"Enhanced storage: Failed - {error_msg}"
                
            except Exception as e:
                logger.error(f"Error in _store_legacy_insights: {e}", exc_info=True)
                return f"Error storing legacy insights: {str(e)}"
        
        # Create the functions with captured db_manager
        func, coroutine = create_tool_func(db_manager)
        
        super().__init__(
            name="store_destination_insights",
            description=(
                "Store enhanced destination insights in the database with evidence hierarchy, "
                "confidence scoring, temporal data, and automatic JSON export. "
                "Input should be a destination object with all enhanced data."
            ),
            func=func,
            coroutine=coroutine
        ) 