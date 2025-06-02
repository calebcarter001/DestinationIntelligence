"""
Enhanced database tools for storing destination insights with full enhanced data model support
"""

import logging
from typing import Dict, Any, Optional, List
from langchain.tools import Tool
import hashlib

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
                    
                    # Store in enhanced database with JSON export
                    results = db_mgr.store_destination(destination, analysis_metadata)
                    
                    # Build response message
                    if results["database_stored"]:
                        message_parts = [f"✅ Successfully stored enhanced insights for {destination.names[0] if destination.names else 'destination'}"]
                        
                        # Add theme summary
                        theme_count = len(destination.themes)
                        verified_count = sum(1 for t in destination.themes if t.get_confidence_level().value in ["verified", "strongly_supported"])
                        message_parts.append(f"📊 Stored {theme_count} themes ({verified_count} high confidence)")
                        
                        # Add dimension summary
                        populated_dims = sum(1 for d in destination.dimensions.values() if d.value is not None)
                        message_parts.append(f"📏 Populated {populated_dims} dimensions")
                        
                        # Add temporal data
                        if destination.temporal_slices:
                            message_parts.append(f"🕐 Stored {len(destination.temporal_slices)} temporal slices")
                        
                        # Add JSON export info
                        if results["json_files_created"]:
                            message_parts.append("\n📁 JSON files exported:")
                            for file_type, path in results["json_files_created"].items():
                                message_parts.append(f"  • {file_type}: {path}")
                        
                        # Add any errors
                        if results["errors"]:
                            message_parts.append("\n⚠️  Warnings:")
                            for error in results["errors"]:
                                message_parts.append(f"  • {error}")
                        
                        return "\n".join(message_parts)
                    else:
                        error_msg = "\n".join(results["errors"]) if results["errors"] else "Unknown error"
                        return f"❌ Failed to store destination insights: {error_msg}"
                        
                except Exception as e:
                    logger.error(f"Error storing enhanced destination insights: {e}")
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
                            for theme_insight in insight_item.validated_themes + insight_item.discovered_themes:
                                theme = Theme(
                                    theme_id=f"{theme_insight.insight_name.lower().replace(' ', '_')}_{datetime.now().timestamp()}",
                                    macro_category=theme_insight.insight_type,
                                    micro_category=theme_insight.insight_type,
                                    name=theme_insight.insight_name,
                                    description=theme_insight.description or "",
                                    fit_score=theme_insight.confidence_score or 0.5,
                                    tags=[]
                                )
                                destination.add_theme(theme)
                                themes_added += 1
                        elif isinstance(insight_item, dict):
                            # Handle dict format
                            theme = Theme(
                                theme_id=f"{insight_item.get('name', 'unknown').lower().replace(' ', '_')}_{datetime.now().timestamp()}",
                                macro_category=insight_item.get('category', 'Other'),
                                micro_category=insight_item.get('subcategory', 'Other'),
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