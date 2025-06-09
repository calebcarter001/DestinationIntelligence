#!/usr/bin/env python3
"""
Enhanced CrewAI-inspired destination analyst with LLM flexibility
Supports both Gemini and OpenAI models via configurable LLM parameter
"""

import asyncio
import logging
from datetime import datetime
from typing import Dict, Any, List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from src.core.enhanced_database_manager import EnhancedDatabaseManager

from src.schemas import PageContent

logger = logging.getLogger(__name__)

class EnhancedCrewAIDestinationAnalyst:
    """Enhanced CrewAI-inspired destination analyst with configurable LLM support"""
    
    def __init__(self, llm, tools: List, db_manager: 'EnhancedDatabaseManager'):
        """
        Initialize with LLM, tools, and a database manager instance
        
        Args:
            llm: Configured LLM instance (Gemini or OpenAI)
            tools: List of tools for the agent
            db_manager: Instance of EnhancedDatabaseManager
        """
        self.llm = llm
        self.tools = tools
        self.db_manager = db_manager
        self.logger = logging.getLogger(__name__)
        
        # Create tool mapping with correct names
        self.tools_dict = {}
        for tool in tools:
            tool_name = getattr(tool, 'name', str(tool))
            self.tools_dict[tool_name] = tool
            
        # ---- TEMPORARY DEBUG ----
        self.logger.info(f"DEBUG: Populated self.tools_dict keys: {list(self.tools_dict.keys())}")
        # ---- END TEMPORARY DEBUG ----
        
        # Create simplified name mappings for easier access
        self.tool_mappings = {
            "discover_and_fetch_content": "discover_and_fetch_web_content_for_destination",
            "process_content_with_vectorize": "process_content_with_vectorize",
            "add_chunks_to_chromadb": "add_processed_chunks_to_chromadb", 
            "semantic_search_chromadb": "semantic_search_chromadb",
            "enhanced_content_analysis": "enhanced_destination_analysis",
            "analyze_themes_from_evidence": "analyze_themes_from_evidence",
            "store_enhanced_destination_insights": "store_destination_insights"
        }
        
        self.logger.info(f"Enhanced CrewAI analyst initialized with {len(tools)} tools")
        self.logger.info(f"Available tools: {list(self.tools_dict.keys())}")
        self.logger.info(f"Tool mappings: {self.tool_mappings}")
        
        # Validate required tools are present in the provided tools list
        missing_tools = []
        for simplified_name, actual_name in self.tool_mappings.items():
            if actual_name not in self.tools_dict:
                missing_tools.append(f"'{simplified_name}' (expected: '{actual_name}')")
        
        if missing_tools:
            self.logger.warning(f"Could not find registered tools for the following mappings: {', '.join(missing_tools)}")
    
    async def analyze_destination(self, destination_name: str, processing_settings: Dict[str, Any]) -> Dict[str, Any]:
        """
        Enhanced destination analysis pipeline
        
        Args:
            destination_name: Name of destination to analyze
            processing_settings: Processing configuration
            
        Returns:
            Dictionary with analysis results
        """
        start_time = datetime.now()
        
        try:
            self.logger.info(f"🚀 Starting enhanced destination analysis for: {destination_name}")
            
            # Step 1: Discover and fetch content
            self.logger.info("Step 1: Discovering web content")
            discovery_tool_name = self.tool_mappings.get("discover_and_fetch_content")
            discovery_tool = self.tools_dict.get(discovery_tool_name)
            if not discovery_tool:
                raise Exception(f"Discovery tool not found. Looking for: {discovery_tool_name}, Available: {list(self.tools_dict.keys())}")
            
            search_results = await discovery_tool._arun(
                destination_name=destination_name
            )
            
            if not search_results or not isinstance(search_results, list):
                raise Exception("No valid content discovered")
            
            pages_processed = len(search_results)
            self.logger.info(f"Step 1 completed: Discovered {pages_processed} pages")
            
            # Step 2: Process content for chunking
            self.logger.info("Step 2: Processing content for chunking")
            vectorize_tool_name = self.tool_mappings.get("process_content_with_vectorize")
            vectorize_tool = self.tools_dict.get(vectorize_tool_name)
            if not vectorize_tool:
                raise Exception(f"Vectorize tool not found. Looking for: {vectorize_tool_name}, Available: {list(self.tools_dict.keys())}")
            
            chunked_content = await vectorize_tool._arun(
                page_content_list=search_results
            )
            
            if not chunked_content or "total_chunks" not in chunked_content:
                raise Exception("Content processing failed")
            
            chunks_created = chunked_content["total_chunks"]
            self.logger.info(f"Step 2 completed: Created {chunks_created} processed chunks")
            
            # Step 3: Store chunks in ChromaDB
            self.logger.info("Step 3: Storing chunks in ChromaDB")
            chroma_add_tool_name = self.tool_mappings.get("add_chunks_to_chromadb")
            chroma_add_tool = self.tools_dict.get(chroma_add_tool_name)
            if not chroma_add_tool:
                raise Exception(f"ChromaDB add tool not found. Looking for: {chroma_add_tool_name}, Available: {list(self.tools_dict.keys())}")
            
            # Extract the chunks list from the dictionary
            chunks_to_store = chunked_content.get("chunks", [])
            if not chunks_to_store:
                raise Exception("No chunks found in processed content")
            
            storage_result = await chroma_add_tool._arun(
                processed_chunks=chunks_to_store
            )
            
            self.logger.info(f"Step 3 completed: {storage_result}")
            
            # Step 4: Search for seed themes
            self.logger.info("Step 4: Searching for seed themes")
            chroma_search_tool_name = self.tool_mappings.get("semantic_search_chromadb")
            chroma_search_tool = self.tools_dict.get(chroma_search_tool_name)
            if not chroma_search_tool:
                raise Exception(f"ChromaDB search tool not found. Looking for: {chroma_search_tool_name}, Available: {list(self.tools_dict.keys())}")
            
            seed_queries = [
                "culture", "history", "nature", "food", "adventure", "art", "architecture",
                "romance", "family", "luxury", "nightlife", "museums", "shopping", "beaches",
                "mountains", "festivals", "traditional", "modern", "authentic", "dining"
            ]
            
            chroma_search_results = await chroma_search_tool._arun(
                query_texts=seed_queries,
                n_results=3
            )
            
            self.logger.info(f"Step 4 completed: Found search results for {len(seed_queries)} theme queries")
            
            # Map search results to themes with evidence
            seed_themes_with_evidence = {}
            if isinstance(chroma_search_results, list) and len(chroma_search_results) == len(seed_queries):
                # chroma_search_results is List[List[ChromaSearchResult]] indexed by query position
                for i, theme_query in enumerate(seed_queries):
                    if i < len(chroma_search_results) and chroma_search_results[i]:
                        seed_themes_with_evidence[theme_query] = chroma_search_results[i]
                    else:
                        seed_themes_with_evidence[theme_query] = []
                self.logger.info(f"Mapped search results to themes with evidence: {list(seed_themes_with_evidence.keys())}")
            else:
                self.logger.warning(f"Unexpected ChromaDB search results format: {type(chroma_search_results)}")
                # Initialize empty evidence for all themes
                for theme in seed_queries:
                    seed_themes_with_evidence[theme] = []
            
            # Step 5: Enhanced content analysis
            self.logger.info(f"Step 5: Enhanced content analysis for {destination_name}")
            enhanced_analysis_tool_name = self.tool_mappings.get("enhanced_content_analysis")
            enhanced_analysis_tool = self.tools_dict.get(enhanced_analysis_tool_name)
            if enhanced_analysis_tool:
                enhanced_analysis_result = await enhanced_analysis_tool._arun(
                    destination_name=destination_name,
                    page_content_list=search_results
                )
                
                attractions_found = len(enhanced_analysis_result.attractions) if hasattr(enhanced_analysis_result, 'attractions') else 0
                hotels_found = len(enhanced_analysis_result.hotels) if hasattr(enhanced_analysis_result, 'hotels') else 0
                restaurants_found = len(enhanced_analysis_result.restaurants) if hasattr(enhanced_analysis_result, 'restaurants') else 0
                
                self.logger.info(f"Step 5 completed: Enhanced analysis generated {attractions_found} attractions, {hotels_found} hotels, {restaurants_found} restaurants")
            else:
                self.logger.warning(f"Enhanced analysis tool not found - skipping. Looking for: {enhanced_analysis_tool_name}, Available: {list(self.tools_dict.keys())}")
                enhanced_analysis_result = {}
                attractions_found = hotels_found = restaurants_found = 0
            
            # Step 6: Enhanced theme analysis
            self.logger.info(f"Step 6: Enhanced theme analysis for {destination_name}")
            theme_analysis_tool_name = self.tool_mappings.get("analyze_themes_from_evidence")
            theme_analysis_tool = self.tools_dict.get(theme_analysis_tool_name)
            if not theme_analysis_tool:
                raise Exception(f"Theme analysis tool not found. Looking for: {theme_analysis_tool_name}, Available: {list(self.tools_dict.keys())}")
            
            # Extract country code from destination name (simplified)
            country_code = "US"  # Default
            if "," in destination_name:
                parts = destination_name.split(",")
                if len(parts) >= 2:
                    country_part = parts[-1].strip()
                    # Map common country names to codes - EXPANDED with Australia and others
                    country_mapping = {
                        "France": "FR", "United States": "US", "USA": "US",
                        "United Kingdom": "GB", "UK": "GB", "Germany": "DE",
                        "Italy": "IT", "Spain": "ES", "Japan": "JP",
                        "Australia": "AU", "Canada": "CA", "Brazil": "BR",
                        "Mexico": "MX", "Netherlands": "NL", "Switzerland": "CH",
                        "Sweden": "SE", "Norway": "NO", "Denmark": "DK",
                        "Austria": "AT", "Belgium": "BE", "Portugal": "PT",
                        "Greece": "GR", "Turkey": "TR", "Poland": "PL",
                        "Czech Republic": "CZ", "Hungary": "HU", "Ireland": "IE",
                        "New Zealand": "NZ", "South Korea": "KR", "China": "CN",
                        "India": "IN", "Thailand": "TH", "Singapore": "SG",
                        "Malaysia": "MY", "Indonesia": "ID", "Philippines": "PH",
                        "Vietnam": "VN", "South Africa": "ZA", "Egypt": "EG",
                        "Morocco": "MA", "Argentina": "AR", "Chile": "CL",
                        "Peru": "PE", "Colombia": "CO", "Ecuador": "EC",
                        "Russia": "RU", "Finland": "FI", "Iceland": "IS"
                    }
                    country_code = country_mapping.get(country_part, "US")
            
            theme_analysis_result = await theme_analysis_tool._arun(
                destination_name=destination_name,
                country_code=country_code,
                text_content_list=search_results,
                seed_themes_with_evidence=seed_themes_with_evidence,
                config=processing_settings
            )
            
            # DEBUG LOGGING: Track theme analysis result
            self.logger.info(f"🔍 DEBUG_TAR_HANDOFF: Theme analysis result type: {type(theme_analysis_result)}")
            if isinstance(theme_analysis_result, dict):
                themes_count = len(theme_analysis_result.get('themes', []))
                self.logger.info(f"🔍 DEBUG_TAR_HANDOFF: Raw themes from Enhanced Theme Analysis Tool: {themes_count}")
                if themes_count > 0:
                    themes = theme_analysis_result.get('themes', [])
                    first_theme = themes[0]
                    # Handle both Theme objects and dictionaries
                    if hasattr(first_theme, 'name'):  # Theme object
                        theme_name = first_theme.name
                    else:  # Dictionary
                        theme_name = first_theme.get('name', 'NO_NAME')
                    self.logger.info(f"🔍 DEBUG_TAR_SAMPLE: First theme type: {type(first_theme)}, name: {theme_name}")
            
            # Fix: theme_analysis_result is a dictionary, not an object with attributes
            # Extract count from the correct structure
            if isinstance(theme_analysis_result, dict):
                total_themes = len(theme_analysis_result.get('themes', []))
                priority_insights = len(theme_analysis_result.get('priority_insights', []))
                # For reporting purposes, treat all themes as discovered since they're generated from content
                validated_themes = 0  # Legacy concept - not used in new enhanced format
                discovered_themes = total_themes
                
                # Log the quality metrics if available
                if 'quality_metrics' in theme_analysis_result:
                    qm = theme_analysis_result['quality_metrics']
                    self.logger.info(f"Quality metrics: {qm.get('themes_discovered', 0)} themes discovered, "
                                   f"{qm.get('themes_validated', 0)} validated, "
                                   f"avg confidence: {qm.get('average_confidence', 0):.3f}")
            else:
                # Fallback for old format (backward compatibility)
                validated_themes = len(theme_analysis_result.validated_themes) if hasattr(theme_analysis_result, 'validated_themes') else 0
                discovered_themes = len(theme_analysis_result.discovered_themes) if hasattr(theme_analysis_result, 'discovered_themes') else 0
                priority_insights = len(theme_analysis_result.priority_insights) if hasattr(theme_analysis_result, 'priority_insights') else 0
                total_themes = validated_themes + discovered_themes
            
            self.logger.info(f"Step 6 completed: Generated {validated_themes} validated themes, {discovered_themes} discovered themes, and {priority_insights} priority insights")
            
            # Extract themes from theme_analysis_result
            themes_data = []
            if isinstance(theme_analysis_result, dict):
                # Ensure we are getting the list of Theme objects
                themes_data = theme_analysis_result.get('themes', [])
                self.logger.info(f"DEBUG_THEME_HANDOFF: Extracted {len(themes_data)} themes from analysis result")
                
                # Also check if themes are in the enhanced format with different structure
                if not themes_data and 'enhanced_themes' in theme_analysis_result:
                    themes_data = theme_analysis_result.get('enhanced_themes', [])
                    self.logger.info(f"DEBUG_THEME_HANDOFF: Found {len(themes_data)} enhanced themes instead")
            
            # Create the Destination object
            from src.core.enhanced_data_models import Destination, Theme
            
            # Handle different theme formats: Theme objects, dictionaries, or mixed
            if themes_data:
                themes_obj_list = []
                for i, theme_item in enumerate(themes_data):
                    try:
                        if hasattr(theme_item, 'name'):  # Already a Theme object
                            themes_obj_list.append(theme_item)
                            self.logger.debug(f"DEBUG_THEME_HANDOFF: Theme {i} '{theme_item.name}' is already an object")
                        elif isinstance(theme_item, dict):  # Dictionary format
                            # More robust re-hydration with all available fields
                            theme_obj = Theme(
                                theme_id=theme_item.get('theme_id', f"theme_{i}"),
                                name=theme_item.get('name', f"Theme {i}"),
                                macro_category=theme_item.get('macro_category', 'General'),
                                micro_category=theme_item.get('micro_category', 'General'),
                                description=theme_item.get('description', ''),
                                fit_score=theme_item.get('fit_score', 0.0),
                                # Include enhanced fields if available
                                confidence_breakdown=theme_item.get('confidence_breakdown'),
                                evidence=theme_item.get('evidence', []),
                                tags=theme_item.get('tags', []),
                                authentic_insights=theme_item.get('authentic_insights', []),
                                local_authorities=theme_item.get('local_authorities', []),
                                seasonal_relevance=theme_item.get('seasonal_relevance', {}),
                                factors=theme_item.get('factors', {}),
                                cultural_summary=theme_item.get('cultural_summary', {}),
                                sentiment_analysis=theme_item.get('sentiment_analysis', {}),
                                temporal_analysis=theme_item.get('temporal_analysis', {})
                            )
                            themes_obj_list.append(theme_obj)
                            self.logger.debug(f"DEBUG_THEME_HANDOFF: Theme {i} '{theme_obj.name}' converted from dict")
                        else:
                            self.logger.warning(f"DEBUG_THEME_HANDOFF: Skipping theme {i} - unknown format: {type(theme_item)}")
                    except Exception as e:
                        self.logger.error(f"DEBUG_THEME_HANDOFF: Error processing theme {i}: {e}")
                
                themes_data = themes_obj_list
                self.logger.info(f"DEBUG_THEME_HANDOFF: Final themes_data contains {len(themes_data)} Theme objects")
            else:
                self.logger.warning("DEBUG_THEME_HANDOFF: No themes found in theme_analysis_result")

            destination_object = Destination(
                id=f"dest_{destination_name.replace(' ', '_').replace(',', '').lower()}",
                names=[destination_name],
                country_code=country_code,
                admin_levels={"country": country_code}, # Simplified
                timezone="UTC", # Placeholder
                themes=themes_data
            )
            
            # Calculate execution time
            end_time = datetime.now()
            duration_seconds = (end_time - start_time).total_seconds()
            
            # Extract priority metrics for summary - Fix: use dict access instead of attributes
            priority_summary = {}
            if isinstance(theme_analysis_result, dict) and 'priority_metrics' in theme_analysis_result:
                pm = theme_analysis_result['priority_metrics']
                if pm:  # Only process if priority_metrics is not None/empty
                    priority_summary = {
                        "safety_score": pm.get('safety_score') if hasattr(pm, 'get') else getattr(pm, 'safety_score', None),
                        "crime_index": pm.get('crime_index') if hasattr(pm, 'get') else getattr(pm, 'crime_index', None),
                        "budget_per_day_low": pm.get('budget_per_day_low') if hasattr(pm, 'get') else getattr(pm, 'budget_per_day_low', None),
                        "budget_per_day_mid": pm.get('budget_per_day_mid') if hasattr(pm, 'get') else getattr(pm, 'budget_per_day_mid', None),
                        "visa_required": pm.get('visa_required') if hasattr(pm, 'get') else getattr(pm, 'visa_required', None),
                        "required_vaccinations": pm.get('required_vaccinations', []) if hasattr(pm, 'get') else getattr(pm, 'required_vaccinations', [])
                    }
            elif hasattr(theme_analysis_result, 'priority_metrics') and theme_analysis_result.priority_metrics:
                # Fallback for old format (backward compatibility)
                pm = theme_analysis_result.priority_metrics
                priority_summary = {
                    "safety_score": getattr(pm, 'safety_score', None),
                    "crime_index": getattr(pm, 'crime_index', None),
                    "budget_per_day_low": getattr(pm, 'budget_per_day_low', None),
                    "budget_per_day_mid": getattr(pm, 'budget_per_day_mid', None),
                    "visa_required": getattr(pm, 'visa_required', None),
                    "required_vaccinations": getattr(pm, 'required_vaccinations', [])
                }
            
            # Build result
            result = {
                "status": "Success",
                "destination_name": destination_name,
                "destination_object": destination_object,
                "execution_method": "Enhanced CrewAI Direct Execution",
                "pages_processed": pages_processed,
                "chunks_created": chunks_created,
                "total_themes": total_themes,
                "validated_themes": validated_themes,
                "discovered_themes": discovered_themes,
                "priority_insights": priority_insights,
                "attractions_found": attractions_found,
                "hotels_found": hotels_found,
                "restaurants_found": restaurants_found,
                "execution_duration_seconds": duration_seconds,
                "enhanced_insights": enhanced_analysis_result,
                "theme_analysis": theme_analysis_result,
                "priority_summary": priority_summary
            }
            
            self.logger.info(f"✅ Enhanced analysis completed successfully for {destination_name}")
            return result
            
        except Exception as e:
            self.logger.error(f"Enhanced CrewAI workflow failed for {destination_name}: {e}", exc_info=True)
            
            end_time = datetime.now()
            duration_seconds = (end_time - start_time).total_seconds()
            
            return {
                "status": "Failed - Enhanced CrewAI Exception",
                "destination_name": destination_name,
                "execution_method": "Enhanced CrewAI Direct Execution",
                "error": str(e),
                "execution_duration_seconds": duration_seconds,
                "pages_processed": 0,
                "chunks_created": 0,
                "total_themes": 0,
                "validated_themes": 0,
                "discovered_themes": 0,
                "priority_insights": 0
            }

def create_enhanced_crewai_destination_analyst(llm, tools: List, db_manager: 'EnhancedDatabaseManager') -> EnhancedCrewAIDestinationAnalyst:
    """
    Factory function to create enhanced CrewAI destination analyst
    
    Args:
        llm: Configured LLM instance (Gemini or OpenAI)
        tools: List of tools for the agent
        db_manager: Instance of EnhancedDatabaseManager
        
    Returns:
        EnhancedCrewAIDestinationAnalyst instance
    """
    return EnhancedCrewAIDestinationAnalyst(llm=llm, tools=tools, db_manager=db_manager) 