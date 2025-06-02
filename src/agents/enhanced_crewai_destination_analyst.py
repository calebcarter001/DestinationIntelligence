#!/usr/bin/env python3
"""
Enhanced CrewAI-inspired destination analyst with LLM flexibility
Supports both Gemini and OpenAI models via configurable LLM parameter
"""

import asyncio
import logging
from datetime import datetime
from typing import Dict, Any, List, Optional
from src.schemas import PageContent

logger = logging.getLogger(__name__)

class EnhancedCrewAIDestinationAnalyst:
    """Enhanced CrewAI-inspired destination analyst with configurable LLM support"""
    
    def __init__(self, llm, tools: List):
        """
        Initialize with LLM and tools
        
        Args:
            llm: Configured LLM instance (Gemini or OpenAI)
            tools: List of tools for the agent
        """
        self.llm = llm
        self.tools = tools
        self.logger = logging.getLogger(__name__)
        
        # Create tool mapping with correct names
        self.tools_dict = {}
        for tool in tools:
            tool_name = getattr(tool, 'name', str(tool))
            self.tools_dict[tool_name] = tool
            
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
        
        # Validate required tools
        required_tools = [
            "discover_and_fetch_content",
            "process_content_with_vectorize", 
            "add_chunks_to_chromadb",
            "semantic_search_chromadb",
            "enhanced_content_analysis",
            "analyze_themes_from_evidence",
            "store_enhanced_destination_insights"
        ]
        
        missing_tools = [tool_name for tool_name in required_tools if tool_name not in self.tools_dict]
        if missing_tools:
            self.logger.warning(f"Missing tools: {missing_tools}")
    
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
                    # Map common country names to codes
                    country_mapping = {
                        "France": "FR", "United States": "US", "USA": "US",
                        "United Kingdom": "GB", "UK": "GB", "Germany": "DE",
                        "Italy": "IT", "Spain": "ES", "Japan": "JP"
                    }
                    country_code = country_mapping.get(country_part, "US")
            
            theme_analysis_result = await theme_analysis_tool._arun(
                destination_name=destination_name,
                country_code=country_code,
                text_content_list=search_results,
                seed_themes_with_evidence=seed_themes_with_evidence,
                config=processing_settings
            )
            
            validated_themes = len(theme_analysis_result.validated_themes) if hasattr(theme_analysis_result, 'validated_themes') else 0
            discovered_themes = len(theme_analysis_result.discovered_themes) if hasattr(theme_analysis_result, 'discovered_themes') else 0
            
            self.logger.info(f"Step 6 completed: Generated {validated_themes} validated themes and {discovered_themes} discovered themes")
            
            # Step 7: Store enhanced results
            self.logger.info("Step 7: Storing enhanced analysis results in database")
            storage_tool_name = self.tool_mappings.get("store_enhanced_destination_insights")
            storage_tool = self.tools_dict.get(storage_tool_name)
            
            if storage_tool:
                # Prepare enhanced destination data
                destination_data = {
                    "destination_name": destination_name,
                    "country_code": country_code,
                    "themes": theme_analysis_result,
                    "enhanced_insights": enhanced_analysis_result,
                    "pages_processed": pages_processed,
                    "chunks_created": chunks_created,
                    "validated_themes": validated_themes,
                    "discovered_themes": discovered_themes
                }
                
                storage_result = await storage_tool._arun(
                    destination_data=destination_data,
                    config=processing_settings
                )
                
                self.logger.info(f"Step 7 completed: {storage_result}")
            else:
                self.logger.warning(f"No database manager found - skipping enhanced storage. Looking for: {storage_tool_name}, Available: {list(self.tools_dict.keys())}")
            
            # Calculate execution time
            end_time = datetime.now()
            duration_seconds = (end_time - start_time).total_seconds()
            
            # Build result
            result = {
                "status": "Success",
                "destination_name": destination_name,
                "execution_method": "Enhanced CrewAI Direct Execution",
                "pages_processed": pages_processed,
                "chunks_created": chunks_created,
                "total_themes": validated_themes + discovered_themes,
                "validated_themes": validated_themes,
                "discovered_themes": discovered_themes,
                "attractions_found": attractions_found,
                "hotels_found": hotels_found,
                "restaurants_found": restaurants_found,
                "execution_duration_seconds": duration_seconds,
                "enhanced_insights": enhanced_analysis_result,
                "theme_analysis": theme_analysis_result
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
                "discovered_themes": 0
            }

def create_enhanced_crewai_destination_analyst(llm, tools: List) -> EnhancedCrewAIDestinationAnalyst:
    """
    Factory function to create enhanced CrewAI destination analyst
    
    Args:
        llm: Configured LLM instance (Gemini or OpenAI)
        tools: List of tools for the agent
        
    Returns:
        EnhancedCrewAIDestinationAnalyst instance
    """
    return EnhancedCrewAIDestinationAnalyst(llm=llm, tools=tools) 