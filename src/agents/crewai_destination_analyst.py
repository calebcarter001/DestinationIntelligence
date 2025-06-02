import os
import logging
from typing import Dict, Any, List
from datetime import datetime

from crewai import Agent, Task, Crew, Process
from langchain_google_genai import ChatGoogleGenerativeAI

from src.tools.web_discovery_tools import DiscoverAndFetchContentTool
from src.tools.vectorize_processing_tool import ProcessContentWithVectorizeTool
from src.tools.chroma_interaction_tools import AddChunksToChromaDBTool, SemanticSearchChromaDBTool
from src.tools.theme_analysis_tool import AnalyzeThemesFromEvidenceTool
from src.tools.database_tools import StoreDestinationInsightsTool
from src.tools.enhanced_content_analysis_tool import EnhancedContentAnalysisTool
from src.tools.enhanced_database_storage_tool import EnhancedDatabaseStorageTool

logger = logging.getLogger(__name__)

class CrewAIDestinationAnalyst:
    """
    CrewAI-based destination intelligence analyst with specialized agents for each workflow step.
    Uses direct tool execution instead of agent tools to avoid compatibility issues.
    """
    
    def __init__(self, gemini_api_key: str, gemini_model_name: str, tools: List[Any]):
        self.gemini_api_key = gemini_api_key
        self.gemini_model_name = gemini_model_name
        self.tools = tools
        
        # Create tools dictionary for easy access
        self.tools_dict = {}
        for tool in tools:
            self.tools_dict[tool.name] = tool
        
        # Initialize LLM
        self.llm = ChatGoogleGenerativeAI(
            model=gemini_model_name,
            google_api_key=gemini_api_key,
            temperature=0.1,
            max_output_tokens=4096
        )
        
        # Create specialized agents (without tools)
        self.research_agent = self._create_research_agent()
        self.processing_agent = self._create_processing_agent()
        self.analysis_agent = self._create_analysis_agent()
        self.storage_agent = self._create_storage_agent()
        
    def _create_research_agent(self) -> Agent:
        """Create agent specialized in web research and content discovery."""
        return Agent(
            role="Web Research Specialist",
            goal="Discover and fetch high-quality web content about travel destinations",
            backstory=(
                "You are an expert web researcher who specializes in finding comprehensive, "
                "accurate, and diverse content about travel destinations. You excel at identifying "
                "authoritative sources and extracting valuable information for analysis."
            ),
            llm=self.llm,
            verbose=True,
            allow_delegation=False,
            max_iter=3
        )
    
    def _create_processing_agent(self) -> Agent:
        """Create agent specialized in content processing and vectorization."""
        return Agent(
            role="Content Processing Specialist", 
            goal="Process and organize content for optimal analysis and storage",
            backstory=(
                "You are a content processing expert who transforms raw web content into "
                "structured, analyzable chunks. You understand how to optimize content for "
                "vector databases and semantic search operations."
            ),
            llm=self.llm,
            verbose=True,
            allow_delegation=False,
            max_iter=3
        )
    
    def _create_analysis_agent(self) -> Agent:
        """Create agent specialized in theme analysis and insights generation."""
        return Agent(
            role="Theme Analysis Expert",
            goal="Analyze content to discover meaningful themes and insights about destinations",
            backstory=(
                "You are a travel intelligence analyst who excels at identifying patterns, "
                "themes, and insights from diverse content sources. You can synthesize "
                "information to create comprehensive destination intelligence reports."
            ),
            llm=self.llm,
            verbose=True,
            allow_delegation=False,
            max_iter=3
        )
    
    def _create_storage_agent(self) -> Agent:
        """Create agent specialized in data storage and management."""
        return Agent(
            role="Data Storage Manager",
            goal="Efficiently store and organize destination intelligence data",
            backstory=(
                "You are a data management specialist who ensures that analyzed destination "
                "intelligence is properly stored, organized, and accessible for future use. "
                "You maintain data integrity and optimize storage structures."
            ),
            llm=self.llm,
            verbose=True,
            allow_delegation=False,
            max_iter=2
        )
    
    async def analyze_destination(self, destination_name: str, processing_settings: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute the complete destination analysis workflow using direct tool execution.
        This approach bypasses CrewAI's tool system to avoid LangChain compatibility issues.
        """
        logger.info(f"Starting CrewAI destination analysis for: {destination_name}")
        start_time = datetime.now()
        
        try:
            # Execute the 6-step workflow directly using our tools
            logger.info("Executing 6-step workflow with direct tool calls...")
            
            # Step 1: Discover Web Content
            logger.info(f"Step 1: Discovering web content for {destination_name}")
            web_discovery_tool = self.tools_dict["discover_and_fetch_web_content_for_destination"]
            page_content_list = await web_discovery_tool._arun(destination_name=destination_name)
            
            if not page_content_list:
                error_msg = f"Step 1 failed: No web content discovered for {destination_name}"
                logger.error(error_msg)
                return {
                    "destination": destination_name,
                    "status": "Failed at Step 1",
                    "error": "No web content discovered",
                    "execution_duration_seconds": (datetime.now() - start_time).total_seconds(),
                    "execution_method": "CrewAI Direct Execution"
                }
            
            logger.info(f"Step 1 completed: Discovered {len(page_content_list)} pages")
            
            # Step 2: Process Content for Chunking
            logger.info(f"Step 2: Processing content for chunking")
            vectorize_tool = self.tools_dict["process_content_with_vectorize"]
            processed_chunks = await vectorize_tool._arun(page_content_list=page_content_list)
            
            if not processed_chunks:
                error_msg = f"Step 2 failed: No processed chunks created"
                logger.error(error_msg)
                return {
                    "destination": destination_name,
                    "status": "Failed at Step 2",
                    "error": "No processed chunks created",
                    "execution_duration_seconds": (datetime.now() - start_time).total_seconds(),
                    "execution_method": "CrewAI Direct Execution"
                }
            
            logger.info(f"Step 2 completed: Created {len(processed_chunks)} processed chunks")
            
            # Step 3: Store Chunks in Vector Database
            logger.info(f"Step 3: Storing chunks in ChromaDB")
            chroma_storage_tool = self.tools_dict["add_processed_chunks_to_chromadb"]
            storage_result = await chroma_storage_tool._arun(processed_chunks=processed_chunks)
            
            if "Error" in storage_result:
                error_msg = f"Step 3 failed: ChromaDB storage error - {storage_result}"
                logger.error(error_msg)
                return {
                    "destination": destination_name,
                    "status": "Failed at Step 3",
                    "error": storage_result,
                    "execution_duration_seconds": (datetime.now() - start_time).total_seconds(),
                    "execution_method": "CrewAI Direct Execution"
                }
            
            logger.info(f"Step 3 completed: {storage_result}")
            
            # Step 4: Search for Seed Themes
            logger.info(f"Step 4: Searching for seed themes")
            semantic_search_tool = self.tools_dict["semantic_search_chromadb"]
            
            # Use the actual seed themes that match what content_intelligence_logic expects
            # These should match the themes in ContentIntelligenceLogic.seed_themes
            seed_theme_queries = ["culture", "history", "nature", "food", "adventure", 
                                  "art", "architecture", "romance", "family", "luxury",
                                  "nightlife", "museums", "shopping", "beaches", "mountains",
                                  "festivals", "traditional", "modern", "authentic", "dining"]
            
            search_results = await semantic_search_tool._arun(
                query_texts=seed_theme_queries,
                n_results=3
            )
            
            if not search_results or "Error" in str(search_results):
                error_msg = f"Step 4 failed: Semantic search failed - {search_results}"
                logger.error(error_msg)
                return {
                    "destination": destination_name,
                    "status": "Failed at Step 4",
                    "error": str(search_results),
                    "execution_duration_seconds": (datetime.now() - start_time).total_seconds(),
                    "execution_method": "CrewAI Direct Execution"
                }
            
            logger.info(f"Step 4 completed: Found search results for {len(seed_theme_queries)} theme queries")
            
            # Transform search results into seed_themes_with_evidence format
            # search_results is a List[List[ChromaSearchResult]] indexed by query position
            seed_themes_with_evidence = {}
            for i, theme_query in enumerate(seed_theme_queries):
                if i < len(search_results) and search_results[i]:
                    # search_results[i] is a List[ChromaSearchResult] for this query
                    seed_themes_with_evidence[theme_query] = search_results[i]
                else:
                    seed_themes_with_evidence[theme_query] = []
            
            logger.info(f"Mapped search results to themes: {list(seed_themes_with_evidence.keys())}")
            
            # Step 5: Enhanced Content Analysis (NEW!)
            logger.info(f"Step 5: Enhanced content analysis for {destination_name}")
            enhanced_analysis_tool = EnhancedContentAnalysisTool(llm=self.llm)
            enhanced_analysis = await enhanced_analysis_tool._arun(
                destination_name=destination_name,
                page_content_list=page_content_list,
                analysis_categories=["attractions", "hotels", "restaurants", "activities", "neighborhoods", "practical_info"]
            )
            
            logger.info(f"Step 5 completed: Enhanced analysis generated {len(enhanced_analysis.attractions)} attractions, "
                       f"{len(enhanced_analysis.hotels)} hotels, {len(enhanced_analysis.restaurants)} restaurants")
            
            # Step 6: Basic Theme Analysis (for compatibility)
            logger.info(f"Step 6: Basic theme analysis for {destination_name}")
            theme_analysis_tool = self.tools_dict["analyze_themes_from_evidence"]
            theme_insights = await theme_analysis_tool._arun(
                destination_name=destination_name,
                original_page_content_list=page_content_list,
                seed_themes_with_evidence=seed_themes_with_evidence,
                config=processing_settings
            )
            
            # Count themes
            total_validated = len(theme_insights.validated_themes) if hasattr(theme_insights, 'validated_themes') else 0
            total_discovered = len(theme_insights.discovered_themes) if hasattr(theme_insights, 'discovered_themes') else 0
            
            logger.info(f"Step 6 completed: Generated {total_validated} validated themes and {total_discovered} discovered themes")
            
            # Step 7: Store Analysis Results (Enhanced + Basic)
            logger.info(f"Step 7: Storing enhanced analysis results in database")
            
            # Store enhanced insights in new detailed tables
            db_manager_found = None
            for tool in self.tools:
                if hasattr(tool, 'db_manager') and tool.db_manager:
                    db_manager_found = tool.db_manager
                    break
                elif hasattr(tool, '_db_manager') and tool._db_manager:
                    db_manager_found = tool._db_manager
                    break
            
            if db_manager_found:
                enhanced_storage_tool = EnhancedDatabaseStorageTool(db_manager=db_manager_found)
                enhanced_storage_result = await enhanced_storage_tool._arun(
                    destination_name=destination_name,
                    enhanced_analysis=enhanced_analysis
                )
                logger.info(f"Enhanced storage result: {enhanced_storage_result}")
            else:
                logger.warning("No database manager found - skipping enhanced storage")
                enhanced_storage_result = "Skipped - no database manager found"
            
            # Also store basic themes (for compatibility)
            storage_tool = self.tools_dict["store_destination_insights"]
            storage_result = await storage_tool._arun(
                destination_name=destination_name,
                insights=[theme_insights]
            )
            
            if "Error" in storage_result:
                error_msg = f"Step 7 failed: Database storage failed - {storage_result}"
                logger.error(error_msg)
                return {
                    "destination": destination_name,
                    "status": "Failed at Step 7",
                    "error": storage_result,
                    "execution_duration_seconds": (datetime.now() - start_time).total_seconds(),
                    "execution_method": "CrewAI Direct Execution"
                }
            
            logger.info(f"Step 7 completed: Basic themes: {storage_result}")
            logger.info(f"Step 7 completed: Enhanced insights: {enhanced_storage_result}")
            
            # All steps completed successfully
            end_time = datetime.now()
            duration = (end_time - start_time).total_seconds()
            
            success_summary = (
                f"CrewAI enhanced analysis completed for {destination_name}. "
                f"Processed {len(page_content_list)} pages, created {len(processed_chunks)} chunks, "
                f"discovered {total_validated + total_discovered} themes, and extracted detailed insights: "
                f"{len(enhanced_analysis.attractions)} attractions, {len(enhanced_analysis.hotels)} hotels, "
                f"{len(enhanced_analysis.restaurants)} restaurants, {len(enhanced_analysis.activities)} activities. "
                f"All 7 steps completed successfully."
            )
            
            return {
                "destination": destination_name,
                "status": "Success",
                "agent_output": success_summary,
                "processed_timestamp": end_time.isoformat(),
                "execution_duration_seconds": duration,
                "execution_method": "CrewAI Enhanced Direct Execution",
                "pages_processed": len(page_content_list),
                "chunks_created": len(processed_chunks),
                "validated_themes": total_validated,
                "discovered_themes": total_discovered,
                "total_themes": total_validated + total_discovered,
                # Enhanced insights
                "attractions_found": len(enhanced_analysis.attractions),
                "hotels_found": len(enhanced_analysis.hotels),
                "restaurants_found": len(enhanced_analysis.restaurants),
                "activities_found": len(enhanced_analysis.activities),
                "neighborhoods_found": len(enhanced_analysis.neighborhoods),
                "practical_info_found": len(enhanced_analysis.practical_info),
                "destination_summary": enhanced_analysis.summary,
                "enhanced_insights": {
                    "attractions": [insight.model_dump() for insight in enhanced_analysis.attractions[:3]],
                    "hotels": [insight.model_dump() for insight in enhanced_analysis.hotels[:3]],
                    "restaurants": [insight.model_dump() for insight in enhanced_analysis.restaurants[:3]],
                    "activities": [insight.model_dump() for insight in enhanced_analysis.activities[:3]]
                }
            }
            
        except Exception as e:
            error_msg = f"CrewAI workflow failed for {destination_name}: {str(e)}"
            logger.error(error_msg, exc_info=True)
            
            return {
                "destination": destination_name,
                "status": "Failed - CrewAI Exception",
                "error": error_msg,
                "execution_duration_seconds": (datetime.now() - start_time).total_seconds(),
                "execution_method": "CrewAI Direct Execution"
            }

def create_crewai_destination_analyst(gemini_api_key: str, gemini_model_name: str, tools: List[Any]) -> CrewAIDestinationAnalyst:
    """Factory function to create a CrewAI destination analyst."""
    return CrewAIDestinationAnalyst(gemini_api_key, gemini_model_name, tools) 