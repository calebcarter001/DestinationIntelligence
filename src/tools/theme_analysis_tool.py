import asyncio
import logging
from typing import List, Dict, Any, Type, Optional

from langchain.tools import StructuredTool
from pydantic import BaseModel, Field

from src.schemas import PageContent, ChromaSearchResult, ThemeInsightOutput, DestinationInsight
from src.core.content_intelligence_logic import ContentIntelligenceLogic # Will need significant updates

logger = logging.getLogger(__name__)

class AnalyzeThemesWithEvidenceInput(BaseModel):
    destination_name: str = Field(description="Name of the destination being analyzed.")
    # Original broad content for discovery & context
    original_page_content_list: List[PageContent] = Field(description="List of original page content objects for broad analysis and discovery.")
    # Focused evidence from Chroma for seed themes
    seed_themes_with_evidence: Optional[Dict[str, List[ChromaSearchResult]]] = Field(None,
        description="Optional. A dictionary where keys are seed theme names and values are lists of relevant ChromaSearchResult chunks for that theme."
    )
    # Configuration for theme extraction, e.g., thresholds
    config: Dict[str, Any] = Field(description="Configuration settings for content intelligence processing.")

class AnalyzeThemesFromEvidenceTool(StructuredTool):
    name: str = "analyze_themes_from_evidence"
    description: str = (
        "Analyzes content to validate seed themes using focused ChromaDB evidence and discover new themes from broader content. "
        "Requires original page content and, optionally, pre-fetched Chroma search results for seed themes."
    )
    args_schema: Type[BaseModel] = AnalyzeThemesWithEvidenceInput
    content_intelligence_logic: ContentIntelligenceLogic # This logic will be updated

    async def _arun(
        self, 
        destination_name: str, 
        original_page_content_list: List[PageContent],
        config: Dict[str, Any], # Make config a direct argument of _arun
        seed_themes_with_evidence: Optional[Dict[str, List[ChromaSearchResult]]] = None
    ) -> ThemeInsightOutput:
        logger.info(f"[ThemeAnalysisTool] Starting theme analysis for {destination_name}.")
        logger.info(f"Received {len(original_page_content_list)} original content pages.")
        if seed_themes_with_evidence:
            logger.info(f"Received evidence for {len(seed_themes_with_evidence)} seed themes from Chroma.")
        
        # The ContentIntelligenceLogic will need to be updated to handle these inputs
        # For now, it might mostly use the original_page_content_list like before,
        # but the plan is to make it use seed_themes_with_evidence for validation.
        
        # Pass the already loaded config directly
        self.content_intelligence_logic.config = config
        
        # Adapt ContentIntelligenceLogic.process_content_for_themes
        # to use both original_page_content_list (for discovery) 
        # and seed_themes_with_evidence (for focused validation)
        
        results = await self.content_intelligence_logic.process_content_for_themes(
            destination_name=destination_name,
            text_content_list=original_page_content_list, # Main content for discovery
            seed_themes_evidence_map=seed_themes_with_evidence # ChromaDB evidence for validation
        )
        
        # Ensure results is ThemeInsightOutput
        if isinstance(results, ThemeInsightOutput):
            logger.info(f"[ThemeAnalysisTool] Analysis for {destination_name} complete. Validated: {len(results.validated_themes)}, Discovered: {len(results.discovered_themes)}")
            return results
        elif isinstance(results, dict): # Handle if it returns a dict that can be parsed
            try:
                output = ThemeInsightOutput(**results)
                logger.info(f"[ThemeAnalysisTool] Analysis for {destination_name} complete. Validated: {len(output.validated_themes)}, Discovered: {len(output.discovered_themes)}")
                return output
            except Exception as e:
                logger.error(f"[ThemeAnalysisTool] Failed to parse dict to ThemeInsightOutput for {destination_name}: {e}", exc_info=True)
                return ThemeInsightOutput(destination_name=destination_name, validated_themes=[], discovered_themes=[]) # Return empty on error
        else:
             logger.error(f"[ThemeAnalysisTool] Unexpected result type from logic for {destination_name}: {type(results)}. Returning empty.")
             return ThemeInsightOutput(destination_name=destination_name, validated_themes=[], discovered_themes=[])

    def _run(self, *args, **kwargs) -> ThemeInsightOutput:
        # Sync wrapper - consider implications
        return asyncio.run(self._arun(*args, **kwargs)) 