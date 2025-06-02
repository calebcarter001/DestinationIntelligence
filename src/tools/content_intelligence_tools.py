import asyncio
import logging
from typing import List, Type, Dict, Any

from langchain.tools import StructuredTool
from pydantic import BaseModel, Field

from src.core.content_intelligence_logic import ContentIntelligenceLogic
from src.schemas import AnalyzeThemesInput, ThemeInsightOutput, PageContent

class AnalyzeContentForThemesTool(StructuredTool):
    name: str = "analyze_content_for_themes"
    description: str = (
        "Analyzes a list of web page contents for a destination to validate seed themes and discover new themes. "
        "Input should be the destination name and a list of page content objects (each with url, title, content, content_length)."
    )
    args_schema: Type[BaseModel] = AnalyzeThemesInput
    content_intelligence_logic: ContentIntelligenceLogic
    config: Dict[str, Any] # Added for consistency, even if not directly used by this tool yet

    def _run(self, destination_name: str, text_content_list: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        logging.info(f"[Tool] Synchronously running analyze_content_for_themes for {destination_name}")
        # Simplified handling for sync call to async logic, see notes in DiscoverAndFetchContentTool
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                return asyncio.run(self._arun(destination_name, text_content_list)) 
            else:
                return loop.run_until_complete(self._arun(destination_name, text_content_list))
        except RuntimeError as e:
            logging.error(f"RuntimeError in AnalyzeContentForThemesTool._run: {e}", exc_info=True)
            raise

    async def _arun(self, destination_name: str, text_content_list: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        logging.info(f"[Tool] Asynchronously running analyze_content_for_themes for {destination_name}")
        
        validated_themes_insights = await self.content_intelligence_logic.validate_themes_with_real_content(
            destination_name, text_content_list
        )
        
        validated_theme_names = [vt.insight_name for vt in validated_themes_insights]
        
        discovered_themes_insights = await self.content_intelligence_logic.discover_new_themes_from_content(
            destination_name, text_content_list, validated_theme_names
        )
        
        all_insights = validated_themes_insights + discovered_themes_insights
        return [insight.__dict__ for insight in all_insights] 