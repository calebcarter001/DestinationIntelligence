import logging
from typing import List, Type, Dict, Any

from langchain.tools import StructuredTool
from pydantic import BaseModel, Field

from src.core.database_manager import DatabaseManager
from src.schemas import StoreInsightsInput, ThemeInsightOutput, DestinationInsight as PydanticDestinationInsight
from src.data_models import DestinationInsight as DataclassDestinationInsight # To reconstruct for DatabaseManager

class StoreDestinationInsightsTool(StructuredTool):
    name: str = "store_destination_insights"
    description: str = "Stores a list of analyzed theme insights for a given destination into the database."
    args_schema: Type[BaseModel] = StoreInsightsInput
    db_manager: DatabaseManager # Field for dependency injection

    def _run(self, destination_name: str, insights: List[ThemeInsightOutput]) -> str:
        logging.info(f"[Tool] Storing {len(insights)} ThemeInsightOutput objects for {destination_name} into the database.")
        try:
            total_stored = 0
            for theme_insight_output in insights:
                # Extract all insights from both validated_themes and discovered_themes
                all_insights = theme_insight_output.validated_themes + theme_insight_output.discovered_themes
                
                for pydantic_insight in all_insights:
                    # Map fields from Pydantic DestinationInsight to dataclass DestinationInsight
                    insight_obj = DataclassDestinationInsight(
                        insight_type=pydantic_insight.insight_type,
                        insight_name=pydantic_insight.insight_name,
                        description=pydantic_insight.description,
                        confidence_score=pydantic_insight.confidence_score,
                        evidence_sources=pydantic_insight.source_urls,  # Map source_urls to evidence_sources
                        content_snippets=pydantic_insight.evidence,     # Map evidence to content_snippets
                        is_discovered_theme=(pydantic_insight in theme_insight_output.discovered_themes)
                        # created_date will be auto-populated by DestinationInsight's __post_init__
                    )
                    self.db_manager.store_real_insight(destination_name, insight_obj)
                    total_stored += 1
                    
            logging.info(f"[Tool] Successfully stored {total_stored} individual insights for {destination_name}.")
            return f"Successfully stored {total_stored} insights for {destination_name}."
        except Exception as e:
            logging.error(f"[Tool] Error storing insights for {destination_name}: {e}", exc_info=True)
            # Return a more informative error message that the agent can potentially use
            return f"Error storing insights for {destination_name}. Details: {str(e)[:100]}"

    async def _arun(self, destination_name: str, insights: List[ThemeInsightOutput]) -> str:
        # For tools that are not I/O bound in their own execution (just calling sync DB methods),
        # _arun can just call _run if DatabaseManager is fully synchronous.
        # If DatabaseManager had async methods, we'd use them here.
        logging.info(f"[Tool] Asynchronously called store_destination_insights for {destination_name}. Running synchronously.")
        return self._run(destination_name, insights) 