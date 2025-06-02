import asyncio
import logging
from typing import List, Type, Dict, Any, Optional
from tqdm.asyncio import tqdm as asyncio_tqdm

from langchain.tools import StructuredTool
from pydantic import BaseModel, Field

from src.core.web_discovery_logic import WebDiscoveryLogic # For fallback
from src.schemas import PageContent # For structuring final output
from .jina_reader_tool import JinaReaderTool # Import the new tool

logger = logging.getLogger(__name__)

class DiscoverAndFetchContentToolInput(BaseModel):
    destination_name: str = Field(description="The name of the destination, e.g., 'Paris, France'.")

class DiscoverAndFetchContentTool(StructuredTool):
    name: str = "discover_and_fetch_web_content_for_destination"
    description: str = (
        "USE THIS TOOL FIRST. Discovers relevant web pages for a given destination, "
        "fetches their content (trying Jina Reader first, then fallback using BeautifulSoup), and extracts text. "
        "Input should be the name of the destination (e.g., 'Paris, France')."
    )
    args_schema: Type[BaseModel] = DiscoverAndFetchContentToolInput
    
    brave_api_key: str # Still needed for Brave Search and BS4 fallback session
    config: Dict[str, Any]
    # jina_api_key: Optional[str] # Removed, JinaReaderTool is now keyless for r.jina.ai

    # Pydantic V2 handles field initialization if types are annotated

    async def _fetch_content_for_single_url(self, url: str, source_metadata: Dict[str, Any]) -> Optional[PageContent]:
        """Helper to fetch content for one URL, trying Jina then fallback."""
        page_content_str = None
        source_title = source_metadata.get("title", url)
        min_len = self.config.get("web_discovery", {}).get("min_content_length_chars", 200)
        jina_reader_endpoint = self.config.get("web_discovery", {}).get("jina_reader_endpoint_template", "https://r.jina.ai/{url}")

        # Try Jina Reader first
        try:
            logger.info(f"[Tool] Attempting Jina Reader for URL: {url}")
            # Instantiate JinaReaderTool without API key, using endpoint from config
            jina_tool = JinaReaderTool(jina_reader_endpoint_template=jina_reader_endpoint) 
            page_content_str = await jina_tool._arun(url=url)
            
            if page_content_str and not page_content_str.startswith("Error:") and len(page_content_str) >= min_len:
                logger.info(f"[Tool] Jina Reader successful for {url}, length {len(page_content_str)}.")
                return PageContent(url=url, title=source_title, content=page_content_str, content_length=len(page_content_str))
            elif page_content_str and page_content_str.startswith("Error:"):
                logger.warning(f"[Tool] Jina Reader reported an error for {url}: {page_content_str}")
            else:
                logger.info(f"[Tool] Jina Reader content for {url} too short or empty (length {len(page_content_str or '')}). Will try BeautifulSoup fallback.")
            page_content_str = None # Ensure fallback if Jina didn't meet criteria
        except Exception as e:
            logger.warning(f"[Tool] Jina Reader failed unexpectedly for {url}: {e}. Will try BeautifulSoup fallback.", exc_info=True)
            page_content_str = None
        
        # Fallback to BeautifulSoup via WebDiscoveryLogic
        if page_content_str is None:
            logger.info(f"[Tool] Using BeautifulSoup fallback for URL: {url}")
            wd_logic_for_fallback = WebDiscoveryLogic(api_key=self.brave_api_key, config=self.config)
            async with wd_logic_for_fallback as wdl:
                page_content_str = await wdl._fetch_page_content(url)
            
            if page_content_str and len(page_content_str) >= min_len:
                logger.info(f"[Tool] BeautifulSoup fallback successful for {url}, length {len(page_content_str)}.")
                return PageContent(url=url, title=source_title, content=page_content_str, content_length=len(page_content_str))
            else:
                logger.info(f"[Tool] BeautifulSoup fallback for {url} yielded content too short or empty (length {len(page_content_str or '')}).")
        return None

    async def _arun(self, destination_name: str) -> List[PageContent]:
        logger.info(f"[Tool] DiscoverAndFetch running for {destination_name}.")
        web_discovery_logic = WebDiscoveryLogic(api_key=self.brave_api_key, config=self.config)
        search_results_lists = []
        async with web_discovery_logic as wdl:
            # Enhanced query templates for richer content discovery
            query_templates = [
                f"what makes {destination_name} special unique attractions",
                f"hidden gems {destination_name} local recommendations", 
                f"{destination_name} culture traditions authentic experiences",
                f"{destination_name} vs similar destinations comparison",
                f"why visit {destination_name} travel guide highlights",
                # NEW: Travel-specific queries for richer content
                f"best hotels {destination_name} where to stay recommendations",
                f"top restaurants {destination_name} food scene dining guide",
                f"things to do {destination_name} activities attractions itinerary",
                f"{destination_name} neighborhoods areas districts guide",
                f"{destination_name} weather best time to visit travel tips",
                f"{destination_name} transportation airport getting around",
                f"{destination_name} shopping markets local crafts",
                f"{destination_name} nightlife entertainment venues",
                f"{destination_name} outdoor activities adventure sports",
                f"travel guide {destination_name} complete visitor information"
            ]
            for template in query_templates:
                query = template.format(destination=destination_name)
                search_results = await wdl._fetch_brave_search(query) 
                if search_results:
                    search_results_lists.extend(search_results)
        
        unique_search_results_by_url: Dict[str, Dict[str, Any]] = {}
        for res in search_results_lists:
            if res.get("url") and res["url"] not in unique_search_results_by_url: # Added get() for safety
                unique_search_results_by_url[res["url"]] = res
        
        top_n_unique_urls_to_fetch = self.config.get("web_discovery", {}).get("max_urls_to_fetch_content_for", 15)
        urls_to_fetch_metadata = list(unique_search_results_by_url.values())[:top_n_unique_urls_to_fetch]

        logger.info(f"[Tool] Will attempt to fetch content for up to {len(urls_to_fetch_metadata)} unique URLs for {destination_name}.")
        content_fetch_tasks = [self._fetch_content_for_single_url(search_meta["url"], search_meta) for search_meta in urls_to_fetch_metadata]
        
        fetched_page_data_list: List[Optional[PageContent]] = []
        if content_fetch_tasks:
            fetched_page_data_list = await asyncio_tqdm.gather(
                *content_fetch_tasks,
                desc=f"Fetching URL Contents for {destination_name} (Jina/BS4)",
                unit="url"
            )
        successful_fetches = [item for item in fetched_page_data_list if item is not None]
        sorted_sources = sorted(successful_fetches, key=lambda x: x.content_length, reverse=True)
        max_sources_to_return_to_agent = self.config.get("web_discovery", {}).get("max_sources_for_agent_processing", 5)
        final_sources_to_return = sorted_sources[:max_sources_to_return_to_agent]
        logger.info(f"[Tool] Successfully fetched content for {len(successful_fetches)} URLs. Returning top {len(final_sources_to_return)} to agent.")
        return final_sources_to_return

    def _run(self, destination_name: str) -> List[PageContent]:
        # Fallback sync execution (problematic if loop is running)
        logging.warning("[DiscoverAndFetchContentTool] _run called; trying to run async logic.")
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running() and not hasattr(loop, '_nest_patched'): # Check if nest_asyncio has patched this loop
                logger.error("Cannot run async tool from a running event loop without nest_asyncio. Returning error.")
                return []
            return asyncio.run(self._arun(destination_name))
        except RuntimeError as e:
            logger.error(f"RuntimeError in DiscoverAndFetchContentTool _run: {e}. This tool is async-first.")
            return []

# Example of how you might structure a more granular tool if needed later:
# class FetchPageContentTool(StructuredTool):
#     name = "fetch_page_content"
#     description = "Fetches and parses the textual content from a single given URL."
#     args_schema: Type[BaseModel] = FetchPageInput
#     brave_api_key: str # Needed for the session, even if not directly for Brave API

#     def _run(self, url: str) -> Dict[str, Any]: # Simplified output for now
#         # Synchronous wrapper for async logic (not ideal)
#         return asyncio.run(self._arun(url))

#     async def _arun(self, url: str) -> Dict[str, Any]:
#         web_discovery_logic = WebDiscoveryLogic(api_key=self.brave_api_key)
#         async with web_discovery_logic as wd_logic:
#             content = await wd_logic._fetch_page_content(url) # Accessing protected method, better to expose it
#         return {"url": url, "content": content, "content_length": len(content)} 