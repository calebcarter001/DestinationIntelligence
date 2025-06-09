import asyncio
import logging
from typing import List, Type, Dict, Any, Optional
from tqdm.asyncio import tqdm as asyncio_tqdm

from langchain.tools import StructuredTool
from pydantic import BaseModel, Field

from src.core.web_discovery_logic import WebDiscoveryLogic # For fallback
from src.schemas import PageContent # For structuring final output
from .jina_reader_tool import JinaReaderTool # Import the new tool
from .priority_data_extraction_tool import PriorityDataExtractor  # Import the priority extractor

logger = logging.getLogger(__name__)

class DiscoverAndFetchContentToolInput(BaseModel):
    destination_name: str = Field(description="The name of the destination, e.g., 'Paris, France'.")

class DiscoverAndFetchContentTool(StructuredTool):
    name: str = "discover_and_fetch_web_content_for_destination"
    description: str = (
        "USE THIS TOOL FIRST. Discovers relevant web pages for a given destination, "
        "fetches their content (trying Jina Reader first, then fallback using BeautifulSoup), "
        "extracts text, and includes priority-focused content for traveler concerns. "
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
        priority_type = source_metadata.get("priority_type", None)
        priority_weight = source_metadata.get("priority_weight", 1.0)

        logger.info(f"[Tool] Attempting to fetch content for URL: {url} (Title: {source_title}, MinLen: {min_len})")

        # Try Jina Reader first
        try:
            logger.info(f"[Tool] Attempting Jina Reader for URL: {url}")
            jina_tool = JinaReaderTool(jina_reader_endpoint_template=jina_reader_endpoint) 
            page_content_str = await jina_tool._arun(url=url)
            
            jina_content_len = len(page_content_str) if page_content_str else 0
            if page_content_str and not page_content_str.startswith("Error:") and jina_content_len >= min_len:
                logger.info(f"[Tool] Jina Reader successful for {url}, Length: {jina_content_len}.")
                page_content = PageContent(
                    url=url, 
                    title=source_title, 
                    content=page_content_str, 
                    content_length=jina_content_len
                )
                if priority_type:
                    page_content.priority_type = priority_type
                    page_content.priority_weight = priority_weight
                return page_content
            elif page_content_str and page_content_str.startswith("Error:"):
                logger.warning(f"[Tool] Jina Reader reported an error for {url}: {page_content_str}")
            elif page_content_str: # Content was fetched but was too short
                logger.info(f"[Tool] Jina Reader content for {url} too short (Length: {jina_content_len}, MinLen: {min_len}). Will try BeautifulSoup fallback.")
            else: # No content string returned
                logger.info(f"[Tool] Jina Reader returned no content for {url}. Will try BeautifulSoup fallback.")
            page_content_str = None # Ensure fallback if Jina didn't meet criteria
        except Exception as e:
            logger.warning(f"[Tool] Jina Reader failed unexpectedly for {url}: {e}. Will try BeautifulSoup fallback.", exc_info=True)
            page_content_str = None
        
        # Fallback to BeautifulSoup via WebDiscoveryLogic
        if page_content_str is None:
            logger.info(f"[Tool] Using BeautifulSoup fallback for URL: {url}")
            # Pass the full config to the fallback logic instance
            wd_logic_for_fallback = WebDiscoveryLogic(api_key=self.brave_api_key, config=self.config) 
            async with wd_logic_for_fallback as wdl:
                # Make sure _fetch_page_content in WebDiscoveryLogic has necessary context if it needs self.config
                page_content_str = await wdl._fetch_page_content(url, destination_name=source_metadata.get("destination_name", "")) # Pass destination_name
            
            bs4_content_len = len(page_content_str) if page_content_str else 0
            if page_content_str and bs4_content_len >= min_len:
                logger.info(f"[Tool] BeautifulSoup fallback successful for {url}, Length: {bs4_content_len}.")
                page_content = PageContent(
                    url=url, 
                    title=source_title, 
                    content=page_content_str, 
                    content_length=bs4_content_len
                )
                if priority_type:
                    page_content.priority_type = priority_type
                    page_content.priority_weight = priority_weight
                return page_content
            elif page_content_str: # Content was fetched by BS4 but was too short
                logger.info(f"[Tool] BeautifulSoup fallback for {url} yielded content too short (Length: {bs4_content_len}, MinLen: {min_len}). Discarding URL.")
            else: # No content string returned by BS4
                 logger.info(f"[Tool] BeautifulSoup fallback for {url} yielded no content. Discarding URL.")
        
        logger.warning(f"[Tool] Failed to fetch valid content for URL: {url} after all attempts.")
        return None

    async def _arun(self, destination_name: str) -> List[PageContent]:
        logger.info(f"[Tool] DiscoverAndFetch running for {destination_name}.")
        web_discovery_logic = WebDiscoveryLogic(api_key=self.brave_api_key, config=self.config)
        all_sources = []
        
        # Check if priority discovery is enabled
        enable_priority_discovery = self.config.get("priority_settings", {}).get("enable_priority_discovery", True)
        
        async with web_discovery_logic as wdl:
            # Get sources from web discovery (includes priority content if enabled)
            sources_with_metadata = await wdl.discover_real_content(destination_name) # RESTORED Original line
            
            all_sources.extend(sources_with_metadata)
        
        # Deduplicate by URL
        unique_sources_by_url: Dict[str, Dict[str, Any]] = {}
        for source in all_sources:
            if source.get("url") and source["url"] not in unique_sources_by_url:
                unique_sources_by_url[source["url"]] = source
        
        # Sort by priority weight first, then by content length
        sorted_sources = sorted(
            list(unique_sources_by_url.values()),
            key=lambda x: (x.get("priority_weight", 1.0), x.get("content_length", 0)),
            reverse=True
        )
        
        # Limit sources to fetch
        top_n_unique_urls_to_fetch = self.config.get("web_discovery", {}).get("max_urls_to_fetch_content_for", 15)
        if enable_priority_discovery:
            top_n_unique_urls_to_fetch = min(top_n_unique_urls_to_fetch * 2, 20)  # Increase limit when priority is enabled
        
        urls_to_fetch_metadata = sorted_sources[:top_n_unique_urls_to_fetch]

        logger.info(f"[Tool] Will attempt to fetch content for up to {len(urls_to_fetch_metadata)} unique URLs for {destination_name}.")
        
        # Log priority type distribution
        priority_types = {}
        for source in urls_to_fetch_metadata:
            ptype = source.get("priority_type", "general")
            priority_types[ptype] = priority_types.get(ptype, 0) + 1
        logger.info(f"[Tool] Priority type distribution: {priority_types}")
        
        content_fetch_tasks = [self._fetch_content_for_single_url(source["url"], source) for source in urls_to_fetch_metadata]
        
        fetched_page_data_list: List[Optional[PageContent]] = []
        if content_fetch_tasks:
            fetched_page_data_list = await asyncio_tqdm.gather(
                *content_fetch_tasks,
                desc=f"Fetching URL Contents for {destination_name} (Jina/BS4)",
                unit="url"
            )
        
        # Filter successful fetches and extract priority data
        successful_fetches = []
        
        # Create semantic priority extractor with LLM
        try:
            from src.core.llm_factory import LLMFactory
            from src.config_loader import load_app_config
            
            config = load_app_config()
            llm = LLMFactory.create_llm(
                provider=config.get("llm_settings", {}).get("provider", "gemini"),
                config=config
            )
            priority_extractor = PriorityDataExtractor(llm=llm)
            logger.info("Semantic priority extractor initialized with configured LLM")
        except Exception as e:
            logger.warning(f"Failed to initialize semantic extractor, using default: {e}")
            priority_extractor = PriorityDataExtractor()
        
        for page_content in fetched_page_data_list:
            if page_content is not None:
                # Extract priority data from content using semantic approach
                try:
                    priority_data = priority_extractor.extract_all_priority_data(
                        page_content.content,
                        page_content.url
                    )
                    # Properly assign priority data to PageContent field (not dynamic dict assignment)
                    page_content.priority_data = priority_data
                        
                    logger.info(f"Semantic extraction completed for {page_content.url}: "
                               f"confidence={priority_data.get('extraction_confidence', 0):.2f}, "
                               f"completeness={priority_data.get('data_completeness', 0):.2f}")
                               
                except Exception as e:
                    logger.warning(f"Failed to extract priority data from {page_content.url}: {e}")
                
                successful_fetches.append(page_content)
        
        # Sort by priority weight and content length
        sorted_sources = sorted(
            successful_fetches, 
            key=lambda x: (
                getattr(x, 'priority_weight', 1.0),
                x.content_length
            ), 
            reverse=True
        )
        
        # MODIFIED: Temporarily increase max_sources_for_agent_processing for more evidence
        default_max_sources = 5 # Was 5, changed to 15, reverting to 5
        max_sources_to_return_to_agent = self.config.get("web_discovery", {}).get("max_sources_for_agent_processing", default_max_sources)
        if enable_priority_discovery:
            # Ensure priority still gets a boost if defined in config, or use a higher default
            priority_max_sources = 20 # Was 10, or min(default_max_sources * 2, 20)
            max_sources_to_return_to_agent = self.config.get("web_discovery", {}).get(
                "max_sources_for_agent_processing_priority", # Assuming a potential separate config for priority
                min(max_sources_to_return_to_agent * 2, priority_max_sources) # Existing logic if no separate config
            )
            # If the above get() still resulted in the non-priority value due to no specific priority config, ensure it's at least our new higher default for priority
            if max_sources_to_return_to_agent <= default_max_sources:
                 max_sources_to_return_to_agent = priority_max_sources

        logger.info(f"[Tool] DEBUG: max_sources_to_return_to_agent set to: {max_sources_to_return_to_agent}")
        final_sources_to_return = sorted_sources[:max_sources_to_return_to_agent]
        
        # Log final distribution
        final_priority_types = {}
        for source in final_sources_to_return:
            ptype = getattr(source, 'priority_type', 'general')
            final_priority_types[ptype] = final_priority_types.get(ptype, 0) + 1
        logger.info(f"[Tool] Final priority type distribution: {final_priority_types}")
        
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