import asyncio
import logging
import aiohttp
from typing import Type, Dict, Any, Optional

from langchain.tools import StructuredTool
from pydantic import BaseModel, Field

# from src.schemas import FetchPageInput # Not strictly needed if using JinaReaderToolInput directly

logger = logging.getLogger(__name__)

class JinaReaderToolInput(BaseModel):
    url: str = Field(description="The URL of the web page to read using Jina Reader.")

class JinaReaderTool(StructuredTool):
    name: str = "jina_reader_fetch_content"
    description: str = (
        "Fetches the main content of a given URL as clean text (potentially Markdown) using the Jina Reader service (r.jina.ai). "
        "Use this for potentially cleaner content extraction than standard scraping. Returns the text content or an error string."
    )
    args_schema: Type[BaseModel] = JinaReaderToolInput
    # No API key field needed if using the public r.jina.ai endpoint without specific auth for this task
    # jina_api_key: Optional[str] = None 

    # Jina Reader public endpoint template
    jina_reader_endpoint_template: str = "https://r.jina.ai/{url}"

    async def _arun(self, url: str) -> str:
        logger.info(f"[JinaReaderTool] Fetching content for URL: {url} using endpoint: {self.jina_reader_endpoint_template.format(url=url)}")
        
        request_url = self.jina_reader_endpoint_template.format(url=url)
        # No specific Authorization header for the public r.jina.ai endpoint for basic reading
        headers = {
            "Accept": "text/markdown, text/plain, */*;q=0.8" # Prefer markdown or plain text
        }

        try:
            async with aiohttp.ClientSession() as session: # Create session inside or ensure it's passed if used heavily
                async with session.get(request_url, headers=headers, timeout=aiohttp.ClientTimeout(total=30)) as resp:
                    logger.debug(f"[JinaReaderTool] Response status for {url}: {resp.status}")
                    if resp.status == 200:
                        # Jina Reader (r.jina.ai/URL) typically returns raw text/markdown directly
                        text_content = await resp.text(encoding='utf-8', errors='ignore')
                        logger.info(f"[JinaReaderTool] Successfully fetched content for {url} (length: {len(text_content)}).")
                        if not text_content.strip():
                            logger.warning(f"[JinaReaderTool] Content for {url} is empty after fetching.")
                            return f"Error: Jina Reader returned empty content for {url}."
                        return text_content
                    else:
                        error_text = await resp.text()
                        logger.error(f"[JinaReaderTool] Error fetching {url}. Status: {resp.status}, Response: {error_text[:500]}")
                        return f"Error: Jina Reader failed for {url}. Status: {resp.status}. Details: {error_text[:200]}"
        except asyncio.TimeoutError:
            logger.warning(f"[JinaReaderTool] Timeout fetching {url}.")
            return f"Error: Timeout while Jina Reader was fetching {url}."
        except Exception as e:
            logger.error(f"[JinaReaderTool] Exception during request to {url}: {e}", exc_info=True)
            return f"Error: Exception while calling Jina Reader for {url}. Details: {str(e)[:100]}"

    def _run(self, url: str) -> str:
        logger.warning("[JinaReaderTool] _run called; ideally _arun should be used by an async agent.")
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                logger.error("[JinaReaderTool] _run cannot reliably execute async _arun from a running event loop without nest_asyncio.")
                # Attempting to run in a new thread as a workaround for nested event loops if this path is ever hit.
                # This is not ideal but better than crashing. Better to ensure agent calls _arun.
                import concurrent.futures
                with concurrent.futures.ThreadPoolExecutor() as executor:
                    future = executor.submit(asyncio.run, self._arun(url=url))
                    return future.result()
            else:
                return loop.run_until_complete(self._arun(url=url))
        except RuntimeError as e:
             logger.error(f"[JinaReaderTool] RuntimeError in _run: {e}. This tool is async-first.")
             return f"Error: RuntimeError in JinaReaderTool sync execution. {e}" 