import asyncio
import logging
import hashlib
from typing import List, Dict, Any, Type, Optional
from pydantic import BaseModel, Field
from langchain.tools import StructuredTool
from src.schemas import PageContent, ProcessedPageChunk

logger = logging.getLogger(__name__)

# Placeholder configuration, actual Vectorize MCP server details would be needed
VECTORIZE_API_ENDPOINT = "YOUR_VECTORIZE_API_ENDPOINT_HERE" # Load from config eventually
VECTORIZE_API_KEY = "YOUR_VECTORIZE_API_KEY_HERE" # Load from config eventually

class VectorizeToolInput(BaseModel):
    page_content_list: List[PageContent] = Field(description="A list of PageContent objects to process and chunk.")
    chunk_size: int = Field(default=1000, description="Target size for text chunks (e.g., in characters or tokens depending on Vectorize service).")
    overlap: int = Field(default=100, description="Overlap between chunks.")

class ProcessContentWithVectorizeTool(StructuredTool):
    name: str = "process_content_with_vectorize"
    description: str = (
        "Processes raw page content using a (simulated) Vectorize service. "
        "Chunks text, prepares it for embedding, and extracts metadata. "
        "Input is a list of PageContent objects. Output is a list of ProcessedPageChunk objects."
    )
    args_schema: Type[BaseModel] = VectorizeToolInput
    config: Dict[str, Any] # To load vectorize_api_endpoint, etc.

    # In a real scenario, this would make an API call to the Vectorize MCP server
    async def _call_vectorize_service(self, text: str, chunk_size: int, overlap: int) -> List[Dict[str, Any]]:
        # SIMULATION: Simple character-based chunking
        # A real Vectorize service would do more sophisticated chunking, cleaning, and possibly embedding
        logger.info(f"[VectorizeTool-SIM] Simulating call to Vectorize for text (first 100 chars): {text[:100]}...")
        simulated_chunks = []
        for i in range(0, len(text), chunk_size - overlap):
            chunk_text = text[i:i + chunk_size]
            if not chunk_text.strip():
                continue
            simulated_chunks.append({"text_chunk": chunk_text, "metadata": {"simulated_processing": True}})
        
        logger.info(f"[VectorizeTool-SIM] Simulated Vectorize returned {len(simulated_chunks)} chunks.")
        return simulated_chunks

    async def _arun(self, page_content_list: List[PageContent], chunk_size: int = 1000, overlap: int = 100) -> Dict[str, Any]:
        logger.info(f"[VectorizeTool] Received {len(page_content_list)} PageContent objects for processing.")
        all_processed_chunks: List[ProcessedPageChunk] = []
        
        # Load actual endpoint/key from config if they were populated
        # vectorize_endpoint = self.config.get("processing_settings", {}).get("web_discovery", {}).get("vectorize_api_endpoint", VECTORIZE_API_ENDPOINT)
        # vectorize_key = self.config.get("processing_settings", {}).get("web_discovery", {}).get("vectorize_api_key", VECTORIZE_API_KEY)

        for idx, page in enumerate(page_content_list):
            if not page.content or not page.content.strip():
                logger.warning(f"[VectorizeTool] Skipping page {page.url} due to empty content.")
                continue
            
            try:
                # In a real implementation, you'd make an async HTTP request here
                # For now, we use the simulation
                raw_chunks_data = await self._call_vectorize_service(page.content, chunk_size, overlap)
                
                for chunk_idx, chunk_data in enumerate(raw_chunks_data):
                    url_hash = hashlib.md5(page.url.encode()).hexdigest()
                    chunk_id = f"{url_hash}_{chunk_idx}"
                    
                    processed_chunk = ProcessedPageChunk(
                        chunk_id=chunk_id,
                        url=page.url,
                        title=page.title,
                        text_chunk=chunk_data["text_chunk"],
                        chunk_order=chunk_idx,
                        metadata={
                            "original_content_length": page.content_length,
                            "processing_method": "vectorize_simulation",
                            **(chunk_data.get("metadata", {}))
                        }
                    )
                    all_processed_chunks.append(processed_chunk)
                logger.info(f"[VectorizeTool] Processed {len(raw_chunks_data)} chunks from {page.url}")
            except Exception as e:
                logger.error(f"[VectorizeTool] Error processing page {page.url} with Vectorize (simulation): {e}", exc_info=True)
                # Optionally return partial results or an error structure

        logger.info(f"[VectorizeTool] Total processed chunks: {len(all_processed_chunks)} from {len(page_content_list)} pages.")
        
        # Return dictionary format expected by enhanced analyst
        return {
            "total_chunks": len(all_processed_chunks),
            "chunks": all_processed_chunks,
            "pages_processed": len(page_content_list),
            "processing_method": "vectorize_simulation"
        }

    def _run(self, page_content_list: List[PageContent], chunk_size: int = 1000, overlap: int = 100) -> Dict[str, Any]:
        # Basic sync wrapper for async, not ideal for production if loop is running
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                # Consider nest_asyncio or running in a separate thread if truly needed sync
                logger.warning("[VectorizeTool] _run called from a running event loop. Simulating with direct async call (may block).")
                # This is a simplified approach; proper handling of nested loops is complex.
                # For agent use, _arun is expected.
                future = asyncio.ensure_future(self._arun(page_content_list, chunk_size, overlap))
                return loop.run_until_complete(future) # This might not work correctly in all nested scenarios
            else:
                return loop.run_until_complete(self._arun(page_content_list, chunk_size, overlap))
        except RuntimeError as e:
            logger.error(f"[VectorizeTool] RuntimeError in _run: {e}. This tool is async-first.")
            return {} 