import asyncio
import logging
from typing import List, Dict, Any, Type, Optional
import platform

import chromadb # Import chromadb
from chromadb.utils import embedding_functions # For default embedding function
from pydantic import BaseModel, Field
from langchain.tools import StructuredTool

from src.schemas import ProcessedPageChunk, ChromaSearchResult, PageContent # Added PageContent for potential direct add

logger = logging.getLogger(__name__)

# --- ChromaDB Client Initialization ---
# This client could be initialized globally or passed around.
# For simplicity in tool usage, tools might initialize their own or expect one.
# Here, we'll assume tools get a path and collection name, then init client.

DEFAULT_CHROMA_COLLECTION_NAME = "destination_content_chunks"

class AddToChromaInput(BaseModel):
    processed_chunks: List[ProcessedPageChunk] = Field(description="List of processed page chunks to add/update in ChromaDB.")
    collection_name: str = Field(default=DEFAULT_CHROMA_COLLECTION_NAME, description="Name of the ChromaDB collection.")

class SemanticSearchChromaInput(BaseModel):
    query_texts: List[str] = Field(description="List of texts to search for similar documents in ChromaDB.")
    collection_name: str = Field(default=DEFAULT_CHROMA_COLLECTION_NAME, description="Name of the ChromaDB collection.")
    n_results: int = Field(default=5, description="Number of results to return per query text.")
    # where_filter: Optional[Dict[str, Any]] = Field(None, description="Optional metadata filter for the search.")

class ChromaDBManager:
    """A simple manager for ChromaDB client and collections, used by tools."""
    def __init__(self, db_path: str, collection_name: str = DEFAULT_CHROMA_COLLECTION_NAME):
        logger.info(f"[ChromaDBManager] Initializing ChromaDB client with path: {db_path}")
        try:
            self.client = chromadb.PersistentClient(path=db_path)
            
            # Check if we're on macOS and use appropriate embedding function
            if platform.system() == "Darwin":  # macOS
                logger.info("[ChromaDBManager] Detected macOS - using SentenceTransformer embedding function to avoid ONNX/CoreML issues")
                # Use SentenceTransformer instead of ONNX on macOS
                self.embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(
                    model_name="all-MiniLM-L6-v2"  # Same model but using SentenceTransformer backend
                )
            else:
                # Use default (ONNX) on other platforms
                logger.info("[ChromaDBManager] Using default ONNX embedding function")
                self.embedding_function = embedding_functions.DefaultEmbeddingFunction()
            
            logger.info(f"[ChromaDBManager] Getting or creating collection: {collection_name}")
            self.collection = self.client.get_or_create_collection(
                name=collection_name,
                embedding_function=self.embedding_function # Use the chosen embedding function
            )
            logger.info(f"[ChromaDBManager] ChromaDB client and collection '{collection_name}' ready.")
        except Exception as e:
            logger.error(f"[ChromaDBManager] Error initializing ChromaDB: {e}", exc_info=True)
            self.client = None
            self.collection = None
            # Raise or handle appropriately for agent to know tool is not functional
            raise

    def add_chunks(self, chunks: List[Any]) -> int:
        """Add multiple processed chunks to ChromaDB collection."""
        try:
            if not chunks:
                logger.warning("[ChromaDBManager] No chunks provided for addition.")
                return 0
            
            # Extract data from chunks using correct field names
            ids = [chunk.chunk_id for chunk in chunks]
            documents = [chunk.text_chunk for chunk in chunks]
            # Handle metadata properly - it could be a dict or a Pydantic object
            metadatas = []
            for chunk in chunks:
                if chunk.metadata:
                    if hasattr(chunk.metadata, 'model_dump'):
                        # Pydantic object
                        metadatas.append(chunk.metadata.model_dump())
                    elif isinstance(chunk.metadata, dict):
                        # Already a dict
                        metadatas.append(chunk.metadata)
                    else:
                        # Convert to dict if possible
                        metadatas.append(dict(chunk.metadata) if hasattr(chunk.metadata, '__dict__') else {})
                else:
                    metadatas.append({})
            
            # Add to collection
            self.collection.add(
                ids=ids,
                documents=documents,
                metadatas=metadatas
            )
            
            logger.info(f"[ChromaDBManager] Successfully added {len(chunks)} chunks to collection '{self.collection.name}'.")
            return len(chunks)
            
        except Exception as e:
            logger.error(f"[ChromaDBManager] Error adding chunks: {e}", exc_info=True)
            return 0

    def search(self, query_texts: List[str], n_results: int = 5) -> List[List[ChromaSearchResult]]:
        if not self.collection:
            logger.error("[ChromaDBManager] Collection not initialized. Cannot search.")
            return [[] for _ in query_texts] # Return list of empty lists for each query

        try:
            results = self.collection.query(
                query_texts=query_texts,
                n_results=n_results,
                include=['documents', 'metadatas', 'distances'] 
            )
            logger.info(f"[ChromaDBManager] Search returned {len(results.get('ids', []))} result sets.")
            
            # Transform results into List[List[ChromaSearchResult]]
            output_results: List[List[ChromaSearchResult]] = []
            if results and results.get('ids'):
                for i in range(len(results['ids'])) : # Iterate over each query's result set
                    query_specific_results = []
                    ids_list = results['ids'][i]
                    docs_list = results['documents'][i]
                    metadatas_list = results['metadatas'][i]
                    distances_list = results['distances'][i]

                    for j in range(len(ids_list)):
                        chunk = ProcessedPageChunk(
                            chunk_id=ids_list[j],
                            url=metadatas_list[j].get('url', 'Unknown URL'), # Get from metadata
                            title=metadatas_list[j].get('title', 'Unknown Title'),
                            text_chunk=docs_list[j],
                            chunk_order=metadatas_list[j].get('chunk_order', -1),
                            metadata=metadatas_list[j]
                        )
                        query_specific_results.append(
                            ChromaSearchResult(document_chunk=chunk, distance=distances_list[j], metadata=metadatas_list[j])
                        )
                    output_results.append(query_specific_results)
            return output_results
        except Exception as e:
            logger.error(f"[ChromaDBManager] Error searching Chroma: {e}", exc_info=True)
            return [[] for _ in query_texts]

# --- LangChain Tools for ChromaDB Interaction ---

class AddChunksToChromaDBTool(StructuredTool):
    name: str = "add_processed_chunks_to_chromadb"
    description: str = (
        "Adds processed text chunks to a ChromaDB collection. "
        "Embeddings will be generated automatically by ChromaDB if not provided. "
        "Input is a list of ProcessedPageChunk objects."
    )
    args_schema: Type[BaseModel] = AddToChromaInput
    # Tool needs access to ChromaDBManager instance or path/collection_name to create one
    chroma_manager: ChromaDBManager # This will be initialized by the main app

    async def _arun(self, processed_chunks: List[ProcessedPageChunk], collection_name: str = DEFAULT_CHROMA_COLLECTION_NAME) -> Dict[str, Any]:
        logger.info(f"[ChromaToolAdd] Adding {len(processed_chunks)} chunks to ChromaDB collection '{collection_name}'.")
        if self.chroma_manager.collection.name != collection_name:
            logger.warning(f"[ChromaToolAdd] Tool initialized with collection '{self.chroma_manager.collection.name}' but called with '{collection_name}'. Re-targeting is not directly supported this way; ensure manager matches.")
            # Or, re-initialize the manager. For now, just log.

        try:
            num_added = self.chroma_manager.add_chunks(chunks=processed_chunks)
            return {"status": "success", "num_added": num_added, "collection_name": collection_name}
        except Exception as e:
            logger.error(f"[ChromaToolAdd] Error in _arun: {e}", exc_info=True)
            return {"status": "error", "message": str(e), "num_added": 0, "collection_name": collection_name}

    def _run(self, processed_chunks: List[ProcessedPageChunk], collection_name: str = DEFAULT_CHROMA_COLLECTION_NAME) -> Dict[str, Any]:
        # Sync wrapper - consider implications for event loops
        return asyncio.run(self._arun(processed_chunks, collection_name))

class SemanticSearchChromaDBTool(StructuredTool):
    name: str = "semantic_search_chromadb"
    description: str = (
        "Performs a semantic search in ChromaDB for text chunks similar to the query texts. "
        "Returns a list of search results for each query."
    )
    args_schema: Type[BaseModel] = SemanticSearchChromaInput
    chroma_manager: ChromaDBManager # Initialized by the main app

    async def _arun(self, query_texts: List[str], collection_name: str = DEFAULT_CHROMA_COLLECTION_NAME, n_results: int = 5) -> List[List[ChromaSearchResult]]:
        logger.info(f"[ChromaToolSearch] Searching for {len(query_texts)} queries in collection '{collection_name}', n_results={n_results}.")
        if self.chroma_manager.collection.name != collection_name:
             logger.warning(f"[ChromaToolSearch] Tool initialized with collection '{self.chroma_manager.collection.name}' but called with '{collection_name}'.")

        try:
            search_results = self.chroma_manager.search(query_texts=query_texts, n_results=n_results)
            # The Pydantic models should handle serialization for the agent if types match
            return search_results
        except Exception as e:
            logger.error(f"[ChromaToolSearch] Error in _arun: {e}", exc_info=True)
            return [[] for _ in query_texts]

    def _run(self, query_texts: List[str], collection_name: str = DEFAULT_CHROMA_COLLECTION_NAME, n_results: int = 5) -> List[List[ChromaSearchResult]]:
        # Sync wrapper
        return asyncio.run(self._arun(query_texts, collection_name, n_results)) 