from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any

class SearchQueryInput(BaseModel):
    destination_name: str = Field(description="The name of the destination to search for, e.g., 'Paris, France'.")
    # query_template_key: Optional[str] = Field(None, description="Specific query template to use, e.g., 'hidden_gems'. If None, multiple general queries may be run.")
    # We will let the agent decide on the query text, or the tool can use its internal templates for a destination.
    # For simplicity, the tool can generate its own queries based on destination for now.

class WebSearchResult(BaseModel):
    url: str = Field(description="URL of the search result.")
    title: str = Field(description="Title of the search result.")
    snippet: Optional[str] = Field(None, description="Snippet or description from the search result.")

class FetchPageInput(BaseModel):
    url: str = Field(description="The URL of the web page to fetch and parse.")

class PageContent(BaseModel):
    url: str = Field(description="The URL of the fetched page.")
    title: Optional[str] = Field(None, description="Title of the page, if available from initial search metadata.")
    content: str = Field(description="Extracted textual content from the page.")
    content_length: int = Field(description="Length of the extracted content.")

class AnalyzeThemesInput(BaseModel):
    destination_name: str = Field(description="Name of the destination being analyzed.")
    text_content_list: List[PageContent] = Field(description="A list of page content objects, each containing text from a web page about the destination.")
    # seed_themes: Optional[List[str]] = Field(None, description="Optional list of seed themes to validate. If None, internal seed themes will be used.")

class DestinationInsight(BaseModel):
    destination_name: str
    insight_type: str = Field(description="Type of insight, e.g., 'Validated Theme', 'Discovered Theme', 'Unique Characteristic'")
    insight_name: str = Field(description="Name of the theme or characteristic, e.g., 'Outdoor Activities', 'Historic Architecture'")
    description: Optional[str] = Field(None, description="Detailed description or explanation of the insight.")
    evidence: List[str] = Field(default_factory=list, description="List of text snippets or URLs supporting the insight.")
    confidence_score: Optional[float] = Field(None, description="Confidence score from 0.0 to 1.0 for the insight's validity.")
    sentiment_score: Optional[float] = Field(None, description="Sentiment score from -1.0 (negative) to 1.0 (positive) related to the insight.")
    sentiment_label: Optional[str] = Field(None, description="Label for sentiment (e.g., POSITIVE, NEGATIVE, NEUTRAL)")
    source_urls: List[str] = Field(default_factory=list, description="List of source URLs from which this insight was derived.")
    # discovery_method: Optional[str] = Field(None, description="How was this discovered? e.g. 'Seed Theme Validation', 'LLM Discovery', 'Content Analysis'")
    # created_at: Optional[datetime] = Field(default_factory=datetime.now)

class ThemeInsightOutput(BaseModel):
    """Output schema for theme analysis, containing lists of validated and discovered themes."""
    destination_name: str
    validated_themes: List[DestinationInsight] = Field(default_factory=list, description="List of themes that were validated based on seed themes.")
    discovered_themes: List[DestinationInsight] = Field(default_factory=list, description="List of new themes discovered from the content.")
    # raw_analysis_summary: Optional[str] = Field(None, description="A brief textual summary of the analysis process or overall findings.")

class StoreInsightsInput(BaseModel):
    destination_name: str = Field(description="Name of the destination.")
    insights: List[ThemeInsightOutput] = Field(description="A list of theme insights to store.")

class FullDestinationAnalysisInput(BaseModel):
    destination_name: str = Field(description="The full name of the destination to analyze, e.g., 'Paris, France'.")

# --- New Schemas for Vectorize and Chroma ---

class ProcessedPageChunk(BaseModel):
    """Represents a chunk of processed text from a web page, ready for embedding and ChromaDB storage."""
    chunk_id: str = Field(description="Unique identifier for this chunk (e.g., url_hash + chunk_index)")
    url: str = Field(description="Original URL of the page this chunk came from.")
    title: Optional[str] = Field(None, description="Title of the original page.")
    text_chunk: str = Field(description="The actual text content of this chunk.")
    chunk_order: int = Field(description="Order of this chunk within the original document.")
    metadata: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Additional metadata like original source or processing details.")

class ChromaSearchResult(BaseModel):
    """Represents a single search result from ChromaDB."""
    document_chunk: ProcessedPageChunk = Field(description="The retrieved document chunk.")
    distance: Optional[float] = Field(None, description="Semantic distance or similarity score (lower is often more similar).")
    metadata: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Metadata associated with the document in ChromaDB.")

# --- Input Schemas for Tools (ensure these are consistent with tool definitions) --- 