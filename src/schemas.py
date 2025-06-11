from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from enum import Enum

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
    # Priority-related fields
    priority_type: Optional[str] = Field(None, description="Type of priority content (e.g., 'safety', 'cost', 'health', 'accessibility', 'weather')")
    priority_weight: Optional[float] = Field(1.0, description="Weight of priority content for ranking (higher is more important)")
    priority_data: Optional[Dict[str, Any]] = Field(None, description="Extracted priority data (safety metrics, costs, health requirements, etc.)")

class EnhancedEvidence(BaseModel):
    """Enhanced evidence object with all analytical fields"""
    id: str = Field(description="Unique identifier for the evidence")
    source_url: str = Field(description="Source URL of the evidence")
    source_category: str = Field(description="Category of the source")
    authority_weight: float = Field(description="Authority weight of the source")
    text_snippet: str = Field(description="Text snippet of the evidence")
    cultural_context: Dict[str, Any] = Field(default_factory=dict, description="Cultural context analysis")
    sentiment: Optional[float] = Field(None, description="Sentiment score")
    relationships: List[Dict[str, Any]] = Field(default_factory=list, description="Relationships with other evidence")
    agent_id: Optional[str] = Field(None, description="ID of the agent that processed this evidence")
    published_date: Optional[str] = Field(None, description="Published date of the content")
    confidence: float = Field(description="Confidence score")
    timestamp: str = Field(description="Processing timestamp")

class PriorityMetrics(BaseModel):
    """Aggregated priority metrics for a destination"""
    safety_score: Optional[float] = Field(None, description="Overall safety score (0-10)")
    crime_index: Optional[float] = Field(None, description="Crime index value")
    tourist_police_available: Optional[bool] = Field(None, description="Whether tourist police are available")
    emergency_contacts: Optional[Dict[str, str]] = Field(None, description="Emergency contact numbers")
    travel_advisory_level: Optional[str] = Field(None, description="Official travel advisory level")
    
    budget_per_day_low: Optional[float] = Field(None, description="Budget traveler daily cost estimate")
    budget_per_day_mid: Optional[float] = Field(None, description="Mid-range traveler daily cost estimate")
    budget_per_day_high: Optional[float] = Field(None, description="Luxury traveler daily cost estimate")
    currency: Optional[str] = Field(None, description="Local currency code")
    meal_cost_budget: Optional[float] = Field(None, description="Budget meal cost")
    meal_cost_mid: Optional[float] = Field(None, description="Mid-range meal cost")
    meal_cost_luxury: Optional[float] = Field(None, description="Luxury meal cost")
    coffee_price: Optional[float] = Field(None, description="Average coffee price")
    beer_price: Optional[float] = Field(None, description="Average beer price")
    public_transport_ticket: Optional[float] = Field(None, description="Public transport ticket price")
    taxi_start: Optional[float] = Field(None, description="Taxi starting fare")
    hotel_budget: Optional[float] = Field(None, description="Budget hotel price per night")
    hotel_mid: Optional[float] = Field(None, description="Mid-range hotel price per night")
    hotel_luxury: Optional[float] = Field(None, description="Luxury hotel price per night")
    
    required_vaccinations: Optional[List[str]] = Field(None, description="List of required vaccinations")
    health_risks: Optional[List[str]] = Field(None, description="List of health risks")
    water_safety: Optional[str] = Field(None, description="Water safety status")
    medical_facility_quality: Optional[str] = Field(None, description="Quality of medical facilities")
    
    visa_required: Optional[bool] = Field(None, description="Whether visa is required")
    visa_on_arrival: Optional[bool] = Field(None, description="Whether visa on arrival is available")
    visa_cost: Optional[float] = Field(None, description="Cost of visa if required")
    english_proficiency: Optional[str] = Field(None, description="Level of English proficiency")
    primary_language: Optional[str] = Field(None, description="Primary language spoken")
    infrastructure_rating: Optional[float] = Field(None, description="Infrastructure quality rating (1-5)")
    
    avg_temp_summer: Optional[float] = Field(None, description="Average summer temperature")
    avg_temp_winter: Optional[float] = Field(None, description="Average winter temperature")
    avg_high_summer: Optional[float] = Field(None, description="Average high in summer")
    avg_high_winter: Optional[float] = Field(None, description="Average high in winter")
    avg_low_summer: Optional[float] = Field(None, description="Average low in summer")
    avg_low_winter: Optional[float] = Field(None, description="Average low in winter")
    rainfall_mm_annual: Optional[float] = Field(None, description="Annual rainfall in mm")
    best_visit_seasons: Optional[List[str]] = Field(None, description="Best seasons to visit")

class AnalyzeThemesInput(BaseModel):
    destination_name: str = Field(description="Name of the destination being analyzed.")
    text_content_list: List[PageContent] = Field(description="A list of page content objects, each containing text from a web page about the destination.")
    # seed_themes: Optional[List[str]] = Field(None, description="Optional list of seed themes to validate. If None, internal seed themes will be used.")

class DestinationInsight(BaseModel):
    destination_name: str
    insight_type: str = Field(description="Type of insight, e.g., 'Validated Theme', 'Discovered Theme', 'Unique Characteristic', 'Priority Concern'")
    insight_name: str = Field(description="Name of the theme or characteristic, e.g., 'Outdoor Activities', 'Historic Architecture', 'Safety Warning'")
    description: Optional[str] = Field(None, description="Detailed description or explanation of the insight.")
    evidence: List[EnhancedEvidence] = Field(default_factory=list, description="List of enhanced evidence objects supporting the insight.")
    confidence_score: Optional[float] = Field(None, description="Confidence score from 0.0 to 1.0 for the insight's validity.")
    sentiment_score: Optional[float] = Field(None, description="Sentiment score from -1.0 (negative) to 1.0 (positive) related to the insight.")
    sentiment_label: Optional[str] = Field(None, description="Label for sentiment (e.g., POSITIVE, NEGATIVE, NEUTRAL)")
    source_urls: List[str] = Field(default_factory=list, description="List of source URLs from which this insight was derived.")
    # Priority-related fields
    priority_category: Optional[str] = Field(None, description="Priority category this insight relates to (safety, cost, health, etc.)")
    priority_impact: Optional[str] = Field(None, description="Impact level: 'high', 'medium', 'low'")
    temporal_relevance: Optional[float] = Field(None, description="How recent/relevant this information is (0-1)")
    # Additional fields to maintain compatibility with Theme objects
    tags: List[str] = Field(default_factory=list, description="List of tags associated with this insight")
    # discovery_method: Optional[str] = Field(None, description="How was this discovered? e.g. 'Seed Theme Validation', 'LLM Discovery', 'Content Analysis'")
    # created_at: Optional[datetime] = Field(default_factory=datetime.now)
    # Enhanced analytical fields
    factors: Optional[Dict[str, Any]] = Field(None, description="Theme factors analysis")
    cultural_summary: Optional[Dict[str, Any]] = Field(None, description="Cultural analysis summary")
    sentiment_analysis: Optional[Dict[str, Any]] = Field(None, description="Detailed sentiment analysis")
    temporal_analysis: Optional[Dict[str, Any]] = Field(None, description="Temporal analysis results")

class ThemeInsightOutput(BaseModel):
    """Enhanced output schema for theme analysis with unified themes list and evidence registry."""
    destination_name: str
    themes: List[DestinationInsight] = Field(default_factory=list, description="List of all discovered themes with enhanced metadata.")
    evidence_registry: Dict[str, Any] = Field(default_factory=dict, description="Registry of unique evidence pieces to avoid duplication.")
    priority_insights: List[DestinationInsight] = Field(default_factory=list, description="List of priority-related insights (safety, cost, health concerns).")
    priority_metrics: Optional[PriorityMetrics] = Field(None, description="Aggregated priority metrics for the destination")
    temporal_slices: List[Dict[str, Any]] = Field(default_factory=list, description="Temporal analysis slices with seasonal information.")
    dimensions: Dict[str, Any] = Field(default_factory=dict, description="Multi-dimensional analysis scores.")
    evidence_summary: Dict[str, Any] = Field(default_factory=dict, description="Summary statistics about evidence collection.")
    quality_metrics: Dict[str, Any] = Field(default_factory=dict, description="Quality metrics for the analysis.")
    authentic_insights: List[Dict[str, Any]] = Field(default_factory=list, description="Authentic insights with local authority validation.")
    
    # Legacy fields for backward compatibility
    validated_themes: List[DestinationInsight] = Field(default_factory=list, description="[DEPRECATED] Use 'themes' instead. Legacy field for backward compatibility.")
    discovered_themes: List[DestinationInsight] = Field(default_factory=list, description="[DEPRECATED] Use 'themes' instead. Legacy field for backward compatibility.")

class StoreInsightsInput(BaseModel):
    destination_name: str = Field(description="Name of the destination.")
    insights: List[ThemeInsightOutput] = Field(description="A list of theme insights to store.")

class FullDestinationAnalysisInput(BaseModel):
    destination_name: str = Field(description="The full name of the destination to analyze, e.g., 'Paris, France'.")

# --- New Enums for Enhanced Data Models ---

class InsightType(Enum):
    SEASONAL = "seasonal"
    SPECIALTY = "specialty"
    INSIDER = "insider"
    CULTURAL = "cultural"
    PRACTICAL = "practical"

class AuthorityType(Enum):
    PRODUCER = "producer"  # maple farmers, distillers
    RESIDENT = "long_term_resident"
    PROFESSIONAL = "industry_professional"
    CULTURAL = "cultural_institution"
    SEASONAL_WORKER = "seasonal_worker"
    OFFICIAL = "official"  # government, tourism boards
    COMMUNITY = "community"  # local community groups

class LocationExclusivity(Enum):
    EXCLUSIVE = "exclusive"  # only here
    SIGNATURE = "signature"  # best known for
    REGIONAL = "regional"   # common in region
    COMMON = "common"       # found elsewhere

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