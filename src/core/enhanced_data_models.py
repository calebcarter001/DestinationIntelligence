from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any, TYPE_CHECKING
from datetime import datetime, date
from enum import Enum
import hashlib
import json

# Forward references for type hints to avoid circular imports
if TYPE_CHECKING:
    from .confidence_scoring import ConfidenceBreakdown, ConfidenceLevel

from .evidence_hierarchy import EvidenceType, SourceCategory
from src.schemas import InsightType, AuthorityType, LocationExclusivity

@dataclass
class Evidence:
    """Evidence supporting a theme or insight"""
    id: str
    source_url: str
    source_category: SourceCategory
    evidence_type: EvidenceType
    authority_weight: float
    text_snippet: str
    timestamp: datetime
    confidence: float
    sentiment: Optional[float] = None
    cultural_context: Optional[Dict[str, Any]] = None  # local_source, language, ownership
    relationships: List[Dict[str, str]] = field(default_factory=list)  # target_id, rel_type
    agent_id: Optional[str] = None
    published_date: Optional[datetime] = None
    factors: Optional[Dict[str, Any]] = None  # Additional analytical factors
    
    def __post_init__(self):
        if not self.id:
            # Generate ID from content hash
            content = f"{self.source_url}:{self.text_snippet[:100]}"
            self.id = hashlib.md5(content.encode()).hexdigest()[:12]

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage/serialization"""
        return {
            "id": self.id,
            "source_url": self.source_url,
            "source_category": self.source_category.value,
            "evidence_type": self.evidence_type.value,
            "authority_weight": self.authority_weight,
            "text_snippet": self.text_snippet,
            "timestamp": self.timestamp.isoformat(),
            "confidence": self.confidence,
            "sentiment": self.sentiment,
            "cultural_context": self.cultural_context,
            "relationships": self.relationships,
            "agent_id": self.agent_id,
            "published_date": self.published_date.isoformat() if self.published_date else None,
            "factors": self.factors
        }

@dataclass
class SeasonalWindow:
    """Time-sensitive availability"""
    start_month: int
    end_month: int
    peak_weeks: List[int]
    booking_lead_time: Optional[str]
    specific_dates: Optional[List[str]]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "start_month": self.start_month,
            "end_month": self.end_month,
            "peak_weeks": self.peak_weeks,
            "booking_lead_time": self.booking_lead_time,
            "specific_dates": self.specific_dates
        }

@dataclass
class LocalAuthority:
    """Enhanced authority for local sources"""
    authority_type: AuthorityType  # PRODUCER, RESIDENT, PROFESSIONAL, CULTURAL
    local_tenure: Optional[int]  # years in location
    expertise_domain: str
    community_validation: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "authority_type": self.authority_type.value,
            "local_tenure": self.local_tenure,
            "expertise_domain": self.expertise_domain,
            "community_validation": self.community_validation
        }

@dataclass
class AuthenticInsight:
    """Enhanced insight with multi-dimensional scoring"""
    insight_type: InsightType  # SEASONAL, SPECIALTY, INSIDER, CULTURAL
    authenticity_score: float
    uniqueness_score: float
    actionability_score: float
    temporal_relevance: float
    location_exclusivity: LocationExclusivity  # EXCLUSIVE, SIGNATURE, REGIONAL, COMMON
    seasonal_window: Optional[SeasonalWindow]
    local_validation_count: int

    def to_dict(self) -> Dict[str, Any]:
        return {
            "insight_type": self.insight_type.value,
            "authenticity_score": self.authenticity_score,
            "uniqueness_score": self.uniqueness_score,
            "actionability_score": self.actionability_score,
            "temporal_relevance": self.temporal_relevance,
            "location_exclusivity": self.location_exclusivity.value,
            "seasonal_window": self.seasonal_window.to_dict() if self.seasonal_window else None,
            "local_validation_count": self.local_validation_count
        }

@dataclass
class Theme:
    """Enhanced theme with taxonomy, evidence, and confidence"""
    theme_id: str
    macro_category: str  # e.g., "Nature & Outdoors"
    micro_category: str  # e.g., "Hiking Trails"
    name: str
    description: str
    fit_score: float  # 0.0-1.0 relevance to destination
    evidence: List[Evidence] = field(default_factory=list)
    confidence_breakdown: Optional['ConfidenceBreakdown'] = None  # Forward reference
    tags: List[str] = field(default_factory=list)
    created_date: datetime = field(default_factory=datetime.now)
    last_validated: Optional[datetime] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    # New fields for enhanced insights
    authentic_insights: List[AuthenticInsight] = field(default_factory=list)
    local_authorities: List[LocalAuthority] = field(default_factory=list)
    seasonal_relevance: Dict[str, float] = field(default_factory=dict)  # month -> relevance
    regional_uniqueness: float = 0.0
    insider_tips: List[str] = field(default_factory=list)
    
    # New fields for analytical data
    factors: Dict[str, Any] = field(default_factory=dict)  # Theme strength factors
    cultural_summary: Dict[str, Any] = field(default_factory=dict)  # Cultural analysis
    sentiment_analysis: Dict[str, Any] = field(default_factory=dict)  # Sentiment patterns
    temporal_analysis: Dict[str, Any] = field(default_factory=dict)  # Temporal patterns
    
    def add_evidence(self, evidence: Evidence):
        """Add evidence and update confidence"""
        self.evidence.append(evidence)
        # Confidence recalculation would happen here
        
    def get_confidence_level(self):
        """Get the confidence level from the confidence breakdown"""
        if self.confidence_breakdown:
            return self.confidence_breakdown.confidence_level
        else:
            # Import here to avoid circular import
            from .confidence_scoring import ConfidenceLevel
            return ConfidenceLevel.INSUFFICIENT
        
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage/serialization"""
        return {
            "theme_id": self.theme_id,
            "macro_category": self.macro_category,
            "micro_category": self.micro_category,
            "name": self.name,
            "description": self.description,
            "fit_score": self.fit_score,
            "evidence": [e.to_dict() for e in self.evidence],
            "confidence_breakdown": self.confidence_breakdown.to_dict() if self.confidence_breakdown else None,
            "tags": self.tags,
            "created_date": self.created_date.isoformat(),
            "last_validated": self.last_validated.isoformat() if self.last_validated else None,
            "metadata": self.metadata,
            "authentic_insights": [ai.to_dict() for ai in self.authentic_insights],
            "local_authorities": [la.to_dict() for la in self.local_authorities],
            "seasonal_relevance": self.seasonal_relevance,
            "regional_uniqueness": self.regional_uniqueness,
            "insider_tips": self.insider_tips,
            "factors": self.factors,
            "cultural_summary": self.cultural_summary,
            "sentiment_analysis": self.sentiment_analysis,
            "temporal_analysis": self.temporal_analysis
        }

@dataclass
class PointOfInterest:
    """Enhanced POI with theme associations and accessibility"""
    poi_id: str
    name: str
    description: str
    location: Dict[str, float]  # lat, lng
    address: Optional[str] = None
    poi_type: str = ""  # attraction, hotel, restaurant, etc.
    theme_tags: List[str] = field(default_factory=list)
    ada_accessible: Optional[bool] = None
    ada_features: List[str] = field(default_factory=list)
    media_urls: List[str] = field(default_factory=list)
    operating_hours: Optional[Dict[str, str]] = None
    price_range: Optional[str] = None
    rating: Optional[float] = None
    review_count: Optional[int] = None
    
@dataclass
class TemporalSlice:
    """Temporal validity for slowly changing dimension (SCD2)"""
    valid_from: datetime
    valid_to: Optional[datetime] = None  # None = current
    season: Optional[str] = None  # spring, summer, fall, winter
    seasonal_highlights: Dict[str, Any] = field(default_factory=dict)
    special_events: List[Dict[str, Any]] = field(default_factory=list)
    weather_patterns: Optional[Dict[str, Any]] = None
    visitor_patterns: Optional[Dict[str, Any]] = None
    
    def is_current(self) -> bool:
        """Check if this slice is currently valid"""
        return self.valid_to is None or datetime.now() < self.valid_to
        
@dataclass
class DimensionValue:
    """Single dimension measurement with metadata"""
    value: Any
    unit: Optional[str] = None
    sample_period: Optional[str] = None  # e.g., "2024-Q1"
    confidence: float = 0.0
    source_evidence_ids: List[str] = field(default_factory=list)
    last_updated: datetime = field(default_factory=datetime.now)

@dataclass
class Destination:
    """Enhanced destination model with all v5 features"""
    # Core identification
    id: str  # geohash8-ISO-CC format
    names: List[str]  # Primary name + alternates
    admin_levels: Dict[str, str]  # country, state, city, etc.
    timezone: str
    population: Optional[int] = None
    country_code: str = ""  # ISO 2-letter code
    
    # Geography
    core_geo: Dict[str, Any] = field(default_factory=dict)  # bbox, elevation, Köppen zone
    
    # Themes with full evidence
    themes: List[Theme] = field(default_factory=list)
    
    # New fields for destination-level aggregation of insights and authorities
    authentic_insights: List[AuthenticInsight] = field(default_factory=list)
    local_authorities: List[LocalAuthority] = field(default_factory=list)
    
    # 60-dimension attribute matrix
    dimensions: Dict[str, DimensionValue] = field(default_factory=dict)
    
    # Points of interest
    pois: List[PointOfInterest] = field(default_factory=list)
    
    # Temporal slices for SCD2
    temporal_slices: List[TemporalSlice] = field(default_factory=list)
    
    # Lineage tracking
    lineage: Dict[str, Any] = field(default_factory=dict)  # agent_id, code_hash, source_hash
    
    # Metadata
    meta: Dict[str, Any] = field(default_factory=dict)
    last_updated: datetime = field(default_factory=datetime.now)
    destination_revision: int = 1
    
    def __post_init__(self):
        if not self.id and self.names:
            # Generate ID from primary name and country
            self.id = self._generate_id(self.names[0], self.country_code)
            
        # Initialize standard dimensions if not present
        self._initialize_dimensions()
    
    def _generate_id(self, primary_name: str, country_code: str) -> str:
        """Generate geohash-style ID"""
        # Simplified version - real implementation would use actual geohash
        name_hash = hashlib.md5(primary_name.lower().encode()).hexdigest()[:8]
        return f"{name_hash}-ISO-{country_code.upper()}"
    
    def _initialize_dimensions(self):
        """Initialize the 60-dimension matrix with defaults"""
        standard_dimensions = [
            # Original dimensions
            "walkability_score", "public_transport_score", "safety_index",
            "cost_of_living_index", "english_proficiency", "wifi_availability",
            "healthcare_quality", "air_quality_index", "noise_level",
            "green_space_percentage", "cultural_diversity_index",
            
            # Vacation-specific additions
            "beach_cleanliness_index", "instagram_worthiness", 
            "scam_hassle_prevalence", "kid_attraction_density",
            "card_acceptance_percentage", "hidden_fee_score",
            "tourist_season_crowding", "local_friendliness",
            "adventure_activity_variety", "nightlife_vibrancy",
            "culinary_diversity", "accommodation_variety",
            "weather_consistency", "natural_beauty_score",
            "historical_significance", "shopping_variety",
            "festival_frequency", "romantic_atmosphere",
            "backpacker_friendliness", "luxury_amenities",
            
            # Environmental & sustainability
            "sustainability_score", "renewable_energy_usage",
            "recycling_availability", "eco_tourism_options",
            
            # Accessibility
            "wheelchair_accessibility", "senior_friendliness",
            "family_facilities_score", "pet_friendliness",
            
            # Digital nomad factors
            "coworking_space_density", "digital_nomad_community",
            "visa_flexibility", "timezone_convenience",
            
            # Health & wellness
            "spa_wellness_options", "fitness_facility_density",
            "outdoor_exercise_options", "meditation_retreat_availability",
            
            # Cultural experiences
            "museum_gallery_density", "live_music_frequency",
            "local_craft_availability", "cooking_class_options",
            
            # Practical factors
            "atm_availability", "pharmacy_density",
            "embassy_consulate_presence", "emergency_service_quality"
        ]
        
        for dim in standard_dimensions:
            if dim not in self.dimensions:
                self.dimensions[dim] = DimensionValue(
                    value=None,
                    confidence=0.0
                )
    
    def get_current_themes(self, min_confidence: float = 0.5) -> List[Theme]:
        """Get themes above confidence threshold for current time slice"""
        current_themes = []
        for theme in self.themes:
            if theme.confidence_breakdown and theme.confidence_breakdown.overall_confidence >= min_confidence:
                current_themes.append(theme)
        return current_themes
    
    def get_temporal_slice(self, target_date: Optional[datetime] = None) -> Optional[TemporalSlice]:
        """Get temporal slice for a specific date"""
        if target_date is None:
            target_date = datetime.now()
            
        for slice in self.temporal_slices:
            if slice.valid_from <= target_date:
                if slice.valid_to is None or target_date < slice.valid_to:
                    return slice
        return None
    
    def add_theme(self, theme: Theme):
        """Add or update a theme"""
        # Check if theme already exists
        existing = next((t for t in self.themes if t.theme_id == theme.theme_id), None)
        if existing:
            # Merge evidence
            existing.evidence.extend(theme.evidence)
            existing.last_validated = datetime.now()
        else:
            self.themes.append(theme)
            
    def update_dimension(self, dimension_name: str, value: Any, 
                        unit: Optional[str] = None, confidence: float = 0.0,
                        evidence_ids: Optional[List[str]] = None):
        """Update a dimension value"""
        self.dimensions[dimension_name] = DimensionValue(
            value=value,
            unit=unit,
            confidence=confidence,
            source_evidence_ids=evidence_ids or [],
            last_updated=datetime.now()
        )
        
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage/serialization"""
        return {
            "id": self.id,
            "names": self.names,
            "admin_levels": self.admin_levels,
            "timezone": self.timezone,
            "population": self.population,
            "country_code": self.country_code,
            "core_geo": self.core_geo,
            "themes": [
                {
                    "theme_id": t.theme_id,
                    "macro_category": t.macro_category,
                    "micro_category": t.micro_category,
                    "name": t.name,
                    "description": t.description,
                    "fit_score": t.fit_score,
                    "evidence_count": len(t.evidence),
                    "confidence_level": t.get_confidence_level().value,
                    "tags": t.tags
                }
                for t in self.themes
            ],
            "dimensions": {
                name: {
                    "value": dv.value,
                    "unit": dv.unit,
                    "confidence": dv.confidence,
                    "last_updated": dv.last_updated.isoformat()
                }
                for name, dv in self.dimensions.items()
                if dv.value is not None
            },
            "poi_count": len(self.pois),
            "temporal_slices": len(self.temporal_slices),
            "authentic_insights": [ai.to_dict() for ai in self.authentic_insights],
            "local_authorities": [la.to_dict() for la in self.local_authorities],
            "last_updated": self.last_updated.isoformat(),
            "destination_revision": self.destination_revision
        } 