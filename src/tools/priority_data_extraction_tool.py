"""
Priority Data Extraction Tool
Uses LLM-based semantic understanding to extract structured travel data
from various sources with high accuracy and robustness.
"""
import json
import logging
from typing import Dict, Any, List, Optional
from datetime import datetime
from dataclasses import dataclass
from pydantic import BaseModel, Field

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from langchain_google_genai import ChatGoogleGenerativeAI

import re
import asyncio
from urllib.parse import urlparse

logger = logging.getLogger(__name__)


# Legacy dataclass structures for backward compatibility
@dataclass
class SafetyMetrics:
    """Safety metrics for a destination"""
    crime_index: Optional[float] = None
    safety_rating: Optional[float] = None
    tourist_police_available: Optional[bool] = None
    emergency_contacts: Dict[str, str] = None
    travel_advisory_level: Optional[str] = None
    recent_incidents: List[Dict[str, Any]] = None
    safe_areas: List[str] = None
    areas_to_avoid: List[str] = None
    
    def __post_init__(self):
        if self.emergency_contacts is None:
            self.emergency_contacts = {}
        if self.recent_incidents is None:
            self.recent_incidents = []
        if self.safe_areas is None:
            self.safe_areas = []
        if self.areas_to_avoid is None:
            self.areas_to_avoid = []


@dataclass
class CostIndicators:
    """Cost indicators for a destination"""
    budget_per_day_low: Optional[float] = None
    budget_per_day_mid: Optional[float] = None
    budget_per_day_high: Optional[float] = None
    meal_cost_average: Optional[float] = None
    accommodation_cost_average: Optional[float] = None
    transport_cost_average: Optional[float] = None
    currency: Optional[str] = None
    exchange_rate_info: Optional[str] = None
    seasonal_price_variation: Dict[str, float] = None
    
    def __post_init__(self):
        if self.seasonal_price_variation is None:
            self.seasonal_price_variation = {}


@dataclass
class HealthRequirements:
    """Health requirements and risks for a destination"""
    required_vaccinations: List[str] = None
    recommended_vaccinations: List[str] = None
    health_risks: List[str] = None
    water_safety: Optional[str] = None
    food_safety_rating: Optional[str] = None
    medical_facility_quality: Optional[str] = None
    health_insurance_required: Optional[bool] = None
    common_health_issues: List[str] = None
    
    def __post_init__(self):
        if self.required_vaccinations is None:
            self.required_vaccinations = []
        if self.recommended_vaccinations is None:
            self.recommended_vaccinations = []
        if self.health_risks is None:
            self.health_risks = []
        if self.common_health_issues is None:
            self.common_health_issues = []


@dataclass
class AccessibilityInfo:
    """Accessibility information for a destination"""
    visa_required: Optional[bool] = None
    visa_on_arrival: Optional[bool] = None
    visa_cost: Optional[float] = None
    direct_flights_from_major_hubs: List[str] = None
    average_flight_time: Optional[float] = None
    local_transport_quality: Optional[str] = None
    english_proficiency: Optional[str] = None
    infrastructure_rating: Optional[float] = None
    
    def __post_init__(self):
        if self.direct_flights_from_major_hubs is None:
            self.direct_flights_from_major_hubs = []


# Pydantic models for LLM structured output
class SafetyMetricsPydantic(BaseModel):
    """Pydantic model for safety metrics"""
    crime_index: Optional[float] = Field(None, description="Crime index value (0-100 scale)")
    safety_rating: Optional[float] = Field(None, description="Safety rating (typically 1-10 scale)")
    tourist_police_available: Optional[bool] = Field(None, description="Whether tourist police are available")
    emergency_contacts: Dict[str, str] = Field(default_factory=dict, description="Emergency contact numbers")
    travel_advisory_level: Optional[str] = Field(None, description="Official travel advisory level")
    safe_areas: List[str] = Field(default_factory=list, description="Areas considered safe for tourists")
    areas_to_avoid: List[str] = Field(default_factory=list, description="Areas to avoid or exercise caution")


class CostIndicatorsPydantic(BaseModel):
    """Pydantic model for cost indicators"""
    budget_per_day_low: Optional[float] = Field(None, description="Low-budget daily cost in USD")
    budget_per_day_mid: Optional[float] = Field(None, description="Mid-range daily cost in USD")
    budget_per_day_high: Optional[float] = Field(None, description="High-end daily cost in USD")
    meal_cost_average: Optional[float] = Field(None, description="Average meal cost in USD")
    accommodation_cost_average: Optional[float] = Field(None, description="Average accommodation cost per night in USD")
    transport_cost_average: Optional[float] = Field(None, description="Average local transport cost in USD")
    currency: Optional[str] = Field(None, description="Local currency code (e.g., USD, EUR, THB)")
    exchange_rate_info: Optional[str] = Field(None, description="Currency exchange rate information")
    seasonal_price_variation: Dict[str, float] = Field(default_factory=dict, description="Seasonal price changes as percentages")


class HealthRequirementsPydantic(BaseModel):
    """Pydantic model for health requirements"""
    required_vaccinations: List[str] = Field(default_factory=list, description="Mandatory vaccinations")
    recommended_vaccinations: List[str] = Field(default_factory=list, description="Recommended vaccinations")
    health_risks: List[str] = Field(default_factory=list, description="Health risks present in the destination")
    water_safety: Optional[str] = Field(None, description="Water safety information")
    food_safety_rating: Optional[str] = Field(None, description="Food safety rating or information")
    medical_facility_quality: Optional[str] = Field(None, description="Quality of medical facilities")
    health_insurance_required: Optional[bool] = Field(None, description="Whether health insurance is required/recommended")
    common_health_issues: List[str] = Field(default_factory=list, description="Common health issues for travelers")


class AccessibilityInfoPydantic(BaseModel):
    """Pydantic model for accessibility information"""
    visa_required: Optional[bool] = Field(None, description="Whether a visa is required")
    visa_on_arrival: Optional[bool] = Field(None, description="Whether visa on arrival is available")
    visa_cost: Optional[float] = Field(None, description="Visa cost in USD")
    direct_flights_from_major_hubs: List[str] = Field(default_factory=list, description="Cities with direct flights")
    average_flight_time: Optional[float] = Field(None, description="Average flight time in hours")
    local_transport_quality: Optional[str] = Field(None, description="Quality of local transportation")
    english_proficiency: Optional[str] = Field(None, description="English proficiency level (High/Moderate/Low)")
    infrastructure_rating: Optional[float] = Field(None, description="Infrastructure quality rating (1-5 scale)")


class ComprehensiveTravelData(BaseModel):
    """Complete travel data extraction result"""
    safety: SafetyMetricsPydantic
    cost: CostIndicatorsPydantic
    health: HealthRequirementsPydantic
    accessibility: AccessibilityInfoPydantic
    extraction_confidence: Optional[float] = Field(None, description="Confidence in extraction quality (0-1)")
    data_completeness: Optional[float] = Field(None, description="Percentage of fields populated")


class PriorityDataExtractor:
    """Extracts priority data from content using semantic analysis."""
    
    def __init__(self, llm=None):
        self.llm = llm
        self.logger = logging.getLogger(__name__)
        
        # Authority scoring
        self.high_authority_domains = [
            'gov', 'edu', 'travel.state.gov', 'who.int',
            'cdc.gov', 'europa.eu', 'un.org'
        ]
        self.medium_authority_domains = [
            'tripadvisor.com', 'booking.com', 'lonelyplanet.com',
            'timeout.com', 'frommers.com', 'fodors.com'
        ]
        
        # Confidence thresholds
        self.min_confidence = 0.3
        self.min_completeness = 0.4
        
        # Extraction patterns
        self.date_patterns = [
            r'last\s+updated:\s*(\d{4}-\d{2}-\d{2})',
            r'updated\s+on\s*(\d{4}-\d{2}-\d{2})',
            r'as\s+of\s*(\d{4}-\d{2}-\d{2})',
            r'current\s+as\s+of\s*(\d{4}-\d{2}-\d{2})'
        ]
        
        # Contradiction patterns
        self.contradiction_pairs = [
            (r'safe', r'dangerous'),
            (r'low crime', r'high crime'),
            (r'safe to drink', r'not safe to drink'),
            (r'affordable', r'expensive'),
            (r'accessible', r'inaccessible')
        ]
    
    def calculate_source_credibility(self, url: str) -> float:
        """Calculate source credibility score."""
        try:
            domain = urlparse(url).netloc.lower()
            
            # Check high authority domains
            if any(auth_domain in domain for auth_domain in self.high_authority_domains):
                return 0.9
            
            # Check medium authority domains
            if any(auth_domain in domain for auth_domain in self.medium_authority_domains):
                return 0.7
            
            # Default credibility for unknown sources
            return 0.5
            
        except Exception as e:
            self.logger.error(f"Error calculating source credibility: {e}")
            return 0.3
    
    def determine_temporal_relevance(self, content: str) -> float:
        """Determine temporal relevance of content."""
        try:
            # Try to find date in content
            for pattern in self.date_patterns:
                match = re.search(pattern, content.lower())
                if match:
                    date_str = match.group(1)
                    content_date = datetime.strptime(date_str, '%Y-%m-%d')
                    days_old = (datetime.now() - content_date).days
                    
                    # Score based on age
                    if days_old < 30:  # Less than a month old
                        return 0.9
                    elif days_old < 90:  # Less than 3 months old
                        return 0.8
                    elif days_old < 180:  # Less than 6 months old
                        return 0.6
                    elif days_old < 365:  # Less than a year old
                        return 0.4
                    else:
                        return 0.2
            
            # No date found
            return 0.5
            
        except Exception as e:
            self.logger.error(f"Error calculating temporal relevance: {e}")
            return 0.3
    
    def extract_safety_metrics(self, content: str, url: str = "unknown") -> Dict[str, Any]:
        """Extract safety-related information from content."""
        return self._extract_safety_metrics(content, url)
    
    def extract_cost_indicators(self, content: str, url: str = "unknown") -> Dict[str, Any]:
        """Extract cost-related information from content."""
        return self._extract_cost_indicators(content, url)
    
    def extract_health_requirements(self, content: str, url: str = "unknown") -> Dict[str, Any]:
        """Extract health-related requirements and information."""
        return self._extract_health_requirements(content, url)
    
    def extract_accessibility_info(self, content: str, url: str = "unknown") -> Dict[str, Any]:
        """Extract accessibility-related information."""
        return self._extract_accessibility_info(content, url)
    
    def _extract_safety_metrics(self, content: str, url: str) -> Dict[str, Any]:
        """Extract safety metrics using semantic analysis."""
        if self.llm:
            return self._extract_with_llm(content, SafetyMetricsPydantic, "safety")
        else:
            # Semantic fallback - more intelligent than pure regex
            return self._extract_safety_semantically(content)
    
    def _extract_cost_indicators(self, content: str, url: str) -> Dict[str, Any]:
        """Extract cost indicators using semantic analysis."""
        if self.llm:
            return self._extract_with_llm(content, CostIndicatorsPydantic, "cost")
        else:
            return self._extract_cost_semantically(content)
    
    def _extract_health_requirements(self, content: str, url: str) -> Dict[str, Any]:
        """Extract health requirements using semantic analysis."""
        if self.llm:
            return self._extract_with_llm(content, HealthRequirementsPydantic, "health")
        else:
            return self._extract_health_semantically(content)
    
    def _extract_accessibility_info(self, content: str, url: str) -> Dict[str, Any]:
        """Extract accessibility info using semantic analysis."""
        if self.llm:
            return self._extract_with_llm(content, AccessibilityInfoPydantic, "accessibility")
        else:
            return self._extract_accessibility_semantically(content)
    
    def _extract_with_llm(self, content: str, model_class, category: str) -> Dict[str, Any]:
        """Extract data using LLM with structured output."""
        try:
            # Create parser for the specific model
            parser = PydanticOutputParser(pydantic_object=model_class)
            
            # Create prompt template
            template = f"""
            Extract {category} information from the following travel content.
            Be precise and only extract information that is clearly stated.
            
            Content: {{content}}
            
            {{format_instructions}}
            """
            
            prompt = ChatPromptTemplate.from_template(template)
            chain = prompt | self.llm | parser
            
            result = chain.invoke({
                "content": content,
                "format_instructions": parser.get_format_instructions()
            })
            
            return result.dict()
            
        except Exception as e:
            self.logger.error(f"LLM extraction failed for {category}: {e}")
            # Fall back to semantic extraction
            return getattr(self, f"_extract_{category}_semantically")(content)
    
    def _extract_safety_semantically(self, content: str) -> Dict[str, Any]:
        """Semantic safety extraction with intelligent pattern matching."""
        safety_data = {
            "crime_index": None,
            "safety_rating": None,
            "tourist_police_available": None,
            "emergency_contacts": {},
            "travel_advisory_level": None,
            "safe_areas": [],
            "areas_to_avoid": [],
            "extraction_method": "semantic_fallback"
        }
        
        content_lower = content.lower()
        extraction_successes = 0
        total_attempts = 7  # Number of extraction categories we attempt
        
        # Semantic crime index extraction
        crime_patterns = [
            r'crime\s+(?:index|rate)[:]\s*(\d+\.?\d*)',
            r'safety\s+index[:]\s*(\d+\.?\d*)',
            r'(\d+\.?\d*)\s+crime\s+(?:index|rate)'
        ]
        for pattern in crime_patterns:
            match = re.search(pattern, content_lower)
            if match:
                safety_data["crime_index"] = float(match.group(1))
                extraction_successes += 1
                break
        
        # Semantic safety rating extraction - improved patterns
        rating_patterns = [
            r'rated\s+(\d+\.?\d*)\s+out\s+of\s+10\s+for\s+safety',  # "rated 6.8 out of 10 for safety"
            r'safety.*?(\d+\.?\d*)\s*(?:out\s+of\s+|\/)?\s*10',      # "safety 6.8/10"
            r'safety\s+(?:rating|score)[:]\s*(\d+\.?\d*)',           # "safety rating: 6.8"
            r'(\d+\.?\d*)\s*\/\s*10\s+(?:for\s+)?safety',           # "6.8/10 safety"
        ]
        for pattern in rating_patterns:
            match = re.search(pattern, content_lower)
            if match:
                safety_data["safety_rating"] = float(match.group(1))
                extraction_successes += 1
                break
        
        # Tourist police detection - improved
        if any(phrase in content_lower for phrase in ['tourist police are available', 'tourist police', 'police for tourists']):
            safety_data["tourist_police_available"] = True
            extraction_successes += 1
        elif any(phrase in content_lower for phrase in ['no tourist police', 'tourist police not available']):
            safety_data["tourist_police_available"] = False
            extraction_successes += 1
        
        # Emergency contacts extraction
        emergency_patterns = [
            (r'emergency[:]\s*(\d+)', 'emergency'),
            (r'police[:]\s*(\d+)', 'police'),
            (r'ambulance[:]\s*(\d+)', 'ambulance'),
            (r'fire[:]\s*(\d+)', 'fire')
        ]
        
        contacts_found = 0
        for pattern, contact_type in emergency_patterns:
            match = re.search(pattern, content_lower)
            if match:
                safety_data["emergency_contacts"][contact_type] = match.group(1)
                contacts_found += 1
        
        if contacts_found > 0:
            extraction_successes += 1
        
        # Travel advisory level
        advisory_patterns = [
            r'travel\s+advisory[:]\s*level\s+(\d+)',
            r'level\s+(\d+)\s*[-\s]*.*?caution',
            r'advisory\s+level[:]\s*(\d+)'
        ]
        
        for pattern in advisory_patterns:
            match = re.search(pattern, content_lower)
            if match:
                safety_data["travel_advisory_level"] = f"Level {match.group(1)}"
                extraction_successes += 1
                break
        
        # Safe areas extraction
        safe_area_patterns = [
            r'safe\s+areas?[:]\s*([^.]+)',
            r'safe\s+(?:districts?|neighborhoods?)[:]\s*([^.]+)',
            r'recommended\s+areas?[:]\s*([^.]+)',
            r'tourist\s+(?:zones?|areas?)[:]\s*([^.]+)'
        ]
        
        for pattern in safe_area_patterns:
            match = re.search(pattern, content_lower)
            if match:
                areas_text = match.group(1)
                # Split by common separators and clean up
                areas = [area.strip() for area in re.split(r'[,;]|and', areas_text) if area.strip()]
                # Clean up extra words like "the", "around"
                cleaned_areas = []
                for area in areas:
                    # Remove common filler words but keep the core location names
                    cleaned_area = re.sub(r'\b(?:the|around|near)\b', '', area).strip()
                    if cleaned_area:
                        cleaned_areas.append(cleaned_area)
                safety_data["safe_areas"].extend(cleaned_areas)
                if cleaned_areas:
                    extraction_successes += 1
                break
        
        # Areas to avoid extraction
        avoid_patterns = [
            r'avoid\s+areas?\s*(?:around|near)?\s*([^.]+)',
            r'dangerous\s+areas?[:]\s*([^.]+)',
            r'areas?\s+to\s+avoid[:]\s*([^.]+)',
            r'(?:stay\s+away\s+from|don\'t\s+go\s+to)[:]\s*([^.]+)'
        ]
        
        for pattern in avoid_patterns:
            match = re.search(pattern, content_lower)
            if match:
                areas_text = match.group(1)
                # Split by common separators and clean up
                areas = [area.strip() for area in re.split(r'[,;]|and', areas_text) if area.strip()]
                # Clean up extra words
                cleaned_areas = []
                for area in areas:
                    # Remove common filler words but keep location names
                    cleaned_area = re.sub(r'\b(?:around|near|some\s+parts?\s+of|late\s+at\s+night)\b', '', area).strip()
                    if cleaned_area:
                        cleaned_areas.append(cleaned_area)
                safety_data["areas_to_avoid"].extend(cleaned_areas)
                if cleaned_areas:
                    extraction_successes += 1
                break
        
        # Calculate extraction confidence
        safety_data["extraction_confidence"] = extraction_successes / total_attempts
        
        return safety_data
    
    def _extract_cost_semantically(self, content: str) -> Dict[str, Any]:
        """Semantic cost extraction with intelligent pattern matching."""
        cost_data = {
            "budget_per_day_low": None,
            "budget_per_day_mid": None, 
            "budget_per_day_high": None,
            "meal_cost_average": None,
            "accommodation_cost_average": None,
            "currency": None,
            "seasonal_price_variation": {},
            "extraction_method": "semantic_fallback"
        }
        
        content_lower = content.lower()
        extraction_successes = 0
        total_attempts = 6  # Budget low, mid, high, currency, meals, accommodation
        
        # Extract budget travelers (low budget)
        budget_low_patterns = [
            r'budget\s+travelers?[:]\s*\$(\d+)[-\s]*(?:to\s+)?\$?(\d+)?\s*per\s+day',
            r'budget[:]\s*\$(\d+)[-\s]*(?:to\s+)?\$?(\d+)?\s*per\s+day',
            r'backpackers?[:]\s*\$(\d+)[-\s]*(?:to\s+)?\$?(\d+)?\s*per\s+day'
        ]
        
        for pattern in budget_low_patterns:
            match = re.search(pattern, content_lower)
            if match:
                cost_data["budget_per_day_low"] = float(match.group(1))
                extraction_successes += 1
                break
        
        # Extract mid-range budget
        budget_mid_patterns = [
            r'mid[-\s]*range[:]\s*\$(\d+)[-\s]*(?:to\s+)?\$?(\d+)?\s*per\s+day',
            r'mid[-\s]*range\s+travelers?[:]\s*\$(\d+)[-\s]*(?:to\s+)?\$?(\d+)?\s*per\s+day',
            r'moderate\s+budget[:]\s*\$(\d+)[-\s]*(?:to\s+)?\$?(\d+)?\s*per\s+day'
        ]
        
        for pattern in budget_mid_patterns:
            match = re.search(pattern, content_lower)
            if match:
                cost_data["budget_per_day_mid"] = float(match.group(1))
                extraction_successes += 1
                break
        
        # Extract luxury/high-end budget
        budget_high_patterns = [
            r'luxury[:]\s*\$(\d+)\+?\s*per\s+day',
            r'high[-\s]*end[:]\s*\$(\d+)\+?\s*per\s+day',
            r'luxury\s+travel[:]\s*\$(\d+)\+?\s*per\s+day'
        ]
        
        for pattern in budget_high_patterns:
            match = re.search(pattern, content_lower)
            if match:
                cost_data["budget_per_day_high"] = float(match.group(1))
                extraction_successes += 1
                break
        
        # Extract meal costs
        meal_patterns = [
            r'meals?\s+cost[:]\s*\$(\d+)[-\s]*(?:to\s+)?\$?(\d+)?',
            r'food\s+costs?[:]\s*\$(\d+)[-\s]*(?:to\s+)?\$?(\d+)?',
            r'dining[:]\s*\$(\d+)[-\s]*(?:to\s+)?\$?(\d+)?'
        ]
        
        meal_costs = []
        for pattern in meal_patterns:
            matches = re.finditer(pattern, content_lower)
            for match in matches:
                meal_costs.append(float(match.group(1)))
                if match.group(2):
                    meal_costs.append(float(match.group(2)))
        
        # Also extract from more detailed meal descriptions
        detailed_meal_patterns = [
            r'\$(\d+)[-\s]*(?:to\s+)?\$?(\d+)?\s+for\s+(?:street\s+food|restaurants?|meals?)',
            r'(?:street\s+food|restaurants?|meals?)[:]\s*\$(\d+)[-\s]*(?:to\s+)?\$?(\d+)?'
        ]
        
        for pattern in detailed_meal_patterns:
            matches = re.finditer(pattern, content_lower)
            for match in matches:
                meal_costs.append(float(match.group(1)))
                if match.group(2):
                    meal_costs.append(float(match.group(2)))
        
        if meal_costs:
            cost_data["meal_cost_average"] = sum(meal_costs) / len(meal_costs)
            extraction_successes += 1
        
        # Extract accommodation costs
        accommodation_patterns = [
            r'(?:budget\s+)?accommodation[:]\s*\$(\d+)[-\s]*(?:to\s+)?\$?(\d+)?\s*per\s+night',
            r'hotels?[:]\s*\$(\d+)[-\s]*(?:to\s+)?\$?(\d+)?\s*per\s+night',
            r'hostels?[:]\s*\$(\d+)[-\s]*(?:to\s+)?\$?(\d+)?\s*per\s+night'
        ]
        
        accommodation_costs = []
        for pattern in accommodation_patterns:
            matches = re.finditer(pattern, content_lower)
            for match in matches:
                accommodation_costs.append(float(match.group(1)))
                if match.group(2):
                    accommodation_costs.append(float(match.group(2)))
        
        if accommodation_costs:
            cost_data["accommodation_cost_average"] = sum(accommodation_costs) / len(accommodation_costs)
            extraction_successes += 1
        
        # Currency extraction with context
        currency_patterns = [
            r'currency[:]\s*([a-z]{3})\s+is\s+the\s+local\s+currency',
            r'local\s+currency[:]\s*([a-z]{3})',
            r'currency[:]\s*([a-z]{3})',
            r'prices?\s+in\s+([a-z]{3})'
        ]
        
        for pattern in currency_patterns:
            match = re.search(pattern, content_lower)
            if match:
                cost_data["currency"] = match.group(1).upper()  # Convert back to uppercase
                extraction_successes += 1
                break
        
        # Extract seasonal price variations
        seasonal_patterns = [
            r'prices?\s+increase\s+(\d+)%\s+during\s+(?:high\s+season|peak|summer)',
            r'prices?\s+decrease\s+(\d+)%\s+during\s+(?:low\s+season|off[-\s]*season)',
            r'(\d+)%\s+(?:more\s+expensive|higher)\s+during\s+(?:high\s+season|peak)',
            r'(\d+)%\s+(?:cheaper|lower)\s+during\s+(?:low\s+season|off[-\s]*season)'
        ]
        
        for pattern in seasonal_patterns:
            match = re.search(pattern, content_lower)
            if match:
                percentage = float(match.group(1))
                if 'increase' in pattern or 'higher' in pattern or 'expensive' in pattern:
                    cost_data["seasonal_price_variation"]["high_season"] = percentage
                elif 'decrease' in pattern or 'lower' in pattern or 'cheaper' in pattern:
                    cost_data["seasonal_price_variation"]["low_season"] = -percentage
        
        # Calculate extraction confidence
        cost_data["extraction_confidence"] = extraction_successes / total_attempts
        
        return cost_data
    
    def _extract_health_semantically(self, content: str) -> Dict[str, Any]:
        """Semantic health extraction with intelligent pattern matching."""
        health_data = {
            "required_vaccinations": [],
            "recommended_vaccinations": [],
            "health_risks": [],
            "water_safety": None,
            "medical_facility_quality": None,
            "health_insurance_required": None,
            "extraction_method": "semantic_fallback"
        }
        
        content_lower = content.lower()
        extraction_successes = 0
        total_attempts = 5  # Required vaccines, recommended vaccines, health risks, water safety, medical facilities
        
        # Extract required vaccinations
        req_patterns = [
            r'required\s+vaccinations?[:]\s*([^.]+)',
            r'mandatory\s+vaccinations?[:]\s*([^.]+)',
            r'must\s+have\s+vaccinations?[:]\s*([^.]+)'
        ]
        
        for pattern in req_patterns:
            match = re.search(pattern, content_lower)
            if match:
                vaccines_text = match.group(1)
                vaccines = [v.strip() for v in re.split(r'[,;]|and', vaccines_text) if v.strip()]
                health_data["required_vaccinations"].extend(vaccines)
                if vaccines:
                    extraction_successes += 1
        
        # Extract recommended vaccinations
        rec_patterns = [
            r'recommended\s+vaccinations?[:]\s*([^.]+)',
            r'suggested\s+vaccinations?[:]\s*([^.]+)',
            r'advisable\s+vaccinations?[:]\s*([^.]+)'
        ]
        
        for pattern in rec_patterns:
            match = re.search(pattern, content_lower)
            if match:
                vaccines_text = match.group(1)
                vaccines = [v.strip() for v in re.split(r'[,;]|and', vaccines_text) if v.strip()]
                health_data["recommended_vaccinations"].extend(vaccines)
                if vaccines:
                    extraction_successes += 1
        
        # Extract health risks - Enhanced to catch specific risks mentioned
        risk_patterns = [
            r'health\s+risks?[:]\s*([^.]+)',
            r'diseases?\s+present[:]\s*([^.]+)',
            r'(\w+\s+(?:fever|virus|disease))\s+is\s+present',
            r'(\w+)\s+risk\s+exists?',
            r'risk\s+of\s+(\w+(?:\s+\w+)?)',
        ]
        
        risks_found = set()
        for pattern in risk_patterns:
            matches = re.finditer(pattern, content_lower)
            for match in matches:
                risk_text = match.group(1)
                # Split multiple risks if in first group
                if ':' in pattern and match.group(1):
                    # Handle "Health risks: Dengue fever is present, especially during rainy season."
                    risk_parts = re.split(r'[,;]|\.', risk_text)
                    for part in risk_parts:
                        part = part.strip()
                        if part and 'especially' not in part and 'during' not in part:
                            risks_found.add(part)
                else:
                    # Handle individual risk matches
                    risks_found.add(risk_text.strip())
        
        if risks_found:
            health_data["health_risks"] = list(risks_found)
            extraction_successes += 1
        
        # Water safety semantic analysis
        if any(phrase in content_lower for phrase in ['water not safe', 'tap water unsafe', 'bottled water recommended']):
            health_data["water_safety"] = "Not safe to drink"
            extraction_successes += 1
        elif any(phrase in content_lower for phrase in ['water safe', 'tap water safe', 'potable water']):
            health_data["water_safety"] = "Safe to drink"
            extraction_successes += 1
        
        # Medical facility quality extraction
        facility_patterns = [
            r'medical\s+facilities?\s+(?:are\s+)?(\w+)',
            r'hospitals?\s+(?:are\s+)?(\w+)',
            r'healthcare\s+is\s+(\w+)'
        ]
        
        for pattern in facility_patterns:
            match = re.search(pattern, content_lower)
            if match:
                quality = match.group(1).strip()
                if quality in ['excellent', 'good', 'poor', 'adequate', 'limited']:
                    health_data["medical_facility_quality"] = quality.title()
                    extraction_successes += 1
                    break
        
        # Calculate extraction confidence
        health_data["extraction_confidence"] = extraction_successes / total_attempts
        
        return health_data
    
    def _extract_accessibility_semantically(self, content: str) -> Dict[str, Any]:
        """Semantic accessibility extraction with intelligent pattern matching."""
        accessibility_data = {
            "visa_required": None,
            "visa_on_arrival": None,
            "visa_cost": None,
            "direct_flights_from_major_hubs": [],
            "average_flight_time": None,
            "english_proficiency": None,
            "infrastructure_rating": None,
            "extraction_method": "semantic_fallback"
        }
        
        content_lower = content.lower()
        extraction_successes = 0
        total_attempts = 6  # Visa required, visa cost, direct flights, english proficiency, infrastructure, flight time
        
        # Semantic visa analysis - improved logic
        if any(phrase in content_lower for phrase in ['visa on-arrival', 'visa on arrival', 'arrival visa']):
            accessibility_data["visa_required"] = True  # Visa is required, but available on arrival
            accessibility_data["visa_on_arrival"] = True
            extraction_successes += 1
        elif any(phrase in content_lower for phrase in ['visa required', 'need visa', 'must have visa']):
            accessibility_data["visa_required"] = True
            extraction_successes += 1
        elif any(phrase in content_lower for phrase in ['no visa required', 'visa-free', 'visa not needed']):
            accessibility_data["visa_required"] = False
            extraction_successes += 1
        
        # Visa cost with better context matching
        visa_cost_patterns = [
            r'visa\s+on[-\s]*arrival\s+for\s+\$(\d+)',
            r'visa.*?\$(\d+).*?for\s+most\s+nationalities',
            r'\$(\d+).*?visa',
            r'visa\s+(?:costs?|fees?).*?\$(\d+)'
        ]
        
        for pattern in visa_cost_patterns:
            match = re.search(pattern, content_lower)
            if match:
                accessibility_data["visa_cost"] = float(match.group(1))
                extraction_successes += 1
                break
        
        # Direct flights extraction
        flight_patterns = [
            r'direct\s+flights?\s+from\s+([^.]+)',
            r'non[-\s]*stop\s+flights?\s+from\s+([^.]+)',
            r'flights?\s+from\s+([^.]+)'
        ]
        
        for pattern in flight_patterns:
            match = re.search(pattern, content_lower)
            if match:
                cities_text = match.group(1)
                cities = [city.strip() for city in re.split(r'[,;]|and', cities_text) if city.strip()]
                accessibility_data["direct_flights_from_major_hubs"].extend(cities)
                if cities:
                    extraction_successes += 1
                break
        
        # English proficiency detection
        if any(phrase in content_lower for phrase in ['english is widely spoken', 'english widely spoken', 'good english']):
            accessibility_data["english_proficiency"] = "High"
            extraction_successes += 1
        elif any(phrase in content_lower for phrase in ['limited english', 'poor english', 'little english']):
            accessibility_data["english_proficiency"] = "Low"
            extraction_successes += 1
        elif any(phrase in content_lower for phrase in ['some english', 'moderate english', 'basic english']):
            accessibility_data["english_proficiency"] = "Moderate"
            extraction_successes += 1
        
        # Infrastructure rating - convert qualitative to quantitative
        infrastructure_patterns = [
            r'infrastructure\s+is\s+(\w+)',
            r'(\w+)\s+infrastructure',
            r'transportation\s+systems?\s+(?:are\s+)?(\w+)'
        ]
        
        infrastructure_mapping = {
            'excellent': 5.0,
            'very good': 4.5,
            'good': 4.0,
            'modern': 4.0,
            'adequate': 3.0,
            'poor': 2.0,
            'limited': 1.5,
            'bad': 1.0
        }
        
        for pattern in infrastructure_patterns:
            match = re.search(pattern, content_lower)
            if match:
                quality = match.group(1).strip()
                if quality in infrastructure_mapping:
                    accessibility_data["infrastructure_rating"] = infrastructure_mapping[quality]
                    extraction_successes += 1
                    break
        
        # Average flight time extraction
        flight_time_patterns = [
            r'average\s+flight\s+time.*?(\d+)\s+hours?',
            r'flight\s+time.*?(\d+)\s+hours?',
            r'(\d+)\s+hours?\s+flight',
            r'takes\s+(\d+)\s+hours?\s+to\s+fly'
        ]
        
        for pattern in flight_time_patterns:
            match = re.search(pattern, content_lower)
            if match:
                accessibility_data["average_flight_time"] = float(match.group(1))
                extraction_successes += 1
                break
        
        # Calculate extraction confidence
        accessibility_data["extraction_confidence"] = extraction_successes / total_attempts
        
        return accessibility_data
    
    def extract_all_priority_data(self, content: str, url: str = "unknown") -> Dict[str, Any]:
        """Extract all priority data with metadata."""
        try:
            # Extract individual components
            safety_metrics = self.extract_safety_metrics(content, url)
            cost_indicators = self.extract_cost_indicators(content, url)
            health_requirements = self.extract_health_requirements(content, url)
            accessibility_info = self.extract_accessibility_info(content, url)
            
            # Calculate metadata
            source_credibility = self.calculate_source_credibility(url)
            temporal_relevance = self.determine_temporal_relevance(content)
            contradictions = self._detect_contradictions(content)
            
            # Calculate sophisticated confidence and completeness
            extraction_confidence = self._calculate_extraction_confidence([
                safety_metrics.get('extraction_confidence', self._calculate_individual_confidence(safety_metrics)),
                cost_indicators.get('extraction_confidence', self._calculate_individual_confidence(cost_indicators)),
                health_requirements.get('extraction_confidence', self._calculate_individual_confidence(health_requirements)),
                accessibility_info.get('extraction_confidence', self._calculate_individual_confidence(accessibility_info))
            ])
            
            data_completeness = self._calculate_data_completeness({
                'safety': safety_metrics,
                'cost': cost_indicators,
                'health': health_requirements,
                'accessibility': accessibility_info
            })
            
            # Apply source and temporal factors to confidence
            final_confidence = extraction_confidence * source_credibility * temporal_relevance
            
            return {
                'safety': safety_metrics,
                'cost': cost_indicators,
                'health': health_requirements,
                'accessibility': accessibility_info,
                'extraction_confidence': final_confidence,
                'data_completeness': data_completeness,
                'source_credibility': source_credibility,
                'temporal_relevance': temporal_relevance,
                'contradictions_detected': contradictions,
                'extraction_timestamp': datetime.now().isoformat(),
                'source_url': url
            }
            
        except Exception as e:
            self.logger.error(f"Error extracting all priority data: {e}")
            return {
                'extraction_confidence': 0.0,
                'data_completeness': 0.0,
                'error': str(e),
                'extraction_timestamp': datetime.now().isoformat(),
                'source_url': url
            }
    
    def _calculate_individual_confidence(self, extracted_data: Dict[str, Any]) -> float:
        """Calculate confidence for individual extraction based on data quality."""
        if not extracted_data or extracted_data.get('error'):
            return 0.0
        
        confidence_factors = []
        
        # Factor 1: Field population rate
        total_fields = 0
        populated_fields = 0
        
        for key, value in extracted_data.items():
            if key not in ['extraction_method', 'error']:
                total_fields += 1
                if value is not None:
                    if isinstance(value, (list, dict)):
                        if value:  # Non-empty list/dict
                            populated_fields += 1
                    else:
                        populated_fields += 1
        
        if total_fields > 0:
            population_rate = populated_fields / total_fields
            confidence_factors.append(population_rate)
        
        # Factor 2: Data type consistency
        type_consistency = 1.0
        expected_types = {
            'crime_index': (int, float),
            'safety_rating': (int, float),
            'budget_per_day_low': (int, float),
            'visa_cost': (int, float),
            'required_vaccinations': list,
            'emergency_contacts': dict,
            'safe_areas': list
        }
        
        for field, expected_type in expected_types.items():
            if field in extracted_data and extracted_data[field] is not None:
                if not isinstance(extracted_data[field], expected_type):
                    type_consistency -= 0.1
        
        confidence_factors.append(max(0.0, type_consistency))
        
        # Factor 3: Extraction method bonus
        if extracted_data.get('extraction_method') == 'semantic_fallback':
            confidence_factors.append(0.8)  # Good semantic extraction
        elif extracted_data.get('extraction_method') == 'llm':
            confidence_factors.append(0.95)  # Excellent LLM extraction
        else:
            confidence_factors.append(0.6)  # Default confidence
        
        # Factor 4: Content quality indicators
        content_quality = 0.7  # Base quality
        
        # Check for specific high-value extractions
        if extracted_data.get('crime_index') is not None:
            content_quality += 0.1
        if extracted_data.get('currency') is not None:
            content_quality += 0.1
        if extracted_data.get('visa_cost') is not None:
            content_quality += 0.1
        if extracted_data.get('required_vaccinations') and len(extracted_data['required_vaccinations']) > 0:
            content_quality += 0.1
        
        confidence_factors.append(min(1.0, content_quality))
        
        # Calculate weighted average
        return sum(confidence_factors) / len(confidence_factors) if confidence_factors else 0.0
    
    def _calculate_extraction_confidence(self, confidences: list) -> float:
        """Calculate overall extraction confidence."""
        if not confidences:
            return 0.0
        return sum(confidences) / len(confidences)
    
    def _calculate_data_completeness(self, data: Dict[str, Any]) -> float:
        """Calculate data completeness score."""
        total_fields = 0
        filled_fields = 0
        
        def count_fields(d):
            nonlocal total_fields, filled_fields
            for k, v in d.items():
                if k not in ['extraction_confidence', 'error']:
                    if isinstance(v, dict):
                        count_fields(v)
                    else:
                        total_fields += 1
                        if v is not None and v != [] and v != {}:
                            filled_fields += 1
        
        for category in data.values():
            if isinstance(category, dict):
                count_fields(category)
        
        return filled_fields / total_fields if total_fields > 0 else 0.0
    
    async def extract_all_priority_data_async(self, content: str, url: str = "unknown") -> Dict[str, Any]:
        """Asynchronous version of extract_all_priority_data."""
        return await asyncio.to_thread(self.extract_all_priority_data, content, url)
    
    def _parse_llm_response(self, response: str) -> Dict[str, Any]:
        """Parse LLM response into structured data."""
        try:
            # Implementation depends on LLM response format
            # This is a placeholder
            return {}
        except Exception as e:
            self.logger.error(f"Error parsing LLM response: {e}")
            return {}
    
    def _extract_context(self, text: str, position: int, window: int) -> str:
        """Extract context around a position in text."""
        start = max(0, position - window)
        end = min(len(text), position + window)
        return text[start:end].strip()
    
    def _update_cost_category(self, indicators: Dict[str, Any], category: str, values: tuple, context: str):
        """Update cost category with extracted values."""
        if not values:
            return
        
        min_val = float(values[0]) if values[0] else 0
        max_val = float(values[1]) if values[1] else min_val
        
        if 'budget' in context.lower() or min_val < 50:
            indicators[category]['budget'] = min_val
        elif 'luxury' in context.lower() or min_val > 200:
            indicators[category]['luxury'] = max_val
        else:
            indicators[category]['mid'] = (min_val + max_val) / 2
    
    def _update_vaccination_info(self, requirements: Dict[str, Any], context: str):
        """Update vaccination information based on context."""
        if 'required' in context.lower():
            vaccines = re.findall(r'(?:required|mandatory).*?([A-Za-z\s,]+)(?:vaccine|vaccination)', context.lower())
            if vaccines:
                requirements['required_vaccinations'].extend(v.strip() for v in vaccines[0].split(','))
        elif 'recommended' in context.lower():
            vaccines = re.findall(r'recommended.*?([A-Za-z\s,]+)(?:vaccine|vaccination)', context.lower())
            if vaccines:
                requirements['recommended_vaccinations'].extend(v.strip() for v in vaccines[0].split(','))
    
    def _classify_water_safety(self, context: str) -> str:
        """Classify water safety based on context."""
        if any(phrase in context.lower() for phrase in ['safe to drink', 'potable']):
            return 'safe'
        elif any(phrase in context.lower() for phrase in ['not safe', 'unsafe', 'boil', 'bottled']):
            return 'unsafe'
        return 'unknown'
    
    def _classify_insurance_requirement(self, context: str) -> str:
        """Classify insurance requirement based on context."""
        if any(phrase in context.lower() for phrase in ['required', 'mandatory', 'must have']):
            return 'required'
        elif any(phrase in context.lower() for phrase in ['recommended', 'advised']):
            return 'recommended'
        return 'unknown'
    
    def _classify_visa_requirement(self, context: str) -> str:
        """Classify visa requirement based on context."""
        if 'not required' in context.lower() or 'visa-free' in context.lower():
            return 'not required'
        elif 'on arrival' in context.lower():
            return 'on arrival'
        elif any(phrase in context.lower() for phrase in ['required', 'needed', 'mandatory']):
            return 'required'
        return 'unknown'
    
    def _classify_language_accessibility(self, context: str) -> str:
        """Classify language accessibility based on context."""
        if any(phrase in context.lower() for phrase in ['widely spoken', 'common', 'good']):
            return 'high'
        elif any(phrase in context.lower() for phrase in ['limited', 'basic', 'poor']):
            return 'low'
        return 'moderate'
    
    def _classify_physical_accessibility(self, context: str) -> str:
        """Classify physical accessibility based on context."""
        if any(phrase in context.lower() for phrase in ['good access', 'accessible', 'available']):
            return 'good'
        elif any(phrase in context.lower() for phrase in ['limited', 'poor', 'difficult']):
            return 'limited'
        return 'moderate'
    
    def _detect_contradictions(self, content: str) -> Dict[str, Any]:
        """Detect contradictions in content."""
        contradictions = []
        content_lower = content.lower()
        
        for positive, negative in self.contradiction_pairs:
            if re.search(positive, content_lower) and re.search(negative, content_lower):
                contradictions.append({
                    'type': f'{positive} vs {negative}',
                    'context': self._extract_context(content_lower, content_lower.find(positive), 50)
                })
        
        return {
            'found': len(contradictions) > 0,
            'details': contradictions
        }


# Factory function for easy instantiation
def create_priority_extractor(llm: Optional[ChatGoogleGenerativeAI] = None) -> PriorityDataExtractor:
    """Create a priority data extractor instance"""
    return PriorityDataExtractor(llm=llm) 