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
    """Semantic-based priority data extractor using LLM understanding"""
    
    def __init__(self, llm: Optional[ChatGoogleGenerativeAI] = None):
        self.logger = logging.getLogger(__name__)
        
        try:
            self.llm = llm or self._create_default_llm()
            self.parser = PydanticOutputParser(pydantic_object=ComprehensiveTravelData)
            
            # Create the extraction prompt template
            self.extraction_prompt = ChatPromptTemplate.from_messages([
                ("system", self._get_system_prompt()),
                ("human", self._get_human_prompt())
            ])
            
            # Create the extraction chain
            self.extraction_chain = self.extraction_prompt | self.llm | self.parser
            self.semantic_enabled = True
            
        except Exception as e:
            self.logger.warning(f"Could not initialize semantic LLM extraction: {e}. Using fallback mode.")
            self.llm = None
            self.parser = None
            self.extraction_prompt = None
            self.extraction_chain = None
            self.semantic_enabled = False
    
    def _create_default_llm(self) -> ChatGoogleGenerativeAI:
        """Create default LLM if none provided"""
        try:
            return ChatGoogleGenerativeAI(
                model="gemini-1.5-flash",
                temperature=0.1,  # Low temperature for consistency
                max_tokens=4000
            )
        except Exception as e:
            # If we can't create the LLM (no API key, etc.), we'll use fallback mode
            raise e
    
    def _get_system_prompt(self) -> str:
        """Get the system prompt for extraction"""
        return """You are an expert travel data analyst with deep knowledge of international travel requirements, safety conditions, costs, and accessibility information.

Your task is to extract comprehensive, accurate travel information from text content and structure it according to the specified schema.

Key Guidelines:
1. **Accuracy**: Only extract information explicitly mentioned or clearly implied in the text
2. **Context Awareness**: Understand what numbers refer to (e.g., distinguish visa cost from meal cost)
3. **Semantic Understanding**: Recognize synonyms and varied expressions (e.g., "tourist police" = "tourism police" = "tourist assistance officers")
4. **Data Validation**: Ensure extracted values make logical sense (e.g., safety ratings within reasonable ranges)
5. **Completeness**: Extract as much relevant information as possible while maintaining accuracy

For missing information:
- Use null/None for optional fields with no information
- Use empty lists/dicts for collection fields with no data
- Do not make up or guess information not present in the text

For ambiguous information:
- Choose the most reasonable interpretation based on context
- For numerical ranges, extract specific values when possible or use midpoint for averages

Currency and Cost Guidelines:
- Convert costs to USD when possible and reasonable
- If original currency is mentioned, note it in the currency field
- For daily budgets, extract low/mid/high ranges when available

Safety Guidelines:
- Extract numerical safety/crime indices when mentioned
- Recognize various travel advisory formats (Level 1-4, color codes, text descriptions)
- Distinguish between tourist police and regular police
- Identify specific safe areas and areas to avoid

Health Guidelines:
- Distinguish between required and recommended vaccinations
- Identify specific disease risks and health concerns
- Categorize water safety clearly (safe/unsafe/bottled recommended)
- Assess medical facility quality from descriptive text

Accessibility Guidelines:
- Understand visa requirements vs. visa on arrival vs. visa-free
- Extract specific visa costs
- Identify major cities with direct flights
- Assess English proficiency and infrastructure quality from context"""

    def _get_human_prompt(self) -> str:
        """Get the human prompt template"""
        return """Extract comprehensive travel information from the following content about a destination.

Content to analyze:
{content}

Source URL (for credibility assessment): {source_url}

Please extract all relevant safety, cost, health, and accessibility information according to the schema. Focus on accuracy and semantic understanding rather than pattern matching.

{format_instructions}"""

    def extract_all_priority_data(self, content: str, source_url: str = None) -> Dict[str, Any]:
        """Extract all priority data using semantic understanding"""
        
        # If semantic extraction is not available, use fallback mode
        if not self.semantic_enabled:
            return self._extract_all_priority_data_fallback(content, source_url)
        
        try:
            # Prepare the input
            format_instructions = self.parser.get_format_instructions()
            
            # Run the extraction
            result = self.extraction_chain.invoke({
                "content": content,
                "source_url": source_url or "Unknown",
                "format_instructions": format_instructions
            })
            
            # Convert to dictionary and add metadata
            result_dict = result.dict()
            
            # Add metadata
            result_dict.update({
                "source_url": source_url,
                "extraction_timestamp": datetime.now().isoformat(),
                "extraction_method": "semantic_llm",
                "source_credibility": self.calculate_source_credibility(source_url),
                "temporal_relevance": self.determine_temporal_relevance(content)
            })
            
            # Calculate completeness and confidence
            result_dict["data_completeness"] = self._calculate_data_completeness(result_dict)
            result_dict["extraction_confidence"] = self._calculate_extraction_confidence(result_dict, content)
            
            return result_dict
            
        except Exception as e:
            self.logger.error(f"Semantic extraction failed: {str(e)}")
            # Fallback to empty structure
            return self._create_empty_result(source_url)
    
    def _extract_all_priority_data_fallback(self, content: str, source_url: str = None) -> Dict[str, Any]:
        """Fallback extraction method when LLM is not available"""
        self.logger.info("Using fallback extraction method (no LLM available)")
        
        return {
            "safety": SafetyMetrics().__dict__,
            "cost": CostIndicators().__dict__,
            "health": HealthRequirements().__dict__,
            "accessibility": AccessibilityInfo().__dict__,
            "source_url": source_url,
            "extraction_timestamp": datetime.now().isoformat(),
            "extraction_method": "fallback_basic",
            "source_credibility": self.calculate_source_credibility(source_url),
            "temporal_relevance": self.determine_temporal_relevance(content),
            "data_completeness": 0.0,
            "extraction_confidence": 0.3
        }
    
    def extract_safety_metrics(self, content: str) -> Dict[str, Any]:
        """Extract only safety metrics with focused semantic analysis"""
        if not self.semantic_enabled:
            return SafetyMetrics().__dict__
            
        safety_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a travel safety expert. Extract safety-related information from travel content.
            Focus on: crime indices, safety ratings, police presence, emergency contacts, travel advisories, safe/unsafe areas.
            Be precise and only extract explicitly mentioned information."""),
            ("human", "Extract safety information from: {content}\n\n{format_instructions}")
        ])
        
        parser = PydanticOutputParser(pydantic_object=SafetyMetricsPydantic)
        chain = safety_prompt | self.llm | parser
        
        try:
            result = chain.invoke({
                "content": content,
                "format_instructions": parser.get_format_instructions()
            })
            return result.dict()
        except Exception as e:
            self.logger.error(f"Safety extraction failed: {str(e)}")
            return SafetyMetrics().__dict__
    
    def extract_cost_indicators(self, content: str) -> Dict[str, Any]:
        """Extract only cost indicators with focused semantic analysis"""
        if not self.semantic_enabled:
            return CostIndicators().__dict__
            
        cost_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a travel cost expert. Extract cost and budget information from travel content.
            Focus on: daily budgets, meal costs, accommodation prices, transport costs, currency, seasonal variations.
            Convert to USD when possible and note original currency."""),
            ("human", "Extract cost information from: {content}\n\n{format_instructions}")
        ])
        
        parser = PydanticOutputParser(pydantic_object=CostIndicatorsPydantic)
        chain = cost_prompt | self.llm | parser
        
        try:
            result = chain.invoke({
                "content": content,
                "format_instructions": parser.get_format_instructions()
            })
            return result.dict()
        except Exception as e:
            self.logger.error(f"Cost extraction failed: {str(e)}")
            return CostIndicators().__dict__
    
    def extract_health_requirements(self, content: str) -> Dict[str, Any]:
        """Extract only health requirements with focused semantic analysis"""
        if not self.semantic_enabled:
            return HealthRequirements().__dict__
            
        health_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a travel health expert. Extract health and medical information from travel content.
            Focus on: vaccinations (required vs recommended), health risks, water/food safety, medical facilities.
            Distinguish clearly between mandatory and optional health measures."""),
            ("human", "Extract health information from: {content}\n\n{format_instructions}")
        ])
        
        parser = PydanticOutputParser(pydantic_object=HealthRequirementsPydantic)
        chain = health_prompt | self.llm | parser
        
        try:
            result = chain.invoke({
                "content": content,
                "format_instructions": parser.get_format_instructions()
            })
            return result.dict()
        except Exception as e:
            self.logger.error(f"Health extraction failed: {str(e)}")
            return HealthRequirements().__dict__
    
    def extract_accessibility_info(self, content: str) -> Dict[str, Any]:
        """Extract only accessibility information with focused semantic analysis"""
        if not self.semantic_enabled:
            return AccessibilityInfo().__dict__
            
        access_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a travel accessibility expert. Extract entry requirements and accessibility information.
            Focus on: visa requirements, flight connections, language barriers, infrastructure quality.
            Distinguish between visa required, visa on arrival, and visa-free entry."""),
            ("human", "Extract accessibility information from: {content}\n\n{format_instructions}")
        ])
        
        parser = PydanticOutputParser(pydantic_object=AccessibilityInfoPydantic)
        chain = access_prompt | self.llm | parser
        
        try:
            result = chain.invoke({
                "content": content,
                "format_instructions": parser.get_format_instructions()
            })
            return result.dict()
        except Exception as e:
            self.logger.error(f"Accessibility extraction failed: {str(e)}")
            return AccessibilityInfo().__dict__
    
    def calculate_source_credibility(self, source_url: str) -> float:
        """Calculate source credibility based on domain and authority"""
        if not source_url:
            return 0.5
        
        url_lower = source_url.lower()
        
        # Government sources (highest credibility)
        gov_domains = ['state.gov', 'gov.uk', 'embassy', 'consulate', 'dfa.ie', 'dfat.gov.au']
        if any(domain in url_lower for domain in gov_domains):
            return 0.95
        
        # International organizations
        intl_orgs = ['who.int', 'cdc.gov', 'iata.org', 'unwto.org']
        if any(org in url_lower for org in intl_orgs):
            return 0.9
        
        # Established travel platforms
        travel_platforms = ['lonelyplanet', 'tripadvisor', 'fodors', 'frommers']
        if any(platform in url_lower for platform in travel_platforms):
            return 0.85
        
        # News sources
        news_sources = ['bbc.com', 'cnn.com', 'reuters.com', 'associated press']
        if any(news in url_lower for news in news_sources):
            return 0.8
        
        # Travel blogs and community
        community_sources = ['reddit', 'forum', 'blog', 'travel.stack']
        if any(community in url_lower for community in community_sources):
            return 0.7
        
        # Default for unknown sources
        return 0.6
    
    def determine_temporal_relevance(self, content: str, extraction_date: datetime = None) -> float:
        """Determine temporal relevance using semantic analysis"""
        if extraction_date is None:
            extraction_date = datetime.now()
        
        content_lower = content.lower()
        current_year = extraction_date.year
        
        # Look for year mentions
        import re
        years = re.findall(r'\b(20\d{2})\b', content)
        if years:
            latest_year = max(int(year) for year in years)
            year_diff = current_year - latest_year
            
            if year_diff == 0:
                return 1.0  # Current year
            elif year_diff <= 1:
                return 0.9  # Last year
            elif year_diff <= 2:
                return 0.8  # Within 2 years
            elif year_diff <= 5:
                return 0.6  # Within 5 years
            else:
                return 0.3  # Older than 5 years
        
        # Look for recency indicators
        recent_indicators = [
            'recently updated', 'latest', 'current', 'new', 'just published',
            'updated', 'revised', 'fresh', 'newest'
        ]
        if any(indicator in content_lower for indicator in recent_indicators):
            return 0.9
        
        # Look for past indicators
        past_indicators = ['last year', 'previous year', 'formerly', 'used to be']
        if any(indicator in content_lower for indicator in past_indicators):
            return 0.7
        
        # Default temporal relevance
        return 0.75
    
    def _calculate_data_completeness(self, result_dict: Dict[str, Any]) -> float:
        """Calculate how complete the extracted data is"""
        total_fields = 0
        populated_fields = 0
        
        for section in ['safety', 'cost', 'health', 'accessibility']:
            if section in result_dict:
                section_data = result_dict[section]
                for key, value in section_data.items():
                    total_fields += 1
                    if value is not None:
                        if isinstance(value, (list, dict)):
                            if len(value) > 0:
                                populated_fields += 1
                        elif isinstance(value, str):
                            if value.strip():
                                populated_fields += 1
                        else:
                            populated_fields += 1
        
        return populated_fields / total_fields if total_fields > 0 else 0.0
    
    def _calculate_extraction_confidence(self, result_dict: Dict[str, Any], content: str) -> float:
        """Calculate confidence in extraction quality"""
        factors = []
        
        # Content length factor (more content = higher confidence potential)
        content_length = len(content)
        if content_length > 1000:
            factors.append(0.9)
        elif content_length > 500:
            factors.append(0.8)
        elif content_length > 200:
            factors.append(0.7)
        else:
            factors.append(0.6)
        
        # Data completeness factor
        completeness = self._calculate_data_completeness(result_dict)
        factors.append(completeness)
        
        # Source credibility factor
        credibility = result_dict.get('source_credibility', 0.6)
        factors.append(credibility)
        
        # Temporal relevance factor  
        temporal = result_dict.get('temporal_relevance', 0.75)
        factors.append(temporal)
        
        # Return weighted average
        return sum(factors) / len(factors)
    
    def _create_empty_result(self, source_url: str = None) -> Dict[str, Any]:
        """Create empty result structure for fallback"""
        return {
            "safety": SafetyMetrics().__dict__,
            "cost": CostIndicators().__dict__,
            "health": HealthRequirements().__dict__,
            "accessibility": AccessibilityInfo().__dict__,
            "source_url": source_url,
            "extraction_timestamp": datetime.now().isoformat(),
            "extraction_method": "semantic_llm_fallback",
            "source_credibility": self.calculate_source_credibility(source_url),
            "temporal_relevance": 0.5,
            "data_completeness": 0.0,
            "extraction_confidence": 0.3
        }


# Factory function for easy instantiation
def create_priority_extractor(llm: Optional[ChatGoogleGenerativeAI] = None) -> PriorityDataExtractor:
    """Create a priority data extractor instance"""
    return PriorityDataExtractor(llm=llm) 