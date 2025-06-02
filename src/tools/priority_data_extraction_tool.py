"""
Priority Data Extraction Tool
Extracts structured data for critical traveler concerns from various sources
"""
import re
import json
import logging
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass
from bs4 import BeautifulSoup

logger = logging.getLogger(__name__)


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


class PriorityDataExtractor:
    """Extracts priority data from text content"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
    def extract_all_priority_data(self, content: str, source_url: str = None) -> Dict[str, Any]:
        """Extract all priority data from content"""
        return {
            "safety": self.extract_safety_metrics(content),
            "cost": self.extract_cost_indicators(content),
            "health": self.extract_health_requirements(content),
            "accessibility": self.extract_accessibility_info(content),
            "source_url": source_url,
            "extraction_timestamp": datetime.now().isoformat()
        }
    
    def extract_safety_metrics(self, content: str) -> Dict[str, Any]:
        """Extract safety-related metrics from content"""
        metrics = SafetyMetrics()
        content_lower = content.lower()
        
        # Crime index/rate patterns
        crime_patterns = [
            r'crime\s+(?:index|rate)[\s:]+(\d+\.?\d*)',
            r'(\d+\.?\d*)\s*(?:%|percent)\s+crime',
            r'crime\s+statistics?[\s:]+(\d+\.?\d*)'
        ]
        
        for pattern in crime_patterns:
            match = re.search(pattern, content_lower)
            if match:
                try:
                    metrics.crime_index = float(match.group(1))
                    break
                except:
                    pass
        
        # Safety rating patterns
        safety_patterns = [
            r'safety\s+(?:rating|score)[\s:]+(\d+\.?\d*)',
            r'rated\s+(\d+\.?\d*)\s+(?:out\s+of\s+\d+\s+)?for\s+safety',
            r'(\d+\.?\d*)/(?:5|10)\s+safety'
        ]
        
        for pattern in safety_patterns:
            match = re.search(pattern, content_lower)
            if match:
                try:
                    metrics.safety_rating = float(match.group(1))
                    break
                except:
                    pass
        
        # Tourist police
        if any(phrase in content_lower for phrase in ['tourist police', 'tourism police', 'police turistico']):
            metrics.tourist_police_available = True
        
        # Emergency contacts
        emergency_patterns = [
            r'emergency[\s:]+(\d{3,})',
            r'police[\s:]+(\d{3,})',
            r'ambulance[\s:]+(\d{3,})',
            r'fire[\s:]+(\d{3,})'
        ]
        
        for pattern in emergency_patterns:
            matches = re.findall(pattern, content_lower)
            for match in matches:
                if 'emergency' in pattern:
                    metrics.emergency_contacts['emergency'] = match
                elif 'police' in pattern:
                    metrics.emergency_contacts['police'] = match
                elif 'ambulance' in pattern:
                    metrics.emergency_contacts['ambulance'] = match
                elif 'fire' in pattern:
                    metrics.emergency_contacts['fire'] = match
        
        # Travel advisory levels
        advisory_patterns = [
            r'level\s+(\d)\s+(?:travel\s+)?advisory',
            r'travel\s+advisory[\s:]+level\s+(\d)',
            r'(?:exercise\s+)?(?:normal|increased|high|extreme)\s+caution'
        ]
        
        for pattern in advisory_patterns:
            match = re.search(pattern, content_lower)
            if match:
                if match.lastindex:
                    metrics.travel_advisory_level = f"Level {match.group(1)}"
                else:
                    if 'normal' in match.group(0):
                        metrics.travel_advisory_level = "Level 1"
                    elif 'increased' in match.group(0):
                        metrics.travel_advisory_level = "Level 2"
                    elif 'high' in match.group(0):
                        metrics.travel_advisory_level = "Level 3"
                    elif 'extreme' in match.group(0):
                        metrics.travel_advisory_level = "Level 4"
                break
        
        # Areas to avoid
        avoid_patterns = [
            r'avoid\s+(?:the\s+)?area[s]?\s+(?:of|around|near)\s+([^.]+)',
            r'dangerous\s+(?:area[s]?|neighborhood[s]?|district[s]?)[\s:]+([^.]+)',
            r'not\s+safe\s+(?:area[s]?|neighborhood[s]?)[\s:]+([^.]+)'
        ]
        
        for pattern in avoid_patterns:
            matches = re.findall(pattern, content_lower)
            for match in matches:
                areas = [area.strip() for area in match.split(',')]
                metrics.areas_to_avoid.extend(areas)
        
        # Safe areas
        safe_patterns = [
            r'safe\s+(?:area[s]?|neighborhood[s]?|district[s]?)[\s:]+([^.]+)',
            r'(?:tourist|touristy)\s+(?:area[s]?|zone[s]?)[\s:]+([^.]+)',
            r'recommended\s+(?:area[s]?|neighborhood[s]?)[\s:]+([^.]+)'
        ]
        
        for pattern in safe_patterns:
            matches = re.findall(pattern, content_lower)
            for match in matches:
                areas = [area.strip() for area in match.split(',')]
                metrics.safe_areas.extend(areas)
        
        return metrics.__dict__
    
    def extract_cost_indicators(self, content: str) -> Dict[str, Any]:
        """Extract cost-related indicators from content"""
        indicators = CostIndicators()
        content_lower = content.lower()
        
        # Daily budget patterns
        budget_patterns = [
            r'budget\s+(?:traveler[s]?|backpacker[s]?)[\s:]+\$?(\d+)[-–]?\$?(\d*)\s*(?:per\s+day|/day)?',
            r'(?:low|cheap|budget)\s+(?:budget|cost)[\s:]+\$?(\d+)\s*(?:per\s+day|/day)?',
            r'mid[\s-]?range[\s:]+\$?(\d+)[-–]?\$?(\d*)\s*(?:per\s+day|/day)?',
            r'(?:luxury|high[\s-]?end)[\s:]+\$?(\d+)[-–]?\$?(\d*)\s*(?:per\s+day|/day)?',
            r'\$?(\d+)[-–]?\$?(\d*)\s*(?:per\s+day|/day)\s+(?:budget|low|cheap)',
            r'\$?(\d+)[-–]?\$?(\d*)\s*(?:per\s+day|/day)\s+(?:mid|medium)',
            r'\$?(\d+)[-–]?\$?(\d*)\s*(?:per\s+day|/day)\s+(?:luxury|high)'
        ]
        
        for pattern in budget_patterns:
            matches = re.findall(pattern, content_lower)
            for match in matches:
                try:
                    amount = float(match[0])
                    if 'budget' in pattern or 'low' in pattern or 'cheap' in pattern:
                        indicators.budget_per_day_low = amount
                    elif 'mid' in pattern or 'medium' in pattern:
                        indicators.budget_per_day_mid = amount
                    elif 'luxury' in pattern or 'high' in pattern:
                        indicators.budget_per_day_high = amount
                except:
                    pass
        
        # Meal costs
        meal_patterns = [
            r'meal[s]?\s+(?:cost[s]?|price[s]?)[\s:]+\$?(\d+\.?\d*)',
            r'(?:lunch|dinner)\s+(?:cost[s]?|price[s]?)[\s:]+\$?(\d+\.?\d*)',
            r'\$?(\d+\.?\d*)\s+(?:for\s+)?(?:a\s+)?meal',
            r'food\s+(?:cost[s]?|budget)[\s:]+\$?(\d+\.?\d*)\s*(?:per\s+day)?'
        ]
        
        meal_costs = []
        for pattern in meal_patterns:
            matches = re.findall(pattern, content_lower)
            for match in matches:
                try:
                    meal_costs.append(float(match))
                except:
                    pass
        
        if meal_costs:
            indicators.meal_cost_average = sum(meal_costs) / len(meal_costs)
        
        # Accommodation costs
        accommodation_patterns = [
            r'(?:hotel[s]?|accommodation[s]?|hostel[s]?)\s+(?:cost[s]?|price[s]?)[\s:]+\$?(\d+\.?\d*)',
            r'\$?(\d+\.?\d*)\s+(?:per\s+)?night\s+(?:hotel|accommodation|hostel)',
            r'(?:budget|cheap)\s+(?:hotel[s]?|accommodation[s]?)[\s:]+\$?(\d+\.?\d*)'
        ]
        
        accommodation_costs = []
        for pattern in accommodation_patterns:
            matches = re.findall(pattern, content_lower)
            for match in matches:
                try:
                    accommodation_costs.append(float(match))
                except:
                    pass
        
        if accommodation_costs:
            indicators.accommodation_cost_average = sum(accommodation_costs) / len(accommodation_costs)
        
        # Currency detection
        currency_patterns = [
            r'currency[\s:]+([A-Z]{3})',
            r'([A-Z]{3})\s+(?:is\s+)?the\s+(?:local\s+)?currency',
            r'prices?\s+(?:are\s+)?in\s+([A-Z]{3})'
        ]
        
        for pattern in currency_patterns:
            match = re.search(pattern, content)
            if match:
                indicators.currency = match.group(1)
                break
        
        # Seasonal variations
        seasonal_patterns = [
            r'(?:high|peak)\s+season[\s:]+(\d+)%?\s+(?:more|higher|increase)',
            r'(?:low|off)\s+season[\s:]+(\d+)%?\s+(?:less|lower|cheaper|decrease)',
            r'prices?\s+(?:increase|rise)\s+(\d+)%?\s+(?:during|in)\s+(?:high|peak)\s+season',
            r'prices?\s+(?:decrease|drop)\s+(\d+)%?\s+(?:during|in)\s+(?:low|off)\s+season'
        ]
        
        for pattern in seasonal_patterns:
            matches = re.findall(pattern, content_lower)
            for match in matches:
                try:
                    percentage = float(match)
                    if 'high' in pattern or 'peak' in pattern:
                        indicators.seasonal_price_variation['high_season'] = percentage
                    else:
                        indicators.seasonal_price_variation['low_season'] = -percentage
                except:
                    pass
        
        return indicators.__dict__
    
    def extract_health_requirements(self, content: str) -> Dict[str, Any]:
        """Extract health-related requirements from content"""
        requirements = HealthRequirements()
        content_lower = content.lower()
        
        # Vaccination patterns
        vaccination_patterns = [
            r'(?:required|mandatory)\s+vaccination[s]?[\s:]+([^.]+)',
            r'vaccination[s]?\s+(?:required|mandatory)[\s:]+([^.]+)',
            r'(?:recommended|suggested)\s+vaccination[s]?[\s:]+([^.]+)',
            r'vaccination[s]?\s+(?:recommended|suggested)[\s:]+([^.]+)'
        ]
        
        for pattern in vaccination_patterns:
            matches = re.findall(pattern, content_lower)
            for match in matches:
                vaccines = [v.strip() for v in match.split(',')]
                if 'required' in pattern or 'mandatory' in pattern:
                    requirements.required_vaccinations.extend(vaccines)
                else:
                    requirements.recommended_vaccinations.extend(vaccines)
        
        # Health risks
        health_risk_patterns = [
            r'health\s+risk[s]?[\s:]+([^.]+)',
            r'(?:malaria|dengue|zika|yellow fever|typhoid)\s+(?:risk|present|found)',
            r'(?:disease[s]?|illness[es]?)\s+(?:to\s+)?(?:watch|be aware|careful)[\s:]+([^.]+)'
        ]
        
        for pattern in health_risk_patterns:
            if 'malaria|dengue' in pattern:
                for disease in ['malaria', 'dengue', 'zika', 'yellow fever', 'typhoid']:
                    if disease in content_lower:
                        requirements.health_risks.append(disease.title())
            else:
                matches = re.findall(pattern, content_lower)
                for match in matches:
                    risks = [r.strip() for r in match.split(',')]
                    requirements.health_risks.extend(risks)
        
        # Water safety
        water_patterns = [
            r'(?:tap\s+)?water\s+(?:is\s+)?(?:safe|unsafe|not safe|drinkable|not drinkable)',
            r'(?:drink|drinking)\s+(?:tap\s+)?water\s+(?:is\s+)?(?:safe|unsafe|not safe)',
            r'(?:bottled|filtered)\s+water\s+(?:recommended|only|advised)'
        ]
        
        for pattern in water_patterns:
            match = re.search(pattern, content_lower)
            if match:
                if any(word in match.group(0) for word in ['safe', 'drinkable']):
                    if 'not' not in match.group(0) and 'unsafe' not in match.group(0):
                        requirements.water_safety = "Safe to drink"
                    else:
                        requirements.water_safety = "Not safe to drink"
                elif 'bottled' in match.group(0) or 'filtered' in match.group(0):
                    requirements.water_safety = "Bottled water recommended"
                break
        
        # Medical facilities
        medical_patterns = [
            r'(?:hospital[s]?|medical facilit(?:y|ies)|healthcare)\s+(?:is\s+)?(?:excellent|good|adequate|poor|limited)',
            r'(?:excellent|good|adequate|poor|limited)\s+(?:hospital[s]?|medical|healthcare)'
        ]
        
        for pattern in medical_patterns:
            match = re.search(pattern, content_lower)
            if match:
                quality_words = ['excellent', 'good', 'adequate', 'poor', 'limited']
                for word in quality_words:
                    if word in match.group(0):
                        requirements.medical_facility_quality = word.capitalize()
                        break
                break
        
        return requirements.__dict__
    
    def extract_accessibility_info(self, content: str) -> Dict[str, Any]:
        """Extract accessibility-related information from content"""
        info = AccessibilityInfo()
        content_lower = content.lower()
        
        # Visa requirements
        visa_patterns = [
            r'visa\s+(?:is\s+)?(?:required|needed|necessary)',
            r'(?:no\s+)?visa\s+(?:required|needed)',
            r'visa[\s-]?(?:on[\s-]?arrival|free)',
            r'(?:require[s]?|need[s]?)\s+(?:a\s+)?visa'
        ]
        
        for pattern in visa_patterns:
            match = re.search(pattern, content_lower)
            if match:
                if 'no visa' in match.group(0):
                    info.visa_required = False
                elif 'on arrival' in match.group(0) or 'on-arrival' in match.group(0):
                    info.visa_on_arrival = True
                    info.visa_required = True
                elif 'free' in match.group(0):
                    info.visa_required = False
                else:
                    info.visa_required = True
                break
        
        # Visa cost
        visa_cost_patterns = [
            r'visa\s+(?:cost[s]?|fee[s]?|price)[\s:]+\$?(\d+)',
            r'\$?(\d+)\s+(?:for\s+)?visa',
            r'visa[\s:]+\$?(\d+)'
        ]
        
        for pattern in visa_cost_patterns:
            match = re.search(pattern, content_lower)
            if match:
                try:
                    info.visa_cost = float(match.group(1))
                    break
                except:
                    pass
        
        # Direct flights
        flight_patterns = [
            r'direct\s+flight[s]?\s+(?:from|to)\s+([^,.]+)',
            r'non[\s-]?stop\s+flight[s]?\s+(?:from|to)\s+([^,.]+)',
            r'([^,.]+)\s+(?:has|have)\s+direct\s+flight[s]?'
        ]
        
        for pattern in flight_patterns:
            matches = re.findall(pattern, content_lower)
            for match in matches:
                cities = [c.strip() for c in match.split('and')]
                info.direct_flights_from_major_hubs.extend(cities)
        
        # English proficiency
        english_patterns = [
            r'english\s+(?:is\s+)?(?:widely|commonly|rarely|not)\s+spoken',
            r'(?:most|many|few|some)\s+(?:people|locals)\s+speak\s+english',
            r'english\s+proficiency\s+(?:is\s+)?(?:high|good|moderate|low|poor)'
        ]
        
        for pattern in english_patterns:
            match = re.search(pattern, content_lower)
            if match:
                if any(word in match.group(0) for word in ['widely', 'commonly', 'most', 'high', 'good']):
                    info.english_proficiency = "High"
                elif any(word in match.group(0) for word in ['some', 'moderate']):
                    info.english_proficiency = "Moderate"
                elif any(word in match.group(0) for word in ['rarely', 'not', 'few', 'low', 'poor']):
                    info.english_proficiency = "Low"
                break
        
        # Infrastructure rating
        infrastructure_patterns = [
            r'infrastructure\s+(?:is\s+)?(?:excellent|good|adequate|poor|limited)',
            r'(?:excellent|good|adequate|poor|limited)\s+infrastructure',
            r'(?:roads?|transport|public transport)\s+(?:is\s+)?(?:excellent|good|adequate|poor|limited)'
        ]
        
        quality_map = {
            'excellent': 5.0,
            'good': 4.0,
            'adequate': 3.0,
            'poor': 2.0,
            'limited': 1.0
        }
        
        for pattern in infrastructure_patterns:
            match = re.search(pattern, content_lower)
            if match:
                for quality, rating in quality_map.items():
                    if quality in match.group(0):
                        info.infrastructure_rating = rating
                        break
                break
        
        return info.__dict__
    
    def calculate_source_credibility(self, source_url: str) -> float:
        """Calculate credibility score based on source"""
        if not source_url:
            return 0.5
        
        # Government and official sources
        if any(domain in source_url.lower() for domain in ['.gov', 'state.', 'embassy', 'consulate']):
            return 0.9
        
        # Major travel platforms
        if any(domain in source_url.lower() for domain in ['tripadvisor', 'lonely planet', 'fodors', 'frommers']):
            return 0.8
        
        # News and media
        if any(domain in source_url.lower() for domain in ['cnn', 'bbc', 'reuters', 'ap news', 'guardian']):
            return 0.75
        
        # Community sources
        if any(domain in source_url.lower() for domain in ['reddit', 'forum', 'facebook', 'twitter']):
            return 0.7
        
        # Default for unknown sources
        return 0.6
    
    def determine_temporal_relevance(self, content: str, extraction_date: datetime = None) -> float:
        """Determine temporal relevance of information"""
        if not extraction_date:
            extraction_date = datetime.now()
        
        # Look for date indicators in content
        current_year = extraction_date.year
        content_lower = content.lower()
        
        # Check for year mentions
        year_pattern = r'20\d{2}'
        years_found = re.findall(year_pattern, content)
        
        if years_found:
            latest_year = max(int(year) for year in years_found)
            years_old = current_year - latest_year
            
            if years_old == 0:
                return 1.0
            elif years_old <= 2:
                return 0.8
            elif years_old <= 5:
                return 0.5
            else:
                return 0.3
        
        # Check for recency indicators
        if any(phrase in content_lower for phrase in ['recently', 'just', 'new', 'latest', 'current']):
            return 0.9
        elif any(phrase in content_lower for phrase in ['last year', 'previous year']):
            return 0.8
        elif any(phrase in content_lower for phrase in ['few years', 'several years']):
            return 0.6
        
        # Default for undated content
        return 0.7 