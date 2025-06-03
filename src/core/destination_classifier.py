from typing import Dict, Any
import logging
from enum import Enum

logger = logging.getLogger(__name__)

class DestinationType(Enum):
    GLOBAL_HUB = "global_hub"        # e.g., Paris, Tokyo, NYC
    REGIONAL = "regional"           # e.g., Bend, Oregon; Kyoto, Japan
    BUSINESS_HUB = "business_hub"   # e.g., Gurgaon, India; Frankfurt, Germany
    REMOTE_GETAWAY = "remote_getaway" # e.g., Patagonia, Bhutan

class SourceStrategy(Enum):
    FILTER_QUALITY_FROM_ABUNDANCE = "filter_quality_from_abundance" # For global hubs
    COMPREHENSIVE_LOCAL_SELECTIVE_NATIONAL = "comprehensive_local_selective_national" # For regional
    BUSINESS_FOCUSED_PRACTICAL = "business_focused_practical" # For business hubs
    ULTRA_LOCAL_NICHE_EXPERT = "ultra_local_niche_expert" # For remote getaways

class DestinationClassifier:
    """Classify destinations by type and determine appropriate source strategies"""
    
    def __init__(self):
        self.logger = logging.getLogger(self.__class__.__name__)
    
    def classify_destination_type(self, destination_data: Dict[str, Any]) -> DestinationType:
        """Classify destination type based on characteristics"""
        
        # Extract key data with safe defaults
        names = destination_data.get("names", [])
        admin_levels = destination_data.get("admin_levels", {})
        population = destination_data.get("population", 0)
        
        # Get primary name
        primary_name = names[0] if names else ""
        location_lower = primary_name.lower()
        
        # Country information
        country = admin_levels.get("country", "").lower()
        
        self.logger.debug(f"Classifying destination: {primary_name}, population: {population}, country: {country}")
        
        # Global hubs - major international cities
        global_hubs = [
            "paris", "london", "tokyo", "new york", "nyc", "new york city",
            "los angeles", "beijing", "shanghai", "dubai", "singapore",
            "rome", "barcelona", "amsterdam", "sydney", "bangkok",
            "istanbul", "mumbai", "delhi", "moscow", "berlin"
        ]
        
        if any(hub in location_lower for hub in global_hubs) or population > 5000000:
            return DestinationType.GLOBAL_HUB
        
        # Business hubs - smaller but business-focused
        business_hubs = [
            "frankfurt", "zurich", "geneva", "luxembourg", "hong kong",
            "gurgaon", "bangalore", "pune", "charlotte", "dallas"
        ]
        
        if any(hub in location_lower for hub in business_hubs) or (
            500000 < population <= 2000000 and any(keyword in location_lower for keyword in ["business", "financial", "tech"])
        ):
            return DestinationType.BUSINESS_HUB
        
        # Remote getaways - very small population or remote areas
        remote_indicators = [
            "island", "mountains", "wilderness", "rural", "remote",
            "patagonia", "bhutan", "faroe", "tibet", "antarctica"
        ]
        
        if population < 100000 or any(indicator in location_lower for indicator in remote_indicators):
            return DestinationType.REMOTE_GETAWAY
        
        # Default to regional for everything else
        return DestinationType.REGIONAL
    
    def get_scoring_weights(self, destination_type: DestinationType) -> Dict[str, float]:
        """Get multi-dimensional scoring weights based on destination type"""
        
        weights_config = {
            DestinationType.GLOBAL_HUB: {
                "authenticity": 0.4,      # Higher weight - filter tourist traps
                "uniqueness": 0.2,        # Lower - many options available
                "actionability": 0.3,     # High - visitors need practical info
                "temporal_relevance": 0.1 # Lower - less seasonal
            },
            DestinationType.REGIONAL: {
                "authenticity": 0.3,      # Important but balanced
                "uniqueness": 0.35,       # Higher - showcase what makes it special
                "actionability": 0.25,    # Important for planning
                "temporal_relevance": 0.1 # Moderate seasonal considerations
            },
            DestinationType.BUSINESS_HUB: {
                "authenticity": 0.2,      # Lower priority
                "uniqueness": 0.2,        # Lower priority
                "actionability": 0.5,     # Highest - business travelers need efficiency
                "temporal_relevance": 0.1 # Lowest - business travel less seasonal
            },
            DestinationType.REMOTE_GETAWAY: {
                "authenticity": 0.35,     # Very important - avoid commercialization
                "uniqueness": 0.4,        # Highest - main draw is uniqueness
                "actionability": 0.15,    # Lower - adventure/exploration mindset
                "temporal_relevance": 0.1 # Higher - weather/access considerations
            }
        }
        
        return weights_config.get(destination_type, weights_config[DestinationType.REGIONAL])
    
    def get_source_strategy(self, destination_type: DestinationType) -> SourceStrategy:
        """Determine appropriate source discovery strategy"""
        
        strategy_mapping = {
            DestinationType.GLOBAL_HUB: SourceStrategy.FILTER_QUALITY_FROM_ABUNDANCE,
            DestinationType.REGIONAL: SourceStrategy.COMPREHENSIVE_LOCAL_SELECTIVE_NATIONAL,
            DestinationType.BUSINESS_HUB: SourceStrategy.BUSINESS_FOCUSED_PRACTICAL,
            DestinationType.REMOTE_GETAWAY: SourceStrategy.ULTRA_LOCAL_NICHE_EXPERT
        }
        
        return strategy_mapping.get(destination_type, SourceStrategy.COMPREHENSIVE_LOCAL_SELECTIVE_NATIONAL) 