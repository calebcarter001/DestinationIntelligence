"""
Priority Aggregation Tool
Aggregates priority data from multiple sources into unified metrics with confidence scoring
"""
import logging
from typing import List, Dict, Any, Optional, Type
from statistics import median, mean
from collections import defaultdict

from langchain.tools import StructuredTool
from pydantic import BaseModel, Field

from src.schemas import PageContent, PriorityMetrics, DestinationInsight
from .priority_data_extraction_tool import PriorityDataExtractor

logger = logging.getLogger(__name__)


class PriorityAggregationInput(BaseModel):
    """Input for priority aggregation tool"""
    destination_name: str = Field(description="Name of the destination being analyzed")
    page_contents: List[PageContent] = Field(description="List of page contents with priority data")
    confidence_threshold: float = Field(0.6, description="Minimum confidence threshold for including data")


class PriorityAggregationTool(StructuredTool):
    """Tool for aggregating priority data from multiple sources"""
    
    name: str = "aggregate_priority_data"
    description: str = (
        "Aggregates priority data (safety, cost, health, accessibility) from multiple sources "
        "into unified metrics with confidence scoring based on source credibility and consensus."
    )
    args_schema: Type[BaseModel] = PriorityAggregationInput
    
    extractor: PriorityDataExtractor = None
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.extractor = PriorityDataExtractor()
    
    async def _arun(
        self, 
        destination_name: str,
        page_contents: List[PageContent],
        confidence_threshold: float = 0.6
    ) -> Dict[str, Any]:
        """Run the priority aggregation asynchronously"""
        return self._run(destination_name, page_contents, confidence_threshold)
    
    def _run(
        self,
        destination_name: str,
        page_contents: List[PageContent],
        confidence_threshold: float = 0.6
    ) -> Dict[str, Any]:
        """Aggregate priority data from multiple sources"""
        
        logger.info(f"Aggregating priority data for {destination_name} from {len(page_contents)} sources")
        
        # Collect all priority data
        all_priority_data = []
        
        for page_content in page_contents:
            if hasattr(page_content, 'priority_data') and page_content.priority_data:
                # Add source credibility score
                credibility = self.extractor.calculate_source_credibility(page_content.url)
                temporal_relevance = self.extractor.determine_temporal_relevance(page_content.content)
                
                priority_data_with_scores = {
                    **page_content.priority_data,
                    "source_credibility": credibility,
                    "temporal_relevance": temporal_relevance,
                    "source_url": page_content.url
                }
                all_priority_data.append(priority_data_with_scores)
        
        # Aggregate by category
        aggregated_metrics = PriorityMetrics()
        priority_insights = []
        
        # Safety aggregation
        safety_data = self._aggregate_safety_data(all_priority_data, confidence_threshold)
        if safety_data:
            aggregated_metrics.safety_score = safety_data.get("safety_score")
            aggregated_metrics.crime_index = safety_data.get("crime_index")
            aggregated_metrics.tourist_police_available = safety_data.get("tourist_police_available")
            aggregated_metrics.emergency_contacts = safety_data.get("emergency_contacts", {})
            aggregated_metrics.travel_advisory_level = safety_data.get("travel_advisory_level")
            
            # Create safety insights
            if safety_data.get("areas_to_avoid"):
                priority_insights.append(DestinationInsight(
                    destination_name=destination_name,
                    insight_type="Priority Concern",
                    insight_name="Areas to Avoid",
                    description=f"Areas reported as potentially unsafe: {', '.join(safety_data['areas_to_avoid'])}",
                    evidence=safety_data.get("evidence", []),
                    confidence_score=safety_data.get("confidence", 0.7),
                    priority_category="safety",
                    priority_impact="high",
                    temporal_relevance=safety_data.get("temporal_relevance", 0.7),
                    source_urls=safety_data.get("source_urls", []),
                    tags=["safety", "areas-to-avoid", "warning", "priority"]
                ))
        
        # Cost aggregation
        cost_data = self._aggregate_cost_data(all_priority_data, confidence_threshold)
        if cost_data:
            aggregated_metrics.budget_per_day_low = cost_data.get("budget_per_day_low")
            aggregated_metrics.budget_per_day_mid = cost_data.get("budget_per_day_mid")
            aggregated_metrics.budget_per_day_high = cost_data.get("budget_per_day_high")
            aggregated_metrics.currency = cost_data.get("currency")
            
            # Create cost insights
            if cost_data.get("seasonal_variations"):
                priority_insights.append(DestinationInsight(
                    destination_name=destination_name,
                    insight_type="Priority Concern",
                    insight_name="Seasonal Price Variations",
                    description=f"Prices vary significantly by season: {cost_data['seasonal_variations']}",
                    evidence=cost_data.get("evidence", []),
                    confidence_score=cost_data.get("confidence", 0.7),
                    priority_category="cost",
                    priority_impact="medium",
                    temporal_relevance=cost_data.get("temporal_relevance", 0.7),
                    source_urls=cost_data.get("source_urls", []),
                    tags=["cost", "seasonal", "budget", "pricing", "priority"]
                ))
        
        # Health aggregation
        health_data = self._aggregate_health_data(all_priority_data, confidence_threshold)
        if health_data:
            aggregated_metrics.required_vaccinations = health_data.get("required_vaccinations", [])
            aggregated_metrics.health_risks = health_data.get("health_risks", [])
            aggregated_metrics.water_safety = health_data.get("water_safety")
            aggregated_metrics.medical_facility_quality = health_data.get("medical_facility_quality")
            
            # Create health insights
            if health_data.get("required_vaccinations"):
                priority_insights.append(DestinationInsight(
                    destination_name=destination_name,
                    insight_type="Priority Concern",
                    insight_name="Required Vaccinations",
                    description=f"Vaccinations required: {', '.join(health_data['required_vaccinations'])}",
                    evidence=health_data.get("evidence", []),
                    confidence_score=health_data.get("confidence", 0.9),
                    priority_category="health",
                    priority_impact="high",
                    temporal_relevance=health_data.get("temporal_relevance", 0.8),
                    source_urls=health_data.get("source_urls", []),
                    tags=["health", "vaccinations", "required", "medical", "priority"]
                ))
        
        # Accessibility aggregation
        accessibility_data = self._aggregate_accessibility_data(all_priority_data, confidence_threshold)
        if accessibility_data:
            aggregated_metrics.visa_required = accessibility_data.get("visa_required")
            aggregated_metrics.visa_on_arrival = accessibility_data.get("visa_on_arrival")
            aggregated_metrics.visa_cost = accessibility_data.get("visa_cost")
            aggregated_metrics.english_proficiency = accessibility_data.get("english_proficiency")
            aggregated_metrics.infrastructure_rating = accessibility_data.get("infrastructure_rating")
            
            # Create accessibility insights
            if accessibility_data.get("visa_required"):
                priority_insights.append(DestinationInsight(
                    destination_name=destination_name,
                    insight_type="Priority Concern",
                    insight_name="Visa Requirements",
                    description=f"Visa required with cost: ${accessibility_data.get('visa_cost', 'Unknown')}",
                    evidence=accessibility_data.get("evidence", []),
                    confidence_score=accessibility_data.get("confidence", 0.8),
                    priority_category="accessibility",
                    priority_impact="high",
                    temporal_relevance=accessibility_data.get("temporal_relevance", 0.8),
                    source_urls=accessibility_data.get("source_urls", []),
                    tags=["visa", "requirements", "travel-documents", "accessibility", "priority"]
                ))
        
        return {
            "priority_metrics": aggregated_metrics,
            "priority_insights": priority_insights,
            "total_sources": len(all_priority_data),
            "aggregation_confidence": self._calculate_overall_confidence(all_priority_data)
        }
    
    def _aggregate_safety_data(self, all_data: List[Dict], threshold: float) -> Dict[str, Any]:
        """Aggregate safety-related data"""
        safety_entries = []
        
        for data in all_data:
            if "safety" in data and data["source_credibility"] * data["temporal_relevance"] >= threshold:
                safety_entries.append(data)
        
        if not safety_entries:
            return {}
        
        # Aggregate crime indices
        crime_indices = [
            entry["safety"]["crime_index"] 
            for entry in safety_entries 
            if entry.get("safety", {}).get("crime_index") is not None
        ]
        
        # Aggregate safety ratings
        safety_ratings = [
            entry["safety"]["safety_rating"] 
            for entry in safety_entries 
            if entry.get("safety", {}).get("safety_rating") is not None
        ]
        
        # Aggregate tourist police availability (majority vote)
        tourist_police_votes = [
            entry["safety"]["tourist_police_available"] 
            for entry in safety_entries 
            if entry.get("safety", {}).get("tourist_police_available") is not None
        ]
        
        # Collect all emergency contacts
        emergency_contacts = {}
        for entry in safety_entries:
            if entry.get("safety", {}).get("emergency_contacts"):
                emergency_contacts.update(entry["safety"]["emergency_contacts"])
        
        # Aggregate areas to avoid
        areas_to_avoid = set()
        for entry in safety_entries:
            if entry.get("safety", {}).get("areas_to_avoid"):
                areas_to_avoid.update(entry["safety"]["areas_to_avoid"])
        
        # Determine travel advisory level (most conservative)
        advisory_levels = [
            entry["safety"]["travel_advisory_level"] 
            for entry in safety_entries 
            if entry.get("safety", {}).get("travel_advisory_level")
        ]
        
        result = {
            "confidence": self._calculate_category_confidence(safety_entries),
            "temporal_relevance": mean([e["temporal_relevance"] for e in safety_entries]),
            "source_urls": list(set(e["source_url"] for e in safety_entries)),
            "evidence": []
        }
        
        if crime_indices:
            result["crime_index"] = median(crime_indices)
            result["evidence"].append(f"Crime index: {result['crime_index']:.1f} (from {len(crime_indices)} sources)")
        
        if safety_ratings:
            result["safety_score"] = mean(safety_ratings)
            result["evidence"].append(f"Safety rating: {result['safety_score']:.1f}/10")
        
        if tourist_police_votes:
            result["tourist_police_available"] = sum(tourist_police_votes) > len(tourist_police_votes) / 2
        
        if emergency_contacts:
            result["emergency_contacts"] = emergency_contacts
        
        if areas_to_avoid:
            result["areas_to_avoid"] = list(areas_to_avoid)
        
        if advisory_levels:
            # Take the highest (most conservative) advisory level
            result["travel_advisory_level"] = max(advisory_levels)
        
        return result
    
    def _aggregate_cost_data(self, all_data: List[Dict], threshold: float) -> Dict[str, Any]:
        """Aggregate cost-related data"""
        cost_entries = []
        
        for data in all_data:
            if "cost" in data and data["source_credibility"] * data["temporal_relevance"] >= threshold:
                cost_entries.append(data)
        
        if not cost_entries:
            return {}
        
        # Aggregate budget estimates
        budget_low = [
            entry["cost"]["budget_per_day_low"] 
            for entry in cost_entries 
            if entry.get("cost", {}).get("budget_per_day_low") is not None
        ]
        
        budget_mid = [
            entry["cost"]["budget_per_day_mid"] 
            for entry in cost_entries 
            if entry.get("cost", {}).get("budget_per_day_mid") is not None
        ]
        
        budget_high = [
            entry["cost"]["budget_per_day_high"] 
            for entry in cost_entries 
            if entry.get("cost", {}).get("budget_per_day_high") is not None
        ]
        
        # Get currency (most common)
        currencies = [
            entry["cost"]["currency"] 
            for entry in cost_entries 
            if entry.get("cost", {}).get("currency")
        ]
        
        result = {
            "confidence": self._calculate_category_confidence(cost_entries),
            "temporal_relevance": mean([e["temporal_relevance"] for e in cost_entries]),
            "source_urls": list(set(e["source_url"] for e in cost_entries)),
            "evidence": []
        }
        
        if budget_low:
            result["budget_per_day_low"] = median(budget_low)
            result["evidence"].append(f"Budget travel: ${result['budget_per_day_low']:.0f}/day")
        
        if budget_mid:
            result["budget_per_day_mid"] = median(budget_mid)
            result["evidence"].append(f"Mid-range travel: ${result['budget_per_day_mid']:.0f}/day")
        
        if budget_high:
            result["budget_per_day_high"] = median(budget_high)
            result["evidence"].append(f"Luxury travel: ${result['budget_per_day_high']:.0f}/day")
        
        if currencies:
            result["currency"] = max(set(currencies), key=currencies.count)
        
        # Check for seasonal variations
        seasonal_variations = []
        for entry in cost_entries:
            if entry.get("cost", {}).get("seasonal_price_variation"):
                seasonal_variations.append(entry["cost"]["seasonal_price_variation"])
        
        if seasonal_variations:
            # Aggregate seasonal variation info
            high_season_increases = [
                var.get("high_season", 0) 
                for var in seasonal_variations 
                if "high_season" in var
            ]
            if high_season_increases:
                avg_increase = mean(high_season_increases)
                result["seasonal_variations"] = f"High season prices increase by ~{avg_increase:.0f}%"
        
        return result
    
    def _aggregate_health_data(self, all_data: List[Dict], threshold: float) -> Dict[str, Any]:
        """Aggregate health-related data"""
        health_entries = []
        
        for data in all_data:
            if "health" in data and data["source_credibility"] * data["temporal_relevance"] >= threshold:
                health_entries.append(data)
        
        if not health_entries:
            return {}
        
        # Collect all vaccinations
        required_vaccinations = set()
        recommended_vaccinations = set()
        health_risks = set()
        
        for entry in health_entries:
            if entry.get("health", {}).get("required_vaccinations"):
                required_vaccinations.update(entry["health"]["required_vaccinations"])
            if entry.get("health", {}).get("recommended_vaccinations"):
                recommended_vaccinations.update(entry["health"]["recommended_vaccinations"])
            if entry.get("health", {}).get("health_risks"):
                health_risks.update(entry["health"]["health_risks"])
        
        # Water safety (most conservative)
        water_safety_ratings = [
            entry["health"]["water_safety"] 
            for entry in health_entries 
            if entry.get("health", {}).get("water_safety")
        ]
        
        # Medical facility quality (mode)
        medical_quality_ratings = [
            entry["health"]["medical_facility_quality"] 
            for entry in health_entries 
            if entry.get("health", {}).get("medical_facility_quality")
        ]
        
        result = {
            "confidence": self._calculate_category_confidence(health_entries),
            "temporal_relevance": mean([e["temporal_relevance"] for e in health_entries]),
            "source_urls": list(set(e["source_url"] for e in health_entries)),
            "evidence": []
        }
        
        if required_vaccinations:
            result["required_vaccinations"] = sorted(list(required_vaccinations))
            result["evidence"].append(f"Required vaccinations: {', '.join(result['required_vaccinations'])}")
        
        if health_risks:
            result["health_risks"] = sorted(list(health_risks))
            result["evidence"].append(f"Health risks: {', '.join(result['health_risks'])}")
        
        if water_safety_ratings:
            # Take most conservative rating
            if any("not safe" in rating.lower() for rating in water_safety_ratings):
                result["water_safety"] = "Not safe to drink"
            elif any("bottled" in rating.lower() for rating in water_safety_ratings):
                result["water_safety"] = "Bottled water recommended"
            else:
                result["water_safety"] = "Safe to drink"
        
        if medical_quality_ratings:
            result["medical_facility_quality"] = max(set(medical_quality_ratings), key=medical_quality_ratings.count)
        
        return result
    
    def _aggregate_accessibility_data(self, all_data: List[Dict], threshold: float) -> Dict[str, Any]:
        """Aggregate accessibility-related data"""
        accessibility_entries = []
        
        for data in all_data:
            if "accessibility" in data and data["source_credibility"] * data["temporal_relevance"] >= threshold:
                accessibility_entries.append(data)
        
        if not accessibility_entries:
            return {}
        
        # Visa requirements (majority vote)
        visa_required_votes = [
            entry["accessibility"]["visa_required"] 
            for entry in accessibility_entries 
            if entry.get("accessibility", {}).get("visa_required") is not None
        ]
        
        # Visa costs
        visa_costs = [
            entry["accessibility"]["visa_cost"] 
            for entry in accessibility_entries 
            if entry.get("accessibility", {}).get("visa_cost") is not None
        ]
        
        # Infrastructure ratings
        infrastructure_ratings = [
            entry["accessibility"]["infrastructure_rating"] 
            for entry in accessibility_entries 
            if entry.get("accessibility", {}).get("infrastructure_rating") is not None
        ]
        
        # English proficiency (mode)
        english_proficiency_ratings = [
            entry["accessibility"]["english_proficiency"] 
            for entry in accessibility_entries 
            if entry.get("accessibility", {}).get("english_proficiency")
        ]
        
        result = {
            "confidence": self._calculate_category_confidence(accessibility_entries),
            "temporal_relevance": mean([e["temporal_relevance"] for e in accessibility_entries]),
            "source_urls": list(set(e["source_url"] for e in accessibility_entries)),
            "evidence": []
        }
        
        if visa_required_votes:
            result["visa_required"] = sum(visa_required_votes) > len(visa_required_votes) / 2
            result["evidence"].append(f"Visa required: {'Yes' if result['visa_required'] else 'No'}")
        
        if visa_costs and result.get("visa_required"):
            result["visa_cost"] = median(visa_costs)
            result["evidence"].append(f"Visa cost: ${result['visa_cost']:.0f}")
        
        if infrastructure_ratings:
            result["infrastructure_rating"] = mean(infrastructure_ratings)
            result["evidence"].append(f"Infrastructure rating: {result['infrastructure_rating']:.1f}/5")
        
        if english_proficiency_ratings:
            result["english_proficiency"] = max(set(english_proficiency_ratings), key=english_proficiency_ratings.count)
            result["evidence"].append(f"English proficiency: {result['english_proficiency']}")
        
        return result
    
    def _calculate_category_confidence(self, entries: List[Dict]) -> float:
        """Calculate confidence score for a category based on source credibility and agreement"""
        if not entries:
            return 0.0
        
        # Factor 1: Average source credibility
        avg_credibility = mean([e["source_credibility"] for e in entries])
        
        # Factor 2: Temporal relevance
        avg_temporal = mean([e["temporal_relevance"] for e in entries])
        
        # Factor 3: Number of sources (more sources = higher confidence, up to a point)
        source_factor = min(len(entries) / 5.0, 1.0)  # Max confidence at 5+ sources
        
        # Combine factors
        confidence = (avg_credibility * 0.4 + avg_temporal * 0.3 + source_factor * 0.3)
        
        return round(confidence, 2)
    
    def _calculate_overall_confidence(self, all_data: List[Dict]) -> float:
        """Calculate overall aggregation confidence"""
        if not all_data:
            return 0.0
        
        credibilities = [d["source_credibility"] for d in all_data]
        temporal_relevances = [d["temporal_relevance"] for d in all_data]
        
        return round(
            mean(credibilities) * 0.5 + mean(temporal_relevances) * 0.5,
            2
        ) 