"""
Priority Export Manager
Handles priority-specific JSON exports and traveler decision scorecards
"""
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional

from .safe_dict_utils import safe_get_nested, safe_get
from src.schemas import PriorityMetrics, DestinationInsight

logger = logging.getLogger(__name__)


class PriorityExportManager:
    """Manages priority-specific exports and scorecards"""
    
    def __init__(self, export_dir: str = "destination_insights/priority_exports"):
        self.export_dir = Path(export_dir)
        self.export_dir.mkdir(parents=True, exist_ok=True)
    
    def export_priority_analysis(
        self,
        destination_name: str,
        priority_metrics: PriorityMetrics,
        priority_insights: List[DestinationInsight],
        validation_results: Optional[Dict] = None
    ) -> str:
        """Export comprehensive priority analysis to JSON"""
        
        timestamp = datetime.now().isoformat()
        
        # Create export structure
        export_data = {
            "destination": destination_name,
            "export_timestamp": timestamp,
            "priority_metrics": self._serialize_metrics(priority_metrics),
            "priority_insights": [self._serialize_insight(insight) for insight in priority_insights],
            "validation": validation_results or {"status": "not_validated"},
            "summary_scores": self._calculate_summary_scores(priority_metrics, priority_insights)
        }
        
        # Save to file
        filename = f"{destination_name.lower().replace(' ', '_')}_priority_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        filepath = self.export_dir / filename
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(export_data, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Priority analysis exported to {filepath}")
        return str(filepath)
    
    def generate_traveler_scorecard(
        self,
        destination_name: str,
        priority_metrics: PriorityMetrics,
        priority_insights: List[DestinationInsight],
        traveler_profile: Optional[Dict] = None
    ) -> Dict[str, Any]:
        """Generate a traveler decision scorecard"""
        
        scorecard = {
            "destination": destination_name,
            "generated_at": datetime.now().isoformat(),
            "overall_score": 0.0,
            "category_scores": {},
            "recommendations": [],
            "warnings": [],
            "highlights": [],
            "decision_factors": {}
        }
        
        # Calculate category scores
        safety_score = self._calculate_safety_score(priority_metrics)
        cost_score = self._calculate_cost_score(priority_metrics, traveler_profile)
        health_score = self._calculate_health_score(priority_metrics)
        accessibility_score = self._calculate_accessibility_score(priority_metrics)
        
        scorecard["category_scores"] = {
            "safety": round(safety_score, 1),
            "cost": round(cost_score, 1),
            "health": round(health_score, 1),
            "accessibility": round(accessibility_score, 1)
        }
        
        # Calculate overall score based on traveler profile
        if traveler_profile:
            weights = traveler_profile.get("priority_weights", {
                "safety": 0.3,
                "cost": 0.3,
                "health": 0.2,
                "accessibility": 0.2
            })
        else:
            weights = {"safety": 0.25, "cost": 0.25, "health": 0.25, "accessibility": 0.25}
        
        scorecard["overall_score"] = round(
            safety_score * weights.get("safety", 0.25) +
            cost_score * weights.get("cost", 0.25) +
            health_score * weights.get("health", 0.25) +
            accessibility_score * weights.get("accessibility", 0.25),
            1
        )
        
        # Generate recommendations and warnings
        scorecard["recommendations"] = self._generate_recommendations(
            priority_metrics, priority_insights, scorecard["category_scores"]
        )
        scorecard["warnings"] = self._generate_warnings(
            priority_metrics, priority_insights, scorecard["category_scores"]
        )
        scorecard["highlights"] = self._generate_highlights(
            priority_metrics, priority_insights, scorecard["category_scores"]
        )
        
        # Decision factors
        scorecard["decision_factors"] = self._analyze_decision_factors(
            priority_metrics, traveler_profile
        )
        
        return scorecard
    
    def export_comparative_analysis(
        self,
        destinations: List[Dict[str, Any]],
        comparison_criteria: Optional[List[str]] = None
    ) -> str:
        """Export comparative analysis between multiple destinations"""
        
        if not comparison_criteria:
            comparison_criteria = ["safety", "cost", "health", "accessibility", "overall"]
        
        comparison = {
            "comparison_timestamp": datetime.now().isoformat(),
            "destinations_compared": len(destinations),
            "criteria": comparison_criteria,
            "comparison_matrix": {},
            "rankings": {},
            "best_for": {}
        }
        
        # Build comparison matrix
        for dest in destinations:
            dest_name = dest["destination"]
            scorecard = dest.get("scorecard", {})
            
            comparison["comparison_matrix"][dest_name] = {
                "overall_score": scorecard.get("overall_score", 0),
                "category_scores": scorecard.get("category_scores", {}),
                "key_metrics": self._extract_key_metrics(dest.get("priority_metrics"))
            }
        
        # Generate rankings
        for criterion in comparison_criteria:
            if criterion == "overall":
                scores = [
                    (d["destination"], safe_get_nested(d, ["scorecard", "overall_score"], 0))
                    for d in destinations
                ]
            else:
                scores = [
                    (d["destination"], safe_get_nested(d, ["scorecard", "category_scores", criterion], 0))
                    for d in destinations
                ]
            
            scores.sort(key=lambda x: x[1], reverse=True)
            comparison["rankings"][criterion] = [
                {"rank": i+1, "destination": dest, "score": score}
                for i, (dest, score) in enumerate(scores)
            ]
        
        # Determine "best for" categories
        comparison["best_for"] = self._determine_best_for(destinations)
        
        # Save comparison
        filename = f"comparison_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        filepath = self.export_dir / filename
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(comparison, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Comparative analysis exported to {filepath}")
        return str(filepath)
    
    def _serialize_metrics(self, metrics: PriorityMetrics) -> Dict[str, Any]:
        """Serialize PriorityMetrics to dict"""
        return {
            "safety": {
                "score": metrics.safety_score,
                "crime_index": metrics.crime_index,
                "tourist_police": metrics.tourist_police_available,
                "emergency_contacts": metrics.emergency_contacts,
                "travel_advisory": metrics.travel_advisory_level
            },
            "cost": {
                "budget_low": metrics.budget_per_day_low,
                "budget_mid": metrics.budget_per_day_mid,
                "budget_high": metrics.budget_per_day_high,
                "currency": metrics.currency,
                "meal_budget": metrics.meal_cost_budget,
                "coffee_price": metrics.coffee_price,
                "transport_ticket": metrics.public_transport_ticket
            },
            "health": {
                "vaccinations": metrics.required_vaccinations,
                "health_risks": metrics.health_risks,
                "water_safety": metrics.water_safety,
                "medical_quality": metrics.medical_facility_quality
            },
            "accessibility": {
                "visa_required": metrics.visa_required,
                "visa_cost": metrics.visa_cost,
                "english_proficiency": metrics.english_proficiency,
                "infrastructure": metrics.infrastructure_rating
            }
        }
    
    def _serialize_insight(self, insight: DestinationInsight) -> Dict[str, Any]:
        """Serialize DestinationInsight to dict"""
        return {
            "name": insight.insight_name,
            "description": insight.description,
            "category": insight.priority_category,
            "impact": insight.priority_impact,
            "confidence": insight.confidence_score,
            "evidence_count": len(insight.evidence) if insight.evidence else 0
        }
    
    def _calculate_summary_scores(
        self,
        metrics: PriorityMetrics,
        insights: List[DestinationInsight]
    ) -> Dict[str, float]:
        """Calculate summary scores for quick reference"""
        
        scores = {
            "safety_score": self._calculate_safety_score(metrics),
            "value_score": self._calculate_value_score(metrics),
            "convenience_score": self._calculate_convenience_score(metrics),
            "health_preparedness_score": self._calculate_health_preparedness_score(metrics)
        }
        
        # Adjust scores based on insights
        high_impact_issues = [i for i in insights if i.priority_impact == "high"]
        if high_impact_issues:
            adjustment = 0.95 ** len(high_impact_issues)
            for key in scores:
                scores[key] *= adjustment
        
        return {k: round(v, 1) for k, v in scores.items()}
    
    def _calculate_safety_score(self, metrics: PriorityMetrics) -> float:
        """Calculate safety score (0-10)"""
        score = 10.0
        
        if metrics.crime_index is not None:
            # Lower crime index = higher score
            score = 10 - (metrics.crime_index / 10)
        
        if metrics.travel_advisory_level:
            # Deduct based on advisory level
            deductions = {1: 0, 2: 2, 3: 4, 4: 8}
            score -= deductions.get(metrics.travel_advisory_level, 0)
        
        if not metrics.tourist_police_available:
            score -= 0.5
        
        return max(0, min(10, score))
    
    def _calculate_cost_score(
        self,
        metrics: PriorityMetrics,
        traveler_profile: Optional[Dict] = None
    ) -> float:
        """Calculate cost/value score (0-10)"""
        
        if not metrics.budget_per_day_low:
            return 5.0  # Default middle score
        
        # Get traveler's budget preference
        if traveler_profile and "daily_budget" in traveler_profile:
            target_budget = traveler_profile["daily_budget"]
        else:
            target_budget = 100  # Default target
        
        # Score based on how well destination matches budget
        if metrics.budget_per_day_low <= target_budget:
            score = 10.0
        else:
            # Deduct points for being over budget
            over_percentage = (metrics.budget_per_day_low - target_budget) / target_budget
            score = max(0, 10 - (over_percentage * 10))
        
        return score
    
    def _calculate_health_score(self, metrics: PriorityMetrics) -> float:
        """Calculate health/medical score (0-10)"""
        score = 10.0
        
        # Deduct for vaccinations
        if metrics.required_vaccinations:
            score -= len(metrics.required_vaccinations) * 0.5
        
        # Deduct for health risks
        if metrics.health_risks:
            score -= len(metrics.health_risks) * 0.75
        
        # Water safety
        if metrics.water_safety:
            if "not safe" in metrics.water_safety.lower():
                score -= 2
            elif "bottled" in metrics.water_safety.lower():
                score -= 1
        
        # Medical facilities
        if metrics.medical_facility_quality:
            quality_scores = {
                "excellent": 0,
                "good": -0.5,
                "adequate": -1,
                "limited": -2,
                "poor": -3
            }
            score += quality_scores.get(metrics.medical_facility_quality.lower(), -1)
        
        return max(0, min(10, score))
    
    def _calculate_accessibility_score(self, metrics: PriorityMetrics) -> float:
        """Calculate accessibility score (0-10)"""
        score = 10.0
        
        # Visa requirements
        if metrics.visa_required:
            score -= 2
            if metrics.visa_cost and metrics.visa_cost > 100:
                score -= 1
        
        # English proficiency
        if metrics.english_proficiency:
            proficiency_scores = {
                "native": 0,
                "excellent": -0.5,
                "good": -1,
                "moderate": -2,
                "basic": -3,
                "none": -4
            }
            score += proficiency_scores.get(metrics.english_proficiency.lower(), -2)
        
        # Infrastructure
        if metrics.infrastructure_rating:
            score = score * 0.5 + (metrics.infrastructure_rating * 2) * 0.5
        
        return max(0, min(10, score))
    
    def _calculate_value_score(self, metrics: PriorityMetrics) -> float:
        """Calculate overall value score"""
        if not metrics.budget_per_day_low:
            return 5.0
        
        # Value = what you get for the price
        base_score = 10.0
        
        # Adjust based on price point
        if metrics.budget_per_day_low < 50:
            base_score = 9.0  # Great value
        elif metrics.budget_per_day_low < 100:
            base_score = 8.0  # Good value
        elif metrics.budget_per_day_low < 200:
            base_score = 7.0  # Moderate value
        else:
            base_score = 6.0  # Premium pricing
        
        # Adjust for infrastructure
        if metrics.infrastructure_rating:
            base_score = base_score * 0.7 + (metrics.infrastructure_rating * 2) * 0.3
        
        return base_score
    
    def _calculate_convenience_score(self, metrics: PriorityMetrics) -> float:
        """Calculate convenience score"""
        score = 10.0
        
        # Visa hassle
        if metrics.visa_required:
            score -= 2
        
        # Language barrier
        if metrics.english_proficiency:
            if metrics.english_proficiency.lower() in ["none", "basic"]:
                score -= 2
            elif metrics.english_proficiency.lower() == "moderate":
                score -= 1
        
        # Infrastructure
        if metrics.infrastructure_rating:
            if metrics.infrastructure_rating < 3:
                score -= 2
            elif metrics.infrastructure_rating < 4:
                score -= 1
        
        return max(0, min(10, score))
    
    def _calculate_health_preparedness_score(self, metrics: PriorityMetrics) -> float:
        """Calculate health preparedness requirements score"""
        score = 10.0
        
        # More requirements = lower score
        if metrics.required_vaccinations:
            score -= len(metrics.required_vaccinations) * 1.0
        
        if metrics.health_risks:
            score -= len(metrics.health_risks) * 0.5
        
        if metrics.water_safety and "not safe" in metrics.water_safety.lower():
            score -= 1
        
        return max(0, min(10, score))
    
    def _generate_recommendations(
        self,
        metrics: PriorityMetrics,
        insights: List[DestinationInsight],
        scores: Dict[str, float]
    ) -> List[str]:
        """Generate recommendations based on analysis"""
        recommendations = []
        
        # Safety recommendations
        if scores.get("safety", 0) < 7:
            recommendations.append("Research specific safety precautions for this destination")
            if metrics.tourist_police_available:
                recommendations.append("Save tourist police contact information before travel")
        
        # Cost recommendations
        if metrics.budget_per_day_low and metrics.budget_per_day_low > 150:
            recommendations.append("Consider traveling during off-season for better prices")
            recommendations.append("Book accommodations in advance for better deals")
        
        # Health recommendations
        if metrics.required_vaccinations:
            recommendations.append(f"Schedule vaccinations at least 4-6 weeks before travel: {', '.join(metrics.required_vaccinations)}")
        
        if metrics.water_safety and "not safe" in metrics.water_safety.lower():
            recommendations.append("Pack water purification tablets or plan to buy bottled water")
        
        # Accessibility recommendations
        if metrics.visa_required:
            recommendations.append("Apply for visa well in advance of travel dates")
        
        if metrics.english_proficiency and metrics.english_proficiency.lower() in ["none", "basic"]:
            recommendations.append("Download offline translation app and learn basic phrases")
        
        return recommendations
    
    def _generate_warnings(
        self,
        metrics: PriorityMetrics,
        insights: List[DestinationInsight],
        scores: Dict[str, float]
    ) -> List[str]:
        """Generate warnings for serious concerns"""
        warnings = []
        
        # Safety warnings
        if metrics.crime_index and metrics.crime_index > 60:
            warnings.append(f"High crime index ({metrics.crime_index}) - exercise increased caution")
        
        if metrics.travel_advisory_level and int(metrics.travel_advisory_level) >= 3:
            warnings.append(f"Level {metrics.travel_advisory_level} travel advisory in effect")
        
        # Health warnings
        serious_risks = ["malaria", "dengue", "yellow fever", "zika"]
        if metrics.health_risks:
            serious = [r for r in metrics.health_risks if any(s in r.lower() for s in serious_risks)]
            if serious:
                warnings.append(f"Serious health risks present: {', '.join(serious)}")
        
        # High impact insights
        high_impact = [i for i in insights if i.priority_impact == "high"]
        for insight in high_impact[:3]:  # Top 3 high impact issues
            warnings.append(f"{insight.insight_name}: {insight.description[:100]}")
        
        return warnings
    
    def _generate_highlights(
        self,
        metrics: PriorityMetrics,
        insights: List[DestinationInsight],
        scores: Dict[str, float]
    ) -> List[str]:
        """Generate positive highlights"""
        highlights = []
        
        # Safety highlights
        if scores.get("safety", 0) >= 8:
            highlights.append("Excellent safety record for tourists")
        
        if metrics.tourist_police_available:
            highlights.append("Dedicated tourist police available")
        
        # Cost highlights
        if metrics.budget_per_day_low and metrics.budget_per_day_low < 50:
            highlights.append("Very affordable destination for budget travelers")
        
        # Health highlights
        if not metrics.required_vaccinations:
            highlights.append("No special vaccinations required")
        
        if metrics.water_safety and "safe" in metrics.water_safety.lower() and "not" not in metrics.water_safety.lower():
            highlights.append("Tap water is safe to drink")
        
        # Accessibility highlights
        if not metrics.visa_required:
            highlights.append("No visa required for most visitors")
        
        if metrics.infrastructure_rating and metrics.infrastructure_rating >= 4:
            highlights.append("Excellent infrastructure and transportation")
        
        return highlights
    
    def _analyze_decision_factors(
        self,
        metrics: PriorityMetrics,
        traveler_profile: Optional[Dict] = None
    ) -> Dict[str, Any]:
        """Analyze key decision factors"""
        
        factors = {
            "best_for": [],
            "not_ideal_for": [],
            "trip_planning_complexity": "low",
            "advance_planning_needed": []
        }
        
        # Determine who it's best for
        if metrics.budget_per_day_low and metrics.budget_per_day_low < 80:
            factors["best_for"].append("budget travelers")
        
        if metrics.infrastructure_rating and metrics.infrastructure_rating >= 4:
            factors["best_for"].append("first-time international travelers")
        
        if metrics.crime_index and metrics.crime_index < 30:
            factors["best_for"].append("solo travelers")
            factors["best_for"].append("families")
        
        # Who it's not ideal for
        if metrics.required_vaccinations and len(metrics.required_vaccinations) > 2:
            factors["not_ideal_for"].append("last-minute travelers")
        
        if metrics.english_proficiency and metrics.english_proficiency.lower() in ["none", "basic"]:
            factors["not_ideal_for"].append("travelers uncomfortable with language barriers")
        
        # Planning complexity
        complexity_score = 0
        if metrics.visa_required:
            complexity_score += 2
        if metrics.required_vaccinations:
            complexity_score += len(metrics.required_vaccinations)
        if metrics.english_proficiency and metrics.english_proficiency.lower() in ["none", "basic"]:
            complexity_score += 1
        
        if complexity_score >= 4:
            factors["trip_planning_complexity"] = "high"
        elif complexity_score >= 2:
            factors["trip_planning_complexity"] = "moderate"
        
        # Advance planning items
        if metrics.visa_required:
            factors["advance_planning_needed"].append("Visa application")
        if metrics.required_vaccinations:
            factors["advance_planning_needed"].append("Vaccination schedule")
        
        return factors
    
    def _extract_key_metrics(self, metrics: Optional[PriorityMetrics]) -> Dict[str, Any]:
        """Extract key metrics for comparison"""
        if not metrics:
            return {}
        
        return {
            "crime_index": metrics.crime_index,
            "daily_budget_low": metrics.budget_per_day_low,
            "visa_required": metrics.visa_required,
            "vaccinations_count": len(metrics.required_vaccinations) if metrics.required_vaccinations else 0,
            "infrastructure_rating": metrics.infrastructure_rating
        }
    
    def _determine_best_for(self, destinations: List[Dict]) -> Dict[str, List[str]]:
        """Determine which destinations are best for different traveler types"""
        
        best_for = {
            "budget_travelers": [],
            "luxury_travelers": [],
            "families": [],
            "solo_travelers": [],
            "adventure_seekers": [],
            "first_timers": []
        }
        
        for dest in destinations:
            name = dest["destination"]
            metrics = dest.get("priority_metrics")
            scorecard = dest.get("scorecard", {})
            
            if not metrics:
                continue
            
            # Budget travelers
            if metrics.budget_per_day_low and metrics.budget_per_day_low < 80:
                best_for["budget_travelers"].append(name)
            
            # Luxury travelers
            if metrics.budget_per_day_high and metrics.budget_per_day_high > 300:
                best_for["luxury_travelers"].append(name)
            
            # Families
            if safe_get_nested(scorecard, ["category_scores", "safety"], 0) >= 8:
                best_for["families"].append(name)
            
            # Solo travelers
            if (safe_get_nested(scorecard, ["category_scores", "safety"], 0) >= 7 and
                metrics.english_proficiency and 
                metrics.english_proficiency.lower() in ["good", "excellent", "native"]):
                best_for["solo_travelers"].append(name)
            
            # First timers
            if (metrics.infrastructure_rating and metrics.infrastructure_rating >= 4 and
                not metrics.visa_required):
                best_for["first_timers"].append(name)
        
        return {k: v for k, v in best_for.items() if v} 