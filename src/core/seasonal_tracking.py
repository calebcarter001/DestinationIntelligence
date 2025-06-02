"""
Seasonal Tracking System
Tracks and analyzes seasonal variations in destination data
"""
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
from statistics import mean, stdev
import json

logger = logging.getLogger(__name__)


class SeasonalTracker:
    """Tracks seasonal variations in destination metrics"""
    
    # Define seasons by month ranges
    SEASONS = {
        "winter": [12, 1, 2],
        "spring": [3, 4, 5],
        "summer": [6, 7, 8],
        "fall": [9, 10, 11]
    }
    
    # Reverse mapping for quick lookup
    MONTH_TO_SEASON = {}
    for season, months in SEASONS.items():
        for month in months:
            MONTH_TO_SEASON[month] = season
    
    def __init__(self):
        self.historical_data = {}
    
    def track_metrics(
        self,
        destination: str,
        metrics: Dict[str, Any],
        timestamp: datetime
    ):
        """Track metrics for a destination at a specific time"""
        
        season = self.MONTH_TO_SEASON[timestamp.month]
        year = timestamp.year
        
        if destination not in self.historical_data:
            self.historical_data[destination] = {}
        
        if year not in self.historical_data[destination]:
            self.historical_data[destination][year] = {}
        
        if season not in self.historical_data[destination][year]:
            self.historical_data[destination][year][season] = []
        
        # Store metrics with timestamp
        self.historical_data[destination][year][season].append({
            "timestamp": timestamp.isoformat(),
            "metrics": metrics
        })
        
        logger.info(f"Tracked metrics for {destination} in {season} {year}")
    
    def analyze_seasonal_patterns(
        self,
        destination: str,
        metric_name: str,
        years_back: int = 2
    ) -> Dict[str, Any]:
        """Analyze seasonal patterns for a specific metric"""
        
        if destination not in self.historical_data:
            return {"error": f"No data for {destination}"}
        
        current_year = datetime.now().year
        start_year = current_year - years_back
        
        seasonal_data = {
            "winter": [],
            "spring": [],
            "summer": [],
            "fall": []
        }
        
        # Collect data across years
        for year in range(start_year, current_year + 1):
            if year in self.historical_data[destination]:
                for season, entries in self.historical_data[destination][year].items():
                    for entry in entries:
                        if metric_name in entry["metrics"]:
                            seasonal_data[season].append(entry["metrics"][metric_name])
        
        # Calculate statistics
        analysis = {
            "destination": destination,
            "metric": metric_name,
            "seasonal_averages": {},
            "seasonal_variations": {},
            "peak_season": None,
            "low_season": None,
            "variation_percentage": 0
        }
        
        # Calculate averages and variations
        valid_seasons = {}
        for season, values in seasonal_data.items():
            if values:
                avg = mean(values)
                analysis["seasonal_averages"][season] = round(avg, 2)
                if len(values) > 1:
                    analysis["seasonal_variations"][season] = round(stdev(values), 2)
                else:
                    analysis["seasonal_variations"][season] = 0
                valid_seasons[season] = avg
        
        # Find peak and low seasons
        if valid_seasons:
            analysis["peak_season"] = max(valid_seasons, key=valid_seasons.get)
            analysis["low_season"] = min(valid_seasons, key=valid_seasons.get)
            
            # Calculate variation percentage
            peak_val = valid_seasons[analysis["peak_season"]]
            low_val = valid_seasons[analysis["low_season"]]
            if low_val > 0:
                analysis["variation_percentage"] = round(
                    ((peak_val - low_val) / low_val) * 100, 1
                )
        
        return analysis
    
    def predict_seasonal_value(
        self,
        destination: str,
        metric_name: str,
        target_season: str
    ) -> Dict[str, Any]:
        """Predict value for a metric in a specific season"""
        
        analysis = self.analyze_seasonal_patterns(destination, metric_name)
        
        if "error" in analysis:
            return analysis
        
        if target_season not in analysis["seasonal_averages"]:
            return {"error": f"No data for {target_season}"}
        
        prediction = {
            "destination": destination,
            "metric": metric_name,
            "season": target_season,
            "predicted_value": analysis["seasonal_averages"][target_season],
            "confidence": self._calculate_prediction_confidence(
                analysis["seasonal_variations"].get(target_season, 0),
                analysis["seasonal_averages"][target_season]
            ),
            "range": self._calculate_prediction_range(
                analysis["seasonal_averages"][target_season],
                analysis["seasonal_variations"].get(target_season, 0)
            )
        }
        
        return prediction
    
    def get_best_season_for_metric(
        self,
        destination: str,
        metric_name: str,
        optimization: str = "minimize"
    ) -> Dict[str, Any]:
        """Get the best season for a specific metric"""
        
        analysis = self.analyze_seasonal_patterns(destination, metric_name)
        
        if "error" in analysis:
            return analysis
        
        if optimization == "minimize":
            best_season = analysis["low_season"]
        else:
            best_season = analysis["peak_season"]
        
        return {
            "destination": destination,
            "metric": metric_name,
            "best_season": best_season,
            "value": analysis["seasonal_averages"].get(best_season),
            "compared_to_worst": analysis["variation_percentage"],
            "recommendation": self._generate_seasonal_recommendation(
                metric_name, best_season, optimization
            )
        }
    
    def generate_seasonal_report(
        self,
        destination: str,
        metrics: List[str]
    ) -> Dict[str, Any]:
        """Generate comprehensive seasonal report for multiple metrics"""
        
        report = {
            "destination": destination,
            "generated_at": datetime.now().isoformat(),
            "metric_analyses": {},
            "overall_recommendations": [],
            "best_visit_times": {}
        }
        
        # Analyze each metric
        for metric in metrics:
            analysis = self.analyze_seasonal_patterns(destination, metric)
            if "error" not in analysis:
                report["metric_analyses"][metric] = analysis
        
        # Generate overall recommendations
        report["overall_recommendations"] = self._generate_overall_recommendations(
            report["metric_analyses"]
        )
        
        # Determine best visit times for different traveler types
        report["best_visit_times"] = self._determine_best_visit_times(
            report["metric_analyses"]
        )
        
        return report
    
    def _calculate_prediction_confidence(
        self,
        variation: float,
        mean_value: float
    ) -> float:
        """Calculate confidence in prediction based on historical variation"""
        
        if mean_value == 0:
            return 0.5
        
        # Coefficient of variation
        cv = variation / mean_value if mean_value else 0
        
        # Higher variation = lower confidence
        if cv < 0.1:
            return 0.9
        elif cv < 0.2:
            return 0.8
        elif cv < 0.3:
            return 0.7
        elif cv < 0.5:
            return 0.6
        else:
            return 0.5
    
    def _calculate_prediction_range(
        self,
        mean_value: float,
        std_dev: float
    ) -> Tuple[float, float]:
        """Calculate prediction range (95% confidence interval)"""
        
        # Approximately 95% of values fall within 2 standard deviations
        lower = max(0, mean_value - 2 * std_dev)
        upper = mean_value + 2 * std_dev
        
        return (round(lower, 2), round(upper, 2))
    
    def _generate_seasonal_recommendation(
        self,
        metric_name: str,
        best_season: str,
        optimization: str
    ) -> str:
        """Generate recommendation text for seasonal timing"""
        
        recommendations = {
            "budget_per_day_low": {
                "minimize": f"Visit in {best_season} for the best budget rates",
                "maximize": f"Premium experiences are available in {best_season}"
            },
            "crime_index": {
                "minimize": f"Safety is highest in {best_season}",
                "maximize": f"Avoid {best_season} if safety is a concern"
            },
            "temperature": {
                "minimize": f"Visit in {best_season} for cooler weather",
                "maximize": f"Visit in {best_season} for warmer weather"
            },
            "rainfall": {
                "minimize": f"Visit in {best_season} for the driest weather",
                "maximize": f"Visit in {best_season} if you enjoy rain"
            },
            "tourist_volume": {
                "minimize": f"Visit in {best_season} to avoid crowds",
                "maximize": f"Visit in {best_season} for peak atmosphere"
            }
        }
        
        if metric_name in recommendations:
            return recommendations[metric_name][optimization]
        else:
            return f"Best time for {metric_name}: {best_season}"
    
    def _generate_overall_recommendations(
        self,
        metric_analyses: Dict[str, Dict]
    ) -> List[str]:
        """Generate overall seasonal recommendations"""
        
        recommendations = []
        
        # Check for budget variations
        if "budget_per_day_low" in metric_analyses:
            budget_analysis = metric_analyses["budget_per_day_low"]
            if budget_analysis["variation_percentage"] > 20:
                recommendations.append(
                    f"Significant price variation ({budget_analysis['variation_percentage']}%) between seasons. "
                    f"Budget travelers should visit in {budget_analysis['low_season']}."
                )
        
        # Check for weather patterns
        temp_analysis = metric_analyses.get("temperature")
        rain_analysis = metric_analyses.get("rainfall")
        
        if temp_analysis and rain_analysis:
            # Find best weather combination
            best_weather_seasons = []
            for season in ["spring", "summer", "fall", "winter"]:
                if (season in temp_analysis["seasonal_averages"] and 
                    season in rain_analysis["seasonal_averages"]):
                    temp = temp_analysis["seasonal_averages"][season]
                    rain = rain_analysis["seasonal_averages"][season]
                    if 15 <= temp <= 25 and rain < 100:  # Comfortable temp, low rain
                        best_weather_seasons.append(season)
            
            if best_weather_seasons:
                recommendations.append(
                    f"Best weather conditions in: {', '.join(best_weather_seasons)}"
                )
        
        # Check for safety variations
        if "crime_index" in metric_analyses:
            safety_analysis = metric_analyses["crime_index"]
            if safety_analysis["variation_percentage"] > 15:
                recommendations.append(
                    f"Safety levels vary by season. Safest time: {safety_analysis['low_season']}"
                )
        
        return recommendations
    
    def _determine_best_visit_times(
        self,
        metric_analyses: Dict[str, Dict]
    ) -> Dict[str, List[str]]:
        """Determine best visit times for different traveler types"""
        
        best_times = {
            "budget_travelers": [],
            "weather_seekers": [],
            "crowd_avoiders": [],
            "adventure_travelers": []
        }
        
        # Budget travelers - lowest costs
        if "budget_per_day_low" in metric_analyses:
            budget_low = metric_analyses["budget_per_day_low"]["low_season"]
            if budget_low:
                best_times["budget_travelers"].append(budget_low)
        
        # Weather seekers - moderate temp, low rain
        if "temperature" in metric_analyses and "rainfall" in metric_analyses:
            temp_avgs = metric_analyses["temperature"]["seasonal_averages"]
            rain_avgs = metric_analyses["rainfall"]["seasonal_averages"]
            
            for season in ["spring", "summer", "fall", "winter"]:
                if season in temp_avgs and season in rain_avgs:
                    if 18 <= temp_avgs[season] <= 28 and rain_avgs[season] < 80:
                        best_times["weather_seekers"].append(season)
        
        # Crowd avoiders - low tourist volume, reasonable weather
        if "tourist_volume" in metric_analyses:
            low_crowd = metric_analyses["tourist_volume"]["low_season"]
            if low_crowd:
                best_times["crowd_avoiders"].append(low_crowd)
        
        # Adventure travelers - season-specific activities
        if "temperature" in metric_analyses:
            temp_avgs = metric_analyses["temperature"]["seasonal_averages"]
            
            # Winter sports
            if "winter" in temp_avgs and temp_avgs["winter"] < 5:
                best_times["adventure_travelers"].append("winter (skiing/snowboarding)")
            
            # Summer activities
            if "summer" in temp_avgs and temp_avgs["summer"] > 20:
                best_times["adventure_travelers"].append("summer (hiking/water sports)")
        
        return {k: v for k, v in best_times.items() if v}


class SeasonalComparator:
    """Compare seasonal patterns across destinations"""
    
    def __init__(self, tracker: SeasonalTracker):
        self.tracker = tracker
    
    def compare_destinations_by_season(
        self,
        destinations: List[str],
        metric: str,
        season: str
    ) -> Dict[str, Any]:
        """Compare multiple destinations for a specific metric in a given season"""
        
        comparison = {
            "metric": metric,
            "season": season,
            "destinations": {},
            "rankings": [],
            "best_value": None,
            "worst_value": None
        }
        
        values = []
        
        for dest in destinations:
            analysis = self.tracker.analyze_seasonal_patterns(dest, metric)
            if "error" not in analysis and season in analysis["seasonal_averages"]:
                value = analysis["seasonal_averages"][season]
                comparison["destinations"][dest] = {
                    "value": value,
                    "variation": analysis["seasonal_variations"].get(season, 0)
                }
                values.append((dest, value))
        
        # Create rankings
        values.sort(key=lambda x: x[1])
        comparison["rankings"] = [
            {"rank": i+1, "destination": dest, "value": val}
            for i, (dest, val) in enumerate(values)
        ]
        
        if values:
            comparison["best_value"] = values[0]
            comparison["worst_value"] = values[-1]
        
        return comparison
    
    def find_complementary_destinations(
        self,
        primary_destination: str,
        travel_months: List[int]
    ) -> List[Dict[str, Any]]:
        """Find destinations with opposite seasonal patterns"""
        
        # Get primary destination's patterns
        primary_patterns = {}
        for metric in ["temperature", "rainfall", "budget_per_day_low"]:
            analysis = self.tracker.analyze_seasonal_patterns(primary_destination, metric)
            if "error" not in analysis:
                primary_patterns[metric] = analysis
        
        if not primary_patterns:
            return []
        
        # Find destinations with complementary patterns
        complementary = []
        
        # This would need access to all destinations in the system
        # For now, returning empty list as placeholder
        # In real implementation, would search through all tracked destinations
        
        return complementary 