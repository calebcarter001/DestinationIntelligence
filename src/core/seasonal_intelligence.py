from typing import List, Dict, Optional, Any, Union
from datetime import datetime
import logging
import re

from src.core.enhanced_data_models import SeasonalWindow, AuthenticInsight

logger = logging.getLogger(__name__)

class SeasonalIntelligence:
    """Handles time-sensitive insights, extracts seasonal patterns, and generates timing recommendations."""
    
    def __init__(self, llm_interface: Any = None):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.llm_interface = llm_interface

    def extract_seasonal_patterns(self, content: Union[List[str], str, None]) -> List[Dict[str, Any]]:
        """Find seasonal availability, timing, etc."""
        patterns = []
        
        if content is None:
            return patterns
        
        # Handle both list and string inputs
        if isinstance(content, list):
            # Filter out None values before joining
            valid_content = [item for item in content if item is not None]
            content_text = " ".join(valid_content)
        else:
            content_text = content or ""
            
        content_lower = content_text.lower()

        # Extract seasonal windows using InsightClassifier logic
        from src.core.insight_classifier import InsightClassifier
        classifier = InsightClassifier()
        
        # Try to extract seasonal window
        seasonal_window = classifier.extract_seasonal_window(content_text)
        if seasonal_window:
            patterns.append({
                "type": "seasonal_window",
                "seasonal_window": seasonal_window,
                "timing": f"Month {getattr(seasonal_window, 'start_month', 'N/A')} to {getattr(seasonal_window, 'end_month', 'N/A')}"
            })

        # Simple regex for common seasonal phrases
        if re.search(r'(maple|sugar)\s*season', content_lower):
            patterns.append({"type": "seasonal_event", "name": "Maple Season", "timing": "Spring"})
        if re.search(r'harvest\s*time', content_lower):
            patterns.append({"type": "seasonal_event", "name": "Harvest Time", "timing": "Fall"})
        if re.search(r'peak\s*season', content_lower):
            patterns.append({"type": "general_season", "name": "Peak Season", "timing": "Unspecified"})
        if re.search(r'fall\s*foliage|autumn\s*colors', content_lower):
            patterns.append({"type": "seasonal_event", "name": "Fall Foliage", "timing": "Fall"})
        if re.search(r'best\s*time\s*to\s*visit\s*(?:is|during)?\s*(january|february|march|april|may|june|july|august|september|october|november|december)', content_lower):
            match = re.search(r'best\s*time\s*to\s*visit\s*(?:is|during)?\s*(january|february|march|april|may|june|july|august|september|october|november|december)', content_lower)
            if match:
                patterns.append({"type": "recommendation", "name": "Best Visit Time", "timing": match.group(1).title()})
        if re.search(r'winter\s*only', content_lower):
            patterns.append({"type": "seasonal_constraint", "name": "Winter Only", "timing": "Winter"})

        return patterns

    def calculate_current_relevance(self, seasonal_window: Optional[SeasonalWindow]) -> float:
        """How relevant is this insight right now?"""
        if not seasonal_window:
            return 0.5  # Default relevance for non-seasonal content

        current_month = datetime.now().month
        relevance = 0.0

        # Handle invalid months gracefully
        start_month = getattr(seasonal_window, 'start_month', 1)
        end_month = getattr(seasonal_window, 'end_month', 12)
        
        if start_month < 1 or start_month > 12:
            start_month = 1
        if end_month < 1 or end_month > 12:
            end_month = 12

        # Check if current month is within the seasonal window
        if start_month <= end_month:
            if start_month <= current_month <= end_month:
                relevance = 1.0
            else:
                # Calculate distance to season
                distance = min(abs(current_month - start_month), abs(current_month - end_month))
                relevance = max(0.1, 1.0 - (distance / 6.0))  # Decay over 6 months
        else:  # Cross-year window (e.g., Dec-Feb)
            if current_month >= start_month or current_month <= end_month:
                relevance = 1.0
            else:
                # Calculate distance for cross-year
                distance_to_start = min(abs(current_month - start_month), 12 - abs(current_month - start_month))
                distance_to_end = min(abs(current_month - end_month), 12 - abs(current_month - end_month))
                distance = min(distance_to_start, distance_to_end)
                relevance = max(0.1, 1.0 - (distance / 6.0))
        
        # Further refine based on peak weeks, specific dates
        peak_weeks = getattr(seasonal_window, 'peak_weeks', [])
        if peak_weeks and datetime.now().isocalendar()[1] in peak_weeks:
            relevance = min(1.0, relevance + 0.2)  # Boost if in peak week
        
        specific_dates = getattr(seasonal_window, 'specific_dates', [])
        if specific_dates:
            today_str = datetime.now().strftime("%m/%d")
            if today_str in specific_dates or datetime.now().strftime("%m/%d/%Y") in specific_dates:
                relevance = 1.0

        return relevance

    def generate_timing_recommendations(self, seasonal_window: Optional[SeasonalWindow], activity: str = "") -> List[str]:
        """Generate when-to-visit recommendations based on seasonal window."""
        recommendations = []
        
        if not seasonal_window:
            recommendations.append(f"The {activity} is available year-round." if activity else "Available year-round.")
            return recommendations

        current_month = datetime.now().month
        current_relevance = self.calculate_current_relevance(seasonal_window)
        
        # Handle invalid months gracefully
        start_month = getattr(seasonal_window, 'start_month', 1)
        end_month = getattr(seasonal_window, 'end_month', 12)
        
        if start_month < 1 or start_month > 12:
            start_month = 1
        if end_month < 1 or end_month > 12:
            end_month = 12

        # Current season recommendations
        if current_relevance > 0.7:
            recommendations.append(f"Now is an excellent time to visit for {activity}." if activity else "Now is an excellent time to visit.")
        elif current_relevance > 0.4:
            recommendations.append(f"This is a good time for {activity}." if activity else "This is a good time to visit.")
        else:
            # Future planning recommendations
            if start_month <= end_month:
                season_months = list(range(start_month, end_month + 1))
            else:
                season_months = list(range(start_month, 13)) + list(range(1, end_month + 1))
            
            # Find next upcoming month in season
            upcoming_months = [m for m in season_months if m > current_month]
            if not upcoming_months:
                upcoming_months = season_months  # Next year
            
            if upcoming_months:
                next_month = min(upcoming_months)
                month_name = datetime(2000, next_month, 1).strftime('%B')
                recommendations.append(f"Plan ahead for {activity} - the season starts in {month_name}." if activity else f"Plan ahead - the season starts in {month_name}.")

        # Booking recommendations
        booking_lead_time = getattr(seasonal_window, 'booking_lead_time', None)
        if booking_lead_time:
            recommendations.append(f"Book {booking_lead_time} for the best {activity} experience." if activity else f"Book {booking_lead_time} for the best experience.")

        return recommendations

def month_name(month_num: int) -> str:
    """Helper to get month name from number."""
    return datetime(2000, month_num, 1).strftime('%B') 