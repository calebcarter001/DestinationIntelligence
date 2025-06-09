from typing import List, Dict, Optional, Any
from dataclasses import dataclass
import re
import logging

from src.schemas import InsightType, LocationExclusivity
from src.core.enhanced_data_models import SeasonalWindow, AuthenticInsight

logger = logging.getLogger(__name__)

class InsightClassifier:
    """Classify insights by type and value, and extract relevant details."""
    
    def __init__(self, llm_interface: Any = None):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.llm_interface = llm_interface

    def classify_insight_type(self, content: str) -> InsightType:
        """Determine if seasonal, specialty, insider, cultural, or practical."""
        if not content:
            return InsightType.PRACTICAL
        
        content_lower = content.lower()

        # Seasonal patterns
        if any(re.search(pattern, content_lower) for pattern in [
            r'season', r'winter', r'summer', r'spring', r'fall', r'autumn', 
            r'harvest', r'peak.*season', r'best.*time.*visit', r'seasonal.*hours'
        ]):
            return InsightType.SEASONAL
        
        # Specialty patterns
        if any(re.search(pattern, content_lower) for pattern in [
            r'famous.*for', r'known.*for', r'specialty', r'signature', r'craft.*beer',
            r'local.*brewery', r'artisan', r'local.*producer', r'unique.*dish'
        ]):
            return InsightType.SPECIALTY
        
        # Insider patterns
        if any(re.search(pattern, content_lower) for pattern in [
            r'locals.*know', r'hidden.*gem', r'secret', r'local.*tip',
            r'only.*residents', r'longtime.*locals'
        ]):
            return InsightType.INSIDER
        
        # Cultural patterns
        if any(re.search(pattern, content_lower) for pattern in [
            r'traditional', r'festival', r'cultural', r'heritage', r'historic',
            r'museum', r'art.*culture', r'indigenous'
        ]):
            return InsightType.CULTURAL
        
        # Default to practical if no specific type is found
        return InsightType.PRACTICAL

    def determine_location_exclusivity(self, content: str) -> LocationExclusivity:
        """Determine how exclusive content is to a location."""
        if not content:
            return LocationExclusivity.COMMON
        
        content_lower = content.lower()

        # Exclusive patterns
        if any(re.search(pattern, content_lower) for pattern in [
            r'only.*place.*world', r'unique.*to.*region', r'found.*nowhere.*else',
            r'exclusively.*available', r'cannot.*be.*found.*anywhere.*else'
        ]):
            return LocationExclusivity.EXCLUSIVE
        
        # Signature patterns
        if any(re.search(pattern, content_lower) for pattern in [
            r'famous.*for.*being.*best', r'premier.*destination', r'renowned.*worldwide',
            r'best.*known.*attraction', r'known.*as.*the.*premier'
        ]):
            return LocationExclusivity.SIGNATURE
        
        # Regional patterns
        if any(re.search(pattern, content_lower) for pattern in [
            r'common.*throughout', r'found.*across.*region', r'popular.*throughout',
            r'regional.*specialty', r'available.*in.*several'
        ]):
            return LocationExclusivity.REGIONAL
        
        # Common patterns
        if any(re.search(pattern, content_lower) for pattern in [
            r'available.*everywhere', r'commonly.*found', r'standard.*tourist',
            r'common.*activity', r'typical.*restaurant.*chain'
        ]):
            return LocationExclusivity.COMMON
        
        # Default to regional
        return LocationExclusivity.REGIONAL

    def extract_seasonal_window(self, content: str) -> Optional[SeasonalWindow]:
        """Extract timing information for seasonal insights."""
        if not content:
            return None
        
        content_lower = content.lower()
        
        start_month = None
        end_month = None
        peak_weeks = []
        booking_lead_time = None
        specific_dates = []

        # Month mapping
        month_map = {
            "january": 1, "jan": 1, "february": 2, "feb": 2, "march": 3, "mar": 3,
            "april": 4, "apr": 4, "may": 5, "june": 6, "jun": 6,
            "july": 7, "jul": 7, "august": 8, "aug": 8, "september": 9, "sep": 9,
            "october": 10, "oct": 10, "november": 11, "nov": 11, "december": 12, "dec": 12
        }

        # Extract month ranges (e.g., "March to November")
        month_range_pattern = r'(january|february|march|april|may|june|july|august|september|october|november|december|jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)\s+(?:to|through|-)\s+(january|february|march|april|may|june|july|august|september|october|november|december|jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)'
        range_match = re.search(month_range_pattern, content_lower)
        
        if range_match:
            start_month = month_map.get(range_match.group(1))
            end_month = month_map.get(range_match.group(2))
        else:
            # Single month extraction
            month_pattern = r'(january|february|march|april|may|june|july|august|september|october|november|december|jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)'
            month_matches = re.findall(month_pattern, content_lower)
            if month_matches:
                start_month = month_map.get(month_matches[0])
                end_month = month_map.get(month_matches[-1]) if len(month_matches) > 1 else start_month

        # Extract booking lead time
        booking_patterns = [
            r'book\s+(\d+\s+(?:weeks?|months?))\s+in\s+advance',
            r'(\d+\s+(?:weeks?|months?))\s+in\s+advance',
            r'book\s+(\d+\s+(?:weeks?|months?))'
        ]
        
        for pattern in booking_patterns:
            booking_match = re.search(pattern, content_lower)
            if booking_match:
                booking_lead_time = booking_match.group(1)
                break

        # Extract specific dates
        date_patterns = [
            r'\d{1,2}\/\d{1,2}(?:\/\d{2,4})?',
            r'\d{1,2}-\d{1,2}(?:-\d{2,4})?'
        ]
        
        for pattern in date_patterns:
            date_matches = re.findall(pattern, content)
            if date_matches:
                specific_dates.extend(date_matches)

        if start_month or end_month or peak_weeks or booking_lead_time or specific_dates:
            return SeasonalWindow(
                start_month=start_month if start_month else 1,
                end_month=end_month if end_month else 12,
                peak_weeks=peak_weeks,
                booking_lead_time=booking_lead_time,
                specific_dates=specific_dates if specific_dates else None
            )
        return None

    def extract_actionable_details(self, content: str) -> Dict[str, str]:
        """Extract what/when/where/how details from content."""
        if not content:
            return {}
        
        details = {}
        content_lower = content.lower()

        # What (activity, item)
        what_patterns = [
            r'(visit|try|see|eat|experience)\s+(?:the\s+)?([^.?!,]+)',
            r'([^.?!,]+(?:restaurant|brewery|museum|gallery|attraction))',
        ]
        
        for pattern in what_patterns:
            what_match = re.search(pattern, content_lower)
            if what_match:
                if len(what_match.groups()) == 2:
                    details["what"] = what_match.group(2).strip()
                else:
                    details["what"] = what_match.group(1).strip()
                break

        # When (time, season, hours)
        when_patterns = [
            r'(?:open|available|hours?)\s+([^.?!]+(?:am|pm|daily|monday|tuesday|wednesday|thursday|friday|saturday|sunday)[^.?!]*)',
            r'best\s+time\s+(?:to\s+visit\s+)?is\s+([^.?!]+)',
            r'(?:during|in)\s+([^.?!]+(?:morning|afternoon|evening|season))'
        ]
        
        for pattern in when_patterns:
            when_match = re.search(pattern, content_lower)
            if when_match:
                details["when"] = when_match.group(1).strip()
                break

        # Where (location, address)
        where_patterns = [
            r'(?:located\s+at|address[:\s]+)\s+([^.?!]+)',
            r'at\s+(\d+[^.?!]*(?:street|st|avenue|ave|road|rd|boulevard|blvd)[^.?!]*)',
            r'(?:in|near)\s+([^.?!]+(?:downtown|district|area|neighborhood)[^.?!]*)'
        ]
        
        for pattern in where_patterns:
            where_match = re.search(pattern, content_lower)
            if where_match:
                details["where"] = where_match.group(1).strip()
                break

        # How (booking, contact, tips)
        how_patterns = [
            r'(call\s+ahead[^.?!]*)',
            r'(?:call|phone)\s+([^.?!]+)',
            r'(?:book|reserve|reservation)[^.?!]*(?:by|at|through)\s+([^.?!]+)',
            r'(?:how\s+to|tips\s+for)\s+([^.?!]+)'
        ]
        
        for pattern in how_patterns:
            how_match = re.search(pattern, content_lower)
            if how_match:
                details["how"] = how_match.group(1).strip()
                break

        return details 