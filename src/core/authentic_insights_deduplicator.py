"""
Authentic Insights Deduplicator
Handles cross-referencing and deduplication of authentic insights across themes and destinations
"""

import hashlib
import uuid
from typing import Dict, List, Set, Any, Optional, Tuple
from datetime import datetime
import logging

from .enhanced_data_models import AuthenticInsight, Theme, Destination
from .safe_dict_utils import safe_get, safe_get_confidence_value, safe_get_nested, safe_get_dict

logger = logging.getLogger(__name__)


class AuthenticInsightsRegistry:
    """Registry for managing deduplicated authentic insights with cross-references"""
    
    def __init__(self):
        self.insights_by_id: Dict[str, AuthenticInsight] = {}
        self.insights_by_hash: Dict[str, str] = {}  # content_hash -> insight_id
        self.theme_insight_mappings: Dict[str, List[str]] = {}  # theme_id -> [insight_ids]
        self.insight_theme_mappings: Dict[str, List[str]] = {}  # insight_id -> [theme_ids]
        self.destination_insight_mappings: Dict[str, List[str]] = {}  # destination_id -> [insight_ids]
        self.logger = logging.getLogger(__name__)
        
    def add_insight(self, insight: AuthenticInsight, 
                   theme_id: Optional[str] = None,
                   destination_id: Optional[str] = None) -> str:
        """
        Add insight to registry with deduplication and cross-referencing
        
        Args:
            insight: AuthenticInsight to add
            theme_id: Optional theme ID to associate with
            destination_id: Optional destination ID to associate with
            
        Returns:
            Insight ID (existing if duplicate, new if unique)
        """
        # Calculate content hash for deduplication
        content_hash = self._calculate_insight_hash(insight)
        
        # Check if insight already exists
        if content_hash in self.insights_by_hash:
            existing_insight_id = self.insights_by_hash[content_hash]
            self.logger.debug(f"Found duplicate insight, using existing ID: {existing_insight_id}")
            
            # Update cross-references for existing insight
            self._update_cross_references(existing_insight_id, theme_id, destination_id)
            return existing_insight_id
        
        # Create new insight ID if not duplicate
        insight_id = str(uuid.uuid4())
        
        # Store insight
        self.insights_by_id[insight_id] = insight
        self.insights_by_hash[content_hash] = insight_id
        
        # Set up cross-references
        self._update_cross_references(insight_id, theme_id, destination_id)
        
        self.logger.debug(f"Added new insight with ID: {insight_id}")
        return insight_id
    
    def get_insight(self, insight_id: str) -> Optional[AuthenticInsight]:
        """Get insight by ID"""
        return self.insights_by_id.get(insight_id)
    
    def get_insights_for_theme(self, theme_id: str) -> List[str]:
        """Get all insight IDs associated with a theme"""
        return self.theme_insight_mappings.get(theme_id, [])
    
    def get_insights_for_destination(self, destination_id: str) -> List[str]:
        """Get all insight IDs associated with a destination"""
        return self.destination_insight_mappings.get(destination_id, [])
    
    def get_themes_for_insight(self, insight_id: str) -> List[str]:
        """Get all theme IDs that reference an insight"""
        return self.insight_theme_mappings.get(insight_id, [])
    
    def get_all_insights(self) -> Dict[str, AuthenticInsight]:
        """Get all insights in the registry"""
        return self.insights_by_id.copy()
    
    def get_cross_reference_statistics(self) -> Dict[str, Any]:
        """Get statistics about cross-references and deduplication"""
        total_insights = len(self.insights_by_id)
        
        # Count insights with multiple theme references
        multi_theme_insights = sum(1 for insight_id in self.insight_theme_mappings 
                                  if len(self.insight_theme_mappings[insight_id]) > 1)
        
        # Count themes with insights
        themes_with_insights = len(self.theme_insight_mappings)
        
        # Count destinations with insights
        destinations_with_insights = len(self.destination_insight_mappings)
        
        return {
            "total_unique_insights": total_insights,
            "insights_with_multiple_theme_refs": multi_theme_insights,
            "themes_with_insights": themes_with_insights,
            "destinations_with_insights": destinations_with_insights,
            "average_insights_per_theme": (
                sum(len(insights) for insights in self.theme_insight_mappings.values()) / 
                themes_with_insights if themes_with_insights > 0 else 0
            ),
            "deduplication_efficiency": self._calculate_deduplication_efficiency()
        }
    
    def _calculate_insight_hash(self, insight: AuthenticInsight) -> str:
        """Calculate content-based hash for insight deduplication"""
        # Handle both object and dictionary formats
        if isinstance(insight, dict):
            # Create hash based on core content for dictionary format
            content_parts = [
                str(safe_get_nested(insight, ['insight_type', 'value'], safe_get(insight, 'insight_type', ''))),
                str(safe_get(insight, 'authenticity_score', 0)),
                str(safe_get(insight, 'uniqueness_score', 0)),
                str(safe_get(insight, 'actionability_score', 0)),
                str(safe_get_nested(insight, ['location_exclusivity', 'value'], safe_get(insight, 'location_exclusivity', ''))),
                str(safe_get(insight, 'local_validation_count', 0))
            ]
            
            # Add seasonal window if present
            seasonal_window = safe_get(insight, 'seasonal_window')
            if seasonal_window:
                content_parts.extend([
                    str(safe_get(seasonal_window, 'start_month', '')),
                    str(safe_get(seasonal_window, 'end_month', '')),
                    str(safe_get(seasonal_window, 'booking_lead_time', ''))
                ])
        else:
            # Handle object format - use safe attribute access
            insight_type = getattr(insight, 'insight_type', None)
            location_exclusivity = getattr(insight, 'location_exclusivity', None)
            
            content_parts = [
                str(insight_type.value if hasattr(insight_type, 'value') else insight_type),
                str(getattr(insight, 'authenticity_score', 0)),
                str(getattr(insight, 'uniqueness_score', 0)),
                str(getattr(insight, 'actionability_score', 0)),
                str(location_exclusivity.value if hasattr(location_exclusivity, 'value') else location_exclusivity),
                str(getattr(insight, 'local_validation_count', 0))
            ]
            
            # Add seasonal window if present - safe access
            seasonal_window = getattr(insight, 'seasonal_window', None)
            if seasonal_window:
                content_parts.extend([
                    str(getattr(seasonal_window, 'start_month', '')),
                    str(getattr(seasonal_window, 'end_month', '')),
                    str(getattr(seasonal_window, 'booking_lead_time', ''))
                ])
        
        content_string = "|".join(content_parts)
        return hashlib.sha256(content_string.encode()).hexdigest()
    
    def _update_cross_references(self, insight_id: str, 
                                theme_id: Optional[str] = None,
                                destination_id: Optional[str] = None):
        """Update cross-reference mappings"""
        if theme_id:
            # Theme -> Insight mapping
            if theme_id not in self.theme_insight_mappings:
                self.theme_insight_mappings[theme_id] = []
            if insight_id not in self.theme_insight_mappings[theme_id]:
                self.theme_insight_mappings[theme_id].append(insight_id)
            
            # Insight -> Theme mapping
            if insight_id not in self.insight_theme_mappings:
                self.insight_theme_mappings[insight_id] = []
            if theme_id not in self.insight_theme_mappings[insight_id]:
                self.insight_theme_mappings[insight_id].append(theme_id)
        
        if destination_id:
            # Destination -> Insight mapping
            if destination_id not in self.destination_insight_mappings:
                self.destination_insight_mappings[destination_id] = []
            if insight_id not in self.destination_insight_mappings[destination_id]:
                self.destination_insight_mappings[destination_id].append(insight_id)
    
    def _calculate_deduplication_efficiency(self) -> float:
        """Calculate deduplication efficiency score (0-1)"""
        if not self.insights_by_id:
            return 1.0
        
        # Calculate how many references would have been duplicated without deduplication
        total_theme_insight_refs = sum(len(insights) for insights in self.theme_insight_mappings.values())
        total_destination_insight_refs = sum(len(insights) for insights in self.destination_insight_mappings.values())
        total_references = total_theme_insight_refs + total_destination_insight_refs
        
        unique_insights = len(self.insights_by_id)
        
        if total_references == 0:
            return 1.0
        
        # Efficiency = unique_insights / total_references (lower means more deduplication)
        return unique_insights / total_references


class AuthenticInsightsDeduplicator:
    """Main deduplicator for processing authentic insights across destinations and themes"""
    
    def __init__(self):
        self.registry = AuthenticInsightsRegistry()
        self.logger = logging.getLogger(__name__)
    
    def process_destination(self, destination: Destination) -> Tuple[Dict[str, AuthenticInsight], Dict[str, Any]]:
        """Process all authentic insights for a destination with deduplication"""
        
        # Safe access to destination ID
        if hasattr(destination, 'id'):
            destination_id = destination.id
        elif isinstance(destination, dict):
            destination_id = destination.get('id', 'unknown_destination')
        else:
            destination_id = 'unknown_destination'
        
        self.logger.info(f"Processing authentic insights for destination: {destination_id}")
        
        # Safe access to themes
        if hasattr(destination, 'themes'):
            themes = destination.themes
        elif isinstance(destination, dict):
            themes = destination.get('themes', [])
        else:
            themes = []
        
        # Process insights from themes
        for theme in themes:
            # Safe access to theme attributes
            if hasattr(theme, 'authentic_insights'):
                insights = theme.authentic_insights
            elif isinstance(theme, dict):
                insights = theme.get('authentic_insights', [])
            else:
                insights = []
                
            for insight in insights:
                # Safe access to theme_id
                if hasattr(theme, 'theme_id'):
                    theme_id = theme.theme_id
                elif isinstance(theme, dict):
                    theme_id = theme.get('theme_id', 'unknown_theme')
                else:
                    theme_id = 'unknown_theme'
                    
                self.registry.add_insight(insight, theme_id=theme_id, destination_id=destination_id)
        
        # Process destination-level insights if they exist
        if hasattr(destination, 'authentic_insights'):
            dest_insights = destination.authentic_insights
        elif isinstance(destination, dict):
            dest_insights = destination.get('authentic_insights', [])
        else:
            dest_insights = []
            
        for insight in dest_insights:
            self.registry.add_insight(insight, destination_id=destination_id)
        
        # Get deduplicated insights and cross-references
        deduplicated_insights = self.registry.get_all_insights()
        cross_references = self.registry.get_cross_reference_statistics()
        
        # Calculate efficiency
        total_processed = self.registry.get_cross_reference_statistics()["total_unique_insights"]
        unique_insights = len(deduplicated_insights)
        efficiency = (unique_insights / total_processed * 100) if total_processed > 0 else 100.0
        
        self.logger.info(f"Processed {unique_insights} unique insights with {efficiency:.2f}% efficiency")
        
        return deduplicated_insights, cross_references
    
    def apply_deduplication_to_destination(self, destination: Destination) -> Destination:
        """
        Apply deduplication to destination by removing duplicate insights and adding references
        
        Args:
            destination: Original destination
            
        Returns:
            Updated destination with deduplicated insights
        """
        # Process insights to get deduplicated registry
        insights_registry, cross_refs = self.process_destination(destination)
        
        # Update destination to use only unique insights
        destination.authentic_insights = list(insights_registry.values())
        
        # Update themes to use insight references instead of duplicating insights
        for theme in destination.themes:
            theme_id = getattr(theme, 'theme_id', None) if hasattr(theme, 'theme_id') else safe_get(theme, 'theme_id', None) if isinstance(theme, dict) else None
            
            # Clear theme-level authentic insights to avoid duplication - safe access
            if isinstance(theme, dict):
                theme['authentic_insights'] = []
            elif hasattr(theme, 'authentic_insights'):
                theme.authentic_insights = []
            
            # Add insight references instead - safe access
            insight_ids = self.registry.get_insights_for_theme(theme_id)
            if isinstance(theme, dict):
                theme['insight_references'] = [{"insight_id": insight_id, "relevance_score": 1.0} 
                                             for insight_id in insight_ids]
            else:
                if not hasattr(theme, 'insight_references'):
                    theme.insight_references = []
                theme.insight_references = [{"insight_id": insight_id, "relevance_score": 1.0} 
                                           for insight_id in insight_ids]
        
        return destination
    
    def get_registry(self) -> AuthenticInsightsRegistry:
        """Get the insights registry"""
        return self.registry
    
    def clear_registry(self):
        """Clear the registry (useful for processing multiple destinations)"""
        self.registry = AuthenticInsightsRegistry()


def deduplicate_authentic_insights_for_export(destination: Destination) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    Convenience function to deduplicate authentic insights for export
    
    Args:
        destination: Destination to process
        
    Returns:
        Tuple of (insights_for_export, cross_reference_mappings)
    """
    deduplicator = AuthenticInsightsDeduplicator()
    insights_registry, cross_refs = deduplicator.process_destination(destination)
    
    # Export insights as dictionaries
    insights_for_export = {}
    for insight_id, insight in insights_registry.items():
        # Proper type checking for insights
        if isinstance(insight, dict):
            insights_for_export[insight_id] = insight
        elif hasattr(insight, 'to_dict') and callable(getattr(insight, 'to_dict')):
            insights_for_export[insight_id] = insight.to_dict()
        else:
            insights_for_export[insight_id] = {"error": "unexpected_insight_type", "type": str(type(insight))}
    
    return insights_for_export, cross_refs 