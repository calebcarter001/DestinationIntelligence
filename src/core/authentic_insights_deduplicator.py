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
        # Create hash based on core content (exclude metadata like timestamps)
        content_parts = [
            str(insight.insight_type.value),
            str(insight.authenticity_score),
            str(insight.uniqueness_score),
            str(insight.actionability_score),
            str(insight.location_exclusivity.value),
            str(insight.local_validation_count)
        ]
        
        # Add seasonal window if present
        if insight.seasonal_window:
            content_parts.extend([
                str(insight.seasonal_window.start_month),
                str(insight.seasonal_window.end_month),
                str(insight.seasonal_window.booking_lead_time)
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
        """
        Process a destination and all its themes to deduplicate authentic insights
        
        Args:
            destination: Destination to process
            
        Returns:
            Tuple of (insights_registry, cross_reference_mappings)
        """
        self.logger.info(f"Processing authentic insights for destination: {destination.id}")
        
        # Process destination-level insights
        for insight in destination.authentic_insights:
            self.registry.add_insight(insight, destination_id=destination.id)
        
        # Process theme-level insights
        for theme in destination.themes:
            theme_id = theme.theme_id
            
            # Process insights associated with this theme
            if hasattr(theme, 'authentic_insights'):
                for insight in theme.authentic_insights:
                    self.registry.add_insight(insight, theme_id=theme_id, destination_id=destination.id)
        
        # Get all unique insights
        insights_registry = self.registry.get_all_insights()
        
        # Build cross-reference mappings
        cross_reference_mappings = {
            "theme_insights": self.registry.theme_insight_mappings.copy(),
            "insight_themes": self.registry.insight_theme_mappings.copy(),
            "destination_insights": self.registry.destination_insight_mappings.copy()
        }
        
        stats = self.registry.get_cross_reference_statistics()
        self.logger.info(f"Processed {stats['total_unique_insights']} unique insights with "
                        f"{stats['deduplication_efficiency']:.2%} efficiency")
        
        return insights_registry, cross_reference_mappings
    
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
            theme_id = theme.theme_id
            
            # Clear theme-level authentic insights to avoid duplication
            if hasattr(theme, 'authentic_insights'):
                theme.authentic_insights = []
            
            # Add insight references instead
            insight_ids = self.registry.get_insights_for_theme(theme_id)
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
    
    # Convert insights to export format
    insights_for_export = {}
    for insight_id, insight in insights_registry.items():
        insights_for_export[insight_id] = insight.to_dict() if hasattr(insight, 'to_dict') else insight
    
    return insights_for_export, cross_refs 