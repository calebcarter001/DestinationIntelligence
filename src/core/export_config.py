"""
Export Configuration System
Provides configurable export modes and smart view generation options
"""

from enum import Enum
from pydantic import BaseModel, Field
from typing import Dict, Any, List, Optional


class ExportMode(Enum):
    """Export mode options"""
    COMPREHENSIVE = "comprehensive"    # Full data with all views
    MINIMAL = "minimal"               # Essential data only
    SUMMARY_ONLY = "summary_only"     # Executive summary only
    EVIDENCE_FOCUSED = "evidence_focused"  # Evidence and relationships only
    THEMES_FOCUSED = "themes_focused"      # Themes and insights only


class ViewGenerationMode(Enum):
    """How views should be generated"""
    PRECOMPUTED = "precomputed"       # Generate all views during export
    DYNAMIC = "dynamic"               # Generate views on-demand
    HYBRID = "hybrid"                 # Key views precomputed, others dynamic


class DeduplicationLevel(Enum):
    """Level of deduplication to apply"""
    STRICT = "strict"                 # Maximum deduplication
    MODERATE = "moderate"             # Balanced approach
    MINIMAL = "minimal"               # Limited deduplication


class ExportConfig(BaseModel):
    """Configuration for destination export"""
    
    # Core export settings
    mode: ExportMode = Field(default=ExportMode.COMPREHENSIVE, description="Export mode")
    version: str = Field(default="3.1", description="Export format version")
    
    # Content inclusion settings
    include_evidence_text: bool = Field(default=True, description="Include full evidence text")
    include_metadata: bool = Field(default=True, description="Include analysis metadata")
    include_relationships: bool = Field(default=True, description="Include relationship mappings")
    include_duplicate_views: bool = Field(default=False, description="Allow duplicate data in views")
    
    # Filtering and limits
    max_evidence_per_theme: Optional[int] = Field(default=None, description="Limit evidence per theme")
    min_confidence_threshold: float = Field(default=0.3, description="Minimum confidence to include")
    max_themes_per_category: Optional[int] = Field(default=None, description="Limit themes per category")
    
    # Deduplication settings
    deduplication_level: DeduplicationLevel = Field(default=DeduplicationLevel.STRICT, description="Deduplication level")
    deduplicate_similar_evidence: bool = Field(default=True, description="Remove similar evidence")
    similarity_threshold: float = Field(default=0.85, description="Similarity threshold for deduplication")
    
    # View generation settings
    view_generation_mode: ViewGenerationMode = Field(default=ViewGenerationMode.PRECOMPUTED, description="View generation mode")
    include_executive_summary: bool = Field(default=True, description="Include executive summary view")
    include_category_views: bool = Field(default=True, description="Include category-organized views")
    include_seasonal_views: bool = Field(default=True, description="Include seasonal analysis views")
    include_quality_dashboard: bool = Field(default=True, description="Include quality metrics dashboard")
    
    # Performance settings
    enable_compression: bool = Field(default=False, description="Enable JSON compression")
    streaming_export: bool = Field(default=False, description="Use streaming for large exports")
    cache_views: bool = Field(default=True, description="Cache generated views")
    
    # Custom view definitions
    custom_views: Dict[str, Dict[str, Any]] = Field(default_factory=dict, description="Custom view definitions")
    
    # Output format settings
    pretty_print: bool = Field(default=True, description="Pretty print JSON output")
    include_timestamps: bool = Field(default=True, description="Include timestamps in output")
    include_export_stats: bool = Field(default=True, description="Include export statistics")


class SmartViewGenerator:
    """Smart view generation with configurable options"""
    
    def __init__(self, config: ExportConfig):
        self.config = config
        
    def generate_views(self, destination_data: Dict[str, Any], 
                      evidence_registry: Dict[str, Any],
                      themes_data: Dict[str, Any]) -> Dict[str, Any]:
        """Generate views based on configuration"""
        views = {}
        
        if self.config.include_executive_summary:
            views["executive_summary"] = self._generate_executive_summary(
                destination_data, themes_data
            )
            
        if self.config.include_category_views:
            views["themes_by_category"] = self._generate_category_views(themes_data)
            views["evidence_by_source"] = self._generate_evidence_by_source_views(evidence_registry)
            
        if self.config.include_seasonal_views:
            views["seasonal_overview"] = self._generate_seasonal_views(themes_data)
            
        if self.config.include_quality_dashboard:
            views["quality_dashboard"] = self._generate_quality_dashboard(
                destination_data, evidence_registry, themes_data
            )
            
        # Generate custom views
        for view_name, view_config in self.config.custom_views.items():
            views[view_name] = self._generate_custom_view(
                view_config, destination_data, evidence_registry, themes_data
            )
            
        return views
    
    def _generate_executive_summary(self, destination_data: Dict[str, Any], 
                                   themes_data: Dict[str, Any]) -> Dict[str, Any]:
        """Generate executive summary view"""
        # Filter high confidence themes
        high_confidence_themes = []
        for theme_id, theme in themes_data.items():
            confidence_breakdown = theme.get("confidence_breakdown", {})
            # Handle None confidence_breakdown
            if confidence_breakdown is None:
                confidence_breakdown = {}
            confidence = confidence_breakdown.get("overall_confidence", 0)
            if confidence >= 0.7:  # High confidence threshold
                high_confidence_themes.append({
                    "theme_id": theme_id,
                    "name": theme["name"],
                    "category": theme["macro_category"],
                    "confidence": confidence,
                    "fit_score": theme["fit_score"]
                })
        
        # Sort by confidence and fit score
        high_confidence_themes.sort(
            key=lambda x: (x["confidence"], x["fit_score"]), 
            reverse=True
        )
        
        return {
            "destination_name": destination_data["names"][0] if destination_data["names"] else "Unknown",
            "country": destination_data["admin_levels"].get("country", "Unknown"),
            "total_themes": len(themes_data),
            "high_confidence_themes": len(high_confidence_themes),
            "top_themes": high_confidence_themes[:5],  # Top 5 themes
            "generated_timestamp": destination_data.get("last_updated"),
            "summary_confidence": sum(t["confidence"] for t in high_confidence_themes) / len(high_confidence_themes) if high_confidence_themes else 0
        }
    
    def _generate_category_views(self, themes_data: Dict[str, Any]) -> Dict[str, List[str]]:
        """Generate themes organized by category"""
        categories = {}
        for theme_id, theme in themes_data.items():
            category = theme["macro_category"]
            if category not in categories:
                categories[category] = []
            categories[category].append(theme_id)
            
        # Sort themes within each category by fit score
        for category in categories:
            theme_scores = [(tid, themes_data[tid]["fit_score"]) for tid in categories[category]]
            theme_scores.sort(key=lambda x: x[1], reverse=True)
            categories[category] = [tid for tid, _ in theme_scores]
            
        return categories
    
    def _generate_evidence_by_source_views(self, evidence_registry: Dict[str, Any]) -> Dict[str, List[str]]:
        """Generate evidence organized by source type"""
        sources = {}
        for evidence_id, evidence in evidence_registry.items():
            source_type = evidence.get("source_category", "unknown")
            if source_type not in sources:
                sources[source_type] = []
            sources[source_type].append(evidence_id)
            
        # Sort by authority weight within each source type
        for source_type in sources:
            evidence_weights = [(eid, evidence_registry[eid].get("authority_weight", 0)) 
                               for eid in sources[source_type]]
            evidence_weights.sort(key=lambda x: x[1], reverse=True)
            sources[source_type] = [eid for eid, _ in evidence_weights]
            
        return sources
    
    def _generate_seasonal_views(self, themes_data: Dict[str, Any]) -> Dict[str, Any]:
        """Generate seasonal analysis views"""
        seasonal_themes = {}
        current_season_themes = []
        
        for theme_id, theme in themes_data.items():
            seasonal_relevance = theme.get("seasonal_relevance", {})
            if seasonal_relevance:
                for season, relevance in seasonal_relevance.items():
                    if relevance > 0.5:  # Significant seasonal relevance
                        if season not in seasonal_themes:
                            seasonal_themes[season] = []
                        seasonal_themes[season].append({
                            "theme_id": theme_id,
                            "name": theme["name"],
                            "relevance": relevance
                        })
                        
                # Check for current season relevance (simplified)
                current_relevance = seasonal_relevance.get("current", 0)
                if current_relevance > 0.7:
                    current_season_themes.append({
                        "theme_id": theme_id,
                        "name": theme["name"],
                        "current_relevance": current_relevance
                    })
        
        # Sort by relevance
        for season in seasonal_themes:
            seasonal_themes[season].sort(key=lambda x: x["relevance"], reverse=True)
            
        current_season_themes.sort(key=lambda x: x["current_relevance"], reverse=True)
        
        return {
            "seasonal_themes": seasonal_themes,
            "current_season_highlights": current_season_themes[:3],  # Top 3 current themes
            "seasons_covered": list(seasonal_themes.keys())
        }
    
    def _generate_quality_dashboard(self, destination_data: Dict[str, Any],
                                   evidence_registry: Dict[str, Any],
                                   themes_data: Dict[str, Any]) -> Dict[str, Any]:
        """Generate quality metrics dashboard"""
        # Evidence quality metrics
        evidence_confidences = [ev.get("confidence", 0) for ev in evidence_registry.values()]
        evidence_authorities = [ev.get("authority_weight", 0) for ev in evidence_registry.values()]
        
        # Theme quality metrics
        theme_confidences = []
        theme_fit_scores = []
        for theme in themes_data.values():
            confidence_breakdown = theme.get("confidence_breakdown", {})
            # Handle None confidence_breakdown
            if confidence_breakdown is not None:
                theme_confidences.append(confidence_breakdown.get("overall_confidence", 0))
            else:
                theme_confidences.append(0)  # Default confidence if breakdown is None
            theme_fit_scores.append(theme.get("fit_score", 0))
        
        # Source diversity
        source_types = set(ev.get("source_category") for ev in evidence_registry.values())
        unique_sources = set(ev.get("source_url") for ev in evidence_registry.values())
        
        return {
            "evidence_quality": {
                "total_evidence": len(evidence_registry),
                "average_confidence": sum(evidence_confidences) / len(evidence_confidences) if evidence_confidences else 0,
                "average_authority": sum(evidence_authorities) / len(evidence_authorities) if evidence_authorities else 0,
                "high_confidence_ratio": len([c for c in evidence_confidences if c > 0.7]) / len(evidence_confidences) if evidence_confidences else 0
            },
            "theme_quality": {
                "total_themes": len(themes_data),
                "average_confidence": sum(theme_confidences) / len(theme_confidences) if theme_confidences else 0,
                "average_fit_score": sum(theme_fit_scores) / len(theme_fit_scores) if theme_fit_scores else 0,
                "high_quality_themes": len([c for c in theme_confidences if c > 0.8])
            },
            "source_diversity": {
                "source_types": len(source_types),
                "unique_sources": len(unique_sources),
                "diversity_score": len(unique_sources) / len(evidence_registry) if evidence_registry else 0
            },
            "overall_quality_score": self._calculate_overall_quality_score(
                evidence_confidences, theme_confidences, len(source_types)
            )
        }
    
    def _generate_custom_view(self, view_config: Dict[str, Any],
                             destination_data: Dict[str, Any],
                             evidence_registry: Dict[str, Any],
                             themes_data: Dict[str, Any]) -> Dict[str, Any]:
        """Generate custom view based on configuration"""
        # This is a simplified implementation - can be extended for specific custom views
        view_type = view_config.get("type", "basic")
        
        if view_type == "filtered_themes":
            # Filter themes by criteria
            criteria = view_config.get("filter_criteria", {})
            min_confidence = criteria.get("min_confidence", 0.5)
            categories = criteria.get("categories", [])
            
            filtered_themes = {}
            for theme_id, theme in themes_data.items():
                confidence_breakdown = theme.get("confidence_breakdown", {})
                # Handle None confidence_breakdown
                if confidence_breakdown is not None:
                    confidence = confidence_breakdown.get("overall_confidence", 0)
                else:
                    confidence = 0
                category = theme.get("macro_category", "")
                
                if confidence >= min_confidence and (not categories or category in categories):
                    filtered_themes[theme_id] = theme
                    
            return {
                "filtered_themes": filtered_themes,
                "filter_criteria": criteria,
                "total_filtered": len(filtered_themes)
            }
        
        return {"type": view_type, "data": "Custom view not implemented"}
    
    def _calculate_overall_quality_score(self, evidence_confidences: List[float],
                                        theme_confidences: List[float],
                                        source_type_count: int) -> float:
        """Calculate overall quality score (0-1)"""
        scores = []
        
        # Evidence quality component
        if evidence_confidences:
            avg_evidence_conf = sum(evidence_confidences) / len(evidence_confidences)
            scores.append(avg_evidence_conf)
        
        # Theme quality component
        if theme_confidences:
            avg_theme_conf = sum(theme_confidences) / len(theme_confidences)
            scores.append(avg_theme_conf)
        
        # Source diversity component (normalized to 0-1, assuming max 10 source types)
        diversity_score = min(source_type_count / 10.0, 1.0)
        scores.append(diversity_score)
        
        return sum(scores) / len(scores) if scores else 0.0


def create_export_config_from_yaml(config_dict: Dict[str, Any]) -> ExportConfig:
    """Create export config from YAML configuration"""
    export_settings = config_dict.get("export_settings", {})
    
    # Map YAML settings to ExportConfig
    return ExportConfig(
        mode=ExportMode(export_settings.get("mode", "comprehensive")),
        include_evidence_text=export_settings.get("include_evidence_text", True),
        include_duplicate_views=export_settings.get("include_duplicate_views", False),
        max_evidence_per_theme=export_settings.get("max_evidence_per_theme"),
        deduplicate_similar_evidence=export_settings.get("deduplicate_similar_evidence", True),
        similarity_threshold=export_settings.get("similarity_threshold", 0.85),
        min_confidence_threshold=export_settings.get("min_confidence_threshold", 0.3),
        view_generation_mode=ViewGenerationMode(export_settings.get("view_generation_mode", "precomputed")),
        enable_compression=export_settings.get("enable_compression", False),
        pretty_print=export_settings.get("pretty_print", True)
    ) 