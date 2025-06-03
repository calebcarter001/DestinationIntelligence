import json
import os
from datetime import datetime
from typing import Dict, Any, List, Optional
import logging

from .enhanced_data_models import Destination, Theme, Evidence, TemporalSlice
from .confidence_scoring import ConfidenceBreakdown, ConfidenceLevel

class JsonExportManager:
    """Manages JSON export of destination insights with full enhanced data"""
    
    def __init__(self, export_base_path: str = "destination_insights"):
        """
        Initialize JSON export manager
        
        Args:
            export_base_path: Base directory for JSON exports
        """
        self.export_base_path = export_base_path
        self.logger = logging.getLogger(__name__)
        
        # Create base export directory
        os.makedirs(export_base_path, exist_ok=True)
        
        # Create subdirectories for organization
        self.paths = {
            "full": os.path.join(export_base_path, "full_insights"),
            "summary": os.path.join(export_base_path, "summaries"),
            "themes": os.path.join(export_base_path, "themes"),
            "evidence": os.path.join(export_base_path, "evidence"),
            "temporal": os.path.join(export_base_path, "temporal"),
            "archive": os.path.join(export_base_path, "archive")
        }
        
        for path in self.paths.values():
            os.makedirs(path, exist_ok=True)
            
    def export_destination_insights(
        self, 
        destination: Destination,
        analysis_metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, str]:
        """
        Export comprehensive destination insights to JSON files
        
        Args:
            destination: Destination object with all enhanced data
            analysis_metadata: Additional metadata about the analysis
            
        Returns:
            Dictionary of file paths created
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        destination_id_safe = destination.id.replace("/", "_").replace(":", "_")
        
        created_files = {}
        
        try:
            # 1. Full comprehensive export
            full_export = self._create_full_export(destination, analysis_metadata)
            full_path = os.path.join(
                self.paths["full"],
                f"{destination_id_safe}_full_{timestamp}.json"
            )
            self._save_json(full_export, full_path)
            created_files["full"] = full_path
            
            # 2. Executive summary
            summary = self._create_executive_summary(destination)
            summary_path = os.path.join(
                self.paths["summary"],
                f"{destination_id_safe}_summary_{timestamp}.json"
            )
            self._save_json(summary, summary_path)
            created_files["summary"] = summary_path
            
            # 3. Themes-only export with confidence details
            themes_export = self._create_themes_export(destination)
            themes_path = os.path.join(
                self.paths["themes"],
                f"{destination_id_safe}_themes_{timestamp}.json"
            )
            self._save_json(themes_export, themes_path)
            created_files["themes"] = themes_path
            
            # 4. Evidence lineage export
            evidence_export = self._create_evidence_export(destination)
            evidence_path = os.path.join(
                self.paths["evidence"],
                f"{destination_id_safe}_evidence_{timestamp}.json"
            )
            self._save_json(evidence_export, evidence_path)
            created_files["evidence"] = evidence_path
            
            # 5. Temporal analysis export
            if destination.temporal_slices:
                temporal_export = self._create_temporal_export(destination)
                temporal_path = os.path.join(
                    self.paths["temporal"],
                    f"{destination_id_safe}_temporal_{timestamp}.json"
                )
                self._save_json(temporal_export, temporal_path)
                created_files["temporal"] = temporal_path
            
            # 6. Create latest symlinks for easy access
            self._create_latest_links(destination_id_safe, created_files)
            
            self.logger.info(f"Successfully exported {len(created_files)} JSON files for {destination.names[0]}")
            
        except Exception as e:
            self.logger.error(f"Error exporting destination insights: {e}")
            raise
            
        return created_files
    
    def _create_full_export(
        self, 
        destination: Destination,
        analysis_metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Create comprehensive export with all data"""
        export = {
            "export_metadata": {
                "version": "2.0",
                "export_timestamp": datetime.now().isoformat(),
                "destination_id": destination.id,
                "destination_revision": destination.destination_revision,
                "export_type": "full_comprehensive"
            },
            "destination": {
                "id": destination.id,
                "names": destination.names,
                "primary_name": destination.names[0] if destination.names else "Unknown",
                "admin_levels": destination.admin_levels,
                "country_code": destination.country_code,
                "timezone": destination.timezone,
                "population": destination.population,
                "geography": destination.core_geo,
                "last_updated": destination.last_updated.isoformat()
            },
            "themes": self._format_themes_with_evidence(destination.themes),
            "dimensions": self._format_dimensions(destination.dimensions),
            "temporal_analysis": self._format_temporal_slices(destination.temporal_slices),
            "points_of_interest": self._format_pois(destination.pois),
            "authentic_insights": [ai.to_dict() for ai in destination.authentic_insights],
            "local_authorities": [la.to_dict() for la in destination.local_authorities],
            "quality_metrics": self._calculate_quality_metrics(destination),
            "lineage": destination.lineage,
            "metadata": destination.meta
        }
        
        if analysis_metadata:
            export["analysis_metadata"] = analysis_metadata
            
        return export
    
    def _create_executive_summary(self, destination: Destination) -> Dict[str, Any]:
        """Create executive summary for quick overview"""
        verified_themes = [t for t in destination.themes
                           if t.get_confidence_level() in [ConfidenceLevel.HIGH, ConfidenceLevel.VERY_HIGH]]
        
        top_dimensions = self._get_top_dimensions(destination.dimensions, n=10)
        
        summary = {
            "export_metadata": {
                "version": "2.0",
                "export_timestamp": datetime.now().isoformat(),
                "export_type": "executive_summary"
            },
            "destination": {
                "name": destination.names[0] if destination.names else "Unknown",
                "country": destination.admin_levels.get("country", "Unknown"),
                "id": destination.id
            },
            "key_insights": {
                "total_themes": len(destination.themes),
                "verified_themes": len(verified_themes),
                "top_themes": [
                    {
                        "name": t.name,
                        "category": t.macro_category,
                        "confidence": t.confidence_breakdown.overall_confidence if t.confidence_breakdown else 0,
                        "evidence_count": len(t.evidence)
                    }
                    for t in sorted(verified_themes, 
                                  key=lambda x: x.confidence_breakdown.overall_confidence if x.confidence_breakdown else 0,
                                  reverse=True)[:5]
                ],
                "destination_strengths": top_dimensions,
                "unique_characteristics": self._extract_unique_characteristics(destination),
                "seasonal_highlights": self._extract_seasonal_highlights(destination)
            },
            "data_quality": {
                "evidence_sources": self._count_evidence_by_type(destination),
                "average_confidence": self._calculate_average_confidence(destination.themes),
                "cultural_diversity_score": self._calculate_cultural_diversity(destination)
            }
        }
        
        return summary
    
    def _create_themes_export(self, destination: Destination) -> Dict[str, Any]:
        """Create detailed themes export with confidence breakdowns"""
        themes_by_category = {}
        
        for theme in destination.themes:
            category = theme.macro_category
            if category not in themes_by_category:
                themes_by_category[category] = []
                
            theme_data = {
                "theme_id": theme.theme_id,
                "name": theme.name,
                "micro_category": theme.micro_category,
                "description": theme.description,
                "fit_score": theme.fit_score,
                "confidence_level": theme.get_confidence_level().value,
                "confidence_breakdown": theme.confidence_breakdown.to_dict() if theme.confidence_breakdown else None,
                "evidence_count": len(theme.evidence),
                "evidence_distribution": self._get_evidence_distribution(theme.evidence),
                "tags": theme.tags,
                "created": theme.created_date.isoformat(),
                "last_validated": theme.last_validated.isoformat() if theme.last_validated else None,
                "authentic_insights": [ai.to_dict() for ai in theme.authentic_insights],
                "local_authorities": [la.to_dict() for la in theme.local_authorities],
                "seasonal_relevance": theme.seasonal_relevance,
                "regional_uniqueness": theme.regional_uniqueness,
                "insider_tips": theme.insider_tips,
                "factors": theme.factors,
                "cultural_summary": theme.cultural_summary,
                "sentiment_analysis": theme.sentiment_analysis,
                "temporal_analysis": theme.temporal_analysis
            }
            
            themes_by_category[category].append(theme_data)
        
        return {
            "export_metadata": {
                "version": "2.0",
                "export_timestamp": datetime.now().isoformat(),
                "export_type": "themes_detailed",
                "destination_id": destination.id
            },
            "themes_by_category": themes_by_category,
            "theme_statistics": {
                "total_themes": len(destination.themes),
                "themes_by_confidence": self._count_themes_by_confidence(destination.themes),
                "categories": list(themes_by_category.keys()),
                "average_evidence_per_theme": sum(len(t.evidence) for t in destination.themes) / len(destination.themes) if destination.themes else 0
            }
        }
    
    def _create_evidence_export(self, destination: Destination) -> Dict[str, Any]:
        """Create evidence lineage export for transparency"""
        all_evidence = []
        evidence_by_source = {}
        
        for theme in destination.themes:
            for evidence in theme.evidence:
                evidence_data = {
                    "id": evidence.id,
                    "theme": theme.name,
                    "source_url": evidence.source_url,
                    "source_category": evidence.source_category.value,
                    "evidence_type": evidence.evidence_type.value,
                    "authority_weight": evidence.authority_weight,
                    "text_snippet": evidence.text_snippet[:500] + "..." if len(evidence.text_snippet) > 500 else evidence.text_snippet,
                    "confidence": evidence.confidence,
                    "timestamp": evidence.timestamp.isoformat(),
                    "cultural_context": evidence.cultural_context,
                    "agent_id": evidence.agent_id,
                    "sentiment": evidence.sentiment,
                    "published_date": evidence.published_date.isoformat() if evidence.published_date else None,
                    "relationships": evidence.relationships
                }
                
                all_evidence.append(evidence_data)
                
                # Group by source
                source = evidence.source_category.value
                if source not in evidence_by_source:
                    evidence_by_source[source] = []
                evidence_by_source[source].append(evidence_data)
        
        return {
            "export_metadata": {
                "version": "2.0",
                "export_timestamp": datetime.now().isoformat(),
                "export_type": "evidence_lineage",
                "destination_id": destination.id
            },
            "evidence_summary": {
                "total_evidence": len(all_evidence),
                "evidence_by_source_type": {k: len(v) for k, v in evidence_by_source.items()},
                "authority_distribution": self._calculate_authority_distribution(destination)
            },
            "evidence_by_source": evidence_by_source,
            "all_evidence": all_evidence
        }
    
    def _create_temporal_export(self, destination: Destination) -> Dict[str, Any]:
        """Create temporal analysis export"""
        temporal_data = []
        
        for slice in destination.temporal_slices:
            slice_data = {
                "valid_from": slice.valid_from.isoformat(),
                "valid_to": slice.valid_to.isoformat() if slice.valid_to else "current",
                "is_current": slice.is_current(),
                "season": slice.season,
                "seasonal_highlights": slice.seasonal_highlights,
                "special_events": slice.special_events,
                "weather_patterns": slice.weather_patterns,
                "visitor_patterns": slice.visitor_patterns
            }
            temporal_data.append(slice_data)
        
        return {
            "export_metadata": {
                "version": "2.0",
                "export_timestamp": datetime.now().isoformat(),
                "export_type": "temporal_analysis",
                "destination_id": destination.id
            },
            "temporal_slices": temporal_data,
            "seasonal_summary": self._create_seasonal_summary(destination.temporal_slices)
        }
    
    def _format_themes_with_evidence(self, themes: List[Theme]) -> List[Dict[str, Any]]:
        """Format themes and their evidence for export"""
        formatted_themes = []
        for theme in themes:
            theme_dict = theme.to_dict() # Use the to_dict method from the Theme dataclass
            
            # Evidence is already formatted by Theme.to_dict() if needed, but let's confirm
            # If evidence objects themselves need to be formatted beyond simple dict, do it here.
            # For now, assuming Evidence.to_dict() is called within Theme.to_dict() for nested objects.
            
            formatted_themes.append(theme_dict)
        return formatted_themes
    
    def _format_dimensions(self, dimensions: Dict[str, Any]) -> Dict[str, Any]:
        """Format dimensions for export"""
        formatted_dimensions = {}
        for name, dv in dimensions.items():
            if dv.value is not None:
                formatted_dimensions[name] = {
                    "value": dv.value,
                    "unit": dv.unit,
                    "confidence": dv.confidence,
                    "last_updated": dv.last_updated.isoformat(),
                    "evidence_count": len(dv.source_evidence_ids)
                }
                
        return formatted_dimensions
    
    def _format_temporal_slices(self, slices: List[TemporalSlice]) -> List[Dict[str, Any]]:
        """Format temporal slices"""
        return [
            {
                "valid_from": s.valid_from.isoformat(),
                "valid_to": s.valid_to.isoformat() if s.valid_to else "current",
                "season": s.season,
                "highlights": s.seasonal_highlights,
                "events": s.special_events
            }
            for s in slices
        ]
    
    def _format_pois(self, pois: List[Any]) -> List[Dict[str, Any]]:
        """Format POIs"""
        return [
            {
                "id": poi.poi_id,
                "name": poi.name,
                "type": poi.poi_type,
                "location": poi.location,
                "theme_associations": poi.theme_tags,
                "accessibility": {
                    "ada_accessible": poi.ada_accessible,
                    "features": poi.ada_features
                }
            }
            for poi in pois
        ]
    
    def _calculate_quality_metrics(self, destination: Destination) -> Dict[str, Any]:
        """Calculate overall quality metrics"""
        total_evidence = sum(len(t.evidence) for t in destination.themes)
        
        confidence_scores = [
            t.confidence_breakdown.overall_confidence 
            for t in destination.themes 
            if t.confidence_breakdown
        ]
        
        return {
            "total_themes": len(destination.themes),
            "total_evidence": total_evidence,
            "average_confidence": sum(confidence_scores) / len(confidence_scores) if confidence_scores else 0,
            "high_confidence_themes": len([
                t for t in destination.themes 
                if t.get_confidence_level() in [ConfidenceLevel.HIGH, ConfidenceLevel.VERY_HIGH]
            ]),
            "dimensions_populated": len([d for d in destination.dimensions.values() if d.value is not None]),
            "temporal_coverage": len(destination.temporal_slices),
            "poi_count": len(destination.pois)
        }
    
    def _get_top_dimensions(self, dimensions: Dict[str, Any], n: int = 10) -> List[Dict[str, Any]]:
        """Get top N dimensions by value and confidence"""
        valid_dims = [
            {
                "name": name,
                "value": dim.value,
                "confidence": dim.confidence
            }
            for name, dim in dimensions.items()
            if dim.value is not None and dim.confidence > 0.5
        ]
        
        # Sort by value * confidence
        return sorted(
            valid_dims, 
            key=lambda x: x["value"] * x["confidence"], 
            reverse=True
        )[:n]
    
    def _extract_unique_characteristics(self, destination: Destination) -> List[str]:
        """Extract unique characteristics from high-confidence themes"""
        characteristics = []
        
        for theme in destination.themes:
            if theme.get_confidence_level() in [ConfidenceLevel.HIGH, ConfidenceLevel.VERY_HIGH]:
                if theme.fit_score > 0.8:  # High relevance to destination
                    characteristics.append(f"{theme.name} ({theme.macro_category})")
                    
        return characteristics[:10]  # Top 10
    
    def _extract_seasonal_highlights(self, destination: Destination) -> Dict[str, List[str]]:
        """Extract seasonal highlights from temporal data"""
        highlights = {}
        
        for slice in destination.temporal_slices:
            if slice.season and slice.seasonal_highlights:
                highlights[slice.season] = [
                    f"{k}: {v}" 
                    for k, v in slice.seasonal_highlights.items()
                ][:3]  # Top 3 per season
                
        return highlights
    
    def _count_evidence_by_type(self, destination: Destination) -> Dict[str, int]:
        """Count evidence by source type"""
        counts = {}
        
        for theme in destination.themes:
            for evidence in theme.evidence:
                source_type = evidence.evidence_type.value
                counts[source_type] = counts.get(source_type, 0) + 1
                
        return counts
    
    def _calculate_average_confidence(self, themes: List[Theme]) -> float:
        """Calculate average confidence across themes"""
        scores = [
            t.confidence_breakdown.overall_confidence 
            for t in themes 
            if t.confidence_breakdown
        ]
        
        return sum(scores) / len(scores) if scores else 0.0
    
    def _calculate_cultural_diversity(self, destination: Destination) -> float:
        """Calculate cultural diversity score based on evidence sources"""
        local_sources = 0
        total_sources = 0
        
        for theme in destination.themes:
            for evidence in theme.evidence:
                total_sources += 1
                if evidence.cultural_context and evidence.cultural_context.get("is_local_source"):
                    local_sources += 1
                    
        if total_sources == 0:
            return 0.0
            
        local_ratio = local_sources / total_sources
        # Optimal is 60% local, 40% international
        return 1.0 - abs(0.6 - local_ratio)
    
    def _get_evidence_distribution(self, evidence_list: List[Evidence]) -> Dict[str, int]:
        """Get distribution of evidence by source category"""
        distribution = {}
        
        for evidence in evidence_list:
            category = evidence.source_category.value
            distribution[category] = distribution.get(category, 0) + 1
            
        return distribution
    
    def _count_themes_by_confidence(self, themes: List[Theme]) -> Dict[str, int]:
        """Count themes by confidence level"""
        counts = {}
        
        for theme in themes:
            level = theme.get_confidence_level().value
            counts[level] = counts.get(level, 0) + 1
            
        return counts
    
    def _calculate_authority_distribution(self, destination: Destination) -> Dict[str, float]:
        """Calculate distribution of evidence by authority weight"""
        ranges = {
            "high (>0.8)": 0,
            "medium (0.5-0.8)": 0,
            "low (<0.5)": 0
        }
        
        total = 0
        for theme in destination.themes:
            for evidence in theme.evidence:
                total += 1
                if evidence.authority_weight > 0.8:
                    ranges["high (>0.8)"] += 1
                elif evidence.authority_weight >= 0.5:
                    ranges["medium (0.5-0.8)"] += 1
                else:
                    ranges["low (<0.5)"] += 1
                    
        # Convert to percentages
        if total > 0:
            for key in ranges:
                ranges[key] = round((ranges[key] / total) * 100, 1)
                
        return ranges
    
    def _create_seasonal_summary(self, slices: List[TemporalSlice]) -> Dict[str, Any]:
        """Create seasonal summary from temporal slices"""
        seasons = {}
        
        for slice in slices:
            if slice.season:
                if slice.season not in seasons:
                    seasons[slice.season] = {
                        "occurrences": 0,
                        "highlights": [],
                        "events": []
                    }
                    
                seasons[slice.season]["occurrences"] += 1
                
                if slice.seasonal_highlights:
                    seasons[slice.season]["highlights"].extend(list(slice.seasonal_highlights.keys()))
                    
                if slice.special_events:
                    seasons[slice.season]["events"].extend(slice.special_events)
                    
        return seasons
    
    def _save_json(self, data: Dict[str, Any], filepath: str):
        """Save data to JSON file with pretty formatting"""
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False, default=str)
            
    def _create_latest_links(self, destination_id: str, created_files: Dict[str, str]):
        """Create 'latest' symlinks for easy access"""
        for file_type, filepath in created_files.items():
            latest_link = os.path.join(
                os.path.dirname(filepath),
                f"{destination_id}_latest.json"
            )
            
            # Remove existing symlink if it exists
            if os.path.islink(latest_link):
                os.unlink(latest_link)
                
            # Create new symlink
            try:
                os.symlink(os.path.basename(filepath), latest_link)
            except OSError:
                # Fall back to copying on systems that don't support symlinks
                import shutil
                shutil.copy2(filepath, latest_link)
    
    def archive_old_exports(self, days_to_keep: int = 30):
        """Archive exports older than specified days"""
        archive_date = datetime.now().timestamp() - (days_to_keep * 24 * 60 * 60)
        
        for directory in [self.paths["full"], self.paths["summary"], self.paths["themes"]]:
            for filename in os.listdir(directory):
                filepath = os.path.join(directory, filename)
                
                # Skip symlinks
                if os.path.islink(filepath):
                    continue
                    
                if os.path.getmtime(filepath) < archive_date:
                    # Move to archive
                    archive_path = os.path.join(self.paths["archive"], filename)
                    os.rename(filepath, archive_path)
                    self.logger.info(f"Archived old export: {filename}") 