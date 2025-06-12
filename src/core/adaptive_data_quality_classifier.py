"""
Adaptive Data Quality Classifier
Analyzes evidence and content to determine data quality level for adaptive processing
"""

import logging
from typing import Dict, Any, List, Optional
from collections import Counter
from urllib.parse import urlparse
import re
from .safe_dict_utils import safe_get_nested

logger = logging.getLogger(__name__)

class AdaptiveDataQualityClassifier:
    """
    Classifies data quality for adaptive processing based on evidence characteristics
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the classifier with configuration settings
        
        Args:
            config: Full application configuration dictionary
        """
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Extract adaptive configuration
        self.heuristics_config = config.get("data_quality_heuristics", {})
        self.fallback_config = config.get("fallback_behavior", {})
        self.override_config = config.get("destination_overrides", {})
        
        # Default thresholds if config is missing
        self.rich_data_indicators = self.heuristics_config.get("rich_data_indicators", {
            "min_evidence_count": 75,
            "min_source_diversity": 4,
            "min_high_authority_ratio": 0.3,
            "min_content_volume": 15000,
            "min_theme_discovery_rate": 25,
            "min_unique_sources": 5
        })
        
        self.poor_data_indicators = self.heuristics_config.get("poor_data_indicators", {
            "max_evidence_count": 30,
            "max_source_diversity": 2,
            "max_high_authority_ratio": 0.1,
            "max_content_volume": 5000,
            "max_theme_discovery_rate": 8,
            "max_unique_sources": 2
        })
        
        # High authority domains for quality assessment
        processing_config = config.get("processing_settings", {})
        priority_config = safe_get_nested(processing_config, ["content_intelligence", "priority_extraction"], {})
        self.high_authority_domains = priority_config.get("high_authority_domains", [])
        self.medium_authority_domains = priority_config.get("medium_authority_domains", [])
        
    def classify_data_quality(
        self,
        destination_name: str,
        evidence_list: List[Any],
        content_list: List[Dict[str, Any]],
        discovered_themes_count: int,
        analysis_metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Classify data quality level for a destination
        
        Args:
            destination_name: Name of the destination
            evidence_list: List of evidence objects
            content_list: List of content dictionaries
            discovered_themes_count: Number of themes discovered before filtering
            analysis_metadata: Additional metadata about the analysis
            
        Returns:
            Dictionary with classification results and metrics
        """
        
        # Check for manual overrides first
        override_result = self._check_manual_overrides(destination_name)
        if override_result:
            return override_result
        
        # Calculate quality metrics
        metrics = self._calculate_quality_metrics(
            evidence_list, content_list, discovered_themes_count
        )
        
        # Classify based on metrics
        classification = self._classify_from_metrics(metrics)
        
        # Build result
        result = {
            "classification": classification,
            "confidence": metrics["classification_confidence"],
            "metrics": metrics,
            "reasoning": self._generate_reasoning(classification, metrics),
            "adaptive_settings": self._get_adaptive_settings(classification)
        }
        
        self.logger.info(f"Data quality classification for '{destination_name}': {classification} "
                        f"(confidence: {metrics['classification_confidence']:.2f})")
        self.logger.debug(f"Classification metrics: {metrics}")
        
        return result
    
    def _check_manual_overrides(self, destination_name: str) -> Optional[Dict[str, Any]]:
        """Check if destination matches any manual override patterns"""
        
        if not self.override_config.get("enabled", False):
            return None
        
        name_lower = destination_name.lower()
        
        # Check major cities
        major_cities = self.override_config.get("major_cities", {})
        if major_cities.get("patterns"):
            for pattern in major_cities["patterns"]:
                if pattern.lower() in name_lower:
                    classification = major_cities.get("force_classification", "rich_data")
                    return {
                        "classification": classification,
                        "confidence": 1.0,
                        "metrics": {"override_applied": True, "override_pattern": pattern},
                        "reasoning": f"Manual override: matches major city pattern '{pattern}'",
                        "adaptive_settings": self._get_adaptive_settings(classification)
                    }
        
        # Check small towns
        small_towns = self.override_config.get("small_towns", {})
        if small_towns.get("patterns"):
            for pattern in small_towns["patterns"]:
                if pattern.lower() in name_lower:
                    classification = small_towns.get("force_classification", "poor_data")
                    return {
                        "classification": classification,
                        "confidence": 1.0,
                        "metrics": {"override_applied": True, "override_pattern": pattern},
                        "reasoning": f"Manual override: matches small town pattern '{pattern}'",
                        "adaptive_settings": self._get_adaptive_settings(classification)
                    }
        
        # Check tourist hotspots
        hotspots = self.override_config.get("tourist_hotspots", {})
        if hotspots.get("patterns"):
            for pattern in hotspots["patterns"]:
                if pattern.lower() in name_lower:
                    classification = hotspots.get("force_classification", "rich_data")
                    return {
                        "classification": classification,
                        "confidence": 1.0,
                        "metrics": {"override_applied": True, "override_pattern": pattern},
                        "reasoning": f"Manual override: matches tourist hotspot pattern '{pattern}'",
                        "adaptive_settings": self._get_adaptive_settings(classification)
                    }
        
        return None
    
    def _calculate_quality_metrics(
        self,
        evidence_list: List[Any],
        content_list: List[Dict[str, Any]],
        discovered_themes_count: int
    ) -> Dict[str, Any]:
        """Calculate quality metrics from evidence and content"""
        
        metrics = {}
        
        # Evidence count
        metrics["evidence_count"] = len(evidence_list)
        
        # Source diversity
        source_domains = set()
        high_authority_sources = 0
        
        for evidence in evidence_list:
            # Handle both Evidence objects and dictionaries
            if hasattr(evidence, 'source_url'):  # Evidence object
                url = evidence.source_url
            else:  # Dictionary
                url = evidence.get('source_url', '')
            
            domain = self._extract_domain(url)
            if domain:
                source_domains.add(domain)
                
                # Check authority level
                if self._is_high_authority_domain(domain):
                    high_authority_sources += 1
        
        metrics["source_diversity"] = len(source_domains)
        metrics["unique_sources"] = len(source_domains)
        metrics["high_authority_ratio"] = (
            high_authority_sources / len(evidence_list) if evidence_list else 0
        )
        
        # Content volume
        total_content_volume = 0
        for content in content_list:
            content_text = content.get("content", "")
            if isinstance(content_text, str):
                total_content_volume += len(content_text)
        
        metrics["content_volume"] = total_content_volume
        
        # Theme discovery rate
        metrics["theme_discovery_rate"] = discovered_themes_count
        
        # Calculate classification confidence
        metrics["classification_confidence"] = self._calculate_classification_confidence(metrics)
        
        return metrics
    
    def _classify_from_metrics(self, metrics: Dict[str, Any]) -> str:
        """Classify data quality based on calculated metrics"""
        
        rich_score = 0
        poor_score = 0
        
        # Evidence count scoring
        if metrics["evidence_count"] >= self.rich_data_indicators["min_evidence_count"]:
            rich_score += 1
        elif metrics["evidence_count"] <= self.poor_data_indicators["max_evidence_count"]:
            poor_score += 1
        
        # Source diversity scoring
        if metrics["source_diversity"] >= self.rich_data_indicators["min_source_diversity"]:
            rich_score += 1
        elif metrics["source_diversity"] <= self.poor_data_indicators["max_source_diversity"]:
            poor_score += 1
        
        # High authority ratio scoring
        if metrics["high_authority_ratio"] >= self.rich_data_indicators["min_high_authority_ratio"]:
            rich_score += 1
        elif metrics["high_authority_ratio"] <= self.poor_data_indicators["max_high_authority_ratio"]:
            poor_score += 1
        
        # Content volume scoring
        if metrics["content_volume"] >= self.rich_data_indicators["min_content_volume"]:
            rich_score += 1
        elif metrics["content_volume"] <= self.poor_data_indicators["max_content_volume"]:
            poor_score += 1
        
        # Theme discovery rate scoring
        if metrics["theme_discovery_rate"] >= self.rich_data_indicators["min_theme_discovery_rate"]:
            rich_score += 1
        elif metrics["theme_discovery_rate"] <= self.poor_data_indicators["max_theme_discovery_rate"]:
            poor_score += 1
        
        # Unique sources scoring (additional check)
        if metrics["unique_sources"] >= self.rich_data_indicators["min_unique_sources"]:
            rich_score += 1
        elif metrics["unique_sources"] <= self.poor_data_indicators["max_unique_sources"]:
            poor_score += 1
        
        # Classification logic
        confidence_threshold = self.fallback_config.get("classification_confidence_threshold", 0.7)
        
        if rich_score >= 4 and metrics["classification_confidence"] >= confidence_threshold:
            return "rich_data"
        elif poor_score >= 4 and metrics["classification_confidence"] >= confidence_threshold:
            return "poor_data"
        else:
            # Fallback to medium data or configured fallback
            return self.fallback_config.get("unknown_data_quality", "medium_data")
    
    def _calculate_classification_confidence(self, metrics: Dict[str, Any]) -> float:
        """Calculate confidence in the classification"""
        
        # Simple confidence calculation based on how extreme the metrics are
        evidence_confidence = 0.0
        if metrics["evidence_count"] >= self.rich_data_indicators["min_evidence_count"]:
            evidence_confidence = min(metrics["evidence_count"] / 100, 1.0)
        elif metrics["evidence_count"] <= self.poor_data_indicators["max_evidence_count"]:
            evidence_confidence = 1.0 - (metrics["evidence_count"] / 50)
        else:
            evidence_confidence = 0.5
        
        # Authority confidence
        authority_confidence = min(metrics["high_authority_ratio"] * 2, 1.0)
        
        # Source diversity confidence
        diversity_confidence = min(metrics["source_diversity"] / 6, 1.0)
        
        # Overall confidence (average of components)
        overall_confidence = (evidence_confidence + authority_confidence + diversity_confidence) / 3
        
        return max(0.0, min(1.0, overall_confidence))
    
    def _generate_reasoning(self, classification: str, metrics: Dict[str, Any]) -> str:
        """Generate human-readable reasoning for the classification"""
        
        reasoning_parts = []
        
        if classification == "rich_data":
            reasoning_parts.append("Rich data classification based on:")
            if metrics["evidence_count"] >= self.rich_data_indicators["min_evidence_count"]:
                reasoning_parts.append(f"• High evidence count ({metrics['evidence_count']})")
            if metrics["source_diversity"] >= self.rich_data_indicators["min_source_diversity"]:
                reasoning_parts.append(f"• Good source diversity ({metrics['source_diversity']} sources)")
            if metrics["high_authority_ratio"] >= self.rich_data_indicators["min_high_authority_ratio"]:
                reasoning_parts.append(f"• High authority sources ({metrics['high_authority_ratio']:.1%})")
        
        elif classification == "poor_data":
            reasoning_parts.append("Poor data classification based on:")
            if metrics["evidence_count"] <= self.poor_data_indicators["max_evidence_count"]:
                reasoning_parts.append(f"• Limited evidence count ({metrics['evidence_count']})")
            if metrics["source_diversity"] <= self.poor_data_indicators["max_source_diversity"]:
                reasoning_parts.append(f"• Low source diversity ({metrics['source_diversity']} sources)")
            if metrics["high_authority_ratio"] <= self.poor_data_indicators["max_high_authority_ratio"]:
                reasoning_parts.append(f"• Few authority sources ({metrics['high_authority_ratio']:.1%})")
        
        else:
            reasoning_parts.append("Medium data classification:")
            reasoning_parts.append("• Metrics fall between rich and poor thresholds")
            reasoning_parts.append(f"• Evidence: {metrics['evidence_count']}, Sources: {metrics['source_diversity']}")
        
        return " ".join(reasoning_parts)
    
    def _get_adaptive_settings(self, classification: str) -> Dict[str, Any]:
        """Get adaptive settings for the given classification"""
        
        config_key = f"{classification}_"
        
        # Extract relevant settings from config
        export_settings = self.config.get("export_settings", {})
        theme_settings = self.config.get("theme_management", {})
        evidence_settings = self.config.get("evidence_filtering", {})
        semantic_settings = self.config.get("semantic_processing", {})
        output_settings = self.config.get("output_control", {})
        
        return {
            "export_mode": export_settings.get(f"{config_key}mode", "themes_focused"),
            "confidence_threshold": export_settings.get(f"{config_key}confidence", 0.55),
            "max_evidence_per_theme": export_settings.get(f"{config_key}max_evidence_per_theme", 5),
            "max_themes": theme_settings.get(f"{config_key}max_themes", 35),
            "min_authority": safe_get_nested(evidence_settings, ["adaptive_quality_thresholds", f"{config_key}min_authority"], 0.5),
            "semantic_intensive": semantic_settings.get(f"{config_key}semantic_intensive", True),
            "output_priority": output_settings.get(f"{config_key}database_priority", False)
        }
    
    def _extract_domain(self, url: str) -> Optional[str]:
        """Extract domain from URL"""
        try:
            parsed = urlparse(url)
            domain = parsed.netloc.lower()
            # Remove www. prefix
            if domain.startswith('www.'):
                domain = domain[4:]
            return domain
        except:
            return None
    
    def _is_high_authority_domain(self, domain: str) -> bool:
        """Check if domain is considered high authority"""
        
        # Check exact matches
        if domain in self.high_authority_domains:
            return True
        
        # Check if domain ends with any high authority domain
        for auth_domain in self.high_authority_domains:
            if domain.endswith(auth_domain):
                return True
        
        # Check medium authority domains as partial high authority
        for auth_domain in self.medium_authority_domains:
            if domain.endswith(auth_domain):
                return True
        
        return False
    
    def get_classification_summary(self) -> Dict[str, Any]:
        """Get summary of classification thresholds and settings"""
        
        return {
            "enabled": self.heuristics_config.get("enabled", True),
            "rich_data_thresholds": self.rich_data_indicators,
            "poor_data_thresholds": self.poor_data_indicators,
            "override_patterns": {
                key: value.get("patterns", []) 
                for key, value in self.override_config.items() 
                if isinstance(value, dict) and "patterns" in value
            }
        } 