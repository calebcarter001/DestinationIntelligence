"""
Specialized agents for destination intelligence validation
"""

import logging
from typing import Dict, List, Optional, Any
from datetime import datetime
import re
from collections import Counter

from .base_agent import BaseAgent, MessageBroker, AgentMessage, MessageType
from ..core.evidence_hierarchy import EvidenceHierarchy, SourceCategory
from ..core.confidence_scoring import ConfidenceScorer, ConfidenceBreakdown, ConfidenceLevel
from ..core.enhanced_data_models import Evidence, Theme, Destination

class ValidationAgent(BaseAgent):
    """
    Agent responsible for validating themes and insights
    
    Applies confidence formula and emits contradictions
    """
    
    def __init__(self, agent_id: str = "validation-001", config: Optional[Dict[str, Any]] = None):
        super().__init__(agent_id, "ValidationAgent", config)
        self.register_handler(MessageType.VALIDATION_REQUEST, self._handle_validation_request)
        
    async def execute_task(self, task_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate themes for a destination
        
        Args:
            task_data: Should contain 'destination' and 'themes'
        """
        destination_name = task_data.get("destination_name", "Unknown")
        themes = task_data.get("themes", [])
        destination_country_code = task_data.get("country_code")
        
        self.logger.info(f"Validating {len(themes)} themes for {destination_name}")
        
        validated_themes = []
        contradictions = []
        
        for theme_data in themes:
            # Extract evidence
            evidence_sources = theme_data.get("evidence_sources", [])
            evidence_texts = theme_data.get("evidence_texts", [])
            sentiment_scores = theme_data.get("sentiment_scores")
            
            # Calculate confidence
            confidence_breakdown = ConfidenceScorer.calculate_confidence(
                evidence_sources=evidence_sources,
                evidence_texts=evidence_texts,
                destination_country_code=destination_country_code,
                sentiment_scores=sentiment_scores
            )
            
            # Check for contradictions
            if confidence_breakdown.consistency < 0.5:
                contradictions.append({
                    "theme": theme_data.get("name"),
                    "reason": "Low consistency score indicates conflicting evidence",
                    "consistency_score": confidence_breakdown.consistency,
                    "evidence_snippets": evidence_texts[:3]  # First 3 as examples
                })
            
            # Add confidence to theme
            theme_data["confidence_breakdown"] = confidence_breakdown.to_dict()
            theme_data["confidence_level"] = confidence_breakdown.confidence_level.value
            theme_data["is_validated"] = confidence_breakdown.total_confidence >= 0.2
            
            validated_themes.append(theme_data)
            
        # Emit contradictions if found
        if contradictions:
            contradiction_msg = self.create_message(
                MessageType.CONTRADICTION,
                {
                    "destination_name": destination_name,
                    "contradictions": contradictions,
                    "timestamp": datetime.now().isoformat()
                }
            )
            # In real implementation, this would be published to the broker
            
        return {
            "validated_themes": validated_themes,
            "total_themes": len(themes),
            "validated_count": sum(1 for t in validated_themes if t["is_validated"]),
            "contradictions_found": len(contradictions),
            "validation_timestamp": datetime.now().isoformat()
        }
    
    async def _handle_validation_request(self, message: AgentMessage) -> Optional[AgentMessage]:
        """Handle validation request messages"""
        result = await self.execute_task(message.payload)
        
        return self.create_message(
            MessageType.VALIDATION_RESPONSE,
            result,
            recipient_id=message.sender_id,
            correlation_id=message.id
        )

class CulturalPerspectiveAgent(BaseAgent):
    """
    Agent that prioritizes local-language and locally owned sources
    
    Enhances evidence with cultural context
    """
    
    def __init__(self, agent_id: str = "cultural-001", config: Optional[Dict[str, Any]] = None):
        super().__init__(agent_id, "CulturalPerspectiveAgent", config)
        
        # Language indicators for different regions
        self.local_language_patterns = {
            "FR": ["french", "français", "fr.", ".fr"],
            "ES": ["spanish", "español", "castellano", ".es"],
            "IT": ["italian", "italiano", ".it"],
            "DE": ["german", "deutsch", ".de"],
            "JP": ["japanese", "日本語", ".jp", "nihongo"],
            "CN": ["chinese", "中文", "mandarin", ".cn"],
            "TH": ["thai", "ไทย", ".th"],
            "GR": ["greek", "ελληνικά", ".gr"],
            "PT": ["portuguese", "português", ".pt", ".br"],
            "RU": ["russian", "русский", ".ru"]
        }
        
    async def execute_task(self, task_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze sources for cultural perspective
        
        Args:
            task_data: Should contain 'sources' and 'destination_country_code'
        """
        sources = task_data.get("sources", [])
        country_code = task_data.get("country_code", "").upper()
        destination_name = task_data.get("destination_name", "Unknown")
        
        self.logger.info(f"Analyzing cultural perspective for {len(sources)} sources from {destination_name}")
        
        enhanced_sources = []
        local_source_count = 0
        language_distribution = Counter()
        
        for source in sources:
            url = source.get("url", "")
            content = source.get("content", "")
            
            # Determine if source is local
            is_local = EvidenceHierarchy.is_local_source(url, country_code)
            
            # Detect language
            detected_language = self._detect_language(url, content, country_code)
            language_distribution[detected_language] += 1
            
            # Check for local ownership indicators
            local_ownership = self._check_local_ownership(url, content)
            
            # Calculate cultural relevance score
            cultural_score = self._calculate_cultural_score(
                is_local, detected_language, local_ownership, country_code
            )
            
            if is_local:
                local_source_count += 1
                
            enhanced_source = {
                **source,
                "cultural_context": {
                    "is_local_source": is_local,
                    "detected_language": detected_language,
                    "local_ownership": local_ownership,
                    "cultural_relevance_score": cultural_score,
                    "analysis_timestamp": datetime.now().isoformat()
                }
            }
            
            enhanced_sources.append(enhanced_source)
            
        # Calculate overall cultural diversity
        cultural_diversity = len(language_distribution) / max(len(self.local_language_patterns), 1)
        local_source_ratio = local_source_count / len(sources) if sources else 0
        
        return {
            "enhanced_sources": enhanced_sources,
            "cultural_metrics": {
                "local_source_count": local_source_count,
                "local_source_ratio": local_source_ratio,
                "language_distribution": dict(language_distribution),
                "cultural_diversity_score": cultural_diversity,
                "optimal_mix_score": self._calculate_optimal_mix_score(local_source_ratio)
            },
            "recommendations": self._generate_recommendations(local_source_ratio, language_distribution)
        }
    
    def _detect_language(self, url: str, content: str, country_code: str) -> str:
        """Detect language of content"""
        combined_text = f"{url} {content}".lower()
        
        # Check for country-specific patterns
        if country_code in self.local_language_patterns:
            for pattern in self.local_language_patterns[country_code]:
                if pattern in combined_text:
                    return f"local_{country_code.lower()}"
                    
        # Default language detection based on common patterns
        if any(eng in combined_text for eng in ["english", "en.", "www.", ".com"]):
            return "english"
            
        return "unknown"
    
    def _check_local_ownership(self, url: str, content: str) -> bool:
        """Check indicators of local ownership"""
        local_indicators = [
            "family-owned", "locally owned", "local business",
            "established in", "founded by", "native",
            "generations", "traditional", "authentic"
        ]
        
        combined_text = f"{url} {content}".lower()
        return any(indicator in combined_text for indicator in local_indicators)
    
    def _calculate_cultural_score(
        self, is_local: bool, language: str, local_ownership: bool, country_code: str
    ) -> float:
        """Calculate cultural relevance score (0-1)"""
        score = 0.0
        
        if is_local:
            score += 0.4
            
        if language.startswith("local_"):
            score += 0.3
        elif language == "english":
            score += 0.1  # Some value for international accessibility
            
        if local_ownership:
            score += 0.3
            
        return min(score, 1.0)
    
    def _calculate_optimal_mix_score(self, local_ratio: float) -> float:
        """Calculate how close to optimal mix (60% local, 40% international)"""
        optimal_ratio = 0.6
        deviation = abs(local_ratio - optimal_ratio)
        
        # Score decreases as we deviate from optimal
        return max(0, 1 - (deviation / optimal_ratio))
    
    def _generate_recommendations(
        self, local_ratio: float, language_dist: Counter
    ) -> List[str]:
        """Generate recommendations for improving cultural perspective"""
        recommendations = []
        
        if local_ratio < 0.4:
            recommendations.append("Seek more local sources to improve authenticity")
        elif local_ratio > 0.8:
            recommendations.append("Include more international perspectives for balance")
            
        if len(language_dist) < 2:
            recommendations.append("Diversify language sources for richer perspective")
            
        return recommendations

class ContradictionDetectionAgent(BaseAgent):
    """
    Agent that detects and resolves contradictions in evidence
    
    Uses authority and recency to resolve conflicts
    """
    
    def __init__(self, agent_id: str = "contradiction-001", config: Optional[Dict[str, Any]] = None):
        super().__init__(agent_id, "ContradictionDetectionAgent", config)
        
        # Contradiction patterns
        self.contradiction_indicators = [
            ("safe", "dangerous"), ("clean", "dirty"), ("expensive", "cheap"),
            ("crowded", "empty"), ("modern", "traditional"), ("quiet", "noisy"),
            ("friendly", "unfriendly"), ("authentic", "touristy"),
            ("well-maintained", "run-down"), ("easy", "difficult")
        ]
        
    async def execute_task(self, task_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Detect contradictions in theme evidence
        
        Args:
            task_data: Should contain 'themes' with evidence
        """
        themes = task_data.get("themes", [])
        destination_name = task_data.get("destination_name", "Unknown")
        
        self.logger.info(f"Detecting contradictions for {len(themes)} themes in {destination_name}")
        
        all_contradictions = []
        resolved_themes = []
        
        for theme in themes:
            theme_name = theme.get("name", "Unknown Theme")
            evidence_list = theme.get("evidence", [])
            
            if len(evidence_list) < 2:
                resolved_themes.append(theme)
                continue
                
            # Detect contradictions
            contradictions = self._detect_contradictions(evidence_list)
            
            if contradictions:
                # Resolve using authority and recency
                resolution = self._resolve_contradictions(contradictions, evidence_list)
                
                all_contradictions.append({
                    "theme": theme_name,
                    "contradictions": contradictions,
                    "resolution": resolution
                })
                
                # Update theme with resolution
                theme["contradiction_resolved"] = True
                theme["resolution_method"] = resolution["method"]
                theme["winning_position"] = resolution["winning_position"]
                
            resolved_themes.append(theme)
            
        return {
            "resolved_themes": resolved_themes,
            "contradictions_found": len(all_contradictions),
            "contradiction_details": all_contradictions,
            "resolution_timestamp": datetime.now().isoformat()
        }
    
    def _detect_contradictions(self, evidence_list: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Detect contradictions in evidence"""
        contradictions = []
        
        # Check each pair of evidence
        for i, evidence1 in enumerate(evidence_list):
            for j, evidence2 in enumerate(evidence_list[i+1:], i+1):
                text1 = evidence1.get("text_snippet", "").lower()
                text2 = evidence2.get("text_snippet", "").lower()
                
                # Check for contradictory terms
                for positive, negative in self.contradiction_indicators:
                    if positive in text1 and negative in text2:
                        contradictions.append({
                            "evidence_1": {
                                "source": evidence1.get("source_url"),
                                "text": evidence1.get("text_snippet"),
                                "position": positive
                            },
                            "evidence_2": {
                                "source": evidence2.get("source_url"),
                                "text": evidence2.get("text_snippet"),
                                "position": negative
                            },
                            "type": f"{positive}_vs_{negative}"
                        })
                    elif negative in text1 and positive in text2:
                        contradictions.append({
                            "evidence_1": {
                                "source": evidence1.get("source_url"),
                                "text": evidence1.get("text_snippet"),
                                "position": negative
                            },
                            "evidence_2": {
                                "source": evidence2.get("source_url"),
                                "text": evidence2.get("text_snippet"),
                                "position": positive
                            },
                            "type": f"{negative}_vs_{positive}"
                        })
                        
        return contradictions
    
    def _resolve_contradictions(
        self, contradictions: List[Dict[str, Any]], evidence_list: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Resolve contradictions using authority and recency"""
        if not contradictions:
            return {"method": "none", "winning_position": "no_contradiction"}
            
        # Score each position based on supporting evidence
        position_scores = {}
        
        for contradiction in contradictions:
            pos1 = contradiction["evidence_1"]["position"]
            pos2 = contradiction["evidence_2"]["position"]
            
            # Find all evidence supporting each position
            for evidence in evidence_list:
                text = evidence.get("text_snippet", "").lower()
                source_url = evidence.get("source_url", "")
                published_date = evidence.get("published_date")
                
                # Get authority weight
                authority, _ = EvidenceHierarchy.get_source_authority(source_url, published_date)
                
                # Calculate recency bonus
                recency_score = 1.0
                if published_date:
                    age_days = (datetime.now() - published_date).days
                    recency_score = max(0.5, 1.0 - (age_days / 365))
                
                combined_score = authority * recency_score
                
                if pos1 in text:
                    position_scores[pos1] = position_scores.get(pos1, 0) + combined_score
                if pos2 in text:
                    position_scores[pos2] = position_scores.get(pos2, 0) + combined_score
                    
        # Determine winning position
        if position_scores:
            winning_position = max(position_scores.items(), key=lambda x: x[1])
            return {
                "method": "authority_and_recency",
                "winning_position": winning_position[0],
                "confidence": winning_position[1] / sum(position_scores.values()),
                "position_scores": position_scores
            }
        else:
            return {
                "method": "unresolved",
                "winning_position": "unclear",
                "reason": "Could not determine authoritative position"
            } 