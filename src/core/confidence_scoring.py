from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
from datetime import datetime
import logging
import numpy as np

from .evidence_hierarchy import EvidenceHierarchy, EvidenceType

logger = logging.getLogger(__name__)

class ConfidenceLevel(Enum):
    """Confidence level classifications"""
    VERIFIED = "verified"                    # > 0.90
    STRONGLY_SUPPORTED = "strongly_supported" # 0.80 - 0.90
    WELL_SUPPORTED = "well_supported"        # 0.70 - 0.80
    PARTIALLY_SUPPORTED = "partially_supported" # 0.50 - 0.70
    EMERGING = "emerging"                    # 0.30 - 0.50
    INSUFFICIENT = "insufficient"            # < 0.30

@dataclass
class ConfidenceBreakdown:
    """Detailed breakdown of confidence score components"""
    source_authority: float
    evidence_diversity: float
    consistency: float
    recency: float
    evidence_quantity: float
    cultural_perspective: float
    total_confidence: float
    confidence_level: ConfidenceLevel
    
    def to_dict(self) -> Dict[str, float]:
        """Convert to dictionary for storage"""
        return {
            "source_authority": self.source_authority,
            "evidence_diversity": self.evidence_diversity,
            "consistency": self.consistency,
            "recency": self.recency,
            "evidence_quantity": self.evidence_quantity,
            "cultural_perspective": self.cultural_perspective,
            "total_confidence": self.total_confidence,
            "confidence_level": self.confidence_level.value
        }

class ConfidenceScorer:
    """
    Implements the 6-factor confidence scoring formula:
    
    confidence = 0.30·source_authority
               + 0.20·evidence_diversity
               + 0.15·consistency
               + 0.10·recency
               + 0.10·evidence_quantity
               + 0.15·cultural_perspective
    """
    
    # Component weights
    WEIGHTS = {
        "source_authority": 0.30,
        "evidence_diversity": 0.20,
        "consistency": 0.15,
        "recency": 0.10,
        "evidence_quantity": 0.10,
        "cultural_perspective": 0.15
    }
    
    # Confidence level thresholds
    CONFIDENCE_THRESHOLDS = {
        ConfidenceLevel.VERIFIED: 0.90,
        ConfidenceLevel.STRONGLY_SUPPORTED: 0.80,
        ConfidenceLevel.WELL_SUPPORTED: 0.70,
        ConfidenceLevel.PARTIALLY_SUPPORTED: 0.50,
        ConfidenceLevel.EMERGING: 0.30,
        ConfidenceLevel.INSUFFICIENT: 0.0
    }
    
    @classmethod
    def calculate_confidence(
        cls,
        evidence_sources: List[str],
        evidence_texts: List[str],
        published_dates: Optional[List[datetime]] = None,
        destination_country_code: Optional[str] = None,
        sentiment_scores: Optional[List[float]] = None
    ) -> ConfidenceBreakdown:
        """
        Calculate comprehensive confidence score for a theme/insight
        
        Args:
            evidence_sources: List of source URLs
            evidence_texts: List of evidence text snippets
            published_dates: List of publication dates (if known)
            destination_country_code: ISO code for cultural perspective
            sentiment_scores: List of sentiment scores for consistency check
            
        Returns:
            ConfidenceBreakdown with all component scores
        """
        if not evidence_sources:
            return cls._empty_confidence()
        
        # Ensure lists are same length
        num_sources = len(evidence_sources)
        if published_dates is None:
            published_dates = [None] * num_sources
        if sentiment_scores is None:
            sentiment_scores = [0.0] * num_sources
            
        # 1. Calculate source authority
        source_authority = cls._calculate_source_authority(
            evidence_sources, published_dates
        )
        
        # 2. Calculate evidence diversity
        evidence_diversity = EvidenceHierarchy.calculate_evidence_diversity(
            evidence_sources
        )
        
        # 3. Calculate consistency
        consistency = cls._calculate_consistency(
            evidence_texts, sentiment_scores
        )
        
        # 4. Calculate recency
        recency = cls._calculate_recency(published_dates)
        
        # 5. Calculate evidence quantity
        evidence_quantity = cls._calculate_evidence_quantity(num_sources)
        
        # 6. Calculate cultural perspective
        cultural_perspective = cls._calculate_cultural_perspective(
            evidence_sources, destination_country_code
        )
        
        # Calculate total confidence
        total_confidence = (
            cls.WEIGHTS["source_authority"] * source_authority +
            cls.WEIGHTS["evidence_diversity"] * evidence_diversity +
            cls.WEIGHTS["consistency"] * consistency +
            cls.WEIGHTS["recency"] * recency +
            cls.WEIGHTS["evidence_quantity"] * evidence_quantity +
            cls.WEIGHTS["cultural_perspective"] * cultural_perspective
        )
        
        # Determine confidence level
        confidence_level = cls._determine_confidence_level(total_confidence)
        
        return ConfidenceBreakdown(
            source_authority=source_authority,
            evidence_diversity=evidence_diversity,
            consistency=consistency,
            recency=recency,
            evidence_quantity=evidence_quantity,
            cultural_perspective=cultural_perspective,
            total_confidence=total_confidence,
            confidence_level=confidence_level
        )
    
    @classmethod
    def _calculate_source_authority(
        cls,
        sources: List[str],
        published_dates: List[Optional[datetime]]
    ) -> float:
        """Calculate average source authority including decay"""
        if not sources:
            return 0.0
            
        authorities = []
        for source, pub_date in zip(sources, published_dates):
            weight, _ = EvidenceHierarchy.get_source_authority(source, pub_date)
            authorities.append(weight)
            
        # Use weighted average, giving more weight to higher authority sources
        weights = np.array(authorities)
        normalized_weights = weights / weights.sum()
        weighted_avg = np.sum(weights * normalized_weights)
        
        return float(weighted_avg)
    
    @classmethod
    def _calculate_consistency(
        cls,
        evidence_texts: List[str],
        sentiment_scores: List[float]
    ) -> float:
        """
        Calculate consistency of evidence
        
        Based on:
        - Sentiment consistency
        - Text similarity (future enhancement)
        - Contradiction detection (future enhancement)
        """
        if len(evidence_texts) < 2:
            return 1.0  # Single source is consistent with itself
            
        # Calculate sentiment consistency
        if sentiment_scores and len(sentiment_scores) > 1:
            sentiment_std = np.std(sentiment_scores)
            # Lower std = higher consistency
            # Map std (0 to 2) to consistency (1 to 0)
            sentiment_consistency = max(0, 1 - (sentiment_std / 2))
        else:
            sentiment_consistency = 0.5  # Neutral if no sentiment data
            
        # TODO: Add text similarity analysis
        # TODO: Add contradiction detection
        
        return sentiment_consistency
    
    @classmethod
    def _calculate_recency(cls, published_dates: List[Optional[datetime]]) -> float:
        """
        Calculate recency score based on publication dates
        
        Recent evidence scores higher
        """
        known_dates = [d for d in published_dates if d is not None]
        
        if not known_dates:
            return 0.5  # Neutral if no dates known
            
        # Calculate average age in days
        current_date = datetime.now()
        ages = [(current_date - d).days for d in known_dates]
        avg_age_days = np.mean(ages)
        
        # Map age to score using exponential decay
        # 0 days = 1.0, 365 days = 0.5, 730 days = 0.25, etc.
        recency_score = 0.5 ** (avg_age_days / 365)
        
        return float(recency_score)
    
    @classmethod
    def _calculate_evidence_quantity(cls, num_sources: int) -> float:
        """
        Calculate score based on quantity of evidence
        
        Uses logarithmic scale with diminishing returns
        """
        if num_sources == 0:
            return 0.0
        
        # Logarithmic scale: 1 source = 0.3, 5 sources = 0.7, 10+ sources = 1.0
        quantity_score = min(1.0, 0.3 + (0.7 * np.log10(num_sources + 1)))
        
        return float(quantity_score)
    
    @classmethod
    def _calculate_cultural_perspective(
        cls,
        sources: List[str],
        destination_country_code: Optional[str]
    ) -> float:
        """
        Calculate cultural perspective score
        
        Higher score for local sources and diverse perspectives
        """
        if not sources:
            return 0.0
            
        local_sources = sum(
            1 for source in sources 
            if EvidenceHierarchy.is_local_source(source, destination_country_code)
        )
        
        # Calculate local source ratio
        local_ratio = local_sources / len(sources)
        
        # Ideal is mix of local and international (60% local, 40% international)
        # Score peaks at 0.6 local ratio, decreases on either side
        if local_ratio <= 0.6:
            cultural_score = local_ratio / 0.6
        else:
            # Decrease score for too much local bias
            cultural_score = 1.0 - ((local_ratio - 0.6) / 0.4) * 0.3
            
        return float(cultural_score)
    
    @classmethod
    def _determine_confidence_level(cls, total_confidence: float) -> ConfidenceLevel:
        """Determine confidence level based on total score"""
        for level, threshold in cls.CONFIDENCE_THRESHOLDS.items():
            if total_confidence >= threshold:
                return level
        return ConfidenceLevel.INSUFFICIENT
    
    @classmethod
    def _empty_confidence(cls) -> ConfidenceBreakdown:
        """Return empty confidence breakdown for no evidence"""
        return ConfidenceBreakdown(
            source_authority=0.0,
            evidence_diversity=0.0,
            consistency=0.0,
            recency=0.0,
            evidence_quantity=0.0,
            cultural_perspective=0.0,
            total_confidence=0.0,
            confidence_level=ConfidenceLevel.INSUFFICIENT
        )
    
    @classmethod
    def explain_confidence(cls, breakdown: ConfidenceBreakdown) -> str:
        """Generate human-readable explanation of confidence score"""
        explanations = []
        
        # Overall assessment
        explanations.append(
            f"Overall Confidence: {breakdown.confidence_level.value.replace('_', ' ').title()} "
            f"({breakdown.total_confidence:.2%})"
        )
        
        # Component breakdown
        explanations.append("\nComponent Scores:")
        explanations.append(f"  • Source Authority: {breakdown.source_authority:.2%}")
        explanations.append(f"  • Evidence Diversity: {breakdown.evidence_diversity:.2%}")
        explanations.append(f"  • Consistency: {breakdown.consistency:.2%}")
        explanations.append(f"  • Recency: {breakdown.recency:.2%}")
        explanations.append(f"  • Evidence Quantity: {breakdown.evidence_quantity:.2%}")
        explanations.append(f"  • Cultural Perspective: {breakdown.cultural_perspective:.2%}")
        
        # Recommendations
        if breakdown.confidence_level == ConfidenceLevel.INSUFFICIENT:
            explanations.append("\nRecommendation: Seek additional evidence from authoritative sources")
        elif breakdown.confidence_level == ConfidenceLevel.EMERGING:
            explanations.append("\nRecommendation: Monitor for additional supporting evidence")
        elif breakdown.confidence_level in [ConfidenceLevel.VERIFIED, ConfidenceLevel.STRONGLY_SUPPORTED]:
            explanations.append("\nRecommendation: High confidence - suitable for decision making")
            
        return "\n".join(explanations) 