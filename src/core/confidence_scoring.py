from typing import Dict, List, Optional, Tuple, Any, TYPE_CHECKING
from dataclasses import dataclass
from enum import Enum
from datetime import datetime
import logging
import numpy as np
import re
import sys

# Forward references for type hints to avoid circular imports
if TYPE_CHECKING:
    from .enhanced_data_models import LocalAuthority, AuthenticInsight

from .evidence_hierarchy import EvidenceHierarchy, EvidenceType

logger = logging.getLogger(__name__)

class ConfidenceLevel(Enum):
    """Confidence level classifications"""
    INSUFFICIENT = "insufficient"  # < 0.3
    LOW = "low"                    # 0.3-0.5
    MODERATE = "moderate"          # 0.5-0.7
    HIGH = "high"                  # 0.7-0.85
    VERY_HIGH = "very_high"        # > 0.85

@dataclass
class ConfidenceBreakdown:
    """Detailed confidence analysis"""
    overall_confidence: float
    confidence_level: ConfidenceLevel
    evidence_count: int
    source_diversity: float
    authority_score: float
    recency_score: float
    consistency_score: float
    factors: Dict[str, float]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage"""
        return {
            "overall_confidence": self.overall_confidence,
            "confidence_level": self.confidence_level.value,
            "evidence_count": self.evidence_count,
            "source_diversity": self.source_diversity,
            "authority_score": self.authority_score,
            "recency_score": self.recency_score,
            "consistency_score": self.consistency_score,
            "factors": self.factors,
            "confidence_level": self.confidence_level.value
        }

@dataclass
class MultiDimensionalScore:
    authenticity: float     # 0-1
    uniqueness: float      # 0-1  
    actionability: float   # 0-1
    temporal_relevance: float # 0-1

    def weighted_average(self, weights: Optional[Dict[str, float]] = None) -> float:
        """Calculate weighted average of all dimensions"""
        if weights is None:
            # Default equal weights
            weights = {
                "authenticity": 0.25,
                "uniqueness": 0.25, 
                "actionability": 0.25,
                "temporal_relevance": 0.25
            }
        
        return (
            self.authenticity * weights.get("authenticity", 0.25) +
            self.uniqueness * weights.get("uniqueness", 0.25) +
            self.actionability * weights.get("actionability", 0.25) +
            self.temporal_relevance * weights.get("temporal_relevance", 0.25)
        )

class AuthenticityScorer:
    """Calculate authenticity scores based on local authorities and source diversity"""
    
    def __init__(self):
        self.logger = logging.getLogger(self.__class__.__name__)
    
    def calculate_local_authority_score(self, authorities: List['LocalAuthority']) -> float:
        """Calculate score based on local authority credentials"""
        if not authorities:
            return 0.0
        
        total_score = 0.0
        for authority in authorities:
            # Base score from authority type
            base_scores = {
                'producer': 0.9,      # Highest for direct producers
                'long_term_resident': 0.8,
                'industry_professional': 0.85,
                'cultural_institution': 0.9,
                'seasonal_worker': 0.7
            }
            
            authority_type_str = authority.authority_type.value
            base_score = base_scores.get(authority_type_str, 0.5)
            
            # Adjust for local tenure
            tenure_bonus = 0.0
            if authority.local_tenure:
                tenure_bonus = min(authority.local_tenure / 20.0, 0.2)  # Max 20% bonus for 20+ years
            
            # Community validation factor
            validation_factor = authority.community_validation
            
            authority_score = (base_score + tenure_bonus) * validation_factor
            total_score += min(authority_score, 1.0)
        
        # Average score, with bonus for multiple authorities
        avg_score = total_score / len(authorities)
        multi_authority_bonus = min(len(authorities) / 10.0, 0.2)  # Up to 20% bonus
        
        return min(avg_score + multi_authority_bonus, 1.0)
    
    def calculate_source_diversity_score(self, evidence_list: List[Any]) -> float:
        """Calculate score based on source diversity"""
        if not evidence_list:
            return 0.0
        
        # Extract source URLs
        source_urls = [evidence.source_url for evidence in evidence_list]
        
        # Calculate diversity using EvidenceHierarchy
        diversity = EvidenceHierarchy.calculate_evidence_diversity(source_urls)
        
        return diversity
    
    def calculate_authenticity(self, authorities: List['LocalAuthority'], 
                             evidence_list: List[Any], content: str) -> float:
        """Calculate overall authenticity score"""
        if not authorities and not evidence_list and not content:
            return 0.0
        
        # Component scores
        authority_score = self.calculate_local_authority_score(authorities)
        diversity_score = self.calculate_source_diversity_score(evidence_list)
        
        # Content authenticity indicators
        content_score = self._analyze_content_authenticity(content)
        
        # Weighted combination
        weights = {
            "authority": 0.5,
            "diversity": 0.3,
            "content": 0.2
        }
        
        total_score = (
            authority_score * weights["authority"] +
            diversity_score * weights["diversity"] +
            content_score * weights["content"]
        )
        
        return min(total_score, 1.0)
    
    def _analyze_content_authenticity(self, content: str) -> float:
        """Analyze content for authenticity indicators"""
        if not content:
            return 0.0
        
        content_lower = content.lower()
        
        # Authentic language indicators
        authentic_indicators = [
            'local', 'authentic', 'traditional', 'family-owned', 'artisan',
            'handmade', 'locally-sourced', 'generations', 'heritage', 'original'
        ]
        
        # Generic/tourist language (negative indicators)
        generic_indicators = [
            'world-class', 'must-see', 'tourist', 'popular', 'famous worldwide',
            'internationally known', 'viral', 'trending'
        ]
        
        authentic_count = sum(1 for indicator in authentic_indicators if indicator in content_lower)
        generic_count = sum(1 for indicator in generic_indicators if indicator in content_lower)
        
        # Score based on ratio
        if authentic_count + generic_count == 0:
            return 0.5  # Neutral
        
        ratio = authentic_count / (authentic_count + generic_count)
        return ratio

class UniquenessScorer:
    """Calculate uniqueness scores based on location exclusivity and rarity"""
    
    def __init__(self):
        self.logger = logging.getLogger(self.__class__.__name__)
    
    def calculate_uniqueness(self, insights: List['AuthenticInsight'], content: str) -> float:
        """Calculate uniqueness score"""
        if not insights and not content:
            return 0.0
        
        # Base uniqueness from location exclusivity
        exclusivity_score = 0.0
        if insights:
            exclusivity_values = {
                'exclusive': 1.0,    # Only here
                'signature': 0.8,    # Best known for
                'regional': 0.5,     # Regional specialty  
                'common': 0.2        # Common elsewhere
            }
            
            exclusivity_scores = []
            for insight in insights:
                exclusivity_str = insight.location_exclusivity.value
                score = exclusivity_values.get(exclusivity_str, 0.3)
                exclusivity_scores.append(score)
            
            exclusivity_score = max(exclusivity_scores) if exclusivity_scores else 0.0
        
        # Content uniqueness indicators
        content_score = self._analyze_content_uniqueness(content)
        
        # Weighted combination
        if insights:
            return exclusivity_score * 0.7 + content_score * 0.3
        else:
            return content_score
    
    def _analyze_content_uniqueness(self, content: str) -> float:
        """Analyze content for uniqueness indicators"""
        if not content:
            return 0.0
        
        content_lower = content.lower()
        
        # Uniqueness indicators
        unique_indicators = [
            'only', 'unique', 'rare', 'exclusive', 'special', 'distinctive',
            'one-of-a-kind', 'nowhere else', 'first of its kind', 'original'
        ]
        
        # Common indicators (negative)
        common_indicators = [
            'typical', 'standard', 'common', 'usual', 'ordinary', 'regular',
            'everywhere', 'anywhere', 'found throughout'
        ]
        
        unique_count = sum(1 for indicator in unique_indicators if indicator in content_lower)
        common_count = sum(1 for indicator in common_indicators if indicator in content_lower)
        
        if unique_count + common_count == 0:
            return 0.5  # Neutral
        
        ratio = unique_count / (unique_count + common_count)
        return ratio

class ActionabilityScorer:
    """Calculate actionability scores based on practical information availability"""
    
    def __init__(self):
        self.logger = logging.getLogger(self.__class__.__name__)
    
    def calculate_actionability(self, content: str) -> float:
        """Calculate actionability based on practical details"""
        if not content:
            return 0.0
        
        # Extract actionable elements
        actionable_elements = self.extract_actionable_elements(content)
        
        # Score based on presence of different types of actionable information
        score_components = {
            'location': 0.25,    # Address, directions
            'timing': 0.25,      # Hours, seasons, best times
            'contact': 0.20,     # Phone, website, booking
            'pricing': 0.15,     # Cost information
            'practical': 0.15    # What to bring, how to prepare
        }
        
        total_score = 0.0
        content_lower = content.lower()
        
        # Location information
        if any(keyword in content_lower for keyword in ['address', 'located at', 'street', 'avenue', 'road']):
            total_score += score_components['location']
        
        # Timing information  
        if any(keyword in content_lower for keyword in ['hours', 'open', 'closed', 'am', 'pm', 'best time']):
            total_score += score_components['timing']
        
        # Contact information
        if any(keyword in content_lower for keyword in ['phone', 'call', 'website', 'book', 'reservation']):
            total_score += score_components['contact']
        
        # Pricing information
        if any(keyword in content_lower for keyword in ['$', 'cost', 'price', 'fee', 'free', 'admission']):
            total_score += score_components['pricing']
        
        # Practical details
        if any(keyword in content_lower for keyword in ['bring', 'wear', 'prepare', 'expect', 'tips']):
            total_score += score_components['practical']
        
        # Bonus for having multiple actionable elements
        if len(actionable_elements) > 3:
            total_score *= 1.2
        
        return min(total_score, 1.0)
    
    def extract_actionable_elements(self, content: str) -> List[str]:
        """Extract specific actionable elements from content"""
        import re
        
        elements = []
        
        # Address patterns
        address_patterns = [
            r'\d+\s+[A-Za-z\s]+(?:Street|St|Avenue|Ave|Road|Rd|Boulevard|Blvd)',
            r'located at [^.!?]+',
            r'address[:\s]+[^.!?]+'
        ]
        
        # Hours patterns
        time_patterns = [
            r'\d{1,2}(?::\d{2})?\s*(?:am|pm)\s*-\s*\d{1,2}(?::\d{2})?\s*(?:am|pm)',
            r'open [^.!?]+',
            r'hours[:\s]+[^.!?]+'
        ]
        
        # Contact patterns
        contact_patterns = [
            r'\(\d{3}\)\s*\d{3}-\d{4}',  # Phone numbers
            r'\d{3}-\d{3}-\d{4}',
            r'call [^.!?]+',
            r'website[:\s]+[^.!?\s]+'
        ]
        
        all_patterns = address_patterns + time_patterns + contact_patterns
        
        for pattern in all_patterns:
            matches = re.findall(pattern, content, re.IGNORECASE)
            elements.extend(matches)
        
        return elements

class ConfidenceScorer:
    """Main confidence scoring system"""
    
    def __init__(self):
        self.evidence_hierarchy = EvidenceHierarchy()
        self.logger = logging.getLogger("app.confidence_scorer")
        self.logger.setLevel(logging.DEBUG)
    
    def calculate_confidence(self, evidence_list: List[Any]) -> ConfidenceBreakdown:
        """Calculate comprehensive confidence breakdown"""
        import sys
        print(f"DEBUG_CS_ENTRY: ENTERING calculate_confidence with {len(evidence_list)} evidence items.", file=sys.stderr)
        self.logger.debug(f"Calculating confidence for {len(evidence_list)} evidence pieces.")

        if not evidence_list:
            print(f"DEBUG_CS_CALC: No evidence, returning INSUFFICIENT.", file=sys.stderr)
            self.logger.debug("No evidence provided, returning INSUFFICIENT confidence.")
            return ConfidenceBreakdown(
                overall_confidence=0.0,
                confidence_level=ConfidenceLevel.INSUFFICIENT,
                evidence_count=0,
                source_diversity=0.0,
                authority_score=0.0,
                recency_score=0.0,
                consistency_score=0.0,
                factors={}
            )
        
        # Calculate component scores
        evidence_count = len(evidence_list)
        source_diversity = self._calculate_source_diversity(evidence_list)
        authority_score = self._calculate_authority_score(evidence_list)
        recency_score = self._calculate_recency_score(evidence_list)
        consistency_score = self._calculate_consistency_score(evidence_list)
        
        weights = {
            "evidence_count": 0.2,
            "source_diversity": 0.25,
            "authority": 0.25,
            "recency": 0.15,
            "consistency": 0.15
        }
        
        evidence_score = min(evidence_count / 10.0, 1.0)
        
        # Log and Print component scores
        print(f"DEBUG_CS_CALC: EvCountRaw={evidence_count}, EvScoreNorm={evidence_score:.4f}, SrcDiv={source_diversity:.4f}, AuthScr={authority_score:.4f}, RecScr={recency_score:.4f}, ConsScr={consistency_score:.4f}", file=sys.stderr)
        self.logger.debug(f"[Confidence Components] Evidence Count Raw: {evidence_count}, Evidence Score (Normalized): {evidence_score:.4f}")
        self.logger.debug(f"[Confidence Components] Source Diversity: {source_diversity:.4f}")
        self.logger.debug(f"[Confidence Components] Authority Score: {authority_score:.4f}")
        self.logger.debug(f"[Confidence Components] Recency Score: {recency_score:.4f}")
        self.logger.debug(f"[Confidence Components] Consistency Score: {consistency_score:.4f}")
        self.logger.debug(f"[Confidence Components] Weights: {weights}")

        overall_confidence = (
            evidence_score * weights["evidence_count"] +
            source_diversity * weights["source_diversity"] +
            authority_score * weights["authority"] +
            recency_score * weights["recency"] +
            consistency_score * weights["consistency"]
        )
        
        print(f"DEBUG_CS_CALC: OverallConfidence={overall_confidence:.4f}", file=sys.stderr)
        self.logger.debug(f"Calculated Overall Confidence: {overall_confidence:.4f}")
        
        confidence_level = self._determine_confidence_level(overall_confidence)
        print(f"DEBUG_CS_CALC: ConfidenceLevel='{confidence_level.value}'", file=sys.stderr)
        self.logger.debug(f"Determined Confidence Level: {confidence_level.value}")
        
        return ConfidenceBreakdown(
            overall_confidence=overall_confidence,
            confidence_level=confidence_level,
            evidence_count=evidence_count,
            source_diversity=source_diversity,
            authority_score=authority_score,
            recency_score=recency_score,
            consistency_score=consistency_score,
            factors={
                "evidence_score": evidence_score,
                "weights": weights
            }
        )
    
    def _calculate_source_diversity(self, evidence_list: List[Any]) -> float:
        """Calculate diversity of evidence sources"""
        source_urls = [evidence.source_url for evidence in evidence_list]
        return EvidenceHierarchy.calculate_evidence_diversity(source_urls)
    
    def _calculate_authority_score(self, evidence_list: List[Any]) -> float:
        """Calculate weighted authority score"""
        import sys # Ensure sys is imported here if not globally
        if not evidence_list:
            print("DEBUG_CS_AUTH: No evidence for authority score.", file=sys.stderr)
            return 0.0
        
        weighted_authorities = []
        for i, evidence in enumerate(evidence_list):
            quality_score = self._evidence_quality_score(evidence)
            weighted_authority = evidence.authority_weight * quality_score
            weighted_authorities.append(weighted_authority)
            if i < 3: 
                print(f"DEBUG_CS_AUTH_EV[{i}]: ev.auth_w={evidence.authority_weight:.4f}, ev.conf(used_by_qual)={evidence.confidence:.4f}, qual_score={quality_score:.4f}, combined={weighted_authority:.4f}", file=sys.stderr)
                self.logger.debug(f"  AuthCalc Ev[{i}]: authority_weight={evidence.authority_weight:.4f}, ev.confidence(used_by_qual)={evidence.confidence:.4f}, quality_score={quality_score:.4f}, combined_ev_auth={weighted_authority:.4f}")

        total_weighted_authority = sum(weighted_authorities)
        calculated_authority_score = total_weighted_authority / len(evidence_list) if evidence_list else 0.0
        print(f"DEBUG_CS_AUTH: TotalWeightedAuth={total_weighted_authority:.4f}, NumEv={len(evidence_list)}, AvgAuthScore={calculated_authority_score:.4f}", file=sys.stderr)
        self.logger.debug(f"  AuthCalc: TotalWeightedAuth={total_weighted_authority:.4f}, NumEvidence={len(evidence_list)}, AvgAuthorityScore={calculated_authority_score:.4f}")
        return calculated_authority_score
    
    def _evidence_quality_score(self, evidence: Any) -> float:
        """Calculate quality score for individual evidence"""
        base_score = evidence.confidence 
        category_str = evidence.source_category.name if hasattr(evidence.source_category, 'name') else str(evidence.source_category)
        
        # REINSTATED: category_multipliers dictionary
        category_multipliers = {
            'GOVERNMENT': 1.0,
            'ACADEMIC': 0.95,
            'BUSINESS': 0.8,
            'GUIDEBOOK': 0.75,
            'BLOG': 0.6,
            'SOCIAL': 0.4,
            'UNKNOWN': 0.3
        }
        multiplier = category_multipliers.get(category_str, 0.5)
        
        final_quality_score = base_score * multiplier
        # self.logger.debug(f"    EvQuality: base_score(ev.confidence)={base_score:.4f}, category={category_str}, multiplier={multiplier:.4f}, final_ev_quality={final_quality_score:.4f}") # Too verbose for every piece
        return final_quality_score
    
    def _calculate_recency_score(self, evidence_list: List[Any]) -> float:
        """Calculate score based on recency of evidence"""
        import sys # Ensure sys is imported here if not globally
        if not evidence_list:
            print("DEBUG_CS_RECENCY: No evidence for recency score.", file=sys.stderr)
            return 0.0
        
        current_time = datetime.now()
        recency_scores = []
        
        for i, evidence in enumerate(evidence_list):
            if evidence.timestamp:
                try:
                    # Attempt to parse if it's a string, otherwise assume it's a datetime object
                    timestamp_obj = evidence.timestamp
                    if isinstance(evidence.timestamp, str):
                        timestamp_obj = datetime.fromisoformat(evidence.timestamp.replace('Z', '+00:00'))
                    
                    days_old = (current_time - timestamp_obj).days
                    recency_score_val = max(0.1, 0.5 ** (days_old / 365))
                    recency_scores.append(recency_score_val)
                    if i < 3:
                        print(f"DEBUG_CS_RECENCY_EV[{i}]: ts={evidence.timestamp}, days_old={days_old}, score={recency_score_val:.4f}", file=sys.stderr)
                        self.logger.debug(f"  RecencyCalc Ev[{i}]: timestamp={evidence.timestamp}, days_old={days_old}, recency_score_val={recency_score_val:.4f}")
                except Exception as e:
                    recency_scores.append(0.2) # Penalize unparseable/problematic timestamps
                    if i < 3:
                        print(f"DEBUG_CS_RECENCY_EV[{i}]: ts={evidence.timestamp}, ERROR parsing: {e}, score=0.2000", file=sys.stderr)
                        self.logger.warning(f"  RecencyCalc Ev[{i}]: Error parsing timestamp '{evidence.timestamp}': {e}. Assigning low score.")
            else:
                recency_scores.append(0.3)
                if i < 3:
                    print(f"DEBUG_CS_RECENCY_EV[{i}]: ts=NULL, score=0.3000", file=sys.stderr)
                    self.logger.debug(f"  RecencyCalc Ev[{i}]: timestamp=NULL, recency_score_val=0.3000")
        
        avg_recency = sum(recency_scores) / len(recency_scores) if recency_scores else 0.0
        print(f"DEBUG_CS_RECENCY: AvgRecency={avg_recency:.4f}", file=sys.stderr)
        self.logger.debug(f"  RecencyCalc: AvgRecency={avg_recency:.4f}")
        return avg_recency
    
    def _calculate_consistency_score(self, evidence_list: List[Any]) -> float:
        """Calculate consistency across evidence sources"""
        import sys # Ensure sys is imported here if not globally
        if len(evidence_list) < 2:
            print(f"DEBUG_CS_CONSIST: <2 evidence, consistency=1.0", file=sys.stderr)
            return 1.0
        
        # For now, use confidence variance as a proxy for consistency
        confidences = [evidence.confidence for evidence in evidence_list]
        
        # Calculate variance
        mean_confidence = sum(confidences) / len(confidences) if confidences else 0.0
        variance = sum((c - mean_confidence) ** 2 for c in confidences) / len(confidences) if confidences else 0.0
        
        # Convert variance to consistency score (lower variance = higher consistency)
        consistency_score_val = max(0.1, 1.0 - variance)
        
        print(f"DEBUG_CS_CONSIST: NumConf={len(confidences)}, MeanConf={mean_confidence:.4f}, Var={variance:.4f}, Score={consistency_score_val:.4f}", file=sys.stderr)
        self.logger.debug(f"  ConsistencyCalc: NumConfidences={len(confidences)}, MeanConfidence={mean_confidence:.4f}, Variance={variance:.4f}, ConsistencyScore={consistency_score_val:.4f}")
        return consistency_score_val
    
    def _determine_confidence_level(self, overall_confidence: float) -> ConfidenceLevel:
        """Determine confidence level from overall score"""
        if overall_confidence < 0.3:
            return ConfidenceLevel.INSUFFICIENT
        elif overall_confidence < 0.5:
            return ConfidenceLevel.LOW
        elif overall_confidence < 0.7:
            return ConfidenceLevel.MODERATE
        elif overall_confidence < 0.85:
            return ConfidenceLevel.HIGH
        else:
            return ConfidenceLevel.VERY_HIGH 