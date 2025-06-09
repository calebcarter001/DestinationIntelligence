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

# Placeholder for line_profiler, kernprof makes @profile available globally
# For local linting/IDE, you might need: 
# try:
#     # This will only be available when running with kernprof
#     profile = profile 
# except NameError:
#     # If not running with kernprof, define a dummy decorator
#     def profile(func):
#         return func

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
    locality_score: float
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
            "locality_score": self.locality_score,
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
            
            authority_type_str = authority.authority_type.value if hasattr(authority.authority_type, 'value') else str(authority.authority_type)
            base_score = base_scores.get(authority_type_str, 0.5)
            
            # Adjust for local tenure
            tenure_bonus = 0.0
            local_tenure = getattr(authority, 'local_tenure', None)
            if local_tenure:
                tenure_bonus = min(local_tenure / 20.0, 0.2)  # Max 20% bonus for 20+ years
            
            # Community validation factor
            validation_factor = getattr(authority, 'community_validation', 0.0)
            
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
                exclusivity_str = insight.location_exclusivity.value if hasattr(insight.location_exclusivity, 'value') else str(insight.location_exclusivity)
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
    """
    Calculates confidence scores for themes based on evidence quality
    """
    
    # @profile # Decorator for the __init__ method if needed for complex initializations
    def __init__(self):
        # Updated weights to include locality
        self.weights = {
            "evidence_count": 0.15,  # Reduced from 0.2
            "source_diversity": 0.20, # Reduced from 0.25
            "authority": 0.25,       # Maintained
            "recency": 0.15,        # Maintained
            "consistency": 0.15,     # Maintained
            "locality": 0.10        # New weight for local relevance
        }
        
        # Stricter thresholds for confidence levels
        self.confidence_thresholds = {
            "high": 0.75,    # Increased from 0.7
            "moderate": 0.6, # Increased from 0.5
            "low": 0.4      # Increased from 0.3
        }

    # @profile
    def calculate_confidence(self, evidence_list: List[Any]) -> ConfidenceBreakdown:
        """
        Calculate confidence score based on evidence quality
        
        Args:
            evidence_list: List of evidence objects
        """
        if not evidence_list:
            return ConfidenceBreakdown(
                overall_confidence=0.1,
                confidence_level=ConfidenceLevel.INSUFFICIENT,
                evidence_count=0,
                source_diversity=0.0,
                authority_score=0.0,
                recency_score=0.0,
                consistency_score=0.0,
                locality_score=0.0,
                factors={}
            )

        # Calculate base metrics
        evidence_count = len(evidence_list)
        evidence_score = min(1.0, evidence_count / 10)  # Cap at 10 pieces of evidence
        
        # Source diversity
        sources = set(ev.source_url for ev in evidence_list if ev.source_url)
        source_diversity = min(1.0, len(sources) / 5)  # Cap at 5 unique sources
        
        # Authority scoring
        authority_weights = []
        for ev in evidence_list:
            quality_score = self._evidence_quality_score(ev)
            authority_weights.append(quality_score)
        
        authority_score = sum(authority_weights) / len(authority_weights) if authority_weights else 0.0
        
        # Recency scoring
        current_date = datetime.now()
        recency_scores = []
        for ev in evidence_list:
            if ev.timestamp:
                # Handle both datetime objects and string timestamps
                if isinstance(ev.timestamp, str):
                    try:
                        timestamp = datetime.fromisoformat(ev.timestamp.replace('Z', '+00:00'))
                    except (ValueError, AttributeError):
                        timestamp = current_date  # Default to current date if parsing fails
                else:
                    timestamp = ev.timestamp
                
                age_days = (current_date - timestamp).days
                recency_score = max(0.0, min(1.0, 1.0 - (age_days / 365)))
                recency_scores.append(recency_score)
            else:
                recency_scores.append(0.5)
        
        avg_recency = sum(recency_scores) / len(recency_scores) if recency_scores else 0.5
        
        # Consistency scoring
        consistency_score = self._calculate_consistency(evidence_list)
        
        # New: Locality scoring
        locality_score = self._calculate_locality_score(evidence_list)
        
        # Calculate weighted score
        weighted_scores = {
            "evidence_count": evidence_score * self.weights["evidence_count"],
            "source_diversity": source_diversity * self.weights["source_diversity"],
            "authority": authority_score * self.weights["authority"],
            "recency": avg_recency * self.weights["recency"],
            "consistency": consistency_score * self.weights["consistency"],
            "locality": locality_score * self.weights["locality"]
        }
        
        overall_confidence = sum(weighted_scores.values())
        
        # Determine confidence level using stricter thresholds
        if overall_confidence >= self.confidence_thresholds["high"]:
            confidence_level = ConfidenceLevel.HIGH
        elif overall_confidence >= self.confidence_thresholds["moderate"]:
            confidence_level = ConfidenceLevel.MODERATE
        elif overall_confidence >= self.confidence_thresholds["low"]:
            confidence_level = ConfidenceLevel.LOW
        else:
            confidence_level = ConfidenceLevel.INSUFFICIENT
        
        return ConfidenceBreakdown(
            overall_confidence=overall_confidence,
            confidence_level=confidence_level,
            evidence_count=evidence_count,
            source_diversity=source_diversity,
            authority_score=authority_score,
            recency_score=avg_recency,
            consistency_score=consistency_score,
            locality_score=locality_score,
            factors=weighted_scores
        )

    # @profile
    def _calculate_locality_score(self, evidence_list: List[Any]) -> float:
        """
        Calculate locality score based on how specific the evidence is to the location
        """
        if not evidence_list:
            return 0.0
            
        locality_indicators = [
            r"local", r"neighborhood", r"district", r"area",
            r"community", r"resident", r"native", r"street",
            r"block", r"quarter", r"precinct", r"vicinity"
        ]
        
        locality_scores = []
        for evidence in evidence_list:
            text = evidence.text_snippet.lower()
            # Count matches of locality indicators
            matches = sum(1 for pattern in locality_indicators if re.search(pattern, text))
            # Score based on number of matches, cap at 3 matches
            score = min(1.0, matches / 3)
            locality_scores.append(score)
            
        return sum(locality_scores) / len(locality_scores) if locality_scores else 0.0

    # @profile
    def _calculate_consistency(self, evidence_list: List[Any]) -> float:
        """
        Calculate consistency score based on sentiment and fact alignment
        """
        if len(evidence_list) < 2:
            return 1.0  # Single piece of evidence is consistent with itself
            
        # Extract key facts and sentiments
        fact_patterns = {}
        sentiments = []
        
        for evidence in evidence_list:
            text = evidence.text_snippet.lower()
            
            # Extract numerical facts
            numbers = re.findall(r'\d+(?:\.\d+)?', text)
            for num in numbers:
                num_idx = text.find(num)
                # Calculate start and end indices for context extraction
                # Window of 30 characters before the number, and 30 characters after the number including the number itself.
                start_idx = max(0, num_idx - 30)
                end_idx = min(len(text), num_idx + len(num) + 30)
                context = text[start_idx:end_idx]
                fact_patterns[context] = float(num)
            
            # Basic sentiment analysis
            positive_words = ["good", "great", "excellent", "best", "beautiful", "safe", "clean"]
            negative_words = ["bad", "poor", "worst", "dangerous", "dirty", "unsafe"]
            
            pos_count = sum(1 for word in positive_words if word in text)
            neg_count = sum(1 for word in negative_words if word in text)
            
            if pos_count > neg_count:
                sentiments.append(1)
            elif neg_count > pos_count:
                sentiments.append(-1)
            else:
                sentiments.append(0)
        
        # Calculate fact consistency
        fact_variance = 0
        if fact_patterns:
            variances = []
            for numbers in fact_patterns.values():
                if isinstance(numbers, list) and len(numbers) > 1:
                    variance = np.var(numbers) / (np.mean(numbers) ** 2)  # Normalized variance
                    variances.append(min(1.0, variance))
            if variances:
                fact_variance = sum(variances) / len(variances)
        
        # Calculate sentiment consistency
        sentiment_consistency = 1.0
        if sentiments:
            sentiment_variance = np.var(sentiments)
            sentiment_consistency = max(0.0, 1.0 - sentiment_variance)
        
        # Combine fact and sentiment consistency
        return (1.0 - fact_variance) * 0.6 + sentiment_consistency * 0.4

    # @profile
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