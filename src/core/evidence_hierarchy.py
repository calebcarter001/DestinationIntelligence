from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
from enum import Enum
import re
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)

class EvidenceType(Enum):
    """Evidence source classification"""
    PRIMARY = "primary"      # .gov, UNESCO, official datasets
    SECONDARY = "secondary"  # Industry reports, vetted guidebooks
    TERTIARY = "tertiary"    # Blogs, social media, user reviews
    
class SourceCategory(Enum):
    """Detailed source categorization for decay functions"""
    GOVERNMENT = "government"
    UNESCO = "unesco"
    ACADEMIC = "academic"
    INDUSTRY = "industry"
    GUIDEBOOK = "guidebook"
    NEWS = "news"
    REVIEW = "review"
    SOCIAL = "social"
    BLOG = "blog"
    UNKNOWN = "unknown"

@dataclass
class EvidenceAuthority:
    """Evidence authority configuration"""
    evidence_type: EvidenceType
    base_weight: float
    decay_half_life_days: int
    category: SourceCategory

class EvidenceHierarchy:
    """Manages evidence classification, weighting, and decay"""
    
    # Authority configurations with weights and decay half-lives
    AUTHORITY_CONFIG = {
        # Primary sources (0.90-0.95)
        SourceCategory.GOVERNMENT: EvidenceAuthority(
            EvidenceType.PRIMARY, 0.95, 365 * 5, SourceCategory.GOVERNMENT  # 5 year half-life
        ),
        SourceCategory.UNESCO: EvidenceAuthority(
            EvidenceType.PRIMARY, 0.95, 365 * 50, SourceCategory.UNESCO  # 50 year half-life
        ),
        SourceCategory.ACADEMIC: EvidenceAuthority(
            EvidenceType.PRIMARY, 0.90, 365 * 10, SourceCategory.ACADEMIC  # 10 year half-life
        ),
        
        # Secondary sources (0.60-0.75)
        SourceCategory.INDUSTRY: EvidenceAuthority(
            EvidenceType.SECONDARY, 0.75, 365 * 3, SourceCategory.INDUSTRY  # 3 year half-life
        ),
        SourceCategory.GUIDEBOOK: EvidenceAuthority(
            EvidenceType.SECONDARY, 0.70, 365 * 2, SourceCategory.GUIDEBOOK  # 2 year half-life
        ),
        SourceCategory.NEWS: EvidenceAuthority(
            EvidenceType.SECONDARY, 0.65, 180, SourceCategory.NEWS  # 6 month half-life
        ),
        
        # Tertiary sources (0.25-0.40)
        SourceCategory.REVIEW: EvidenceAuthority(
            EvidenceType.TERTIARY, 0.40, 365, SourceCategory.REVIEW  # 1 year half-life
        ),
        SourceCategory.BLOG: EvidenceAuthority(
            EvidenceType.TERTIARY, 0.35, 180, SourceCategory.BLOG  # 6 month half-life
        ),
        SourceCategory.SOCIAL: EvidenceAuthority(
            EvidenceType.TERTIARY, 0.25, 90, SourceCategory.SOCIAL  # 3 month half-life
        ),
        SourceCategory.UNKNOWN: EvidenceAuthority(
            EvidenceType.TERTIARY, 0.30, 365, SourceCategory.UNKNOWN  # 1 year default
        )
    }
    
    # URL patterns for source classification
    SOURCE_PATTERNS = {
        SourceCategory.GOVERNMENT: [
            r'\.gov(\.[a-z]{2})?(/|$)',
            r'\.gob\.',
            r'\.gouv\.',
            r'government\.',
            r'official\.',
            r'state\.',
            r'city\.',
            r'tourism\..*\.gov'
        ],
        SourceCategory.UNESCO: [
            r'unesco\.org',
            r'whc\.unesco',
            r'en\.unesco'
        ],
        SourceCategory.ACADEMIC: [
            r'\.edu(/|$)',
            r'\.ac\.',
            r'university',
            r'journal',
            r'academic',
            r'scholar\.google',
            r'jstor\.org',
            r'pubmed'
        ],
        SourceCategory.INDUSTRY: [
            r'travel-industry',
            r'tourism-board',
            r'destination-marketing',
            r'phocuswright',
            r'skift\.com',
            r'traveldailynews'
        ],
        SourceCategory.GUIDEBOOK: [
            r'lonelyplanet',
            r'fodors',
            r'frommers',
            r'roughguides',
            r'tripadvisor\.com/Tourism',
            r'timeout\.com'
        ],
        SourceCategory.NEWS: [
            r'cnn\.com/travel',
            r'bbc\.com/travel',
            r'nytimes\.com/.*travel',
            r'theguardian\.com/travel',
            r'reuters\.com',
            r'bloomberg\.com'
        ],
        SourceCategory.REVIEW: [
            r'tripadvisor\.com/.*Review',
            r'yelp\.com',
            r'booking\.com/reviews',
            r'hotels\.com/.*reviews',
            r'airbnb\.com/.*reviews',
            r'viator\.com'
        ],
        SourceCategory.BLOG: [
            r'blogspot\.com',
            r'wordpress\.com',
            r'medium\.com',
            r'/blog/',
            r'travelblog',
            r'nomadic',
            r'backpacker'
        ],
        SourceCategory.SOCIAL: [
            r'instagram\.com',
            r'facebook\.com',
            r'twitter\.com',
            r'x\.com',
            r'reddit\.com',
            r'pinterest\.com',
            r'tiktok\.com'
        ]
    }
    
    @classmethod
    def classify_source(cls, url: str) -> SourceCategory:
        """Classify a URL into a source category"""
        url_lower = url.lower()
        
        for category, patterns in cls.SOURCE_PATTERNS.items():
            for pattern in patterns:
                if re.search(pattern, url_lower):
                    logger.debug(f"Classified {url} as {category.value}")
                    return category
        
        logger.debug(f"Could not classify {url}, defaulting to UNKNOWN")
        return SourceCategory.UNKNOWN
    
    @classmethod
    def get_source_authority(cls, url: str, published_date: Optional[datetime] = None) -> Tuple[float, EvidenceType]:
        """
        Get the authority weight for a source, including time decay
        
        Returns:
            Tuple of (weight, evidence_type)
        """
        category = cls.classify_source(url)
        authority = cls.AUTHORITY_CONFIG[category]
        
        # Apply time decay if published date is known
        if published_date:
            current_weight = cls._apply_time_decay(
                authority.base_weight,
                published_date,
                authority.decay_half_life_days
            )
        else:
            current_weight = authority.base_weight
            
        return current_weight, authority.evidence_type
    
    @classmethod
    def _apply_time_decay(cls, base_weight: float, published_date: datetime, half_life_days: int) -> float:
        """Apply exponential decay to evidence weight based on age"""
        age_days = (datetime.now() - published_date).days
        
        # Exponential decay formula: weight = base_weight * (0.5)^(age/half_life)
        decay_factor = 0.5 ** (age_days / half_life_days)
        decayed_weight = base_weight * decay_factor
        
        # Ensure minimum weight of 0.1
        return max(decayed_weight, 0.1)
    
    @classmethod
    def calculate_evidence_diversity(cls, sources: List[str]) -> float:
        """
        Calculate diversity score based on variety of source types
        
        Returns score from 0.0 to 1.0
        """
        if not sources:
            return 0.0
            
        categories = set()
        evidence_types = set()
        
        for source in sources:
            category = cls.classify_source(source)
            categories.add(category)
            evidence_types.add(cls.AUTHORITY_CONFIG[category].evidence_type)
        
        # Diversity based on number of unique categories and evidence types
        category_diversity = len(categories) / len(SourceCategory)
        type_diversity = len(evidence_types) / len(EvidenceType)
        
        # Weight category diversity more heavily
        diversity_score = (0.7 * category_diversity) + (0.3 * type_diversity)
        
        return min(diversity_score, 1.0)
    
    @classmethod
    def is_local_source(cls, url: str, destination_country_code: Optional[str] = None) -> bool:
        """
        Determine if a source is local to the destination
        
        Args:
            url: Source URL
            destination_country_code: ISO country code of destination
            
        Returns:
            True if source appears to be local
        """
        # Check for country-specific TLDs
        if destination_country_code:
            country_tld = f".{destination_country_code.lower()}"
            if country_tld in url.lower():
                return True
        
        # Check for local indicators
        local_indicators = [
            'local', 'native', 'resident', 'municipal',
            'community', 'neighborhood', 'district'
        ]
        
        url_lower = url.lower()
        return any(indicator in url_lower for indicator in local_indicators) 