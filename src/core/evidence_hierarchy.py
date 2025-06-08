from typing import Dict, List, Optional, Tuple, TYPE_CHECKING
from datetime import datetime, timedelta
from enum import Enum
import re
from dataclasses import dataclass
import logging
import math

# Forward references for type hints to avoid circular imports
if TYPE_CHECKING:
    from .enhanced_data_models import LocalAuthority

from src.schemas import AuthorityType

logger = logging.getLogger(__name__)

class EvidenceType(Enum):
    """Evidence source classification"""
    PRIMARY = "primary"      # Direct, first-hand
    SECONDARY = "secondary"  # Curated, reviewed
    TERTIARY = "tertiary"    # Referenced, cited

class SourceCategory(Enum):
    """Source category classification"""
    GOVERNMENT = "government"
    ACADEMIC = "academic"
    BUSINESS = "business"
    GUIDEBOOK = "guidebook"
    BLOG = "blog"
    SOCIAL = "social"
    UNKNOWN = "unknown"

@dataclass
class AuthorityConfig:
    """Configuration for source authority"""
    evidence_type: EvidenceType
    base_weight: float
    decay_half_life_days: int

class EvidenceHierarchy:
    """Enhanced evidence hierarchy with local authority classification"""
    
    # Authority configuration by source category
    AUTHORITY_CONFIG = {
        SourceCategory.GOVERNMENT: AuthorityConfig(EvidenceType.PRIMARY, 0.95, 730),
        SourceCategory.ACADEMIC: AuthorityConfig(EvidenceType.PRIMARY, 0.90, 1095),
        SourceCategory.BUSINESS: AuthorityConfig(EvidenceType.SECONDARY, 0.75, 365),
        SourceCategory.GUIDEBOOK: AuthorityConfig(EvidenceType.SECONDARY, 0.70, 365),
        SourceCategory.BLOG: AuthorityConfig(EvidenceType.TERTIARY, 0.50, 180),
        SourceCategory.SOCIAL: AuthorityConfig(EvidenceType.TERTIARY, 0.30, 90),
        SourceCategory.UNKNOWN: AuthorityConfig(EvidenceType.TERTIARY, 0.20, 90)
    }
    
    # URL patterns for source classification
    SOURCE_PATTERNS = {
        SourceCategory.GOVERNMENT: [
            r'\.gov',
            r'\.ca\.gov',
            r'\.state\.',
            r'\.us\.gov',
            r'gov\.uk',
            r'gouvernement',
            r'municipal',
            r'city\.',
            r'county\.',
            r'tourism\.gov'
        ],
        SourceCategory.ACADEMIC: [
            r'\.edu',
            r'\.ac\.uk',
            r'university',
            r'college',
            r'scholar\.google',
            r'researchgate',
            r'academia\.edu',
            r'journal\.',
            r'academic'
        ],
        SourceCategory.BUSINESS: [
            r'\.com\/business',
            r'\.biz',
            r'chamber\.org',
            r'business',
            r'corp\.',
            r'company',
            r'enterprise'
        ],
        SourceCategory.GUIDEBOOK: [
            r'lonelyplanet',
            r'tripadvisor',
            r'fodors',
            r'frommers',
            r'roughguides',
            r'timeout',
            r'travel\+leisure',
            r'natgeo',
            r'guidebook'
        ],
        SourceCategory.BLOG: [
            r'blog',
            r'wordpress',
            r'blogspot',
            r'medium\.com',
            r'substack',
            r'personal',
            r'diary'
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

    LOCAL_AUTHORITY_PATTERNS = {
        AuthorityType.PRODUCER: [
            r'maple.*farm', r'distillery', r'winery', r'brewery',
            r'artisan', r'craftsman', r'local.*producer', r'farm', r'vineyard',
            r'bakery', r'market.*vendor', r'local.*business.*owner'
        ],
        AuthorityType.RESIDENT: [
            r'local.*blog', r'resident.*forum', r'neighborhood',
            r'community.*group', r'nextdoor', r'local.*facebook',
            r'lived.*here', r'born.*here', r'resident.*for', r'local.*for.*years'
        ],
        AuthorityType.PROFESSIONAL: [
            r'tour.*guide', r'sommelier', r'chef', r'hospitality',
            r'local.*expert', r'concierge', r'professional.*chef',
            r'sushi.*chef', r'years.*experience', r'tourism.*board',
            r'certified.*guide', r'official.*guide', r'licensed.*guide',
            r'industry.*professional', r'hospitality.*professional',
            r'culinary.*expert', r'food.*expert', r'travel.*professional'
        ]
    }

    SEASONAL_INDICATORS = [
        r'maple.*season', r'harvest.*time', r'peak.*season',
        r'best.*time.*visit', r'seasonal.*hours', r'winter.*hours',
        r'summer.*season', r'fall.*foliage', r'spring.*opening',
        r'holiday.*hours', r'seasonal.*closure'
    ]

    @staticmethod
    def classify_source(url: str) -> SourceCategory:
        """Classify source URL into category"""
        url_lower = url.lower()
        
        for category, patterns in EvidenceHierarchy.SOURCE_PATTERNS.items():
            for pattern in patterns:
                if re.search(pattern, url_lower):
                    return category
        
        return SourceCategory.UNKNOWN
    
    @staticmethod
    def get_source_authority(url: str, timestamp: Optional[datetime] = None) -> Tuple[float, EvidenceType]:
        """Get authority weight and evidence type for source"""
        category = EvidenceHierarchy.classify_source(url)
        config = EvidenceHierarchy.AUTHORITY_CONFIG[category]
        
        base_weight = config.base_weight
        evidence_type = config.evidence_type
        
        # Apply temporal decay if timestamp provided
        if timestamp:
            days_old = (datetime.now() - timestamp).days
            decay_factor = 0.5 ** (days_old / config.decay_half_life_days)
            weight = base_weight * decay_factor
        else:
            weight = base_weight
        
        return weight, evidence_type
    
    @staticmethod
    def calculate_evidence_diversity(source_urls: List[str]) -> float:
        """Calculate diversity score for evidence sources"""
        if not source_urls:
            return 0.0
        
        # Count sources by category
        category_counts = {}
        for url in source_urls:
            category = EvidenceHierarchy.classify_source(url)
            category_counts[category] = category_counts.get(category, 0) + 1
        
        # Calculate Shannon diversity index
        total_sources = len(source_urls)
        diversity = 0.0
        
        for count in category_counts.values():
            if count > 0:
                proportion = count / total_sources
                # Use natural logarithm for Shannon diversity
                diversity -= proportion * math.log(proportion) if proportion > 0 else 0
        
        # Normalize to 0-1 scale (max diversity is log of number of categories)
        max_diversity = math.log(len(SourceCategory))
        return min(diversity / max_diversity, 1.0) if max_diversity > 0 else 0.0
    
    @staticmethod
    def is_local_source(url: str, country_code: Optional[str] = None) -> bool:
        """Determine if source appears to be local"""
        url_lower = url.lower()
        
        # Country-specific TLD patterns
        if country_code:
            country_patterns = {
                'CA': [r'\.ca(?:/|$)', r'canada'],
                'US': [r'\.us(?:/|$)', r'america', r'usa'],
                'UK': [r'\.uk(?:/|$)', r'britain', r'england'],
                'AU': [r'\.au(?:/|$)', r'australia'],
                'DE': [r'\.de(?:/|$)', r'germany', r'deutschland']
            }
            
            patterns = country_patterns.get(country_code, [])
            for pattern in patterns:
                if re.search(pattern, url_lower):
                    return True
        
        # Local keywords
        local_indicators = [
            r'local', r'community', r'neighborhood', r'resident',
            r'municipal', r'city', r'town', r'village'
        ]
        
        for pattern in local_indicators:
            if re.search(pattern, url_lower):
                return True
        
        return False
    
    @staticmethod
    def classify_local_authority(url: str, content: str) -> 'LocalAuthority':
        """Classify and create LocalAuthority from URL and content"""
        # Import here to avoid circular imports
        from .enhanced_data_models import LocalAuthority
        
        url_lower = url.lower()
        content_lower = content.lower()
        combined_text = f"{url_lower} {content_lower}"
        
        # Check for authority type patterns
        authority_type = AuthorityType.RESIDENT  # Default
        expertise_domain = "general local knowledge"
        community_validation = 0.2  # Default low validation
        local_tenure = None
        
        # Professional patterns (check first for higher priority)
        for pattern in EvidenceHierarchy.LOCAL_AUTHORITY_PATTERNS[AuthorityType.PROFESSIONAL]:
            if re.search(pattern, combined_text):
                authority_type = AuthorityType.PROFESSIONAL
                expertise_domain = "professional local services"
                community_validation = 0.9
                
                # Extract years of experience if mentioned
                experience_match = re.search(r'(\d+).*years.*experience', content_lower)
                if experience_match:
                    local_tenure = int(experience_match.group(1))
                    community_validation = min(0.95, 0.7 + (local_tenure * 0.02))  # Higher validation for more experience
                
                # Determine specific expertise domain
                if any(term in combined_text for term in ['chef', 'culinary', 'food', 'restaurant']):
                    expertise_domain = "culinary expertise"
                elif any(term in combined_text for term in ['guide', 'tour', 'tourism']):
                    expertise_domain = "tourism and local guidance"
                elif any(term in combined_text for term in ['hospitality', 'hotel', 'accommodation']):
                    expertise_domain = "hospitality services"
                break
        
        # Producer patterns
        if authority_type == AuthorityType.RESIDENT:  # Only check if not already classified as professional
            for pattern in EvidenceHierarchy.LOCAL_AUTHORITY_PATTERNS[AuthorityType.PRODUCER]:
                if re.search(pattern, combined_text):
                    authority_type = AuthorityType.PRODUCER
                    expertise_domain = "local production/artisan work"
                    community_validation = 0.8
                    break
        
        # Resident patterns (lowest priority)
        if authority_type == AuthorityType.RESIDENT:
            for pattern in EvidenceHierarchy.LOCAL_AUTHORITY_PATTERNS[AuthorityType.RESIDENT]:
                if re.search(pattern, combined_text):
                    authority_type = AuthorityType.RESIDENT
                    expertise_domain = "community knowledge"
                    community_validation = 0.6
                    
                    # Extract tenure if mentioned
                    tenure_match = re.search(r'(?:lived|been|resident).*?(\d+).*?years?', content_lower)
                    if tenure_match:
                        local_tenure = int(tenure_match.group(1))
                    break
        
        # Additional expertise domain refinement
        if 'expert' in content_lower or 'specialist' in content_lower:
            expertise_domain = f"specialized {expertise_domain}"
            community_validation = min(community_validation + 0.1, 1.0)
        
        if 'official' in content_lower or 'certified' in content_lower or 'licensed' in content_lower:
            community_validation = min(community_validation + 0.1, 1.0)
            if 'tourism' in content_lower:
                expertise_domain = "official tourism guidance"
        
        return LocalAuthority(
            authority_type=authority_type,
            local_tenure=local_tenure,
            expertise_domain=expertise_domain,
            community_validation=community_validation
        ) 