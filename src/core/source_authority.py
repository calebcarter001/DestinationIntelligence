import re
from urllib.parse import urlparse

# Define authority tiers and their weights
# These are examples; expand and refine this list significantly
KNOWN_AUTHORITY_DOMAINS = {
    # High Authority (Official, Major News, Academic)
    "wikipedia.org": 0.9,
    "bbc.com": 0.85,
    "reuters.com": 0.85,
    "apnews.com": 0.85,
    "nytimes.com": 0.8,
    "theguardian.com": 0.8,
    "nationalgeographic.com": 0.8,
    ".gov": 0.9,  # Government sites
    ".edu": 0.75, # Educational institutions
    "who.int": 0.9, # World Health Organization
    "unesco.org": 0.85, # UNESCO

    # Medium Authority (Reputable specific interest, established blogs)
    "lonelyplanet.com": 0.7,
    "tripadvisor.com": 0.6, # User-generated but moderated, widely used
    "fodors.com": 0.7,
    "frommers.com": 0.7,
    "cntraveler.com": 0.7, # Conde Nast Traveler

    # Low Authority (General blogs, forums - use with caution or for diversity)
    "blogspot.com": 0.3,
    "wordpress.com": 0.3, # Free hosted blogs
    # Add more specific known low-authority domains if necessary
}

# Default weight for unknown sources
DEFAULT_AUTHORITY_WEIGHT = 0.2

def get_domain(url: str) -> str:
    """Extract the main domain from a URL (e.g., 'example.com')."""
    try:
        parsed_url = urlparse(url)
        netloc = parsed_url.netloc
        if not netloc:
            return ""
        
        parts = netloc.split('.')
        if len(parts) > 1:
            # Handle cases like www.example.co.uk -> example.co.uk
            # and example.com -> example.com
            # This is a simplification; robust domain parsing can be complex.
            if parts[-2] in ['co', 'com', 'org', 'net', 'gov', 'edu'] and len(parts) > 2:
                return '.'.join(parts[-3:]) 
            return '.'.join(parts[-2:])
        return netloc
    except Exception:
        return ""

def get_authority_weight(url: str) -> float:
    """
    Get the authority weight for a given URL based on its domain.
    """
    if not url:
        return DEFAULT_AUTHORITY_WEIGHT

    domain = get_domain(url)
    if not domain:
        return DEFAULT_AUTHORITY_WEIGHT

    # Check for exact domain match
    if domain in KNOWN_AUTHORITY_DOMAINS:
        return KNOWN_AUTHORITY_DOMAINS[domain]

    # Check for TLD matches (like .gov, .edu)
    for tld, weight in KNOWN_AUTHORITY_DOMAINS.items():
        if domain.endswith(tld) and tld.startswith('.'): # Ensure it's a TLD pattern
            return weight
            
    # Check for subdomain of known authority (e.g. travel.nytimes.com)
    for known_domain, weight in KNOWN_AUTHORITY_DOMAINS.items():
        if not known_domain.startswith('.') and domain.endswith(known_domain): # e.g. domain = 'sub.nytimes.com', known_domain = 'nytimes.com'
             return weight


    return DEFAULT_AUTHORITY_WEIGHT

if __name__ == '__main__':
    # Test cases
    urls_to_test = [
        ("https://www.nytimes.com/section/travel", 0.8),
        ("http://example.blogspot.com/article.html", 0.3),
        ("https://www.officialsite.gov/info", 0.9),
        ("https://www.unknownsite.com", 0.2),
        ("https://www.wikipedia.org/wiki/Chicago", 0.9),
        ("https://www.lonelyplanet.com/usa/chicago", 0.7),
        ("http://some.subdomain.nyc.gov",0.9),
        ("http://news.stanford.edu/story",0.75)
    ]
    for url, expected_weight in urls_to_test:
        weight = get_authority_weight(url)
        print(f"URL: {url}, Domain: {get_domain(url)}, Weight: {weight}, Expected: {expected_weight}, Match: {abs(weight - expected_weight) < 0.01}") 