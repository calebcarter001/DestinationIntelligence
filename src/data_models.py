from dataclasses import dataclass, field
from typing import List
from datetime import datetime

@dataclass
class DestinationInsight:
    """Individual insight about a destination"""
    insight_type: str
    insight_name: str 
    description: str
    confidence_score: float
    evidence_sources: List[str] = field(default_factory=list) # Real URLs
    content_snippets: List[str] = field(default_factory=list) # Actual content
    is_discovered_theme: bool = False
    created_date: str = None
    tags: List[str] = field(default_factory=list)  # Tags for categorization
    
    def __post_init__(self):
        if self.created_date is None:
            self.created_date = datetime.now().isoformat() 