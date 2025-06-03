from typing import List, Any
import logging
from abc import ABC, abstractmethod

from src.core.destination_classifier import SourceStrategy, DestinationType, DestinationClassifier

logger = logging.getLogger(__name__)

class SourceDiscoveryStrategy(ABC):
    """Abstract base class for source discovery strategies."""
    
    @abstractmethod
    def get_sources(self, location: str, search_tool: Any) -> List[str]:
        """Abstract method to get sources based on the strategy."""
        pass

class GlobalHubStrategy(SourceDiscoveryStrategy):
    """For Global Hubs (e.g., Paris, Tokyo, NYC) - filters quality from abundance."""

    def get_sources(self, location: str, search_tool: Any) -> List[str]:
        self.logger.info(f"Applying Global Hub strategy for {location}")
        # Example: prioritize well-known international travel sites, official tourism boards
        queries = [
            f"official tourism {location} site:*.gov OR site:*.org",
            f"best attractions {location} travel guide",
            f"cultural experiences {location} top sites"
        ]
        all_urls = []
        for query in queries:
            results = search_tool.run_query(query=query, num_results=5) # Assuming run_query method
            all_urls.extend([r['url'] for r in results])
        return all_urls

class RegionalDestinationStrategy(SourceDiscoveryStrategy):
    """For Regional Destinations (e.g., Bend, Oregon) - comprehensive local + selective national."""

    def get_sources(self, location: str, search_tool: Any) -> List[str]:
        self.logger.info(f"Applying Regional Destination strategy for {location}")
        # Example: prioritize local blogs, community sites, then national travel sites
        queries = [
            f"local blog {location} community",
            f"things to do {location} local guide",
            f"unique experiences {location} site:*.com OR site:*.org"
        ]
        all_urls = []
        for query in queries:
            results = search_tool.run_query(query=query, num_results=5)
            all_urls.extend([r['url'] for r in results])
        return all_urls

class BusinessHubStrategy(SourceDiscoveryStrategy):
    """For Business Hubs (e.g., Gurgaon, India) - local expertise + business intelligence."""

    def get_sources(self, location: str, search_tool: Any) -> List[str]:
        self.logger.info(f"Applying Business Hub strategy for {location}")
        # Example: prioritize business news, corporate travel blogs, local professional groups
        queries = [
            f"business travel {location} guide",
            f"corporate events {location} networking",
            f"expat community {location} local insights"
        ]
        all_urls = []
        for query in queries:
            results = search_tool.run_query(query=query, num_results=5)
            all_urls.extend([r['url'] for r in results])
        return all_urls

class RemoteGetawayStrategy(SourceDiscoveryStrategy):
    """For Remote Getaways (e.g., Patagonia) - niche community + local guides."""

    def get_sources(self, location: str, search_tool: Any) -> List[str]:
        self.logger.info(f"Applying Remote Getaway strategy for {location}")
        # Example: prioritize adventure blogs, specialized tour operators, forums for specific activities
        queries = [
            f"adventure travel {location} hiking guide",
            f"eco-tourism {location} remote lodges",
            f"travel forum {location} hidden gems"
        ]
        all_urls = []
        for query in queries:
            results = search_tool.run_query(query=query, num_results=5)
            all_urls.extend([r['url'] for r in results])
        return all_urls


class RegionalSourceStrategy:
    """Manages different source strategies by region/type."""

    def __init__(self, llm_interface: Any = None):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.llm_interface = llm_interface
        self.strategies = {
            SourceStrategy.FILTER_QUALITY_FROM_ABUNDANCE: GlobalHubStrategy(),
            SourceStrategy.COMPREHENSIVE_LOCAL_SELECTIVE_NATIONAL: RegionalDestinationStrategy(),
            SourceStrategy.BUSINESS_FOCUSED_PRACTICAL: BusinessHubStrategy(),
            SourceStrategy.ULTRA_LOCAL_NICHE_EXPERT: RemoteGetawayStrategy()
        }
    
    def get_strategy_for_location(self, location: str, destination_type: DestinationType) -> SourceDiscoveryStrategy:
        """Return appropriate strategy based on destination type."""
        strategy_enum = DestinationClassifier().get_source_strategy(destination_type) # Use DestinationClassifier
        self.logger.info(f"Determined strategy {strategy_enum.value} for {location} (type: {destination_type.value})")
        return self.strategies.get(strategy_enum, RegionalDestinationStrategy()) # Default to regional 