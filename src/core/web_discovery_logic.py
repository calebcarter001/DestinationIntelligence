import asyncio
import aiohttp
import logging
from typing import Dict, List, Optional, Any
from retry import retry
from bs4 import BeautifulSoup
from tqdm import tqdm # For synchronous loops
from tqdm.asyncio import tqdm as asyncio_tqdm # For asyncio.gather

# Adjusted import path for caching module
from ..caching import read_from_cache, write_to_cache, CACHE_EXPIRY_DAYS, PAGE_CONTENT_CACHE_EXPIRY_DAYS

class WebDiscoveryLogic:
    """Core logic for web discovery using Brave Search API with caching."""
    
    def __init__(self, api_key: str, config: Dict[str, Any]):
        self.api_key = api_key
        self.logger = logging.getLogger(__name__ + '.WebDiscoveryLogic') # Updated logger name
        self.session = None
        # Get web_discovery specific settings, providing defaults if keys are missing
        wd_config = config.get("web_discovery", {})
        self.cache_settings = config.get("caching", {})
        
        self.query_templates = [
            "what makes {destination} special unique attractions",
            "hidden gems {destination} local recommendations", 
            "{destination} culture traditions authentic experiences",
            "{destination} vs similar destinations comparison",
            "why visit {destination} travel guide highlights"
        ]
        
        # Priority-focused query templates
        self.priority_query_templates = {
            "safety": [
                "{destination} crime rate tourist areas safety",
                "{destination} travel advisory warnings dangerous areas avoid",
                "{destination} tourist police emergency contacts safety tips",
                "{destination} safe neighborhoods recommended areas tourists",
                "{destination} scams tourist safety concerns warnings"
            ],
            "cost": [
                "{destination} daily budget cost breakdown travelers",
                "{destination} cheap eats budget accommodation backpacker",
                "{destination} price comparison neighboring countries costs",
                "{destination} seasonal prices peak off-season costs",
                "{destination} free activities budget travel tips money"
            ],
            "health": [
                "{destination} vaccination requirements health risks travelers",
                "{destination} hospitals medical facilities quality healthcare",
                "{destination} water safety food hygiene stomach problems",
                "{destination} disease outbreaks health warnings CDC",
                "{destination} travel insurance medical emergency costs"
            ],
            "weather": [
                "{destination} best time visit weather monthly climate",
                "{destination} rainy season hurricane monsoon weather patterns",
                "{destination} temperature rainfall sunshine hours monthly",
                "{destination} weather affect travel plans activities",
                "{destination} climate change extreme weather events"
            ],
            "accessibility": [
                "{destination} visa requirements entry process tourists",
                "{destination} direct flights major cities connections",
                "{destination} public transport infrastructure getting around",
                "{destination} english spoken language barrier communication",
                "{destination} disabled access wheelchair friendly facilities"
            ]
        }
        
        self.search_results_count = wd_config.get("search_results_per_query", 5)
        self.min_content_length = wd_config.get("min_content_length_chars", 200) # Updated default
        # Use the new byte limit config key
        self.max_page_content_bytes = wd_config.get("max_page_content_bytes", 2 * 1024 * 1024) # Default 2MB
        self.brave_cache_expiry = self.cache_settings.get("brave_search_expiry_days", 7)
        self.page_cache_expiry = self.cache_settings.get("page_content_expiry_days", 30)
        self.max_sources_for_agent = wd_config.get("max_sources_for_agent_processing", 5)
        
        # Priority settings
        self.priority_settings = config.get("priority_settings", {})
        self.enable_priority_discovery = self.priority_settings.get("enable_priority_discovery", True)
        self.priority_weights = self.priority_settings.get("priority_weights", {
            "safety": 1.5,
            "cost": 1.3,
            "health": 1.2,
            "accessibility": 1.1,
            "weather": 1.0
        })
    
    async def __aenter__(self):
        self.session = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=15),
            headers={'User-Agent': 'Mozilla/5.0 (compatible; DestinationBot/1.0; +http://example.com/bot)'}
        )
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()
    
    @retry(tries=3, delay=2, backoff=2, jitter=(1,3))
    async def _fetch_brave_search(self, query: str) -> List[Dict]:
        """Fetch real results from Brave Search API, using cache."""
        cache_key_parts = ["brave_search", query]
        cached_results = read_from_cache(cache_key_parts, self.brave_cache_expiry)
        if cached_results is not None:
            return cached_results

        self.logger.info(f"🔍 Brave Search (API): {query}")
        
        try:
            await asyncio.sleep(1.1) 

            async with self.session.get(
                "https://api.search.brave.com/res/v1/web/search",
                headers={"X-Subscription-Token": self.api_key, "Accept": "application/json"},
                params={
                    "q": query,
                    "count": self.search_results_count, 
                    "freshness": "pw",
                    "country": "US",
                    "spellcheck": "true"
                }
            ) as resp:
                if resp.status == 429:
                    self.logger.warning(f"Brave Search API rate limited for '{query}'. Will retry if attempts left.")
                    error_text = await resp.text()
                    raise Exception(f"Brave Search API rate limited (429): {error_text[:200]}")
                if resp.status != 200:
                    error_text = await resp.text()
                    self.logger.error(f"Brave Search API error {resp.status} for '{query}': {error_text[:200]}")
                    return [] 
                
                data = await resp.json()
                results = []
                
                for r in data.get("web", {}).get("results", []):
                    if r.get("url") and r.get("title"):
                        results.append({
                            "url": r["url"],
                            "title": r.get("title", ""),
                            "snippet": r.get("description", ""),
                            "age": r.get("age", ""),
                            "language": r.get("language", "en")
                        })
                
                self.logger.info(f"   Found {len(results)} search results from API for '{query}'")
                if results: 
                    write_to_cache(cache_key_parts, results)
                return results
                
        except aiohttp.ClientError as e: 
            self.logger.error(f"AIOHTTP ClientError during Brave Search for '{query}': {e}")
            return []
        except Exception as e:
            self.logger.error(f"Generic exception during Brave Search for '{query}': {e}")
            if "Brave Search API rate limited (429)" not in str(e): 
                 raise 
            return [] 

    @retry(tries=2, delay=1, backoff=2)
    async def _fetch_page_content(self, url: str) -> str:
        cache_key_parts = ["page_content_v2_bytes", url] # Changed cache key prefix due to logic change
        cached_content = read_from_cache(cache_key_parts, self.page_cache_expiry)
        if cached_content is not None:
            return cached_content

        self.logger.info(f"📄 Fetching content (WEB): {url} (limit {self.max_page_content_bytes} bytes)")
        html_text = ""
        try:
            async with self.session.get(url, timeout=aiohttp.ClientTimeout(total=20), allow_redirects=True) as resp: # Increased timeout for larger reads
                if resp.status != 200:
                    self.logger.warning(f"Failed to fetch {url}, status: {resp.status}")
                    return ""
                
                content_type = resp.headers.get('content-type', '').lower()
                if 'text/html' not in content_type:
                    self.logger.info(f"Skipping non-HTML content at {url} (type: {content_type})")
                    return ""
                
                try:
                    # Read up to the byte limit
                    html_bytes = await resp.content.read(self.max_page_content_bytes)
                    # Try to decode, ignoring errors. Common encodings first.
                    try:
                        html_text = html_bytes.decode('utf-8', errors='ignore')
                    except UnicodeDecodeError:
                        try:
                            html_text = html_bytes.decode('latin-1', errors='ignore')
                        except UnicodeDecodeError:
                            self.logger.warning(f"Could not decode content from {url} with utf-8 or latin-1 after reading bytes.")
                            return ""
                    if not html_text:
                        self.logger.warning(f"Content decoded to empty string for {url} after reading {len(html_bytes)} bytes.")
                        return ""

                except asyncio.TimeoutError:
                    self.logger.warning(f"Timeout during content read for {url} (limit {self.max_page_content_bytes} bytes)")
                    return ""
                except Exception as e:
                    self.logger.error(f"Error reading content bytes from {url}: {e}", exc_info=True)
                    return ""

                if not html_text: return "" # If decoding failed or produced nothing

                soup = BeautifulSoup(html_text, 'html.parser')
                
                for noisy_tag in soup(["script", "style", "nav", "footer", "header", "aside", "form", "noscript", "iframe", "svg", "img"]):
                    noisy_tag.decompose()
                
                content_selectors = [
                    'article[class*="content"]', 'main[class*="content"]', 
                    'article', 'main', '.post-content', '.entry-content', 
                    'div[class*="content"]', 'div[class*="main"]', 'div[class*="article"]',
                    '[role="main"]'
                ]
                
                text_parts = []
                main_content_element = None
                for selector in content_selectors:
                    main_content_element = soup.select_one(selector)
                    if main_content_element:
                        break
                
                if not main_content_element: 
                    main_content_element = soup.find('body')

                if main_content_element:
                    for element in main_content_element.find_all(['p', 'h1', 'h2', 'h3', 'h4', 'li', 'div'], recursive=True):
                        if element.name == 'div' and not any(cls in element.get('class', []) for cls in ['content', 'article', 'post', 'text', 'story']):
                            if len(element.find_all(['p','h1','h2','h3','h4','li'])) == 0: 
                                continue

                        text = element.get_text(separator=' ', strip=True)
                        if text and len(text.split()) > 3: 
                            text_parts.append(text)
                    
                    full_text = ' \n\n '.join(text_parts) 
                    cleaned_text = ' '.join(full_text.split())
                    # Character based truncation still happens *after* text extraction from potentially byte-limited HTML
                    # This is because self.max_page_content_chars is still what this part of logic uses.
                    # We should use a different variable for the final text character limit if desired, 
                    # or ensure this class uses one consistent way (chars or influence from bytes).
                    # For now, the byte limit is on raw HTML fetch, and this char limit is on extracted text.
                    # Let's ensure this uses a character limit that might be different from the byte limit on fetch.
                    # The config has `max_page_content_chars_per_source` which I used for `self.max_page_content_chars`.
                    # The intent was to replace it. I'll assume the character limit is now gone in spirit due to byte limit.
                    # For safety, let's apply a very large character limit here after extraction, or remove it if byte limit is primary.
                    # Re-reading: config has `max_page_content_bytes` and I used it for `self.max_page_content_bytes`.
                    # The old `self.max_page_content_chars` is still used for final_text. This needs to be harmonized.
                    # Let's assume max_page_content_chars is effectively superseded if content read was limited by bytes.
                    # The cleaned_text here is from the (potentially byte-limited) html_text.
                    final_text = cleaned_text # No further char limit here if byte limit was primary
                    # Or, if we want a *character* limit on the *extracted* text, distinct from byte limit of HTML:
                    # final_text_char_limit = wd_config.get("extracted_text_char_limit", 500000) # e.g. 500k chars
                    # final_text = cleaned_text[:final_text_char_limit]

                    if final_text:
                         write_to_cache(cache_key_parts, final_text)
                    return final_text
                
                self.logger.info(f"No substantial content extracted from {url}")
                return ""
                
        except aiohttp.ClientError as e:
            self.logger.warning(f"AIOHTTP ClientError fetching content from {url}: {e}")
            return ""
        except asyncio.TimeoutError:
            self.logger.warning(f"Timeout fetching content from {url}")
            return ""
        except Exception as e:
            self.logger.error(f"Generic error fetching/parsing content from {url}: {e}", exc_info=True)
            return ""
    
    def generate_priority_focused_queries(self, destination: str) -> Dict[str, List[str]]:
        """Generate priority-focused queries for critical traveler concerns"""
        priority_queries = {}
        
        for priority_type, templates in self.priority_query_templates.items():
            priority_queries[priority_type] = [
                template.format(destination=destination) 
                for template in templates
            ]
        
        return priority_queries
    
    async def discover_priority_content(self, destination: str, priority_types: List[str] = None) -> Dict[str, List[Dict]]:
        """Discover content focused on specific traveler priorities"""
        if priority_types is None:
            priority_types = list(self.priority_query_templates.keys())
        
        self.logger.info(f"🎯 Starting priority content discovery for: {destination}")
        priority_results = {}
        
        # Generate priority queries
        all_priority_queries = self.generate_priority_focused_queries(destination)
        
        for priority_type in priority_types:
            if priority_type not in all_priority_queries:
                continue
                
            self.logger.info(f"Searching for {priority_type} information...")
            type_results = []
            
            # Search for each query in this priority category
            for query in all_priority_queries[priority_type]:
                search_results = await self._fetch_brave_search(query)
                type_results.extend(search_results)
            
            # Fetch content for unique URLs
            processed_urls = set()
            content_tasks = []
            metadata_list = []
            
            for result in type_results:
                url = result["url"]
                if url not in processed_urls:
                    content_tasks.append(self._fetch_page_content(url))
                    metadata_list.append(result)
                    processed_urls.add(url)
            
            if content_tasks:
                fetched_contents = await asyncio.gather(*content_tasks)
                
                sources_with_content = []
                for i, content in enumerate(fetched_contents):
                    if content and len(content) >= self.min_content_length:
                        source = metadata_list[i].copy()
                        source["content"] = content
                        source["content_length"] = len(content)
                        source["priority_type"] = priority_type
                        source["priority_weight"] = self.priority_weights.get(priority_type, 1.0)
                        sources_with_content.append(source)
                
                # Sort by content length and limit
                sources_with_content.sort(key=lambda x: x["content_length"], reverse=True)
                priority_results[priority_type] = sources_with_content[:self.max_sources_for_agent]
            else:
                priority_results[priority_type] = []
            
            self.logger.info(f"✅ Found {len(priority_results[priority_type])} sources for {priority_type}")
        
        return priority_results
    
    async def discover_real_content(self, destination: str) -> List[Dict]:
        """Discover real web content about destination - now includes priority content if enabled"""
        self.logger.info(f"🌐 Starting real content discovery for: {destination}")
        all_sources_with_content = []
        
        # First, get priority-focused content if enabled
        if self.enable_priority_discovery:
            priority_results = await self.discover_priority_content(destination)
            
            # Add priority sources with their weights
            for priority_type, sources in priority_results.items():
                for source in sources:
                    all_sources_with_content.append(source)
        
        # Then, get general content using existing templates
        list_of_search_results_lists = []
        # Wrap query_templates with tqdm for search progress
        for template in tqdm(self.query_templates, desc=f"Brave Search Queries for {destination}", unit="query"):
            query = template.format(destination=destination)
            search_results = await self._fetch_brave_search(query) 
            list_of_search_results_lists.append(search_results)

        processed_urls = set()
        # Collect URLs already processed from priority searches
        for source in all_sources_with_content:
            processed_urls.add(source["url"])
        
        content_fetch_tasks = []
        search_results_for_metadata = []

        for search_results in list_of_search_results_lists:
            for result in search_results[:self.search_results_count]: 
                url = result["url"]
                if url not in processed_urls:
                    content_fetch_tasks.append(self._fetch_page_content(url))
                    search_results_for_metadata.append(result) 
                    processed_urls.add(url)

        self.logger.info(f"Fetching content for {len(content_fetch_tasks)} additional URLs for {destination}...")
        if content_fetch_tasks:
            fetched_contents = await asyncio_tqdm.gather(
                *content_fetch_tasks, 
                desc=f"Fetching URL Contents for {destination}", 
                unit="url"
            )
        else: fetched_contents = []

        for i, content in enumerate(fetched_contents):
            source_metadata = search_results_for_metadata[i]
            if content and len(content) >= self.min_content_length: 
                source_metadata["content"] = content
                source_metadata["content_length"] = len(content)
                source_metadata["priority_weight"] = 1.0  # Default weight for non-priority content
                all_sources_with_content.append(source_metadata)
        
        unique_sources_dict = {source["url"]: source for source in all_sources_with_content}
        # Sort by priority weight first, then by content length
        sorted_sources = sorted(
            list(unique_sources_dict.values()), 
            key=lambda x: (x.get("priority_weight", 1.0), x.get("content_length", 0)), 
            reverse=True
        )
        
        # Limit the number of sources returned, but ensure we have a good mix
        # Take more sources if priority discovery is enabled
        max_sources = self.max_sources_for_agent
        if self.enable_priority_discovery:
            max_sources = min(max_sources * 2, 10)  # Double the sources but cap at 10
        
        final_sources_to_return = sorted_sources[:max_sources]
        
        self.logger.info(f"✅ Discovered {len(sorted_sources)} unique sources with substantial content for {destination}. Returning top {len(final_sources_to_return)} to agent.")
        return final_sources_to_return 