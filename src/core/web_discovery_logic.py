import asyncio
import aiohttp
import logging
from typing import Dict, List, Optional, Any
from retry import retry
from bs4 import BeautifulSoup
from tqdm import tqdm # For synchronous loops
from tqdm.asyncio import tqdm as asyncio_tqdm # For asyncio.gather
import backoff
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import requests
import json
from urllib.parse import urlparse
import re
import hashlib
from nltk import sent_tokenize

# Adjusted import path for caching module
from ..caching import read_from_cache, write_to_cache, CACHE_EXPIRY_DAYS, PAGE_CONTENT_CACHE_EXPIRY_DAYS

class WebDiscoveryLogic:
    """Core logic for web discovery using Brave Search API with caching."""
    
    def __init__(self, api_key: str, config: Dict[str, Any]):
        self.api_key = api_key
        self.config = config  # Assign config to self.config
        self.logger = logging.getLogger("app.web_discovery") 
        self.logger.setLevel(logging.DEBUG) # Ensure DEBUG level for detailed logs
        self.session = self._create_robust_session()
        # Get web_discovery specific settings, providing defaults if keys are missing
        wd_config = self.config.get("web_discovery", {})
        self.cache_settings = self.config.get("caching", {})
        
        self.query_templates = [
            # Core destination categories - ENHANCED for local customs
            "{destination} local customs traditions etiquette culture",
            "{destination} neighborhood culture local life experiences",
            "{destination} community activities local events festivals",
            "{destination} local expressions slang social norms",
            "{destination} indigenous culture aboriginal traditions",
            "{destination} multicultural communities ethnic neighborhoods",
            "{destination} local habits lifestyle daily routines",
            "{destination} street culture urban life social scenes",
            
            # Traditional discovery queries (restored as requested)
            "{destination} best restaurants food scene dining guide",
            "{destination} top attractions must see places visit", 
            "{destination} entertainment nightlife bars clubs",
            "{destination} museums galleries arts culture sites",
            "{destination} shopping districts markets local stores",
            "{destination} outdoor activities nature parks hiking",
            "{destination} family activities kids children fun",
            
            # Enhanced local discovery queries (restored as requested)
            "{destination} travel guide complete visitor information",
            "{destination} insider tips local secrets recommendations",
            "{destination} neighborhoods districts areas explore",
            "{destination} day trips nearby attractions excursions",
            
            # Keep original queries (restored as requested)
            "what makes {destination} special unique attractions",
            "hidden gems {destination} local recommendations", 
            "{destination} culture traditions authentic experiences",
        ]
        
        # Thematic query templates for focused content discovery - ENHANCED LOCAL FOCUS
        self.thematic_query_templates = {
            "local_customs": [
                # PRIMARY FOCUS: Local customs and social norms
                "{destination} local customs traditions social etiquette",
                "{destination} social norms cultural etiquette behavior",
                "{destination} greeting customs handshakes social interactions",
                "{destination} tipping culture service expectations customs",
                "{destination} dress code cultural norms appropriate clothing",
                "{destination} conversation topics cultural sensitivity taboos",
                "{destination} business etiquette professional customs meetings",
                "{destination} dining etiquette table manners local customs",
                "{destination} gift giving customs cultural traditions protocols",
                "{destination} time concepts punctuality cultural expectations",
                "{destination} personal space cultural norms social distance",
                "{destination} local expressions idioms slang phrases",
                # Generic patterns that work for any destination
                "{destination} coffee culture cafe etiquette local customs",
                "{destination} beach culture surf etiquette social norms",
                "{destination} pub culture drinking customs social traditions",
                "{destination} local language expressions common phrases"
            ],
            "community_activities": [
                # FOCUS: Community events and local activities
                "{destination} community events local gatherings festivals",
                "{destination} neighborhood activities community centers programs",
                "{destination} volunteer opportunities community involvement",
                "{destination} local sports clubs community teams",
                "{destination} community gardens local agriculture initiatives",
                "{destination} neighborhood walks tours community guides",
                "{destination} local markets community vendors events",
                "{destination} cultural festivals community celebrations",
                "{destination} community workshops classes local skills",
                "{destination} neighborhood associations community groups",
                "{destination} local meetups social groups activities",
                "{destination} community volunteering local causes",
                # Generic community activities for any destination
                "{destination} community festivals local celebrations events",
                "{destination} neighborhood groups community activities",
                "{destination} local sports culture community involvement",
                "{destination} markets community vendors local culture"
            ],
            "neighborhood_culture": [
                # FOCUS: Neighborhood-specific cultural insights
                "{destination} neighborhoods local character community identity",
                "{destination} district culture local personalities communities",
                "{destination} local communities ethnic neighborhoods diversity",
                "{destination} street art local artists community expression",
                "{destination} neighborhood stories local history legends",
                "{destination} community spaces local gathering places",
                "{destination} local businesses neighborhood character shops",
                "{destination} residential culture local living experiences",
                "{destination} neighborhood traditions local customs",
                "{destination} community landmarks local significance places",
                # Generic neighborhood patterns for any destination
                "{destination} historic districts local culture community character",
                "{destination} alternative culture local scene neighborhoods",
                "{destination} historical community local stories districts",
                "{destination} beach culture local community lifestyle areas",
                "{destination} inner city local culture community activities",
                "{destination} suburban local lifestyle community areas"
            ],
            "local_activities": [
                # FOCUS: Authentic local activities and experiences (RESTORED)
                "{destination} local activities authentic experiences off beaten path",
                "{destination} seasonal activities local traditions community events",
                "{destination} weekend activities local lifestyle experiences",
                "{destination} free local activities community experiences",
                "{destination} unique experiences only locals know about",
                "{destination} local workshops artisan crafts community classes",
                "{destination} community sports local games traditional activities",
                "{destination} local music scene community venues performances",
                "{destination} outdoor activities local recreation community spaces",
                "{destination} local fitness culture community activities",
                "{destination} artisan markets local crafts community vendors",
                "{destination} community education local learning opportunities",
                # Generic local activities for any destination
                "{destination} harbor swimming local swimming spots culture",
                "{destination} bushwalking local tracks community groups",
                "{destination} cycling culture local routes community rides",
                "{destination} outdoor fitness culture local beach workouts"
            ],
            "activities": [
                "{destination} outdoor activities adventure sports",
                "{destination} family activities kids children",
                "{destination} shopping districts markets stores",
                "{destination} tours excursions day trips",
                # Enhanced local/niche queries
                "{destination} unique experiences only locals know",
                "{destination} seasonal activities local traditions",
                "{destination} artisan workshops local crafts",
                "{destination} community activities local culture"
            ],
            "food_dining": [
                # RESTORED: Enhanced food and dining options
                "{destination} best restaurants local cuisine dining",
                "{destination} street food markets food scene", 
                "{destination} bars breweries pubs local drinks",
                "{destination} cafes coffee shops breakfast",
                # Enhanced local/niche queries (RESTORED)
                "{destination} hidden gems local favorites restaurants",
                "{destination} traditional dishes regional specialties",
                "{destination} local markets vendors food scene",
                "{destination} hole in the wall authentic eats",
                # Additional local food culture
                "{destination} local cuisine traditional dishes regional specialties",
                "{destination} food customs dining etiquette local traditions",
                "{destination} ethnic cuisine multicultural food neighborhoods",
                "{destination} local drinks traditional beverages culture"
            ],
            "attractions": [
                # RESTORED: Enhanced attractions with more features
                "{destination} top attractions tourist sites must see",
                "{destination} museums galleries cultural attractions",
                "{destination} historic landmarks monuments heritage",
                "{destination} parks gardens outdoor spaces nature",
                # Enhanced local/niche queries (RESTORED)
                "{destination} off beaten path hidden attractions",
                "{destination} local secrets insider recommendations",
                "{destination} lesser known historical sites",
                "{destination} neighborhood gems community favorites",
                # Additional attraction features
                "{destination} local attractions community significance cultural sites",
                "{destination} neighborhood landmarks local history community",
                "{destination} off beaten path hidden local gems",
                "{destination} cultural sites local significance community importance",
                "{destination} local artists galleries community art spaces"
            ],
            "entertainment": [
                # RESTORED: Entertainment category
                "{destination} nightlife entertainment districts bars",
                "{destination} live music venues concerts shows",
                "{destination} theaters performances arts events",
                "{destination} festivals events calendar entertainment",
                # Enhanced local/niche queries (RESTORED)
                "{destination} underground music scene venues",
                "{destination} local festivals community events",
                "{destination} quirky unique entertainment experiences",
                "{destination} arts scene galleries local artists",
                # Additional entertainment options
                "{destination} local entertainment community venues events",
                "{destination} neighborhood entertainment local scene community",
                "{destination} community performances local arts culture",
                "{destination} local nightlife community gathering places"
            ],
            # New category for hyper-local content
            "local_culture": [
                "{destination} local traditions customs culture",
                "{destination} community gatherings local events",
                "{destination} historical stories local legends",
                "{destination} dialect slang local expressions",
                "{destination} local heroes famous residents",
                "{destination} neighborhood characters stories"
            ]
        }
        
        # Priority-focused query templates - ENHANCED with local customs
        self.priority_query_templates = {
            "local_etiquette": [
                # NEW PRIORITY: Local customs and etiquette for travelers
                "{destination} cultural etiquette visitors travelers customs",
                "{destination} social norms tourists should know customs",
                "{destination} dos and donts cultural mistakes avoid",
                "{destination} tipping etiquette service customs travelers",
                "{destination} greeting customs cultural norms visitors",
                "{destination} dress code cultural expectations tourists",
                "{destination} conversation topics cultural sensitivity travelers",
                "{destination} business etiquette cultural norms visitors",
                "{destination} dining etiquette cultural customs tourists",
                "{destination} cultural faux pas mistakes tourists avoid"
            ],
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
        
        self.search_results_count = wd_config.get("search_results_per_query", 10) # Defaulted to 10 based on config
        self.min_content_length = wd_config.get("min_content_length_chars", 200) # Updated default
        self.max_page_content_bytes = wd_config.get("max_page_content_bytes", 2 * 1024 * 1024) # Default 2MB
        self.brave_cache_expiry = self.cache_settings.get("brave_search_expiry_days", 7)
        self.page_cache_expiry = wd_config.get("page_content_expiry_days", 30)
        self.max_sources_for_agent = wd_config.get("max_sources_for_agent_processing", 15) 
        self.max_urls_to_fetch_content_for = wd_config.get("max_urls_to_fetch_content_for", 20) # Explicitly initialize this attribute
        
        # NEW: Read granular controls for candidate pool sizing
        self.max_thematic_queries_per_theme = wd_config.get("max_thematic_queries_per_theme", 4)
        self.max_thematic_urls_to_fetch_per_theme = wd_config.get("max_thematic_urls_to_fetch_per_theme", 5)
        self.max_general_results_to_process_per_query = wd_config.get("max_general_results_to_process_per_query", 5)
        self.max_thematic_sources_to_select_per_theme = wd_config.get("max_thematic_sources_to_select_per_theme", 5)
        self.max_priority_sources_to_select_per_type = wd_config.get("max_priority_sources_to_select_per_type", 5)

        # NEW: Read raw HTML caching setting
        self.cache_raw_html_content = wd_config.get("cache_raw_html_content", True)

        # Priority settings
        self.priority_settings = config.get("priority_settings", {})
        self.enable_priority_discovery = self.priority_settings.get("enable_priority_discovery", True)
        self.priority_weights = self.priority_settings.get("priority_weights", {
            "local_etiquette": 1.6,  # NEW: Highest priority for local customs
            "safety": 1.5,
            "cost": 1.3,
            "health": 1.2,
            "accessibility": 1.1,
            "weather": 1.0
        })
        
        # Enhanced settings
        self.timeout_seconds = wd_config.get('timeout_seconds', 30)
        self.max_retries = 3
        self.retry_delay = 1
        self.supported_content_types = ['text/html', 'application/xhtml+xml']
        self.supported_encodings = ['utf-8', 'latin-1', 'cp1252', 'iso-8859-1']
        self.noisy_elements = [
            "script", "style", "nav", "footer", "header", "aside",
            "form", "noscript", "iframe", "svg", "img", "button",
            "meta", "link", "input", "select", "textarea"
        ]
        self.content_selectors = [
            'article[class*="content"]', 'main[class*="content"]',
            'article', 'main', '.post-content', '.entry-content',
            'div[class*="content"]', 'div[class*="main"]',
            'div[class*="article"]', '[role="main"]',
            'div[class*="blog"]', 'div[class*="post"]'
        ]
        
        # Quality thresholds
        self.min_quality_score = 0.6  # Increased from 0.5
        self.min_paragraph_length = 25 # Increased from 20
        self.max_boilerplate_ratio = 0.25 # Tightened from 0.3
    
    def _create_robust_session(self):
        session = requests.Session()
        retry_strategy = Retry(
            total=3,  # number of retries
            backoff_factor=1,  # wait 1, 2, 4 seconds between retries
            status_forcelist=[429, 500, 502, 503, 504],  # retry on these status codes
            allowed_methods=["GET", "HEAD"]
        )
        adapter = HTTPAdapter(max_retries=retry_strategy)
        session.mount("http://", adapter)
        session.mount("https://", adapter)
        session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        })
        return session

    @retry(tries=3, delay=2, backoff=2, jitter=(1,3))
    async def _fetch_page_content(self, url: str, destination_name: Optional[str] = None) -> Optional[str]:
        """Fetches page content using aiohttp, with robust error handling and timeout."""
        # Use aiohttp for asynchronous fetching
        html_text = None
        cache_key_raw_html = None

        if self.cache_raw_html_content:
            # Construct a cache key for the raw HTML content
            # Using a simple prefix and the URL. Ensure the URL is safe for a filename or use a hash.
            # For simplicity, let's assume the caching utility handles problematic characters or we use a hash.
            # Example: cache_key_raw_html = f"raw_html_{hashlib.md5(url.encode()).hexdigest()}"
            # For now, using a simpler key, assuming caching utility handles it.
            cache_key_raw_html = ["raw_html_content", url] 
            self.logger.debug(f"Attempting to load raw HTML for {url} from cache with key: {cache_key_raw_html}")
            cached_html = read_from_cache(cache_key_raw_html, self.page_cache_expiry)
            if cached_html:
                self.logger.info(f"CACHE HIT: Loaded raw HTML for {url} from cache (Length: {len(cached_html)}).")
                html_text = cached_html
            else:
                self.logger.debug(f"CACHE MISS: Raw HTML for {url} not found in cache.")

        if html_text is None: # If not loaded from cache, proceed to live fetch
            if not self.session:
                self.logger.warning("No session available for content fetching")
                return None
            
            self.logger.info(f"LIVE FETCH: Fetching page content for {url}")
            try:
                async with self.session.get(
                    url,
                    timeout=float(self.timeout_seconds),
                    allow_redirects=True,
                    headers={
                        'User-Agent': 'Mozilla/5.0 (compatible; DestinationBot/1.0; +http://example.com/bot)',
                        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9',
                        'Accept-Language': 'en-US,en;q=0.9',
                        'Accept-Encoding': 'gzip, deflate, br'
                    }
                ) as resp:
                    if resp.status != 200:
                        self.logger.warning(f"Failed to fetch {url}, status: {resp.status}")
                        return None
                    
                    content_type = resp.headers.get('content-type', '').lower()
                    if not any(supported_type in content_type for supported_type in self.supported_content_types):
                        self.logger.info(f"Skipping unsupported content type at {url} (type: {content_type})")
                        return None
                    
                    try:
                        html_bytes = await asyncio.wait_for(
                            resp.content.read(self.max_page_content_bytes),
                            timeout=20
                        )
                        self.logger.debug(f"Successfully read {len(html_bytes)} bytes from {url}")
                        
                        decode_errors = []
                        for encoding in self.supported_encodings:
                            try:
                                fetched_html_text_live = html_bytes.decode(encoding)
                                if fetched_html_text_live:
                                    html_text = fetched_html_text_live # Assign to the main html_text variable
                                    break
                            except UnicodeDecodeError as e:
                                decode_errors.append(f"{encoding}: {str(e)}")
                        
                        if not html_text:
                            self.logger.warning(
                                f"Could not decode content from {url} with any encoding. "
                                f"Errors: {'; '.join(decode_errors)}"
                            )
                            return None
                        
                        # If live fetch was successful and caching is enabled, save to cache
                        if self.cache_raw_html_content and html_text and cache_key_raw_html:
                            self.logger.info(f"CACHE WRITE: Saving raw HTML for {url} to cache (Length: {len(html_text)}).")
                            write_to_cache(cache_key_raw_html, html_text)
                            
                    except asyncio.TimeoutError:
                        self.logger.warning(f"Timeout during content read for {url}")
                        return None
                    except Exception as e:
                        self.logger.error(f"Error reading content from {url}: {e}", exc_info=True)
                        return None
                        
            except aiohttp.ClientError as e:
                self.logger.warning(f"Client error fetching {url}: {e}")
                return None
            except asyncio.TimeoutError:
                self.logger.warning(f"Connection timeout for {url}")
                return None
            except Exception as e:
                self.logger.error(f"Unexpected error fetching {url}: {e}", exc_info=True)
                return None

        # Ensure html_text is not None before proceeding to parsing
        if html_text is None:
            self.logger.warning(f"HTML text is None for {url} after fetch/cache attempt. Returning empty.")
            return None

        # Process HTML content (moved out of the live fetch block)
        try:
            soup = BeautifulSoup(html_text, 'html.parser')
            
            for noisy_tag in soup(self.noisy_elements):
                noisy_tag.decompose()
            
            text_parts = []
            main_content_element = None
            
            for selector in self.content_selectors:
                main_content_element = soup.select_one(selector)
                if main_content_element:
                    break
            
            if not main_content_element:
                main_content_element = soup.find('body')
                self.logger.debug(f"No specific content selector matched for {url}, using body tag.")
            else:
                # Safely get and format classes for logging
                class_list = main_content_element.get("class", [])
                class_str = ' '.join(class_list) if class_list else 'none'
                self.logger.debug(f"Using selector '{main_content_element.name}' with classes '{class_str}' for {url}")
            
            if main_content_element:
                for element in main_content_element.find_all(['p', 'h1', 'h2', 'h3', 'h4', 'h5', 'li', 'div'], recursive=True):
                    if element.name == 'div':
                        if not any(cls in element.get('class', []) for cls in ['content', 'article', 'post', 'text', 'story']):
                            if len(element.find_all(['p', 'h1', 'h2', 'h3', 'h4', 'h5', 'li'])) == 0:
                                continue
                    
                    text = element.get_text(separator=' ', strip=True)
                    if text and len(text.split()) >= 3:  
                        if element.name.startswith('h'):
                            text = f"\n\n{text}\n"
                        elif element.name == 'p':
                            text = f"{text}\n\n"
                        elif element.name == 'li':
                            text = f"• {text}\n"
                        text_parts.append(text)
                
                full_text = ' '.join(text_parts)
                cleaned_text = ' '.join(full_text.split())
                self.logger.debug(f"Parsed text length for {url}: {len(cleaned_text)} characters.")
                
                if cleaned_text:
                    is_relevant = self._validate_content_relevance(cleaned_text, destination_name, url)
                    if is_relevant:
                        quality_score = self._calculate_quality_score(cleaned_text, url)
                        if quality_score >= self.min_quality_score:
                            self.logger.info(
                                f"Successfully extracted and validated content from {url}: "
                                f"{len(cleaned_text)} chars, quality score: {quality_score:.2f}"
                            )
                            return cleaned_text
                        else:
                            self.logger.info(
                                f"Content quality score too low for {url}: {quality_score:.2f} (min: {self.min_quality_score}). Discarding."
                            )
                    else:
                        self.logger.info(f"Content validation failed for {url} (relevance/boilerplate). Discarding.")
                else:
                    self.logger.info(f"Cleaned text is empty for {url}. Discarding.")
            
            self.logger.info(f"No substantial content extracted or validated from {url} after parsing.")
            return None
        except Exception as e:
            self.logger.error(f"Error processing HTML content for {url}: {e}", exc_info=True)
            return None

    async def __aenter__(self):
        self.session = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=float(self.timeout_seconds)),
            headers={
                'User-Agent': 'Mozilla/5.0 (compatible; DestinationBot/1.0; +http://example.com/bot)',
                'Accept-Language': 'en-US,en;q=0.9'
            }
        )
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()
    
    @retry(tries=3, delay=2, backoff=2, jitter=(1,3))
    async def _fetch_brave_search(self, query: str, destination_country_code: Optional[str] = "US") -> List[Dict]:
        """
        Fetches search results from Brave Search API for a given query.
        Includes new parameters for result_filter, extra_snippets, and freshness.
        Potentially enriches location results with details from Local Search API.
        """
        # Construct cache key as a list of parts
        cache_key_parts = ["brave_search", query, destination_country_code or "US"]
        # Add other dynamic parameters to cache_key_parts if they vary per call and should influence caching
        # For example, if result_filter, freshness, etc., could change for the same query/country:
        # wd_config = self.config.get("web_discovery", {})
        # cache_key_parts.append(f"filter_{wd_config.get('brave_result_filter', 'default')}")
        # cache_key_parts.append(f"freshness_{wd_config.get('brave_freshness', 'any')}")

        use_cache = self.cache_settings.get("use_cache", True)
        if use_cache:
            # Use CACHE_EXPIRY_DAYS from src.caching for expiry
            cached_data = read_from_cache(cache_key_parts, CACHE_EXPIRY_DAYS)
            if cached_data:
                self.logger.debug(f"Cache hit for Brave search query: {query} with parts: {cache_key_parts}")
                return cached_data

        headers = {
            "Accept": "application/json",
            "X-Subscription-Token": self.api_key
        }
        
        wd_config = self.config.get("web_discovery", {})
        result_filter = wd_config.get("brave_result_filter")
        # Convert boolean config values to string "true"/"false" for the API
        extra_snippets_bool = wd_config.get("brave_extra_snippets", False) # Default to False if not present
        extra_snippets_str = "true" if extra_snippets_bool else "false"
        
        text_decorations_bool = wd_config.get("brave_text_decorations", True) # Default to True
        text_decorations_str = "true" if text_decorations_bool else "false"
        
        spellcheck_bool = wd_config.get("brave_spellcheck", True) # Default to True
        spellcheck_str = "true" if spellcheck_bool else "false"
        
        freshness = wd_config.get("brave_freshness")
        
        params = {
                    "q": query,
            "country": destination_country_code or "US",
            "search_lang": wd_config.get("brave_search_lang", "en"),
            "ui_lang": wd_config.get("brave_ui_lang", "en-US"),
            "count": wd_config.get("brave_search_count", 20),
            "offset": wd_config.get("brave_search_offset", 0),
            "safesearch": wd_config.get("brave_safesearch", "moderate"),
            "text_decorations": text_decorations_str, # Use string version
            "spellcheck": spellcheck_str # Use string version
        }

        if result_filter:
            params["result_filter"] = result_filter
        # extra_snippets is a boolean in API, but needs to be string "true"/"false" for query param
        params["extra_snippets"] = extra_snippets_str # Use string version
        
        if freshness:
            params["freshness"] = freshness
            
        self.logger.debug(f"Brave Search API request params: {params}")
        raw_web_search_results = []

        try:
            async with self.session.get("https://api.search.brave.com/res/v1/web/search", headers=headers, params=params) as response:
                response.raise_for_status()
                data = await response.json()
                
                if not data or (not any(k in data for k in ['web', 'discussions', 'locations', 'news'])):
                    self.logger.warning(f"Brave Search for '{query}' returned no substantive results. Data: {data}")
                    return [] 
                
                if 'web' in data and data['web'].get('results'):
                    raw_web_search_results.extend(data['web']['results'])
                for key in ['discussions', 'locations', 'news', 'videos', 'faq', 'infobox']:
                    if key in data and data[key].get('results'):
                        for item in data[key]['results']:
                            item['result_type'] = key 
                        raw_web_search_results.extend(data[key]['results'])
                
                # --- Integration of Local Search API --- 
                location_ids_to_fetch = [] 
                if "locations" in (result_filter or ""):
                    for result in raw_web_search_results:
                        # Assuming the ID is present in a field like 'location_id' or 'id' for location type results
                        # This part is speculative and depends on the actual structure of Brave Web Search API's location results
                        if result.get('result_type') == 'locations' and result.get('id'):
                            location_ids_to_fetch.append(result['id']) 
                        elif result.get('result_type') == 'locations' and result.get('local_id'): # Another common pattern
                             location_ids_to_fetch.append(result['local_id'])
                
                if location_ids_to_fetch:
                    self.logger.info(f"Found {len(location_ids_to_fetch)} location IDs from web search to enrich: {location_ids_to_fetch}")
                    detailed_location_data = await self._fetch_brave_local_search_by_ids(list(set(location_ids_to_fetch))) # Use set to ensure unique IDs
                    
                    # Merge or associate detailed_location_data with raw_web_search_results
                    # This is a complex step. For now, we can log or store them separately.
                    # A simple approach: add to the original location result item
                    if detailed_location_data:
                        enriched_results_temp = []
                        id_to_detail_map = {loc_detail.get('id'): loc_detail for loc_detail in detailed_location_data} 
                        
                        for result in raw_web_search_results:
                            loc_id = result.get('id') or result.get('local_id')
                            if result.get('result_type') == 'locations' and loc_id and loc_id in id_to_detail_map:
                                result['local_search_details'] = id_to_detail_map[loc_id]
                                self.logger.debug(f"Enriched location {loc_id} with local search details.")
                            enriched_results_temp.append(result)
                        raw_web_search_results = enriched_results_temp

                if use_cache:
                    # Correct call to write_to_cache with list of key parts and data
                    write_to_cache(cache_key_parts, raw_web_search_results)
                
                self.logger.info(f"Brave Search successful for query: {query}. Found {len(raw_web_search_results)} total items after potential enrichment.")
                return raw_web_search_results
        except aiohttp.ClientResponseError as e:
            self.logger.error(f"Brave Search API request failed for query '{query}': {e.status} {e.message} - {await response.text()}")
            if e.status == 401: self.logger.error("Brave API Key is invalid or expired.")
            elif e.status == 429: self.logger.error("Brave API rate limit exceeded.")
            return []
        except Exception as e:
            self.logger.error(f"An unexpected error occurred during Brave Search for query '{query}': {e}", exc_info=True)
            return []

    @retry(tries=3, delay=2, backoff=2, jitter=(1,3))
    async def _fetch_brave_local_search_by_ids(self, location_ids: List[str]) -> List[Dict]:
        """
        Fetches detailed information for given location IDs using Brave Local Search API.
        Optionally also fetches AI-generated descriptions for the same locations.
        """
        if not location_ids:
            self.logger.debug("No location IDs provided to _fetch_brave_local_search_by_ids.")
            return []

        # Fetch POI details
        poi_details = await self._fetch_brave_pois(location_ids)
        if not poi_details:
            return []  # No point in fetching descriptions if POIs failed

        wd_config = self.config.get("web_discovery", {})
        fetch_descriptions = wd_config.get("brave_fetch_local_descriptions", False)

        if fetch_descriptions:
            descriptions = await self._fetch_brave_local_descriptions(location_ids)
            if descriptions:
                # Create a map for easy lookup
                description_map = {desc.get('id'): desc.get('description') for desc in descriptions}
                # Enrich poi_details with descriptions
                for poi in poi_details:
                    if poi.get('id') in description_map:
                        poi['ai_description'] = description_map[poi.get('id')]
                        self.logger.debug(f"Enriched POI {poi.get('id')} with AI description.")

        return poi_details

    async def _fetch_brave_pois(self, location_ids: List[str]) -> List[Dict]:
        """Fetches POI data from https://api.search.brave.com/res/v1/local/pois"""
        # This function is refactored from the original _fetch_brave_local_search_by_ids
        # to specifically handle the /pois endpoint.
        wd_config = self.config.get("web_discovery", {})
        search_lang = wd_config.get("brave_local_search_lang", "en")
        ui_lang = wd_config.get("brave_local_ui_lang", "en-US")
        units = wd_config.get("brave_local_units")

        headers = {"Accept": "application/json", "X-Subscription-Token": self.api_key}
        params = {
            "ids": ",".join(location_ids),
            "search_lang": search_lang,
            "ui_lang": ui_lang,
        }
        if units:
            params["units"] = units

        self.logger.info(f"Fetching Brave POIs for {len(location_ids)} IDs.")
        try:
            async with self.session.get("https://api.search.brave.com/res/v1/local/pois", headers=headers,
                                        params=params) as response:
                response.raise_for_status()
                data = await response.json()
                local_results = data.get("results", data if isinstance(data, list) else [])
                if not local_results:
                    self.logger.warning(f"Brave POI Search for IDs '{location_ids}' returned no results.")
                    return []
                self.logger.info(f"Brave POI Search successful for {len(local_results)} IDs.")
                return local_results
        except aiohttp.ClientResponseError as e:
            self.logger.error(f"Brave POI Search API request failed for IDs '{location_ids}': {e.status} {e.message}")
            return []
        except Exception as e:
            self.logger.error(f"An unexpected error occurred during Brave POI Search for IDs '{location_ids}': {e}",
                             exc_info=True)
            return []

    async def _fetch_brave_local_descriptions(self, location_ids: List[str]) -> List[Dict]:
        """Fetches AI-generated descriptions from https://api.search.brave.com/res/v1/local/descriptions"""
        headers = {"Accept": "application/json", "X-Subscription-Token": self.api_key}
        params = {"ids": ",".join(location_ids)}

        self.logger.info(f"Fetching Brave Local Descriptions for {len(location_ids)} IDs.")
        try:
            async with self.session.get("https://api.search.brave.com/res/v1/local/descriptions", headers=headers,
                                        params=params) as response:
                response.raise_for_status()
                data = await response.json()
                descriptions = data.get("results", data if isinstance(data, list) else [])
                if not descriptions:
                    self.logger.warning(f"Brave Local Descriptions for IDs '{location_ids}' returned no results.")
                    return []
                self.logger.info(
                    f"Brave Local Descriptions successful for {len(descriptions)} IDs.")
                return descriptions
        except aiohttp.ClientResponseError as e:
            self.logger.error(
                f"Brave Local Descriptions API request failed for IDs '{location_ids}': {e.status} {e.message}")
            return []
        except Exception as e:
            self.logger.error(
                f"An unexpected error occurred during Brave Local Descriptions for IDs '{location_ids}': {e}",
                exc_info=True)
            return [] 

    def generate_priority_focused_queries(self, destination: str) -> Dict[str, List[str]]:
        """Generate priority-focused queries for critical traveler concerns"""
        priority_queries = {}
        
        for priority_type, templates in self.priority_query_templates.items():
            priority_queries[priority_type] = [
                template.format(destination=destination) 
                for template in templates
            ]
        
        return priority_queries
    
    async def discover_priority_content(self, destination: str, priority_types: List[str] = None, destination_country_code: Optional[str] = "US") -> Dict[str, List[Dict]]:
        """Discover content focused on specific traveler priorities"""
        if priority_types is None:
            priority_types = list(self.priority_query_templates.keys())
        
        self.logger.info(f"🎯 Starting priority content discovery for: {destination} (Country: {destination_country_code})")
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
                search_results = await self._fetch_brave_search(query, destination_country_code=destination_country_code)
                type_results.extend(search_results)
            
            # Fetch content for unique URLs
            processed_urls = set() # Keep track of URLs processed within this priority type
            content_tasks = []
            metadata_list = []
            
            # Deduplicate URLs before fetching content
            unique_search_results_for_type = list({result['url']: result for result in type_results}.values())

            # Limit how many unique URLs we fetch content for this priority type
            # self.max_urls_to_fetch_content_for is a general cap on total fetches, 
            # but here we are fetching per priority type. Let's use a reasonable limit or make it configurable.
            # For now, using a portion of search_results_count or a new config.
            # Let's assume self.max_priority_sources_to_select_per_type can guide the fetch limit here too.
            urls_to_fetch_for_priority_type = unique_search_results_for_type[:self.max_urls_to_fetch_content_for] 

            for result_meta in urls_to_fetch_for_priority_type:
                url = result_meta["url"]
                # This processed_urls is local to the function/priority_type, which is fine.
                if url not in processed_urls: 
                    content_tasks.append(self._fetch_page_content(url, destination))
                    metadata_list.append(result_meta) # metadata_list corresponds to content_tasks
                    processed_urls.add(url)
            
            sources_with_content_for_type = []
            if content_tasks:
                fetched_contents = await asyncio_tqdm.gather(
                    *content_tasks,
                    desc=f"Fetching Priority Content for {priority_type}",
                    unit="url"
                )
                
                for i, content in enumerate(fetched_contents):
                    current_metadata = metadata_list[i] # Get current metadata for logging in both cases
                    if content and len(content) >= self.min_content_length:
                        source = current_metadata.copy()
                        source["content"] = content
                        source["content_length"] = len(content)
                        source["priority_type"] = priority_type 
                        source["priority_weight"] = self.priority_weights.get(priority_type, 1.0)
                        sources_with_content_for_type.append(source)
                        self.logger.debug(f"    Added content for {priority_type} URL: {source['url']} (Length: {len(content)})")
                    else:
                        # Use current_metadata['url'] for logging when content is discarded
                        self.logger.debug(f"    Discarded content for {priority_type} URL: {current_metadata['url']} (Length: {len(content or [])}, Min: {self.min_content_length})")
                
            # Sort by content length
            sources_with_content_for_type.sort(key=lambda x: x["content_length"], reverse=True)
            # This function returns candidates; discover_real_content will do further selection based on its slots.
            priority_results[priority_type] = sources_with_content_for_type
            
            self.logger.info(f"✅ Found {len(priority_results[priority_type])} sources with content for {priority_type}")
        
        await asyncio.sleep(5.0) # Pause after priority content
        
        return priority_results
    
    async def discover_real_content(self, destination: str) -> List[Dict]:
        """Enhanced discovery with thematic balance - now prioritizing thematic content"""
        self.logger.info(f"🌐 Starting enhanced thematic content discovery for: {destination}")
        processed_urls = set() 
        
        total_slots = self.max_sources_for_agent
        
        # Define tourist-specific query templates
        # These will be used if the destination is considered specific (not a broad country)
        self.tourist_destination_query_templates = [ # Renamed from tourist_city_query_templates
            "best tourist attractions in {primary_destination_name}",
            "top things to do for visitors in {primary_destination_name}",
            "must-see sights in {primary_destination_name} for tourists",
            "unique {primary_destination_name} experiences for travelers",
            "{primary_destination_name} sightseeing tours and guides",
            "{primary_destination_name} travel guide for first-time visitors",
            "exploring {primary_destination_name} as a tourist what to see",
            "{primary_destination_name} iconic landmarks and hidden gems for tourists",
            "itinerary for {primary_destination_name} tourist visit",
            "{primary_destination_name} cultural experiences for tourists"
        ]

        primary_destination_name = self._extract_primary_destination_name(destination)
        extracted_destination_country_code = self._extract_country_code_from_destination(destination)

        tourist_specific_slots = 0
        # Determine if the query is for a specific destination (not just a country)
        # Heuristic: if it contains a comma, or if it's a multi-word name assumed to be specific (e.g., "Yellowstone National Park")
        # A very short name without a comma might be a country, e.g., "France"
        is_specific_destination_query = (',' in destination) or (len(destination.split()) > 1 and len(destination) > 4)

        if is_specific_destination_query:
            tourist_specific_slots = int(total_slots * 0.40) 
            thematic_slots = int(total_slots * 0.30)
            priority_slots = int(total_slots * 0.20) 
            general_slots = total_slots - thematic_slots - priority_slots - tourist_specific_slots
        else: # For broader queries (e.g., a whole country entered as the destination)
            thematic_slots = int(total_slots * 0.65) 
            priority_slots = int(total_slots * 0.25)
        general_slots = total_slots - thematic_slots - priority_slots
            # No tourist_specific_slots for broad country-level queries
        
        general_slots = max(0, general_slots)
        
        self.logger.info(
            f"🎯 Content discovery allocation for '{destination}' (Country: {extracted_destination_country_code}, Specific: {is_specific_destination_query}): "
            f"{tourist_specific_slots} tourist, {thematic_slots} thematic, "
            f"{general_slots} general, {priority_slots} priority = {total_slots} total (target)"
        )
        
        tourist_sources_candidates = []
        thematic_sources_candidates = []
        general_sources_candidates = []
        priority_sources_candidates = []
        
        # STEP 0: Get Tourist-Specific Content (VERY HIGH PRIORITY for specific destinations)
        if is_specific_destination_query and tourist_specific_slots > 0:
            self.logger.info(f"🏖️ Step 0: Discovering Tourist-Specific content for: {primary_destination_name}")
            tourist_search_results_to_fetch = []
            
            current_tourist_templates = self.tourist_destination_query_templates

            for template in current_tourist_templates:
                query = template.format(primary_destination_name=primary_destination_name)
                search_results = await self._fetch_brave_search(query, destination_country_code=extracted_destination_country_code)
                for result in search_results[:self.max_general_results_to_process_per_query]: 
                    if result["url"] not in processed_urls:
                        result['content_type'] = 'tourist' 
                        result['priority_weight'] = 2.0 # Highest inherent weight
                        tourist_search_results_to_fetch.append(result)
                        processed_urls.add(result["url"])
            
            content_fetch_tasks_tourist = []
            metadata_for_tourist_content = []
            for source_meta in tourist_search_results_to_fetch:
                content_fetch_tasks_tourist.append(self._fetch_page_content(source_meta["url"], primary_destination_name))
                metadata_for_tourist_content.append(source_meta)

            self.logger.info(f"📄 Fetching content for {len(content_fetch_tasks_tourist)} tourist-specific URLs for {primary_destination_name}...")
            if content_fetch_tasks_tourist:
                fetched_tourist_contents = await asyncio_tqdm.gather(
                    *content_fetch_tasks_tourist,
                    desc=f"Fetching Tourist Content for {primary_destination_name}",
                    unit="url"
                )
                for i, content in enumerate(fetched_tourist_contents):
                    source_metadata = metadata_for_tourist_content[i]
                    if content and len(content) >= self.min_content_length:
                        source_metadata["content"] = content
                        source_metadata["content_length"] = len(content)
                        tourist_sources_candidates.append(source_metadata)
            self.logger.info(f"✅ Tourist-specific step: Gathered {len(tourist_sources_candidates)} tourist source candidates with content for {primary_destination_name}")

        await asyncio.sleep(5.0) # Pause after tourist content

        # STEP 1: Get thematic content (HIGH PRIORITY)
        if hasattr(self, 'thematic_query_templates') and thematic_slots > 0: 
            self.logger.info(f"🎨 Step 1: Discovering thematic content candidates (Max queries per theme: {self.max_thematic_queries_per_theme}, Max URLs to fetch per theme: {self.max_thematic_urls_to_fetch_per_theme})")
            thematic_results_by_theme = await self.discover_thematic_content(destination, destination_country_code=extracted_destination_country_code)
            
            for theme_category, sources_from_theme in thematic_results_by_theme.items():
                count_for_this_theme = 0
                for source in sources_from_theme: # These sources already have content
                    if source["url"] not in processed_urls:
                        if count_for_this_theme < self.max_thematic_sources_to_select_per_theme:
                            source["content_type"] = "thematic" # Ensure type
                            source["theme_category"] = theme_category # Ensure category is set
                            thematic_sources_candidates.append(source)
                            processed_urls.add(source["url"])
                            count_for_this_theme += 1
                            self.logger.debug(f"    Added THEMATIC content for '{theme_category}' URL: {source['url']} (Length: {len(source['content'])})")
            
            self.logger.info(f"✅ Thematic step: Gathered {len(thematic_sources_candidates)} thematic source candidates")
        
        await asyncio.sleep(5.0) # Pause after thematic content
        
        # STEP 2: Get general content (MEDIUM PRIORITY)
        general_search_results_to_fetch = [] 
        if general_slots > 0: # Check against allocated slots
            self.logger.info(f"🌍 Step 2: Discovering general content candidates (Max results per query: {self.max_general_results_to_process_per_query})")
            
            general_queries = self.query_templates[:6] 
            for template in tqdm(general_queries, desc=f"General queries for {destination}"):
                query = template.format(destination=destination)
                search_results = await self._fetch_brave_search(query, destination_country_code=extracted_destination_country_code)
                self.logger.debug(f"  General query '{query}' got {len(search_results)} Brave results.")
                
                # Process based on new config: max_general_results_to_process_per_query
                for result in search_results[:self.max_general_results_to_process_per_query]:  
                    if result["url"] not in processed_urls:
                        result['theme_type'] = 'general' # Mark for fetching
                        # result['priority_weight'] = 1.2 # Default weight for general
                        result['priority_weight'] = 0.5 # Significantly lower than thematic (1.5) or priority types
                        general_search_results_to_fetch.append(result) 
                        processed_urls.add(result["url"]) 
        
            content_fetch_tasks = []
            metadata_for_general_content = []
            for source_meta in general_search_results_to_fetch:
                content_fetch_tasks.append(self._fetch_page_content(source_meta["url"], destination))
                metadata_for_general_content.append(source_meta)

            self.logger.info(f"📄 Fetching content for {len(content_fetch_tasks)} general URLs...")
            if content_fetch_tasks:
                fetched_general_contents = await asyncio_tqdm.gather(
                    *content_fetch_tasks, 
                    desc=f"Fetching General Content for {destination}", 
                    unit="url"
                )
                for i, content in enumerate(fetched_general_contents):
                    source_metadata = metadata_for_general_content[i]
                    if content and len(content) >= self.min_content_length: 
                        source_metadata["content"] = content
                        source_metadata["content_length"] = len(content)
                        source_metadata["content_type"] = "general" 
                        general_sources_candidates.append(source_metadata)
                        self.logger.debug(f"    Added GENERAL content for URL: {source_metadata['url']} (Length: {len(content)})")
                    else:
                        self.logger.debug(f"    Discarded GENERAL content for URL: {source_metadata['url']} (Length: {len(content or [])}, Min: {self.min_content_length})")
            
            self.logger.info(f"✅ General step: Gathered {len(general_sources_candidates)} general source candidates with content")
        
        await asyncio.sleep(5.0) # Pause after general content
        
        # STEP 3: Get priority content (PRIORITY)
        if self.enable_priority_discovery and priority_slots > 0: # Check against allocated slots
            self.logger.info(f"🎯 Step 3: Discovering priority content candidates")
            
            priority_results_by_type = await self.discover_priority_content(destination, 
                                                                   priority_types=['safety', 'cost', 'local_etiquette', 'health', 'accessibility'], 
                                                                   destination_country_code=extracted_destination_country_code) 
            
            for priority_type, sources_from_type in priority_results_by_type.items():
                count_for_this_type = 0
                for source in sources_from_type: # These sources already have content
                    if source["url"] not in processed_urls:
                        if count_for_this_type < self.max_priority_sources_to_select_per_type:
                            source['content_type'] = 'priority' # Ensure type
                            # priority_type and priority_weight are set by discover_priority_content
                            priority_sources_candidates.append(source)
                            processed_urls.add(source["url"])
                            count_for_this_type += 1
            
            self.logger.info(f"✅ Priority step: Gathered {len(priority_sources_candidates)} priority source candidates with content")
        
        # STEP 4: Combine, Score, and Select Final Sources
        all_candidate_sources = tourist_sources_candidates + thematic_sources_candidates + general_sources_candidates + priority_sources_candidates
        
        self.logger.info(
            f"Combining candidates: {len(tourist_sources_candidates)} tourist, "
            f"{len(thematic_sources_candidates)} thematic, {len(general_sources_candidates)} general, "
            f"{len(priority_sources_candidates)} priority = {len(all_candidate_sources)} total candidates with content."
        )

        if not all_candidate_sources:
            self.logger.warning(f"No candidate sources found for {destination} after all discovery steps.")
            return []

        # Apply quality scoring to all candidate sources that have content
        for source in all_candidate_sources:
            # Content should exist at this point as helpers only return sources with content
            source['quality_score'] = self._calculate_source_quality_score(source) 
            # Recalculate final_priority_score based on its inherent weight and new quality_score
            inherent_weight = source.get('priority_weight', 1.0) # thematic=1.5, general=1.2, priority types vary
            source['final_priority_score'] = inherent_weight * source['quality_score']
        
        # Sort all candidates by their final_priority_score
        all_candidate_sources.sort(key=lambda x: x.get('final_priority_score', 0), reverse=True)
        
        # Select the top N sources based on the overall limit max_sources_for_agent
        final_selected_sources = all_candidate_sources[:self.max_sources_for_agent]

        # Log the final distribution and metrics for the selected sources
        thematic_final_count = len([s for s in final_selected_sources if s.get("content_type") == "thematic"])
        general_final_count = len([s for s in final_selected_sources if s.get("content_type") == "general"])
        priority_final_count = len([s for s in final_selected_sources if s.get("content_type") == "priority"])
        tourist_final_count = len([s for s in final_selected_sources if s.get("content_type") == "tourist"])
        
        avg_quality_final = sum(s.get('quality_score', 0.0) for s in final_selected_sources) / len(final_selected_sources) if final_selected_sources else 0
        
        self.logger.info(f"✅ Final selection for {destination} ({self.max_sources_for_agent} limit applied):")
        self.logger.info(
            f"   📊 Distribution: {tourist_final_count} tourist, {thematic_final_count} thematic, "
            f"{general_final_count} general, {priority_final_count} priority = {len(final_selected_sources)} total sources."
        )
        self.logger.info(f"   🏆 Average quality score of selected: {avg_quality_final:.2f}")
        
        # ... (rest of the logging as before, using final_selected_sources) ...
        
        self._log_discovery_metrics(final_selected_sources, destination) 
        
        return final_selected_sources

    def _validate_content_relevance(self, content: str, destination: str, url: str) -> bool:
        """Enhanced content validation with quality scoring"""
        try:
            # Basic validation
            if not content or len(content.strip()) < self.min_content_length:
                self.logger.info(f"Content filtered: Too short from {url}")
                return False

            # Check for destination keywords
            destination_parts = destination.lower().split(',')
            city = destination_parts[0].strip()
            country = destination_parts[-1].strip() if len(destination_parts) > 1 else ""
            
            content_lower = content.lower()
            
            # Check for destination mentions
            if city not in content_lower and (country and country not in content_lower):
                self.logger.info(f"Content filtered: No destination keywords found in {url}")
                return False

            # Check for boilerplate content
            boilerplate_patterns = [
                r'cookie[s]? policy',
                r'privacy policy',
                r'terms of service',
                r'accept all cookies',
                r'website uses cookies',
                r'sign up for newsletter'
            ]
            
            boilerplate_matches = sum(1 for pattern in boilerplate_patterns if re.search(pattern, content_lower))
            # Adjusted boilerplate check: Consider total lines. If content is very short, this ratio might be sensitive.
            # For now, increasing the ratio to be more lenient.
            if len(content.split('\n')) > 0 and (boilerplate_matches / len(content.split('\n'))) > self.max_boilerplate_ratio:
                self.logger.info(f"Content filtered: Too much boilerplate content (by line ratio) in {url}")
                return False

            # Check for meaningful paragraphs - Relaxed this check significantly
            paragraphs = [p.strip() for p in content.split('\n') if p.strip()] # Re-split by single newline for paragraph definition
            # Instead of one long paragraph, check for a decent amount of total text in paragraphs
            total_meaningful_words = 0
            min_words_for_paragraph_to_count = 5 # Lowered from 20
            for p_text in paragraphs:
                words_in_p = len(p_text.split())
                if words_in_p >= min_words_for_paragraph_to_count:
                    total_meaningful_words += words_in_p
            
            # Require at least, say, 100 meaningful words in total from shortish paragraphs, or one longer one
            if not any(len(p.split()) >= 20 for p in paragraphs) and total_meaningful_words < 100:
                self.logger.info(f"Content filtered: Not enough meaningful paragraph content (total words < 100 from paras >={min_words_for_paragraph_to_count} words, and no single para >= 20 words) in {url}")
                return False

            return True

        except Exception as e:
            self.logger.error(f"Error validating content from {url}: {e}")
            return False

    def _calculate_quality_score(self, content: str, url: str) -> float:
        """Calculates a quality score for the given content."""
        
        lines = [line.strip() for line in content.split('\\n') if line.strip()]
        if not lines:
            return 0.0
            
        num_words = len(content.split())
        num_sentences = len(sent_tokenize(content)) if num_words > 0 else 0
        avg_sentence_length = num_words / num_sentences if num_sentences > 0 else 0
        
        # Penalize very short content more heavily
        if num_words < 50:
            length_score = (num_words / 50.0) * 0.2 # Max 0.2 for very short content
        else:
            length_score = min(1.0, num_words / 1000.0) * 0.4 # Weight 0.4

        # Reward complexity, but don't over-penalize short sentences
        complexity_score = min(1.0, avg_sentence_length / 25.0) * 0.3 # Weight 0.3

        # Keyword density as a measure of focus
        # This is a simple example; a more advanced version could use TF-IDF
        keywords = ['attractions', 'activities', 'guide', 'visit', 'eat', 'see', 'do', 'experience']
        keyword_count = sum(1 for word in keywords if word in content.lower())
        keyword_score = min(1.0, keyword_count / 5.0) * 0.3 # Weight 0.3

        # Combine scores
        total_score = length_score + complexity_score + keyword_score
        
        self.logger.debug(f"Quality score for {url}: {total_score:.4f} "
                          f"(Length: {length_score:.2f}, Complexity: {complexity_score:.2f}, "
                          f"Keywords: {keyword_score:.2f})")
                          
        return total_score

    def _calculate_source_quality_score(self, source: Dict) -> float:
        """Calculate quality score for a source"""
        score = 1.0
        url = source.get("url", "").lower()
        title = source.get("title", "").lower()
        
        # Boost authoritative domains
        authority_domains = [
            "wikipedia.org", "tripadvisor.com", "lonelyplanet.com",
            "timeout.com", "fodors.com", "frommers.com", 
            "visitnsw.com", "sydney.com", "australia.com",
            "gov.au", "museum", "official"
        ]
        
        for domain in authority_domains:
            if domain in url:
                score += 0.5
                break
        
        # Boost local and niche content indicators
        local_indicators = [
            "local", "hidden", "secret", "insider", "neighborhood", 
            "community", "authentic", "traditional", "locals", "native",
            "indigenous", "heritage", "historic", "unique", "quirky",
            "off beaten path", "lesser known", "underground", "artisan"
        ]
        
        local_boost = 0
        for indicator in local_indicators:
            if indicator in title:
                local_boost += 0.1  # Each local indicator adds 0.1
                
        score += min(local_boost, 0.4)  # Cap local boost at 0.4
        
        # Boost local domain extensions and city-specific sites
        local_domain_patterns = [
            ".gov.au", ".com.au", ".org.au",  # Australian domains
            "visit", "discover", "explore",   # Tourism sites
            "local", "city", "town"           # Local sites
        ]
        
        for pattern in local_domain_patterns:
            if pattern in url:
                score += 0.2
                break
        
        # Penalize news sites with non-destination content
        news_domains = ["travelandtourworld.com", "cnn.com", "bbc.com"]
        for domain in news_domains:
            if domain in url and "travel advisory" in title:
                score -= 0.3
        
        # Boost destination-specific content
        destination_indicators = ["guide", "attractions", "restaurants", "hotels", "visit"]
        for indicator in destination_indicators:
            if indicator in title:
                score += 0.2
                break
        
        return max(0.1, min(2.5, score))  # Increased max to 2.5 to allow for local content boost

    async def discover_thematic_content(self, destination: str, destination_country_code: Optional[str] = "US") -> Dict[str, List[Dict]]:
        """Discover content organized by themes"""
        self.logger.info(f"🎨 Starting thematic content discovery for: {destination} (Country: {destination_country_code})")
        thematic_results = {} 
        
        for theme_category, templates in self.thematic_query_templates.items():
            self.logger.info(f"Searching for thematic category '{theme_category}' content...")
            theme_search_results_raw = [] 
            
            for template in templates[:self.max_thematic_queries_per_theme]: 
                query = template.format(destination=destination)
                search_results = await self._fetch_brave_search(query, destination_country_code=destination_country_code)
                self.logger.debug(f"  Thematic query for '{theme_category}': '{query}' got {len(search_results)} Brave results.")
                
                for result in search_results: 
                    result['theme_category'] = theme_category 
                    result['priority_weight'] = 1.5  # Default weight for thematic content
                    theme_search_results_raw.append(result)
            
            # Deduplicate URLs before fetching content for this theme category
            unique_urls_for_theme_meta = list({s['url']: s for s in theme_search_results_raw}.values())

            # Limit how many unique URLs we fetch content for, per theme category
            urls_to_fetch_meta = unique_urls_for_theme_meta[:self.max_thematic_urls_to_fetch_per_theme]
            
            content_fetch_tasks = [self._fetch_page_content(meta['url'], destination) for meta in urls_to_fetch_meta]
            
            sources_with_content_for_theme = []
            if content_fetch_tasks:
                fetched_theme_contents = await asyncio.gather(*content_fetch_tasks)
                
                for i, content in enumerate(fetched_theme_contents):
                    if content and len(content) >= self.min_content_length:
                        source_metadata = urls_to_fetch_meta[i].copy() 
                        source_metadata['content'] = content
                        source_metadata['content_length'] = len(content)
                        sources_with_content_for_theme.append(source_metadata)
                        self.logger.debug(f"    Added THEMATIC content for '{theme_category}' URL: {source_metadata['url']} (Length: {len(content)})")
                    else:
                        self.logger.debug(f"    Discarded THEMATIC content for '{theme_category}' URL: {urls_to_fetch_meta[i]['url']} (Length: {len(content or [])}, Min: {self.min_content_length})")
                
            thematic_results[theme_category] = sources_with_content_for_theme
            self.logger.info(f"✅ Found {len(sources_with_content_for_theme)} sources with content for theme '{theme_category}' from {len(urls_to_fetch_meta)} URLs fetched.")
        
        return thematic_results

    def _log_discovery_metrics(self, all_sources: List[Dict], destination: str):
        """Log discovery metrics for monitoring and optimization"""
        if not all_sources:
            self.logger.warning(f"No sources discovered for {destination}")
            return
        
        # Content type distribution
        content_types = {}
        theme_categories = {}
        priority_types = {}
        quality_scores = []
        content_lengths = []
        
        for source in all_sources:
            # Content type analysis
            content_type = source.get('content_type', 'unknown')
            content_types[content_type] = content_types.get(content_type, 0) + 1
            
            # Theme category analysis (for thematic content)
            if content_type == 'thematic':
                theme_cat = source.get('theme_category', 'unknown')
                theme_categories[theme_cat] = theme_categories.get(theme_cat, 0) + 1
            
            # Priority type analysis (for priority content)
            if content_type == 'priority':
                priority_type = source.get('priority_type', 'unknown')
                priority_types[priority_type] = priority_types.get(priority_type, 0) + 1
            
            # Quality and content metrics
            quality_scores.append(source.get('quality_score', 1.0))
            content_lengths.append(len(source.get('content', '')))
        
        # Calculate metrics
        avg_quality = sum(quality_scores) / len(quality_scores)
        avg_content_length = sum(content_lengths) / len(content_lengths)
        
        # Content quality assessment
        boilerplate_count = 0
        for source in all_sources:
            content = source.get('content', '').lower()
            if any(pattern in content for pattern in ['cookie', 'privacy policy', 'terms of service']):
                boilerplate_count += 1
        
        boilerplate_percentage = (boilerplate_count / len(all_sources)) * 100
        
        # Log comprehensive metrics
        self.logger.info(f"📊 Discovery metrics for {destination}:")
        self.logger.info(f"   📈 Content distribution: {content_types}")
        
        if theme_categories:
            self.logger.info(f"   🎨 Thematic breakdown: {theme_categories}")
        
        if priority_types:
            self.logger.info(f"   🎯 Priority breakdown: {priority_types}")
        
        self.logger.info(f"   🏆 Quality metrics:")
        self.logger.info(f"      • Average quality score: {avg_quality:.2f}")
        self.logger.info(f"      • Average content length: {avg_content_length:.0f} chars")
        self.logger.info(f"      • Boilerplate content: {boilerplate_percentage:.1f}%")
        
        # Performance insights
        thematic_count = content_types.get('thematic', 0)
        general_count = content_types.get('general', 0)
        priority_count = content_types.get('priority', 0)
        tourist_count = content_types.get('tourist', 0)
        
        self.logger.info(f"   📋 Performance insights:")
        if thematic_count > 0:
            self.logger.info(f"      • Thematic content prioritization: {'✅ Working' if thematic_count >= priority_count else '⚠️ Needs adjustment'}")
        if tourist_count > 0: 
            # Adjusted logic: Consider tourist content good if it's a decent portion of non-priority content
            non_priority_total = tourist_count + thematic_count + general_count
            tourist_ratio_of_non_priority = tourist_count / non_priority_total if non_priority_total > 0 else 0
            self.logger.info(f"      • Tourist-specific content presence: {'✅ Good portion' if tourist_ratio_of_non_priority > 0.3 else '⚠️ Low tourist content (' + str(round(tourist_ratio_of_non_priority*100)) + '% of non-priority), consider adjusting slots/queries'}")
        
        if avg_quality > 1.2:
            self.logger.info(f"      • Source quality: ✅ Good (>{avg_quality:.2f})")
        elif avg_quality > 1.0:
            self.logger.info(f"      • Source quality: ⚠️ Moderate ({avg_quality:.2f})")
        else:
            self.logger.info(f"      • Source quality: ❌ Poor ({avg_quality:.2f})")
        
        if boilerplate_percentage < 20:
            self.logger.info(f"      • Content relevance: ✅ Good ({boilerplate_percentage:.1f}% boilerplate)")
        else:
            self.logger.info(f"      • Content relevance: ⚠️ Needs filtering ({boilerplate_percentage:.1f}% boilerplate)")
        
        # Optimization recommendations
        recommendations = []
        if thematic_count > 0 and thematic_count < priority_count: 
            recommendations.append("Increase thematic content allocation or review priority content weight")
        
        if tourist_count > 0: # Check tourist_count > 0
            non_priority_total_rec = tourist_count + thematic_count + general_count
            tourist_ratio_rec = tourist_count / non_priority_total_rec if non_priority_total_rec > 0 else 0
            if tourist_ratio_rec < 0.33: # If less than a third of non-priority is tourist
                 recommendations.append("Boost tourist-specific query impact or slot allocation for specific destination queries")
        elif tourist_count == 0:
            recommendations.append("No tourist-specific content found for specific destination query; check templates/slots.")

        if avg_quality < 1.2:
            recommendations.append("Improve source quality scoring")
        if boilerplate_percentage > 25:
            recommendations.append("Enhance content relevance filtering")
        if avg_content_length < 1000:
            recommendations.append("Target longer, more comprehensive content")
        
        if recommendations:
            self.logger.info(f"   💡 Optimization recommendations:")
            for rec in recommendations:
                self.logger.info(f"      • {rec}") 

    def _extract_country_code_from_destination(self, destination: str) -> Optional[str]:
        """Extracts a potential country code (e.g., US, FR) from a destination string."""
        if ',' in destination:
            parts = destination.split(',')
            last_part = parts[-1].strip()
            if len(last_part) == 2 and last_part.isalpha():
                return last_part.upper()
            # More comprehensive country name to code mapping
            country_map_simple = {
                "united states": "US", "usa": "US", "u.s.": "US", "u.s.a.": "US",
                "france": "FR",
                "united kingdom": "GB", "uk": "GB", "britain": "GB",
                "canada": "CA",
                "germany": "DE",
                "japan": "JP",
                "australia": "AU",
                "italy": "IT",
                "spain": "ES",
                "china": "CN"
                # Add more common names as needed
            }
            if last_part.lower() in country_map_simple:
                return country_map_simple[last_part.lower()]
        self.logger.debug(f"Could not determine country code from destination: {destination}, defaulting to US for search.")
        return "US" # Default to US if not determinable, to ensure Brave search has a country

    def _extract_primary_destination_name(self, destination: str) -> str:
        """Extracts the primary, most specific part of the destination name."""
        if ',' in destination:
            # Return the part before the first comma, assuming it's the most specific
            return destination.split(',')[0].strip() 
        # If no comma, assume the whole string is the primary destination name (e.g., "Tuscany", "Yellowstone National Park")
        return destination.strip()