import logging
import requests
import json
from bs4 import BeautifulSoup
from ..core.llm_factory import LLMFactory
from ..core.safe_dict_utils import safe_get, safe_get_confidence_value, safe_get_nested, safe_get_dict
import calendar
from ..core.enhanced_data_models import DimensionValue, Destination
from typing import Dict, Any

def _clean_llm_json_response(response_text: str) -> Dict[str, Any]:
    """
    Cleans and parses a JSON string from an LLM response,
    handling common formatting issues like code blocks.
    """
    if not isinstance(response_text, str):
        return {}
        
    clean_text = response_text.strip().lstrip("```json").rstrip("```").strip()
    
    try:
        return json.loads(clean_text)
    except json.JSONDecodeError:
        # Fallback for more complex cleaning if needed in the future
        return {}

class BaseEnrichmentAgent:
    """Base class for agents that enrich destination data."""
    module_name = "OVERRIDE_IN_SUBCLASS"

    def __init__(self, config, llm):
        self.app_config = config
        self.enrichment_config = self.app_config.get('data_enrichment', {})
        self.llm = llm
        self.logger = logging.getLogger(f"app.enrichment.{self.__class__.__name__}")
        
        # FIXED: Proper configuration path checking
        enrichment_enabled = self.enrichment_config.get('enabled', False)
        module_enabled = self.enrichment_config.get('enrichment_modules', {}).get(self.module_name, False)
        self.is_enabled = enrichment_enabled and module_enabled
        
        self.logger.debug(f"Enrichment enabled: {enrichment_enabled}, Module '{self.module_name}' enabled: {module_enabled}, Final enabled: {self.is_enabled}")

    def run(self, destination: Destination):
        """
        Runs the enrichment process on the destination object in place.
        """
        if not self.is_enabled:
            self.logger.debug(f"'{self.module_name}' is disabled, skipping.")
            return destination
        
        self.logger.info(f"Running {self.__class__.__name__} for {destination.names[0]}...")
        try:
            self._enrich(destination)
        except Exception as e:
            self.logger.error(f"Error during {self.__class__.__name__} enrichment: {e}", exc_info=True)
        
        return destination

    def _enrich(self, destination: Destination):
        """Subclasses must implement this method to perform enrichment in place."""
        raise NotImplementedError

class DemographicAndGeographicAgent(BaseEnrichmentAgent):
    module_name = "demographic_and_geographic_lookup"

    def _enrich(self, destination: Destination):
        """
        Enriches the destination with population, area, and primary language
        by searching for and parsing its Wikipedia page.
        """
        dest_name = destination.names[0]
        self.logger.info(f"Fetching demographic/geographic data for {dest_name} from Wikipedia.")

        brave_api_key = self.app_config.get('api_keys', {}).get('brave_search')
        if not brave_api_key:
            self.logger.warning("Brave API key not found. Skipping demographic enrichment.")
            return

        search_query = f"{dest_name} wikipedia"
        search_url = "https://api.search.brave.com/res/v1/web/search"
        headers = {"Accept": "application/json", "X-Subscription-Token": brave_api_key}
        params = {"q": search_query, "count": 1}

        try:
            # 1. Find the Wikipedia URL
            response = requests.get(search_url, headers=headers, params=params, timeout=10)
            response.raise_for_status()
            search_results = response.json()
            
            wiki_url = safe_get(safe_get(safe_get(search_results, "web", {}), "results", [{}])[0], "url")
            if not wiki_url or "wikipedia.org" not in wiki_url:
                self.logger.warning(f"No valid Wikipedia page found for {dest_name}.")
                return
            
            self.logger.debug(f"Found Wikipedia URL for {dest_name}: {wiki_url}")

            # 2. Fetch and parse page content
            page_response = requests.get(wiki_url, timeout=15)
            page_response.raise_for_status()
            soup = BeautifulSoup(page_response.content, 'html.parser')
            
            infobox = soup.find('table', class_='infobox')
            page_text = infobox.get_text(separator='\n', strip=True) if infobox else ""
            
            if len(page_text) < 200:
                content_div = soup.find('div', id='mw-content-text')
                page_text += content_div.get_text(separator=' ', strip=True) if content_div else ""

            max_text_length = 25000
            page_text = page_text[:max_text_length]
            self.logger.debug(f"--- Wikipedia Text for {dest_name} (first 500 chars) ---\n{page_text[:500]}\n--------------------")

            # 3. Use LLM to extract facts
            prompt = f"""
            From the following text from a Wikipedia page for "{dest_name}", extract the following information.
            Provide the response in a valid JSON format with ONLY the following keys: "population", "area_km2", "primary_language".

            - For "population", find the total population of the main city or administrative area. Provide an integer value, removing commas.
            - For "area_km2", find the total area in square kilometers. Provide a float value, looking for the 'Total' area if multiple are listed.
            - For "primary_language", find the primary official or most spoken language. Provide a string.

            If a value cannot be found, use null for that key. Do not add any commentary or introductory text.

            Page Text:
            ---
            {page_text}
            ---
            """
            
            llm_response = self.llm.invoke(prompt)
            llm_content = llm_response if isinstance(llm_response, str) else llm_response.content
            self.logger.debug(f"LLM Raw Response for {dest_name}:\n---\n{llm_content}\n---")
            
            extracted_data = _clean_llm_json_response(llm_content)
            
            if not extracted_data:
                self.logger.warning(f"Could not parse JSON from LLM for {dest_name}.")
                return

            # Safely access the data and update the destination object
            destination.population = safe_get(extracted_data, 'population')
            destination.area_km2 = safe_get(extracted_data, 'area_km2')
            destination.primary_language = safe_get(extracted_data, 'primary_language')
            
            self.logger.info(f"Successfully enriched {dest_name} with demographic data.")

        except requests.exceptions.RequestException as e:
            self.logger.error(f"Could not fetch data for {dest_name} from network: {e}")

        # No return needed as the object is modified in place

class VibeAndGastronomyAgent(BaseEnrichmentAgent):
    module_name = "vibe_and_gastronomy_analysis"

    def _enrich(self, destination: Destination):
        """
        Analyzes web content to determine vibe descriptors and gastronomic culture.
        """
        dest_name = destination.names[0]
        self.logger.info(f"Analyzing vibe and gastronomy for {dest_name}.")
        
        prompt = f"""
        Based on general knowledge and common descriptions of "{dest_name}", provide a JSON object with two keys:
        1. "vibe_descriptors": A list of 5-7 descriptive adjectives (e.g., "historic", "modern", "relaxed", "bustling", "artsy").
        2. "gastronomic_culture": A short phrase describing the dominant food scene (e.g., "diverse street food", "classic fine dining", "farm-to-table focus").
        
        Example for a fictional city:
        {{
            "vibe_descriptors": ["coastal", "laid-back", "surf-centric", "bohemian", "sunny"],
            "gastronomic_culture": "Fresh seafood and casual beachside cafes"
        }}
        """
        
        try:
            llm_response = self.llm.invoke(prompt)
            llm_content = llm_response if isinstance(llm_response, str) else llm_response.content
            
            clean_response = llm_content.strip().lstrip("```json").rstrip("```").strip()
            extracted_data = json.loads(clean_response)
            
            if safe_get(extracted_data, 'vibe_descriptors'):
                destination.vibe_descriptors = extracted_data['vibe_descriptors']
                self.logger.info(f"Updated vibe for {dest_name}: {destination.vibe_descriptors}")

            if safe_get(extracted_data, 'gastronomic_culture'):
                destination.dimensions['gastronomic_culture'] = DimensionValue(value=extracted_data['gastronomic_culture'], confidence=0.7)
                self.logger.info(f"Updated gastronomic culture for {dest_name}: {extracted_data['gastronomic_culture']}")

        except (json.JSONDecodeError, TypeError, ValueError) as e:
            self.logger.error(f"Failed to parse LLM response for vibe/gastronomy of {dest_name}: {e}")
            self.logger.debug(f"LLM Response was: {llm_content}")
            
        return destination

class HistoricalAndCulturalAgent(BaseEnrichmentAgent):
    module_name = "historical_and_cultural_lookup"

    def _enrich(self, destination: Destination):
        """
        Enriches the destination with historical summary, UNESCO sites, and dominant religions
        by parsing its Wikipedia page.
        """
        dest_name = destination.names[0]
        self.logger.info(f"Fetching historical/cultural data for {dest_name} from Wikipedia.")

        brave_api_key = self.app_config.get('api_keys', {}).get('brave_search')
        if not brave_api_key:
            self.logger.warning("Brave API key not found. Skipping historical/cultural enrichment.")
            return
        
        search_query = f"{dest_name} wikipedia"
        search_url = "https://api.search.brave.com/res/v1/web/search"
        headers = {"Accept": "application/json", "X-Subscription-Token": brave_api_key}
        params = {"q": search_query, "count": 1}

        try:
            response = requests.get(search_url, headers=headers, params=params, timeout=10)
            response.raise_for_status()
            search_results = response.json()
            
            wiki_url = safe_get(safe_get(safe_get(search_results, "web", {}), "results", [{}])[0], "url")
            if not wiki_url or "wikipedia.org" not in wiki_url:
                self.logger.warning(f"No valid Wikipedia page found for {dest_name}.")
                return
            
            page_response = requests.get(wiki_url, timeout=15)
            page_response.raise_for_status()
            soup = BeautifulSoup(page_response.content, 'html.parser')
            page_text = soup.get_text(separator=' ', strip=True)[:25000]

            prompt = f"""
            From the following text of a Wikipedia page for "{dest_name}", extract the following information.
            Provide the response in a valid JSON format with keys: "historical_summary", "unesco_sites", "dominant_religions".

            - "historical_summary": A concise 1-2 sentence summary of the city's history.
            - "unesco_sites": A list of strings of any UNESCO World Heritage sites mentioned. If none, provide an empty list.
            - "dominant_religions": A list of strings of the primary or most practiced religions.

            Page Text:
            ---
            {page_text}
            ---
            """
            
            llm_response = self.llm.invoke(prompt)
            llm_content = llm_response if isinstance(llm_response, str) else llm_response.content
            clean_response = llm_content.strip().lstrip("```json").rstrip("```").strip()
            extracted_data = json.loads(clean_response)
            
            if safe_get(extracted_data, 'historical_summary'):
                destination.historical_summary = extracted_data['historical_summary']
                self.logger.info(f"Updated historical summary for {dest_name}.")
            if safe_get(extracted_data, 'unesco_sites'):
                destination.unesco_sites = extracted_data['unesco_sites']
                self.logger.info(f"Updated UNESCO sites for {dest_name}: {destination.unesco_sites}")
            if safe_get(extracted_data, 'dominant_religions'):
                destination.dominant_religions = extracted_data['dominant_religions']
                self.logger.info(f"Updated dominant religions for {dest_name}: {destination.dominant_religions}")

        except (requests.exceptions.RequestException, json.JSONDecodeError, TypeError, ValueError) as e:
            self.logger.error(f"Failed during historical/cultural enrichment for {dest_name}: {e}")

        return destination

class EconomicAndDevelopmentAgent(BaseEnrichmentAgent):
    module_name = "economic_and_development_lookup"

    def _enrich(self, destination: Destination):
        """
        Enriches the destination with GDP per capita and HDI.
        This is a placeholder and would ideally use a structured data source like World Bank API.
        For now, it uses an LLM to extract this from a web search.
        """
        dest_name = destination.names[0]
        country_name = destination.admin_levels.get("country", "")
        self.logger.info(f"Fetching economic data for {dest_name} / {country_name}.")
        
        prompt = f"""
        For the country of "{country_name}", provide a JSON object with the latest approximate values for the following keys:
        1. "gdp_per_capita_usd": The Gross Domestic Product per capita in USD. Provide an integer.
        2. "hdi": The Human Development Index. Provide a float.

        Example:
        {{
            "gdp_per_capita_usd": 65000,
            "hdi": 0.926
        }}
        """
        try:
            llm_response = self.llm.invoke(prompt)
            llm_content = llm_response if isinstance(llm_response, str) else llm_response.content
            clean_response = llm_content.strip().lstrip("```json").rstrip("```").strip()
            extracted_data = json.loads(clean_response)

            if safe_get(extracted_data, 'gdp_per_capita_usd'):
                destination.gdp_per_capita_usd = int(extracted_data['gdp_per_capita_usd'])
                self.logger.info(f"Updated GDP per capita for {country_name}: ${destination.gdp_per_capita_usd}")
            if safe_get(extracted_data, 'hdi'):
                destination.hdi = float(extracted_data['hdi'])
                self.logger.info(f"Updated HDI for {country_name}: {destination.hdi}")

        except (json.JSONDecodeError, TypeError, ValueError) as e:
            self.logger.error(f"Failed to parse LLM response for economic data of {country_name}: {e}")

        return destination

class TourismAndTrendAgent(BaseEnrichmentAgent):
    module_name = "tourism_and_trend_analysis"

    def _enrich(self, destination: Destination):
        """
        Enriches with annual tourist arrivals and classifies popularity.
        """
        dest_name = destination.names[0]
        self.logger.info(f"Fetching tourism data for {dest_name}.")
        
        prompt = f"""
        Based on general knowledge about tourism for "{dest_name}", provide a JSON object with keys:
        1. "annual_tourist_arrivals": A rough integer estimate of the number of annual visitors.
        2. "popularity_stage": A classification from ["emerging", "mature", "overtouristed"].
        3. "visa_info_url": The best URL for official visa information for tourists visiting this destination.

        Example:
        {{
            "annual_tourist_arrivals": 8500000,
            "popularity_stage": "mature",
            "visa_info_url": "https://travel.state.gov/content/travel/en/us-visas.html"
        }}
        """
        try:
            llm_response = self.llm.invoke(prompt)
            llm_content = llm_response if isinstance(llm_response, str) else llm_response.content
            clean_response = llm_content.strip().lstrip("```json").rstrip("```").strip()
            extracted_data = json.loads(clean_response)

            if safe_get(extracted_data, 'annual_tourist_arrivals'):
                destination.annual_tourist_arrivals = int(extracted_data['annual_tourist_arrivals'])
                self.logger.info(f"Updated tourist arrivals for {dest_name}: {destination.annual_tourist_arrivals}")
            if safe_get(extracted_data, 'popularity_stage'):
                destination.popularity_stage = extracted_data['popularity_stage']
                self.logger.info(f"Updated popularity stage for {dest_name}: {destination.popularity_stage}")
            if safe_get(extracted_data, 'visa_info_url'):
                destination.visa_info_url = extracted_data['visa_info_url']
                self.logger.info(f"Updated visa info URL for {dest_name}.")

        except (json.JSONDecodeError, TypeError, ValueError) as e:
            self.logger.error(f"Failed to parse LLM response for tourism data of {dest_name}: {e}")

        return destination

class EventCalendarAgent(BaseEnrichmentAgent):
    module_name = "event_calendar_extraction"

    def _enrich(self, destination: Destination):
        from ..core.enhanced_data_models import SpecialEvent
        from datetime import date

        dest_name = destination.names[0]
        self.logger.info(f"Fetching event calendar for {dest_name}.")

        prompt = f"""
        List the top 3-5 major annual events or festivals for "{dest_name}". 
        Provide the response as a JSON list, where each object has keys: "name", "month", "genre", "scale".

        - "month": The primary month the event occurs (e.g., "July").
        - "genre": e.g., "Music", "Culture", "Food", "Film".
        - "scale": e.g., "Local", "National", "International".

        Example:
        [
            {{"name": "Taste of the City", "month": "August", "genre": "Food", "scale": "National"}},
            {{"name": "International Film Festival", "month": "October", "genre": "Film", "scale": "International"}}
        ]
        """
        try:
            llm_response = self.llm.invoke(prompt)
            llm_content = llm_response if isinstance(llm_response, str) else llm_response.content
            clean_response = llm_content.strip().lstrip("```json").rstrip("```").strip()
            events_data = json.loads(clean_response)

            if not destination.temporal_slices:
                self.logger.warning(f"No temporal slices exist for {dest_name} to add events to.")
                return

            current_year = date.today().year
            month_map = {name: num for num, name in enumerate(calendar.month_name) if num}

            for event in events_data:
                month_num = month_map.get(safe_get(event, "month"))
                if not month_num: continue
                
                # Create a dummy date, we only care about the event itself for now
                start_date = date(current_year, month_num, 1)
                end_date = date(current_year, month_num, 28)

                special_event = SpecialEvent(
                    name=safe_get(event, "name"),
                    start_date=start_date,
                    end_date=end_date,
                    genre=safe_get(event, "genre"),
                    scale=safe_get(event, "scale")
                )
                destination.temporal_slices[0].special_events.append(special_event)
            
            self.logger.info(f"Added {len(events_data)} events to the calendar for {dest_name}.")

        except (json.JSONDecodeError, TypeError, ValueError) as e:
            self.logger.error(f"Failed to parse LLM response for event data of {dest_name}: {e}")

        return destination 