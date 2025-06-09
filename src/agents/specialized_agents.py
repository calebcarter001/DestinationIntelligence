"""
Specialized agents for destination intelligence validation
"""

import logging
from typing import Dict, List, Optional, Any
from datetime import datetime
import re
import json
import sys
import numpy as np # For dot product and norm, if calculating cosine similarity manually
from sklearn.metrics.pairwise import cosine_similarity # Ensure cosine_similarity is imported
from sentence_transformers import SentenceTransformer # Ensure SentenceTransformer is imported
from collections import Counter # ADDED Counter import

from .base_agent import BaseAgent, MessageBroker, AgentMessage, MessageType
from ..core.evidence_hierarchy import EvidenceHierarchy, SourceCategory
from ..core.confidence_scoring import ConfidenceScorer, ConfidenceBreakdown, ConfidenceLevel
from ..core.enhanced_data_models import Evidence, Theme, Destination

# Placeholder for where you initialize your sentence model
# In a real app, this might come from a factory or be passed in.
# For this example, we'll assume it gets initialized in __init__.
# from sentence_transformers import SentenceTransformer # Add this to your main imports if not there

# Placeholder for line_profiler, kernprof makes @profile available globally
# For local linting/IDE, you might need: 
# try:
#     # This will only be available when running with kernprof
#     profile = profile 
# except NameError:
#     # If not running with kernprof, define a dummy decorator
#     def profile(func):
#         return func

class ValidationAgent(BaseAgent):
    """
    Agent responsible for validating themes and insights
    
    Applies confidence formula and emits contradictions
    """
    
    def __init__(self, agent_id: str = "validation-001", config: Optional[Dict[str, Any]] = None):
        super().__init__(agent_id, "ValidationAgent", config)
        self.register_handler(MessageType.VALIDATION_REQUEST, self._handle_validation_request)
        
        # Initialize Sentence Model
        try:
            self.sentence_model = SentenceTransformer('all-MiniLM-L6-v2') 
            self.logger.info("SentenceTransformer model 'all-MiniLM-L6-v2' loaded for ValidationAgent.")
        except ImportError:
            self.logger.error("SentenceTransformer library not found. Please install it: pip install sentence-transformers")
            self.sentence_model = None
        except Exception as e:
            self.logger.error(f"Error loading SentenceTransformer model: {e}. Semantic relevance will be impaired.")
            self.sentence_model = None

        # Theme Category Archetypes (destination-agnostic, but with a tourist lens)
        self.theme_category_archetypes = {
            "attractions": [
                "popular tourist attractions and sightseeing spots",
                "must-see landmarks, iconic buildings, and monuments",
                "observation decks, viewpoints, and photo opportunities",
                "guided tours, city walks, and visitor excursions",
                "unique architectural sights and significant structures"
            ],
            "culture": [
                "museums, art galleries, and exhibition centers for visitors",
                "historical sites, heritage locations, and cultural landmarks",
                "performing arts venues, theaters, and live shows for tourists",
                "local festivals, cultural events, and public celebrations",
                "local traditions, customs, and immersive cultural experiences"
            ],
            "dining": [
                "local cuisine, regional specialties, and authentic food experiences",
                "recommended restaurants, cafes, and eateries for tourists",
                "food markets, street food stalls, and culinary tours",
                "fine dining options and notable gastronomic experiences",
                "local breweries, wineries, and unique beverage tasting"
            ],
            "entertainment": [
                "nightlife options, bars, and clubs for visitors",
                "live music venues, concerts, and performance spaces",
                "theaters, stage shows, and evening entertainment",
                "comedy clubs, entertainment districts, and social hubs",
                "cinemas, amusement parks, and leisure activity centers"
            ],
            "nature_outdoors": [ # Renamed for clarity
                "parks, gardens, and public green spaces for recreation",
                "natural landmarks, scenic areas, and beautiful landscapes",
                "outdoor recreational activities, adventure sports for tourists",
                "hiking trails, walking paths, and cycling routes for visitors",
                "wildlife viewing, nature reserves, and ecological tours"
            ],
            "shopping": [
                "main shopping districts, popular retail areas, and unique boutiques",
                "local markets, artisan craft shops, and souvenir stores",
                "shopping malls, department stores, and brand outlets",
                "antique shops, flea markets, and specialty stores",
                "tax-free shopping and visitor discount opportunities"
            ],
            "neighborhoods_districts": [ # Renamed for clarity
                "exploring distinct neighborhoods and local districts",
                "historic quarters, charming old towns, and heritage areas",
                "vibrant local communities and cultural enclaves",
                "trendy districts, bohemian areas, and artistic zones",
                "waterfront areas, promenades, and riverside districts"
            ],
            "transportation_accessibility": [ # Renamed for clarity
                "public transit options for tourists (metro, bus, tram)",
                "city sightseeing buses and tourist transport passes",
                "walkable areas, pedestrian zones, and scenic walking routes",
                "bike rentals and cycling paths for visitors",
                "airport transfers and main transportation hubs accessibility"
            ],
            "services_amenities_tourist": [ # NEW CATEGORY
                "tourist information centers and visitor services",
                "accommodation options (hotels, guesthouses, rentals) for visitors",
                "currency exchange, ATM locations, and banking for travelers",
                "public restrooms and essential traveler amenities in key tourist areas",
                "luggage storage solutions and travel convenience services for tourists"
            ]
        }

        # Activity-based Archetypes (focused on tourist actions)
        self.activity_archetypes = [
            "sightseeing at major landmarks, exploring iconic attractions, and capturing photo opportunities",
            "engaging in cultural immersion, learning local history through museums, and visiting heritage sites",
            "dining at notable local restaurants, trying signature city dishes, and enjoying guided culinary tours",
            "attending live performances, unique city shows, concerts, and experiencing distinctive nightlife",
            "exploring unique urban parks and engaging in city-specific recreational activities",
            "shopping for local souvenirs, artisan crafts, and items unique to the destination",
            "experiencing unique local relaxation spots and engaging in distinct city-based leisure activities",
            "participating in local festivals, significant community events, or public celebrations open to visitors",
            "taking guided city tours, themed excursions, or day trips to significant nearby attractions",
            "people-watching in vibrant public spaces and soaking up the unique city atmosphere"
        ]

        # Traveler Interest Archetypes (more specific interests of visitors)
        self.traveler_interest_archetypes = [
            "family-friendly attractions, theme parks, and activities suitable for children in the city",
            "historical tours, exploring ancient ruins, and learning about the destination's heritage preservation",
            "visiting prominent art galleries, modern art museums, and discovering local street art tours",
            "urban nature experiences, city botanical gardens, and unique local wildlife viewing opportunities",
            "gourmet food tourism, participating in local cooking classes, and guided wine or craft beer tasting tours",
            "experiencing vibrant live music scenes, energetic nightlife districts, and evening entertainment shows",
            "luxury retail therapy, finding designer boutiques, and shopping for unique local artisan crafts",
            "authentic cultural immersion by interacting with local communities and learning traditions",
            "architectural sightseeing tours, learning about city planning, and appreciating unique urban design",
            "attending major sporting events, trying out adventure sports available to visitors, and using city recreational facilities",
            "destination-specific wellness programs, culturally unique spa treatments, and urban relaxation therapies",
            "guided photography tours focused on capturing iconic cityscapes and landmark views"
        ]

        # Initialize embeddings
        if self.sentence_model:
            # Create embeddings for all archetype categories
            self.category_embeddings = {}
            for category, archetypes in self.theme_category_archetypes.items():
                category_embeddings = self.sentence_model.encode(archetypes)
                self.category_embeddings[category] = category_embeddings

            # Create embeddings for activity archetypes
            self.activity_embeddings = self.sentence_model.encode(self.activity_archetypes)

            # Create embeddings for traveler interest archetypes
            self.interest_embeddings = self.sentence_model.encode(self.traveler_interest_archetypes)

            self.logger.info(f"Initialized embeddings for {len(self.theme_category_archetypes)} categories, "
                           f"{len(self.activity_archetypes)} activities, and "
                           f"{len(self.traveler_interest_archetypes)} traveler interests.")
        else:
            self.category_embeddings = {}
            self.activity_embeddings = []
            self.interest_embeddings = []
            self.logger.warning("No sentence model available, embeddings will be empty.")

        # Enhanced stop lists and geographic hierarchy
        self.generic_theme_stop_list = [
            "general", "information", "overview", "details", "introduction",
            "history", "culture", "people", "population", "geography",
            "climate", "weather", "economy", "politics", "government",
            "city", "town", "area", "region", "country", "nation", "world",
            "facts", "data", "statistics", "demographics",
            "united states", "america", "usa", "u.s.a.", "u.s.",  # Added explicit country variations
            "state", "states", "province", "provinces", "territory",  # Added administrative divisions
            "north", "south", "east", "west", "central",  # Added directional terms
            "international", "global", "worldwide"  # Added scope terms
        ]
        
        # Geographic hierarchy validation
        self.country_name_map = {
            "US": ["united states", "america", "usa", "u.s.a.", "u.s."],
            "UK": ["united kingdom", "great britain", "britain"],
            "CA": ["canada"],
            # Add more as needed
        }
        
        # State/Region validation for city analysis
        self.us_states = {
            "illinois": ["chicago"],
            "new york": ["new york city", "buffalo", "albany"],
            "california": ["los angeles", "san francisco", "san diego"],
            # Add more as needed
        }

    # @profile
    def _calculate_semantic_scores_from_embedding(self, theme_embedding: np.ndarray, theme_name: str) -> float:
        """Calculate semantic similarity scores from a pre-computed theme embedding."""
        if not self.sentence_model: # Should not happen if embedding is pre-computed, but good check
            self.logger.warning(f"Sentence model not available during semantic score calculation from embedding for '{theme_name}'.")
            return 0.5

        try:
            # Calculate similarity with category archetypes
            category_scores = []
            if self.category_embeddings:
                for category, embeddings in self.category_embeddings.items():
                    if embeddings is not None and len(embeddings) > 0:
                        similarities = cosine_similarity([theme_embedding], embeddings)[0]
                        category_scores.append(max(similarities) if len(similarities) > 0 else 0.0)
                    else:
                        category_scores.append(0.0)
            
            # Calculate similarity with activity archetypes
            activity_score = 0.0
            if self.activity_embeddings is not None and len(self.activity_embeddings) > 0:
                activity_similarities = cosine_similarity([theme_embedding], self.activity_embeddings)[0]
                activity_score = max(activity_similarities) if len(activity_similarities) > 0 else 0.0
            
            # Calculate similarity with traveler interest archetypes
            interest_score = 0.0
            if self.interest_embeddings is not None and len(self.interest_embeddings) > 0:
                interest_similarities = cosine_similarity([theme_embedding], self.interest_embeddings)[0]
                interest_score = max(interest_similarities) if len(interest_similarities) > 0 else 0.0
            
            # Weighted combination of scores
            max_category_score = max(category_scores) if category_scores else 0.0
            semantic_score = (
                max_category_score * 0.40 +  # Category relevance (restored to 0.4 for stronger tourist focus)
                activity_score * 0.40 +      # Activity relevance (increased to 0.4 for stronger action focus)
                interest_score * 0.20        # Traveler interest relevance (reduced to 0.2 as supplementary)
            )
            
            self.logger.debug(
                f"Theme '{theme_name}' (from embedding) semantic scores: "
                f"category={max_category_score:.3f}, "
                f"activity={activity_score:.3f}, "
                f"interest={interest_score:.3f}, "
                f"combined={semantic_score:.3f}"
            )
            
            return semantic_score
            
        except Exception as e:
            self.logger.error(f"Error calculating semantic scores from embedding for theme '{theme_name}': {e}", exc_info=True)
            return 0.5  # Default score on error

    # @profile
    def _calculate_semantic_relevance(self, theme_data: Dict[str, Any]) -> float:
        """Calculate semantic relevance using multiple archetype categories (single theme processing)."""
        if not self.sentence_model:
            self.logger.warning(f"Sentence model not available for semantic relevance calculation of theme '{theme_data.get('name', '')}'.")
            return 0.5  # Default if no model available

        theme_name = theme_data.get('name', '')
        theme_description = theme_data.get('description', '')
        theme_category = theme_data.get('macro_category', '')
        theme_subcategory = theme_data.get('micro_category', '')
        
        theme_text = f"{theme_name}. {theme_description}. {theme_category} - {theme_subcategory}"
        
        try:
            theme_embedding = self.sentence_model.encode(theme_text)
            return self._calculate_semantic_scores_from_embedding(theme_embedding, theme_name)
            
        except Exception as e:
            self.logger.error(f"Error calculating semantic relevance for theme '{theme_name}': {e}", exc_info=True)
            return 0.5  # Default score on error

    # @profile
    def _calculate_traveler_relevance(self, theme_data: Dict[str, Any], 
                                      destination_name: str, 
                                      destination_country_code: str,
                                      semantic_score: float) -> float:
        """Calculate how relevant a theme is for travelers."""
        theme_name = theme_data.get("name", "").lower()
        theme_description = theme_data.get("description", "").lower()
        
        # Penalize themes that are just geographic names or too general
        if any(term in theme_name for term in self.generic_theme_stop_list):
            return 0.1  # Severely penalize generic geographic themes
        
        # Check if theme matches country name variations
        country_variations = self.country_name_map.get(destination_country_code, [])
        if any(country.lower() in theme_name for country in country_variations):
            return 0.1  # Severely penalize country-level themes
            
        # Penalize state/region level themes
        if destination_country_code == "US":
            for state, cities in self.us_states.items():
                if state.lower() in theme_name and destination_name.lower() not in cities:
                    return 0.2  # Penalize state-level themes
        
        # Calculate base relevance from semantic score
        base_relevance = semantic_score
        
        # Boost for specific tourist indicators in theme name or description
        tourist_indicators = [
            "attraction", "tour", "visit", "experience", "activity",
            "sightseeing", "guide", "tourist", "traveler", "destination"
        ]
        
        indicator_boost = sum(0.1 for indicator in tourist_indicators 
                            if indicator in theme_name or indicator in theme_description)
        
        # Cap the total boost
        total_boost = min(indicator_boost, 0.3)
        
        # Combine base relevance with tourist-specific boosts
        final_relevance = min(base_relevance + total_boost, 1.0)
        
        return final_relevance

    async def execute_task(self, task_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate themes for a destination
        
        Args:
            task_data: Should contain 'destination' and 'themes'
        """
        destination_name = task_data.get("destination_name", "Unknown")
        themes = task_data.get("themes", [])
        destination_country_code = task_data.get("country_code")
        
        self.logger.info(f"Validating {len(themes)} themes for {destination_name} (Country: {destination_country_code})")
        
        validated_themes = []
        contradictions = []
        
        # ---- Batch calculate semantic scores ----
        theme_texts_for_batch_embedding = []
        if self.sentence_model and themes:
            for theme_data_item in themes:
                theme_name_item = theme_data_item.get('name', '')
                theme_description_item = theme_data_item.get('description', '')
                theme_category_item = theme_data_item.get('macro_category', '')
                theme_subcategory_item = theme_data_item.get('micro_category', '')
                theme_texts_for_batch_embedding.append(
                    f"{theme_name_item}. {theme_description_item}. {theme_category_item} - {theme_subcategory_item}"
                )
        
        batch_theme_embeddings = None
        if self.sentence_model and theme_texts_for_batch_embedding:
            try:
                self.logger.info(f"Generating batch embeddings for {len(theme_texts_for_batch_embedding)} themes.")
                batch_theme_embeddings = self.sentence_model.encode(theme_texts_for_batch_embedding, batch_size=32) # Default batch_size is often 32
                self.logger.info(f"Successfully generated batch embeddings.")
            except Exception as e:
                self.logger.error(f"Error during batch theme embedding: {e}", exc_info=True)
                # Fallback: batch_theme_embeddings will remain None, individual processing will occur or default scores used.

        for i, theme_data in enumerate(themes):
            # Handle both Theme objects and dictionaries
            if hasattr(theme_data, 'name'):
                theme_name = theme_data.name
            else:
                theme_name = theme_data.get("name", "Unknown Theme")
            
            # Handle both evidence formats - dictionary key vs original objects
            if hasattr(theme_data, 'evidence'):
                evidence_list = theme_data.evidence
            else:
                evidence_list = theme_data.get("evidence", [])
                if not evidence_list:
                    # Fallback to original_evidence_objects from validation
                    evidence_list = theme_data.get("original_evidence_objects", [])
            
            if not evidence_list:
                self.logger.warning(f"Theme '{theme_name}' received no original evidence objects for validation. Assigning default INSUFFICIENT confidence.")
                confidence_scorer = ConfidenceScorer()
                confidence_breakdown = confidence_scorer.calculate_confidence([]) 
            else:
                confidence_scorer = ConfidenceScorer()
                confidence_breakdown = confidence_scorer.calculate_confidence(evidence_list)
            
            # ADDED: Debugging print and log for ConfidenceScorer's output as seen by ValidationAgent
            diag_theme_name = theme_name
            diag_overall_conf = confidence_breakdown.overall_confidence
            diag_conf_level = confidence_breakdown.confidence_level.value
            
            # Proper type checking for confidence_breakdown
            if isinstance(confidence_breakdown, dict):
                diag_breakdown_dict = confidence_breakdown
            elif hasattr(confidence_breakdown, 'to_dict') and callable(getattr(confidence_breakdown, 'to_dict')):
                diag_breakdown_dict = confidence_breakdown.to_dict()
            else:
                diag_breakdown_dict = {"error": "unexpected_confidence_breakdown_type", "type": str(type(confidence_breakdown))}

            print(f"DEBUG_VA_CONF_OUT: Theme='{diag_theme_name}', OverallConf={diag_overall_conf:.4f}, Level='{diag_conf_level}'", file=sys.stderr)
            self.logger.info(f"VALIDATION_AGENT_CS_RESULT: Theme='{diag_theme_name}', OverallConf={diag_overall_conf:.4f}, Level='{diag_conf_level}', Breakdown={json.dumps(diag_breakdown_dict)}")

            # Check for contradictions (this logic can remain, but might be more effective 
            # if it also had access to richer evidence details if it needs to re-evaluate)
            # For now, it uses evidence_texts which might need to be derived if not directly available
            # from original_evidence_objects in the expected format for contradiction checks.
            # Let's assume for now contradiction logic might need its own evidence representation if it can't use original_evidence_objects directly.
            # We will rely on the ContradictionDetectionAgent to handle this with the data it receives.
            evidence_texts_for_contradiction = [ev.text_snippet for ev in evidence_list] # Reconstruct if needed

            if confidence_breakdown.consistency_score < 0.5:
                contradictions.append({
                    "theme": theme_name,
                    "reason": "Low consistency score indicates conflicting evidence",
                    "consistency_score": confidence_breakdown.consistency_score,
                    "evidence_snippets": evidence_texts_for_contradiction[:3]  # First 3 as examples
                })
            
            # Add confidence to theme - proper type checking
            if isinstance(confidence_breakdown, dict):
                theme_data["confidence_breakdown"] = confidence_breakdown
            elif hasattr(confidence_breakdown, 'to_dict') and callable(getattr(confidence_breakdown, 'to_dict')):
                theme_data["confidence_breakdown"] = confidence_breakdown.to_dict()
            else:
                theme_data["confidence_breakdown"] = {"error": "unexpected_confidence_breakdown_type", "type": str(type(confidence_breakdown))}
            
            theme_data["confidence_level"] = confidence_breakdown.confidence_level.value
            # The threshold for is_validated can remain, but its meaning is now based on richer scoring.
            theme_data["is_validated"] = confidence_breakdown.overall_confidence >= 0.2
            
            # --- Calculate Semantic Score (using batch or individual) ---
            current_semantic_score = 0.5 # Default
            if batch_theme_embeddings is not None and i < len(batch_theme_embeddings):
                current_theme_embedding = batch_theme_embeddings[i]
                current_semantic_score = self._calculate_semantic_scores_from_embedding(current_theme_embedding, diag_theme_name)
            elif self.sentence_model: # Fallback to individual calculation if batch failed or not applicable
                current_semantic_score = self._calculate_semantic_relevance(theme_data)
            else:
                self.logger.warning(f"No sentence model or batch embeddings for theme '{diag_theme_name}', using default semantic score.")

            # --- Calculate Traveler Relevance and Adjusted Confidence (using the calculated semantic_score) ---
            relevance_factor = self._calculate_traveler_relevance(
                theme_data, 
                destination_name, 
                destination_country_code if destination_country_code else "", # Pass empty string if None
                current_semantic_score # Pass the calculated semantic score
            )
            theme_data['traveler_relevance_factor'] = relevance_factor
            
            original_overall_confidence = confidence_breakdown.overall_confidence
            adjusted_overall_confidence = original_overall_confidence * relevance_factor
            theme_data['adjusted_overall_confidence'] = adjusted_overall_confidence
            
            self.logger.info(
                f"Theme: '{diag_theme_name}' | Orig. Conf: {original_overall_confidence:.4f} | "
                f"Relevance Factor: {relevance_factor:.2f} | Adj. Conf: {adjusted_overall_confidence:.4f}"
            )
            
            # --- CRITICAL DEBUG & FIX: Preserve the original fit_score --- 
            original_fit_score_from_input = theme_data.get("fit_score") # Get what was passed in
            
            if original_fit_score_from_input is not None:
                theme_data["fit_score"] = original_fit_score_from_input # Ensure it is preserved
            else:
                theme_data["fit_score"] = 0.0 # Fallback, though this indicates a problem upstream
            
            validated_themes.append(theme_data)
            
        # Emit contradictions if found
        if contradictions:
            contradiction_msg = self.create_message(
                MessageType.CONTRADICTION,
                {
                    "destination_name": destination_name,
                    "contradictions": contradictions,
                    "timestamp": datetime.now().isoformat()
                }
            )
            # In real implementation, this would be published to the broker
            
        return {
            "validated_themes": validated_themes,
            "total_themes": len(themes),
            "validated_count": sum(1 for t in validated_themes if t["is_validated"]),
            "contradictions_found": len(contradictions),
            "validation_timestamp": datetime.now().isoformat()
        }
    
    async def _handle_validation_request(self, message: AgentMessage) -> Optional[AgentMessage]:
        """Handle validation request messages"""
        result = await self.execute_task(message.payload)
        
        return self.create_message(
            MessageType.VALIDATION_RESPONSE,
            result,
            recipient_id=message.sender_id,
            correlation_id=message.id
        )

class CulturalPerspectiveAgent(BaseAgent):
    """
    Agent that prioritizes local-language and locally owned sources
    
    Enhances evidence with cultural context
    """
    
    def __init__(self, agent_id: str = "cultural-001", config: Optional[Dict[str, Any]] = None):
        super().__init__(agent_id, "CulturalPerspectiveAgent", config)
        
        # Language indicators for different regions
        self.local_language_patterns = {
            "FR": ["french", "français", "fr.", ".fr"],
            "ES": ["spanish", "español", "castellano", ".es"],
            "IT": ["italian", "italiano", ".it"],
            "DE": ["german", "deutsch", ".de"],
            "JP": ["japanese", "日本語", ".jp", "nihongo"],
            "CN": ["chinese", "中文", "mandarin", ".cn"],
            "TH": ["thai", "ไทย", ".th"],
            "GR": ["greek", "ελληνικά", ".gr"],
            "PT": ["portuguese", "português", ".pt", ".br"],
            "RU": ["russian", "русский", ".ru"]
        }
        
    async def execute_task(self, task_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze sources for cultural perspective
        
        Args:
            task_data: Should contain 'sources' and 'destination_country_code'
        """
        sources = task_data.get("sources", [])
        country_code = task_data.get("country_code", "").upper()
        destination_name = task_data.get("destination_name", "Unknown")
        
        self.logger.info(f"Analyzing cultural perspective for {len(sources)} sources from {destination_name}")
        
        enhanced_sources = []
        local_source_count = 0
        language_distribution = Counter()
        
        for source in sources:
            url = source.get("url", "")
            content = source.get("content", "")
            
            # Determine if source is local
            is_local = EvidenceHierarchy.is_local_source(url, country_code)
            
            # Detect language
            detected_language = self._detect_language(url, content, country_code)
            language_distribution[detected_language] += 1
            
            # Check for local ownership indicators
            local_ownership = self._check_local_ownership(url, content)
            
            # Calculate cultural relevance score
            cultural_score = self._calculate_cultural_score(
                is_local, detected_language, local_ownership, country_code
            )
            
            if is_local:
                local_source_count += 1
                
            enhanced_source = {
                **source,
                "cultural_context": {
                    "is_local_source": is_local,
                    "detected_language": detected_language,
                    "local_ownership": local_ownership,
                    "cultural_relevance_score": cultural_score,
                    "analysis_timestamp": datetime.now().isoformat()
                }
            }
            
            enhanced_sources.append(enhanced_source)
            
        # Calculate overall cultural diversity
        cultural_diversity = len(language_distribution) / max(len(self.local_language_patterns), 1)
        local_source_ratio = local_source_count / len(sources) if sources else 0
        
        return {
            "enhanced_sources": enhanced_sources,
            "cultural_metrics": {
                "local_source_count": local_source_count,
                "local_source_ratio": local_source_ratio,
                "language_distribution": dict(language_distribution),
                "cultural_diversity_score": cultural_diversity,
                "optimal_mix_score": self._calculate_optimal_mix_score(local_source_ratio)
            },
            "recommendations": self._generate_recommendations(local_source_ratio, language_distribution)
        }
    
    def _detect_language(self, url: str, content: str, country_code: str) -> str:
        """Detect language of content"""
        combined_text = f"{url} {content}".lower()
        
        # Check for country-specific patterns
        if country_code in self.local_language_patterns:
            for pattern in self.local_language_patterns[country_code]:
                if pattern in combined_text:
                    return f"local_{country_code.lower()}"
                    
        # Default language detection based on common patterns
        if any(eng in combined_text for eng in ["english", "en.", "www.", ".com"]):
            return "english"
            
        return "unknown"
    
    def _check_local_ownership(self, url: str, content: str) -> bool:
        """Check indicators of local ownership"""
        local_indicators = [
            "family-owned", "locally owned", "local business",
            "established in", "founded by", "native",
            "generations", "traditional", "authentic"
        ]
        
        combined_text = f"{url} {content}".lower()
        return any(indicator in combined_text for indicator in local_indicators)
    
    def _calculate_cultural_score(
        self, is_local: bool, language: str, local_ownership: bool, country_code: str
    ) -> float:
        """Calculate cultural relevance score (0-1)"""
        score = 0.0
        
        if is_local:
            score += 0.4
            
        if language.startswith("local_"):
            score += 0.3
        elif language == "english":
            score += 0.1  # Some value for international accessibility
            
        if local_ownership:
            score += 0.3
            
        return min(score, 1.0)
    
    def _calculate_optimal_mix_score(self, local_ratio: float) -> float:
        """Calculate how close to optimal mix (60% local, 40% international)"""
        optimal_ratio = 0.6
        deviation = abs(local_ratio - optimal_ratio)
        
        # Score decreases as we deviate from optimal
        return max(0, 1 - (deviation / optimal_ratio))
    
    def _generate_recommendations(
        self, local_ratio: float, language_dist: Counter
    ) -> List[str]:
        """Generate recommendations for improving cultural perspective"""
        recommendations = []
        
        if local_ratio < 0.4:
            recommendations.append("Seek more local sources to improve authenticity")
        elif local_ratio > 0.8:
            recommendations.append("Include more international perspectives for balance")
            
        if len(language_dist) < 2:
            recommendations.append("Diversify language sources for richer perspective")
            
        return recommendations

class ContradictionDetectionAgent(BaseAgent):
    """
    Agent that detects and resolves contradictions in evidence
    
    Uses authority and recency to resolve conflicts
    """
    
    def __init__(self, agent_id: str = "contradiction-001", config: Optional[Dict[str, Any]] = None):
        super().__init__(agent_id, "ContradictionDetectionAgent", config)
        
        # Contradiction patterns
        self.contradiction_indicators = [
            ("safe", "dangerous"), ("clean", "dirty"), ("expensive", "cheap"),
            ("crowded", "empty"), ("modern", "traditional"), ("quiet", "noisy"),
            ("friendly", "unfriendly"), ("authentic", "touristy"),
            ("well-maintained", "run-down"), ("easy", "difficult")
        ]
        
    async def execute_task(self, task_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Detect contradictions in theme evidence
        
        Args:
            task_data: Should contain 'themes' with evidence
        """
        themes = task_data.get("themes", [])
        destination_name = task_data.get("destination_name", "Unknown")
        
        self.logger.info(f"Detecting contradictions for {len(themes)} themes in {destination_name}")
        
        all_contradictions = []
        resolved_themes = []
        
        for theme in themes:
            # Handle both Theme objects and dictionaries
            if hasattr(theme, 'name'):
                theme_name = theme.name
            else:
                theme_name = theme.get("name", "Unknown Theme")
            
            # Handle both evidence formats - dictionary key vs original objects
            if hasattr(theme, 'evidence'):
                evidence_list = theme.evidence
            else:
                evidence_list = theme.get("evidence", [])
                if not evidence_list:
                    # Fallback to original_evidence_objects from validation
                    evidence_list = theme.get("original_evidence_objects", [])
            
            if len(evidence_list) < 2:
                resolved_themes.append(theme)
                continue
                
            # Detect contradictions
            contradictions = self._detect_contradictions(evidence_list)
            
            if contradictions:
                # Resolve using authority and recency
                resolution = self._resolve_contradictions(contradictions, evidence_list)
                
                all_contradictions.append({
                    "theme": theme_name,
                    "contradictions": contradictions,
                    "resolution": resolution
                })
                
                # Update theme with resolution - handle both Theme objects and dictionaries
                if hasattr(theme, '__dict__'):  # Theme object
                    theme.metadata = theme.metadata or {}
                    theme.metadata["contradiction_resolved"] = True
                    theme.metadata["resolution_method"] = resolution["method"]
                    theme.metadata["winning_position"] = resolution["winning_position"]
                else:  # Dictionary
                    theme["contradiction_resolved"] = True
                    theme["resolution_method"] = resolution["method"]
                    theme["winning_position"] = resolution["winning_position"]
                
            resolved_themes.append(theme)
            
        return {
            "resolved_themes": resolved_themes,
            "contradictions_found": len(all_contradictions),
            "contradiction_details": all_contradictions,
            "resolution_timestamp": datetime.now().isoformat()
        }
    
    def _detect_contradictions(self, evidence_list: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Detect contradictions in evidence"""
        contradictions = []
        
        # Check each pair of evidence
        for i, evidence1 in enumerate(evidence_list):
            for j, evidence2 in enumerate(evidence_list[i+1:], i+1):
                # Handle both Evidence objects and dictionaries
                if hasattr(evidence1, 'text_snippet'):  # Evidence object
                    text1 = evidence1.text_snippet.lower() if evidence1.text_snippet else ""
                    source1 = evidence1.source_url if hasattr(evidence1, 'source_url') else ""
                    snippet1 = evidence1.text_snippet if evidence1.text_snippet else ""
                else:  # Dictionary
                    text1 = evidence1.get("text_snippet", "").lower()
                    source1 = evidence1.get("source_url", "")
                    snippet1 = evidence1.get("text_snippet", "")
                
                if hasattr(evidence2, 'text_snippet'):  # Evidence object
                    text2 = evidence2.text_snippet.lower() if evidence2.text_snippet else ""
                    source2 = evidence2.source_url if hasattr(evidence2, 'source_url') else ""
                    snippet2 = evidence2.text_snippet if evidence2.text_snippet else ""
                else:  # Dictionary
                    text2 = evidence2.get("text_snippet", "").lower()
                    source2 = evidence2.get("source_url", "")
                    snippet2 = evidence2.get("text_snippet", "")
                
                # Check for contradictory terms
                for positive, negative in self.contradiction_indicators:
                    if positive in text1 and negative in text2:
                        contradictions.append({
                            "evidence_1": {
                                "source": source1,
                                "text": snippet1,
                                "position": positive
                            },
                            "evidence_2": {
                                "source": source2,
                                "text": snippet2,
                                "position": negative
                            },
                            "type": f"{positive}_vs_{negative}"
                        })
                    elif negative in text1 and positive in text2:
                        contradictions.append({
                            "evidence_1": {
                                "source": source1,
                                "text": snippet1,
                                "position": negative
                            },
                            "evidence_2": {
                                "source": source2,
                                "text": snippet2,
                                "position": positive
                            },
                            "type": f"{negative}_vs_{positive}"
                        })
                        
        return contradictions
    
    def _resolve_contradictions(
        self, contradictions: List[Dict[str, Any]], evidence_list: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Resolve contradictions using authority and recency"""
        if not contradictions:
            return {"method": "none", "winning_position": "no_contradiction"}
            
        # Score each position based on supporting evidence
        position_scores = {}
        
        for contradiction in contradictions:
            pos1 = contradiction["evidence_1"]["position"]
            pos2 = contradiction["evidence_2"]["position"]
            
            # Find all evidence supporting each position
            for evidence in evidence_list:
                # Handle both Evidence objects and dictionaries
                if hasattr(evidence, 'text_snippet'):  # Evidence object
                    text = evidence.text_snippet.lower() if evidence.text_snippet else ""
                    source_url = evidence.source_url if hasattr(evidence, 'source_url') else ""
                    published_date = evidence.published_date if hasattr(evidence, 'published_date') else None
                else:  # Dictionary
                    text = evidence.get("text_snippet", "").lower()
                    source_url = evidence.get("source_url", "")
                    published_date = evidence.get("published_date")
                
                # Get authority weight
                authority, _ = EvidenceHierarchy.get_source_authority(source_url, published_date)
                
                # Calculate recency bonus
                recency_score = 1.0
                if published_date:
                    # Handle both datetime objects and string timestamps
                    if isinstance(published_date, str):
                        try:
                            timestamp = datetime.fromisoformat(published_date.replace('Z', '+00:00'))
                        except (ValueError, AttributeError):
                            timestamp = datetime.now()  # Default to current date if parsing fails
                    else:
                        timestamp = published_date
                    
                    age_days = (datetime.now() - timestamp).days
                    recency_score = max(0.5, 1.0 - (age_days / 365))
                
                combined_score = authority * recency_score
                
                if pos1 in text:
                    position_scores[pos1] = position_scores.get(pos1, 0) + combined_score
                if pos2 in text:
                    position_scores[pos2] = position_scores.get(pos2, 0) + combined_score
                    
        # Determine winning position
        if position_scores:
            winning_position = max(position_scores.items(), key=lambda x: x[1])
            return {
                "method": "authority_and_recency",
                "winning_position": winning_position[0],
                "confidence": winning_position[1] / sum(position_scores.values()),
                "position_scores": position_scores
            }
        else:
            return {
                "method": "unresolved",
                "winning_position": "unclear",
                "reason": "Could not determine authoritative position"
            } 