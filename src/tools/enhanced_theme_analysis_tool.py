from langchain.tools import Tool
from pydantic import BaseModel, Field
from typing import List, Dict, Optional, Any
from datetime import datetime
import logging
import hashlib

from ..core.evidence_hierarchy import EvidenceHierarchy, SourceCategory
from ..core.confidence_scoring import ConfidenceScorer
from ..core.enhanced_data_models import Evidence, Theme, Destination, TemporalSlice
from ..agents.specialized_agents import ValidationAgent, CulturalPerspectiveAgent, ContradictionDetectionAgent
from ..schemas import DestinationInsight

logger = logging.getLogger(__name__)

class EnhancedThemeAnalysisInput(BaseModel):
    """Input for enhanced theme analysis"""
    destination_name: str = Field(description="Name of the destination being analyzed")
    country_code: str = Field(description="ISO 2-letter country code of the destination")
    text_content_list: List[Dict[str, Any]] = Field(description="List of content with URLs and text")
    analyze_temporal: bool = Field(default=True, description="Whether to analyze temporal aspects")
    min_confidence: float = Field(default=0.5, description="Minimum confidence threshold")

class EnhancedThemeAnalysisTool:
    """
    Advanced theme analysis with evidence hierarchy, confidence scoring, and multi-agent validation
    """
    
    def __init__(self):
        self.name = "enhanced_theme_analysis"
        self.description = (
            "Perform advanced theme analysis with evidence classification, "
            "confidence scoring, cultural perspective, and contradiction detection"
        )
        self.logger = self._setup_logger()
        
        # Initialize specialized agents
        self.validation_agent = ValidationAgent()
        self.cultural_agent = CulturalPerspectiveAgent()
        self.contradiction_agent = ContradictionDetectionAgent()
        
        # Enhanced taxonomy with generic themes + destination-specific ones
        self.theme_taxonomy = {
            "Nature & Outdoor": [
                "Hiking & Trails", "Mountains & Peaks", "Parks & Recreation", "Nature Viewing",
                "Outdoor Adventures", "Rivers & Lakes", "Scenic Views", "Wildlife",
                "Camping & RV", "Rock Climbing", "Kayaking & Rafting", "Fishing",
                "Biking & Cycling", "Cross-Country Skiing", "Snowshoeing"
            ],
            "Cultural & Arts": [
                "Museums & Galleries", "Architecture", "Historic Sites", "Local Arts",
                "Cultural Heritage", "Music & Performances", "Festivals & Events",
                "Art Studios", "Public Art", "Cultural Centers"
            ],
            "Food & Dining": [
                "Restaurants", "Local Cuisine", "Breweries & Distilleries", "Cafes & Coffee",
                "Food Markets", "Farm-to-Table", "Fine Dining", "Food Festivals",
                "Local Specialties", "Wine Tasting", "Food Tours"
            ],
            "Entertainment & Nightlife": [
                "Nightlife", "Bars & Pubs", "Live Music Venues", "Dance Clubs",
                "Comedy Shows", "Theater & Performances", "Casinos", "Rooftop Bars",
                "Entertainment Districts", "Social Venues"
            ],
            "Adventure & Sports": [
                "Adventure Sports", "Water Sports", "Winter Sports", "Extreme Sports",
                "Golf", "Tennis", "Fitness & Wellness", "Sports Events",
                "Adventure Tours", "Outdoor Recreation", "Skiing & Snowboarding"
            ],
            "Shopping & Local Craft": [
                "Shopping", "Local Markets", "Boutiques", "Craft Shops", "Artisan Goods",
                "Farmers Markets", "Antiques", "Local Products", "Specialty Stores",
                "Shopping Districts", "Handmade Crafts"
            ],
            "Family & Education": [
                "Family Activities", "Kid-Friendly", "Educational", "Science Centers",
                "Zoos & Aquariums", "Children's Museums", "Playgrounds",
                "Family Entertainment", "Learning Experiences"
            ],
            "Accommodation & Stay": [
                "Hotels", "Resorts", "Unique Stays", "Luxury Accommodation",
                "Budget Options", "Vacation Rentals", "Bed & Breakfasts",
                "Camping & Glamping", "Historic Inns"
            ],
            "Health & Wellness": [
                "Spas & Wellness", "Hot Springs", "Yoga & Meditation", "Fitness",
                "Health Retreats", "Therapeutic", "Natural Healing", "Relaxation"
            ],
            "Transportation & Access": [
                "Transportation", "Accessibility", "Getting Around", "Parking",
                "Public Transit", "Walkability", "Bike-Friendly"
            ]
        }
        
    async def analyze_themes(self, input_data: EnhancedThemeAnalysisInput) -> Dict[str, Any]:
        """
        Perform comprehensive theme analysis with enhanced features
        """
        self.logger.info(f"Starting enhanced theme analysis for {input_data.destination_name}")
        
        # Step 1: Extract and classify evidence
        all_evidence = await self._extract_evidence(
            input_data.text_content_list,
            input_data.country_code
        )
        
        # Step 2: Discover themes from evidence
        discovered_themes = await self._discover_themes(
            all_evidence,
            input_data.destination_name
        )
        
        # Step 3: Cultural perspective analysis
        cultural_result = await self.cultural_agent.execute_task({
            "sources": [
                {
                    "url": ev.source_url,
                    "content": ev.text_snippet
                }
                for ev in all_evidence
            ],
            "country_code": input_data.country_code,
            "destination_name": input_data.destination_name
        })
        
        # Step 4: Validate themes with confidence scoring
        validation_result = await self.validation_agent.execute_task({
            "destination_name": input_data.destination_name,
            "themes": [
                {
                    "name": theme.name,
                    "evidence_sources": [ev.source_url for ev in theme.evidence],
                    "evidence_texts": [ev.text_snippet for ev in theme.evidence],
                    "sentiment_scores": [ev.sentiment for ev in theme.evidence if ev.sentiment]
                }
                for theme in discovered_themes
            ],
            "country_code": input_data.country_code
        })
        
        # Step 5: Detect and resolve contradictions
        contradiction_result = await self.contradiction_agent.execute_task({
            "themes": validation_result["validated_themes"],
            "destination_name": input_data.destination_name
        })
        
        # Step 6: Build enhanced themes with full metadata
        enhanced_themes = self._build_enhanced_themes(
            contradiction_result["resolved_themes"],
            all_evidence,
            cultural_result
        )
        
        # Step 7: Create temporal slices if requested
        temporal_slices = []
        if input_data.analyze_temporal:
            temporal_slices = self._analyze_temporal_aspects(enhanced_themes, all_evidence)
        
        # Step 8: Calculate destination dimensions
        dimensions = self._calculate_dimensions(enhanced_themes, all_evidence)
        
        return {
            "destination_name": input_data.destination_name,
            "country_code": input_data.country_code,
            "themes": enhanced_themes,
            "temporal_slices": temporal_slices,
            "dimensions": dimensions,
            "evidence_summary": {
                "total_evidence": len(all_evidence),
                "source_distribution": self._get_source_distribution(all_evidence),
                "cultural_metrics": cultural_result["cultural_metrics"]
            },
            "quality_metrics": {
                "themes_discovered": len(discovered_themes),
                "themes_validated": validation_result["validated_count"],
                "contradictions_found": contradiction_result["contradictions_found"],
                "average_confidence": self._calculate_average_confidence(enhanced_themes)
            },
            "analysis_timestamp": datetime.now().isoformat()
        }
    
    async def _extract_evidence(
        self, content_list: List[Dict[str, Any]], country_code: str
    ) -> List[Evidence]:
        """Extract and classify evidence from content"""
        all_evidence = []
        
        self.logger.info(f"Starting evidence extraction from {len(content_list)} content items")
        
        for idx, content_item in enumerate(content_list):
            url = content_item.get("url", "")
            text = content_item.get("content", "")
            title = content_item.get("title", "")
            
            self.logger.info(f"Processing content item {idx}: url={url[:50]}..., title={title}, content_length={len(text) if text else 0}")
            
            # Skip empty content
            if not text or len(text.strip()) < 50:
                self.logger.info(f"Skipping content item {idx}: insufficient text (length={len(text) if text else 0})")
                continue
            
            # Split into evidence chunks (simplified - in production would use NLP)
            chunks = self._split_into_chunks(text, chunk_size=500)
            self.logger.info(f"Split content item {idx} into {len(chunks)} chunks")
            
            for chunk_idx, chunk_text in enumerate(chunks):
                # Classify source
                source_category = EvidenceHierarchy.classify_source(url)
                authority_weight, evidence_type = EvidenceHierarchy.get_source_authority(url)
                
                # Create evidence object
                evidence = Evidence(
                    id="",  # Will be auto-generated
                    source_url=url,
                    source_category=source_category,
                    evidence_type=evidence_type,
                    authority_weight=authority_weight,
                    text_snippet=chunk_text,
                    timestamp=datetime.now(),
                    confidence=authority_weight,  # Initial confidence = authority
                    cultural_context={
                        "source_title": title,
                        "chunk_index": chunk_idx
                    },
                    agent_id="enhanced_theme_analysis"
                )
                
                all_evidence.append(evidence)
                
        self.logger.info(f"Extracted {len(all_evidence)} evidence pieces from {len(content_list)} sources")
        return all_evidence
    
    async def _discover_themes(
        self, evidence_list: List[Evidence], destination_name: str
    ) -> List[Theme]:
        """Discover themes from evidence using taxonomy"""
        discovered_themes = []
        theme_evidence_map = {}
        
        # Simple keyword-based discovery (in production would use NLP/LLM)
        for evidence in evidence_list:
            text_lower = evidence.text_snippet.lower()
            
            for macro_category, micro_categories in self.theme_taxonomy.items():
                for micro_category in micro_categories:
                    # Check if micro category keywords appear in text
                    if self._check_theme_match(micro_category, text_lower):
                        theme_key = f"{macro_category}|{micro_category}"
                        
                        if theme_key not in theme_evidence_map:
                            theme_evidence_map[theme_key] = []
                        theme_evidence_map[theme_key].append(evidence)
        
        # Create theme objects - REDUCED REQUIREMENT to 1 evidence piece
        for theme_key, evidence_list in theme_evidence_map.items():
            if len(evidence_list) < 1:  # Require at least 1 piece of evidence (was 2)
                continue
                
            macro, micro = theme_key.split("|")
            
            theme = Theme(
                theme_id=hashlib.md5(theme_key.encode()).hexdigest()[:12],
                macro_category=macro,
                micro_category=micro,
                name=micro,
                description=f"{micro} experiences and attractions in {destination_name}",
                fit_score=min(1.0, len(evidence_list) / 5),  # Adjusted fit score calculation
                evidence=evidence_list,
                tags=self._generate_tags(micro),
                created_date=datetime.now()
            )
            
            discovered_themes.append(theme)
            
        return discovered_themes
    
    def _check_theme_match(self, theme_name: str, text: str) -> bool:
        """Check if theme keywords match in text"""
        # Simple keyword matching - in production would use NLP
        theme_keywords = theme_name.lower().replace("&", "").split()
        
        # Enhanced keyword expansions for better matching
        keyword_expansions = {
            # Nature & Outdoor themes
            "hiking": ["hike", "trail", "trek", "walk", "path", "outdoor", "nature", "adventure", "mountain", "cascade", "forest", "wilderness"],
            "trails": ["trail", "hike", "hiking", "path", "trek", "walk", "nature", "outdoor", "forest", "mountain"],
            "mountains": ["mountain", "peak", "summit", "cascade", "range", "alpine", "elevation", "climb", "vista", "ridge"],
            "peaks": ["peak", "summit", "mountain", "elevation", "climb", "vista", "high", "point"],
            "parks": ["park", "garden", "recreation", "public", "green", "space", "nature", "outdoor"],
            "nature": ["nature", "natural", "park", "forest", "wildlife", "scenic", "landscape", "outdoor", "wilderness"],
            "outdoor": ["outdoor", "nature", "adventure", "recreation", "sports", "activities", "hiking", "camping"],
            "rivers": ["river", "water", "deschutes", "cascade", "stream", "flowing", "rapids", "waterway"],
            "lakes": ["lake", "river", "water", "deschutes", "cascade", "stream", "pond", "reservoir"],
            "scenic": ["scenic", "view", "vista", "landscape", "beautiful", "panoramic", "overlook"],
            "wildlife": ["wildlife", "animals", "birds", "nature", "deer", "elk", "bear", "bird watching"],
            "camping": ["camp", "camping", "campground", "outdoor", "tent", "rv", "recreational vehicle"],
            "climbing": ["climb", "climbing", "rock", "bouldering", "mountaineering", "ascent"],
            "kayaking": ["kayak", "kayaking", "paddle", "water", "river", "lake", "rafting"],
            "rafting": ["raft", "rafting", "whitewater", "river", "rapids", "paddle"],
            "fishing": ["fish", "fishing", "angling", "catch", "trout", "salmon", "stream", "river"],
            "biking": ["bike", "biking", "cycling", "trail", "mountain", "road", "bicycle"],
            "cycling": ["cycle", "cycling", "bike", "bicycle", "trail", "road", "mountain"],
            "skiing": ["ski", "skiing", "snow", "winter", "mountain", "resort", "downhill", "cross-country"],
            "snowshoeing": ["snowshoe", "snowshoeing", "winter", "snow", "trail", "hiking"],
            
            # Cultural & Arts themes
            "museums": ["museum", "gallery", "exhibit", "collection", "art", "history", "cultural", "center"],
            "galleries": ["gallery", "art", "museum", "exhibit", "collection", "studio", "artwork"],
            "architecture": ["architecture", "building", "historic", "design", "structure", "construction"],
            "historic": ["historic", "history", "heritage", "old", "traditional", "antique", "vintage"],
            "arts": ["art", "arts", "gallery", "creative", "studio", "craft", "design", "artistic"],
            "heritage": ["heritage", "history", "historic", "cultural", "tradition", "legacy"],
            "music": ["music", "concert", "performance", "band", "live", "venue", "entertainment"],
            "festivals": ["festival", "event", "celebration", "music", "arts", "community", "annual"],
            "cultural": ["cultural", "culture", "heritage", "history", "art", "local", "traditional"],
            
            # Food & Dining themes
            "restaurants": ["restaurant", "dining", "eat", "meal", "food", "cuisine", "eatery"],
            "dining": ["dining", "restaurant", "food", "cuisine", "eat", "meal", "culinary", "taste"],
            "cuisine": ["cuisine", "food", "dining", "restaurant", "culinary", "local", "specialty"],
            "breweries": ["brewery", "beer", "brew", "craft", "ale", "hops", "brewing"],
            "distilleries": ["distillery", "spirits", "whiskey", "vodka", "gin", "craft", "alcohol"],
            "cafes": ["cafe", "coffee", "espresso", "latte", "cappuccino", "roastery"],
            "coffee": ["coffee", "cafe", "espresso", "roast", "beans", "brew", "cappuccino"],
            "markets": ["market", "farmers", "local", "produce", "vendors", "food", "fresh"],
            "fine": ["fine", "upscale", "elegant", "gourmet", "sophisticated", "high-end"],
            "local": ["local", "regional", "native", "indigenous", "community", "authentic"],
            "specialties": ["specialty", "special", "unique", "signature", "famous", "notable"],
            "wine": ["wine", "vineyard", "tasting", "cellar", "vintage", "grape"],
            
            # Entertainment & Nightlife themes
            "nightlife": ["nightlife", "bar", "club", "night", "party", "entertainment", "live", "music", "venue"],
            "bars": ["bar", "pub", "tavern", "lounge", "cocktail", "drink", "nightlife"],
            "pubs": ["pub", "bar", "tavern", "beer", "drink", "local", "gathering"],
            "venues": ["venue", "location", "place", "establishment", "facility"],
            "clubs": ["club", "nightclub", "dance", "party", "music", "nightlife"],
            "comedy": ["comedy", "comedian", "humor", "funny", "laughs", "entertainment"],
            "theater": ["theater", "theatre", "performance", "play", "show", "drama"],
            "entertainment": ["entertainment", "show", "performance", "venue", "event"],
            
            # Adventure & Sports themes
            "adventure": ["adventure", "sport", "recreation", "outdoor", "activity", "experience", "thrill"],
            "sports": ["sport", "sports", "recreation", "activity", "fitness", "exercise", "game"],
            "water": ["water", "aquatic", "swimming", "boating", "kayaking", "rafting"],
            "winter": ["winter", "snow", "ski", "skiing", "snowboard", "cold", "season"],
            "extreme": ["extreme", "adventure", "adrenaline", "thrill", "exciting", "challenging"],
            "golf": ["golf", "course", "green", "fairway", "club", "tournament"],
            "tennis": ["tennis", "court", "racquet", "match", "game"],
            "fitness": ["fitness", "gym", "exercise", "workout", "health", "training"],
            "wellness": ["wellness", "health", "spa", "relaxation", "therapeutic", "healing"],
            
            # Shopping & Local Craft themes
            "shopping": ["shop", "shopping", "store", "retail", "purchase", "buy"],
            "boutiques": ["boutique", "shop", "store", "fashion", "clothing", "unique"],
            "craft": ["craft", "handmade", "artisan", "handcrafted", "local", "art"],
            "artisan": ["artisan", "craft", "handmade", "artist", "maker", "creator"],
            "antiques": ["antique", "vintage", "old", "collectible", "historic", "rare"],
            "handmade": ["handmade", "craft", "artisan", "local", "unique", "custom"],
            
            # Family & Education themes
            "family": ["family", "kids", "children", "child", "friendly", "fun", "activity"],
            "kids": ["kids", "children", "child", "family", "young", "playground"],
            "children": ["children", "kids", "child", "family", "young", "educational"],
            "educational": ["educational", "learning", "education", "school", "teaching"],
            "science": ["science", "scientific", "research", "discovery", "technology"],
            "zoos": ["zoo", "animals", "wildlife", "conservation", "nature"],
            "aquariums": ["aquarium", "fish", "marine", "underwater", "ocean"],
            
            # Accommodation & Stay themes
            "hotels": ["hotel", "accommodation", "stay", "resort", "lodge", "inn", "motel"],
            "resorts": ["resort", "hotel", "luxury", "vacation", "accommodation"],
            "unique": ["unique", "special", "unusual", "distinctive", "one-of-a-kind"],
            "luxury": ["luxury", "upscale", "premium", "high-end", "exclusive", "elegant"],
            "budget": ["budget", "affordable", "cheap", "inexpensive", "value"],
            "vacation": ["vacation", "holiday", "rental", "stay", "getaway"],
            "bed": ["bed", "breakfast", "b&b", "inn", "accommodation"],
            "breakfast": ["breakfast", "b&b", "morning", "meal", "inn"],
            "glamping": ["glamping", "camping", "luxury", "tent", "outdoor"],
            
            # Health & Wellness themes
            "spas": ["spa", "wellness", "relaxation", "massage", "treatment", "therapeutic"],
            "springs": ["springs", "hot", "thermal", "natural", "healing", "mineral"],
            "yoga": ["yoga", "meditation", "mindfulness", "relaxation", "spiritual"],
            "meditation": ["meditation", "mindfulness", "peaceful", "quiet", "spiritual"],
            "therapeutic": ["therapeutic", "healing", "treatment", "wellness", "health"],
            "relaxation": ["relaxation", "calm", "peaceful", "tranquil", "rest"],
            
            # Transportation & Access themes  
            "transportation": ["transportation", "transport", "travel", "getting", "around"],
            "accessibility": ["accessibility", "accessible", "disabled", "wheelchair", "mobility"],
            "parking": ["parking", "park", "car", "vehicle", "lot"],
            "transit": ["transit", "bus", "public", "transportation", "commute"],
            "walkability": ["walkable", "walking", "pedestrian", "foot", "stroll"],
            "friendly": ["friendly", "accessible", "easy", "convenient", "welcoming"]
        }
        
        # Check direct keyword matches
        for keyword in theme_keywords:
            if keyword in text:
                return True
                
        # Check expanded keyword matches
        for keyword in theme_keywords:
            if keyword in keyword_expansions:
                if any(exp in text for exp in keyword_expansions[keyword]):
                    return True
            # Also check if any expansion base matches this keyword
            for base, expansions in keyword_expansions.items():
                if keyword == base or any(word in keyword for word in base.split()):
                    if any(exp in text for exp in expansions):
                        return True
                        
        return False
    
    def _generate_tags(self, theme_name: str) -> List[str]:
        """Generate relevant tags for a theme"""
        # Simple tag generation
        base_tags = theme_name.lower().replace("&", "").split()
        
        # Add related tags
        tag_additions = {
            "hiking": ["outdoor", "nature", "adventure"],
            "beaches": ["water", "relax", "summer"],
            "museums": ["culture", "history", "art"],
            "dining": ["food", "culinary", "taste"],
            "nightlife": ["evening", "social", "fun"],
            "family": ["kids", "children", "all-ages"]
        }
        
        tags = list(base_tags)
        for keyword, additions in tag_additions.items():
            if keyword in theme_name.lower():
                tags.extend(additions)
                
        return list(set(tags))  # Remove duplicates
    
    def _build_enhanced_themes(
        self, 
        validated_themes: List[Dict[str, Any]], 
        all_evidence: List[Evidence],
        cultural_result: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Build enhanced theme objects with full metadata"""
        enhanced_themes = []
        
        self.logger.info(f"Building enhanced themes: {len(validated_themes)} validated themes, {len(all_evidence)} evidence pieces")
        
        for idx, theme_data in enumerate(validated_themes):
            theme_name = theme_data["name"]
            self.logger.info(f"Processing theme {idx}: '{theme_name}'")
            
            # Extract confidence breakdown
            confidence_breakdown = theme_data.get("confidence_breakdown", {})
            
            # Build evidence list with cultural context
            theme_evidence = []
            evidence_matches = 0
            for ev_idx, ev in enumerate(all_evidence):
                # Match evidence to theme (simplified)
                if self._check_theme_match(theme_name, ev.text_snippet):
                    evidence_matches += 1
                    self.logger.info(f"  Evidence match {evidence_matches} for '{theme_name}': {ev.text_snippet[:100]}...")
                    
                    # Find cultural context for this source
                    cultural_context = None
                    for enhanced_source in cultural_result.get("enhanced_sources", []):
                        if enhanced_source.get("url") == ev.source_url:
                            cultural_context = enhanced_source.get("cultural_context")
                            break
                    
                    theme_evidence.append({
                        "id": ev.id,
                        "source_url": ev.source_url,
                        "source_category": ev.source_category.value,
                        "authority_weight": ev.authority_weight,
                        "text_snippet": ev.text_snippet[:200] + "...",
                        "cultural_context": cultural_context
                    })
            
            self.logger.info(f"  Theme '{theme_name}' matched {len(theme_evidence)} evidence pieces")
            
            enhanced_theme = {
                "theme_id": hashlib.md5(theme_data["name"].encode()).hexdigest()[:12],
                "name": theme_data["name"],
                "macro_category": self._get_macro_category(theme_data["name"]),
                "confidence_level": theme_data.get("confidence_level", "unknown"),
                "confidence_score": confidence_breakdown.get("total_confidence", 0.0),
                "confidence_breakdown": confidence_breakdown,
                "evidence_count": len(theme_evidence),
                "evidence_summary": theme_evidence[:5],  # Top 5 evidence pieces
                "is_validated": theme_data.get("is_validated", False),
                "contradiction_status": {
                    "has_contradictions": theme_data.get("contradiction_resolved", False),
                    "resolution": theme_data.get("winning_position", "none")
                },
                "tags": self._generate_tags(theme_data["name"])
            }
            
            enhanced_themes.append(enhanced_theme)
            
        return enhanced_themes
    
    def _get_macro_category(self, theme_name: str) -> str:
        """Get macro category for a theme"""
        for macro, micros in self.theme_taxonomy.items():
            if any(micro.lower() in theme_name.lower() for micro in micros):
                return macro
        return "Other"
    
    def _analyze_temporal_aspects(
        self, themes: List[Dict[str, Any]], evidence: List[Evidence]
    ) -> List[Dict[str, Any]]:
        """Analyze temporal aspects of themes"""
        # Simplified temporal analysis
        # In production would analyze seasonal mentions, events, etc.
        
        temporal_slices = []
        
        # Create a current/default slice
        current_slice = {
            "valid_from": datetime.now().isoformat(),
            "valid_to": None,
            "season": self._get_current_season(),
            "theme_strengths": {
                theme["name"]: theme["confidence_score"]
                for theme in themes
            },
            "seasonal_highlights": self._extract_seasonal_highlights(evidence)
        }
        
        temporal_slices.append(current_slice)
        
        return temporal_slices
    
    def _get_current_season(self) -> str:
        """Get current season"""
        month = datetime.now().month
        if month in [12, 1, 2]:
            return "winter"
        elif month in [3, 4, 5]:
            return "spring"
        elif month in [6, 7, 8]:
            return "summer"
        else:
            return "fall"
    
    def _extract_seasonal_highlights(self, evidence: List[Evidence]) -> Dict[str, Any]:
        """Extract seasonal highlights from evidence"""
        # Simplified - in production would use NLP
        seasonal_keywords = {
            "summer": ["summer", "beach", "swimming", "warm", "sunny"],
            "winter": ["winter", "snow", "ski", "cold", "christmas"],
            "spring": ["spring", "flower", "bloom", "easter"],
            "fall": ["fall", "autumn", "foliage", "harvest"]
        }
        
        highlights = {}
        for season, keywords in seasonal_keywords.items():
            mentions = 0
            for ev in evidence:
                text_lower = ev.text_snippet.lower()
                mentions += sum(1 for kw in keywords if kw in text_lower)
            if mentions > 0:
                highlights[season] = {"mention_count": mentions}
                
        return highlights
    
    def _calculate_dimensions(
        self, themes: List[Dict[str, Any]], evidence: List[Evidence]
    ) -> Dict[str, Any]:
        """Calculate destination dimensions based on themes and evidence"""
        dimensions = {}
        
        # Map themes to dimensions (simplified)
        theme_dimension_mapping = {
            "beaches": {"beach_cleanliness_index": 0.8},
            "hiking": {"outdoor_exercise_options": 0.9, "nature_score": 0.9},
            "museums": {"cultural_diversity_index": 0.8, "museum_gallery_density": 0.9},
            "nightlife": {"nightlife_vibrancy": 0.9},
            "family": {"family_facilities_score": 0.8, "kid_attraction_density": 0.8},
            "shopping": {"shopping_variety": 0.8},
            "dining": {"culinary_diversity": 0.8}
        }
        
        # Calculate dimension values based on theme presence and confidence
        for theme in themes:
            theme_name_lower = theme["name"].lower()
            confidence = theme["confidence_score"]
            
            for keyword, dim_values in theme_dimension_mapping.items():
                if keyword in theme_name_lower:
                    for dim_name, base_value in dim_values.items():
                        if dim_name not in dimensions:
                            dimensions[dim_name] = {
                                "value": base_value * confidence,
                                "confidence": confidence,
                                "evidence_themes": [theme["name"]]
                            }
                        else:
                            # Average if multiple themes contribute
                            current = dimensions[dim_name]
                            new_value = (current["value"] + base_value * confidence) / 2
                            new_confidence = (current["confidence"] + confidence) / 2
                            current["value"] = new_value
                            current["confidence"] = new_confidence
                            current["evidence_themes"].append(theme["name"])
                            
        return dimensions
    
    def _split_into_chunks(self, text: str, chunk_size: int = 500) -> List[str]:
        """Split text into chunks"""
        words = text.split()
        chunks = []
        
        for i in range(0, len(words), chunk_size):
            chunk = " ".join(words[i:i + chunk_size])
            if len(chunk.strip()) > 50:  # Minimum chunk size
                chunks.append(chunk)
                
        return chunks
    
    def _get_source_distribution(self, evidence: List[Evidence]) -> Dict[str, int]:
        """Get distribution of evidence by source category"""
        distribution = {}
        for ev in evidence:
            category = ev.source_category.value
            distribution[category] = distribution.get(category, 0) + 1
        return distribution
    
    def _calculate_average_confidence(self, themes: List[Dict[str, Any]]) -> float:
        """Calculate average confidence across themes"""
        if not themes:
            return 0.0
            
        total_confidence = sum(theme.get("confidence_score", 0.0) for theme in themes)
        return total_confidence / len(themes)

    def _setup_logger(self):
        """Setup logger for the enhanced theme analysis tool"""
        import logging
        return logging.getLogger(__name__)

def create_enhanced_theme_analysis_tool() -> Tool:
    """Factory function to create the tool for LangChain"""
    tool_instance = EnhancedThemeAnalysisTool()
    
    async def run_analysis(
        destination_name: str,
        country_code: str,
        text_content_list: List[Dict[str, Any]],
        analyze_temporal: bool = True,
        min_confidence: float = 0.5
    ) -> Dict[str, Any]:
        """Wrapper function for the tool"""
        input_data = EnhancedThemeAnalysisInput(
            destination_name=destination_name,
            country_code=country_code,
            text_content_list=text_content_list,
            analyze_temporal=analyze_temporal,
            min_confidence=min_confidence
        )
        return await tool_instance.analyze_themes(input_data)
    
    return Tool(
        name=tool_instance.name,
        description=tool_instance.description,
        func=run_analysis
    )

class EnhancedAnalyzeThemesFromEvidenceTool(Tool):
    """
    Enhanced LangChain Tool for theme analysis with agent orchestration integration
    """
    
    def __init__(self, agent_orchestrator=None, llm=None):
        """
        Initialize enhanced theme analysis tool
        
        Args:
            agent_orchestrator: Optional agent orchestrator for multi-agent validation
            llm: Optional LLM instance for enhanced analysis
        """
        super().__init__(
            name="analyze_themes_from_evidence",
            description="Enhanced theme analysis with evidence-based confidence scoring and multi-agent validation",
            func=self._create_analysis_function(agent_orchestrator, llm)
        )
    
    def _create_analysis_function(self, agent_orchestrator, llm):
        """Create the analysis function with captured dependencies"""
        def create_tool_func(orchestrator, llm_instance):
            theme_analyzer = EnhancedThemeAnalysisTool()
            
            # Store LLM reference for potential use in analysis
            if llm_instance:
                theme_analyzer.llm = llm_instance
            
            # If agent orchestrator provided, replace the analyzer's agents
            if orchestrator:
                # Get agents from orchestrator
                for agent_id, agent in orchestrator.broker.agents.items():
                    if isinstance(agent, ValidationAgent):
                        theme_analyzer.validation_agent = agent
                    elif isinstance(agent, CulturalPerspectiveAgent):
                        theme_analyzer.cultural_agent = agent
                    elif isinstance(agent, ContradictionDetectionAgent):
                        theme_analyzer.contradiction_agent = agent
            
            async def _arun(
                destination_name: str,
                country_code: str = "US",
                text_content_list: Optional[List[Dict[str, Any]]] = None,
                evidence_snippets: Optional[List[Dict[str, Any]]] = None,
                seed_themes_with_evidence: Optional[Dict[str, Any]] = None,
                config=None,  # Added to satisfy LangChain's requirements
                **kwargs
            ) -> Dict[str, Any]:
                """Enhanced theme analysis execution"""
                try:
                    logger.info(f"=== Enhanced Theme Analysis Starting for {destination_name} ===")
                    logger.info(f"Received parameters:")
                    logger.info(f"  - text_content_list: {type(text_content_list)}, length: {len(text_content_list) if text_content_list else 0}")
                    logger.info(f"  - evidence_snippets: {type(evidence_snippets)}, length: {len(evidence_snippets) if evidence_snippets else 0}")
                    logger.info(f"  - seed_themes_with_evidence: {type(seed_themes_with_evidence)}, keys: {list(seed_themes_with_evidence.keys()) if seed_themes_with_evidence else []}")
                    
                    # Handle backward compatibility with evidence_snippets
                    if text_content_list is None and evidence_snippets is not None:
                        logger.info("Converting evidence_snippets to text_content_list format")
                        text_content_list = [
                            {
                                "url": snippet.get("source_url", ""),
                                "content": snippet.get("content", ""),
                                "title": snippet.get("title", "")
                            }
                            for snippet in evidence_snippets
                        ]
                    elif text_content_list is None:
                        logger.warning("No content provided - initializing empty list")
                        text_content_list = []
                    
                    # If we have PageContent objects, convert them to the expected format
                    formatted_content_list = []
                    logger.info(f"Converting {len(text_content_list) if text_content_list else 0} content items")
                    
                    for idx, item in enumerate(text_content_list or []):
                        logger.info(f"Processing item {idx}: type={type(item)}")
                        
                        if hasattr(item, 'url') and hasattr(item, 'content'):
                            # It's a PageContent object
                            formatted_item = {
                                "url": item.url,
                                "content": item.content,
                                "title": getattr(item, 'title', '')
                            }
                            formatted_content_list.append(formatted_item)
                            logger.info(f"Converted PageContent {idx}: url={item.url[:50]}..., content_length={len(item.content) if item.content else 0}")
                        elif isinstance(item, dict):
                            # Already in dict format
                            formatted_content_list.append(item)
                            logger.info(f"Dict item {idx}: url={item.get('url', 'N/A')[:50]}..., content_length={len(item.get('content', ''))}")
                        else:
                            logger.warning(f"Unexpected content item type: {type(item)} at index {idx}")
                    
                    logger.info(f"Formatted content list has {len(formatted_content_list)} items")
                    
                    # Log sample of formatted content
                    if formatted_content_list:
                        first_item = formatted_content_list[0]
                        logger.info(f"First formatted item preview:")
                        logger.info(f"  - URL: {first_item.get('url', 'N/A')[:100]}")
                        logger.info(f"  - Title: {first_item.get('title', 'N/A')[:100]}")
                        logger.info(f"  - Content length: {len(first_item.get('content', ''))}")
                        if first_item.get('content'):
                            logger.info(f"  - Content preview: {first_item['content'][:200]}...")
                    
                    # Run enhanced analysis
                    input_data = EnhancedThemeAnalysisInput(
                        destination_name=destination_name,
                        country_code=country_code,
                        text_content_list=formatted_content_list,
                        analyze_temporal=kwargs.get("analyze_temporal", True),
                        min_confidence=kwargs.get("min_confidence", 0.5)
                    )
                    
                    logger.info(f"Created input data with {len(input_data.text_content_list)} content items")
                    
                    result = await theme_analyzer.analyze_themes(input_data)
                    
                    # Convert result to backward compatible format if needed
                    # The basic tool expects validated_themes and discovered_themes
                    if "themes" in result:
                        # Create ThemeInsightOutput-like structure for backward compatibility
                        from src.schemas import ThemeInsightOutput, DestinationInsight
                        
                        validated_themes = []
                        discovered_themes = []
                        
                        for theme in result["themes"]:
                            # Preserve evidence information in description and create rich evidence list
                            evidence_list = []
                            evidence_string_list = []
                            for ev in theme.get("evidence_summary", []):
                                evidence_dict = {
                                    "source_url": ev.get("source_url", ""),
                                    "source_category": ev.get("source_category", ""),
                                    "authority_weight": ev.get("authority_weight", 0.0),
                                    "text_snippet": ev.get("text_snippet", ""),
                                    "cultural_context": ev.get("cultural_context")
                                }
                                evidence_list.append(evidence_dict)
                                
                                # Create string version for DestinationInsight schema compatibility
                                evidence_string = f"[{ev.get('source_category', 'Unknown')}] {ev.get('text_snippet', '')[:100]}..."
                                evidence_string_list.append(evidence_string)
                            
                            # Create rich description with evidence count and confidence info
                            description = (
                                f"{theme.get('name', '')} experiences in {destination_name}. "
                                f"Confidence: {theme.get('confidence_level', 'unknown')} "
                                f"({theme.get('confidence_score', 0.0):.2f}). "
                                f"Evidence: {theme.get('evidence_count', 0)} pieces. "
                                f"Category: {theme.get('macro_category', 'Other')}"
                            )
                            
                            theme_insight = DestinationInsight(
                                destination_name=destination_name,
                                insight_type=theme.get("macro_category", "Other"),
                                insight_name=theme.get("name", "Unknown"),
                                description=description,
                                confidence_score=theme.get("confidence_score", 0.0),
                                evidence=evidence_string_list,  # Use string list for schema compatibility
                                source_urls=[ev.get("source_url", "") for ev in theme.get("evidence_summary", [])]
                            )
                            
                            # Store enhanced metadata in the evidence field for later processing (instead of as attributes)
                            # Enhanced data is preserved in evidence_details list and description
                            
                            if theme.get("is_validated", False):
                                validated_themes.append(theme_insight)
                            else:
                                discovered_themes.append(theme_insight)
                        
                        # Return backward compatible format
                        return ThemeInsightOutput(
                            destination_name=destination_name,
                            validated_themes=validated_themes,
                            discovered_themes=discovered_themes
                        )
                    
                    # Log summary
                    logger.info(
                        f"Enhanced analysis complete for {destination_name}: "
                        f"{result.get('quality_metrics', {}).get('themes_validated', 0)} themes validated, "
                        f"avg confidence {result.get('quality_metrics', {}).get('average_confidence', 0.0):.2f}"
                    )
                    
                    return result
                    
                except Exception as e:
                    logger.error(f"Error in enhanced theme analysis: {e}", exc_info=True)
                    # Return minimal result on error
                    from src.schemas import ThemeInsightOutput
                    return ThemeInsightOutput(
                        destination_name=destination_name,
                        validated_themes=[],
                        discovered_themes=[]
                    )
            
            def _run(**kwargs):
                """Synchronous wrapper"""
                import asyncio
                # Use asyncio.run() which creates a new event loop
                return asyncio.run(_arun(**kwargs))
            
            return _run
        
        # Create the functions with captured dependencies
        func = create_tool_func(agent_orchestrator, llm)
        
        return func 