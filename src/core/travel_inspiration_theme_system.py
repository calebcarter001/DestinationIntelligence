#!/usr/bin/env python3
"""
Travel Inspiration Theme System
Redesigned to prioritize travel inspiration over local administrative details
"""

import re
from typing import Dict, List, Set, Any, Optional
from dataclasses import dataclass
from enum import Enum
import logging

logger = logging.getLogger(__name__)

class ThemeCategory(Enum):
    """Travel-focused theme categories in priority order"""
    POPULAR = "popular"      # Trending, must-see, Instagram-worthy
    POI = "poi"             # Specific places, landmarks, attractions
    CULTURAL = "cultural"    # Authentic local experiences
    PRACTICAL = "practical" # Essential travel information

@dataclass
class TravelTheme:
    """Travel-focused theme with inspiration scoring"""
    name: str
    category: ThemeCategory
    inspiration_score: float  # 0-1, how inspiring for travelers
    specificity_score: float  # 0-1, how specific vs generic
    actionability_score: float # 0-1, how actionable for travelers
    evidence_count: int
    poi_names: List[str] = None  # Specific POI names mentioned
    trending_indicators: List[str] = None  # Social media, reviews, etc.
    
class TravelInspirationThemeSystem:
    """Redesigned theme system focused on travel inspiration"""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        
        # POPULAR THEME PATTERNS - What travelers actually want to see
        self.popular_patterns = {
            "must_see_attractions": [
                r"must see", r"don't miss", r"bucket list", r"iconic", r"famous for",
                r"world renowned", r"legendary", r"unmissable", r"top attraction"
            ],
            "instagram_worthy": [
                r"instagram", r"photo", r"scenic", r"viewpoint", r"panoramic",
                r"sunset", r"sunrise", r"photography", r"picture perfect"
            ],
            "trending_experiences": [
                r"trending", r"popular", r"viral", r"everyone's talking about",
                r"hot spot", r"buzzing", r"latest", r"new attraction"
            ],
            "unique_experiences": [
                r"unique", r"one of a kind", r"nowhere else", r"only place",
                r"exclusive", r"rare", r"special", r"extraordinary"
            ]
        }
        
        # POI EXTRACTION PATTERNS - Specific places and venues
        self.poi_patterns = {
            "landmarks": [
                r"observatory", r"monument", r"tower", r"bridge", r"cathedral",
                r"castle", r"palace", r"fort", r"lighthouse", r"statue"
            ],
            "natural_attractions": [
                r"national park", r"state park", r"canyon", r"mountain", r"peak",
                r"lake", r"river", r"falls", r"waterfall", r"trail", r"forest"
            ],
            "venues": [
                r"brewery", r"restaurant", r"cafe", r"bar", r"hotel", r"resort",
                r"museum", r"gallery", r"theater", r"venue", r"center", r"market"
            ],
            "districts": [
                r"downtown", r"historic district", r"old town", r"quarter",
                r"neighborhood", r"area", r"district", r"strip", r"square"
            ]
        }
        
        # CULTURAL THEME PATTERNS - Authentic experiences
        self.cultural_patterns = {
            "authentic_experiences": [
                r"authentic", r"traditional", r"local", r"heritage", r"culture",
                r"customs", r"rituals", r"festivals", r"celebrations"
            ],
            "local_specialties": [
                r"local specialty", r"regional", r"native", r"indigenous",
                r"signature", r"famous for", r"known for"
            ],
            "arts_and_crafts": [
                r"artisan", r"craft", r"handmade", r"pottery", r"weaving",
                r"art", r"gallery", r"studio", r"workshop"
            ]
        }
        
        # PRACTICAL THEME PATTERNS - Essential travel info (lowest priority)
        self.practical_patterns = {
            "transportation": [r"airport", r"train", r"bus", r"parking", r"uber"],
            "accommodation": [r"hotel", r"hostel", r"airbnb", r"lodging"],
            "safety": [r"safe", r"crime", r"police", r"emergency"],
            "costs": [r"price", r"cost", r"budget", r"expensive", r"cheap"]
        }
        
        # Theme limits by category (prioritize inspiration)
        self.category_limits = {
            ThemeCategory.POPULAR: 3,    # Top 3 popular attractions
            ThemeCategory.POI: 4,        # Top 4 specific places
            ThemeCategory.CULTURAL: 2,   # Top 2 cultural experiences
            ThemeCategory.PRACTICAL: 1   # Only 1 practical theme
        }
        
        # Minimum scores for theme inclusion
        self.min_scores = {
            "inspiration_score": 0.7,    # Must be inspiring
            "specificity_score": 0.6,    # Must be specific enough
            "actionability_score": 0.5   # Must be actionable
        }

    def extract_travel_themes(self, evidence_list: List[Dict[str, Any]], 
                            destination_name: str) -> List[TravelTheme]:
        """Extract travel-focused themes from evidence"""
        
        logger.info(f"🎯 Extracting travel themes for {destination_name}")
        
        # Step 1: Extract POI names from evidence
        poi_names = self._extract_poi_names(evidence_list, destination_name)
        logger.info(f"📍 Found {len(poi_names)} specific POIs: {poi_names[:5]}")
        
        # Step 2: Categorize evidence by theme type
        categorized_evidence = self._categorize_evidence(evidence_list)
        
        # Step 3: Generate themes by category (in priority order)
        all_themes = []
        
        # POPULAR themes (highest priority)
        popular_themes = self._generate_popular_themes(
            categorized_evidence.get("popular", []), poi_names, destination_name
        )
        all_themes.extend(popular_themes[:self.category_limits[ThemeCategory.POPULAR]])
        
        # POI themes (second priority)
        poi_themes = self._generate_poi_themes(
            categorized_evidence.get("poi", []), poi_names, destination_name
        )
        all_themes.extend(poi_themes[:self.category_limits[ThemeCategory.POI]])
        
        # CULTURAL themes (third priority)
        cultural_themes = self._generate_cultural_themes(
            categorized_evidence.get("cultural", []), poi_names, destination_name
        )
        all_themes.extend(cultural_themes[:self.category_limits[ThemeCategory.CULTURAL]])
        
        # PRACTICAL themes (lowest priority)
        practical_themes = self._generate_practical_themes(
            categorized_evidence.get("practical", []), destination_name
        )
        all_themes.extend(practical_themes[:self.category_limits[ThemeCategory.PRACTICAL]])
        
        # Step 4: Filter by quality scores
        quality_themes = self._filter_by_quality(all_themes)
        
        logger.info(f"✅ Generated {len(quality_themes)} high-quality travel themes")
        return quality_themes

    def _extract_poi_names(self, evidence_list: List[Dict[str, Any]], 
                          destination_name: str) -> List[str]:
        """Extract specific POI names from evidence"""
        poi_names = set()
        
        for evidence in evidence_list:
            text = evidence.get("text_snippet", "")
            
            # Extract proper nouns that could be POI names
            # Look for capitalized phrases that aren't the destination name
            proper_nouns = re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', text)
            
            for noun in proper_nouns:
                # Skip if it's the destination name
                if destination_name.lower() in noun.lower():
                    continue
                    
                # Check if it matches POI patterns
                noun_lower = noun.lower()
                for category, patterns in self.poi_patterns.items():
                    for pattern in patterns:
                        if re.search(pattern, noun_lower) or any(
                            poi_word in noun_lower for poi_word in 
                            ["observatory", "brewery", "museum", "park", "center", "trail"]
                        ):
                            poi_names.add(noun)
                            break
        
        return list(poi_names)

    def _categorize_evidence(self, evidence_list: List[Dict[str, Any]]) -> Dict[str, List]:
        """Categorize evidence by theme type"""
        categorized = {
            "popular": [],
            "poi": [],
            "cultural": [],
            "practical": []
        }
        
        for evidence in evidence_list:
            text = evidence.get("text_snippet", "").lower()
            
            # Check for popular indicators (highest priority)
            if self._matches_patterns(text, self.popular_patterns):
                categorized["popular"].append(evidence)
            # Check for POI indicators
            elif self._matches_patterns(text, self.poi_patterns):
                categorized["poi"].append(evidence)
            # Check for cultural indicators
            elif self._matches_patterns(text, self.cultural_patterns):
                categorized["cultural"].append(evidence)
            # Check for practical indicators (lowest priority)
            elif self._matches_patterns(text, self.practical_patterns):
                categorized["practical"].append(evidence)
            else:
                # Default to POI if it mentions specific places
                if any(word in text for word in ["visit", "see", "go to", "located", "attraction"]):
                    categorized["poi"].append(evidence)
        
        return categorized

    def _matches_patterns(self, text: str, pattern_dict: Dict[str, List[str]]) -> bool:
        """Check if text matches any patterns in the pattern dictionary"""
        for category, patterns in pattern_dict.items():
            for pattern in patterns:
                if re.search(pattern, text):
                    return True
        return False

    def _generate_popular_themes(self, evidence_list: List[Dict], 
                               poi_names: List[str], destination_name: str) -> List[TravelTheme]:
        """Generate popular/trending themes"""
        themes = []
        
        # Group evidence by trending topics
        trending_topics = self._extract_trending_topics(evidence_list)
        
        for topic, topic_evidence in trending_topics.items():
            if len(topic_evidence) >= 2:  # Require multiple mentions
                theme = TravelTheme(
                    name=f"{topic} in {destination_name}",
                    category=ThemeCategory.POPULAR,
                    inspiration_score=self._calculate_inspiration_score(topic_evidence, "popular"),
                    specificity_score=self._calculate_specificity_score(topic, poi_names),
                    actionability_score=self._calculate_actionability_score(topic_evidence),
                    evidence_count=len(topic_evidence),
                    poi_names=[poi for poi in poi_names if poi.lower() in topic.lower()],
                    trending_indicators=self._extract_trending_indicators(topic_evidence)
                )
                themes.append(theme)
        
        # Sort by inspiration score
        return sorted(themes, key=lambda t: t.inspiration_score, reverse=True)

    def _generate_poi_themes(self, evidence_list: List[Dict], 
                           poi_names: List[str], destination_name: str) -> List[TravelTheme]:
        """Generate specific POI themes"""
        themes = []
        
        # Create themes for each significant POI
        for poi_name in poi_names:
            poi_evidence = [e for e in evidence_list 
                          if poi_name.lower() in e.get("text_snippet", "").lower()]
            
            if poi_evidence:
                theme = TravelTheme(
                    name=poi_name,
                    category=ThemeCategory.POI,
                    inspiration_score=self._calculate_inspiration_score(poi_evidence, "poi"),
                    specificity_score=1.0,  # POIs are always specific
                    actionability_score=self._calculate_actionability_score(poi_evidence),
                    evidence_count=len(poi_evidence),
                    poi_names=[poi_name]
                )
                themes.append(theme)
        
        return sorted(themes, key=lambda t: t.inspiration_score, reverse=True)

    def _generate_cultural_themes(self, evidence_list: List[Dict], 
                                poi_names: List[str], destination_name: str) -> List[TravelTheme]:
        """Generate cultural experience themes"""
        themes = []
        
        cultural_topics = self._extract_cultural_topics(evidence_list)
        
        for topic, topic_evidence in cultural_topics.items():
            if len(topic_evidence) >= 2:
                theme = TravelTheme(
                    name=f"{topic} Experience",
                    category=ThemeCategory.CULTURAL,
                    inspiration_score=self._calculate_inspiration_score(topic_evidence, "cultural"),
                    specificity_score=self._calculate_specificity_score(topic, poi_names),
                    actionability_score=self._calculate_actionability_score(topic_evidence),
                    evidence_count=len(topic_evidence),
                    poi_names=[poi for poi in poi_names if poi.lower() in topic.lower()]
                )
                themes.append(theme)
        
        return sorted(themes, key=lambda t: t.inspiration_score, reverse=True)

    def _generate_practical_themes(self, evidence_list: List[Dict], 
                                 destination_name: str) -> List[TravelTheme]:
        """Generate essential practical themes (minimal)"""
        themes = []
        
        # Only create one essential practical theme
        if evidence_list:
            theme = TravelTheme(
                name=f"Travel Essentials for {destination_name}",
                category=ThemeCategory.PRACTICAL,
                inspiration_score=0.3,  # Low inspiration
                specificity_score=0.5,
                actionability_score=0.8,  # High actionability
                evidence_count=len(evidence_list)
            )
            themes.append(theme)
        
        return themes

    def _extract_trending_topics(self, evidence_list: List[Dict]) -> Dict[str, List]:
        """Extract trending topics from popular evidence"""
        topics = {}
        
        for evidence in evidence_list:
            text = evidence.get("text_snippet", "")
            
            # Extract key phrases that indicate popular attractions
            for pattern_category, patterns in self.popular_patterns.items():
                for pattern in patterns:
                    matches = re.finditer(pattern, text.lower())
                    for match in matches:
                        # Extract surrounding context as topic
                        start = max(0, match.start() - 20)
                        end = min(len(text), match.end() + 20)
                        context = text[start:end].strip()
                        
                        # Clean up the topic name
                        topic = self._clean_topic_name(context)
                        if topic and len(topic) > 3:
                            if topic not in topics:
                                topics[topic] = []
                            topics[topic].append(evidence)
        
        return topics

    def _extract_cultural_topics(self, evidence_list: List[Dict]) -> Dict[str, List]:
        """Extract cultural topics from cultural evidence"""
        topics = {}
        
        cultural_keywords = ["heritage", "tradition", "culture", "festival", "art", "craft"]
        
        for evidence in evidence_list:
            text = evidence.get("text_snippet", "")
            
            for keyword in cultural_keywords:
                if keyword in text.lower():
                    # Use the keyword as the topic
                    topic = keyword.title()
                    if topic not in topics:
                        topics[topic] = []
                    topics[topic].append(evidence)
        
        return topics

    def _clean_topic_name(self, context: str) -> str:
        """Clean and extract a meaningful topic name from context"""
        # Remove common words and clean up
        words = context.split()
        meaningful_words = [w for w in words if len(w) > 3 and w.lower() not in 
                          ["the", "and", "for", "with", "this", "that", "from", "they"]]
        
        if meaningful_words:
            return " ".join(meaningful_words[:3]).title()
        return ""

    def _calculate_inspiration_score(self, evidence_list: List[Dict], category: str) -> float:
        """Calculate how inspiring this theme is for travelers"""
        score = 0.5  # Base score
        
        inspiration_keywords = [
            "amazing", "stunning", "breathtaking", "incredible", "spectacular",
            "beautiful", "gorgeous", "magnificent", "awesome", "fantastic"
        ]
        
        for evidence in evidence_list:
            text = evidence.get("text_snippet", "").lower()
            
            # Boost for inspiration keywords
            keyword_count = sum(1 for keyword in inspiration_keywords if keyword in text)
            score += keyword_count * 0.1
            
            # Category-specific boosts
            if category == "popular":
                if any(word in text for word in ["must", "famous", "iconic", "trending"]):
                    score += 0.2
            elif category == "poi":
                if any(word in text for word in ["visit", "see", "attraction", "landmark"]):
                    score += 0.15
        
        return min(1.0, score)

    def _calculate_specificity_score(self, topic: str, poi_names: List[str]) -> float:
        """Calculate how specific (vs generic) this theme is"""
        # Higher score for specific POI names
        if any(poi.lower() in topic.lower() for poi in poi_names):
            return 0.9
        
        # Medium score for specific activities
        specific_words = ["observatory", "brewery", "trail", "museum", "park"]
        if any(word in topic.lower() for word in specific_words):
            return 0.7
        
        # Lower score for generic terms
        generic_words = ["culture", "heritage", "experience", "activity"]
        if any(word in topic.lower() for word in generic_words):
            return 0.4
        
        return 0.6

    def _calculate_actionability_score(self, evidence_list: List[Dict]) -> float:
        """Calculate how actionable this theme is for travelers"""
        score = 0.5
        
        actionable_keywords = [
            "visit", "go to", "see", "experience", "try", "book", "reserve",
            "open", "hours", "location", "address", "website", "phone"
        ]
        
        for evidence in evidence_list:
            text = evidence.get("text_snippet", "").lower()
            keyword_count = sum(1 for keyword in actionable_keywords if keyword in text)
            score += keyword_count * 0.05
        
        return min(1.0, score)

    def _extract_trending_indicators(self, evidence_list: List[Dict]) -> List[str]:
        """Extract indicators that show this is trending"""
        indicators = []
        
        trending_patterns = [
            r"viral", r"trending", r"popular", r"instagram", r"social media",
            r"everyone's talking", r"hot spot", r"buzzing"
        ]
        
        for evidence in evidence_list:
            text = evidence.get("text_snippet", "")
            for pattern in trending_patterns:
                if re.search(pattern, text.lower()):
                    indicators.append(pattern)
        
        return list(set(indicators))

    def _filter_by_quality(self, themes: List[TravelTheme]) -> List[TravelTheme]:
        """Filter themes by quality scores"""
        quality_themes = []
        
        for theme in themes:
            if (theme.inspiration_score >= self.min_scores["inspiration_score"] and
                theme.specificity_score >= self.min_scores["specificity_score"] and
                theme.actionability_score >= self.min_scores["actionability_score"]):
                quality_themes.append(theme)
        
        return quality_themes

    def get_theme_display_order(self) -> List[ThemeCategory]:
        """Get the display order for theme categories"""
        return [
            ThemeCategory.POPULAR,
            ThemeCategory.POI,
            ThemeCategory.CULTURAL,
            ThemeCategory.PRACTICAL
        ] 