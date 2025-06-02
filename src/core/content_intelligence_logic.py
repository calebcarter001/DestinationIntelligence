import logging
import re
from typing import Dict, List, Optional, Any, Tuple
from transformers import pipeline
import os
from tqdm import tqdm # For sync iteration

# Adjusted import path for data_models
from ..data_models import DestinationInsight
from ..schemas import PageContent, ThemeInsightOutput, ChromaSearchResult, DestinationInsight as PydanticDestinationInsight # Added ChromaSearchResult and PydanticDestinationInsight

class ContentIntelligenceLogic:
    """Core logic for content analysis using DistilBERT."""
    
    def __init__(self, config: Dict[str, Any]):
        self.logger = logging.getLogger(__name__ + '.ContentIntelligenceLogic') # Updated logger name
        self.sentiment_analyzer = None
        self.config = config.get("content_intelligence", {})
        self._load_sentiment_model()
        
        self.seed_themes = config.get("seed_themes", [
            "culture", "history", "nature", "food", "adventure", "art", "architecture",
            "romance", "family", "luxury", "budget", "nightlife", "museums", "shopping",
            "beaches", "mountains", "festivals", "traditional", "modern", "authentic",
            "dining", "entertainment", "relaxation", "scenic", "urban", "coastal",
            "historical", "contemporary", "activities", "experiences", "atmosphere",
            "wellness", "spirituality", "photography", "wildlife", "trekking", "hiking",
            "sustainability", "eco-tourism", "local life", "crafts", "markets", "music"
        ])
        
        self.theme_keywords = config.get("theme_keywords", {
            "romantic": ["romantic", "couples getaway", "love", "intimate dining", "honeymoon spot", "charming ambiance"],
            "cultural": ["cultural heritage", "local traditions", "historic sites", "museum visits", "art galleries", "traditional performances"],
            "artistic": ["art scene", "creative workshops", "galleries", "street art", "artisan crafts", "cultural expressions"],
            "culinary": ["gourmet food", "local cuisine", "fine dining", "street food tour", "cooking class", "wine tasting"],
            "adventure": ["outdoor adventure", "thrill-seeking", "extreme sports", "hiking trails", "water sports", "zip-lining"],
            "luxury": ["luxury travel", "upscale resorts", "premium services", "exclusive experiences", "high-end shopping", "private tours"],
            "family friendly": ["family vacation", "kids activities", "child-friendly attractions", "theme parks", "educational fun"], 
            "nature & outdoors": ["natural beauty", "wildlife viewing", "national parks", "scenic landscapes", "botanical gardens", "bird watching"], 
            "historical sites": ["ancient ruins", "historical landmarks", "world heritage sites", "medieval castles", "battlefields"], 
            "modern architecture": ["contemporary design", "innovative buildings", "skyline views", "modern art museums", "architectural tours"], 
            "nightlife": ["bars", "clubs", "live music", "evening entertainment", "rooftop bars"],
            "shopping": ["boutiques", "local markets", "shopping malls", "designer stores", "souvenirs"],
            "wellness & spa": ["spa retreats", "yoga sessions", "meditation", "health resorts", "holistic therapies"],
            "sustainability": ["eco-friendly", "sustainable tourism", "conservation efforts", "green initiatives"]
        })
        for theme in self.seed_themes:
            if theme not in self.theme_keywords and theme.replace(" ", "_") not in self.theme_keywords: 
                self.theme_keywords[theme] = [theme] if " " not in theme else [theme, theme.replace(" ", "_")]

        self.min_validated_theme_confidence = self.config.get("min_validated_theme_confidence", 0.60)
        self.min_discovered_theme_confidence = self.config.get("min_discovered_theme_confidence", 0.60)
        self.max_discovered_themes_per_destination = self.config.get("max_discovered_themes_per_destination", 5)

    def _load_sentiment_model(self):
        self.logger.info("🤖 Loading DistilBERT model (sentiment)...")
        try:
            device_to_use = 'mps' if os.name == 'darwin' else -1
            self.sentiment_analyzer = pipeline("sentiment-analysis", model="distilbert-base-uncased", device=device_to_use)
            if device_to_use == 'mps' and self.sentiment_analyzer.device.type == 'mps':
                self.logger.info("✅ DistilBERT sentiment model loaded on MPS (macOS).")
            else:
                self.logger.info("✅ DistilBERT sentiment model loaded on CPU.")
        except Exception as e:
            self.logger.error(f"Failed to load DistilBERT model (tried device: {device_to_use}): {e}. Sentiment analysis will be impacted. Falling back to CPU if possible.")
            try:
                self.sentiment_analyzer = pipeline("sentiment-analysis", model="distilbert-base-uncased", device=-1)
                self.logger.info("✅ DistilBERT sentiment model successfully loaded on CPU as fallback.")
            except Exception as e_cpu:
                 self.logger.error(f"Failed to load DistilBERT model on CPU as fallback: {e_cpu}. Sentiment features disabled.")
                 self.sentiment_analyzer = None

    def _analyze_theme_in_content(self, content: str, theme: str) -> Optional[Dict]:
        content_lower = content.lower()
        keywords_to_check = []
        if theme in self.theme_keywords:
            keywords_to_check = self.theme_keywords[theme]
        elif theme.lower() in self.theme_keywords:
             keywords_to_check = self.theme_keywords[theme.lower()]
        else: 
            keywords_to_check = [theme.lower()]

        if " " in theme:
            parts = theme.split()
            for part in parts:
                if part in self.theme_keywords:
                    keywords_to_check.extend(self.theme_keywords[part])
                elif len(part) > 3 :
                    keywords_to_check.append(part)
        
        keywords_to_check = list(set(kw.lower() for kw in keywords_to_check))

        found_keywords_details = []
        for keyword in keywords_to_check:
            escaped_keyword = re.escape(keyword)
            try:
                for match in re.finditer(r'\b' + escaped_keyword + r'\b', content_lower):
                    start, end = match.span()
                    snippet_start = max(0, start - 50)
                    snippet_end = min(len(content_lower), end + 50)
                    snippet = content[snippet_start:snippet_end]
                    original_cased_keyword_match = re.search(re.escape(keyword), snippet, re.IGNORECASE)
                    display_keyword = original_cased_keyword_match.group(0) if original_cased_keyword_match else keyword
                    found_keywords_details.append({"keyword": display_keyword, "snippet": snippet.strip()})
            except re.error as re_err:
                self.logger.warning(f"Regex error for keyword '{keyword}' in theme '{theme}': {re_err}")
                continue 

        if not found_keywords_details:
            return None

        unique_keywords_found = set(det["keyword"].lower() for det in found_keywords_details)
        num_unique_keywords = len(unique_keywords_found)
        total_mentions = len(found_keywords_details)

        confidence = 0.0
        if num_unique_keywords == 1:
            confidence = 0.3 + min(total_mentions / 10, 0.2)
        elif num_unique_keywords == 2:
            confidence = 0.5 + min(total_mentions / 10, 0.2)
        elif num_unique_keywords >= 3:
            confidence = 0.7 + min(total_mentions / 10, 0.2)

        if total_mentions > 1 and content_lower: 
            mention_density_factor = min( (total_mentions * 1000) / len(content_lower), 1.0) 
            if mention_density_factor > 0.1: 
                 confidence = min(confidence + 0.1, 1.0)
        
        return {
            "theme": theme,
            "confidence": round(confidence, 2),
            "mentions_details": found_keywords_details[:3],
            "keyword_variety_count": num_unique_keywords,
            "total_mentions": total_mentions
        }

    async def validate_themes_with_real_content(self, destination: str, sources: List[PageContent]) -> List[DestinationInsight]:
        self.logger.info(f"🎯 Validating {len(self.seed_themes)} seed themes for {destination} using {len(sources)} sources.")
        validated_insights_map = {}

        # Wrap sources with tqdm for progress
        for source_idx, source_obj in enumerate(tqdm(sources, desc=f"Analyzing sources for {destination}", unit="source")):
            # Access attributes directly from PageContent object
            content = source_obj.content 
            source_url = source_obj.url
            if not content:
                continue
            
            for theme_idx, theme in enumerate(self.seed_themes):
                analysis = self._analyze_theme_in_content(content, theme)
                
                if analysis and analysis["confidence"] > 0.35: 
                    if theme not in validated_insights_map:
                        validated_insights_map[theme] = {
                            "confidences": [],
                            "all_snippets": [],
                            "source_urls": set(),
                            "mention_counts": 0,
                            "keyword_variety_sum": 0
                        }
                    
                    validated_insights_map[theme]["confidences"].append(analysis["confidence"])
                    validated_insights_map[theme]["source_urls"].add(source_url)
                    validated_insights_map[theme]["mention_counts"] += analysis["total_mentions"]
                    validated_insights_map[theme]["keyword_variety_sum"] += analysis["keyword_variety_count"]
                    
                    for detail in analysis["mentions_details"]:
                        snippet_with_source = f'"{detail["snippet"]}" (Source: {source_url})'
                        if not any(s.startswith(f'"{detail["snippet"][:50]}"') for s in validated_insights_map[theme]["all_snippets"]):
                             validated_insights_map[theme]["all_snippets"].append(snippet_with_source)
        
        final_insights = []
        for theme, data in validated_insights_map.items():
            if not data["confidences"]:
                continue

            avg_confidence_from_analyses = sum(data["confidences"]) / len(data["confidences"])
            num_unique_sources = len(data["source_urls"])

            source_boost = 0.0
            if num_unique_sources == 2:
                source_boost = 0.1
            elif num_unique_sources >= 3:
                source_boost = 0.2
            
            avg_keyword_variety = data["keyword_variety_sum"] / len(data["confidences"])
            variety_boost = 0.0
            if avg_keyword_variety >= 2:
                variety_boost = 0.05
            if avg_keyword_variety >=3:
                variety_boost = 0.1
                
            final_confidence = min(avg_confidence_from_analyses + source_boost + variety_boost, 1.0)

            if final_confidence >= self.min_validated_theme_confidence: 
                insight = DestinationInsight(
                    insight_type="theme_validation",
                    insight_name=theme,
                    description=f"{destination} shows characteristics of '{theme}'. Found across {num_unique_sources} sources with {data['mention_counts']} total mentions.",
                    confidence_score=round(final_confidence,2),
                    evidence_sources=list(data["source_urls"])[:3],
                    content_snippets=data["all_snippets"][:3],
                    is_discovered_theme=False,
                    tags=[theme.lower().replace(" ", "-"), "validated", "theme"]
                )
                final_insights.append(insight)
        
        self.logger.info(f"✅ Validated {len(final_insights)} themes for {destination} with confidence >= {self.min_validated_theme_confidence}.")
        return sorted(final_insights, key=lambda x: x.confidence_score, reverse=True)

    async def discover_new_themes_from_content(self, destination: str, sources: List[PageContent], validated_theme_names: List[str]) -> List[DestinationInsight]:
        self.logger.info(f"🔍 Discovering new themes for {destination} from {len(sources)} sources.")
        
        all_content_lower = " ".join([source_obj.content.lower() for source_obj in sources if source_obj.content])
        if not all_content_lower:
            self.logger.info("No content available for new theme discovery.")
            return []

        discovery_patterns = [
            r'famous for its ([\w\s]{2,30}?)\b',
            r'known for ([\w\s]{2,30}?)\b',
            r'renowned for ([\w\s]{2,30}?)\b',
            r'offers unique ([\w\s]{2,30}?)\b',
            r'experience the ([\w\s]{2,30}?)\b',
            r'discover its ([\w\s]{2,30}?)\b',
            r'a hub for ([\w\s]{2,30}?)\b',
            r'highlights include ([\w\s]{2,30}?)\b',
            r'popular ([\w\s]{2,25})\b\s*(?:attractions?|experiences?|activities?|spots?)\b',
            r'(authentic|traditional|local|vibrant|scenic|historic|cultural|artistic|charming|bustling|serene|picturesque)\s+([\w\s]{2,20})\b'
        ]
        
        stop_words = set([
            "a", "an", "the", "is", "are", "was", "were", "be", "been", "being", "have", "has", "had", "do", "does", "did",
            "and", "but", "or", "so", "if", "as", "of", "at", "by", "for", "with", "about", "against", "between", "into",
            "through", "during", "before", "after", "above", "below", "to", "from", "up", "down", "in", "out", "on", "off",
            "over", "under", "again", "further", "then", "once", "here", "there", "when", "where", "why", "how", "all",
            "any", "both", "each", "few", "more", "most", "other", "some", "such", "no", "nor", "not", "only", "own",
            "same", "so", "than", "too", "very", "s", "t", "can", "will", "just", "don", "should", "now", "d", "ll", "m",
            "o", "re", "ve", "y", "ain", "aren", "couldn", "didn", "doesn", "hadn", "hasn", "haven", "isn", "ma", "mightn",
            "mustn", "needn", "shan", "shouldn", "wasn", "weren", "won", "wouldn", "it", "its", "itself", "them", "their",
            "theirs", "themselves", "what", "which", "who", "whom", "this", "that", "these", "those", "am"
        ])

        candidate_themes = {}

        # Can add tqdm here if source iteration is slow, but it's mostly CPU bound string ops after content aggregation.
        # For now, focusing on I/O bound tqdm usage.
        for source_obj in sources: # Iterate over PageContent objects
            content = source_obj.content
            source_url = source_obj.url
            if not content:
                continue
            
            content_lower_for_match = content.lower()
            
            for pattern in discovery_patterns:
                try:
                    matches = re.findall(pattern, content_lower_for_match)
                    for match_group in matches:
                        theme_phrase = " ".join(m.strip() for m in (match_group if isinstance(match_group, tuple) else (match_group,)))
                        theme_phrase = theme_phrase.strip().lower()
                        
                        words = [word for word in theme_phrase.split() if word not in stop_words and len(word) > 2 and word.isalpha()]
                        cleaned_theme = " ".join(words)

                        if not cleaned_theme or len(cleaned_theme.split()) > 4 or len(cleaned_theme.split()) == 0: 
                            continue
                        if cleaned_theme in self.seed_themes or cleaned_theme in validated_theme_names:
                            continue
                        if any(stop_word in cleaned_theme.split() for stop_word in ["destination", "experience", "attraction", "place", "city", "area", "region", "country"]):
                            continue

                        if cleaned_theme not in candidate_themes:
                            candidate_themes[cleaned_theme] = {"count": 0, "snippets": [], "source_urls": set()}
                        
                        candidate_themes[cleaned_theme]["count"] += 1
                        candidate_themes[cleaned_theme]["source_urls"].add(source_url)
                        
                        try:
                            original_match_obj = re.search(re.escape(theme_phrase), content, re.IGNORECASE)
                            if original_match_obj:
                                start_orig, end_orig = original_match_obj.span()
                                snippet_start = max(0, start_orig - 50)
                                snippet_end = min(len(content), end_orig + 50)
                                snippet = f'"{content[snippet_start:snippet_end].strip()}" (Source: {source_url})'
                                if not any(s.startswith(f'"{content[snippet_start:snippet_end][:50]}"') for s in candidate_themes[cleaned_theme]["snippets"]):
                                    candidate_themes[cleaned_theme]["snippets"].append(snippet)
                        except Exception:
                            pass 
                except re.error: 
                    self.logger.warning(f"Regex error with pattern: {pattern}")
                    continue

        discovered_insights = []
        for theme, data in candidate_themes.items():
            if data["count"] < 2 and len(data["source_urls"]) < 2 : 
                continue

            confidence = 0.5 
            if data["count"] >= 3: confidence += 0.1
            if data["count"] >= 5: confidence += 0.1
            if len(data["source_urls"]) >= 2: confidence += 0.1
            if len(data["source_urls"]) >= 3: confidence += 0.1
            
            num_words_in_theme = len(theme.split())
            if num_words_in_theme == 2: confidence += 0.05
            if num_words_in_theme == 3: confidence += 0.05

            confidence = min(round(confidence, 2), 1.0)

            if confidence >= self.min_discovered_theme_confidence:
                 insight = DestinationInsight(
                    insight_type="discovered_theme",
                    insight_name=theme,
                    description=f"Potential new theme '{theme}' discovered for {destination}. Found {data['count']} times across {len(data['source_urls'])} sources.",
                    confidence_score=confidence,
                    evidence_sources=list(data["source_urls"])[:3],
                    content_snippets=data["snippets"][:2],
                    is_discovered_theme=True,
                    tags=[theme.lower().replace(" ", "-"), "discovered", "theme", "new"]
                )
                 discovered_insights.append(insight)
        
        final_discovered = sorted(discovered_insights, key=lambda x: x.confidence_score, reverse=True)[:self.max_discovered_themes_per_destination]
        self.logger.info(f"🆕 Discovered {len(final_discovered)} new themes for {destination} with confidence >= {self.min_discovered_theme_confidence}: {[t.insight_name for t in final_discovered]}")
        return final_discovered 

    async def _get_sentiment(self, text: str) -> Tuple[float, str]:
        if not self.sentiment_analyzer:
            self.logger.warning("Sentiment analyzer not loaded. Returning neutral sentiment.")
            return 0.0, "NEUTRAL"
        try:
            # Ensure text is not excessively long for the model
            max_length = self.sentiment_analyzer.tokenizer.model_max_length
            truncated_text = text[:max_length]
            result = self.sentiment_analyzer(truncated_text)[0]
            score = result['score']
            label = result['label']
            if label == 'NEGATIVE':
                score = -score
            return score, label
        except Exception as e:
            self.logger.error(f"Error during sentiment analysis: {e}", exc_info=True)
            return 0.0, "NEUTRAL"

    async def process_content_for_themes(
        self,
        destination_name: str, 
        text_content_list: List[PageContent], # This is the original_page_content_list
        seed_themes_evidence_map: Optional[Dict[str, List[ChromaSearchResult]]] = None # New input
    ) -> ThemeInsightOutput:
        self.logger.info(f"Processing content for themes for destination: {destination_name}")
        validated_themes: List[DestinationInsight] = []
        discovered_themes_candidates: Dict[str, Dict[str, Any]] = {}

        seed_themes_from_config = self.config.get("content_intelligence", {}).get("seed_themes", self.seed_themes)
        min_validated_confidence = self.config.get("content_intelligence", {}).get("min_validated_theme_confidence", self.min_validated_theme_confidence)
        min_discovered_confidence = self.config.get("content_intelligence", {}).get("min_discovered_theme_confidence", self.min_discovered_theme_confidence)
        max_discovered_themes = self.config.get("content_intelligence", {}).get("max_discovered_themes_per_destination", self.max_discovered_themes_per_destination)

        # --- 1. Validate Seed Themes ---
        self.logger.info(f"Validating {len(seed_themes_from_config)} seed themes...")
        for theme_name in seed_themes_from_config:
            self.logger.debug(f"Validating seed theme: {theme_name}")
            theme_evidence_snippets: List[str] = []
            theme_source_urls: set[str] = set()
            total_mentions = 0
            overall_sentiment_score = 0.0
            content_sources_count = 0

            if seed_themes_evidence_map and theme_name in seed_themes_evidence_map:
                self.logger.info(f"Using ChromaDB evidence for seed theme: {theme_name}")
                chroma_evidences = seed_themes_evidence_map[theme_name]
                for chroma_result in chroma_evidences:
                    chunk = chroma_result.document_chunk
                    # Simple check for theme name in chunk (can be more sophisticated)
                    if theme_name.lower() in chunk.text_chunk.lower(): 
                        total_mentions += 1 # Or count actual occurrences
                        theme_evidence_snippets.append(f"(From Chunk {chunk.chunk_id}): " + chunk.text_chunk[:250] + "...")
                        theme_source_urls.add(chunk.url)
                        sentiment_score, _ = await self._get_sentiment(chunk.text_chunk)
                        overall_sentiment_score += sentiment_score
                        content_sources_count +=1
            else:
                self.logger.info(f"No ChromaDB evidence for '{theme_name}', or map not provided. Searching in original content.")
                for page in text_content_list:
                    if theme_name.lower() in page.content.lower(): # Simple keyword check
                        total_mentions += page.content.lower().count(theme_name.lower())
                        theme_evidence_snippets.append(page.content[:250] + "...") # Example snippet
                        theme_source_urls.add(page.url)
                        sentiment_score, _ = await self._get_sentiment(page.content)
                        overall_sentiment_score += sentiment_score
                        content_sources_count +=1
            
            if total_mentions > 0:
                confidence = min(1.0, total_mentions / 5.0) # Basic confidence
                avg_sentiment = overall_sentiment_score / content_sources_count if content_sources_count > 0 else 0.0
                _, sentiment_label = await self._get_sentiment(str(avg_sentiment)) # Get label from score

                if confidence >= min_validated_confidence:
                    validated_themes.append(DestinationInsight(
                        insight_type="Validated Theme",
                        insight_name=theme_name,
                        description=f"The theme '{theme_name}' was validated with {total_mentions} mentions.",
                        evidence_sources=list(theme_source_urls),
                        content_snippets=theme_evidence_snippets[:3], # Max 3 snippets
                        confidence_score=confidence,
                        is_discovered_theme=False,
                        tags=[theme_name.lower().replace(" ", "-"), "validated", "theme"]
                    ))
                    self.logger.debug(f"Theme '{theme_name}' validated with confidence {confidence:.2f}.")
                else:
                    self.logger.debug(f"Theme '{theme_name}' mentioned but low confidence {confidence:.2f}.")
            else:
                self.logger.debug(f"Seed theme '{theme_name}' not found in provided content.")

        # --- 2. Discover New Themes (Simplified - still uses original_page_content_list) ---
        # This part would ideally use a more sophisticated discovery mechanism if we had an LLM call here
        # or more advanced NLP. For now, it's a placeholder based on keyword counting in broad content.
        self.logger.info("Discovering new themes from original page content list...")
        combined_text_for_discovery = " ".join([page.content for page in text_content_list])
        # Placeholder: very basic "discovery" - count common words/phrases (not robust)
        # In a real scenario, this would involve NLP techniques (e.g., NMF, LDA, LLM summarization/extraction)
        # For now, this part will be very rudimentary and likely not discover much effectively without more work.
        # We'll skip sophisticated discovery for this refactoring pass to focus on Chroma integration.
        self.logger.warning("Rudimentary theme discovery is active. This needs significant improvement for real use.")
        potential_new_themes = ["craft breweries", "mountain biking", "art scene", "local markets", "historic downtown"]
        discovered_theme_insights: List[DestinationInsight] = []

        for potential_theme in potential_new_themes:
            if potential_theme.lower() in combined_text_for_discovery.lower():
                mentions = combined_text_for_discovery.lower().count(potential_theme.lower())
                if mentions > 2: # Arbitrary threshold for discovery
                    confidence = min(1.0, mentions / 10.0) # Different scale for discovery
                    if confidence >= min_discovered_confidence and len(discovered_theme_insights) < max_discovered_themes:
                        # Gather some evidence for this discovered theme
                        disc_evidence_snippets: List[str] = []
                        disc_source_urls: set[str] = set()
                        disc_sentiment_score = 0.0
                        disc_content_sources_count = 0

                        for page in text_content_list:
                            if potential_theme.lower() in page.content.lower():
                                disc_evidence_snippets.append(page.content[:150] + "...")
                                disc_source_urls.add(page.url)
                                sentiment_s, _ = await self._get_sentiment(page.content)
                                disc_sentiment_score += sentiment_s
                                disc_content_sources_count += 1
                        
                        avg_disc_sentiment = disc_sentiment_score / disc_content_sources_count if disc_content_sources_count > 0 else 0.0
                        _, disc_sentiment_label = await self._get_sentiment(str(avg_disc_sentiment))

                        discovered_theme_insights.append(DestinationInsight(
                            insight_type="Discovered Theme",
                            insight_name=potential_theme,
                            description=f"Discovered theme '{potential_theme}' with {mentions} mentions.",
                            evidence_sources=list(disc_source_urls),
                            content_snippets=disc_evidence_snippets[:2],
                            confidence_score=confidence,
                            is_discovered_theme=True,
                            tags=[potential_theme.lower().replace(" ", "-"), "discovered", "theme", "new"]
                        ))
                        self.logger.info(f"Discovered theme: {potential_theme} with confidence {confidence:.2f}")

        # Convert dataclass instances to Pydantic schema format
        pydantic_validated_themes = []
        for theme in validated_themes:
            pydantic_theme = PydanticDestinationInsight(
                destination_name=destination_name,
                insight_type=theme.insight_type,
                insight_name=theme.insight_name,
                description=theme.description,
                evidence=theme.content_snippets,  # Map content_snippets to evidence
                confidence_score=theme.confidence_score,
                source_urls=theme.evidence_sources,
                tags=getattr(theme, 'tags', [])  # Include tags if available
            )
            pydantic_validated_themes.append(pydantic_theme)
        
        pydantic_discovered_themes = []
        for theme in discovered_theme_insights:
            pydantic_theme = PydanticDestinationInsight(
                destination_name=destination_name,
                insight_type=theme.insight_type,
                insight_name=theme.insight_name,
                description=theme.description,
                evidence=theme.content_snippets,  # Map content_snippets to evidence
                confidence_score=theme.confidence_score,
                source_urls=theme.evidence_sources,
                tags=getattr(theme, 'tags', [])  # Include tags if available
            )
            pydantic_discovered_themes.append(pydantic_theme)

        return ThemeInsightOutput(
            destination_name=destination_name,
            validated_themes=pydantic_validated_themes, 
            discovered_themes=pydantic_discovered_themes
        ) 