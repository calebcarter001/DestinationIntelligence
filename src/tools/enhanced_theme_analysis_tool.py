from langchain.tools import Tool
from pydantic import BaseModel, Field
from typing import List, Dict, Optional, Any, Set
from datetime import datetime
import logging
import hashlib
import uuid

from ..core.evidence_hierarchy import EvidenceHierarchy, SourceCategory
from ..core.confidence_scoring import ConfidenceScorer
from ..core.enhanced_data_models import Evidence, Theme, Destination, TemporalSlice
from ..agents.specialized_agents import ValidationAgent, CulturalPerspectiveAgent, ContradictionDetectionAgent
from ..schemas import DestinationInsight, PageContent, PriorityMetrics
from ..tools.priority_aggregation_tool import PriorityAggregationTool
from ..tools.priority_data_extraction_tool import PriorityDataExtractor

logger = logging.getLogger(__name__)

class EnhancedThemeAnalysisInput(BaseModel):
    """Input for enhanced theme analysis"""
    destination_name: str = Field(description="Name of the destination being analyzed")
    country_code: str = Field(description="ISO 2-letter country code of the destination")
    text_content_list: List[Dict[str, Any]] = Field(description="List of content with URLs and text")
    analyze_temporal: bool = Field(default=True, description="Whether to analyze temporal aspects")
    min_confidence: float = Field(default=0.5, description="Minimum confidence threshold")
    config: Optional[Dict[str, Any]] = Field(default=None, description="Configuration settings for analysis")
    agent_orchestrator: Optional[Any] = Field(default=None, description="Agent orchestrator for validation")

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
            input_data.destination_name,
            input_data.country_code
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
                    "macro_category": theme.macro_category, # Pass through
                    "micro_category": theme.micro_category, # Pass through
                    "tags": theme.tags, # Pass through initial tags
                    "fit_score": theme.fit_score, # Pass through initial fit_score
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
        
        # Aggregate priority data if enabled
        priority_metrics = None
        priority_insights = []
        
        if input_data.config and input_data.config.get("priority_settings", {}).get("enable_priority_discovery"):
            logger.info("Aggregating priority data from content sources")
            aggregator = PriorityAggregationTool()
            
            # Ensure all page contents have priority data
            enhanced_pages = []
            extractor = PriorityDataExtractor()
            
            for page in input_data.text_content_list:
                # Handle both dict and object types
                if isinstance(page, dict):
                    # For dictionary objects
                    if 'priority_data' not in page or not page['priority_data']:
                        page['priority_data'] = extractor.extract_all_priority_data(
                            page.get("content", ""), page.get("url", "")
                        )
                else:
                    # For objects with attributes
                    if not hasattr(page, 'priority_data') or not page.priority_data:
                        page.priority_data = extractor.extract_all_priority_data(
                            getattr(page, "content", ""), getattr(page, "url", "")
                        )
                enhanced_pages.append(page)
            
            # Run aggregation
            agg_result = aggregator._run(
                destination_name=input_data.destination_name,
                page_contents=enhanced_pages
            )
            
            priority_metrics = agg_result.get("priority_metrics")
            priority_insights = agg_result.get("priority_insights", [])
            
            logger.info(f"Aggregated priority data: {len(priority_insights)} priority insights generated")
        
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
            "analysis_timestamp": datetime.now().isoformat(),
            "priority_metrics": priority_metrics,
            "priority_insights": priority_insights
        }
    
    async def _extract_evidence(
        self, content_list: List[Dict[str, Any]], country_code: str
    ) -> List[Evidence]:
        """Extract and classify evidence from content with enhanced context awareness"""
        all_evidence = []
        destination_entities = set()  # Track local entities
        
        self.logger.info(f"Starting enhanced evidence extraction from {len(content_list)} content items")
        
        # First pass: collect destination-specific entities
        for content_item in content_list:
            text = content_item.get("content", "")
            if isinstance(text, str) and len(text.strip()) >= 50:
                # Extract potential local entities (simplified - in production use NER)
                local_entities = self._extract_local_entities(text, country_code)
                destination_entities.update(local_entities)
        
        self.logger.info(f"Identified {len(destination_entities)} local entities")
        
        # Second pass: extract evidence with context
        for idx, content_item in enumerate(content_list):
            url = content_item.get("url", "")
            text = content_item.get("content", "")
            title = content_item.get("title", "")
            
            if not isinstance(text, str) or len(text.strip()) < 50:
                continue
                
            # Smart chunking based on content structure
            chunks = self._smart_chunk_content(text)
            self.logger.info(f"Split content item {idx} into {len(chunks)} smart chunks")
            
            # Track relationships between chunks
            chunk_relationships = self._analyze_chunk_relationships(chunks)
            
            for chunk_idx, chunk_data in enumerate(chunks):
                chunk_text = chunk_data["text"]
                chunk_context = chunk_data.get("context", {})
                
                # Classify source with enhanced metadata
                source_category = EvidenceHierarchy.classify_source(url)
                authority_weight, evidence_type = EvidenceHierarchy.get_source_authority(url)
                
                # Identify local entities in this chunk
                local_entities_in_chunk = [
                    entity for entity in destination_entities
                    if entity.lower() in chunk_text.lower()
                ]
                
                # Enhanced cultural context
                cultural_context = {
                    "source_title": title,
                    "chunk_index": chunk_idx,
                    "is_local_source": EvidenceHierarchy.is_local_source(url, country_code),
                    "local_entities": local_entities_in_chunk,
                    "content_type": chunk_context.get("content_type", "general"),
                    "related_chunks": chunk_relationships.get(chunk_idx, []),
                    "semantic_topics": chunk_context.get("topics", []),
                    "local_relevance_score": len(local_entities_in_chunk) / max(len(chunk_text.split()), 1),
                    "context_relationships": chunk_relationships.get(chunk_idx, []),
                    "content_structure": chunk_context
                }
                
                evidence = Evidence(
                    id=f"{idx}-{chunk_idx}",
                    source_url=url,
                    source_category=source_category,
                    evidence_type=evidence_type,
                    authority_weight=authority_weight,
                    text_snippet=chunk_text,
                    timestamp=datetime.now(),
                    confidence=authority_weight,
                    cultural_context=cultural_context,
                    agent_id="enhanced_theme_analysis",
                    published_date=content_item.get("published_date")
                )
                
                all_evidence.append(evidence)
        
        return all_evidence
    
    def _smart_chunk_content(self, text: str) -> List[Dict[str, Any]]:
        """Smart content chunking that preserves context and structure"""
        chunks = []
        
        # Split on meaningful boundaries (paragraphs, sections)
        raw_sections = text.split('\n\n')
        
        current_chunk = {"text": "", "context": {"content_type": "general"}}
        current_length = 0
        
        for section in raw_sections:
            section = section.strip()
            if not section:
                continue
                
            # Detect section type and context
            section_type = self._detect_section_type(section)
            section_length = len(section.split())
            
            # Start new chunk if current is too large or context changes
            if current_length + section_length > 300 or section_type != current_chunk["context"]["content_type"]:
                if current_chunk["text"]:
                    chunks.append(current_chunk)
                current_chunk = {"text": "", "context": {"content_type": section_type}}
                current_length = 0
            
            # Add section to current chunk
            if current_chunk["text"]:
                current_chunk["text"] += "\n\n"
            current_chunk["text"] += section
            current_length += section_length
            
            # Extract and store additional context
            current_chunk["context"].update(self._extract_section_context(section, section_type))
        
        # Add final chunk
        if current_chunk["text"]:
            chunks.append(current_chunk)
        
        return chunks
    
    def _detect_section_type(self, text: str) -> str:
        """Detect the type of content section"""
        text_lower = text.lower()
        
        # Check for various content types
        if any(marker in text_lower for marker in ["location:", "address:", "directions:", "getting there"]):
            return "location"
        elif any(marker in text_lower for marker in ["hours:", "opening hours", "schedule:", "times:"]):
            return "operational"
        elif any(marker in text_lower for marker in ["$", "price:", "cost:", "fee:", "admission:"]):
            return "pricing"
        elif any(marker in text_lower for marker in ["tip:", "note:", "warning:", "important:"]):
            return "advisory"
        elif any(marker in text_lower for marker in ["history:", "background:", "about:"]):
            return "background"
        elif any(marker in text_lower for marker in ["activity:", "experience:", "tour:", "attraction:"]):
            return "activity"
        
        return "general"
    
    def _extract_section_context(self, text: str, section_type: str) -> Dict[str, Any]:
        """Extract additional context from a section"""
        context = {
            "content_type": section_type,
            "topics": [],
            "entities": [],
            "temporal_indicators": []
        }
        
        # Extract topics (simplified - in production use topic modeling)
        text_lower = text.lower()
        for category, keywords in self.theme_taxonomy.items():
            if any(keyword.lower() in text_lower for keyword in keywords):
                context["topics"].append(category)
        
        # Extract temporal indicators
        temporal_patterns = [
            "summer", "winter", "spring", "fall", "season",
            "morning", "afternoon", "evening",
            "monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday",
            "january", "february", "march", "april", "may", "june",
            "july", "august", "september", "october", "november", "december"
        ]
        context["temporal_indicators"] = [
            pattern for pattern in temporal_patterns
            if pattern in text_lower
        ]
        
        return context
    
    def _analyze_chunk_relationships(self, chunks: List[Dict[str, Any]]) -> Dict[int, List[int]]:
        """Analyze relationships between content chunks"""
        relationships = {}
        
        for i, chunk1 in enumerate(chunks):
            relationships[i] = []
            
            # Find related chunks based on shared topics and entities
            for j, chunk2 in enumerate(chunks):
                if i != j:
                    relationship_score = self._calculate_chunk_relationship(
                        chunk1["context"],
                        chunk2["context"]
                    )
                    if relationship_score > 0.3:  # Threshold for relationship
                        relationships[i].append(j)
        
        return relationships
    
    def _calculate_chunk_relationship(
        self, context1: Dict[str, Any], context2: Dict[str, Any]
    ) -> float:
        """Calculate relationship strength between two chunks"""
        score = 0.0
        
        # Compare topics
        shared_topics = set(context1.get("topics", [])) & set(context2.get("topics", []))
        if shared_topics:
            score += len(shared_topics) * 0.2
        
        # Compare entities
        shared_entities = set(context1.get("entities", [])) & set(context2.get("entities", []))
        if shared_entities:
            score += len(shared_entities) * 0.3
        
        # Compare temporal indicators
        shared_temporal = set(context1.get("temporal_indicators", [])) & set(context2.get("temporal_indicators", []))
        if shared_temporal:
            score += len(shared_temporal) * 0.1
        
        return min(score, 1.0)
    
    def _extract_local_entities(self, text: str, country_code: str) -> Set[str]:
        """Extract potential local entities from text"""
        entities = set()
        
        # Simple pattern matching for potential local entities
        # In production, use NER and location database
        words = text.split()
        for i in range(len(words) - 1):
            word_pair = " ".join(words[i:i+2])
            if (
                any(char.isupper() for char in words[i]) and
                not any(word.lower() in word_pair.lower() for word in ["the", "and", "or", "but"])
            ):
                entities.add(word_pair)
        
        return entities
    
    async def _discover_themes(
        self, evidence_list: List[Evidence], destination_name: str, country_code: str = None
    ) -> List[Theme]:
        """Discover themes from evidence using enhanced analysis"""
        discovered_themes = []
        theme_evidence_map = {}
        local_theme_candidates = set()
        
        # First pass: identify local themes
        for evidence in evidence_list:
            local_entities = evidence.cultural_context.get("local_entities", [])
            if local_entities:
                for entity in local_entities:
                    # Sanitize entity name to avoid issues with pipe character
                    sanitized_entity = entity.replace("|", " ")
                    local_theme_candidates.add(sanitized_entity)
        
        self.logger.info(f"Identified {len(local_theme_candidates)} potential local themes")
        
        # Second pass: map evidence to themes
        for evidence in evidence_list:
            text_lower = evidence.text_snippet.lower()
            content_type = evidence.cultural_context.get("content_type", "general")
            semantic_topics = evidence.cultural_context.get("semantic_topics", [])
            
            # Check generic themes from taxonomy
            for macro_category, micro_categories in self.theme_taxonomy.items():
                for micro_category in micro_categories:
                    if self._check_theme_match(micro_category, text_lower):
                        theme_key = f"{macro_category}|{micro_category}"
                        if theme_key not in theme_evidence_map:
                            theme_evidence_map[theme_key] = {
                                "evidence": [],
                                "local_context": set(),
                                "content_types": set(),
                                "related_themes": set(),
                                "temporal_aspects": set()
                            }
                        theme_map = theme_evidence_map[theme_key]
                        theme_map["evidence"].append(evidence)
                        theme_map["content_types"].add(content_type)
                        theme_map["temporal_aspects"].update(
                            evidence.cultural_context.get("temporal_indicators", [])
                        )
                        
                        for topic in semantic_topics:
                            if topic != macro_category:
                                theme_map["related_themes"].add(topic)
            
            # Check local themes
            for local_theme in local_theme_candidates:
                if local_theme.lower() in text_lower:
                    macro_category = self._categorize_local_theme(
                        local_theme, text_lower, evidence.cultural_context
                    )
                    # Ensure local_theme is sanitized here as well before key creation
                    sanitized_local_theme = local_theme.replace("|", " ") 
                    theme_key = f"{macro_category}|{sanitized_local_theme}"
                    
                    if theme_key not in theme_evidence_map:
                        theme_evidence_map[theme_key] = {
                            "evidence": [],
                            "local_context": set(),
                            "content_types": set(),
                            "related_themes": set(),
                            "temporal_aspects": set()
                        }
                    theme_map = theme_evidence_map[theme_key]
                    theme_map["evidence"].append(evidence)
                    theme_map["local_context"].add(local_theme) # Store original name in context
                    theme_map["content_types"].add(content_type)
                    theme_map["temporal_aspects"].update(
                        evidence.cultural_context.get("temporal_indicators", [])
                    )
        
        # Create enhanced themes with rich context
        for theme_key, theme_data in theme_evidence_map.items():
            evidence_list = theme_data["evidence"]
            evidence_count = len(evidence_list)
            
            if evidence_count < 1:
                continue
            
            # Enhanced confidence scoring
            confidence_components = self._calculate_enhanced_confidence(
                evidence_list,
                theme_data["content_types"],
                theme_data["local_context"],
                theme_data["temporal_aspects"]
            )
            
            macro, micro = theme_key.split("|")
            
            # Combined confidence score with adjusted weights
            confidence_score = (
                confidence_components["evidence_quality"] * 0.35 +      # Increased from 0.3
                confidence_components["source_diversity"] * 0.25 +     # Increased from 0.2
                confidence_components["temporal_coverage"] * 0.15 +       # Reduced from 0.2
                confidence_components["content_completeness"] * 0.25 +       # Increased from 0.2
                confidence_components["local_relevance"] * 0.05                # Additional bonus
            )
            
            # Robustly split theme_key to handle potential multiple pipes in the micro part
            split_parts = theme_key.split("|", 1)
            if len(split_parts) == 2:
                macro, micro = split_parts
            else:
                self.logger.warning(f"Unexpected theme_key format: {theme_key}. Using fallback split.")
                macro = split_parts[0] 
                micro = "|".join(split_parts[1:]) if len(split_parts) > 1 else theme_key

            # Create rich description
            description = self._generate_rich_description(
                micro,
                destination_name,
                theme_data, # Pass the whole theme_data for context
                confidence_components # Pass components for description context
            )
            
            theme = Theme(
                theme_id=hashlib.md5(theme_key.encode()).hexdigest()[:12],
                macro_category=macro,  # This is correctly the macro category
                micro_category=micro,  # This is correctly the micro category
                name=micro,  # The name is the micro category
                description=description,
                fit_score=confidence_components["total_score"], # Use the total_score from components
                evidence=evidence_list, # Use the original evidence_list for this theme
                tags=self._generate_enhanced_tags(
                    micro,
                    theme_data["local_context"],
                    theme_data["related_themes"]
                ),
                created_date=datetime.now(),
                metadata={ # Populate the metadata field correctly
                    "local_context": list(theme_data["local_context"]),
                    "content_types": list(theme_data["content_types"]),
                    "related_themes_from_discovery": list(theme_data["related_themes"]),
                    "temporal_aspects": list(theme_data["temporal_aspects"]),
                    "confidence_components": confidence_components,
                    "raw_evidence_count": evidence_count
                }
            )
            
            # Calculate confidence breakdown using the main ConfidenceScorer
            # Pass the country code if available
            theme.confidence_breakdown = ConfidenceScorer.calculate_confidence(
                evidence_sources=[ev.source_url for ev in evidence_list],
                evidence_texts=[ev.text_snippet for ev in evidence_list],
                published_dates=[ev.published_date for ev in evidence_list],
                destination_country_code=country_code,
                sentiment_scores=[ev.sentiment for ev in evidence_list if ev.sentiment is not None]
            )
            
            discovered_themes.append(theme)
        
        # Post-process themes to identify relationships
        self._enhance_theme_relationships(discovered_themes)
        
        return discovered_themes
        
    def _categorize_local_theme(
        self, local_theme: str, context: str, cultural_context: Dict[str, Any]
    ) -> str:
        """Categorize a local theme into the best fitting macro category"""
        # Check content type first
        content_type = cultural_context.get("content_type", "general")
        if content_type == "activity":
            return "Adventure & Sports"
        elif content_type == "location":
            return "Nature & Outdoor"
        
        # Check context against taxonomy
        best_category = None
        best_score = 0
        
        for macro_category, keywords in self.theme_taxonomy.items():
            score = sum(1 for keyword in keywords if keyword.lower() in context.lower())
            if score > best_score:
                best_score = score
                best_category = macro_category
        
        return best_category or "Other"
        
    def _calculate_enhanced_confidence(
        self,
        evidence_list: List[Evidence],
        content_types: Set[str],
        local_context: Set[str],
        temporal_aspects: Set[str]
    ) -> Dict[str, float]:
        """Calculate enhanced confidence scores with multiple components"""
        components = {
            "evidence_quality": 0.0,
            "source_diversity": 0.0,
            "local_relevance": 0.0,
            "temporal_coverage": 0.0,
            "content_completeness": 0.0,
            "total_score": 0.0
        }
        
        # Evidence quality (authority and recency)
        authority_scores = [ev.authority_weight for ev in evidence_list]
        components["evidence_quality"] = (
            max(authority_scores) * 0.7 +  # Max authority
            sum(authority_scores) / len(authority_scores) * 0.3  # Average authority
        )
        
        # Source diversity
        unique_sources = len(set(ev.source_url for ev in evidence_list))
        components["source_diversity"] = min((unique_sources / 3) * 1.2, 1.0)
        
        # Local relevance
        if local_context:
            components["local_relevance"] = min(len(local_context) * 0.3, 1.0)
        
        # Temporal coverage
        if temporal_aspects:
            season_count = len([
                aspect for aspect in temporal_aspects
                if aspect in ["summer", "winter", "spring", "fall"]
            ])
            components["temporal_coverage"] = min(season_count / 4, 1.0)
        
        # Content completeness
        content_type_scores = {
            "activity": 0.3,
            "location": 0.2,
            "operational": 0.15,
            "pricing": 0.15,
            "background": 0.1,
            "advisory": 0.1
        }
        completeness_score = sum(
            content_type_scores.get(ct, 0.05)
            for ct in content_types
        )
        components["content_completeness"] = min(completeness_score, 1.0)
        
        # Calculate total score with weighted components
        components["total_score"] = (
            components["evidence_quality"] * 0.3 +
            components["source_diversity"] * 0.2 +
            components["local_relevance"] * 0.2 +
            components["temporal_coverage"] * 0.15 +
            components["content_completeness"] * 0.15
        )
        
        return components
        
    def _generate_rich_description(
        self,
        theme_name: str,
        destination_name: str,
        theme_data: Dict[str, Any],
        confidence: Dict[str, float]
    ) -> str:
        """Generate rich theme description with context"""
        parts = [f"{theme_name} experiences and attractions in {destination_name}."]
        
        # Add local context if available
        if theme_data["local_context"]:
            local_attractions = list(theme_data["local_context"])
            if len(local_attractions) == 1:
                parts.append(f"Features {local_attractions[0]}.")
            elif len(local_attractions) > 1:
                attractions_text = ", ".join(local_attractions[:-1]) + f" and {local_attractions[-1]}"
                parts.append(f"Notable attractions include {attractions_text}.")
        
        # Add temporal aspects if available
        seasons = [s for s in theme_data["temporal_aspects"] if s in ["summer", "winter", "spring", "fall"]]
        if seasons:
            seasons_text = ", ".join(seasons[:-1]) + f" and {seasons[-1]}" if len(seasons) > 1 else seasons[0]
            parts.append(f"Best experienced during {seasons_text}.")
        
        # Add confidence context
        confidence_level = "High" if confidence["total_score"] > 0.8 else "Medium" if confidence["total_score"] > 0.5 else "Emerging"
        parts.append(f"Confidence: {confidence_level} ({confidence['total_score']:.2f})")
        
        return " ".join(parts)
        
    def _generate_enhanced_tags(
        self,
        theme_name: str,
        local_context: Set[str],
        related_themes: Set[str]
    ) -> List[str]:
        """Generate enhanced tags including local context"""
        tags = set(self._generate_tags(theme_name))  # Start with base tags
        
        # Add local context as tags
        tags.update(local_context)
        
        # Add related theme keywords
        for related_theme in related_themes:
            theme_words = related_theme.lower().replace("&", "").split()
            tags.update(theme_words)
        
        return list(tags)
        
    def _enhance_theme_relationships(self, themes: List[Theme]) -> None:
        """Enhance themes with relationship information"""
        for theme in themes:
            related_themes_data = []
            
            # Safely get theme_topics, ensuring it's a list of strings
            current_related_list = theme.metadata.get("related_themes_from_discovery", [])
            if not isinstance(current_related_list, list):
                self.logger.warning(f"Theme {theme.name} metadata['related_themes_from_discovery'] is not a list: {type(current_related_list)}. Skipping relationship enhancement for this part.")
                theme_topics = set()
            else:
                theme_topics = set(str(item) for item in current_related_list if isinstance(item, (str, int, float, bool)))
            
            for other_theme in themes:
                if other_theme.theme_id != theme.theme_id:
                    # Safely get other_topics
                    other_related_list = other_theme.metadata.get("related_themes_from_discovery", [])
                    if not isinstance(other_related_list, list):
                        self.logger.warning(f"Other theme {other_theme.name} metadata['related_themes_from_discovery'] is not a list: {type(other_related_list)}. Skipping comparison.")
                        other_topics = set()
                    else:
                        other_topics = set(str(item) for item in other_related_list if isinstance(item, (str, int, float, bool)))
                    
                    shared_topics = theme_topics & other_topics
                    
                    if shared_topics:
                        # Calculate relationship strength (ensure denominator is not zero)
                        denominator = max(len(theme_topics), len(other_topics), 1)
                        relationship_strength = len(shared_topics) / denominator
                        
                        related_themes_data.append({
                            "theme_id": other_theme.theme_id,
                            "name": other_theme.name,
                            "shared_topics": list(shared_topics),
                            "relationship_strength": round(relationship_strength, 3)
                        })
            
            # Store the calculated relationships back into the theme's metadata
            # This key should be distinct if "related_themes_from_discovery" is just raw data
            theme.metadata["calculated_relationships"] = sorted(
                related_themes_data,
                key=lambda x: x["relationship_strength"],
                reverse=True
            )
            # Optionally, clear or rename the original "related_themes_from_discovery" if it's no longer needed in this exact form
            # For now, let's keep it for debugging, but consider if it should be cleaned up or transformed.
    
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
            
            # Get macro and micro categories from theme data or derive them if not present
            macro_category = theme_data.get("macro_category")
            micro_category = theme_data.get("micro_category")
            
            # If categories not present in theme data, try to derive them
            if not macro_category or not micro_category:
                for macro, micros in self.theme_taxonomy.items():
                    for micro in micros:
                        if self._check_theme_match(micro, theme_name.lower()):
                            macro_category = macro
                            micro_category = micro
                            break
                    if macro_category:  # Found a match
                        break
                        
            # Fallback if still not found
            if not macro_category:
                macro_category = self._get_macro_category(theme_name)
            if not micro_category:
                micro_category = theme_name  # Use theme name as micro category
            
            enhanced_theme = {
                "theme_id": hashlib.md5(theme_name.encode()).hexdigest()[:12],
                "name": theme_name,
                "macro_category": macro_category,
                "micro_category": micro_category,  # Now including micro category
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
                "tags": self._generate_tags(theme_name)
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
        temporal_slices = []
        current_season = self._get_current_season()
        
        # Create seasonal slices for the next year
        seasons = ["winter", "spring", "summer", "fall"]
        current_season_idx = seasons.index(current_season)
        
        # Get current date for slice timing
        now = datetime.now()
        current_month = now.month
        current_year = now.year
        
        # Calculate season start dates
        season_months = {
            "winter": [12, 1, 2],
            "spring": [3, 4, 5],
            "summer": [6, 7, 8],
            "fall": [9, 10, 11]
        }
        
        # Create slices for current and next 3 seasons
        for i in range(4):
            season_idx = (current_season_idx + i) % 4
            season = seasons[season_idx]
            
            # Calculate valid dates for the season
            if season_months[season][0] == 12:  # Winter starts in December
                start_month = 12
                start_year = current_year if current_month == 12 else current_year + (i // 4)
            else:
                start_month = season_months[season][0]
                start_year = current_year + ((current_month + (i * 3)) // 12)
            
            valid_from = datetime(start_year, start_month, 1)
            valid_to = None if i == 0 else datetime(
                start_year + (1 if start_month + 2 > 12 else 0),
                (start_month + 2) % 12 + 1,
                1
            )
            
            # Calculate seasonal theme strengths
            theme_strengths = {}
            for theme in themes:
                base_strength = theme["confidence_score"]
                seasonal_modifier = self._calculate_seasonal_modifier(theme["name"], season)
                theme_strengths[theme["name"]] = base_strength * seasonal_modifier
            
            # Extract seasonal highlights
            seasonal_highlights = self._extract_seasonal_highlights(evidence, season)
            
            # Add predicted activities and events
            predicted_activities = self._predict_seasonal_activities(season, themes)
            
            slice_data = {
                "valid_from": valid_from.isoformat(),
                "valid_to": valid_to.isoformat() if valid_to else None,
                "season": season,
                "theme_strengths": theme_strengths,
                "seasonal_highlights": seasonal_highlights,
                "predicted_activities": predicted_activities,
                "is_current": i == 0,
                "confidence": 1.0 - (i * 0.2)  # Decreasing confidence for future predictions
            }
            
            temporal_slices.append(slice_data)
        
        return temporal_slices
    
    def _calculate_seasonal_modifier(self, theme_name: str, season: str) -> float:
        """Calculate how a theme's strength varies by season"""
        seasonal_modifiers = {
            "winter": {
                "skiing": 1.5, "snowboarding": 1.5, "snow": 1.5,
                "winter sports": 1.5, "christmas": 1.5, "ice skating": 1.5,
                "hot springs": 1.3, "spa": 1.2, "indoor": 1.2,
                "beach": 0.6, "swimming": 0.6, "water sports": 0.7
            },
            "spring": {
                "hiking": 1.3, "gardens": 1.4, "flowers": 1.4,
                "outdoor": 1.3, "festivals": 1.3, "nature": 1.3,
                "skiing": 0.7, "beach": 0.8, "winter sports": 0.6
            },
            "summer": {
                "beach": 1.5, "swimming": 1.5, "water": 1.4,
                "outdoor": 1.4, "hiking": 1.3, "camping": 1.4,
                "festivals": 1.3, "nightlife": 1.2,
                "skiing": 0.5, "winter sports": 0.5
            },
            "fall": {
                "foliage": 1.5, "hiking": 1.3, "nature": 1.3,
                "harvest": 1.4, "festivals": 1.2, "outdoor": 1.2,
                "beach": 0.7, "swimming": 0.7, "water sports": 0.8
            }
        }
        
        theme_lower = theme_name.lower()
        modifier = 1.0  # Default no modification
        
        # Check for seasonal modifiers
        for keyword, strength in seasonal_modifiers[season].items():
            if keyword in theme_lower:
                modifier = strength
                break
        
        return modifier
    
    def _predict_seasonal_activities(self, season: str, themes: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Predict likely activities and events for the season"""
        seasonal_activities = {
            "winter": [
                "Skiing and Snowboarding",
                "Winter Festivals and Markets",
                "Hot Springs and Spa Visits",
                "Indoor Cultural Activities",
                "Snow Sports and Recreation"
            ],
            "spring": [
                "Hiking and Nature Walks",
                "Garden and Park Visits",
                "Spring Festivals",
                "Outdoor Sports",
                "Cultural Events"
            ],
            "summer": [
                "Water Sports and Activities",
                "Outdoor Concerts and Events",
                "Beach and Lake Recreation",
                "Summer Festivals",
                "Evening Entertainment"
            ],
            "fall": [
                "Fall Foliage Tours",
                "Harvest Festivals",
                "Hiking and Nature Photography",
                "Wine Tasting",
                "Cultural Events"
            ]
        }
        
        # Match activities to existing themes
        predicted = []
        base_activities = seasonal_activities[season]
        
        for activity in base_activities:
            matching_themes = []
            for theme in themes:
                if any(word.lower() in theme["name"].lower() for word in activity.lower().split()):
                    matching_themes.append(theme["name"])
            
            if matching_themes:
                predicted.append({
                    "activity": activity,
                    "likelihood": "high" if len(matching_themes) > 1 else "medium",
                    "related_themes": matching_themes
                })
            else:
                # Include some activities even without direct theme matches
                predicted.append({
                    "activity": activity,
                    "likelihood": "low",
                    "related_themes": []
                })
        
        return predicted
    
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
    
    def _extract_seasonal_highlights(self, evidence: List[Evidence], season: str) -> Dict[str, Any]:
        """Extract seasonal highlights from evidence"""
        # Simplified - in production would use NLP
        seasonal_keywords = {
            "winter": ["winter", "snow", "ski", "cold", "christmas"],
            "spring": ["spring", "flower", "bloom", "easter"],
            "summer": ["summer", "beach", "swimming", "warm", "sunny"],
            "fall": ["fall", "autumn", "foliage", "harvest"]
        }
        
        highlights = {}
        for keyword in seasonal_keywords[season]:
            mentions = 0
            for ev in evidence:
                text_lower = ev.text_snippet.lower()
                mentions += sum(1 for kw in keyword.split() if kw in text_lower)
            if mentions > 0:
                highlights[keyword] = {"mention_count": mentions}
                
        return highlights
    
    def _calculate_dimensions(
        self, themes: List[Dict[str, Any]], evidence: List[Evidence]
    ) -> Dict[str, Any]:
        """Calculate destination dimensions based on themes and evidence"""
        dimensions = {}
        
        # Map themes to dimensions (expanded)
        theme_dimension_mapping = {
            # Nature & Outdoor dimensions
            "hiking": {"outdoor_activity_score": 0.9, "nature_accessibility": 0.9, "trail_quality": 0.8},
            "mountains": {"scenic_beauty": 0.9, "outdoor_activity_score": 0.9, "adventure_potential": 0.8},
            "parks": {"nature_accessibility": 0.9, "family_friendly": 0.8, "outdoor_spaces": 0.9},
            "nature": {"natural_beauty": 0.9, "wildlife_viewing": 0.8, "eco_tourism": 0.8},
            "wildlife": {"wildlife_diversity": 0.9, "nature_experience": 0.8, "photo_opportunities": 0.8},
            "camping": {"outdoor_accommodation": 0.8, "adventure_potential": 0.7, "nature_immersion": 0.9},
            "rivers": {"water_activities": 0.9, "scenic_beauty": 0.8, "outdoor_recreation": 0.8},
            "lakes": {"water_activities": 0.9, "scenic_beauty": 0.8, "outdoor_recreation": 0.8},
            
            # Cultural & Arts dimensions
            "museums": {"cultural_richness": 0.9, "educational_value": 0.8, "rainy_day_activities": 0.7},
            "galleries": {"arts_scene": 0.9, "cultural_richness": 0.8, "creative_atmosphere": 0.9},
            "historic": {"historical_significance": 0.9, "cultural_preservation": 0.8, "educational_value": 0.8},
            "architecture": {"architectural_interest": 0.9, "photo_opportunities": 0.8, "cultural_richness": 0.8},
            "festivals": {"cultural_events": 0.9, "entertainment_options": 0.8, "local_experience": 0.9},
            
            # Food & Dining dimensions
            "restaurants": {"dining_variety": 0.9, "culinary_quality": 0.8, "food_scene": 0.9},
            "breweries": {"nightlife_options": 0.8, "local_craft": 0.9, "social_atmosphere": 0.8},
            "cafes": {"coffee_culture": 0.9, "social_spaces": 0.8, "local_experience": 0.7},
            "markets": {"local_food_access": 0.9, "cultural_immersion": 0.8, "shopping_variety": 0.8},
            
            # Entertainment dimensions
            "nightlife": {"evening_entertainment": 0.9, "social_scene": 0.8, "urban_vibrancy": 0.8},
            "bars": {"nightlife_options": 0.9, "social_atmosphere": 0.8, "entertainment_variety": 0.7},
            "music": {"entertainment_options": 0.9, "cultural_events": 0.8, "nightlife_quality": 0.8},
            "theater": {"cultural_entertainment": 0.9, "evening_activities": 0.8, "arts_scene": 0.8},
            
            # Shopping & Local Craft
            "shopping": {"retail_variety": 0.9, "shopping_experience": 0.8, "urban_amenities": 0.7},
            "boutiques": {"shopping_quality": 0.9, "local_craft": 0.8, "unique_offerings": 0.9},
            "craft": {"local_artisans": 0.9, "cultural_products": 0.8, "shopping_authenticity": 0.9},
            
            # Family & Education
            "family": {"family_friendly": 0.9, "kid_activities": 0.9, "safety": 0.8},
            "educational": {"learning_opportunities": 0.9, "family_friendly": 0.8, "cultural_education": 0.8},
            "science": {"educational_value": 0.9, "family_activities": 0.8, "rainy_day_options": 0.8},
            
            # Accommodation
            "hotels": {"accommodation_variety": 0.9, "lodging_quality": 0.8, "tourist_infrastructure": 0.8},
            "resorts": {"luxury_options": 0.9, "amenities": 0.9, "service_quality": 0.8},
            "unique": {"unique_experiences": 0.9, "memorable_stays": 0.8, "local_character": 0.9},
            
            # Health & Wellness
            "spas": {"wellness_facilities": 0.9, "relaxation_options": 0.9, "luxury_services": 0.8},
            "yoga": {"wellness_activities": 0.9, "spiritual_atmosphere": 0.8, "health_focus": 0.8},
            "meditation": {"spiritual_atmosphere": 0.9, "wellness_options": 0.8, "peaceful_settings": 0.9}
        }
        
        # Calculate dimension values based on theme presence and confidence
        for theme in themes:
            theme_name_lower = theme["name"].lower()
            confidence = theme["confidence_score"]
            
            # Check for exact matches first
            matched = False
            for keyword, dim_values in theme_dimension_mapping.items():
                if keyword in theme_name_lower:
                    matched = True
                    for dim_name, base_value in dim_values.items():
                        if dim_name not in dimensions:
                            dimensions[dim_name] = {
                                "value": base_value * confidence,
                                "confidence": confidence,
                                "evidence_themes": [theme["name"]]
                            }
                        else:
                            # Use max value instead of average for stronger dimensions
                            current = dimensions[dim_name]
                            new_value = max(current["value"], base_value * confidence)
                            new_confidence = max(current["confidence"], confidence)
                            current["value"] = new_value
                            current["confidence"] = new_confidence
                            if theme["name"] not in current["evidence_themes"]:
                                current["evidence_themes"].append(theme["name"])
            
            # If no exact match, try to find related dimensions
            if not matched:
                words = theme_name_lower.split()
                for word in words:
                    if len(word) > 3:  # Only check significant words
                        for keyword, dim_values in theme_dimension_mapping.items():
                            if word in keyword or keyword in word:
                                for dim_name, base_value in dim_values.items():
                                    if dim_name not in dimensions:
                                        dimensions[dim_name] = {
                                            "value": base_value * confidence * 0.8,  # Slightly lower confidence for partial matches
                                            "confidence": confidence * 0.8,
                                            "evidence_themes": [theme["name"]]
                                        }
                                    else:
                                        current = dimensions[dim_name]
                                        new_value = max(current["value"], base_value * confidence * 0.8)
                                        new_confidence = max(current["confidence"], confidence * 0.8)
                                        current["value"] = new_value
                                        current["confidence"] = new_confidence
                                        if theme["name"] not in current["evidence_themes"]:
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
            
            # Import priority aggregation tool
            from .priority_aggregation_tool import PriorityAggregationTool
            priority_aggregator = PriorityAggregationTool()
            
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
                    page_content_objects = []  # Keep original PageContent objects for priority analysis
                    
                    logger.info(f"Converting {len(text_content_list) if text_content_list else 0} content items")
                    
                    for idx, item in enumerate(text_content_list or []):
                        logger.info(f"Processing item {idx}: type={type(item)}")
                        
                        if hasattr(item, 'url') and hasattr(item, 'content'):
                            # It's a PageContent object
                            page_content_objects.append(item)  # Keep original for priority analysis
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
                    
                    # Load config from app
                    from src.config_loader import load_app_config
                    app_config = load_app_config()
                    
                    # Run enhanced analysis
                    input_data = EnhancedThemeAnalysisInput(
                        destination_name=destination_name,
                        country_code=country_code,
                        text_content_list=formatted_content_list,
                        analyze_temporal=kwargs.get("analyze_temporal", True),
                        min_confidence=kwargs.get("min_confidence", 0.5),
                        config=app_config,  # Pass the app config
                        agent_orchestrator=orchestrator  # Pass the orchestrator if available
                    )
                    
                    logger.info(f"Created input data with {len(input_data.text_content_list)} content items")
                    
                    result = await theme_analyzer.analyze_themes(input_data)
                    
                    # Aggregate priority data if we have PageContent objects with priority data
                    priority_metrics = None
                    priority_insights = []
                    
                    if page_content_objects and any(hasattr(pc, 'priority_data') for pc in page_content_objects):
                        logger.info("Aggregating priority data from content sources")
                        try:
                            priority_result = await priority_aggregator._arun(
                                destination_name=destination_name,
                                page_contents=page_content_objects,
                                confidence_threshold=0.6
                            )
                            
                            priority_metrics = priority_result.get("priority_metrics")
                            priority_insights = priority_result.get("priority_insights", [])
                            
                            logger.info(f"Aggregated priority data: {len(priority_insights)} priority insights generated")
                            
                        except Exception as e:
                            logger.error(f"Error aggregating priority data: {e}")
                    
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
                            
                            # Generate tags for the theme
                            theme_tags = theme_analyzer._generate_tags(theme.get("name", ""))
                            
                            theme_insight = DestinationInsight(
                                destination_name=destination_name,
                                insight_type=theme.get("macro_category", "Other"),
                                insight_name=theme.get("name", "Unknown"),
                                description=description,
                                confidence_score=theme.get("confidence_score", 0.0),
                                evidence=evidence_string_list,  # Use string list for schema compatibility
                                source_urls=[ev.get("source_url", "") for ev in theme.get("evidence_summary", [])],
                                tags=theme_tags  # Now supported in schema
                            )
                            
                            # Store enhanced metadata in the evidence field for later processing (instead of as attributes)
                            # Enhanced data is preserved in evidence_details list and description
                            
                            if theme.get("is_validated", False):
                                validated_themes.append(theme_insight)
                            else:
                                discovered_themes.append(theme_insight)
                        
                        # Return backward compatible format with priority data
                        output = ThemeInsightOutput(
                            destination_name=destination_name,
                            validated_themes=validated_themes,
                            discovered_themes=discovered_themes,
                            priority_insights=priority_insights,
                            priority_metrics=priority_metrics
                        )
                        
                        # Note: Raw themes data is available in the result dict but cannot be attached
                        # to Pydantic model as dynamic field. Use the validated/discovered themes instead.
                        
                        return output
                    
                    # Log summary
                    logger.info(
                        f"Enhanced analysis complete for {destination_name}: "
                        f"{result.get('quality_metrics', {}).get('themes_validated', 0)} themes validated, "
                        f"{len(priority_insights)} priority insights, "
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