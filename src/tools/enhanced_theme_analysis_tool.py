from langchain.tools import Tool
from pydantic import BaseModel, Field
from typing import List, Dict, Optional, Any, Set
from datetime import datetime
import logging
import hashlib
import uuid
import re
from collections import Counter
import sys # Add at the top of the file

from ..core.evidence_hierarchy import EvidenceHierarchy, SourceCategory, EvidenceType
from ..core.confidence_scoring import ConfidenceScorer, AuthenticityScorer, UniquenessScorer, ActionabilityScorer, MultiDimensionalScore
from ..core.enhanced_data_models import Evidence, Theme, Destination, TemporalSlice, AuthenticInsight, SeasonalWindow, LocalAuthority
from ..agents.specialized_agents import ValidationAgent, CulturalPerspectiveAgent, ContradictionDetectionAgent
from ..schemas import DestinationInsight, PageContent, PriorityMetrics, InsightType, LocationExclusivity, AuthorityType
from ..tools.priority_aggregation_tool import PriorityAggregationTool
from ..tools.priority_data_extraction_tool import PriorityDataExtractor
from ..core.insight_classifier import InsightClassifier
from ..core.seasonal_intelligence import SeasonalIntelligence
from ..core.source_authority import get_authority_weight # ADDED IMPORT

logger = logging.getLogger(__name__)

class EvidenceRegistry:
    """Registry for deduplicating evidence and using references"""
    
    def __init__(self):
        self.evidence_by_hash = {}  # content_hash -> evidence_data
        self.evidence_by_id = {}    # evidence_id -> evidence_data
        self.hash_to_id = {}        # content_hash -> evidence_id
        
    def _generate_content_hash(self, text_snippet: str, source_url: str) -> str:
        """Generate a content hash to detect duplicate evidence"""
        # Normalize text for comparison
        normalized_text = re.sub(r'\s+', ' ', text_snippet.lower().strip())
        
        # Create hash from normalized text + source URL
        content = f"{normalized_text}|{source_url}"
        return hashlib.md5(content.encode('utf-8')).hexdigest()[:12]
    
    def _calculate_similarity(self, text1: str, text2: str) -> float:
        """Calculate text similarity between two evidence snippets"""
        # Simple word-based similarity
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())
        
        if not words1 or not words2:
            return 0.0
            
        intersection = len(words1.intersection(words2))
        union = len(words1.union(words2))
        
        return intersection / union if union > 0 else 0.0
    
    def add_evidence(self, evidence: Evidence, similarity_threshold: float = 0.85) -> str:
        """
        Add evidence to registry with deduplication.
        Returns evidence ID (existing if duplicate, new if unique)
        """
        content_hash = self._generate_content_hash(evidence.text_snippet, evidence.source_url)
        
        # Check for exact hash match
        if content_hash in self.hash_to_id:
            existing_id = self.hash_to_id[content_hash]
            logger.info(f"Found exact duplicate evidence (hash: {content_hash}), reusing ID: {existing_id}")
            return existing_id
        
        # Check for similar content from same source
        for existing_hash, existing_evidence in self.evidence_by_hash.items():
            if (existing_evidence["source_url"] == evidence.source_url and
                self._calculate_similarity(existing_evidence["text_snippet"], evidence.text_snippet) > similarity_threshold):
                
                existing_id = self.hash_to_id[existing_hash]
                logger.info(f"Found similar evidence (similarity: {self._calculate_similarity(existing_evidence['text_snippet'], evidence.text_snippet):.2f}), reusing ID: {existing_id}")
                return existing_id
        
        # Create new evidence entry
        evidence_id = f"ev_{len(self.evidence_by_id)}"
        
        evidence_data = {
            "id": evidence_id,
            "content_hash": content_hash,
            "source_url": evidence.source_url,
            "source_category": evidence.source_category.value,
            "evidence_type": evidence.evidence_type.value,
            "authority_weight": evidence.authority_weight,
            "text_snippet": evidence.text_snippet,
            "cultural_context": evidence.cultural_context,
            "sentiment": evidence.sentiment,
            "relationships": evidence.relationships,
            "agent_id": evidence.agent_id,
            "published_date": evidence.published_date.isoformat() if evidence.published_date else None,
            "confidence": evidence.confidence,
            "timestamp": evidence.timestamp.isoformat(),
            "factors": getattr(evidence, 'factors', {})
        }
        
        self.evidence_by_hash[content_hash] = evidence_data
        self.evidence_by_id[evidence_id] = evidence_data
        self.hash_to_id[content_hash] = evidence_id
        
        logger.info(f"Added new evidence with ID: {evidence_id}")
        return evidence_id
    
    def get_evidence(self, evidence_id: str) -> Dict[str, Any]:
        """Get evidence data by ID"""
        return self.evidence_by_id.get(evidence_id, {})
    
    def get_all_evidence(self) -> Dict[str, Dict[str, Any]]:
        """Get all evidence in the registry"""
        return self.evidence_by_id.copy()
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get deduplication statistics"""
        return {
            "total_evidence": len(self.evidence_by_id),
            "unique_content_hashes": len(self.evidence_by_hash),
            "deduplication_ratio": 1.0 - (len(self.evidence_by_id) / len(self.hash_to_id)) if self.hash_to_id else 0.0
        }

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
        
        # Initialize evidence registry for deduplication
        self.evidence_registry = EvidenceRegistry()
        
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
        print(f"DEBUG_ETA: ENTERING analyze_themes for {input_data.destination_name}", file=sys.stderr)
        self.logger.info(f"Starting enhanced theme analysis for {input_data.destination_name}")
        
        # Step 1: Extract and classify evidence
        all_evidence = await self._extract_evidence(
            input_data.text_content_list,
            input_data.country_code
        )
        print(f"DEBUG_ETA: STEP 1 (extract_evidence) COMPLETED. Found {len(all_evidence)} evidence pieces.", file=sys.stderr)

        # Step 2: Discover themes from evidence
        discovered_themes = await self._discover_themes(
            all_evidence,
            input_data.destination_name,
            input_data.country_code
        )
        print(f"DEBUG_ETA: STEP 2 (_discover_themes) COMPLETED. Discovered {len(discovered_themes)} raw themes.", file=sys.stderr)

        # Step 3: Cultural perspective analysis
        cultural_result = await self.cultural_agent.execute_task({
            "sources": [
                {
                    "url": ev.source_url,
                    "content": ev.text_snippet
                }
                for ev in all_evidence # Use all_evidence from Step 1
            ],
            "country_code": input_data.country_code,
            "destination_name": input_data.destination_name
        })
        print(f"DEBUG_ETA: STEP 3 (cultural_agent) COMPLETED.", file=sys.stderr)

        # Step 4: Validate themes with confidence scoring
        validation_task_input_themes = []
        for theme in discovered_themes: # discovered_themes are List[Theme]
            theme_input_data = {
                "name": theme.name,
                "macro_category": theme.macro_category, 
                "micro_category": theme.micro_category, 
                "tags": theme.tags, 
                "fit_score": theme.fit_score, 
                "original_evidence_objects": theme.evidence 
            }
            validation_task_input_themes.append(theme_input_data)

        print(f"DEBUG_ETA: PREPARING TO CALL ValidationAgent with {len(validation_task_input_themes)} themes.", file=sys.stderr)
        validation_result = await self.validation_agent.execute_task({
            "destination_name": input_data.destination_name,
            "themes": validation_task_input_themes, 
            "country_code": input_data.country_code
        })
        print(f"DEBUG_ETA: STEP 4 (validation_agent) COMPLETED. Validated count: {validation_result.get('validated_count', 'N/A')}", file=sys.stderr)

        # Step 5: Detect and resolve contradictions
        contradiction_result = await self.contradiction_agent.execute_task({
            "themes": validation_result["validated_themes"],
            "destination_name": input_data.destination_name
        })
        print(f"DEBUG_ETA: STEP 5 (contradiction_agent) COMPLETED. Contradictions found: {contradiction_result.get('contradictions_found', 'N/A')}", file=sys.stderr)

        # Step 6: Build enhanced themes with full metadata
        enhanced_themes = self._build_enhanced_themes(
            contradiction_result["resolved_themes"],
            all_evidence, # Pass all_evidence here for _build_enhanced_themes
            cultural_result
        )
        print(f"DEBUG_ETA: STEP 6 (_build_enhanced_themes) COMPLETED. Built {len(enhanced_themes)} enhanced themes.", file=sys.stderr)

        # Step 7: Create temporal slices if requested
        temporal_slices = []
        if input_data.analyze_temporal:
            temporal_slices = self._analyze_temporal_aspects(enhanced_themes, all_evidence)
        
        # Initialize new components
        insight_classifier = InsightClassifier()

        # New processing steps
        classified_insights = []
        # Step 8: Classify insights by type, calculate multi-dimensional scores, extract seasonal and temporal data
        for theme_data in enhanced_themes:
            # Ensure theme_data is a dictionary 
            if not isinstance(theme_data, dict):
                self.logger.warning(f"Skipping invalid theme_data: {theme_data}")
                continue

            # Get theme description from available fields
            theme_description = None
            if 'description' in theme_data:
                theme_description = theme_data['description']
            elif 'name' in theme_data:
                # Create description from theme name if description is missing
                theme_description = f"{theme_data['name']} experiences in {input_data.destination_name}. {theme_data.get('micro_category', '')} category."
            else:
                self.logger.warning(f"Skipping theme without description or name: {theme_data}")
                continue

            insight_type = insight_classifier.classify_insight_type(theme_description)
            seasonal_window = insight_classifier.extract_seasonal_window(theme_description)
            location_exclusivity = insight_classifier.determine_location_exclusivity(theme_description)
            actionable_details = insight_classifier.extract_actionable_details(theme_description)

            # For simplicity, let's create an AuthenticInsight from the theme
            # In a real scenario, this would involve more detailed parsing and LLM calls
            # to generate authentic insights based on discovered themes and evidence.
            
            # Generate local authorities based on content analysis
            evidence_hierarchy = EvidenceHierarchy()
            local_authorities_for_theme = []
            
            # Analyze theme evidence for local authority indicators
            # Get actual evidence data from evidence references since we use reference-based architecture
            theme_evidence = []
            evidence_refs = theme_data.get("evidence_references", [])
            
            self.logger.info(f"Processing theme '{theme_data.get('name')}' with {len(evidence_refs)} evidence references")
            
            # Get actual evidence data from registry using references
            for evidence_ref in evidence_refs:
                evidence_id = evidence_ref.get("evidence_id")
                if evidence_id:
                    evidence_data = self.evidence_registry.get_evidence(evidence_id)
                    if evidence_data:
                        theme_evidence.append(evidence_data)
                        self.logger.info(f"Retrieved evidence {evidence_id} from registry")
            
            self.logger.info(f"Retrieved {len(theme_evidence)} actual evidence pieces from registry")
            
            for evidence_item in theme_evidence:
                source_url = evidence_item.get("source_url", "")
                text_snippet = evidence_item.get("text_snippet", "")
                
                self.logger.info(f"Checking evidence: URL={source_url[:50]}..., snippet={text_snippet[:100]}...")
                
                # Check for local authority patterns
                local_authority = evidence_hierarchy.classify_local_authority(source_url, text_snippet)
                self.logger.info(f"Classified authority: type={local_authority.authority_type.value}, domain={local_authority.expertise_domain}, validation={local_authority.community_validation}")
                
                if local_authority.authority_type != AuthorityType.RESIDENT:  # If we found a specific authority type
                    # Use the authority data from the classification
                    local_authorities_for_theme.append(local_authority)
                    self.logger.info(f"Added authority: {local_authority.authority_type.value}")
            
            # If no specific authorities found, create a default one based on source quality
            if not local_authorities_for_theme and theme_evidence:
                self.logger.info("No specific authorities found, checking for high authority evidence...")
                highest_authority_evidence = max(theme_evidence, key=lambda x: x.get("authority_weight", 0))
                auth_weight = highest_authority_evidence.get("authority_weight", 0)
                self.logger.info(f"Highest authority weight: {auth_weight}")
                
                if auth_weight > 0.6:
                    local_authority = LocalAuthority(
                        authority_type=AuthorityType.PROFESSIONAL,
                        local_tenure=2,  # Assume moderate tenure for professional sources
                        expertise_domain=theme_data.get('name', ''),
                        community_validation=auth_weight
                    )
                    local_authorities_for_theme.append(local_authority)
                    self.logger.info(f"Added default professional authority with weight {auth_weight}")
            
            self.logger.info(f"Final authorities for theme: {len(local_authorities_for_theme)}")
            
            # Calculate local validation count
            local_validation_count = len([la for la in local_authorities_for_theme if la.community_validation > 0.7])
            
            # Enhanced seasonal intelligence
            seasonal_intelligence = SeasonalIntelligence()
            seasonal_patterns = seasonal_intelligence.extract_seasonal_patterns([theme_description])
            current_relevance = seasonal_intelligence.calculate_current_relevance(seasonal_window) if seasonal_window else 0.5
            
            # Calculate temporal relevance score based on seasonal analysis
            temporal_relevance_score = current_relevance
            
            # Enhanced calculation of authenticity, uniqueness, and actionability scores
            authenticity_scorer = AuthenticityScorer()
            uniqueness_scorer = UniquenessScorer()
            actionability_scorer = ActionabilityScorer()
            
            # Use enhanced implementations with the available data
            authenticity_score = authenticity_scorer.calculate_authenticity(
                [authority for authority in local_authorities_for_theme],  # local authorities
                [],  # evidence list - empty for now
                theme_description  # content
            )
            uniqueness_score = uniqueness_scorer.calculate_uniqueness(
                classified_insights,  # list of insights so far
                theme_description  # content
            )
            actionability_score = actionability_scorer.calculate_actionability(theme_description)

            authentic_insight = AuthenticInsight(
                insight_type=insight_type,  # Pass the enum directly
                authenticity_score=authenticity_score,
                uniqueness_score=uniqueness_score,
                actionability_score=actionability_score,
                temporal_relevance=temporal_relevance_score,
                location_exclusivity=location_exclusivity,  # Pass the enum directly
                seasonal_window=seasonal_window,
                local_validation_count=local_validation_count
            )
            classified_insights.append(authentic_insight)

            # Update the original theme with the new authentic insights and other fields
            if 'authentic_insights' not in theme_data: # Check if the list exists
                theme_data['authentic_insights'] = []
            theme_data['authentic_insights'].append(authentic_insight.to_dict())
            
            # Enhanced seasonal_relevance and regional_uniqueness in theme
            theme_data['seasonal_relevance'] = self._extract_seasonal_relevance(theme_description, seasonal_patterns)
            theme_data['regional_uniqueness'] = uniqueness_score
            theme_data['insider_tips'] = self._extract_insider_tips(theme_description, actionable_details)
            theme_data['local_authorities'] = [la.to_dict() for la in local_authorities_for_theme] # Convert to dict for theme


        # Step 9: Calculate destination dimensions (already exists as Step 8)
        dimensions = self._calculate_dimensions(enhanced_themes, all_evidence)

        # Step 10: Extract seasonal and temporal data (integrated into classification above)
        # Step 11: Validate with local sources (will be done in Sprint 3 with dedicated agent)
        
        # Initialize priority variables (will be enhanced in future sprints)
        priority_metrics = None
        priority_insights = []

        # Update the return result to include authentic insights
        return {
            "destination_name": input_data.destination_name,
            "country_code": input_data.country_code,
            "themes": enhanced_themes, # Now includes evidence references instead of duplicated evidence
            "evidence_registry": self.evidence_registry.get_all_evidence(),  # Single source of all unique evidence
            "temporal_slices": temporal_slices,
            "dimensions": dimensions,
            "evidence_summary": {
                "total_evidence": len(all_evidence),
                "unique_evidence": self.evidence_registry.get_statistics()["total_evidence"],
                "deduplication_stats": self.evidence_registry.get_statistics(),
                "source_distribution": self._get_source_distribution(all_evidence),
                "cultural_metrics": cultural_result["cultural_metrics"]
            },
            "quality_metrics": {
                "themes_discovered": len(discovered_themes),
                "themes_validated": validation_result["validated_count"],
                "contradictions_found": contradiction_result["contradictions_found"],
                "average_confidence": self._calculate_average_confidence(enhanced_themes),
                "evidence_efficiency": self.evidence_registry.get_statistics()["deduplication_ratio"]
            },
            "analysis_timestamp": datetime.now().isoformat(),
            "priority_metrics": priority_metrics,
            "priority_insights": priority_insights,
            "authentic_insights": [ai.to_dict() for ai in classified_insights] # Include all generated authentic insights
        }
    
    async def _extract_evidence(
        self, content_list: List[Dict[str, Any]], country_code: str
    ) -> List[Evidence]:
        """Extract and classify evidence from content with enhanced context awareness"""
        all_evidence = []
        agent_id = f"enhanced_theme_analyzer_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        self.logger.info(f"Extracting evidence from {len(content_list)} content sources")
        
        for idx, content_item in enumerate(content_list):
            url = content_item.get("url", f"unknown_source_{idx}")
            raw_content = content_item.get("content", "")
            title = content_item.get("title", "")
            
            if not raw_content or len(raw_content) < 50:
                self.logger.warning(f"Skipping content item {idx}: insufficient content ({len(raw_content)} chars)")
                continue
            
            # Classify source category and evidence type
            source_category = self._classify_source_category(url, title, raw_content)
            evidence_type = self._classify_evidence_type(raw_content, source_category)
            authority_weight = self._calculate_authority_weight(source_category, url, raw_content)
            
            # Extract published date with better heuristics
            published_date = self._extract_published_date(raw_content, url)
            
            # Smart content chunking
            chunks = self._smart_chunk_content(raw_content)
            
            self.logger.info(f"Processing source {idx}: {url[:80]}... -> {len(chunks)} chunks, category: {source_category.value}")
            
            for chunk_idx, chunk in enumerate(chunks):
                chunk_text = chunk["text"]
                if len(chunk_text) < 100:  # Skip very short chunks
                    continue
                
                # Enhanced sentiment analysis
                sentiment_score = self._analyze_enhanced_sentiment(chunk_text)
                
                # Enhanced cultural context extraction
                cultural_context = self._extract_enhanced_cultural_context(
                    chunk_text, url, country_code, title
                )
                
                # Extract relationships with other content
                relationships = self._extract_content_relationships(
                    chunk_text, url, all_evidence, chunk_idx
                )
                
                # Calculate additional factors for this evidence
                evidence_factors = self._calculate_evidence_factors(
                    chunk_text, url, source_category, authority_weight
                )
                
                evidence = Evidence(
                    id=f"{idx}-{chunk_idx}",
                    source_url=url,
                    source_category=source_category,
                    evidence_type=evidence_type,
                    authority_weight=authority_weight,
                    text_snippet=chunk_text,
                    timestamp=datetime.now(),
                    confidence=authority_weight,
                    sentiment=sentiment_score,
                    cultural_context=cultural_context,
                    relationships=relationships,
                    agent_id=agent_id,
                    published_date=published_date,
                    factors=evidence_factors  # Add the factors field
                )
                
                all_evidence.append(evidence)
        
        self.logger.info(f"Extracted {len(all_evidence)} evidence pieces from {len(content_list)} sources")
        return all_evidence

    def _analyze_enhanced_sentiment(self, text: str) -> float:
        """Enhanced sentiment analysis with destination-specific context"""
        # Basic sentiment indicators
        positive_indicators = [
            "amazing", "beautiful", "wonderful", "fantastic", "incredible", "stunning",
            "breathtaking", "magnificent", "spectacular", "charming", "delightful",
            "must-see", "highly recommend", "loved", "perfect", "excellent"
        ]
        
        negative_indicators = [
            "terrible", "awful", "horrible", "disappointing", "overrated", "crowded",
            "expensive", "tourist trap", "avoid", "waste", "not worth", "dirty",
            "unsafe", "scam", "rude", "poor"
        ]
        
        neutral_indicators = [
            "okay", "average", "decent", "standard", "typical", "normal", "fine"
        ]
        
        text_lower = text.lower()
        
        positive_count = sum(1 for word in positive_indicators if word in text_lower)
        negative_count = sum(1 for word in negative_indicators if word in text_lower)
        neutral_count = sum(1 for word in neutral_indicators if word in text_lower)
        
        total_indicators = positive_count + negative_count + neutral_count
        
        if total_indicators == 0:
            return 0.0  # Neutral if no indicators found
        
        # Calculate weighted sentiment
        sentiment = (positive_count - negative_count) / total_indicators
        
        # Normalize to 0-1 scale (0 = very negative, 0.5 = neutral, 1 = very positive)
        return max(0.0, min(1.0, (sentiment + 1) / 2))

    def _extract_enhanced_cultural_context(
        self, text: str, url: str, country_code: str, title: str = ""
    ) -> Dict[str, Any]:
        """Extract enhanced cultural context with comprehensive analysis"""
        context = {
            "is_local_source": False,
            "local_entities": [],
            "content_type": "general",
            "language_indicators": [],
            "cultural_markers": [],
            "geographic_specificity": 0.0,
            "content_quality_score": 0.0,
            "author_perspective": "unknown",
            "temporal_indicators": []
        }
        
        text_lower = text.lower()
        title_lower = title.lower()
        
        # Determine if source is local
        local_indicators = [
            "local", "native", "born here", "lived here", "from here",
            "our city", "our town", "we locals", "as a local"
        ]
        context["is_local_source"] = any(indicator in text_lower for indicator in local_indicators)
        
        # Extract local entities (simplified - could use NER)
        local_entity_patterns = [
            r'\b[A-Z][a-z]+ [A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b',  # Proper nouns
            r'\b[A-Z][a-z]+\s+(?:Street|Road|Avenue|Boulevard|Plaza|Square|Market|Temple|Museum|Beach|Park)\b'
        ]
        
        import re
        for pattern in local_entity_patterns:
            matches = re.findall(pattern, text)
            context["local_entities"].extend(matches[:5])  # Limit to 5 per pattern
        
        # Classify content type
        if any(word in text_lower for word in ["restaurant", "food", "dining", "meal", "cuisine"]):
            context["content_type"] = "culinary"
        elif any(word in text_lower for word in ["activity", "tour", "experience", "adventure", "visit"]):
            context["content_type"] = "activity"
        elif any(word in text_lower for word in ["transport", "travel", "bus", "train", "flight", "taxi"]):
            context["content_type"] = "transportation"
        elif any(word in text_lower for word in ["culture", "history", "tradition", "heritage", "festival"]):
            context["content_type"] = "cultural"
        elif any(word in text_lower for word in ["price", "cost", "budget", "expensive", "cheap", "money"]):
            context["content_type"] = "pricing"
        elif any(word in text_lower for word in ["safety", "security", "danger", "crime", "safe"]):
            context["content_type"] = "safety"
        
        # Detect language indicators
        non_english_patterns = [
            r'[àáâäçèéêëìíîïñòóôöùúûü]',  # Romance languages
            r'[αβγδεζηθικλμνξοπρστυφχψω]',   # Greek
            r'[абвгдежзийклмнопрстуфхцчшщъыьэюя]',  # Cyrillic
            r'[一-龯]',  # Chinese characters
            r'[ひらがなカタカナ]'  # Japanese
        ]
        
        for pattern in non_english_patterns:
            if re.search(pattern, text, re.IGNORECASE):
                context["language_indicators"].append("non_english_text")
                break
        
        # Cultural markers
        cultural_markers = [
            "traditional", "authentic", "cultural", "heritage", "festival",
            "ceremony", "ritual", "custom", "local way", "indigenous"
        ]
        context["cultural_markers"] = [marker for marker in cultural_markers if marker in text_lower]
        
        # Geographic specificity (0-1 scale)
        geographic_terms = len(context["local_entities"])
        specific_locations = len([e for e in context["local_entities"] if any(loc_type in e.lower() 
                                 for loc_type in ["street", "road", "avenue", "plaza", "square"])])
        context["geographic_specificity"] = min(1.0, (geographic_terms * 0.1) + (specific_locations * 0.2))
        
        # Content quality score based on various factors
        quality_factors = [
            len(text) > 200,  # Substantial content
            len(context["local_entities"]) > 0,  # Specific references
            any(word in text_lower for word in ["because", "since", "due to", "therefore"]),  # Explanatory
            len(text.split('.')) > 3,  # Multiple sentences
            context["is_local_source"],  # Local perspective
            len(context["cultural_markers"]) > 0  # Cultural depth
        ]
        context["content_quality_score"] = sum(quality_factors) / len(quality_factors)
        
        # Author perspective
        if context["is_local_source"]:
            context["author_perspective"] = "local_resident"
        elif any(word in text_lower for word in ["visited", "trip", "travel", "vacation"]):
            context["author_perspective"] = "tourist"
        elif any(word in text_lower for word in ["guide", "recommend", "should", "must"]):
            context["author_perspective"] = "advisor"
        
        # Temporal indicators
        temporal_words = ["seasonal", "summer", "winter", "spring", "fall", "holiday", "weekend", "daily"]
        context["temporal_indicators"] = [word for word in temporal_words if word in text_lower]
        
        return context

    def _extract_content_relationships(
        self, text: str, url: str, existing_evidence: List[Evidence], chunk_idx: int
    ) -> List[Dict[str, str]]:
        """Extract relationships with other content pieces"""
        relationships = []
        
        # Find thematic similarities with existing evidence
        text_lower = text.lower()
        key_terms = self._extract_key_terms(text_lower)
        
        for evidence in existing_evidence[-5:]:  # Check last 5 pieces for efficiency
            evidence_terms = self._extract_key_terms(evidence.text_snippet.lower())
            
            # Calculate similarity
            common_terms = set(key_terms) & set(evidence_terms)
            if len(common_terms) >= 2:  # At least 2 common terms
                similarity_strength = "high" if len(common_terms) >= 4 else "medium"
                relationships.append({
                    "target_id": evidence.id,
                    "relationship_type": "thematic_similarity",
                    "strength": similarity_strength,
                    "common_terms": list(common_terms)[:3]  # Store up to 3 common terms
                })
        
        return relationships

    def _extract_key_terms(self, text: str) -> List[str]:
        """Extract key terms from text for relationship analysis"""
        # Simple keyword extraction - could be enhanced with NLP
        important_words = []
        
        # Travel-specific important terms
        travel_terms = [
            "beach", "mountain", "city", "temple", "museum", "restaurant", "hotel",
            "market", "festival", "culture", "food", "activity", "tour", "experience",
            "shopping", "nightlife", "nature", "historic", "architecture", "art"
        ]
        
        for term in travel_terms:
            if term in text:
                important_words.append(term)
        
        # Extract proper nouns (simplified)
        import re
        proper_nouns = re.findall(r'\b[A-Z][a-z]+\b', text)
        important_words.extend(proper_nouns[:5])  # Limit to 5
        
        return important_words

    def _calculate_evidence_factors(
        self, text: str, url: str, source_category: SourceCategory, authority_weight: float
    ) -> Dict[str, Any]:
        """Calculate additional factors for evidence analysis"""
        factors = {
            "content_length": len(text),
            "sentence_count": len(text.split('.')),
            "readability_score": self._calculate_readability(text),
            "specificity_score": self._calculate_specificity(text),
            "actionability_score": self._calculate_actionability(text),
            "recency_indicators": self._extract_recency_indicators(text),
            "authority_signals": self._extract_authority_signals(text, url)
        }
        
        return factors

    def _calculate_readability(self, text: str) -> float:
        """Simple readability score (0-1, higher = more readable)"""
        words = text.split()
        if not words:
            return 0.0
        
        avg_word_length = sum(len(word) for word in words) / len(words)
        sentence_count = len(text.split('.'))
        avg_sentence_length = len(words) / max(sentence_count, 1)
        
        # Simple readability heuristic
        readability = 1.0 - min(1.0, (avg_word_length - 4) / 10 + (avg_sentence_length - 15) / 20)
        return max(0.0, readability)

    def _calculate_specificity(self, text: str) -> float:
        """Calculate how specific/detailed the content is (0-1)"""
        specificity_indicators = [
            "address", "phone", "hours", "price", "$", "€", "£", "¥",
            "monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday",
            "am", "pm", "open", "closed", "reservation", "booking"
        ]
        
        text_lower = text.lower()
        specific_count = sum(1 for indicator in specificity_indicators if indicator in text_lower)
        
        # Also count numbers as specificity indicators
        import re
        number_count = len(re.findall(r'\d+', text))
        
        total_specificity = specific_count + (number_count * 0.5)
        return min(1.0, total_specificity / 10)

    def _calculate_actionability(self, text: str) -> float:
        """Calculate how actionable the content is (0-1)"""
        actionable_words = [
            "visit", "go", "try", "book", "reserve", "call", "check",
            "avoid", "recommend", "should", "must", "need to", "make sure",
            "remember", "bring", "wear", "take", "use"
        ]
        
        text_lower = text.lower()
        actionable_count = sum(1 for word in actionable_words if word in text_lower)
        
        return min(1.0, actionable_count / 5)

    def _extract_recency_indicators(self, text: str) -> List[str]:
        """Extract indicators of content recency"""
        recency_indicators = []
        
        recent_terms = [
            "recently", "latest", "new", "updated", "current", "now",
            "2024", "2023", "this year", "last month", "recently opened"
        ]
        
        text_lower = text.lower()
        for term in recent_terms:
            if term in text_lower:
                recency_indicators.append(term)
        
        return recency_indicators

    def _extract_authority_signals(self, text: str, url: str) -> List[str]:
        """Extract signals indicating authoritative content"""
        authority_signals = []
        
        # Check URL for authority signals
        if any(domain in url.lower() for domain in [
            "tripadvisor", "lonelyplanet", "fodors", "frommers", "timeout",
            "official", "tourism", "gov", "city", "museum"
        ]):
            authority_signals.append("authoritative_domain")
        
        # Check text for authority signals
        authority_terms = [
            "official", "certified", "licensed", "expert", "professional",
            "years of experience", "local guide", "tourism board", "verified"
        ]
        
        text_lower = text.lower()
        for term in authority_terms:
            if term in text_lower:
                authority_signals.append(f"authority_term_{term.replace(' ', '_')}")
        
        return authority_signals
    
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
            confidence_scorer = ConfidenceScorer()
            theme.confidence_breakdown = confidence_scorer.calculate_confidence(evidence_list)
            
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
        """Build enhanced theme objects with deduplicated evidence references"""
        enhanced_themes = []
        
        # First pass: Register all evidence to enable deduplication
        self.logger.info(f"Registering {len(all_evidence)} evidence pieces for deduplication...")
        for evidence in all_evidence:
            self.evidence_registry.add_evidence(evidence)
        
        # Log deduplication statistics
        stats = self.evidence_registry.get_statistics()
        self.logger.info(f"Evidence deduplication stats: {stats['total_evidence']} unique evidence, "
                        f"deduplication ratio: {stats['deduplication_ratio']:.2%}")
        
        self.logger.info(f"Building enhanced themes: {len(validated_themes)} validated themes")
        
        for idx, theme_data in enumerate(validated_themes):
            theme_name = theme_data["name"]
            self.logger.info(f"Processing theme {idx}: '{theme_name}'")
            
            # Extract confidence breakdown
            confidence_breakdown = theme_data.get("confidence_breakdown", {})
            
            # Build evidence reference list AND actual evidence list for storage compatibility
            theme_evidence_refs = []
            theme_evidence_objects = []  # Add actual Evidence objects for storage
            evidence_matches = 0
            
            for evidence in all_evidence:
                # Match evidence to theme (simplified)
                if self._check_theme_match(theme_name, evidence.text_snippet):
                    evidence_matches += 1
                    
                    # Get evidence ID from registry (deduplicated)
                    evidence_id = self.evidence_registry.add_evidence(evidence)
                    
                    # Store only the reference and minimal metadata for theme context
                    evidence_ref = {
                        "evidence_id": evidence_id,
                        "relevance_score": self._calculate_theme_relevance(theme_name, evidence.text_snippet),
                        "theme_context": evidence.text_snippet[:100] + "..." if len(evidence.text_snippet) > 100 else evidence.text_snippet
                    }
                    
                    theme_evidence_refs.append(evidence_ref)
                    theme_evidence_objects.append(evidence)  # Add actual Evidence object
                    
                    self.logger.info(f"  Evidence match {evidence_matches} for '{theme_name}': {evidence_id}")
            
            self.logger.info(f"  Theme '{theme_name}' references {len(theme_evidence_refs)} evidence pieces")
            
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
            
            # Calculate theme-level factors using evidence references
            theme_factors = self._calculate_theme_factors_from_refs(theme_evidence_refs)
            
            enhanced_theme = {
                "theme_id": hashlib.md5(theme_name.encode()).hexdigest()[:12],
                "name": theme_name,
                "macro_category": macro_category,
                "micro_category": micro_category,
                "confidence_level": theme_data.get("confidence_level", "unknown"),
                "confidence_score": confidence_breakdown.get("overall_confidence", 0.0),
                "confidence_breakdown": confidence_breakdown,
                "factors": theme_factors,
                "evidence_count": len(theme_evidence_refs),
                "evidence_references": theme_evidence_refs,  # Use references instead of full evidence
                "evidence": theme_evidence_objects,  # Add actual Evidence objects for storage compatibility
                "is_validated": theme_data.get("is_validated", False),
                "contradiction_status": {
                    "has_contradictions": theme_data.get("contradiction_resolved", False),
                    "resolution": theme_data.get("winning_position", "none")
                },
                "tags": self._generate_tags(theme_name),
                "cultural_summary": self._generate_cultural_summary_from_refs(theme_evidence_refs),
                "sentiment_analysis": self._analyze_theme_sentiment_from_refs(theme_evidence_refs),
                "temporal_analysis": self._analyze_theme_temporal_aspects_from_refs(theme_evidence_refs)
            }
            
            enhanced_themes.append(enhanced_theme)
            
        return enhanced_themes
    
    def _calculate_theme_relevance(self, theme_name: str, evidence_text: str) -> float:
        """Calculate how relevant evidence is to a specific theme"""
        theme_words = set(theme_name.lower().split())
        evidence_words = set(evidence_text.lower().split())
        
        # Simple word overlap scoring
        overlap = len(theme_words.intersection(evidence_words))
        max_possible = len(theme_words)
        
        if max_possible == 0:
            return 0.0
            
        return min(overlap / max_possible, 1.0)
    
    def _calculate_theme_factors_from_refs(self, evidence_refs: List[Dict]) -> Dict[str, Any]:
        """Calculate theme factors using evidence references"""
        if not evidence_refs:
            return {}
        
        # Get actual evidence data from registry
        evidence_data = []
        for ref in evidence_refs:
            evidence = self.evidence_registry.get_evidence(ref["evidence_id"])
            if evidence:
                evidence_data.append(evidence)
        
        if not evidence_data:
            return {}
        
        factors = {
            "source_diversity": len(set(ev.get("source_url", "") for ev in evidence_data)),
            "authority_distribution": self._calculate_authority_distribution(evidence_data),
            "sentiment_consistency": self._calculate_sentiment_consistency(evidence_data),
            "cultural_breadth": self._calculate_cultural_breadth(evidence_data),
            "temporal_freshness": self._calculate_temporal_freshness(evidence_data),
            "geographic_specificity": self._calculate_geographic_specificity_avg(evidence_data),
            "content_quality_avg": self._calculate_content_quality_avg(evidence_data)
        }
        
        return factors
    
    def _generate_cultural_summary_from_refs(self, evidence_refs: List[Dict]) -> Dict[str, Any]:
        """Generate cultural summary using evidence references"""
        if not evidence_refs:
            return {
                "total_sources": 0,
                "local_sources": 0,
                "international_sources": 0,
                "local_ratio": 0.0,
                "primary_languages": {},
                "cultural_balance": "no-data"
            }
        
        # Get actual evidence data from registry
        evidence_data = []
        for ref in evidence_refs:
            evidence = self.evidence_registry.get_evidence(ref["evidence_id"])
            if evidence:
                evidence_data.append(evidence)
        
        return self._generate_cultural_summary(evidence_data)
    
    def _analyze_theme_sentiment_from_refs(self, evidence_refs: List[Dict]) -> Dict[str, Any]:
        """Analyze theme sentiment using evidence references"""
        # Get actual evidence data from registry
        evidence_data = []
        for ref in evidence_refs:
            evidence = self.evidence_registry.get_evidence(ref["evidence_id"])
            if evidence:
                evidence_data.append(evidence)
        
        return self._analyze_theme_sentiment(evidence_data)
    
    def _analyze_theme_temporal_aspects_from_refs(self, evidence_refs: List[Dict]) -> Dict[str, Any]:
        """Analyze temporal aspects using evidence references"""
        # Get actual evidence data from registry
        evidence_data = []
        for ref in evidence_refs:
            evidence = self.evidence_registry.get_evidence(ref["evidence_id"])
            if evidence:
                evidence_data.append(evidence)
        
        return self._analyze_theme_temporal_aspects(evidence_data)
    
    def _calculate_authority_distribution(self, evidence_list: List[Dict]) -> Dict[str, float]:
        """Calculate distribution of evidence by authority level"""
        if not evidence_list:  # Handle empty evidence list
            return {
                "high_authority_ratio": 0.0,
                "medium_authority_ratio": 0.0,
                "low_authority_ratio": 0.0,
                "authority_score": 0.0
            }
            
        high_authority = sum(1 for ev in evidence_list if ev.get("authority_weight", 0) > 0.8)
        medium_authority = sum(1 for ev in evidence_list if 0.5 < ev.get("authority_weight", 0) <= 0.8)
        low_authority = sum(1 for ev in evidence_list if ev.get("authority_weight", 0) <= 0.5)
        total = len(evidence_list)
        
        return {
            "high_authority_ratio": high_authority / total,
            "medium_authority_ratio": medium_authority / total,
            "low_authority_ratio": low_authority / total,
            "authority_score": sum(ev.get("authority_weight", 0) for ev in evidence_list) / total
        }
    
    def _calculate_sentiment_consistency(self, evidence_list: List[Dict]) -> Dict[str, float]:
        """Calculate sentiment consistency across evidence"""
        sentiments = [ev.get("sentiment", 0) for ev in evidence_list if ev.get("sentiment") is not None]
        
        if not sentiments:
            return {"consistency": 0.0, "average_sentiment": 0.0, "sentiment_range": 0.0, "positive_evidence_ratio": 0.0}
        
        avg_sentiment = sum(sentiments) / len(sentiments)
        sentiment_variance = sum((s - avg_sentiment) ** 2 for s in sentiments) / len(sentiments)
        sentiment_consistency = 1.0 - min(sentiment_variance, 1.0)  # High variance = low consistency
        
        return {
            "consistency": sentiment_consistency,
            "average_sentiment": avg_sentiment,
            "sentiment_range": max(sentiments) - min(sentiments) if sentiments else 0,
            "positive_evidence_ratio": sum(1 for s in sentiments if s > 0.1) / len(sentiments)
        }
    
    def _calculate_cultural_breadth(self, evidence_list: List[Dict]) -> Dict[str, float]:
        """Calculate cultural breadth of evidence sources"""
        if not evidence_list:  # Handle empty evidence list
            return {
                "local_source_ratio": 0.0,
                "language_diversity": 0,
                "cultural_balance_score": 0.0
            }
            
        local_sources = sum(1 for ev in evidence_list 
                          if ev.get("cultural_context", {}).get("is_local_source", False))
        total_sources = len(evidence_list)
        
        # Language diversity
        languages = set()
        for ev in evidence_list:
            lang_indicators = ev.get("cultural_context", {}).get("language_indicators", [])
            languages.update(lang_indicators)
        
        return {
            "local_source_ratio": local_sources / total_sources,
            "language_diversity": len(languages),
            "cultural_balance_score": min(abs(0.6 - (local_sources / total_sources)) + 0.5, 1.0)
        }
    
    def _calculate_temporal_freshness(self, evidence_list: List[Dict]) -> Dict[str, float]:
        """Calculate temporal freshness of evidence"""
        now = datetime.now()
        freshness_scores = []
        
        for ev in evidence_list:
            pub_date_str = ev.get("published_date")
            if pub_date_str:
                try:
                    pub_date = datetime.fromisoformat(pub_date_str.replace('Z', '+00:00'))
                    days_old = (now - pub_date).days
                    # Freshness decreases over time, with 50% at 1 year
                    freshness = max(0, 1.0 - (days_old / 365))
                    freshness_scores.append(freshness)
                except:
                    freshness_scores.append(0.5)  # Default for unparseable dates
            else:
                freshness_scores.append(0.3)  # Default for missing dates
        
        return {
            "average_freshness": sum(freshness_scores) / len(freshness_scores) if freshness_scores else 0,
            "freshest_content": max(freshness_scores) if freshness_scores else 0,
            "oldest_content": min(freshness_scores) if freshness_scores else 0
        }
    
    def _calculate_geographic_specificity_avg(self, evidence_list: List[Dict]) -> float:
        """Calculate average geographic specificity"""
        specificities = []
        for ev in evidence_list:
            geo_spec = ev.get("cultural_context", {}).get("geographic_specificity", 0)
            if isinstance(geo_spec, (int, float)):
                specificities.append(geo_spec)
        
        return sum(specificities) / len(specificities) if specificities else 0
    
    def _calculate_content_quality_avg(self, evidence_list: List[Dict]) -> float:
        """Calculate average content quality"""
        qualities = []
        for ev in evidence_list:
            quality = ev.get("cultural_context", {}).get("content_quality_score", 0)
            if isinstance(quality, (int, float)):
                qualities.append(quality)
        
        return sum(qualities) / len(qualities) if qualities else 0
    
    def _generate_cultural_summary(self, evidence_list: List[Dict]) -> Dict[str, Any]:
        """Generate summary of cultural aspects of the theme"""
        if not evidence_list:  # Handle empty evidence list
            return {
                "total_sources": 0,
                "local_sources": 0,
                "international_sources": 0,
                "local_ratio": 0.0,
                "primary_languages": {},
                "cultural_balance": "no-data"
            }
            
        local_count = sum(1 for ev in evidence_list 
                         if ev.get("cultural_context", {}).get("is_local_source", False))
        
        # Collect language indicators
        all_languages = []
        for ev in evidence_list:
            lang_indicators = ev.get("cultural_context", {}).get("language_indicators", [])
            all_languages.extend(lang_indicators)
        
        from collections import Counter
        language_freq = Counter(all_languages)
        
        return {
            "total_sources": len(evidence_list),
            "local_sources": local_count,
            "international_sources": len(evidence_list) - local_count,
            "local_ratio": local_count / len(evidence_list),
            "primary_languages": dict(language_freq.most_common(3)),
            "cultural_balance": "local-heavy" if local_count / len(evidence_list) > 0.7 else 
                              "international-heavy" if local_count / len(evidence_list) < 0.3 else "balanced"
        }
    
    def _analyze_theme_sentiment(self, evidence_list: List[Dict]) -> Dict[str, Any]:
        """Analyze sentiment patterns across theme evidence"""
        sentiments = [ev.get("sentiment", 0) for ev in evidence_list if ev.get("sentiment") is not None]
        
        if not sentiments:
            return {"overall": "neutral", "confidence": 0.0, "distribution": {}}
        
        avg_sentiment = sum(sentiments) / len(sentiments)
        positive_count = sum(1 for s in sentiments if s > 0.1)
        negative_count = sum(1 for s in sentiments if s < -0.1)
        neutral_count = len(sentiments) - positive_count - negative_count
        
        # Determine overall sentiment
        if avg_sentiment > 0.2:
            overall = "positive"
        elif avg_sentiment < -0.2:
            overall = "negative"
        else:
            overall = "neutral"
        
        return {
            "overall": overall,
            "average_score": avg_sentiment,
            "confidence": abs(avg_sentiment),
            "distribution": {
                "positive": positive_count,
                "negative": negative_count,
                "neutral": neutral_count
            },
            "consistency": 1.0 - (max(sentiments) - min(sentiments)) / 2.0 if len(sentiments) > 1 else 1.0
        }
    
    def _analyze_theme_temporal_aspects(self, evidence_list: List[Dict]) -> Dict[str, Any]:
        """Analyze temporal aspects of theme evidence"""
        # Collect temporal markers from cultural context
        all_temporal_markers = []
        for ev in evidence_list:
            markers = ev.get("cultural_context", {}).get("temporal_markers", [])
            all_temporal_markers.extend(markers)
        
        from collections import Counter
        temporal_freq = Counter(all_temporal_markers)
        
        # Analyze publication dates
        pub_dates = []
        for ev in evidence_list:
            pub_date_str = ev.get("published_date")
            if pub_date_str:
                try:
                    pub_date = datetime.fromisoformat(pub_date_str.replace('Z', '+00:00'))
                    pub_dates.append(pub_date)
                except:
                    pass
        
        # Calculate temporal spread
        temporal_spread = 0
        if len(pub_dates) > 1:
            oldest = min(pub_dates)
            newest = max(pub_dates)
            temporal_spread = (newest - oldest).days
        
        return {
            "temporal_markers": dict(temporal_freq.most_common(5)),
            "evidence_span_days": temporal_spread,
            "newest_evidence": max(pub_dates).isoformat() if pub_dates else None,
            "oldest_evidence": min(pub_dates).isoformat() if pub_dates else None,
            "seasonal_indicators": [marker for marker in temporal_freq.keys() 
                                  if marker in ["summer", "winter", "spring", "fall", "season"]]
        }
    
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
            
        overall_confidence = sum(theme.get("confidence_score", 0.0) for theme in themes)
        return overall_confidence / len(themes)

    def _setup_logger(self):
        """Setup logger for the enhanced theme analysis tool"""
        import logging
        return logging.getLogger(__name__)

    def _extract_seasonal_relevance(self, content: str, seasonal_patterns: List[Dict] = None) -> Dict[str, float]:
        """Extract seasonal relevance scores by month using detected patterns."""
        relevance = {
            "january": 0.0,
            "february": 0.0,
            "march": 0.0,
            "april": 0.0,
            "may": 0.0,
            "june": 0.0,
            "july": 0.0,
            "august": 0.0,
            "september": 0.0,
            "october": 0.0,
            "november": 0.0,
            "december": 0.0
        }
        
        # If we have seasonal patterns, use them
        if seasonal_patterns:
            for pattern in seasonal_patterns:
                pattern_type = pattern.get('pattern_type', '')
                start_month = pattern.get('start_month', 1)
                end_month = pattern.get('end_month', 12)
                confidence = pattern.get('confidence', 0.5)
                
                # Map pattern to months with confidence score
                month_names = ["january", "february", "march", "april", "may", "june",
                             "july", "august", "september", "october", "november", "december"]
                
                # Handle year-wrapping (e.g., Dec-Jan-Feb)
                if start_month <= end_month:
                    months_in_pattern = list(range(start_month, end_month + 1))
                else:
                    months_in_pattern = list(range(start_month, 13)) + list(range(1, end_month + 1))
                
                for month_num in months_in_pattern:
                    month_name = month_names[month_num - 1]
                    relevance[month_name] = max(relevance[month_name], confidence)
        
        # Fallback: Basic text analysis for seasonal keywords
        content_lower = content.lower()
        seasonal_keywords = {
            "winter": ["winter", "snow", "ski", "cold", "christmas", "ice", "skiing", "snowboard"],
            "spring": ["spring", "flower", "bloom", "easter", "cherry blossom", "mild", "warming"],
            "summer": ["summer", "beach", "swimming", "warm", "sunny", "hot", "festival", "vacation"],
            "fall": ["fall", "autumn", "foliage", "harvest", "halloween", "changing leaves", "crisp"]
        }
        
        # Map seasons to months
        season_months = {
            "winter": [12, 1, 2],
            "spring": [3, 4, 5], 
            "summer": [6, 7, 8],
            "fall": [9, 10, 11]
        }
        
        # Check for seasonal keywords
        for season, keywords in seasonal_keywords.items():
            keyword_count = sum(1 for keyword in keywords if keyword in content_lower)
            if keyword_count > 0:
                # Calculate relevance based on keyword frequency
                keyword_relevance = min(keyword_count * 0.2, 1.0)
                
                # Apply to relevant months
                month_names = ["january", "february", "march", "april", "may", "june",
                             "july", "august", "september", "october", "november", "december"]
                for month_num in season_months[season]:
                    month_name = month_names[month_num - 1]
                    relevance[month_name] = max(relevance[month_name], keyword_relevance)
        
        return relevance

    def _extract_insider_tips(self, content: str, actionable_details: List[str] = None) -> List[str]:
        """Extract insider tips from content using actionable details and pattern matching."""
        tips = []
        
        # First, use any actionable details provided
        if actionable_details:
            # Filter actionable details that sound like insider tips
            for detail in actionable_details:
                # Look for tip-like language patterns
                if any(indicator in detail.lower() for indicator in [
                    "tip:", "secret", "insider", "local", "hidden", "avoid", "best time",
                    "pro tip", "locals know", "off the beaten", "lesser known"
                ]):
                    tips.append(detail.strip())
        
        # Pattern-based extraction from content
        content_lower = content.lower()
        
        # Look for explicit tip markers
        tip_patterns = [
            r'insider tip[:\s]([^.]+)',
            r'pro tip[:\s]([^.]+)', 
            r'local tip[:\s]([^.]+)',
            r'secret[:\s]([^.]+)',
            r'hidden gem[:\s]([^.]+)',
            r'locals know[:\s]([^.]+)',
            r'best kept secret[:\s]([^.]+)',
            r'off the beaten path[:\s]([^.]+)'
        ]
        
        import re
        for pattern in tip_patterns:
            matches = re.finditer(pattern, content, re.IGNORECASE)
            for match in matches:
                tip_text = match.group(1).strip()
                if len(tip_text) > 10 and tip_text not in tips:  # Avoid duplicates and short tips
                    tips.append(tip_text)
        
        # Look for advice-like sentences with specific indicators
        sentences = content.split('.')
        for sentence in sentences:
            sentence = sentence.strip()
            if len(sentence) < 20:  # Skip very short sentences
                continue
                
            # Check for advice patterns
            advice_indicators = [
                "make sure to", "don't forget to", "be sure to", "remember to",
                "avoid", "watch out for", "best time to", "ideal time",
                "locals recommend", "locals suggest", "word of advice",
                "you should", "it's worth", "don't miss"
            ]
            
            if any(indicator in sentence.lower() for indicator in advice_indicators):
                # Clean up the sentence and add as tip
                clean_tip = sentence.strip()
                if clean_tip and clean_tip not in tips:
                    tips.append(clean_tip)
        
        # Look for timing and practical advice
        timing_patterns = [
            r'(arrive early|go early|visit early)([^.]+)',
            r'(best time|ideal time|perfect time)([^.]+)',
            r'(avoid crowds|fewer crowds|less crowded)([^.]+)',
            r'(book in advance|reserve ahead|make reservations)([^.]+)'
        ]
        
        for pattern in timing_patterns:
            matches = re.finditer(pattern, content, re.IGNORECASE)
            for match in matches:
                full_match = match.group(0).strip()
                if len(full_match) > 15 and full_match not in tips:
                    tips.append(full_match)
        
        # Limit to most relevant tips and clean them up
        if len(tips) > 5:
            tips = tips[:5]  # Keep top 5 tips
        
        # Clean up tips - remove incomplete sentences and improve formatting
        cleaned_tips = []
        for tip in tips:
            # Remove leading/trailing whitespace and normalize
            tip = tip.strip()
            if len(tip) > 10:  # Keep only substantial tips
                # Capitalize first letter if needed
                if tip and not tip[0].isupper():
                    tip = tip[0].upper() + tip[1:]
                cleaned_tips.append(tip)
        
        return cleaned_tips
    
    def _analyze_sentiment(self, text: str) -> float:
        """Analyze sentiment of text content"""
        # Simple sentiment analysis based on positive/negative words
        positive_words = [
            "amazing", "beautiful", "wonderful", "fantastic", "excellent", "great", "love",
            "perfect", "stunning", "incredible", "awesome", "brilliant", "outstanding",
            "magnificent", "spectacular", "charming", "delightful", "enjoyable", "pleasant",
            "recommend", "must-see", "must-visit", "best", "favorite", "special"
        ]
        
        negative_words = [
            "terrible", "awful", "horrible", "bad", "worst", "hate", "disappointing",
            "overpriced", "crowded", "dirty", "dangerous", "avoid", "waste", "boring",
            "expensive", "poor", "limited", "difficult", "problem", "issue", "warning",
            "closed", "unavailable", "cancelled", "broken", "unsafe"
        ]
        
        text_lower = text.lower()
        positive_count = sum(1 for word in positive_words if word in text_lower)
        negative_count = sum(1 for word in negative_words if word in text_lower)
        
        # Calculate sentiment score between -1 and 1
        total_words = len(text.split())
        if total_words == 0:
            return 0.0
            
        positive_score = positive_count / total_words
        negative_score = negative_count / total_words
        
        # Normalize to -1 to 1 scale
        if positive_score + negative_score == 0:
            return 0.0
        
        sentiment = (positive_score - negative_score) / (positive_score + negative_score + 0.1)
        return max(-1.0, min(1.0, sentiment))
    
    def _extract_published_date(self, content: str, url: str) -> Optional[datetime]:
        """Extract published date from content or URL"""
        import re
        
        # Look for date patterns in content
        date_patterns = [
            r'(\d{4}-\d{2}-\d{2})',  # YYYY-MM-DD
            r'(\d{2}/\d{2}/\d{4})',  # MM/DD/YYYY
            r'(\d{1,2}\s+(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+\d{4})',  # DD Month YYYY
            r'((?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+\d{1,2},?\s+\d{4})'  # Month DD, YYYY
        ]
        
        for pattern in date_patterns:
            matches = re.findall(pattern, content, re.IGNORECASE)
            if matches:
                try:
                    # Try to parse the first match
                    date_str = matches[0]
                    # Simple parsing - in production would use more robust date parsing
                    if '-' in date_str:
                        return datetime.strptime(date_str, '%Y-%m-%d')
                    elif '/' in date_str:
                        return datetime.strptime(date_str, '%m/%d/%Y')
                except:
                    continue
        
        # If no date found, return None
        return None
    
    def _build_evidence_relationships(
        self, chunk_idx: int, chunk_relationships: Dict, content_idx: int, chunks: List
    ) -> List[Dict[str, str]]:
        """Build relationships between evidence pieces"""
        relationships = []
        
        # Add relationships to related chunks in same content
        for related_chunk_idx in chunk_relationships.get(chunk_idx, []):
            relationships.append({
                "target_id": f"{content_idx}-{related_chunk_idx}",
                "relationship_type": "content_sequence",
                "strength": "high"
            })
        
        # Add relationship to source content
        relationships.append({
            "target_id": f"content_{content_idx}",
            "relationship_type": "source_document", 
            "strength": "direct"
        })
        
        # Add thematic relationships based on chunk context
        current_chunk = chunks[chunk_idx]
        current_topics = current_chunk.get("context", {}).get("topics", [])
        
        for other_idx, other_chunk in enumerate(chunks):
            if other_idx != chunk_idx:
                other_topics = other_chunk.get("context", {}).get("topics", [])
                shared_topics = set(current_topics) & set(other_topics)
                if shared_topics:
                    relationships.append({
                        "target_id": f"{content_idx}-{other_idx}",
                        "relationship_type": "thematic_similarity",
                        "strength": "medium",
                        "shared_topics": list(shared_topics)
                    })
        
        return relationships
    
    def _extract_title_from_content(self, text: str) -> str:
        """Extract potential title from content text"""
        lines = text.split('\n')
        # Look for first substantial line that could be a title
        for line in lines[:3]:  # Check first 3 lines
            line = line.strip()
            if len(line) > 10 and len(line) < 100:  # Reasonable title length
                return line
        
        # Fallback: use first sentence
        sentences = text.split('.')
        if sentences:
            first_sentence = sentences[0].strip()
            if len(first_sentence) < 150:  # Not too long for a title
                return first_sentence
        
        return "Content Extract"
    
    def _detect_language_indicators(self, text: str, url: str) -> List[str]:
        """Detect language indicators in text and URL"""
        indicators = []
        
        # URL-based indicators
        url_lower = url.lower()
        if '.fr' in url_lower or '/fr/' in url_lower:
            indicators.append("french")
        if '.es' in url_lower or '/es/' in url_lower:
            indicators.append("spanish")
        if '.de' in url_lower or '/de/' in url_lower:
            indicators.append("german")
        if '.it' in url_lower or '/it/' in url_lower:
            indicators.append("italian")
        
        # Text-based indicators (simple character pattern detection)
        text_sample = text.lower()[:500]  # First 500 chars
        
        # Common non-English words/patterns
        if any(word in text_sample for word in ['le ', 'la ', 'des ', 'une ', 'avec']):
            indicators.append("french_content")
        if any(word in text_sample for word in ['el ', 'la ', 'los ', 'las ', 'con ']):
            indicators.append("spanish_content")
        if any(word in text_sample for word in ['der ', 'die ', 'das ', 'und ', 'mit ']):
            indicators.append("german_content")
        
        # Default to English if no other indicators
        if not indicators:
            indicators.append("english")
        
        return indicators
    
    def _assess_content_quality(self, text: str) -> float:
        """Assess the quality of content based on various factors"""
        score = 0.0
        
        # Length factor (not too short, not too long)
        word_count = len(text.split())
        if 50 <= word_count <= 500:
            score += 0.3
        elif 20 <= word_count < 50 or 500 < word_count <= 1000:
            score += 0.2
        elif word_count > 1000:
            score += 0.1
        
        # Sentence structure (variety in sentence length)
        sentences = text.split('.')
        if len(sentences) > 1:
            sentence_lengths = [len(s.split()) for s in sentences if s.strip()]
            if sentence_lengths:
                avg_length = sum(sentence_lengths) / len(sentence_lengths)
                if 8 <= avg_length <= 20:  # Good average sentence length
                    score += 0.2
        
        # Information density (specific terms, numbers, names)
        specific_indicators = len([
            word for word in text.split()
            if word[0].isupper() or word.isdigit() or '$' in word or '%' in word
        ])
        density = specific_indicators / max(len(text.split()), 1)
        if 0.1 <= density <= 0.3:
            score += 0.3
        
        # Grammar indicators (proper punctuation)
        punctuation_count = sum(1 for char in text if char in '.!?:;')
        if punctuation_count > len(text.split()) * 0.05:  # Reasonable punctuation
            score += 0.2
        
        return min(score, 1.0)
    
    def _assess_geographic_specificity(self, text: str, local_entities: List[str]) -> float:
        """Assess how geographically specific the content is"""
        # Base score from local entities
        entity_score = min(len(local_entities) * 0.2, 0.6)
        
        # Look for geographic terms
        geographic_terms = [
            'located', 'address', 'street', 'avenue', 'road', 'near', 'downtown',
            'neighborhood', 'district', 'area', 'region', 'north', 'south', 'east', 'west',
            'miles', 'kilometers', 'km', 'minutes', 'walk', 'drive', 'direction'
        ]
        
        text_lower = text.lower()
        geo_term_count = sum(1 for term in geographic_terms if term in text_lower)
        geo_score = min(geo_term_count * 0.1, 0.4)
        
        return min(entity_score + geo_score, 1.0)

    def _classify_source_category(self, url: str, title: str, content: str) -> SourceCategory:
        """Classify the source category based on URL, title, and content"""
        url_lower = url.lower()
        title_lower = title.lower()
        content_lower = content.lower()
        
        # Government sources
        if any(domain in url_lower for domain in ['.gov', 'tourism', 'official', 'city', 'state']):
            return SourceCategory.GOVERNMENT
        
        # Academic sources
        if any(domain in url_lower for domain in ['.edu', 'university', 'research', 'study']):
            return SourceCategory.ACADEMIC
        
        # Business sources
        if any(domain in url_lower for domain in ['business', '.biz', 'company', 'corp']):
            return SourceCategory.BUSINESS
        
        # Travel guides
        if any(domain in url_lower for domain in ['lonelyplanet', 'fodors', 'frommers', 'tripadvisor', 'guidebook']):
            return SourceCategory.GUIDEBOOK
        
        # Blogs and social media
        if any(domain in url_lower for domain in ['blog', 'facebook', 'instagram', 'twitter', 'reddit']):
            return SourceCategory.BLOG
        
        # Social media
        if any(domain in url_lower for domain in ['social', 'facebook', 'instagram', 'twitter', 'tiktok']):
            return SourceCategory.SOCIAL
        
        # Default to unknown for unrecognized sources
        return SourceCategory.UNKNOWN

    def _classify_evidence_type(self, content: str, source_category: SourceCategory) -> EvidenceType:
        """Classify the evidence type based on content and source"""
        content_lower = content.lower()
        
        # Primary evidence indicators
        if any(indicator in content_lower for indicator in ['official', 'certified', 'verified', 'authorized']):
            return EvidenceType.PRIMARY
        
        # Secondary evidence indicators
        if source_category in [SourceCategory.GOVERNMENT, SourceCategory.ACADEMIC]:
            return EvidenceType.SECONDARY
        
        if any(indicator in content_lower for indicator in ['research', 'study', 'survey', 'statistics']):
            return EvidenceType.SECONDARY
        
        # Everything else is tertiary
        return EvidenceType.TERTIARY

    def _calculate_authority_weight(self, source_category: SourceCategory, url: str, content: str) -> float:
        """Calculate authority weight for evidence based on URL using the centralized function."""
        # The content and source_category parameters are no longer used by this specific implementation
        # but are kept for compatibility with the calling signature if other tools or future versions
        # might use them or if this method is overridden with different logic.
        return get_authority_weight(url) # MODIFIED LINE

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

class EnhancedAnalyzeThemesFromEvidenceTool:
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
        self.name = "analyze_themes_from_evidence"
        self.description = "Enhanced theme analysis with evidence-based confidence scoring and multi-agent validation"
        
        # Store attributes without Pydantic validation
        self.agent_orchestrator = agent_orchestrator
        self.llm = llm
        
        # Initialize the tool analyzer
        self.theme_analyzer = EnhancedThemeAnalysisTool()
        
        # Import priority aggregation tool
        from .priority_aggregation_tool import PriorityAggregationTool
        self.priority_aggregator = PriorityAggregationTool()
        
        # Store LLM reference for potential use in analysis
        if llm:
            self.theme_analyzer.llm = llm
        
        # If agent orchestrator provided, replace the analyzer's agents
        if agent_orchestrator:
            # Get agents from orchestrator
            for agent_id, agent in agent_orchestrator.broker.agents.items():
                if isinstance(agent, ValidationAgent):
                    self.theme_analyzer.validation_agent = agent
                elif isinstance(agent, CulturalPerspectiveAgent):
                    self.theme_analyzer.cultural_agent = agent
                elif isinstance(agent, ContradictionDetectionAgent):
                    self.theme_analyzer.contradiction_agent = agent
    
    async def _arun(
        self,
        destination_name: str,
        country_code: str = "US",
        text_content_list: Optional[List[Dict[str, Any]]] = None,
        evidence_snippets: Optional[List[Dict[str, Any]]] = None,
        seed_themes_with_evidence: Optional[Dict[str, Any]] = None,
        config=None,
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
                agent_orchestrator=self.agent_orchestrator  # Pass the orchestrator if available
            )
            
            logger.info(f"Created input data with {len(input_data.text_content_list)} content items")
            
            result = await self.theme_analyzer.analyze_themes(input_data)
            
            # Aggregate priority data if we have PageContent objects with priority data
            priority_metrics = None
            priority_insights = []
            
            if page_content_objects and any(hasattr(pc, 'priority_data') for pc in page_content_objects):
                logger.info("Aggregating priority data from content sources")
                try:
                    priority_result = await self.priority_aggregator._arun(
                        destination_name=destination_name,
                        page_contents=page_content_objects,
                        confidence_threshold=0.6
                    )
                    
                    priority_metrics = priority_result.get("priority_metrics")
                    priority_insights = priority_result.get("priority_insights", [])
                    
                    logger.info(f"Aggregated priority data: {len(priority_insights)} priority insights generated")
                    
                except Exception as e:
                    logger.error(f"Error aggregating priority data: {e}")
            
            # Don't convert the enhanced result to backward compatibility - pass it through directly!
            # The storage tool can handle the enhanced format
            logger.info(f"Returning enhanced result directly with {len(result['themes'])} themes and evidence registry")
            
            # Add priority data to the result
            result["priority_metrics"] = priority_metrics
            result["priority_insights"] = priority_insights
            
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
    
    def _run(self, **kwargs):
        """Synchronous wrapper - properly implemented"""
        import asyncio
        try:
            # Try to run in the current event loop if one exists
            loop = asyncio.get_running_loop()
            # If there's a running loop, we need to use a different approach
            import concurrent.futures
            import threading
            
            result = None
            exception = None
            
            def run_in_thread():
                nonlocal result, exception
                try:
                    # Create new event loop in thread
                    new_loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(new_loop)
                    result = new_loop.run_until_complete(self._arun(**kwargs))
                    new_loop.close()
                except Exception as e:
                    exception = e
            
            thread = threading.Thread(target=run_in_thread)
            thread.start()
            thread.join()
            
            if exception:
                raise exception
            return result
            
        except RuntimeError:
            # No event loop running, safe to use asyncio.run()
            return asyncio.run(self._arun(**kwargs))