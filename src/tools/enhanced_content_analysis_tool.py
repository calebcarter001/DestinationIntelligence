import logging
from typing import List, Dict, Any, Type, Optional
from langchain.tools import StructuredTool
from pydantic import BaseModel, Field
from langchain_google_genai import ChatGoogleGenerativeAI

from src.schemas import PageContent, PriorityMetrics, DestinationInsight

logger = logging.getLogger(__name__)

class EnhancedContentAnalysisInput(BaseModel):
    destination_name: str = Field(description="Name of the destination to analyze")
    page_content_list: List[PageContent] = Field(description="List of web page content to analyze")
    analysis_categories: List[str] = Field(
        default=["attractions", "hotels", "restaurants", "activities", "practical_info", "neighborhoods"],
        description="Categories of information to extract"
    )

class StructuredDestinationInsight(BaseModel):
    """Rich, structured insight about a destination"""
    category: str = Field(description="Category (attractions, hotels, restaurants, etc.)")
    name: str = Field(description="Name of the place/thing")
    description: str = Field(description="Detailed description")
    highlights: List[str] = Field(default_factory=list, description="Key highlights or features")
    practical_info: Dict[str, str] = Field(default_factory=dict, description="Practical details (price range, location, hours, etc.)")
    source_evidence: str = Field(description="Direct quote or evidence from source content")
    confidence_score: float = Field(description="Confidence in this insight (0.0-1.0)")

class EnhancedDestinationAnalysis(BaseModel):
    destination_name: str
    attractions: List[StructuredDestinationInsight] = Field(default_factory=list)
    hotels: List[StructuredDestinationInsight] = Field(default_factory=list)
    restaurants: List[StructuredDestinationInsight] = Field(default_factory=list)
    activities: List[StructuredDestinationInsight] = Field(default_factory=list)
    neighborhoods: List[StructuredDestinationInsight] = Field(default_factory=list)
    practical_info: List[StructuredDestinationInsight] = Field(default_factory=list)
    summary: str = Field(default="", description="Rich summary of what makes this destination special")
    priority_metrics: Optional[PriorityMetrics] = Field(default=None, description="Priority metrics for the destination")
    priority_insights: List[DestinationInsight] = Field(default_factory=list, description="Priority-related insights")

class EnhancedContentAnalysisTool(StructuredTool):
    name: str = "enhanced_destination_analysis"
    description: str = (
        "Performs sophisticated analysis of destination content to extract detailed, actionable insights "
        "including specific attractions, hotels, restaurants, activities, and practical travel information."
    )
    args_schema: Type[BaseModel] = EnhancedContentAnalysisInput
    
    def __init__(self, llm: ChatGoogleGenerativeAI, **kwargs):
        super().__init__(**kwargs)
        # Store LLM as a private attribute to avoid Pydantic field conflicts
        self._llm = llm
    
    async def _arun(self, destination_name: str, page_content_list: List[PageContent], analysis_categories: List[str] = None) -> EnhancedDestinationAnalysis:
        logger.info(f"[EnhancedAnalysis] Starting enhanced analysis for {destination_name}")
        
        if not analysis_categories:
            analysis_categories = ["attractions", "hotels", "restaurants", "activities", "practical_info", "neighborhoods"]
        
        # Combine all content for analysis
        combined_content = "\n\n".join([
            f"URL: {page.url}\nTitle: {page.title}\nContent: {page.content[:2000]}..."
            for page in page_content_list
        ])
        
        # Enhanced analysis results
        result = EnhancedDestinationAnalysis(destination_name=destination_name)
        
        for category in analysis_categories:
            insights = await self._analyze_category(destination_name, combined_content, category)
            
            # Assign insights to appropriate category
            if category == "attractions":
                result.attractions = insights
            elif category == "hotels":
                result.hotels = insights
            elif category == "restaurants":
                result.restaurants = insights
            elif category == "activities":
                result.activities = insights
            elif category == "neighborhoods":
                result.neighborhoods = insights
            elif category == "practical_info":
                result.practical_info = insights
        
        # Generate rich summary
        result.summary = await self._generate_destination_summary(destination_name, combined_content)
        
        logger.info(f"[EnhancedAnalysis] Analysis complete for {destination_name}. "
                   f"Found {len(result.attractions)} attractions, {len(result.hotels)} hotels, "
                   f"{len(result.restaurants)} restaurants, {len(result.activities)} activities")
        
        return result
    
    async def _analyze_category(self, destination_name: str, content: str, category: str) -> List[StructuredDestinationInsight]:
        """Extract structured insights for a specific category"""
        
        category_prompts = {
            "attractions": f"""
            Extract specific attractions, landmarks, and points of interest in {destination_name}.
            For each attraction, provide:
            - Name (e.g., "Pike Place Market", "Space Needle")
            - Detailed description of what makes it special
            - Key features and unique aspects
            - Visitor experience details (what to expect)
            - Location and practical info
            
            Focus on finding CONCRETE, NAMED places mentioned in the content.
            """,
            "hotels": f"""
            Extract specific hotels and accommodations in {destination_name}.
            For each hotel, provide:
            - Hotel name (e.g., "Hyatt Regency", "Four Seasons")
            - Star rating or quality level
            - Key amenities and features
            - Target audience (luxury, business, family, etc.)
            - Location and practical details
            
            Focus on finding CONCRETE, NAMED hotels mentioned in the content.
            """,
            "restaurants": f"""
            Extract restaurants, food venues, and dining experiences in {destination_name}.
            For each venue, provide:
            - Restaurant name (e.g., "Le Bernardin", "Joe's Crab Shack")
            - Cuisine type and specialties
            - Signature dishes or menu highlights
            - Atmosphere and dining experience
            - Location and practical info
            
            Focus on finding CONCRETE, NAMED restaurants mentioned in the content.
            """,
            "activities": f"""
            Extract specific activities and experiences in {destination_name}.
            For each activity, provide:
            - Activity name (e.g., "Mount Hood Skiing", "Columbia River Kayaking")
            - Detailed description of the experience
            - What makes it special/unique
            - Target audience and skill level
            - Location and practical details
            
            Focus on finding CONCRETE, NAMED activities mentioned in the content.
            """,
            "neighborhoods": f"""
            Extract neighborhoods, districts, and areas in {destination_name}.
            For each area, provide:
            - Neighborhood name (e.g., "Pearl District", "Downtown")
            - Character and atmosphere
            - Key attractions and features
            - What makes it special
            - Location and practical info
            
            Focus on finding CONCRETE, NAMED neighborhoods mentioned in the content.
            """,
            "practical_info": f"""
            Extract practical travel information for {destination_name}.
            Include details about:
            - Best times to visit (specific months/seasons)
            - Weather patterns and what to expect
            - Transportation options and getting around
            - Cost information and budgeting
            - Essential travel tips
            
            Focus on finding CONCRETE, SPECIFIC details mentioned in the content.
            """
        }
        
        prompt = f"""
        {category_prompts.get(category, f"Extract {category} information for {destination_name}.")}
        
        Content to analyze:
        {content[:3000]}
        
        You MUST return ONLY a valid JSON array of objects. Do not include any explanation, markdown formatting, or other text.
        Each object must have exactly these fields:
        {{
            "name": "specific name here",
            "description": "detailed description here",
            "highlights": ["key feature 1", "key feature 2"],
            "practical_info": {{"price_range": "$$", "location": "downtown"}},
            "source_evidence": "direct quote from content showing this information",
            "confidence_score": 0.85
        }}
        
        Focus on concrete, specific, actionable information. If you cannot find any {category}, return an empty array: []
        
        JSON Response:
        [
            {{
                "name": "Example Place",
                "description": "A detailed description of the place",
                "highlights": ["Key feature 1", "Key feature 2"],
                "practical_info": {{"price_range": "$$", "location": "downtown"}},
                "source_evidence": "Direct quote from content",
                "confidence_score": 0.85
            }}
        ]
        """
        
        try:
            response = await self._llm.ainvoke(prompt)
            response_text = response.content.strip()
            
            # Clean up common JSON formatting issues
            if response_text.startswith("```json"):
                response_text = response_text[7:]
            if response_text.endswith("```"):
                response_text = response_text[:-3]
            if response_text.startswith("```"):
                response_text = response_text[3:]
            
            response_text = response_text.strip()
            
            # If empty or non-JSON response, return empty list
            if not response_text or response_text.lower() in ['none', 'no information found', 'n/a']:
                logger.warning(f"[EnhancedAnalysis] Empty or invalid response for {category}: {response_text[:100]}")
                return []
            
            # Parse LLM response into structured insights
            import json
            try:
                insights_data = json.loads(response_text)
            except json.JSONDecodeError as json_err:
                logger.error(f"[EnhancedAnalysis] JSON decode error for {category}: {json_err}")
                logger.error(f"[EnhancedAnalysis] Raw response: {response_text[:500]}")
                return []
            
            if not isinstance(insights_data, list):
                logger.warning(f"[EnhancedAnalysis] Response is not a list for {category}, attempting to wrap...")
                insights_data = [insights_data] if isinstance(insights_data, dict) else []
            
            insights = []
            for item in insights_data:
                if not isinstance(item, dict):
                    continue
                    
                insight = StructuredDestinationInsight(
                    category=category,
                    name=item.get("name", ""),
                    description=item.get("description", ""),
                    highlights=item.get("highlights", []),
                    practical_info=item.get("practical_info", {}),
                    source_evidence=item.get("source_evidence", ""),
                    confidence_score=item.get("confidence_score", 0.5)
                )
                insights.append(insight)
            
            return insights[:5]  # Limit to top 5 per category
            
        except Exception as e:
            logger.error(f"[EnhancedAnalysis] Error analyzing {category}: {e}")
            logger.error(f"[EnhancedAnalysis] Response content: {response.content[:500] if 'response' in locals() else 'No response'}")
            return []
    
    async def _generate_destination_summary(self, destination_name: str, content: str) -> str:
        """Generate a rich summary of what makes the destination special"""
        
        prompt = f"""
        Based on the following content about {destination_name}, write a compelling 2-3 paragraph summary 
        that captures what makes this destination unique and special. Focus on:
        
        1. The distinctive character and atmosphere
        2. Top reasons people visit
        3. The overall experience visitors can expect
        
        Content:
        {content[:4000]}
        
        Write in an engaging, informative style similar to high-quality travel guides.
        """
        
        try:
            response = await self._llm.ainvoke(prompt)
            return response.content.strip()
        except Exception as e:
            logger.error(f"[EnhancedAnalysis] Error generating summary: {e}")
            return f"A comprehensive analysis of {destination_name} reveals a destination rich in attractions, culture, and experiences."
    
    def _run(self, destination_name: str, page_content_list: List[PageContent], analysis_categories: List[str] = None) -> EnhancedDestinationAnalysis:
        import asyncio
        return asyncio.run(self._arun(destination_name, page_content_list, analysis_categories)) 