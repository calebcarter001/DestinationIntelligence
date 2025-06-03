#!/usr/bin/env python3

import asyncio
import sys
import os
from datetime import datetime
import pytest
import sqlite3
import json
import logging
from typing import Dict, Any

# Add project root to path
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.tools.enhanced_theme_analysis_tool import EnhancedThemeAnalysisTool

@pytest.mark.asyncio
async def test_enhanced_fields():
    """Test to see why enhanced fields are not being populated"""
    
    # Create analyzer
    analyzer = EnhancedThemeAnalysisTool()
    
    # Test sample content
    sample_content = [{
        "url": "https://example.com/bali-guide",
        "content": "Bali offers amazing beaches and beautiful sunsets. The local culture is vibrant and welcoming. Many visitors enjoy the affordable cost of living and tropical weather. Swimming in the crystal clear waters is a popular activity during summer months.",
        "title": "Bali Travel Guide"
    }]
    
    print("🔍 Testing Enhanced Analysis Components...")
    
    # Test sentiment analysis
    text = sample_content[0]["content"]
    sentiment = analyzer._analyze_sentiment(text)
    print(f"1. Sentiment Analysis: {sentiment}")
    
    # Test published date extraction
    published_date = analyzer._extract_published_date(text, sample_content[0]["url"])
    print(f"2. Published Date: {published_date}")
    
    # Test language indicators
    language_indicators = analyzer._detect_language_indicators(text, sample_content[0]["url"])
    print(f"3. Language Indicators: {language_indicators}")
    
    # Test content quality
    content_quality = analyzer._assess_content_quality(text)
    print(f"4. Content Quality: {content_quality}")
    
    # Test local entities
    local_entities = analyzer._extract_local_entities(text, "ID")
    print(f"5. Local Entities: {list(local_entities)}")
    
    print("\n📊 Testing Evidence Extraction...")
    evidence_list = await analyzer._extract_evidence(sample_content, "ID")
    
    if evidence_list:
        ev = evidence_list[0]
        print(f"Evidence ID: {ev.id}")
        print(f"Sentiment: {ev.sentiment}")
        print(f"Agent ID: {ev.agent_id}")
        print(f"Published Date: {ev.published_date}")
        print(f"Cultural Context keys: {list(ev.cultural_context.keys()) if ev.cultural_context else 'None'}")
        print(f"Relationships count: {len(ev.relationships)}")
        
        # Test evidence serialization
        print("\n🔧 Testing Evidence Serialization...")
        evidence_dict = ev.to_dict()
        print(f"Serialized sentiment: {evidence_dict.get('sentiment')}")
        print(f"Serialized agent_id: {evidence_dict.get('agent_id')}")
        print(f"Serialized cultural_context: {bool(evidence_dict.get('cultural_context'))}")
    else:
        print("❌ No evidence extracted!")

if __name__ == "__main__":
    asyncio.run(test_enhanced_fields()) 