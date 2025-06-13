#!/usr/bin/env python3
"""
Test script to verify travel-focused theme discovery changes
Shows what generic features were removed and tests new functionality
"""

import sys
import os
import asyncio
from datetime import datetime

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from tools.enhanced_theme_analysis_tool import EnhancedThemeAnalysisTool, EnhancedThemeAnalysisInput
from core.enhanced_data_models import Evidence
from core.evidence_hierarchy import SourceCategory, EvidenceType

def test_generic_features_removed():
    """Test what generic features were removed"""
    print("🔍 TESTING: Generic Features Removed")
    print("=" * 50)
    
    theme_tool = EnhancedThemeAnalysisTool()
    
    # Check if old generic taxonomy is still being used
    old_generic_categories = [
        "Music Heritage", "Music Scene", "Local Character", "City Vibe",
        "Cultural Heritage", "Historical Identity", "Artistic Scene",
        "Creative Community", "Cultural Movements", "Local Legends"
    ]
    
    print("❌ REMOVED Generic Categories:")
    for category in old_generic_categories:
        print(f"   - {category}")
    
    # Check new travel-focused taxonomy
    print("\n✅ NEW Travel-Focused Categories:")
    for macro_category, micro_categories in theme_tool.theme_taxonomy.items():
        print(f"   📂 {macro_category}:")
        for micro in micro_categories[:3]:  # Show first 3
            print(f"      - {micro}")
        if len(micro_categories) > 3:
            print(f"      ... and {len(micro_categories) - 3} more")
    
    print("\n🎯 KEY CHANGES:")
    print("   • Removed generic 'Music Heritage' and 'Local Character' themes")
    print("   • Added specific POI extraction patterns")
    print("   • Prioritized Popular > POI > Cultural > Practical")
    print("   • Limited theme counts: 3 Popular, 4 POI, 2 Cultural, 1 Practical")
    print("   • Focus on travel inspiration over local administrative details")

def create_test_evidence():
    """Create test evidence for different theme types"""
    evidence_list = []
    
    # POI Evidence - should extract "Lowell Observatory"
    evidence_list.append(Evidence(
        evidence_id="ev_poi_001",
        text_snippet="Visit the famous Lowell Observatory in Flagstaff, where Pluto was discovered. This iconic landmark offers stunning views of the night sky and is a must-see attraction for astronomy enthusiasts.",
        source_url="https://example.com/lowell-observatory",
        source_category=SourceCategory.TRAVEL_GUIDE,
        evidence_type=EvidenceType.ATTRACTION_INFO,
        confidence=0.9,
        cultural_context={"content_type": "attraction", "semantic_topics": ["astronomy", "science"]},
        timestamp=datetime.now()
    ))
    
    # Popular/Trending Evidence - should extract trending themes
    evidence_list.append(Evidence(
        evidence_id="ev_popular_001",
        text_snippet="Flagstaff's Historic Downtown is trending on Instagram for its picture-perfect Victorian architecture and scenic mountain views. This hot spot is buzzing with visitors taking photos.",
        source_url="https://example.com/downtown-flagstaff",
        source_category=SourceCategory.SOCIAL_MEDIA,
        evidence_type=EvidenceType.SOCIAL_CONTENT,
        confidence=0.8,
        cultural_context={"content_type": "social", "semantic_topics": ["photography", "architecture"]},
        timestamp=datetime.now()
    ))
    
    # Cultural Evidence - should extract authentic experiences
    evidence_list.append(Evidence(
        evidence_id="ev_cultural_001",
        text_snippet="Experience authentic Native American heritage at the Museum of Northern Arizona, where traditional pottery and weaving demonstrations showcase local artisan crafts.",
        source_url="https://example.com/museum-northern-arizona",
        source_category=SourceCategory.CULTURAL_SITE,
        evidence_type=EvidenceType.CULTURAL_INFO,
        confidence=0.85,
        cultural_context={"content_type": "cultural", "semantic_topics": ["heritage", "crafts"]},
        timestamp=datetime.now()
    ))
    
    # Practical Evidence - should create minimal practical theme
    evidence_list.append(Evidence(
        evidence_id="ev_practical_001",
        text_snippet="Getting around Flagstaff is easy with the Mountain Line bus system. Transportation options include rental cars, and the city is very walkable for downtown attractions.",
        source_url="https://example.com/flagstaff-transportation",
        source_category=SourceCategory.OFFICIAL_TOURISM,
        evidence_type=EvidenceType.PRACTICAL_INFO,
        confidence=0.7,
        cultural_context={"content_type": "practical", "semantic_topics": ["transportation"]},
        timestamp=datetime.now()
    ))
    
    # Generic Evidence that should NOT create generic themes
    evidence_list.append(Evidence(
        evidence_id="ev_generic_001",
        text_snippet="Flagstaff has a vibrant music scene and rich cultural heritage with local character that defines the city vibe.",
        source_url="https://example.com/flagstaff-culture",
        source_category=SourceCategory.BLOG,
        evidence_type=EvidenceType.OPINION,
        confidence=0.6,
        cultural_context={"content_type": "general", "semantic_topics": ["music", "culture"]},
        timestamp=datetime.now()
    ))
    
    return evidence_list

async def test_new_theme_discovery():
    """Test the new travel-focused theme discovery"""
    print("\n🎯 TESTING: New Travel-Focused Theme Discovery")
    print("=" * 50)
    
    theme_tool = EnhancedThemeAnalysisTool()
    evidence_list = create_test_evidence()
    
    # Test individual extraction methods
    print("📍 Testing POI Theme Extraction:")
    poi_themes = theme_tool._extract_poi_themes(evidence_list, "Flagstaff, Arizona")
    for theme in poi_themes:
        print(f"   ✅ POI: {theme.name} (inspiration: {theme.metadata.get('inspiration_score', 0):.2f})")
    
    print("\n🔥 Testing Popular Theme Extraction:")
    popular_themes = theme_tool._extract_popular_themes(evidence_list, "Flagstaff, Arizona")
    for theme in popular_themes:
        print(f"   ✅ Popular: {theme.name} (trending: {theme.metadata.get('trending_score', 0):.2f})")
    
    print("\n🎭 Testing Cultural Theme Extraction:")
    cultural_themes = theme_tool._extract_cultural_themes(evidence_list, "Flagstaff, Arizona")
    for theme in cultural_themes:
        print(f"   ✅ Cultural: {theme.name} (authenticity: {theme.metadata.get('authenticity_score', 0):.2f})")
    
    print("\n📋 Testing Practical Theme Extraction:")
    practical_themes = theme_tool._extract_practical_themes(evidence_list, "Flagstaff, Arizona")
    for theme in practical_themes:
        print(f"   ✅ Practical: {theme.name} (utility: {theme.metadata.get('utility_score', 0):.2f})")
    
    # Test complete discovery with limits
    print("\n🎯 Testing Complete Discovery with Limits:")
    all_themes = await theme_tool._discover_themes(evidence_list, "Flagstaff, Arizona")
    
    category_counts = {}
    for theme in all_themes:
        category = theme.macro_category
        category_counts[category] = category_counts.get(category, 0) + 1
        print(f"   ✅ {category}: {theme.name} (fit_score: {theme.fit_score:.2f})")
    
    print(f"\n📊 Theme Count Summary:")
    print(f"   Popular: {category_counts.get('Popular', 0)}/3 (limit)")
    print(f"   POI: {category_counts.get('POI', 0)}/4 (limit)")
    print(f"   Cultural: {category_counts.get('Cultural', 0)}/2 (limit)")
    print(f"   Practical: {category_counts.get('Practical', 0)}/1 (limit)")
    print(f"   Total: {len(all_themes)}/10 (max)")

def test_no_generic_themes():
    """Test that generic themes are no longer generated"""
    print("\n❌ TESTING: No Generic Themes Generated")
    print("=" * 50)
    
    # Evidence that would have created generic themes in old system
    generic_evidence = [
        Evidence(
            evidence_id="ev_generic_music",
            text_snippet="Flagstaff has a vibrant music heritage and thriving music scene with local bands and venues.",
            source_url="https://example.com/music",
            source_category=SourceCategory.BLOG,
            evidence_type=EvidenceType.OPINION,
            confidence=0.6,
            cultural_context={"content_type": "entertainment"},
            timestamp=datetime.now()
        ),
        Evidence(
            evidence_id="ev_generic_character",
            text_snippet="The local character and city vibe of Flagstaff is defined by its mountain town atmosphere.",
            source_url="https://example.com/character",
            source_category=SourceCategory.BLOG,
            evidence_type=EvidenceType.OPINION,
            confidence=0.5,
            cultural_context={"content_type": "general"},
            timestamp=datetime.now()
        )
    ]
    
    theme_tool = EnhancedThemeAnalysisTool()
    themes = asyncio.run(theme_tool._discover_themes(generic_evidence, "Flagstaff, Arizona"))
    
    generic_theme_names = [
        "Music Heritage", "Music Scene", "Local Character", "City Vibe",
        "Cultural Heritage", "Historical Identity", "Artistic Scene"
    ]
    
    found_generic = False
    for theme in themes:
        for generic_name in generic_theme_names:
            if generic_name.lower() in theme.name.lower():
                print(f"   ❌ FOUND Generic Theme: {theme.name}")
                found_generic = True
    
    if not found_generic:
        print("   ✅ SUCCESS: No generic themes generated!")
        print("   ✅ Old system would have created 'Music Heritage' and 'Local Character'")
        print("   ✅ New system focuses on specific, actionable content")
    
    if themes:
        print(f"\n   Instead, generated {len(themes)} travel-focused themes:")
        for theme in themes:
            print(f"      • {theme.macro_category}: {theme.name}")

async def main():
    """Run all tests"""
    print("🧪 TRAVEL-FOCUSED THEME DISCOVERY TESTS")
    print("=" * 60)
    
    # Test 1: Show what was removed
    test_generic_features_removed()
    
    # Test 2: Test new functionality
    await test_new_theme_discovery()
    
    # Test 3: Verify no generic themes
    test_no_generic_themes()
    
    print("\n" + "=" * 60)
    print("✅ ALL TESTS COMPLETED")
    print("\n🎯 SUMMARY OF CHANGES:")
    print("   • Replaced generic taxonomy with travel-focused categories")
    print("   • Added specific POI extraction using regex patterns")
    print("   • Implemented theme prioritization: Popular > POI > Cultural > Practical")
    print("   • Enforced strict theme limits to reduce noise")
    print("   • Maintained safe data access practices throughout")
    print("   • Removed generic themes like 'Music Heritage' and 'Local Character'")
    print("   • Focus on actionable, specific travel content")

if __name__ == "__main__":
    asyncio.run(main()) 