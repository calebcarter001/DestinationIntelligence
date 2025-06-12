#!/usr/bin/env python3
"""
Test to verify that the evidence linking fix in enhanced_theme_analysis_tool.py works correctly
"""

import sys
import os
import asyncio
from unittest.mock import Mock

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from src.tools.enhanced_theme_analysis_tool import EnhancedThemeAnalysisTool, EnhancedThemeAnalysisInput
from src.core.enhanced_data_models import Evidence
from src.core.evidence_hierarchy import SourceCategory, EvidenceType
from datetime import datetime

async def test_evidence_linking_fix():
    """Test that themes are properly linked to evidence objects"""
    
    print("🔧 Testing Evidence Linking Fix...")
    
    # Create test evidence with correct enum values
    test_evidence = [
        Evidence(
            id="test_ev_1",
            text_snippet="Pike Place Market is a famous public market in Seattle.",
            source_url="https://example.com/pike-place",
            source_category=SourceCategory.GUIDEBOOK,  # Fixed enum value
            evidence_type=EvidenceType.PRIMARY,        # Fixed enum value  
            authority_weight=0.8,
            cultural_context={"local_entities": ["Pike Place Market"]},
            sentiment=0.7,
            relationships=[],
            agent_id="test_agent",
            published_date=datetime.now(),
            confidence=0.8,
            timestamp=datetime.now()
        ),
        Evidence(
            id="test_ev_2", 
            text_snippet="Seattle's coffee culture is legendary with many local roasteries.",
            source_url="https://example.com/coffee-culture",
            source_category=SourceCategory.BLOG,       # Fixed enum value
            evidence_type=EvidenceType.SECONDARY,      # Fixed enum value
            authority_weight=0.7,
            cultural_context={"local_entities": ["Seattle"]},
            sentiment=0.8,
            relationships=[],
            agent_id="test_agent",
            published_date=datetime.now(),
            confidence=0.7,
            timestamp=datetime.now()
        )
    ]
    
    # Create test content
    test_content = [
        {"url": "https://example.com/pike-place", "content": "Pike Place Market is a famous public market in Seattle."},
        {"url": "https://example.com/coffee-culture", "content": "Seattle's coffee culture is legendary with many local roasteries."}
    ]
    
    # Initialize the tool
    tool = EnhancedThemeAnalysisTool()
    
    # Create mock input
    input_data = EnhancedThemeAnalysisInput(
        destination_name="Seattle, United States",
        country_code="US", 
        text_content_list=test_content,
        analyze_temporal=True,
        min_confidence=0.5
    )
    
    # Run analysis
    result = await tool.analyze_themes(input_data)
    
    # Verify results
    themes = result.get("themes", [])
    print(f"✅ Generated {len(themes)} themes")
    
    # Check evidence linking
    evidence_linked_themes = 0
    total_evidence_refs = 0
    total_evidence_objects = 0
    
    for theme in themes:
        evidence_refs = theme.get("evidence_references", [])
        evidence_objects = theme.get("evidence", [])
        
        total_evidence_refs += len(evidence_refs)
        total_evidence_objects += len(evidence_objects)
        
        if evidence_refs or evidence_objects:
            evidence_linked_themes += 1
            print(f"  Theme '{theme.get('name')}': {len(evidence_refs)} refs, {len(evidence_objects)} objects")
    
    print(f"\n📊 EVIDENCE LINKING RESULTS:")
    print(f"   Themes with evidence: {evidence_linked_themes}/{len(themes)}")
    print(f"   Total evidence references: {total_evidence_refs}")
    print(f"   Total evidence objects: {total_evidence_objects}")
    
    # Verify fix success
    if evidence_linked_themes > 0 and total_evidence_objects > 0:
        print(f"\n✅ EVIDENCE LINKING FIX: SUCCESS!")
        print(f"   Themes now have evidence objects attached")
        print(f"   This should resolve 'No evidence stored for theme' warnings")
        return True
    else:
        print(f"\n❌ EVIDENCE LINKING FIX: FAILED!")
        print(f"   Themes still have no evidence objects") 
        return False

def main():
    """Run the evidence linking test"""
    try:
        result = asyncio.run(test_evidence_linking_fix())
        if result:
            print(f"\n🎉 Evidence linking fix verified successfully!")
            return 0
        else:
            print(f"\n💥 Evidence linking fix needs more work")
            return 1
    except Exception as e:
        print(f"\n💥 Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    exit(main()) 