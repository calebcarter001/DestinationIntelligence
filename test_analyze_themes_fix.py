#!/usr/bin/env python3
"""
Test script to validate the analyze_themes.py fix
"""

import sys
import os
import sqlite3
import json
from datetime import datetime

# Add the current directory to the path so we can import analyze_themes
sys.path.insert(0, os.getcwd())

def test_get_processing_type():
    """Test the get_processing_type function with database values"""
    print("🧪 Testing get_processing_type function...")
    
    try:
        from analyze_themes import get_processing_type
        
        # Test database values
        test_cases = [
            ("Popular", "must_see"),
            ("POI", "must_see"),
            ("Cultural", "experiences"),
            ("Practical", "essentials"),
            ("Unknown", "unknown"),
            (None, "unknown"),
            ("", "unknown")
        ]
        
        for input_val, expected in test_cases:
            result = get_processing_type(input_val)
            if result == expected:
                print(f"  ✅ {input_val} -> {result}")
            else:
                print(f"  ❌ {input_val} -> {result} (expected {expected})")
                return False
        
        print("  ✅ get_processing_type function working correctly")
        return True
        
    except Exception as e:
        print(f"  ❌ Error testing get_processing_type: {e}")
        return False

def test_calculate_cultural_intelligence_metrics():
    """Test the calculate_cultural_intelligence_metrics function"""
    print("🧪 Testing calculate_cultural_intelligence_metrics function...")
    
    try:
        from analyze_themes import calculate_cultural_intelligence_metrics
        
        # Create test data with different processing types
        test_themes = [
            {
                'name': 'Space Needle',
                'macro_category': 'Popular',
                'overall_confidence': 0.8,
                'description': 'Famous landmark'
            },
            {
                'name': 'Pike Place Market',
                'macro_category': 'POI',
                'overall_confidence': 0.9,
                'description': 'Historic market'
            },
            {
                'name': 'Local Culture',
                'macro_category': 'Cultural',
                'overall_confidence': 0.7,
                'description': 'Cultural experience'
            },
            {
                'name': 'Transportation',
                'macro_category': 'Practical',
                'overall_confidence': 0.6,
                'description': 'Getting around'
            }
        ]
        
        # Test the function
        metrics = calculate_cultural_intelligence_metrics(test_themes)
        
        # Validate the results
        expected_keys = [
            "total_themes", "category_breakdown", "theme_distribution",
            "avg_confidence_by_type", "high_confidence_themes",
            "distinctiveness_analysis", "authenticity_indicators",
            "top_cultural_themes", "top_practical_themes", "cultural_practical_ratio"
        ]
        
        for key in expected_keys:
            if key not in metrics:
                print(f"  ❌ Missing key: {key}")
                return False
        
        # Check that processing types are handled correctly
        if metrics["category_breakdown"]["must_see"] != 2:  # Popular + POI
            print(f"  ❌ Expected 2 must_see themes, got {metrics['category_breakdown']['must_see']}")
            return False
            
        if metrics["category_breakdown"]["experiences"] != 1:  # Cultural
            print(f"  ❌ Expected 1 experiences theme, got {metrics['category_breakdown']['experiences']}")
            return False
            
        if metrics["category_breakdown"]["essentials"] != 1:  # Practical
            print(f"  ❌ Expected 1 essentials theme, got {metrics['category_breakdown']['essentials']}")
            return False
        
        print("  ✅ calculate_cultural_intelligence_metrics function working correctly")
        print(f"    - Total themes: {metrics['total_themes']}")
        print(f"    - Must see: {metrics['category_breakdown']['must_see']}")
        print(f"    - Experiences: {metrics['category_breakdown']['experiences']}")
        print(f"    - Essentials: {metrics['category_breakdown']['essentials']}")
        return True
        
    except Exception as e:
        print(f"  ❌ Error testing calculate_cultural_intelligence_metrics: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_database_integration():
    """Test loading themes from the actual database"""
    print("🧪 Testing database integration...")
    
    try:
        from analyze_themes import load_themes_from_db
        
        # Test with actual database
        db_path = "enhanced_destination_intelligence.db"
        if not os.path.exists(db_path):
            print("  ⚠️ Database not found, skipping database integration test")
            return True
        
        # Test loading themes for Flagstaff
        themes = load_themes_from_db(db_path, "Flagstaff, Arizona")
        
        if not themes:
            print("  ⚠️ No themes found for Flagstaff, Arizona")
            return True
        
        print(f"  ✅ Loaded {len(themes)} themes from database")
        
        # Check that processing_type is added to each theme
        for theme in themes[:3]:  # Check first 3 themes
            if 'processing_type' not in theme:
                print(f"  ❌ Theme missing processing_type: {theme.get('name', 'Unknown')}")
                return False
            print(f"    - {theme['name']}: {theme['macro_category']} -> {theme['processing_type']}")
        
        print("  ✅ Database integration working correctly")
        return True
        
    except Exception as e:
        print(f"  ❌ Error testing database integration: {e}")
        return False

def test_full_analyze_themes_script():
    """Test running the full analyze_themes.py script"""
    print("🧪 Testing full analyze_themes.py script...")
    
    try:
        # Import and run the main function
        import analyze_themes
        
        # Test with a small subset to avoid long execution
        print("  Running analyze_themes for one destination...")
        
        # This will test the full pipeline
        analyze_themes.fetch_and_save_comprehensive_report("dest_flagstaff_arizona")
        
        print("  ✅ Full script executed without errors")
        return True
        
    except Exception as e:
        print(f"  ❌ Error running full script: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all tests"""
    print("🚀 Starting analyze_themes.py fix validation tests...")
    print("=" * 60)
    
    tests = [
        test_get_processing_type,
        test_calculate_cultural_intelligence_metrics,
        test_database_integration,
        test_full_analyze_themes_script
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        try:
            if test():
                passed += 1
            print()
        except Exception as e:
            print(f"  ❌ Test failed with exception: {e}")
            print()
    
    print("=" * 60)
    print(f"📊 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! The fix is working correctly.")
        return True
    else:
        print("❌ Some tests failed. The fix needs more work.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 