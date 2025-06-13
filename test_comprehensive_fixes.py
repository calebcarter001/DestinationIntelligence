#!/usr/bin/env python3
"""
Comprehensive Test Script for Destination Intelligence Fixes
Tests all the major fixes we've implemented to ensure they work correctly.
"""

import sys
import os
import json
import traceback
from datetime import datetime

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

def test_database_manager_with_dict():
    """Test that database manager can handle dictionary destinations"""
    print("🧪 Testing Database Manager with Dictionary Destinations...")
    
    try:
        from src.core.enhanced_database_manager import EnhancedDatabaseManager
        
        # Create test destination as dictionary
        test_dest = {
            'id': 'test_dict_dest',
            'names': ['Test Dictionary City'],
            'themes': [],
            'country_code': 'US',
            'timezone': 'UTC',
            'population': 100000,
            'meta': {},
            'dimensions': {},
            'temporal_slices': [],
            'pois': [],
            'authentic_insights': [],
            'local_authorities': []
        }
        
        db_manager = EnhancedDatabaseManager()
        result = db_manager.store_destination(test_dest)
        db_manager.close_db()
        
        if result['database_stored']:
            print("✅ Database Manager Dictionary Test: PASSED")
            return True
        else:
            print(f"❌ Database Manager Dictionary Test: FAILED - {result['errors']}")
            return False
            
    except Exception as e:
        print(f"❌ Database Manager Dictionary Test: FAILED - {e}")
        traceback.print_exc()
        return False

def test_database_manager_with_object():
    """Test that database manager can handle Destination objects"""
    print("🧪 Testing Database Manager with Destination Objects...")
    
    try:
        from src.core.enhanced_database_manager import EnhancedDatabaseManager
        
        # Create test destination as object-like class
        class TestDestination:
            def __init__(self):
                self.id = 'test_obj_dest'
                self.names = ['Test Object City']
                self.country_code = 'US'
                self.timezone = 'UTC'
                self.population = 100000
                self.themes = []
                self.meta = {}
                self.dimensions = {}
                self.temporal_slices = []
                self.pois = []
                self.authentic_insights = []
                self.local_authorities = []
        
        test_dest = TestDestination()
        
        db_manager = EnhancedDatabaseManager()
        result = db_manager.store_destination(test_dest)
        db_manager.close_db()
        
        if result['database_stored']:
            print("✅ Database Manager Object Test: PASSED")
            return True
        else:
            print(f"❌ Database Manager Object Test: FAILED - {result['errors']}")
            return False
            
    except Exception as e:
        print(f"❌ Database Manager Object Test: FAILED - {e}")
        traceback.print_exc()
        return False

def test_json_export_with_dict():
    """Test that JSON export manager can handle dictionary destinations"""
    print("🧪 Testing JSON Export Manager with Dictionary Destinations...")
    
    try:
        from src.core.consolidated_json_export_manager import ConsolidatedJsonExportManager
        import tempfile
        
        # Create test destination as dictionary
        test_dest = {
            'id': 'test_json_dict_dest',
            'names': ['Test JSON Dictionary City'],
            'themes': [],
            'country_code': 'US',
            'timezone': 'UTC',
            'population': 100000,
            'meta': {},
            'dimensions': {},
            'temporal_slices': [],
            'pois': [],
            'authentic_insights': [],
            'local_authorities': [],
            'destination_revision': 1,
            'lineage': {}
        }
        
        with tempfile.TemporaryDirectory() as temp_dir:
            json_manager = ConsolidatedJsonExportManager(temp_dir)
            result_path = json_manager.export_destination_insights(test_dest)
            
            if os.path.exists(result_path):
                print("✅ JSON Export Dictionary Test: PASSED")
                return True
            else:
                print("❌ JSON Export Dictionary Test: FAILED - No file created")
                return False
                
    except Exception as e:
        print(f"❌ JSON Export Dictionary Test: FAILED - {e}")
        traceback.print_exc()
        return False

def test_theme_wrapper_compatibility():
    """Test that ThemeWrapper objects work correctly"""
    print("🧪 Testing ThemeWrapper Compatibility...")
    
    try:
        from src.tools.enhanced_theme_analysis_tool import ThemeWrapper
        
        # Create test theme data
        theme_data = {
            'theme_id': 'test_theme_1',
            'name': 'Test Theme',
            'description': 'A test theme',
            'macro_category': 'test',
            'micro_category': 'test',
            'fit_score': 0.8,
            'tags': ['test'],
            'evidence': [],
            'confidence_breakdown': {
                'overall_confidence': 0.7
            }
        }
        
        # Create ThemeWrapper
        wrapper = ThemeWrapper(theme_data)
        
        # Test .get() method
        theme_id = wrapper.get('theme_id', 'default')
        name = wrapper.get('name', 'default')
        
        # Test attribute access
        attr_theme_id = wrapper.theme_id
        attr_name = wrapper.name
        
        if (theme_id == 'test_theme_1' and name == 'Test Theme' and 
            attr_theme_id == 'test_theme_1' and attr_name == 'Test Theme'):
            print("✅ ThemeWrapper Compatibility Test: PASSED")
            return True
        else:
            print("❌ ThemeWrapper Compatibility Test: FAILED - Access methods not working")
            return False
            
    except Exception as e:
        print(f"❌ ThemeWrapper Compatibility Test: FAILED - {e}")
        traceback.print_exc()
        return False

def test_enhanced_theme_analysis_tool():
    """Test that Enhanced Theme Analysis Tool has all required methods"""
    print("🧪 Testing Enhanced Theme Analysis Tool Methods...")
    
    try:
        from src.tools.enhanced_theme_analysis_tool import EnhancedThemeAnalysisTool
        
        tool = EnhancedThemeAnalysisTool()
        
        # Check for required methods
        required_methods = [
            '_extract_authentic_insights_from_evidence',
            '_analyze_temporal_aspects',
            '_extract_published_date',
            '_clean_poi_name',
            '_extract_popular_themes',
            '_extract_cultural_themes',
            '_extract_practical_themes'
        ]
        
        missing_methods = []
        for method in required_methods:
            if not hasattr(tool, method):
                missing_methods.append(method)
        
        if not missing_methods:
            print("✅ Enhanced Theme Analysis Tool Methods Test: PASSED")
            return True
        else:
            print(f"❌ Enhanced Theme Analysis Tool Methods Test: FAILED - Missing: {missing_methods}")
            return False
            
    except Exception as e:
        print(f"❌ Enhanced Theme Analysis Tool Methods Test: FAILED - {e}")
        traceback.print_exc()
        return False

def test_safe_attribute_access():
    """Test that safe attribute access patterns work correctly"""
    print("🧪 Testing Safe Attribute Access Patterns...")
    
    try:
        # Test with object
        class TestObj:
            def __init__(self):
                self.id = 'test_id'
                self.names = ['Test Name']
        
        # Test with dictionary
        test_dict = {
            'id': 'test_id',
            'names': ['Test Name']
        }
        
        # Safe access function
        def safe_get_attr(obj, attr, default=None):
            if hasattr(obj, attr):
                return getattr(obj, attr, default)
            elif isinstance(obj, dict):
                return obj.get(attr, default)
            else:
                return default
        
        # Test object access
        obj = TestObj()
        obj_id = safe_get_attr(obj, 'id', 'default')
        obj_names = safe_get_attr(obj, 'names', [])
        
        # Test dict access
        dict_id = safe_get_attr(test_dict, 'id', 'default')
        dict_names = safe_get_attr(test_dict, 'names', [])
        
        # Test missing attribute
        missing = safe_get_attr(obj, 'missing', 'default')
        
        if (obj_id == 'test_id' and obj_names == ['Test Name'] and
            dict_id == 'test_id' and dict_names == ['Test Name'] and
            missing == 'default'):
            print("✅ Safe Attribute Access Test: PASSED")
            return True
        else:
            print("❌ Safe Attribute Access Test: FAILED")
            return False
            
    except Exception as e:
        print(f"❌ Safe Attribute Access Test: FAILED - {e}")
        traceback.print_exc()
        return False

def run_all_tests():
    """Run all tests and report results"""
    print("🚀 Running Comprehensive Fix Validation Tests")
    print("=" * 60)
    
    tests = [
        test_database_manager_with_dict,
        test_database_manager_with_object,
        test_json_export_with_dict,
        test_theme_wrapper_compatibility,
        test_enhanced_theme_analysis_tool,
        test_safe_attribute_access
    ]
    
    results = []
    for test in tests:
        try:
            result = test()
            results.append(result)
            print()
        except Exception as e:
            print(f"❌ Test {test.__name__} crashed: {e}")
            results.append(False)
            print()
    
    # Summary
    passed = sum(results)
    total = len(results)
    
    print("=" * 60)
    print(f"📊 TEST SUMMARY: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 ALL TESTS PASSED! System is ready for production.")
        return True
    else:
        print(f"⚠️  {total - passed} tests failed. Please review and fix.")
        return False

if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1) 