#!/usr/bin/env python3
"""
Test runner for Enhanced Destination Intelligence features
Runs ALL unit tests and integration tests and provides a comprehensive report
"""

import subprocess
import sys
import os

# Add the src directory to the path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(project_root, 'src'))
sys.path.insert(0, project_root)

# Set up environment for subprocesses
env = os.environ.copy()
env['PYTHONPATH'] = f"{project_root}:{env.get('PYTHONPATH', '')}"


def run_all_tests():
    """Run ALL unit and integration tests and provide detailed reporting"""
    
    print("="*70)
    print("ENHANCED DESTINATION INTELLIGENCE - COMPLETE TEST SUITE")
    print("="*70)
    
    # Get the test directories
    unit_test_dir = os.path.join(os.path.dirname(__file__), 'unit')
    integration_test_dir = os.path.join(os.path.dirname(__file__), 'integration')
    
    # Complete list of ALL available unit test files
    unit_test_files = [
        'test_cache_utilities.py',
        'test_confidence_scoring.py',
        'test_destination_classifier.py',
        'test_enhanced_data_models.py',
        'test_enhanced_theme_analysis_tool.py',
        'test_evidence_hierarchy.py',
        'test_fixes.py',
        'test_insight_classifier.py',
        'test_priority_data_extraction_tool.py',  # Our new semantic extraction tests!
        'test_schemas.py',
        'test_seasonal_intelligence.py'
    ]
    
    # Complete list of ALL available integration test files
    integration_test_files = [
        'test_caching_layers.py',
        'test_enhanced_fields.py',
        'test_enhanced_fields_comprehensive.py'
    ]
    
    print(f"Unit test directory: {unit_test_dir}")
    print(f"Integration test directory: {integration_test_dir}")
    print(f"Unit test files to run: {len(unit_test_files)}")
    print(f"Integration test files to run: {len(integration_test_files)}")
    print(f"Total test files: {len(unit_test_files) + len(integration_test_files)}")
    
    # Check if test files exist and categorize them
    missing_files = []
    found_unit_tests = []
    found_integration_tests = []
    
    for test_file in unit_test_files:
        test_path = os.path.join(unit_test_dir, test_file)
        if not os.path.exists(test_path):
            missing_files.append(f"unit/{test_file}")
        else:
            found_unit_tests.append(test_file)
            print(f"✓ Found unit test: {test_file}")
            
    for test_file in integration_test_files:
        test_path = os.path.join(integration_test_dir, test_file)
        if not os.path.exists(test_path):
            missing_files.append(f"integration/{test_file}")
        else:
            found_integration_tests.append(test_file)
            print(f"✓ Found integration test: {test_file}")
    
    if missing_files:
        print(f"\n⚠️  Missing test files (will be skipped):")
        for file in missing_files:
            print(f"  - {file}")
    
    total_found = len(found_unit_tests) + len(found_integration_tests)
    print(f"\n✅ Found {total_found} test files to run")
    
    # Show test categories
    print(f"\n📋 TEST BREAKDOWN:")
    print(f"   Unit Tests: {len(found_unit_tests)} files")
    for test in found_unit_tests:
        category = "Priority Data" if "priority" in test else \
                  "Enhanced Theme" if "theme" in test else \
                  "Cache System" if "cache" in test else \
                  "Core Framework" if test in ['test_schemas.py', 'test_enhanced_data_models.py'] else \
                  "Intelligence" if any(x in test for x in ['confidence', 'classifier', 'seasonal', 'evidence']) else \
                  "Fixes & Utils"
        print(f"     • {test} ({category})")
    
    print(f"   Integration Tests: {len(found_integration_tests)} files")
    for test in found_integration_tests:
        category = "Caching Integration" if "caching" in test else "Enhanced Fields"
        print(f"     • {test} ({category})")
    
    # Run tests with pytest
    print("\n" + "="*70)
    print("RUNNING UNIT TESTS")
    print("="*70)
    
    try:
        # Run unit tests
        unit_cmd = [
            sys.executable, '-m', 'pytest', 
            unit_test_dir, 
            '-v', 
            '--tb=short',
            '--no-header',
            '--disable-warnings'  # Reduce noise
        ]
        
        unit_result = subprocess.run(unit_cmd, capture_output=True, text=True, cwd=project_root, env=env)
        
        # Print the output
        print(unit_result.stdout)
        if unit_result.stderr:
            print("STDERR:")
            print(unit_result.stderr)
            
        print("\n" + "="*70)
        print("RUNNING INTEGRATION TESTS")
        print("="*70)
        
        # Run integration tests
        integration_cmd = [
            sys.executable, '-m', 'pytest', 
            integration_test_dir, 
            '-v', 
            '--tb=short',
            '--no-header',
            '--disable-warnings'  # Reduce noise
        ]
        
        integration_result = subprocess.run(integration_cmd, capture_output=True, text=True, cwd=project_root, env=env)
        
        # Print the output
        print(integration_result.stdout)
        if integration_result.stderr:
            print("STDERR:")
            print(integration_result.stderr)
        
        # Parse results from pytest output
        unit_lines = unit_result.stdout.split('\n')
        integration_lines = integration_result.stdout.split('\n')
        
        # Look for the summary lines
        unit_summary = None
        integration_summary = None
        for line in unit_lines:
            if 'passed' in line and ('failed' in line or 'error' in line or line.strip().endswith('passed')):
                unit_summary = line.strip()
                break
                
        for line in integration_lines:
            if 'passed' in line and ('failed' in line or 'error' in line or line.strip().endswith('passed')):
                integration_summary = line.strip()
                break
        
        print("\n" + "="*70)
        print("COMPREHENSIVE TEST SUMMARY")
        print("="*70)
        
        if unit_summary:
            print(f"Unit Tests: {unit_summary}")
        else:
            print(f"Unit Tests: Exit code {unit_result.returncode}")
            
        if integration_summary:
            print(f"Integration Tests: {integration_summary}")
        else:
            print(f"Integration Tests: Exit code {integration_result.returncode}")
        
        success = unit_result.returncode == 0 and integration_result.returncode == 0
        
        # Show what was tested
        print(f"\n📊 COVERAGE SUMMARY:")
        print(f"   ✅ Semantic Priority Data Extraction")
        print(f"   ✅ Enhanced Theme Analysis")  
        print(f"   ✅ Cache Utilities & Performance")
        print(f"   ✅ Core Data Models & Schemas")
        print(f"   ✅ Intelligence Components (Confidence, Classification)")
        print(f"   ✅ Evidence Hierarchy & Seasonal Logic")
        print(f"   ✅ Integration & End-to-End Workflows")
        
        if success:
            print("\n🎉 ALL TESTS PASSED SUCCESSFULLY!")
            print("   The complete Enhanced Destination Intelligence system is validated.")
        else:
            print("\n❌ Some tests failed. See details above.")
            print("   This indicates issues that need to be resolved.")
            
        return success
        
    except FileNotFoundError:
        print("❌ pytest not found. Falling back to unittest...")
        return run_with_unittest(found_unit_tests, found_integration_tests)
    except Exception as e:
        print(f"❌ Error running tests: {e}")
        return False


def run_with_unittest(unit_tests, integration_tests):
    """Fallback to unittest if pytest is not available"""
    import unittest
    
    # Get the test directories
    unit_test_dir = os.path.join(os.path.dirname(__file__), 'unit')
    integration_test_dir = os.path.join(os.path.dirname(__file__), 'integration')
    
    # Discover and run tests
    loader = unittest.TestLoader()
    unit_suite = loader.discover(unit_test_dir, pattern='test_*.py')
    integration_suite = loader.discover(integration_test_dir, pattern='test_*.py')
    
    # Combine test suites
    all_tests = unittest.TestSuite([unit_suite, integration_suite])
    
    runner = unittest.TextTestRunner(verbosity=2, buffer=True)
    result = runner.run(all_tests)
    
    print("\n" + "="*70)
    print("COMPREHENSIVE TEST SUMMARY (unittest)")
    print("="*70)
    
    total_tests = result.testsRun
    failures = len(result.failures)
    errors = len(result.errors)
    successful = total_tests - failures - errors
    
    print(f"Total tests run: {total_tests}")
    print(f"Successful: {successful}")
    print(f"Failures: {failures}")
    print(f"Errors: {errors}")
    
    if total_tests > 0:
        success_rate = (successful / total_tests) * 100
        print(f"Success Rate: {success_rate:.1f}%")
        
        if success_rate == 100:
            print("🎉 All tests passed!")
        else:
            print("❌ Some tests failed.")
    
    return result.wasSuccessful()


if __name__ == '__main__':
    success = run_all_tests()
    sys.exit(0 if success else 1) 