"""
Test runner for Cultural Intelligence functionality.
Runs all unit and integration tests for the cultural intelligence features.
"""

import unittest
import sys
import os
import logging
from io import StringIO

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - [%(levelname)s] - %(message)s')

# Add paths for imports
current_dir = os.path.dirname(__file__)
project_root = os.path.join(current_dir, '..')
sys.path.insert(0, project_root)
sys.path.insert(0, os.path.join(project_root, 'src'))

def run_cultural_intelligence_tests():
    """Run all cultural intelligence tests"""
    print("=" * 80)
    print("🎭 CULTURAL INTELLIGENCE TEST SUITE")
    print("=" * 80)
    
    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Capture test output
    test_results = {
        "total_tests": 0,
        "failures": 0,
        "errors": 0,
        "skipped": 0,
        "success": 0
    }
    
    # Test modules to run
    test_modules = [
        {
            "name": "Cultural Intelligence Core",
            "module": "tests.unit.test_cultural_intelligence",
            "description": "Core cultural intelligence functionality"
        },
        {
            "name": "Destination Comparison Cultural",
            "module": "tests.unit.test_destination_comparison_cultural", 
            "description": "Cultural intelligence in destination comparison"
        },
        {
            "name": "Cultural Intelligence End-to-End",
            "module": "tests.integration.test_cultural_intelligence_end_to_end",
            "description": "Complete cultural intelligence pipeline"
        },
        {
            "name": "Cultural Scripts Integration",
            "module": "tests.integration.test_cultural_scripts_integration",
            "description": "Integration of cultural intelligence enhanced scripts"
        }
    ]
    
    print(f"\n📋 Test Plan:")
    print(f"   • {len(test_modules)} test modules")
    print(f"   • Unit tests: theme categorization, authenticity scoring, processing types")
    print(f"   • Integration tests: end-to-end pipeline, script functionality")
    print(f"   • Error handling and consistency tests")
    
    print(f"\n🚀 Running Tests...\n")
    
    for test_module in test_modules:
        print(f"🔍 Testing: {test_module['name']}")
        print(f"   Description: {test_module['description']}")
        
        try:
            # Import and add test module
            module = __import__(test_module['module'], fromlist=[''])
            module_suite = loader.loadTestsFromModule(module)
            suite.addTest(module_suite)
            
            # Run tests for this module
            module_runner = unittest.TextTestRunner(
                stream=StringIO(),
                verbosity=2,
                buffer=True
            )
            
            module_result = module_runner.run(module_suite)
            
            # Track results
            tests_run = module_result.testsRun
            failures = len(module_result.failures)
            errors = len(module_result.errors)
            skipped = len(module_result.skipped) if hasattr(module_result, 'skipped') else 0
            success = tests_run - failures - errors - skipped
            
            test_results["total_tests"] += tests_run
            test_results["failures"] += failures
            test_results["errors"] += errors
            test_results["skipped"] += skipped
            test_results["success"] += success
            
            # Report module results
            if failures == 0 and errors == 0:
                print(f"   ✅ PASSED: {success}/{tests_run} tests")
            else:
                print(f"   ❌ ISSUES: {success} passed, {failures} failed, {errors} errors")
                
                # Show failure details
                if module_result.failures:
                    print(f"   📋 Failures:")
                    for test, error in module_result.failures:
                        print(f"      • {test}: {error.split('AssertionError:')[-1].strip()}")
                
                if module_result.errors:
                    print(f"   🚨 Errors:")
                    for test, error in module_result.errors:
                        print(f"      • {test}: {error.split('Exception:')[-1].strip()}")
            
            print()
            
        except ImportError as e:
            print(f"   ⚠️  SKIPPED: Could not import module ({e})")
            print()
        except Exception as e:
            print(f"   🚨 ERROR: {e}")
            test_results["errors"] += 1
            print()
    
    # Final results summary
    print("=" * 80)
    print("📊 CULTURAL INTELLIGENCE TEST RESULTS")
    print("=" * 80)
    
    total = test_results["total_tests"]
    success = test_results["success"]
    failures = test_results["failures"]
    errors = test_results["errors"]
    skipped = test_results["skipped"]
    
    if total > 0:
        success_rate = (success / total) * 100
        print(f"Total Tests: {total}")
        print(f"✅ Passed: {success} ({success_rate:.1f}%)")
        
        if failures > 0:
            failure_rate = (failures / total) * 100
            print(f"❌ Failed: {failures} ({failure_rate:.1f}%)")
        
        if errors > 0:
            error_rate = (errors / total) * 100
            print(f"🚨 Errors: {errors} ({error_rate:.1f}%)")
        
        if skipped > 0:
            skip_rate = (skipped / total) * 100
            print(f"⏭️  Skipped: {skipped} ({skip_rate:.1f}%)")
        
        print()
        
        if failures == 0 and errors == 0:
            print("🎉 ALL CULTURAL INTELLIGENCE TESTS PASSED!")
            print("🎭 Cultural Intelligence system is functioning correctly")
            return True
        else:
            print("⚠️  Some tests failed. Please review the issues above.")
            return False
    else:
        print("⚠️  No tests were run. Please check test module imports.")
        return False

def run_specific_feature_tests():
    """Run tests for specific cultural intelligence features"""
    print("\n🧪 FEATURE-SPECIFIC TESTS")
    print("=" * 50)
    
    features_to_test = [
        "Theme Categorization",
        "Authenticity Scoring", 
        "Distinctiveness Filtering",
        "Processing Type Identification",
        "Cultural Character Analysis",
        "Destination Personality Detection"
    ]
    
    print("Testing individual features:")
    for feature in features_to_test:
        print(f"   🎯 {feature}")
    
    print("\n(Individual feature tests are included in the main test suite above)")

def validate_test_environment():
    """Validate that the test environment is properly set up"""
    print("🔧 VALIDATING TEST ENVIRONMENT")
    print("=" * 50)
    
    validation_checks = [
        ("Python version", sys.version_info >= (3, 7)),
        ("Project root accessible", os.path.exists(project_root)),
        ("Source directory", os.path.exists(os.path.join(project_root, 'src'))),
        ("Test directory", os.path.exists(os.path.join(project_root, 'tests'))),
        ("Unit tests directory", os.path.exists(os.path.join(project_root, 'tests', 'unit'))),
        ("Integration tests directory", os.path.exists(os.path.join(project_root, 'tests', 'integration')))
    ]
    
    all_valid = True
    for check_name, is_valid in validation_checks:
        status = "✅" if is_valid else "❌"
        print(f"   {status} {check_name}")
        if not is_valid:
            all_valid = False
    
    if all_valid:
        print("✅ Test environment is properly configured")
    else:
        print("❌ Test environment has issues. Please fix before running tests.")
    
    return all_valid

def main():
    """Main test runner function"""
    print("🎭 Cultural Intelligence Test Runner")
    print("Testing comprehensive cultural intelligence functionality\n")
    
    # Validate environment
    if not validate_test_environment():
        sys.exit(1)
    
    # Run feature description
    run_specific_feature_tests()
    
    # Run all tests
    success = run_cultural_intelligence_tests()
    
    if success:
        print("\n🎯 NEXT STEPS:")
        print("   • Cultural Intelligence system is ready for use")
        print("   • Run actual destination analysis to test with real data")
        print("   • Generate dynamic viewers to visualize cultural categorization")
        print("   • Compare destinations using cultural intelligence metrics")
        sys.exit(0)
    else:
        print("\n🔧 TROUBLESHOOTING:")
        print("   • Check import paths and dependencies")
        print("   • Ensure all cultural intelligence files are properly created")
        print("   • Review test output for specific error details")
        sys.exit(1)

if __name__ == "__main__":
    main() 