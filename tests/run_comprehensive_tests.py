"""
Comprehensive Test Runner
Runs all the new tests that would have caught the evidence_quality KeyError and other issues.
"""

import unittest
import sys
import os

# Add the root directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

def run_comprehensive_tests():
    """Run all comprehensive tests"""
    print("🧪 Comprehensive Test Suite")
    print("=" * 60)
    print("Running tests that would have caught the evidence_quality KeyError...")
    print()
    
    # Test categories to run
    test_categories = [
        ("🔧 Unit Tests", [
            "tests.unit.test_enhanced_theme_analysis_tool_unit",
            "tests.unit.test_evidence_processing_unit",
            "tests.unit.test_theme_discovery_unit",
            "tests.unit.test_cultural_intelligence_unit"
        ]),
        ("🔗 Integration Tests", [
            "tests.integration.test_theme_generation_pipeline",
            "tests.integration.test_database_integration",
            "tests.integration.test_chromadb_integration",
            "tests.integration.test_configuration_integration"
        ]),
        ("📊 Data Model Tests", [
            "tests.datamodel.test_schema_validation",
            "tests.datamodel.test_data_transformation"
        ]),
        ("🚨 Error Handling Tests", [
            "tests.error_handling.test_graceful_degradation",
            "tests.error_handling.test_exception_handling"
        ]),
        ("⚙️ Configuration Tests", [
            "tests.config.test_cultural_intelligence_config"
        ]),
        ("🌊 End-to-End Tests", [
            "tests.e2e.test_complete_app_execution"
        ])
    ]
    
    overall_results = {
        "total_tests": 0,
        "passed": 0,
        "failed": 0,
        "errors": 0,
        "skipped": 0
    }
    
    for category_name, test_modules in test_categories:
        print(f"\n{category_name}")
        print("=" * 40)
        
        category_results = {"tests": 0, "passed": 0, "failed": 0, "errors": 0, "skipped": 0}
        
        for module_name in test_modules:
            try:
                # Load the test module
                module = __import__(module_name, fromlist=[''])
                
                # Create test suite
                loader = unittest.TestLoader()
                suite = loader.loadTestsFromModule(module)
                
                # Run tests
                runner = unittest.TextTestRunner(verbosity=1, stream=open(os.devnull, 'w'))
                result = runner.run(suite)
                
                # Collect results
                tests_run = result.testsRun
                failures = len(result.failures)
                errors = len(result.errors)
                skipped = len(result.skipped) if hasattr(result, 'skipped') else 0
                passed = tests_run - failures - errors - skipped
                
                category_results["tests"] += tests_run
                category_results["passed"] += passed
                category_results["failed"] += failures
                category_results["errors"] += errors
                category_results["skipped"] += skipped
                
                # Print individual module results
                module_short = module_name.split('.')[-1]
                status = "✅ PASS" if failures == 0 and errors == 0 else "❌ FAIL"
                print(f"  {module_short:<35} {tests_run:>2} tests  {status}")
                
                if failures > 0:
                    print(f"    └─ {failures} failures")
                if errors > 0:
                    print(f"    └─ {errors} errors")
                if skipped > 0:
                    print(f"    └─ {skipped} skipped")
                    
            except ImportError as e:
                print(f"  {module_name:<35} IMPORT ERROR: {e}")
                category_results["errors"] += 1
            except Exception as e:
                print(f"  {module_name:<35} ERROR: {e}")
                category_results["errors"] += 1
        
        # Print category summary
        print(f"\n  📊 {category_name} Summary:")
        print(f"     Total: {category_results['tests']}, " +
              f"✅ Passed: {category_results['passed']}, " +
              f"❌ Failed: {category_results['failed']}, " +
              f"🚨 Errors: {category_results['errors']}, " +
              f"⏭️ Skipped: {category_results['skipped']}")
        
        # Add to overall results
        for key in overall_results:
            if key == "total_tests":
                overall_results[key] += category_results["tests"]
            else:
                overall_results[key] += category_results.get(key.replace("total_", ""), 0)
    
    # Print final summary
    print("\n" + "=" * 60)
    print("📈 COMPREHENSIVE TEST SUMMARY")
    print("=" * 60)
    
    print(f"Total Tests Run: {overall_results['total_tests']}")
    print(f"✅ Passed: {overall_results['passed']}")
    print(f"❌ Failed: {overall_results['failed']}")
    print(f"🚨 Errors: {overall_results['errors']}")
    print(f"⏭️ Skipped: {overall_results['skipped']}")
    
    success_rate = (overall_results['passed'] / max(overall_results['total_tests'], 1)) * 100
    print(f"\n📊 Success Rate: {success_rate:.1f}%")
    
    print("\n💡 Key Insight: These tests focus on the PRODUCTION layer")
    print("   (theme generation) rather than just the CONSUMPTION layer")
    print("   (scripts that use themes), catching bugs at the source!")
    
    return overall_results['failed'] == 0 and overall_results['errors'] == 0

if __name__ == "__main__":
    success = run_comprehensive_tests()
    exit(0 if success else 1) 