#!/usr/bin/env python3

import sys
import os
import subprocess
import json
from datetime import datetime

# Add project root to path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

def run_cache_tests():
    """Run comprehensive cache testing suite"""
    print("🧪 Running Cache System Tests")
    print("=" * 50)
    
    # Test configurations
    test_commands = [
        {
            "name": "File Cache Tests",
            "command": "python -m pytest tests/integration/test_caching_layers.py::TestFileCaching -v -s",
            "description": "Test basic file-based caching system"
        },
        {
            "name": "ChromaDB Cache Tests", 
            "command": "python -m pytest tests/integration/test_caching_layers.py::TestChromaDBCaching -v -s",
            "description": "Test ChromaDB vector database caching"
        },
        {
            "name": "Database Cache Tests",
            "command": "python -m pytest tests/integration/test_caching_layers.py::TestDatabaseCaching -v -s", 
            "description": "Test enhanced database caching and storage"
        },
        {
            "name": "Cache Integration Tests",
            "command": "python -m pytest tests/integration/test_caching_layers.py::TestCacheIntegration -v -s",
            "description": "Test integration between different cache layers"
        },
        {
            "name": "Cache Utilities Tests",
            "command": "python -m pytest tests/unit/test_cache_utilities.py -v -s",
            "description": "Test cache utility functions and helpers"
        }
    ]
    
    results = {}
    total_tests = len(test_commands)
    passed_tests = 0
    
    for i, test_config in enumerate(test_commands, 1):
        print(f"\n[{i}/{total_tests}] {test_config['name']}")
        print(f"Description: {test_config['description']}")
        print(f"Command: {test_config['command']}")
        print("-" * 50)
        
        try:
            # Run the test command
            result = subprocess.run(
                test_config['command'].split(),
                capture_output=True,
                text=True,
                cwd=PROJECT_ROOT
            )
            
            if result.returncode == 0:
                print(f"✅ PASSED: {test_config['name']}")
                results[test_config['name']] = "PASSED"
                passed_tests += 1
            else:
                print(f"❌ FAILED: {test_config['name']}")
                print(f"STDERR: {result.stderr}")
                results[test_config['name']] = "FAILED"
                
        except Exception as e:
            print(f"❌ ERROR: {test_config['name']} - {str(e)}")
            results[test_config['name']] = f"ERROR: {str(e)}"
    
    # Summary
    print("\n" + "=" * 50)
    print("📊 CACHE TESTS SUMMARY")
    print("=" * 50)
    print(f"Total Tests: {total_tests}")
    print(f"Passed: {passed_tests}")
    print(f"Failed: {total_tests - passed_tests}")
    print(f"Success Rate: {(passed_tests/total_tests)*100:.1f}%")
    
    print("\nDetailed Results:")
    for test_name, status in results.items():
        status_icon = "✅" if status == "PASSED" else "❌"
        print(f"  {status_icon} {test_name}: {status}")
    
    # Save results
    timestamp = datetime.now().isoformat()
    report = {
        "timestamp": timestamp,
        "total_tests": total_tests,
        "passed_tests": passed_tests,
        "success_rate": (passed_tests/total_tests)*100,
        "results": results
    }
    
    report_path = os.path.join(PROJECT_ROOT, "tests", "cache_test_report.json")
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2)
    
    print(f"\n📋 Report saved to: {report_path}")
    
    return passed_tests == total_tests


if __name__ == "__main__":
    success = run_cache_tests()
    sys.exit(0 if success else 1) 