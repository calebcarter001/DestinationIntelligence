#!/usr/bin/env python3

import os
import sys
import subprocess
import asyncio
from datetime import datetime

# Add project root to path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

def run_command(command, description=""):
    """Run a command and return the result"""
    print(f"\n🔍 {description}")
    print("-" * 50)
    
    try:
        result = subprocess.run(
            command,
            shell=True,
            capture_output=True,
            text=True,
            cwd=PROJECT_ROOT
        )
        
        if result.returncode == 0:
            print(f"✅ {description} - PASSED")
            if result.stdout:
                print(result.stdout)
            return True
        else:
            print(f"❌ {description} - FAILED")
            if result.stderr:
                print(f"Error: {result.stderr}")
            if result.stdout:
                print(f"Output: {result.stdout}")
            return False
            
    except Exception as e:
        print(f"❌ {description} - ERROR: {e}")
        return False

def main():
    """Run comprehensive cache testing suite"""
    print("🧪 DESTINATION INTELLIGENCE - CACHE TESTING SUITE")
    print("=" * 60)
    print(f"📅 Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"📁 Project root: {PROJECT_ROOT}")
    
    test_results = {}
    total_tests = 0
    passed_tests = 0
    
    # Test categories
    test_categories = [
        {
            "name": "File Cache Unit Tests",
            "command": "python -m pytest tests/unit/test_cache_utilities.py -v -s",
            "description": "Core file caching utility tests"
        },
        {
            "name": "Cache Layer Integration Tests", 
            "command": "python -m pytest tests/integration/test_caching_layers.py -v -s",
            "description": "Multi-layer cache integration tests"
        },
        {
            "name": "Cache Performance Benchmarks",
            "command": "python -m pytest tests/performance/test_cache_performance.py -v -s",
            "description": "Cache performance and benchmark tests"
        },
        {
            "name": "Enhanced Fields Comprehensive Test",
            "command": "python -m pytest tests/integration/test_enhanced_fields_comprehensive.py::test_database_persistence_and_export -v -s",
            "description": "Enhanced fields cache persistence test"
        }
    ]
    
    # Run all test categories
    for category in test_categories:
        total_tests += 1
        success = run_command(category["command"], category["description"])
        test_results[category["name"]] = success
        if success:
            passed_tests += 1
    
    # Performance benchmark (separate run)
    print("\n🚀 Running Comprehensive Performance Benchmark")
    print("-" * 50)
    
    try:
        # Import and run the comprehensive performance test directly
        sys.path.insert(0, os.path.join(PROJECT_ROOT, "tests", "performance"))
        from test_cache_performance import run_comprehensive_performance_test
        run_comprehensive_performance_test()
        test_results["Performance Benchmark"] = True
        total_tests += 1
        passed_tests += 1
    except Exception as e:
        print(f"❌ Performance Benchmark - ERROR: {e}")
        test_results["Performance Benchmark"] = False
        total_tests += 1
    
    # Summary report
    print("\n" + "=" * 60)
    print("📊 CACHE TESTING SUMMARY REPORT")
    print("=" * 60)
    
    for test_name, result in test_results.items():
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"{test_name:.<45} {status}")
    
    print(f"\n📈 Overall Results:")
    print(f"   Total Tests: {total_tests}")
    print(f"   Passed: {passed_tests}")
    print(f"   Failed: {total_tests - passed_tests}")
    print(f"   Success Rate: {(passed_tests / total_tests * 100):.1f}%")
    
    # Cache inspection
    print(f"\n🗄️ Cache Directory Inspection:")
    cache_dir = os.path.join(PROJECT_ROOT, "cache")
    if os.path.exists(cache_dir):
        cache_files = len([f for f in os.listdir(cache_dir) if f.endswith('.json')])
        print(f"   Cache files found: {cache_files}")
        
        # Show cache usage
        total_size = 0
        for root, dirs, files in os.walk(cache_dir):
            for file in files:
                file_path = os.path.join(root, file)
                if os.path.exists(file_path):
                    total_size += os.path.getsize(file_path)
        
        if total_size > 0:
            size_mb = total_size / (1024 * 1024)
            print(f"   Total cache size: {size_mb:.2f} MB")
        else:
            print(f"   Cache directory is empty")
    else:
        print(f"   Cache directory not found: {cache_dir}")
    
    # ChromaDB inspection
    chroma_dir = os.path.join(PROJECT_ROOT, "chroma_db")
    if os.path.exists(chroma_dir):
        print(f"\n🧠 ChromaDB Inspection:")
        chroma_size = 0
        chroma_files = 0
        for root, dirs, files in os.walk(chroma_dir):
            for file in files:
                file_path = os.path.join(root, file)
                if os.path.exists(file_path):
                    chroma_size += os.path.getsize(file_path)
                    chroma_files += 1
        
        if chroma_size > 0:
            size_mb = chroma_size / (1024 * 1024)
            print(f"   ChromaDB files: {chroma_files}")
            print(f"   ChromaDB size: {size_mb:.2f} MB")
        else:
            print(f"   ChromaDB directory is empty")
    
    print(f"\n🏁 Cache testing completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Exit with appropriate code
    if passed_tests == total_tests:
        print("🎉 All cache tests passed successfully!")
        sys.exit(0)
    else:
        print(f"⚠️  {total_tests - passed_tests} test(s) failed")
        sys.exit(1)

if __name__ == "__main__":
    main() 