#!/usr/bin/env python3

import os
import sys
import json
import tempfile
import shutil
import pytest
from datetime import datetime, timedelta
from unittest.mock import patch, mock_open
import hashlib

# Add project root to path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.caching import read_from_cache, write_to_cache, get_cache_path, CACHE_EXPIRY_DAYS


class TestCacheKeyGeneration:
    """Test cache key generation and path handling"""
    
    def test_simple_key_generation(self):
        """Test basic key generation"""
        key_parts = ["simple", "test"]
        path = get_cache_path(key_parts)
        
        expected_key = "simple_test"
        expected_hash = hashlib.md5(expected_key.encode('utf-8')).hexdigest()
        
        assert path.endswith(f"{expected_hash}.json")
        assert "cache" in path

    def test_url_sanitization(self):
        """Test URL sanitization in cache keys"""
        key_parts = ["https://example.com/page", "test:data"]
        path = get_cache_path(key_parts)
        
        # URLs should be sanitized (no https://, /, :)
        expected_key = "example.com_page_test_data"
        expected_hash = hashlib.md5(expected_key.encode('utf-8')).hexdigest()
        
        assert path.endswith(f"{expected_hash}.json")

    def test_special_characters_in_keys(self):
        """Test handling of special characters in cache keys"""
        key_parts = ["test/path", "data:value", "query?param=1"]
        path = get_cache_path(key_parts)
        
        # Should sanitize specific characters (/, :, https://, http://)
        expected_key = "test_path_data_value_query?param=1"
        expected_hash = hashlib.md5(expected_key.encode('utf-8')).hexdigest()
        
        assert path.endswith(f"{expected_hash}.json")

    def test_empty_key_parts(self):
        """Test handling of empty key parts"""
        key_parts = ["", "test", ""]
        path = get_cache_path(key_parts)
        
        expected_key = "_test_"
        expected_hash = hashlib.md5(expected_key.encode('utf-8')).hexdigest()
        
        assert path.endswith(f"{expected_hash}.json")

    def test_unicode_characters(self):
        """Test handling of unicode characters in keys"""
        key_parts = ["测试", "тест", "🚀"]
        path = get_cache_path(key_parts)
        
        expected_key = "测试_тест_🚀"
        expected_hash = hashlib.md5(expected_key.encode('utf-8')).hexdigest()
        
        assert path.endswith(f"{expected_hash}.json")


class TestCacheIO:
    """Test cache input/output operations"""
    
    def setup_method(self):
        """Setup temporary cache directory"""
        self.temp_dir = tempfile.mkdtemp()
        
        # Patch cache directory
        import src.caching
        self.original_cache_dir = src.caching.CACHE_DIR
        src.caching.CACHE_DIR = self.temp_dir
    
    def teardown_method(self):
        """Cleanup temporary directory"""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
        
        # Restore cache directory
        import src.caching
        src.caching.CACHE_DIR = self.original_cache_dir

    def test_cache_directory_creation(self):
        """Test automatic cache directory creation"""
        key_parts = ["test", "directory"]
        write_to_cache(key_parts, {"test": "data"})
        
        cache_path = get_cache_path(key_parts)
        assert os.path.exists(os.path.dirname(cache_path))

    def test_cache_file_permissions(self):
        """Test cache file permissions and accessibility"""
        key_parts = ["permissions", "test"]
        test_data = {"accessible": True}
        
        write_to_cache(key_parts, test_data)
        cache_path = get_cache_path(key_parts)
        
        # File should be readable
        assert os.access(cache_path, os.R_OK)
        # File should be writable
        assert os.access(cache_path, os.W_OK)

    def test_large_data_caching(self):
        """Test caching of large data structures"""
        key_parts = ["large", "data"]
        
        # Create large test data
        large_data = {
            "numbers": list(range(10000)),
            "strings": [f"string_{i}" for i in range(1000)],
            "nested": {
                "level1": {
                    "level2": {
                        "level3": {"deep": "data"}
                    }
                }
            }
        }
        
        write_to_cache(key_parts, large_data)
        cached_data = read_from_cache(key_parts, 1)
        
        assert cached_data == large_data
        assert len(cached_data["numbers"]) == 10000
        assert cached_data["nested"]["level1"]["level2"]["level3"]["deep"] == "data"

    def test_cache_with_none_values(self):
        """Test caching data with None values"""
        key_parts = ["none", "values"]
        test_data = {
            "valid": "data",
            "none_value": None,
            "empty_list": [],
            "empty_dict": {}
        }
        
        write_to_cache(key_parts, test_data)
        cached_data = read_from_cache(key_parts, 1)
        
        assert cached_data == test_data
        assert cached_data["none_value"] is None

    def test_concurrent_cache_access(self):
        """Test concurrent access to cache files"""
        import threading
        import time
        
        key_base = ["concurrent", "test"]
        test_data = {"concurrent": True, "data": list(range(100))}
        
        # Results queue for thread-safe collection
        results = {}
        errors = []
        lock = threading.Lock()
        
        def cache_worker(worker_id):
            try:
                data = {"worker": worker_id, "timestamp": time.time()}
                write_to_cache([*key_base, str(worker_id)], data)
                cached = read_from_cache([*key_base, str(worker_id)], 1)
                
                with lock:
                    results[worker_id] = cached
            except Exception as e:
                with lock:
                    errors.append(e)
        
        # Create multiple threads
        threads = []
        for i in range(5):
            thread = threading.Thread(target=cache_worker, args=(i,))
            threads.append(thread)
            thread.start()
        
        # Wait for all threads to complete
        for thread in threads:
            thread.join()
        
        # Verify no errors and all data cached correctly
        assert len(errors) == 0
        assert len(results) == 5
        
        for worker_id in range(5):
            assert worker_id in results
            assert results[worker_id]["worker"] == worker_id


class TestCacheExpiration:
    """Test cache expiration logic"""
    
    def setup_method(self):
        """Setup temporary cache directory"""
        self.temp_dir = tempfile.mkdtemp()
        
        # Patch cache directory
        import src.caching
        self.original_cache_dir = src.caching.CACHE_DIR
        src.caching.CACHE_DIR = self.temp_dir
    
    def teardown_method(self):
        """Cleanup temporary directory"""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
        
        # Restore cache directory
        import src.caching
        src.caching.CACHE_DIR = self.original_cache_dir

    def test_fresh_cache_retrieval(self):
        """Test retrieval of fresh cache data"""
        key_parts = ["fresh", "data"]
        test_data = {"fresh": True, "timestamp": datetime.now().isoformat()}
        
        write_to_cache(key_parts, test_data)
        
        # Should retrieve fresh data
        cached_data = read_from_cache(key_parts, expiry_days=1)
        assert cached_data == test_data

    def test_expired_cache_handling(self):
        """Test handling of expired cache entries"""
        key_parts = ["expired", "data"]
        test_data = {"expired": True}
        
        # Write cache with manual timestamp manipulation
        cache_path = get_cache_path(key_parts)
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        
        expired_timestamp = (datetime.now() - timedelta(days=10)).isoformat()
        cache_content = {
            "timestamp": expired_timestamp,
            "data": test_data
        }
        
        with open(cache_path, 'w') as f:
            json.dump(cache_content, f)
        
        # Should return None for expired cache
        cached_data = read_from_cache(key_parts, expiry_days=7)
        assert cached_data is None

    def test_different_expiry_periods(self):
        """Test different expiry periods"""
        key_parts = ["expiry", "periods"]
        test_data = {"test": "data"}
        
        # Write cache with 5-day-old timestamp
        cache_path = get_cache_path(key_parts)
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        
        old_timestamp = (datetime.now() - timedelta(days=5)).isoformat()
        cache_content = {
            "timestamp": old_timestamp,
            "data": test_data
        }
        
        with open(cache_path, 'w') as f:
            json.dump(cache_content, f)
        
        # Should be expired with 3-day expiry
        assert read_from_cache(key_parts, expiry_days=3) is None
        
        # Should be valid with 7-day expiry
        assert read_from_cache(key_parts, expiry_days=7) == test_data
        
        # Should be valid with 30-day expiry
        assert read_from_cache(key_parts, expiry_days=30) == test_data

    def test_invalid_timestamp_format(self):
        """Test handling of invalid timestamp formats"""
        key_parts = ["invalid", "timestamp"]
        test_data = {"test": "data"}
        
        cache_path = get_cache_path(key_parts)
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        
        # Invalid timestamp format
        cache_content = {
            "timestamp": "not-a-valid-timestamp",
            "data": test_data
        }
        
        with open(cache_path, 'w') as f:
            json.dump(cache_content, f)
        
        # Should handle gracefully and return None
        cached_data = read_from_cache(key_parts, expiry_days=7)
        assert cached_data is None


class TestCacheErrorHandling:
    """Test cache error handling and resilience"""
    
    def setup_method(self):
        """Setup temporary cache directory"""
        self.temp_dir = tempfile.mkdtemp()
        
        # Patch cache directory
        import src.caching
        self.original_cache_dir = src.caching.CACHE_DIR
        src.caching.CACHE_DIR = self.temp_dir
    
    def teardown_method(self):
        """Cleanup temporary directory"""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
        
        # Restore cache directory
        import src.caching
        src.caching.CACHE_DIR = self.original_cache_dir

    def test_non_serializable_data_handling(self):
        """Test handling of non-JSON-serializable data"""
        key_parts = ["non", "serializable"]
        
        # Data that can't be JSON serialized
        class NonSerializable:
            def __init__(self):
                self.data = "test"
        
        non_serializable_data = {"object": NonSerializable()}
        
        # Should handle gracefully without crashing
        try:
            write_to_cache(key_parts, non_serializable_data)
            # If it doesn't raise an exception, the data shouldn't be cached
            cached_data = read_from_cache(key_parts, 1)
            assert cached_data is None
        except (TypeError, ValueError):
            # Expected behavior - can't serialize
            pass

    def test_permission_denied_handling(self):
        """Test handling of permission denied errors"""
        key_parts = ["permission", "test"]
        
        # Mock permission denied error
        with patch('builtins.open', side_effect=PermissionError("Permission denied")):
            # Should handle gracefully
            write_to_cache(key_parts, {"test": "data"})
            cached_data = read_from_cache(key_parts, 1)
            assert cached_data is None

    def test_disk_full_handling(self):
        """Test handling of disk full errors"""
        key_parts = ["disk", "full"]
        
        # Mock disk full error
        with patch('builtins.open', side_effect=OSError("No space left on device")):
            # Should handle gracefully
            write_to_cache(key_parts, {"test": "data"})
            # No assertion needed - just shouldn't crash

    def test_corrupted_json_recovery(self):
        """Test recovery from corrupted JSON files"""
        key_parts = ["corrupted", "json"]
        
        # Create corrupted cache file
        cache_path = get_cache_path(key_parts)
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        
        with open(cache_path, 'w') as f:
            f.write('{"invalid": json content')
        
        # Should handle corrupted JSON gracefully
        cached_data = read_from_cache(key_parts, 1)
        assert cached_data is None
        
        # Should be able to write new data after corruption
        new_data = {"recovered": True}
        write_to_cache(key_parts, new_data)
        
        recovered_data = read_from_cache(key_parts, 1)
        assert recovered_data == new_data

    def test_missing_cache_directory_handling(self):
        """Test handling when cache directory doesn't exist"""
        # Use non-existent directory
        non_existent_dir = "/tmp/non_existent_cache_dir_12345"
        
        import src.caching
        original_cache_dir = src.caching.CACHE_DIR
        src.caching.CACHE_DIR = non_existent_dir
        
        try:
            key_parts = ["missing", "dir"]
            test_data = {"test": "data"}
            
            # Should create directory and cache data
            write_to_cache(key_parts, test_data)
            cached_data = read_from_cache(key_parts, 1)
            
            assert cached_data == test_data
            assert os.path.exists(non_existent_dir)
            
        finally:
            # Cleanup
            if os.path.exists(non_existent_dir):
                shutil.rmtree(non_existent_dir)
            src.caching.CACHE_DIR = original_cache_dir


if __name__ == "__main__":
    # Run tests with verbose output
    pytest.main([__file__, "-v", "-s"]) 