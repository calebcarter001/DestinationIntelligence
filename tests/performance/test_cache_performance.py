#!/usr/bin/env python3

import asyncio
import sys
import os
import time
import tempfile
import shutil
import statistics
from typing import List, Dict, Any
import pytest
from unittest.mock import patch, AsyncMock

# Add project root to path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.caching import read_from_cache, write_to_cache, get_cache_path
from src.core.web_discovery_logic import WebDiscoveryLogic
from src.tools.chroma_interaction_tools import ChromaDBManager


class CachePerformanceBenchmark:
    """Performance benchmarking for cache layers"""
    
    def __init__(self):
        self.temp_dir = tempfile.mkdtemp()
        self.results = {}
        
        # Patch cache directory
        import src.caching
        self.original_cache_dir = src.caching.CACHE_DIR
        src.caching.CACHE_DIR = self.temp_dir
    
    def cleanup(self):
        """Cleanup benchmark environment"""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
        
        # Restore cache directory
        import src.caching
        src.caching.CACHE_DIR = self.original_cache_dir
    
    def time_function(self, func, *args, **kwargs):
        """Time a function execution"""
        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        end_time = time.perf_counter()
        return result, end_time - start_time
    
    async def time_async_function(self, func, *args, **kwargs):
        """Time an async function execution"""
        start_time = time.perf_counter()
        result = await func(*args, **kwargs)
        end_time = time.perf_counter()
        return result, end_time - start_time


class TestFileCachePerformance:
    """Test file cache performance characteristics"""
    
    def setup_method(self):
        """Setup performance test environment"""
        self.benchmark = CachePerformanceBenchmark()
    
    def teardown_method(self):
        """Cleanup performance test environment"""
        self.benchmark.cleanup()

    def test_cache_write_performance(self):
        """Benchmark cache write performance"""
        # Test data of varying sizes
        test_sizes = [
            ("small", {"data": "small test data"}),
            ("medium", {"data": ["item"] * 1000}),
            ("large", {"data": list(range(10000))})
        ]
        
        write_times = {}
        
        for size_name, data in test_sizes:
            key_parts = ["performance", "write", size_name]
            
            # Measure multiple writes and take average
            times = []
            for i in range(10):
                _, duration = self.benchmark.time_function(
                    write_to_cache, [*key_parts, str(i)], data
                )
                times.append(duration)
            
            avg_time = statistics.mean(times)
            write_times[size_name] = avg_time
            
            print(f"Cache write ({size_name}): {avg_time:.4f}s avg")
        
        # Verify reasonable performance (should be under 0.1s for all sizes)
        for size_name, avg_time in write_times.items():
            assert avg_time < 0.1, f"Cache write too slow for {size_name}: {avg_time}s"

    def test_cache_read_performance(self):
        """Benchmark cache read performance"""
        # Pre-populate cache with test data
        test_data = {"numbers": list(range(5000)), "strings": [f"item_{i}" for i in range(1000)]}
        key_parts = ["performance", "read"]
        write_to_cache(key_parts, test_data)
        
        # Measure read performance
        read_times = []
        for i in range(50):
            _, duration = self.benchmark.time_function(
                read_from_cache, key_parts, expiry_days=1
            )
            read_times.append(duration)
        
        avg_read_time = statistics.mean(read_times)
        print(f"Cache read average: {avg_read_time:.4f}s")
        
        # Cache reads should be very fast (under 0.01s)
        assert avg_read_time < 0.01, f"Cache read too slow: {avg_read_time}s"

    def test_cache_hit_vs_miss_performance(self):
        """Compare cache hit vs miss performance"""
        key_parts = ["performance", "hit_miss"]
        test_data = {"cached": True, "data": list(range(1000))}
        
        # Measure cache miss (first read)
        _, miss_time = self.benchmark.time_function(
            read_from_cache, key_parts, expiry_days=1
        )
        
        # Write to cache
        write_to_cache(key_parts, test_data)
        
        # Measure cache hit
        hit_times = []
        for _ in range(10):
            _, duration = self.benchmark.time_function(
                read_from_cache, key_parts, expiry_days=1
            )
            hit_times.append(duration)
        
        avg_hit_time = statistics.mean(hit_times)
        
        print(f"Cache miss time: {miss_time:.4f}s")
        print(f"Cache hit time: {avg_hit_time:.4f}s")
        
        # Cache hits should be fast, but cache miss is very fast too (just file existence check)
        # The important thing is that both are fast
        assert avg_hit_time < 0.01  # Cache hits should be very fast
        assert miss_time < 0.01     # Cache misses should also be fast (just file check)

    def test_concurrent_cache_performance(self):
        """Test cache performance under concurrent access"""
        import threading
        import queue
        
        key_base = ["performance", "concurrent"]
        test_data = {"concurrent": True, "data": list(range(100))}
        
        # Results queue for thread-safe collection
        results_queue = queue.Queue()
        
        def cache_worker(worker_id):
            # Each worker performs multiple cache operations
            worker_times = []
            for i in range(10):
                key_parts = [*key_base, str(worker_id), str(i)]
                
                # Write
                start = time.perf_counter()
                write_to_cache(key_parts, test_data)
                write_time = time.perf_counter() - start
                
                # Read
                start = time.perf_counter()
                read_from_cache(key_parts, 1)
                read_time = time.perf_counter() - start
                
                worker_times.append((write_time, read_time))
            
            results_queue.put((worker_id, worker_times))
        
        # Run concurrent workers
        threads = []
        num_workers = 5
        
        start_concurrent = time.perf_counter()
        for worker_id in range(num_workers):
            thread = threading.Thread(target=cache_worker, args=(worker_id,))
            threads.append(thread)
            thread.start()
        
        # Wait for completion
        for thread in threads:
            thread.join()
        
        total_concurrent_time = time.perf_counter() - start_concurrent
        
        # Collect results
        all_write_times = []
        all_read_times = []
        
        while not results_queue.empty():
            worker_id, times = results_queue.get()
            for write_time, read_time in times:
                all_write_times.append(write_time)
                all_read_times.append(read_time)
        
        avg_concurrent_write = statistics.mean(all_write_times)
        avg_concurrent_read = statistics.mean(all_read_times)
        
        print(f"Concurrent cache performance:")
        print(f"  Total time ({num_workers} workers): {total_concurrent_time:.4f}s")
        print(f"  Avg write time: {avg_concurrent_write:.4f}s")
        print(f"  Avg read time: {avg_concurrent_read:.4f}s")
        
        # Should handle concurrent access reasonably well
        assert avg_concurrent_write < 0.05
        assert avg_concurrent_read < 0.01


class TestWebDiscoveryPerformance:
    """Test web discovery cache performance"""
    
    def setup_method(self):
        """Setup web discovery performance test"""
        self.benchmark = CachePerformanceBenchmark()
        
        # Mock config
        self.config = {
            "web_discovery": {
                "search_results_per_query": 3,
                "min_content_length_chars": 100,
                "max_page_content_bytes": 1024 * 1024,
                "max_sources_for_agent_processing": 5
            },
            "caching": {
                "brave_search_expiry_days": 7,
                "page_content_expiry_days": 30
            },
            "priority_settings": {
                "enable_priority_discovery": False
            }
        }
    
    def teardown_method(self):
        """Cleanup web discovery performance test"""
        self.benchmark.cleanup()

    @pytest.mark.asyncio
    async def test_brave_search_cache_performance(self):
        """Test Brave Search cache performance benefits"""
        api_key = "test_api_key"
        
        async with WebDiscoveryLogic(api_key, self.config) as discovery:
            query = "test performance query"
            
            # Mock API response
            mock_results = [
                {"url": f"https://example{i}.com", "title": f"Test {i}", "snippet": f"Content {i}"}
                for i in range(5)
            ]
            
            with patch.object(discovery.session, 'get') as mock_get:
                mock_response = AsyncMock()
                mock_response.status = 200
                mock_response.json = AsyncMock(return_value={
                    "web": {"results": mock_results}
                })
                mock_get.return_value.__aenter__ = AsyncMock(return_value=mock_response)
                
                # First call - API + cache write
                _, api_time = await self.benchmark.time_async_function(
                    discovery._fetch_brave_search, query
                )
                
                # Subsequent calls - cache hits
                cache_times = []
                for _ in range(10):
                    _, duration = await self.benchmark.time_async_function(
                        discovery._fetch_brave_search, query
                    )
                    cache_times.append(duration)
                
                avg_cache_time = statistics.mean(cache_times)
                
                print(f"Brave Search performance:")
                print(f"  API call time: {api_time:.4f}s")
                print(f"  Cache hit time: {avg_cache_time:.4f}s")
                print(f"  Performance improvement: {api_time / avg_cache_time:.1f}x")
                
                # Cache should be significantly faster than API
                assert avg_cache_time < api_time / 2  # At least 2x faster
                assert avg_cache_time < 0.01  # Very fast cache access

    @pytest.mark.asyncio
    async def test_page_content_cache_performance(self):
        """Test page content cache performance benefits"""
        api_key = "test_api_key"
        
        async with WebDiscoveryLogic(api_key, self.config) as discovery:
            url = "https://example.com/performance-test"
            
            # Large mock HTML content
            mock_html = """
            <html><body>
                <article>
                    <h1>Performance Test Article</h1>
                    {}
                </article>
            </body></html>
            """.format("\n".join([f"<p>Paragraph {i} with content for performance testing.</p>" for i in range(100)]))
            
            with patch.object(discovery.session, 'get') as mock_get:
                mock_response = AsyncMock()
                mock_response.status = 200
                mock_response.headers = {'content-type': 'text/html'}
                mock_response.content.read = AsyncMock(return_value=mock_html.encode('utf-8'))
                mock_get.return_value.__aenter__ = AsyncMock(return_value=mock_response)
                
                # First call - fetch + processing + cache write
                _, fetch_time = await self.benchmark.time_async_function(
                    discovery._fetch_page_content, url
                )
                
                # Subsequent calls - cache hits
                cache_times = []
                for _ in range(10):
                    _, duration = await self.benchmark.time_async_function(
                        discovery._fetch_page_content, url
                    )
                    cache_times.append(duration)
                
                avg_cache_time = statistics.mean(cache_times)
                
                print(f"Page content performance:")
                print(f"  Fetch + process time: {fetch_time:.4f}s")
                print(f"  Cache hit time: {avg_cache_time:.4f}s") 
                print(f"  Performance improvement: {fetch_time / avg_cache_time:.1f}x")
                
                # Cache should be much faster than fetching and processing
                assert avg_cache_time < fetch_time / 5  # At least 5x faster
                assert avg_cache_time < 0.01  # Very fast cache access


class TestChromaDBPerformance:
    """Test ChromaDB vector cache performance"""
    
    def setup_method(self):
        """Setup ChromaDB performance test"""
        self.benchmark = CachePerformanceBenchmark()
        self.temp_db_dir = os.path.join(self.benchmark.temp_dir, "chroma")
        self.collection_name = "performance_test"
    
    def teardown_method(self):
        """Cleanup ChromaDB performance test"""
        self.benchmark.cleanup()

    def test_chromadb_storage_performance(self):
        """Test ChromaDB storage performance"""
        from src.schemas import ProcessedPageChunk
        
        manager = ChromaDBManager(
            db_path=self.temp_db_dir,
            collection_name=self.collection_name
        )
        
        # Create test chunks of varying sizes
        small_chunks = [
            ProcessedPageChunk(
                chunk_id=f"small_{i}",
                url=f"https://example.com/small/{i}",
                title=f"Small Chunk {i}",
                text_chunk=f"Small content for chunk {i}.",
                chunk_order=0,
                metadata={"size": "small", "index": i}
            )
            for i in range(50)
        ]
        
        large_chunks = [
            ProcessedPageChunk(
                chunk_id=f"large_{i}",
                url=f"https://example.com/large/{i}",
                title=f"Large Chunk {i}",
                text_chunk=" ".join([f"Large content sentence {j} for chunk {i}." for j in range(50)]),
                chunk_order=0,
                metadata={"size": "large", "index": i}
            )
            for i in range(20)
        ]
        
        # Test small chunks performance
        _, small_time = self.benchmark.time_function(
            manager.add_chunks, small_chunks
        )
        
        # Test large chunks performance
        _, large_time = self.benchmark.time_function(
            manager.add_chunks, large_chunks
        )
        
        print(f"ChromaDB storage performance:")
        print(f"  50 small chunks: {small_time:.4f}s ({small_time/50:.4f}s per chunk)")
        print(f"  20 large chunks: {large_time:.4f}s ({large_time/20:.4f}s per chunk)")
        
        # Should handle chunk storage efficiently
        assert small_time / 50 < 0.1  # Under 0.1s per small chunk
        assert large_time / 20 < 0.2   # Under 0.2s per large chunk

    def test_chromadb_search_performance(self):
        """Test ChromaDB search performance"""
        from src.schemas import ProcessedPageChunk
        
        manager = ChromaDBManager(
            db_path=self.temp_db_dir,
            collection_name=self.collection_name
        )
        
        # Pre-populate with search data
        search_chunks = [
            ProcessedPageChunk(
                chunk_id=f"search_{i}",
                url=f"https://example.com/search/{i}",
                title=f"Search Test {i}",
                text_chunk=f"This is search content about topic {i % 10}. Contains keywords for semantic matching.",
                chunk_order=0,
                metadata={"topic": i % 10}
            )
            for i in range(200)
        ]
        
        manager.add_chunks(search_chunks)
        
        # Test search performance
        search_queries = [
            ["search content topic"],
            ["semantic matching keywords"],
            ["test data example"],
            ["content about topic"]
        ]
        
        search_times = []
        for query in search_queries:
            _, duration = self.benchmark.time_function(
                manager.search, query, n_results=10
            )
            search_times.append(duration)
        
        avg_search_time = statistics.mean(search_times)
        
        print(f"ChromaDB search performance:")
        print(f"  Average search time: {avg_search_time:.4f}s")
        print(f"  Searches per second: {1/avg_search_time:.1f}")
        
        # Should provide fast semantic search
        assert avg_search_time < 0.5  # Under 0.5s per search
        assert 1/avg_search_time > 2   # At least 2 searches per second


def run_comprehensive_performance_test():
    """Run comprehensive performance test suite"""
    print("🚀 Running Comprehensive Cache Performance Tests")
    print("=" * 60)
    
    # File cache performance
    print("\n📁 File Cache Performance Tests")
    print("-" * 40)
    file_benchmark = TestFileCachePerformance()
    file_benchmark.setup_method()
    try:
        file_benchmark.test_cache_write_performance()
        file_benchmark.test_cache_read_performance()
        file_benchmark.test_cache_hit_vs_miss_performance()
        file_benchmark.test_concurrent_cache_performance()
        print("✅ File cache performance tests passed")
    finally:
        file_benchmark.teardown_method()
    
    # Web discovery performance
    print("\n🌐 Web Discovery Performance Tests")
    print("-" * 40)
    web_benchmark = TestWebDiscoveryPerformance()
    web_benchmark.setup_method()
    try:
        asyncio.run(web_benchmark.test_brave_search_cache_performance())
        asyncio.run(web_benchmark.test_page_content_cache_performance())
        print("✅ Web discovery performance tests passed")
    finally:
        web_benchmark.teardown_method()
    
    # ChromaDB performance
    print("\n🧠 ChromaDB Performance Tests")
    print("-" * 40)
    chroma_benchmark = TestChromaDBPerformance()
    chroma_benchmark.setup_method()
    try:
        chroma_benchmark.test_chromadb_storage_performance()
        chroma_benchmark.test_chromadb_search_performance()
        print("✅ ChromaDB performance tests passed")
    finally:
        chroma_benchmark.teardown_method()
    
    print("\n🎉 All performance tests completed successfully!")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--comprehensive":
        run_comprehensive_performance_test()
    else:
        # Run with pytest
        pytest.main([__file__, "-v", "-s"]) 