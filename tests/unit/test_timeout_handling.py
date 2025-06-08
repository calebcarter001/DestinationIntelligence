import pytest
import asyncio
from unittest.mock import Mock, patch, AsyncMock
import aiohttp
from src.core.web_discovery_logic import WebDiscoveryLogic

class TestTimeoutHandling:
    """Test suite for timeout handling functionality"""
    
    @pytest.fixture
    def web_discovery(self):
        """Create a WebDiscoveryLogic instance with test config"""
        config = {
            "web_discovery": {
                "timeout_seconds": 5,  # Short timeout for testing
                "max_retries": 3
            }
        }
        return WebDiscoveryLogic("test_api_key", config)

    async def test_connection_timeout(self, web_discovery):
        """Test handling of connection timeouts"""
        mock_session = AsyncMock()
        mock_session.get.side_effect = asyncio.TimeoutError()

        with patch.object(web_discovery, 'session', mock_session):
            content = await web_discovery._fetch_page_content("https://test.com", "Test")
            
            assert content == ""  # Should return empty string on timeout
            assert mock_session.get.call_count > 1  # Should have retried

    async def test_read_timeout(self, web_discovery):
        """Test handling of read timeouts"""
        # Mock response that times out during read
        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.headers = {'content-type': 'text/html'}
        mock_response.content.read.side_effect = asyncio.TimeoutError()

        mock_session = AsyncMock()
        mock_session.get.return_value.__aenter__.return_value = mock_response

        with patch.object(web_discovery, 'session', mock_session):
            content = await web_discovery._fetch_page_content("https://test.com", "Test")
            
            assert content == ""  # Should return empty string on timeout
            assert mock_response.content.read.call_count >= 1  # Should have attempted read

    async def test_retry_behavior(self, web_discovery):
        """Test retry behavior with different error types"""
        error_sequence = [
            asyncio.TimeoutError(),  # First attempt: timeout
            aiohttp.ClientError(),   # Second attempt: network error
            None                     # Third attempt: success
        ]
        
        mock_response_success = AsyncMock()
        mock_response_success.status = 200
        mock_response_success.headers = {'content-type': 'text/html'}
        mock_response_success.content.read.return_value = b"Success after retries"

        mock_session = AsyncMock()
        mock_session.get.side_effect = [
            error_sequence[0],
            error_sequence[1],
            mock_response_success  # Third attempt succeeds
        ]

        with patch.object(web_discovery, 'session', mock_session):
            content = await web_discovery._fetch_page_content("https://test.com", "Test")
            
            assert content == "Success after retries"
            assert mock_session.get.call_count == 3  # Should have tried all attempts

    async def test_max_retries(self, web_discovery):
        """Test that max retries limit is respected"""
        mock_session = AsyncMock()
        mock_session.get.side_effect = asyncio.TimeoutError()

        with patch.object(web_discovery, 'session', mock_session):
            content = await web_discovery._fetch_page_content("https://test.com", "Test")
            
            assert content == ""  # Should return empty string after all retries fail
            assert mock_session.get.call_count <= web_discovery.max_retries  # Should not exceed max retries

    async def test_partial_content_timeout(self, web_discovery):
        """Test handling of timeouts during partial content reads"""
        # Mock response that succeeds connection but times out during content read
        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.headers = {'content-type': 'text/html'}
        
        # Simulate timeout during content read
        async def mock_read(*args, **kwargs):
            await asyncio.sleep(0.1)  # Small delay
            raise asyncio.TimeoutError()
            
        mock_response.content.read = mock_read

        mock_session = AsyncMock()
        mock_session.get.return_value.__aenter__.return_value = mock_response

        with patch.object(web_discovery, 'session', mock_session):
            content = await web_discovery._fetch_page_content("https://test.com", "Test")
            
            assert content == ""  # Should return empty string on partial timeout
            assert mock_session.get.call_count >= 1  # Should have attempted at least once

    async def test_custom_timeout_configs(self, web_discovery):
        """Test custom timeout configurations"""
        # Test with different timeout values
        timeout_configs = [
            {"timeout_seconds": 1},   # Very short timeout
            {"timeout_seconds": 10},  # Longer timeout
            {"timeout_seconds": 30}   # Default timeout
        ]

        for config in timeout_configs:
            web_discovery.timeout_seconds = config["timeout_seconds"]
            
            mock_session = AsyncMock()
            mock_session.get.side_effect = asyncio.TimeoutError()

            with patch.object(web_discovery, 'session', mock_session):
                content = await web_discovery._fetch_page_content("https://test.com", "Test")
                
                assert content == ""  # Should handle timeout consistently
                assert mock_session.get.call_count > 0  # Should have attempted

    async def test_mixed_error_handling(self, web_discovery):
        """Test handling of mixed error types in sequence"""
        error_scenarios = [
            # Scenario 1: Network errors
            [
                aiohttp.ClientError(),        # Connection error
                aiohttp.ServerDisconnectedError(),  # Server disconnect
                asyncio.TimeoutError()        # Timeout
            ],
            # Scenario 2: Timeout variations
            [
                asyncio.TimeoutError(),       # General timeout
                aiohttp.ClientTimeout(),      # Client timeout
                aiohttp.ServerTimeoutError()  # Server timeout
            ],
            # Scenario 3: Mixed errors
            [
                aiohttp.ClientError(),        # Network error
                asyncio.TimeoutError(),       # Timeout
                Exception("Unknown error")    # Generic error
            ]
        ]

        for error_sequence in error_scenarios:
            mock_session = AsyncMock()
            mock_session.get.side_effect = error_sequence

            with patch.object(web_discovery, 'session', mock_session):
                content = await web_discovery._fetch_page_content("https://test.com", "Test")
                
                assert content == ""  # Should handle all error types gracefully
                assert mock_session.get.call_count <= len(error_sequence)  # Should not exceed sequence length

    async def test_timeout_with_partial_success(self, web_discovery):
        """Test timeout handling with partial success cases"""
        # Scenario 1: Timeout after successful connection
        mock_response_1 = AsyncMock()
        mock_response_1.status = 200
        mock_response_1.headers = {'content-type': 'text/html'}
        mock_response_1.content.read.side_effect = asyncio.TimeoutError()

        # Scenario 2: Timeout during headers
        mock_response_2 = AsyncMock()
        mock_response_2.status = 200
        mock_response_2.headers.side_effect = asyncio.TimeoutError()

        # Scenario 3: Timeout after partial content
        async def mock_partial_read(*args, **kwargs):
            return b"Partial content before timeout"
        
        mock_response_3 = AsyncMock()
        mock_response_3.status = 200
        mock_response_3.headers = {'content-type': 'text/html'}
        mock_response_3.content.read.side_effect = [
            mock_partial_read(),
            asyncio.TimeoutError()
        ]

        scenarios = [
            (mock_response_1, ""),  # Should return empty on content timeout
            (mock_response_2, ""),  # Should return empty on headers timeout
            (mock_response_3, "")   # Should return empty on partial content timeout
        ]

        for mock_response, expected_result in scenarios:
            mock_session = AsyncMock()
            mock_session.get.return_value.__aenter__.return_value = mock_response

            with patch.object(web_discovery, 'session', mock_session):
                content = await web_discovery._fetch_page_content("https://test.com", "Test")
                assert content == expected_result 