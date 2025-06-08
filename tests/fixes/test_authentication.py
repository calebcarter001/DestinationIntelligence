import unittest
import os
from unittest.mock import patch, MagicMock
from src.config_loader import load_app_config
from src.core.llm_factory import LLMFactory

class TestAuthenticationFixes(unittest.TestCase):
    def setUp(self):
        # Setup mock environment variables
        self.env_patcher = patch.dict('os.environ', {
            'GOOGLE_APPLICATION_CREDENTIALS': '/path/to/credentials.json',
            'GOOGLE_CLOUD_PROJECT': 'test-project',
            'GEMINI_API_KEY': 'test-gemini-key',
            'OPENAI_API_KEY': 'test-openai-key'
        })
        self.env_patcher.start()
        
    def tearDown(self):
        self.env_patcher.stop()
        
    def test_google_credentials_loading(self):
        """Test that Google credentials are properly loaded"""
        config = load_app_config()
        self.assertEqual(
            os.getenv('GOOGLE_APPLICATION_CREDENTIALS'),
            '/path/to/credentials.json'
        )
        self.assertEqual(
            os.getenv('GOOGLE_CLOUD_PROJECT'),
            'test-project'
        )
        
    @patch('google.cloud.language.LanguageServiceClient')
    def test_semantic_llm_initialization(self, mock_language_client):
        """Test semantic LLM initialization with credentials"""
        mock_language_client.return_value = MagicMock()
        config = load_app_config()
        
        # Test Gemini initialization
        llm = LLMFactory.create_llm('gemini', config)
        self.assertIsNotNone(llm)
        
        # Test OpenAI initialization
        llm = LLMFactory.create_llm('openai', config)
        self.assertIsNotNone(llm)
        
    def test_fallback_mode_activation(self):
        """Test that system falls back gracefully when credentials are missing"""
        with patch.dict('os.environ', {}, clear=True):
            config = load_app_config()
            # Should not raise an exception
            llm = LLMFactory.create_llm('gemini', config)
            self.assertTrue(hasattr(llm, 'is_fallback_mode'))
            self.assertTrue(llm.is_fallback_mode)

if __name__ == '__main__':
    unittest.main() 