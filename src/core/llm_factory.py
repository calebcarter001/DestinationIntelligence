"""
LLM Factory for creating Gemini or OpenAI models based on configuration
"""

import logging
from typing import Dict, Any, Union
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI

logger = logging.getLogger(__name__)

class LLMFactory:
    """Factory for creating LLM instances based on provider"""
    
    @staticmethod
    def create_llm(provider: str, config: Dict[str, Any]) -> Union[ChatGoogleGenerativeAI, ChatOpenAI]:
        """
        Create an LLM instance based on the provider
        
        Args:
            provider: 'gemini' or 'openai'
            config: Configuration dictionary with API keys and model names
            
        Returns:
            Configured LLM instance
        """
        provider = provider.lower()
        api_keys = config.get("api_keys", {})
        llm_settings = config.get("llm_settings", {})
        
        if provider == "gemini":
            return LLMFactory._create_gemini_llm(api_keys, llm_settings)
        elif provider == "openai":
            return LLMFactory._create_openai_llm(api_keys, llm_settings)
        else:
            raise ValueError(f"Unsupported LLM provider: {provider}. Supported: 'gemini', 'openai'")
    
    @staticmethod
    def _create_gemini_llm(api_keys: Dict[str, Any], llm_settings: Dict[str, Any]) -> ChatGoogleGenerativeAI:
        """Create a Gemini LLM instance"""
        gemini_api_key = api_keys.get("gemini_api_key")
        if not gemini_api_key:
            raise ValueError("Gemini API key not found in configuration")
        
        model_name = llm_settings.get("gemini_model_name", "gemini-1.5-flash-latest")
        
        logger.info(f"Initializing Gemini LLM with model: {model_name}")
        
        return ChatGoogleGenerativeAI(
            model=model_name,
            google_api_key=gemini_api_key,
            temperature=0.3,
            max_tokens=4096,
            timeout=60,
            max_retries=3
        )
    
    @staticmethod
    def _create_openai_llm(api_keys: Dict[str, Any], llm_settings: Dict[str, Any]) -> ChatOpenAI:
        """Create an OpenAI LLM instance"""
        openai_api_key = api_keys.get("openai_api_key")
        if not openai_api_key:
            raise ValueError("OpenAI API key not found in configuration")
        
        model_name = llm_settings.get("openai_model_name", "gpt-4o-mini")
        
        logger.info(f"Initializing OpenAI LLM with model: {model_name}")
        
        return ChatOpenAI(
            model_name=model_name,
            openai_api_key=openai_api_key,
            temperature=0.3,
            max_tokens=4096,
            timeout=60,
            max_retries=3
        )
    
    @staticmethod
    def validate_api_key(provider: str, api_key: str) -> bool:
        """
        Validate API key format for the given provider
        
        Args:
            provider: 'gemini' or 'openai'
            api_key: The API key to validate
            
        Returns:
            True if key appears valid, False otherwise
        """
        if not api_key or not isinstance(api_key, str):
            return False
        
        provider = provider.lower()
        
        # Check for placeholder values
        placeholder_indicators = [
            "YOUR_", "YOURKEY", "PLACEHOLDER", "REPLACE", "ENTER"
        ]
        
        if any(indicator in api_key.upper() for indicator in placeholder_indicators):
            return False
        
        if provider == "gemini":
            # Gemini keys typically start with certain patterns
            return len(api_key) > 30 and not api_key.isdigit()
        elif provider == "openai":
            # OpenAI keys typically start with "sk-"
            return api_key.startswith("sk-") and len(api_key) > 40
        
        return True
    
    @staticmethod
    def get_available_providers(config: Dict[str, Any]) -> list:
        """
        Get list of available providers based on valid API keys
        
        Args:
            config: Configuration dictionary
            
        Returns:
            List of available provider names
        """
        available = []
        api_keys = config.get("api_keys", {})
        
        if LLMFactory.validate_api_key("gemini", api_keys.get("gemini_api_key")):
            available.append("gemini")
        
        if LLMFactory.validate_api_key("openai", api_keys.get("openai_api_key")):
            available.append("openai")
        
        return available 