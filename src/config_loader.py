import yaml
import os
import logging
from dotenv import load_dotenv
from typing import Dict, Any
from pathlib import Path

logger = logging.getLogger(__name__)

def load_app_config() -> Dict[str, Any]:
    """
    Load configuration from .env and config.yaml files with enhanced OpenAI support.
    """
    # Load .env first
    env_path = Path(__file__).parent.parent / ".env"
    if env_path.exists():
        load_dotenv(env_path)
        logger.info(f".env file loaded from {env_path}")
    else:
        logger.warning(f"No .env file found at {env_path}")

    # Load config.yaml from config directory
    config_path = Path(__file__).parent.parent / "config" / "config.yaml"
    app_config = {}
    
    if config_path.exists():
        try:
            with open(config_path, 'r') as f:
                app_config = yaml.safe_load(f)
            logger.info(f"config.yaml loaded from {config_path}")
        except yaml.YAMLError as e:
            logger.error(f"Error parsing config.yaml: {e}")
            raise
        except Exception as e:
            logger.error(f"Error loading config.yaml: {e}")
            raise
    else:
        logger.warning(f"No config.yaml found at {config_path}")
    
    # Initialize api_keys section if not present
    if "api_keys" not in app_config:
        app_config["api_keys"] = {}
    
    # Override/add API keys from environment variables
    env_api_keys = {
        "gemini_api_key": os.getenv("GEMINI_API_KEY"),
        "brave_search": os.getenv("BRAVE_SEARCH_API_KEY"),
        "openai_api_key": os.getenv("OPENAI_API_KEY"),  # Added OpenAI support
        "jina_api_key": os.getenv("JINA_API_KEY")
    }
    
    for key, value in env_api_keys.items():
        if value:
            app_config["api_keys"][key] = value
    
    # Override LLM settings from environment variables
    if "llm_settings" not in app_config:
        app_config["llm_settings"] = {}
    
    # Gemini model settings
    gemini_model_env = os.getenv("GEMINI_MODEL_NAME")
    if gemini_model_env:
        app_config["llm_settings"]["gemini_model_name"] = gemini_model_env
        logger.info(f"GEMINI_MODEL_NAME loaded from .env: {gemini_model_env}")
    
    # OpenAI model settings
    openai_model_env = os.getenv("OPENAI_MODEL_NAME")
    if openai_model_env:
        app_config["llm_settings"]["openai_model_name"] = openai_model_env
        logger.info(f"OPENAI_MODEL_NAME loaded from .env: {openai_model_env}")
    
    # Set default models if not specified
    if "gemini_model_name" not in app_config["llm_settings"]:
        app_config["llm_settings"]["gemini_model_name"] = "gemini-1.5-flash-latest"
    
    if "openai_model_name" not in app_config["llm_settings"]:
        app_config["llm_settings"]["openai_model_name"] = "gpt-4o-mini"
    
    # LLM provider preference
    llm_provider_env = os.getenv("LLM_PROVIDER")
    if llm_provider_env:
        app_config["llm_settings"]["provider"] = llm_provider_env.lower()
        logger.info(f"LLM_PROVIDER loaded from .env: {llm_provider_env}")
    
    return app_config 