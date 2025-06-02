import yaml
import os
import logging
from dotenv import load_dotenv
from typing import Dict, Any

logger = logging.getLogger(__name__)

def load_app_config() -> Dict[str, Any]:
    """Load application configuration from .env and config.yaml.
    API keys and critical LLM settings (like model name) are sourced from .env.
    Other operational settings are sourced from config.yaml.
    Environment variables from .env take precedence if a key exists in both (though this should be avoided).
    """
    project_root = os.path.join(os.path.dirname(__file__), '..')
    dotenv_path = os.path.join(project_root, '.env')
    
    if os.path.exists(dotenv_path):
        load_dotenv(dotenv_path=dotenv_path, override=True)
        logger.info(f".env file loaded from {dotenv_path}")
    else:
        logger.warning(f".env file not found at {dotenv_path}. Critical settings like API keys might be missing.")

    app_config: Dict[str, Any] = {}

    # Load general config from config.yaml first
    config_yaml_path = os.path.join(project_root, 'config.yaml')
    if os.path.exists(config_yaml_path):
        try:
            with open(config_yaml_path, 'r') as f:
                yaml_config = yaml.safe_load(f)
            if yaml_config:
                app_config.update(yaml_config)
            logger.info(f"config.yaml loaded from {config_yaml_path}")
        except yaml.YAMLError as e:
            logger.error(f"Error parsing config.yaml: {e}")
    else:
        logger.warning(f"config.yaml not found at {config_yaml_path}. Proceeding with .env and defaults.")

    # API keys are exclusively from .env (loaded into os.environ by load_dotenv)
    app_config["api_keys"] = {
        "brave_search": os.getenv("BRAVE_SEARCH_API_KEY"),
        "gemini_api_key": os.getenv("GEMINI_API_KEY"),
        "jina_api_key": os.getenv("JINA_API_KEY")
    }
    
    # LLM settings, model name from .env, with a fallback default if not in .env OR config.yaml
    # Initialize llm_settings if not present from config.yaml
    if "llm_settings" not in app_config:
        app_config["llm_settings"] = {}
    
    # Get GEMINI_MODEL_NAME from environment (set by .env), overriding config.yaml if it was there.
    gemini_model_from_env = os.getenv("GEMINI_MODEL_NAME")
    if gemini_model_from_env:
        app_config["llm_settings"]["gemini_model_name"] = gemini_model_from_env
        logger.info(f"GEMINI_MODEL_NAME loaded from .env: {gemini_model_from_env}")
    elif not app_config.get("llm_settings", {}).get("gemini_model_name"):
        # If not in .env AND not in config.yaml (or config.yaml had no llm_settings)
        default_model = "gemini-1.5-flash-latest"
        app_config["llm_settings"]["gemini_model_name"] = default_model
        logger.warning(f"GEMINI_MODEL_NAME not found in .env or config.yaml, defaulting to {default_model}")
    # If it was in config.yaml but not .env, it remains as loaded from yaml.

    return app_config 