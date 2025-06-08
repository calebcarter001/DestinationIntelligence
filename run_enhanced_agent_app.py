#!/usr/bin/env python3

import asyncio
import json
import logging
import os
import sys
import argparse
from datetime import datetime
import yaml
from tqdm.asyncio import tqdm as asyncio_tqdm
from tqdm import tqdm
from typing import Dict, Any
import warnings

# Filter the specific UserWarning from langchain_google_genai
warnings.filterwarnings("ignore", category=UserWarning, module="langchain_google_genai\.chat_models")

# Ensure the project root and src directory are in the Python path
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.config_loader import load_app_config
from src.core.llm_factory import LLMFactory
from src.core.enhanced_database_manager import EnhancedDatabaseManager
from src.core.content_intelligence_logic import ContentIntelligenceLogic
from src.tools.web_discovery_tools import DiscoverAndFetchContentTool
from src.tools.enhanced_content_analysis_tool import EnhancedContentAnalysisTool
from src.tools.enhanced_database_tools import StoreEnhancedDestinationInsightsTool
from src.tools.vectorize_processing_tool import ProcessContentWithVectorizeTool
from src.tools.chroma_interaction_tools import ChromaDBManager, AddChunksToChromaDBTool, SemanticSearchChromaDBTool
from src.tools.enhanced_theme_analysis_tool import EnhancedAnalyzeThemesFromEvidenceTool
from src.agents.enhanced_crewai_destination_analyst import create_enhanced_crewai_destination_analyst
from src.agents.base_agent import MessageBroker, AgentOrchestrator
from src.agents.specialized_agents import ValidationAgent, CulturalPerspectiveAgent, ContradictionDetectionAgent

def parse_arguments():
    """Parse command line arguments for LLM provider selection"""
    parser = argparse.ArgumentParser(description='Enhanced Destination Intelligence Application')
    parser.add_argument(
        '--provider', 
        choices=['gemini', 'openai'], 
        help='LLM provider to use (gemini or openai). If not specified, will auto-detect from available API keys.'
    )
    parser.add_argument(
        '--model', 
        help='Specific model name to use (overrides default for the provider)'
    )
    parser.add_argument(
        '--list-providers', 
        action='store_true',
        help='List available LLM providers based on configured API keys and exit'
    )
    return parser.parse_args()

# --- Directory Setup ---
LOGS_DIR = os.path.join(PROJECT_ROOT, "logs")
OUTPUTS_DIR = os.path.join(PROJECT_ROOT, "outputs")
COMPLETED_PROCESSING_DIR = os.path.join(OUTPUTS_DIR, "completed_processing")
FAILED_PROCESSING_DIR = os.path.join(OUTPUTS_DIR, "failed_processing")

os.makedirs(LOGS_DIR, exist_ok=True)
os.makedirs(COMPLETED_PROCESSING_DIR, exist_ok=True)
os.makedirs(FAILED_PROCESSING_DIR, exist_ok=True)

# --- Logging Setup ---
run_timestamp_for_log = datetime.now().strftime('%Y%m%d_%H%M%S')
main_log_file_path = os.path.join(LOGS_DIR, f"enhanced_app_run_{run_timestamp_for_log}.log")

# Configure root logger
app_logger = logging.getLogger()
app_logger.setLevel(logging.DEBUG)
if app_logger.hasHandlers(): 
    app_logger.handlers.clear()

formatter = logging.Formatter("%(asctime)s - [%(levelname)s] - %(name)s - %(module)s.%(funcName)s:%(lineno)d - %(message)s")

console_handler = logging.StreamHandler(sys.stdout)
console_handler.setFormatter(formatter)
console_handler.setLevel(logging.INFO)
app_logger.addHandler(console_handler)

# Main application file log (captures root logger at DEBUG)
file_handler_main = logging.FileHandler(main_log_file_path)
file_handler_main.setFormatter(formatter)
file_handler_main.setLevel(logging.INFO)
app_logger.addHandler(file_handler_main)

# --- Dedicated File Log for src.tools to ensure DEBUG from enhanced_theme_analysis_tool ---
tools_logger = logging.getLogger("src.tools")
tools_logger.setLevel(logging.DEBUG)
tools_logger.addHandler(file_handler_main) # Send to the same main log file
tools_logger.propagate = False # Prevent duplicate messages if root logger also handles it

# --- Dedicated File Log for ConfidenceScorer ---
pcs_log_path = os.path.join(LOGS_DIR, f"confidence_scorer_{run_timestamp_for_log}.log")
cs_file_handler = logging.FileHandler(pcs_log_path)
cs_file_handler.setFormatter(formatter)
cs_file_handler.setLevel(logging.DEBUG)
cs_logger = logging.getLogger("app.confidence_scorer") # Use the new prefixed name
cs_logger.setLevel(logging.DEBUG)
cs_logger.addHandler(cs_file_handler)
cs_logger.propagate = False # Do not propagate to root logger to avoid duplicate file entries

# --- Dedicated File Log for WebDiscoveryLogic ---
wd_log_path = os.path.join(LOGS_DIR, f"web_discovery_{run_timestamp_for_log}.log")
wd_file_handler = logging.FileHandler(wd_log_path)
wd_file_handler.setFormatter(formatter)
wd_file_handler.setLevel(logging.DEBUG)
wd_logger = logging.getLogger("app.web_discovery") # Use the new prefixed name
wd_logger.setLevel(logging.DEBUG)
wd_logger.addHandler(wd_file_handler)
wd_logger.propagate = False # Do not propagate to root logger

# Configure LangChain logger
langchain_log_file_path = os.path.join(LOGS_DIR, f"enhanced_langchain_agent_trace_{run_timestamp_for_log}.log")

langchain_loggers = [
    'langchain.agents.agent',
    'langchain.agents.agent_executor',
    'langchain_core.agents.agent',
    'langchain_community.agents.agent',
    'langchain.schema.runnable',
    'langchain_core.runnables',
    'langchain'
]

for logger_name in langchain_loggers:
    lc_logger = logging.getLogger(logger_name)
    lc_logger.setLevel(logging.DEBUG)
    lc_logger.propagate = False
    
    if not any(isinstance(h, logging.FileHandler) and h.baseFilename.endswith(f"enhanced_langchain_agent_trace_{run_timestamp_for_log}.log") for h in lc_logger.handlers):
        file_handler_lc = logging.FileHandler(langchain_log_file_path)
        file_handler_lc.setFormatter(formatter)
        lc_logger.addHandler(file_handler_lc)

logger = logging.getLogger(__name__)

async def run_enhanced_analysis_for_destination(enhanced_analyst, destination_name: str, processing_settings: Dict[str, Any]):
    """Execute enhanced destination analysis using CrewAI workflow orchestration."""
    logger.info(f"Starting enhanced CrewAI destination analysis for: {destination_name}")
    result = await enhanced_analyst.analyze_destination(destination_name, processing_settings)
    return result

async def main_agent_orchestration():
    # Parse command line arguments
    args = parse_arguments()
    
    # Initial print statements for immediate console feedback
    print("🚀 ENHANCED CrewAI-Inspired Destination Intelligence Application")
    print("🤖 Using specialized agents with enhanced evidence hierarchy, confidence scoring, and multi-agent validation")
    print("📊 Features: Evidence-based confidence, cultural perspective, temporal intelligence, JSON exports")
    print(f"📄 Main application log: {main_log_file_path}")
    print(f"📄 LangChain agent trace log: {langchain_log_file_path}")
    print("=" * 70)
    
    logger.info("🚀🚀🚀 ENHANCED CrewAI-Inspired Destination Intelligence Application Starting 🚀🚀🚀")
    logger.info("Using enhanced architecture with evidence hierarchy, confidence scoring, and multi-agent validation")

    run_start_time = datetime.now()
    overall_results = {
        "run_start_timestamp": run_start_time.isoformat(),
        "execution_method": "Enhanced CrewAI Direct Execution",
        "destinations_processed_details": [],
        "run_status": "Incomplete",
        "total_destinations": 0,
        "successful_destinations": 0,
        "failed_destinations": 0,
        "run_end_timestamp": None,
        "total_duration_seconds": None,
        "application_log_file": main_log_file_path,
        "agent_trace_log_file": langchain_log_file_path
    }

    try:
        app_config = load_app_config()
        config_load_success = True
    except Exception as e:
        logger.critical(f"Configuration loading failed: {e}", exc_info=True)
        overall_results["run_status"] = "Failed - Config Error"
        config_load_success = False

    if not config_load_success:
        results_filename = f"enhanced_config_error_run_{run_start_time.strftime('%Y%m%d_%H%M%S')}.json"
        results_filepath = os.path.join(FAILED_PROCESSING_DIR, results_filename)
        try:
            with open(results_filepath, 'w') as f:
                json.dump(overall_results, f, indent=4)
            logger.info(f"📜 Configuration error results saved to: {results_filepath}")
        except IOError:
            logger.error(f"❌ Could not save configuration error results to JSON.")
        return

    # Handle --list-providers argument
    if args.list_providers:
        available_providers = LLMFactory.get_available_providers(app_config)
        print("Available LLM Providers:")
        for provider in available_providers:
            api_keys = app_config.get("api_keys", {})
            llm_settings = app_config.get("llm_settings", {})
            if provider == "gemini":
                model = llm_settings.get("gemini_model_name", "gemini-1.5-flash-latest")
                print(f"  ✅ Gemini (model: {model})")
            elif provider == "openai":
                model = llm_settings.get("openai_model_name", "gpt-4o-mini")
                print(f"  ✅ OpenAI (model: {model})")
        
        if not available_providers:
            print("  ❌ No valid API keys found. Please check your .env file.")
        return

    # Determine LLM provider
    available_providers = LLMFactory.get_available_providers(app_config)
    
    if args.provider:
        if args.provider not in available_providers:
            logger.critical(f"Requested provider '{args.provider}' is not available. Available: {available_providers}")
            print(f"❌ Provider '{args.provider}' is not available. Available providers: {available_providers}")
            return
        selected_provider = args.provider
    else:
        if not available_providers:
            logger.critical("No valid API keys found for any LLM provider")
            print("❌ No valid API keys found. Please configure GEMINI_API_KEY or OPENAI_API_KEY in your .env file")
            return
        selected_provider = available_providers[0]  # Use first available
        logger.info(f"Auto-selected provider: {selected_provider} (from available: {available_providers})")

    # Override model if specified via command line
    if args.model:
        llm_settings = app_config.get("llm_settings", {})
        if selected_provider == "gemini":
            llm_settings["gemini_model_name"] = args.model
        elif selected_provider == "openai":
            llm_settings["openai_model_name"] = args.model
        logger.info(f"Model overridden via command line: {args.model}")

    # Create LLM instance
    try:
        llm = LLMFactory.create_llm(selected_provider, app_config)
        llm_settings = app_config.get("llm_settings", {})
        if selected_provider == "gemini":
            model_name = llm_settings.get("gemini_model_name", "gemini-1.5-flash-latest")
        else:
            model_name = llm_settings.get("openai_model_name", "gpt-4o-mini")
        
        logger.info(f"Using {selected_provider.upper()} LLM with model: {model_name}")
        print(f"🤖 Using {selected_provider.upper()} LLM with model: {model_name}")
    except Exception as e:
        logger.critical(f"Failed to initialize LLM: {e}")
        overall_results["run_status"] = f"Failed - LLM Initialization Error: {str(e)[:100]}"
        results_filename = f"enhanced_llm_error_run_{run_start_time.strftime('%Y%m%d_%H%M%S')}.json"
        results_filepath = os.path.join(FAILED_PROCESSING_DIR, results_filename)
        try:
            with open(results_filepath, 'w') as f:
                json.dump(overall_results, f, indent=4)
        except IOError:
            pass
        return

    # Continue with existing initialization...
    api_keys_config = app_config.get("api_keys", {})
    brave_api_key = api_keys_config.get("brave_search")
    processing_settings = app_config.get("processing_settings", {})

    # Validate Brave API key
    if not brave_api_key or not isinstance(brave_api_key, str) or "YOUR_BRAVE" in brave_api_key.upper():
        logger.critical("❌ Brave Search API key is missing or invalid")
        overall_results["run_status"] = "Failed - Brave API Key Invalid"
        results_filename = f"enhanced_api_key_error_run_{run_start_time.strftime('%Y%m%d_%H%M%S')}.json"
        results_filepath = os.path.join(FAILED_PROCESSING_DIR, results_filename)
        try:
            with open(results_filepath, 'w') as f:
                json.dump(overall_results, f, indent=4)
        except IOError:
            pass
        return

    # Enhanced database and storage setup
    try:
        insights_export_dir = os.path.join(PROJECT_ROOT, "destination_insights")
        logger.info(f"Initializing Enhanced Database Manager with JSON export to: {insights_export_dir}")
        
        db_path = app_config.get("database", {}).get("path", "enhanced_destination_intelligence.db")
        db_manager = EnhancedDatabaseManager(
            db_path=db_path,
            json_export_path=insights_export_dir,
            config=app_config  # Pass full app config for adaptive processing
        )

        # Initialize ChromaDB Manager
        chroma_db_config_path = app_config.get("database", {}).get("chroma_db_path", "./chroma_db")
        if not os.path.isabs(chroma_db_config_path):
            chroma_db_path = os.path.join(PROJECT_ROOT, chroma_db_config_path)
        else:
            chroma_db_path = chroma_db_config_path
        os.makedirs(chroma_db_path, exist_ok=True)
        logger.info(f"ChromaDB persistent path set to: {chroma_db_path}")
        
        default_chroma_collection = "destination_content_chunks"
        chroma_manager_instance = ChromaDBManager(db_path=chroma_db_path, collection_name=default_chroma_collection)

        # Initialize Core Logic Classes
        content_intelligence_logic = ContentIntelligenceLogic(config=processing_settings)
        
        # Initialize Multi-Agent System for validation
        logger.info("Initializing Multi-Agent System for validation...")
        message_broker = MessageBroker()
        orchestrator = AgentOrchestrator(message_broker)
        
        validation_agent = ValidationAgent("validation_agent", message_broker)
        cultural_agent = CulturalPerspectiveAgent("cultural_agent", message_broker)  
        contradiction_agent = ContradictionDetectionAgent("contradiction_agent", message_broker)
        
        message_broker.register_agent(validation_agent)
        message_broker.register_agent(cultural_agent)
        message_broker.register_agent(contradiction_agent)

        # Initialize Enhanced Tools with LLM
        tools = [
            DiscoverAndFetchContentTool(
                brave_api_key=brave_api_key,
                config=processing_settings
            ),
            ProcessContentWithVectorizeTool(
                config=processing_settings
            ),
            AddChunksToChromaDBTool(
                chroma_manager=chroma_manager_instance
            ),
            SemanticSearchChromaDBTool(
                chroma_manager=chroma_manager_instance
            ),
            EnhancedContentAnalysisTool(
                llm=llm
            ),
            EnhancedAnalyzeThemesFromEvidenceTool(
                agent_orchestrator=orchestrator,
                llm=llm
            ),
            StoreEnhancedDestinationInsightsTool(
                db_manager=db_manager
            )
        ]
        
        # Create Enhanced CrewAI analyst
        enhanced_analyst = create_enhanced_crewai_destination_analyst(
            llm=llm,  # Pass the LLM instance directly
            tools=tools
        )
        
        all_destinations = app_config.get("destinations_to_process", ["Paris, France"])
        max_dest = processing_settings.get("max_destinations_to_process", 1)
        destinations_to_process = all_destinations[:max_dest] if max_dest > 0 and max_dest <= len(all_destinations) else all_destinations
        
        logger.info(f"Enhanced agent will process {len(destinations_to_process)} destinations: {', '.join(destinations_to_process)}")
        overall_results["total_destinations"] = len(destinations_to_process)

        for dest_name in tqdm(destinations_to_process, desc="Processing Destinations", file=sys.stderr):
            destination_result_data = await run_enhanced_analysis_for_destination(enhanced_analyst, dest_name, processing_settings)
            
            # Extract only metadata for run summary (remove detailed insights to prevent duplication)
            destination_summary = {
                "status": destination_result_data.get("status"),
                "destination_name": destination_result_data.get("destination_name"),
                "execution_method": destination_result_data.get("execution_method"),
                "pages_processed": destination_result_data.get("pages_processed", 0),
                "chunks_created": destination_result_data.get("chunks_created", 0),
                "total_themes": destination_result_data.get("total_themes", 0),
                "validated_themes": destination_result_data.get("validated_themes", 0),
                "discovered_themes": destination_result_data.get("discovered_themes", 0),
                "priority_insights": destination_result_data.get("priority_insights", 0),
                "attractions_found": destination_result_data.get("attractions_found", 0),
                "hotels_found": destination_result_data.get("hotels_found", 0),
                "restaurants_found": destination_result_data.get("restaurants_found", 0),
                "execution_duration_seconds": destination_result_data.get("execution_duration_seconds", 0),
                "error": destination_result_data.get("error")  # Only include if there was an error
                # NOTE: Deliberately excluding "enhanced_insights" and "theme_analysis" to prevent duplication
                # These detailed insights are stored via the consolidated export system in destination_insights/
            }
            
            overall_results["destinations_processed_details"].append(destination_summary)
            
            if destination_result_data.get("status") == "Success":
                overall_results["successful_destinations"] += 1
            else:
                overall_results["failed_destinations"] += 1

            # Enhanced summary display
            if destination_result_data.get("status") == "Success":
                pages_processed = destination_result_data.get("pages_processed", 0)
                chunks_created = destination_result_data.get("chunks_created", 0)
                validated_themes = destination_result_data.get("validated_themes", 0)
                total_themes = destination_result_data.get("total_themes", 0)
                duration = destination_result_data.get("execution_duration_seconds", 0)
                
                logger.info(f"--- Enhanced Summary for {dest_name} ---")
                logger.info(f"✅ Status: SUCCESS")
                logger.info(f"📄 Pages processed: {pages_processed}")
                logger.info(f"📦 Chunks created: {chunks_created}")
                logger.info(f"🎯 Total themes: {total_themes} ({validated_themes} validated)")
                logger.info(f"⏱️  Execution time: {duration:.2f} seconds")
                logger.info(f"🤖 LLM Provider: {selected_provider.upper()}")
            else:
                error = destination_result_data.get("error", "Unknown error")
                logger.error(f"--- Enhanced Summary for {dest_name} ---")
                logger.error(f"❌ Status: FAILED")
                logger.error(f"🚨 Error: {error}")
                
            logger.info("=" * 50)

        # Determine overall status
        if overall_results["total_destinations"] > 0:
            if overall_results["successful_destinations"] == overall_results["total_destinations"]:
                overall_results["run_status"] = "Complete - All Successful"
            elif overall_results["successful_destinations"] > 0:
                overall_results["run_status"] = "Complete - Partial Success"
            else:
                overall_results["run_status"] = "Complete - All Failed"
        else:
            overall_results["run_status"] = "Complete - No Destinations To Process"
        
        logger.info(f"Enhanced Overall Run Status: {overall_results['run_status']}")

    except Exception as e:
        logger.critical(f"Critical error during enhanced agentic orchestration: {e}", exc_info=True)
        overall_results["run_status"] = f"Critical Failure: {str(e)[:150]}"
    finally:
        if 'db_manager' in locals():
            db_manager.close_db()
        
        run_end_time = datetime.now()
        overall_results["run_end_timestamp"] = run_end_time.isoformat()
        overall_results["total_duration_seconds"] = (run_end_time - run_start_time).total_seconds()
        
        status_suffix = overall_results["run_status"].replace(" ", "_").replace("-", "").replace("(", "").replace(")", "").lower()
        
        if "Success" in overall_results["run_status"] or "Partial" in overall_results["run_status"]:
            results_output_dir = COMPLETED_PROCESSING_DIR
        else:
            results_output_dir = FAILED_PROCESSING_DIR
            
        results_filename = f"enhanced_agent_run_{run_start_time.strftime('%Y%m%d_%H%M%S')}_{status_suffix}.json"
        results_filepath = os.path.join(results_output_dir, results_filename)
        try:
            os.makedirs(results_output_dir, exist_ok=True)
            with open(results_filepath, 'w') as f:
                # Convert any Pydantic objects to dict for JSON serialization
                def convert_to_serializable(obj):
                    if hasattr(obj, 'dict'):  # Pydantic object
                        return obj.dict()
                    elif isinstance(obj, dict):
                        return {k: convert_to_serializable(v) for k, v in obj.items()}
                    elif isinstance(obj, list):
                        return [convert_to_serializable(item) for item in obj]
                    else:
                        return obj
                
                serializable_results = convert_to_serializable(overall_results)
                json.dump(serializable_results, f, indent=4)
            logger.info(f"📜 Enhanced overall results saved to: {results_filepath}")
        except IOError as e:
            logger.error(f"❌ Could not save enhanced overall results to JSON: {e}")

if __name__ == "__main__":
    if os.name == 'nt' and hasattr(asyncio, 'WindowsSelectorEventLoopPolicy'):
        try:
            asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())
        except AttributeError:
            asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    asyncio.run(main_agent_orchestration()) 