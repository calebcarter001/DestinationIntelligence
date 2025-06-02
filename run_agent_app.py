import asyncio
import json
import logging
import os
import sys # Add sys for path manipulation
from datetime import datetime
import yaml
from tqdm.asyncio import tqdm as asyncio_tqdm # For async iteration
from tqdm import tqdm # For sync iteration (if any)
from typing import Dict, Any
import warnings # Import warnings module

# Filter the specific UserWarning from langchain_google_genai
warnings.filterwarnings("ignore", category=UserWarning, module="langchain_google_genai\.chat_models")

# Ensure the project root and src directory are in the Python path
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.config_loader import load_app_config
from src.core.database_manager import DatabaseManager
from src.core.content_intelligence_logic import ContentIntelligenceLogic
from src.tools.web_discovery_tools import DiscoverAndFetchContentTool
from src.tools.content_intelligence_tools import AnalyzeContentForThemesTool
from src.tools.database_tools import StoreDestinationInsightsTool
from src.tools.vectorize_processing_tool import ProcessContentWithVectorizeTool
from src.tools.chroma_interaction_tools import ChromaDBManager, AddChunksToChromaDBTool, SemanticSearchChromaDBTool
from src.tools.theme_analysis_tool import AnalyzeThemesFromEvidenceTool
from src.agents.destination_analyst_agent import create_destination_analyst_agent
from src.agents.crewai_destination_analyst import create_crewai_destination_analyst

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
main_log_file_path = os.path.join(LOGS_DIR, f"app_run_{run_timestamp_for_log}.log") # Renamed for clarity

# Configure root logger (for our application code)
app_logger = logging.getLogger() 
app_logger.setLevel(logging.INFO)
if app_logger.hasHandlers(): app_logger.handlers.clear()

formatter = logging.Formatter("%(asctime)s - [%(levelname)s] - %(name)s - %(module)s.%(funcName)s:%(lineno)d - %(message)s")

console_handler = logging.StreamHandler(sys.stdout)
console_handler.setFormatter(formatter)
console_handler.setLevel(logging.INFO) 
app_logger.addHandler(console_handler)

file_handler_main = logging.FileHandler(main_log_file_path)
file_handler_main.setFormatter(formatter)
file_handler_main.setLevel(logging.INFO)
app_logger.addHandler(file_handler_main)

# Configure LangChain's specific logger for verbose agent trace
# This captures the AgentExecutor verbose output to a separate file.
langchain_log_file_path = os.path.join(LOGS_DIR, f"langchain_agent_trace_{run_timestamp_for_log}.log")

# Configure multiple LangChain loggers to capture agent execution
langchain_loggers = [
    'langchain.agents.agent',
    'langchain.agents.agent_executor', 
    'langchain_core.agents.agent',
    'langchain_community.agents.agent',
    'langchain.schema.runnable',
    'langchain_core.runnables',
    'langchain'  # fallback
]

for logger_name in langchain_loggers:
    lc_logger = logging.getLogger(logger_name)
    lc_logger.setLevel(logging.DEBUG)  # Use DEBUG for maximum detail
    lc_logger.propagate = False  # Keep separate from main log
    
    # Add file handler if not already added
    if not any(isinstance(h, logging.FileHandler) and h.baseFilename.endswith(f"langchain_agent_trace_{run_timestamp_for_log}.log") for h in lc_logger.handlers):
        file_handler_lc = logging.FileHandler(langchain_log_file_path)
        file_handler_lc.setFormatter(formatter)
        lc_logger.addHandler(file_handler_lc)

# Also configure the root langchain logger as a fallback
langchain_root_logger = logging.getLogger('langchain')
langchain_root_logger.setLevel(logging.DEBUG)
if not any(isinstance(h, logging.FileHandler) and h.baseFilename.endswith(f"langchain_agent_trace_{run_timestamp_for_log}.log") for h in langchain_root_logger.handlers):
    file_handler_lc_root = logging.FileHandler(langchain_log_file_path)
    file_handler_lc_root.setFormatter(formatter) 
    langchain_root_logger.addHandler(file_handler_lc_root)

logger = logging.getLogger(__name__) # Logger for this specific file (run_agent_app.py)

async def run_analysis_for_destination(crewai_analyst, destination_name: str, processing_settings: Dict[str, Any]):
    """
    Execute destination analysis using CrewAI workflow orchestration.
    """
    logger.info(f"Starting CrewAI destination analysis for: {destination_name}")
    
    # Use CrewAI's analyze_destination method (now async)
    result = await crewai_analyst.analyze_destination(destination_name, processing_settings)
    
    return result

async def main_agent_orchestration():
    # Initial print statements for immediate console feedback
    print("🚀 CrewAI-Inspired Destination Intelligence Application")
    print("🤖 Using specialized agents with direct tool execution for reliable workflows")
    print(f"📄 Main application log: {main_log_file_path}")
    print(f"📄 LangChain agent trace log: {langchain_log_file_path}") # Keeping for tool logs
    print("=" * 70)
    
    logger.info("🚀🚀🚀 CrewAI-Inspired Destination Intelligence Application Starting 🚀🚀🚀")
    logger.info("Using specialized agents with direct tool execution for reliable workflow execution")

    run_start_time = datetime.now()
    overall_results = {
        "run_start_timestamp": run_start_time.isoformat(),
        "execution_method": "CrewAI-Inspired Direct Execution",
        "destinations_processed_details": [], "run_status": "Incomplete",
        "total_destinations": 0, "successful_destinations": 0, "failed_destinations": 0,
        "run_end_timestamp": None, "total_duration_seconds": None,
        "application_log_file": main_log_file_path,
        "agent_trace_log_file": langchain_log_file_path
    }

    try:
        app_config = load_app_config() # This now handles .env and config.yaml
        config_load_success = True 
    except FileNotFoundError as e: 
        logger.critical(f"Configuration loading failed: {e}")
        overall_results["run_status"] = "Failed - Config File Missing"
        config_load_success = False
    except yaml.YAMLError as e:
        logger.critical(f"Error parsing config.yaml: {e}")
        overall_results["run_status"] = "Failed - Config YAML Error"
        config_load_success = False
    except Exception as e:
        logger.critical(f"Unexpected error loading configuration: {e}", exc_info=True)
        overall_results["run_status"] = "Failed - Unknown Config Error"
        config_load_success = False

    results_output_dir = FAILED_PROCESSING_DIR
    if not config_load_success:
        results_filename = f"config_error_run_{run_start_time.strftime('%Y%m%d_%H%M%S')}.json"
        results_filepath = os.path.join(results_output_dir, results_filename)
        try:
            with open(results_filepath, 'w') as f: json.dump(overall_results, f, indent=4)
            logger.info(f"📜 Configuration error results saved to: {results_filepath}")
        except IOError: logger.error(f"❌ Could not save configuration error results to JSON.")
        return

    # Retrieve settings from the potentially merged app_config
    api_keys_config = app_config.get("api_keys", {})
    gemini_api_key = api_keys_config.get("gemini_api_key")
    brave_api_key = api_keys_config.get("brave_search")
    # jina_api_key = api_keys_config.get("jina_api_key") # No longer needed from config if tool is keyless
    vectorize_api_endpoint = app_config.get("processing_settings", {}).get("web_discovery", {}).get("vectorize_api_endpoint", "")
    vectorize_api_key = app_config.get("processing_settings", {}).get("web_discovery", {}).get("vectorize_api_key", "")
    
    # llm_settings should be loaded from config.yaml, potentially overridden by .env if defined there by load_app_config
    llm_settings_config = app_config.get("llm_settings", {})
    gemini_model_name = llm_settings_config.get("gemini_model_name", "gemini-1.5-flash-latest") # Default here if still not found
    
    processing_settings = app_config.get("processing_settings", {})

    # --- API Key Validation --- 
    gemini_key_valid = True
    if not gemini_api_key:
        logger.critical("❌ Gemini API key (GEMINI_API_KEY in .env) is missing.")
        gemini_key_valid = False
    elif not isinstance(gemini_api_key, str):
        logger.critical("❌ Gemini API key (GEMINI_API_KEY in .env) is not a string.")
        gemini_key_valid = False
    elif "YOUR_GEMINI_API_KEY_HERE" in gemini_api_key.upper():
        logger.critical("❌ Gemini API key (GEMINI_API_KEY in .env) appears to be a placeholder (YOUR_GEMINI_API_KEY_HERE).")
        gemini_key_valid = False
    elif "YOUR_ACTUAL_GEMINI_KEY_GOES_HERE" in gemini_api_key.upper():
        logger.critical("❌ Gemini API key (GEMINI_API_KEY in .env) appears to be a placeholder (YOUR_ACTUAL_GEMINI_KEY_GOES_HERE).")
        gemini_key_valid = False
    elif "YOURKEY" in gemini_api_key.upper() and len(gemini_api_key) < 30: # Generic short placeholder check
        logger.critical("❌ Gemini API key (GEMINI_API_KEY in .env) appears to be a generic placeholder.")
        gemini_key_valid = False

    brave_key_valid = True
    if not brave_api_key:
        logger.critical("❌ Brave Search API key (BRAVE_SEARCH_API_KEY in .env) is missing.")
        brave_key_valid = False
    elif not isinstance(brave_api_key, str):
        logger.critical("❌ Brave Search API key (BRAVE_SEARCH_API_KEY in .env) is not a string.")
        brave_key_valid = False
    elif "YOUR_BRAVE_SEARCH_API_KEY_HERE" in brave_api_key.upper():
        logger.critical("❌ Brave Search API key (BRAVE_SEARCH_API_KEY in .env) appears to be a placeholder (YOUR_BRAVE_SEARCH_API_KEY_HERE).")
        brave_key_valid = False
    elif "YOUR_ACTUAL_BRAVE_KEY_GOES_HERE" in brave_api_key.upper():
        logger.critical("❌ Brave Search API key (BRAVE_SEARCH_API_KEY in .env) appears to be a placeholder (YOUR_ACTUAL_BRAVE_KEY_GOES_HERE).")
        brave_key_valid = False
    elif "YOURKEY" in brave_api_key.upper() and len(brave_api_key) < 20: # Generic short placeholder check
        logger.critical("❌ Brave Search API key (BRAVE_SEARCH_API_KEY in .env) appears to be a generic placeholder.")
        brave_key_valid = False

    api_keys_ok_for_run = True
    if not gemini_key_valid and not brave_key_valid:
        overall_results["run_status"] = "Failed - Multiple API Keys Invalid"
        api_keys_ok_for_run = False
    elif not gemini_key_valid:
        overall_results["run_status"] = "Failed - Gemini API Key Invalid"
        api_keys_ok_for_run = False
    elif not brave_key_valid:
        overall_results["run_status"] = "Failed - Brave API Key Invalid"
        api_keys_ok_for_run = False
    # --- End API Key Validation ---

    logger.info(f"Using Gemini model: {gemini_model_name}")

    if not api_keys_ok_for_run:
        print("Please check your .env file for API key issues as logged above.")
        results_filename = f"api_key_error_run_{run_start_time.strftime('%Y%m%d_%H%M%S')}.json"
        results_filepath = os.path.join(results_output_dir, results_filename) 
        try:
            with open(results_filepath, 'w') as f: json.dump(overall_results, f, indent=4)
            logger.info(f"📜 API key error results saved to: {results_filepath}")
        except IOError: logger.error(f"❌ Could not save API key error results to JSON.")
        return
    
    db_manager = None
    chroma_manager_instance = None # For ChromaDB
    crewai_analyst = None
    try:
        # Initialize SQLite DB Manager
        db_path = app_config.get("database", {}).get("path", "real_destination_intelligence.db")
        db_manager = DatabaseManager(db_path=db_path)

        # Initialize ChromaDB Manager
        chroma_db_config_path = app_config.get("database", {}).get("chroma_db_path", "./chroma_data")
        # Ensure the path is absolute or correctly relative to PROJECT_ROOT
        if not os.path.isabs(chroma_db_config_path):
            chroma_db_path = os.path.join(PROJECT_ROOT, chroma_db_config_path)
        else:
            chroma_db_path = chroma_db_config_path
        os.makedirs(chroma_db_path, exist_ok=True) # Ensure directory exists
        logger.info(f"ChromaDB persistent path set to: {chroma_db_path}")
        # You might want to make collection name configurable too via config.yaml
        default_chroma_collection = "destination_content_chunks"
        chroma_manager_instance = ChromaDBManager(db_path=chroma_db_path, collection_name=default_chroma_collection)

        # Initialize Core Logic Classes
        content_intelligence_logic = ContentIntelligenceLogic(config=processing_settings) # Pass full processing_settings for now
        
        # Initialize Tools
        tools = [
            DiscoverAndFetchContentTool(
                brave_api_key=brave_api_key, 
                config=processing_settings
            ),
            ProcessContentWithVectorizeTool(config=processing_settings), # Pass config for potential API keys
            AddChunksToChromaDBTool(chroma_manager=chroma_manager_instance),
            SemanticSearchChromaDBTool(chroma_manager=chroma_manager_instance),
            AnalyzeThemesFromEvidenceTool(
                content_intelligence_logic=content_intelligence_logic 
                # config will be passed by the agent based on its prompt
            ),
            StoreDestinationInsightsTool(db_manager=db_manager)
        ]
        
        # Create CrewAI analyst instead of agent executor
        crewai_analyst = create_crewai_destination_analyst(
            gemini_api_key=gemini_api_key, 
            gemini_model_name=gemini_model_name, 
            tools=tools
        )
        all_destinations = app_config.get("destinations_to_process", ["Paris, France"])
        max_dest = processing_settings.get("max_destinations_to_process", 1)
        destinations_to_process = all_destinations[:max_dest] if max_dest > 0 and max_dest <= len(all_destinations) else (all_destinations if max_dest == 0 or max_dest is None else [])
        if not destinations_to_process and all_destinations :
             logger.warning(f"max_destinations_to_process ({max_dest}) is invalid or exceeds list length. Processing all {len(all_destinations)} destinations.")
             destinations_to_process = all_destinations
        elif max_dest > 0 : logger.info(f"Processing the first {len(destinations_to_process)} of {len(all_destinations)} destinations based on max_destinations_to_process: {max_dest}.")
        
        logger.info(f"Agent will process {len(destinations_to_process)} destinations: {', '.join(destinations_to_process)}")
        overall_results["total_destinations"] = len(destinations_to_process)

        for dest_name in tqdm(destinations_to_process, desc="Processing Destinations", file=sys.stderr, bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}{postfix}]'):
            destination_result_data = await run_analysis_for_destination(crewai_analyst, dest_name, processing_settings)
            overall_results["destinations_processed_details"].append(destination_result_data)
            
            # Refined success/failure counting based on more nuanced status
            if destination_result_data.get("status") == "Success":
                overall_results["successful_destinations"] +=1
            elif destination_result_data.get("status") == "Completed with Agent-Reported Issues":
                # Count as successful for now, but could be a separate category
                overall_results["successful_destinations"] +=1 
                logger.warning(f"Destination {dest_name} completed with agent-reported issues.")
            else: # Any other status is a failure
                overall_results["failed_destinations"] +=1

            # Enhanced summary display for the processed destination
            if destination_result_data.get("status") == "Success":
                agent_output = destination_result_data.get("agent_output", "")
                pages_processed = destination_result_data.get("pages_processed", 0)
                chunks_created = destination_result_data.get("chunks_created", 0)
                total_themes = destination_result_data.get("total_themes", 0)
                validated_themes = destination_result_data.get("validated_themes", 0)
                discovered_themes = destination_result_data.get("discovered_themes", 0)
                duration = destination_result_data.get("execution_duration_seconds", 0)
                execution_method = destination_result_data.get("execution_method", "CrewAI")
                
                # Enhanced insights metrics
                attractions_found = destination_result_data.get("attractions_found", 0)
                hotels_found = destination_result_data.get("hotels_found", 0)
                restaurants_found = destination_result_data.get("restaurants_found", 0)
                activities_found = destination_result_data.get("activities_found", 0)
                neighborhoods_found = destination_result_data.get("neighborhoods_found", 0)
                practical_info_found = destination_result_data.get("practical_info_found", 0)
                destination_summary = destination_result_data.get("destination_summary", "")
                enhanced_insights = destination_result_data.get("enhanced_insights", {})
                
                logger.info(f"--- Summary for {dest_name} ---")
                logger.info(f"✅ Status: SUCCESS ({execution_method})")
                logger.info(f"📄 Pages processed: {pages_processed}")
                logger.info(f"📦 Chunks created: {chunks_created}")
                logger.info(f"🎯 Total themes discovered: {total_themes} ({validated_themes} validated, {discovered_themes} new)")
                
                # Enhanced insights summary
                logger.info(f"🏛️  Attractions found: {attractions_found}")
                logger.info(f"🏨 Hotels found: {hotels_found}")
                logger.info(f"🍽️  Restaurants found: {restaurants_found}")
                logger.info(f"🎯 Activities found: {activities_found}")
                logger.info(f"🏘️  Neighborhoods found: {neighborhoods_found}")
                logger.info(f"ℹ️  Practical info items: {practical_info_found}")
                
                logger.info(f"⏱️  Execution time: {duration:.2f} seconds")
                logger.info(f"📝 Destination Summary: {destination_summary[:200]}..." if len(destination_summary) > 200 else f"📝 Destination Summary: {destination_summary}")
                
                # Show sample enhanced insights
                if enhanced_insights:
                    logger.info("🌟 Sample Enhanced Insights:")
                    
                    if enhanced_insights.get("attractions"):
                        top_attraction = enhanced_insights["attractions"][0]
                        logger.info(f"   🏛️  Top Attraction: {top_attraction.get('name', 'N/A')} - {top_attraction.get('description', 'N/A')[:100]}...")
                    
                    if enhanced_insights.get("hotels"):
                        top_hotel = enhanced_insights["hotels"][0]
                        logger.info(f"   🏨 Featured Hotel: {top_hotel.get('name', 'N/A')} - {top_hotel.get('description', 'N/A')[:100]}...")
                    
                    if enhanced_insights.get("restaurants"):
                        top_restaurant = enhanced_insights["restaurants"][0]
                        logger.info(f"   🍽️  Featured Restaurant: {top_restaurant.get('name', 'N/A')} - {top_restaurant.get('description', 'N/A')[:100]}...")
                    
                    if enhanced_insights.get("activities"):
                        top_activity = enhanced_insights["activities"][0]
                        logger.info(f"   🎯 Featured Activity: {top_activity.get('name', 'N/A')} - {top_activity.get('description', 'N/A')[:100]}...")
                
                logger.info(f"✅ All 7 workflow steps completed successfully (including enhanced analysis)")
            else:
                # Handle failed destinations
                status = destination_result_data.get("status", "Unknown")
                error = destination_result_data.get("error", "No error details available")
                execution_method = destination_result_data.get("execution_method", "CrewAI")
                
                logger.error(f"--- Summary for {dest_name} ---")
                logger.error(f"❌ Status: {status} ({execution_method})")
                logger.error(f"🚨 Error: {error}")
                
            logger.info("=" * 50)

        if overall_results["total_destinations"] > 0:
            if overall_results["successful_destinations"] == overall_results["total_destinations"]: overall_results["run_status"] = "Complete - All Successful"
            elif overall_results["successful_destinations"] > 0: overall_results["run_status"] = "Complete - Partial Success"
            else: overall_results["run_status"] = "Complete - All Failed"
        else: overall_results["run_status"] = "Complete - No Destinations To Process"
        logger.info(f"Overall Run Status: {overall_results['run_status']}")

    except Exception as e:
        logger.critical(f"Critical error during agentic orchestration: {e}", exc_info=True)
        overall_results["run_status"] = f"Critical Failure: {str(e)[:150]}"
    finally:
        if db_manager: db_manager.close_db()
        # ChromaDB client in ChromaDBManager doesn't have an explicit close, 
        # relies on PersistentClient handling shutdown or garbage collection.
        run_end_time = datetime.now()
        overall_results["run_end_timestamp"] = run_end_time.isoformat()
        overall_results["total_duration_seconds"] = (run_end_time - run_start_time).total_seconds()
        status_suffix = overall_results["run_status"].replace(" ", "_").replace("-", "").replace("(", "").replace(")", "").lower()
        
        if "Success" in overall_results["run_status"] or "Partial" in overall_results["run_status"] :
            results_output_dir = COMPLETED_PROCESSING_DIR
        else:
            results_output_dir = FAILED_PROCESSING_DIR
            
        results_filename = f"agent_run_{run_start_time.strftime('%Y%m%d_%H%M%S')}_{status_suffix}.json"
        results_filepath = os.path.join(results_output_dir, results_filename)
        try:
            os.makedirs(results_output_dir, exist_ok=True) # Ensure directory exists
            with open(results_filepath, 'w') as f: json.dump(overall_results, f, indent=4)
            logger.info(f"📜 Overall results saved to: {results_filepath}")
        except IOError as e: logger.error(f"❌ Could not save overall results to JSON: {e}")
        except Exception as e_json: logger.error(f"❌ Error saving JSON results to {results_filename}: {e_json}")

if __name__ == "__main__":
    if os.name == 'nt' and hasattr(asyncio, 'WindowsSelectorEventLoopPolicy'):
        try: asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())
        except AttributeError: asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    asyncio.run(main_agent_orchestration()) 