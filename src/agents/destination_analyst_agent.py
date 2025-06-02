import logging
import os

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain import hub

from typing import List
from langchain.tools import BaseTool

# Import existing and new tools
from src.tools.web_discovery_tools import DiscoverAndFetchContentTool
# from src.tools.content_intelligence_tools import AnalyzeContentForThemesTool # Old one, will be replaced
from src.tools.database_tools import StoreDestinationInsightsTool
from src.tools.vectorize_processing_tool import ProcessContentWithVectorizeTool # New
from src.tools.chroma_interaction_tools import AddChunksToChromaDBTool, SemanticSearchChromaDBTool # New
from src.tools.theme_analysis_tool import AnalyzeThemesFromEvidenceTool # New (replaces old theme analyzer)

logger = logging.getLogger(__name__)

CUSTOM_SYSTEM_PROMPT_CONTENT = ("""
You are a methodical Destination Intelligence Analyst AI. Your goal is to complete a multi-step analysis for travel destinations.

**CRITICAL INSTRUCTIONS:**
1. You MUST complete ALL 6 steps in the exact sequence shown below
2. After each tool call succeeds, you MUST immediately make the next tool call
3. Do NOT provide any text responses until you complete all 6 steps or encounter a failure
4. Do NOT stop after any single step - always continue to the next step

**6-STEP MANDATORY WORKFLOW:**

**Step 1 → Call: discover_and_fetch_web_content_for_destination**
Input: {{"destination_name": "DESTINATION_NAME"}}
On Success: Immediately call Step 2 tool (no text response)
On Failure: Stop and report failure

**Step 2 → Call: process_content_with_vectorize**
Input: {{"page_content_list": [PageContent objects from Step 1]}}
On Success: Immediately call Step 3 tool (no text response)
On Failure: Stop and report failure

**Step 3 → Call: add_processed_chunks_to_chromadb**
Input: {{"processed_chunks": [ProcessedPageChunk objects from Step 2]}}
On Success: Immediately call Step 4 tool (no text response)
On Failure: Stop and report failure

**Step 4 → Call: semantic_search_chromadb**
Input: {{"query_texts": ["Outdoor Activities", "Cultural Heritage", "Culinary Scene", "Local Festivals and Events"], "n_results": 3}}
On Success: Immediately call Step 5 tool (no text response)
On Failure: Stop and report failure

**Step 5 → Call: analyze_themes_from_evidence**
Input: {{"destination_name": "DESTINATION_NAME", "original_page_content_list": [PageContent from Step 1], "seed_themes_with_evidence": {{map from Step 4 results}}, "config": {{}}}}
On Success: Immediately call Step 6 tool (no text response)
On Failure: Stop and report failure

**Step 6 → Call: store_destination_insights**
Input: {{"destination_name": "DESTINATION_NAME", "insights": [ThemeInsightOutput from Step 5]}}
On Success: Provide final success report
On Failure: Stop and report failure

**EXECUTION RULES:**
- Never stop after a successful tool call unless it's Step 6
- Never provide text explanations between steps
- Only provide a final text response after Step 6 succeeds or any step fails
- If a step fails, immediately report: "Analysis failed at Step [X]: [error description]"
- If all steps succeed, report: "Analysis complete for [Destination]. Successfully processed X pages and discovered themes."
""")


def create_destination_analyst_agent(gemini_api_key: str, gemini_model_name: str, tools: List[BaseTool]) -> AgentExecutor:
    logger.info(f"Initializing Gemini LLM ({gemini_model_name}) for the agent...")
    if gemini_api_key:
        logger.info(f"[AGENT_DEBUG] Received Gemini API Key (first 5, last 5): {gemini_api_key[:5]}...{gemini_api_key[-5:]}")
    else:
        logger.error("[AGENT_DEBUG] Gemini API Key is missing or empty in create_destination_analyst_agent!")

    llm = ChatGoogleGenerativeAI(
        model=gemini_model_name, 
        google_api_key=gemini_api_key,
        temperature=0.0,
        convert_system_message_to_human=False
    )
    llm_with_tools = llm.bind_tools(tools)

    prompt = ChatPromptTemplate.from_messages([
        ("system", CUSTOM_SYSTEM_PROMPT_CONTENT),
        MessagesPlaceholder(variable_name="chat_history", optional=True),
        ("human", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad"), 
    ])

    # logger.info(f"Using prompt for tool calling agent: {prompt.pretty_repr()}") # Commented out due to ValueError

    agent = create_tool_calling_agent(llm_with_tools, tools, prompt)
    
    agent_executor = AgentExecutor(
        agent=agent, 
        tools=tools, 
        verbose=True, 
        handle_parsing_errors=True,
        max_iterations=20,  # Increased to allow for all 6 steps
        early_stopping_method="force",  # Changed from "generate" to "force"
        return_intermediate_steps=True,
    )
    logger.info("Tool-calling Destination Analyst Agent Executor created.")
    return agent_executor 