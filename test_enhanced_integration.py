#!/usr/bin/env python
"""
Test script to verify enhanced architecture integration
"""

import sys
import os
import asyncio
import logging

# Add project root to path
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_imports():
    """Test all imports work correctly"""
    logger.info("Testing imports...")
    
    try:
        # Core components
        from src.core.evidence_hierarchy import EvidenceHierarchy, SourceCategory
        logger.info("✓ Evidence Hierarchy imported")
        
        from src.core.confidence_scoring import ConfidenceScorer, ConfidenceLevel
        logger.info("✓ Confidence Scoring imported")
        
        from src.core.enhanced_data_models import Destination, Theme, Evidence, TemporalSlice
        logger.info("✓ Enhanced Data Models imported")
        
        from src.core.enhanced_database_manager import EnhancedDatabaseManager
        logger.info("✓ Enhanced Database Manager imported")
        
        from src.core.json_export_manager import JsonExportManager
        logger.info("✓ JSON Export Manager imported")
        
        # Agents
        from src.agents.base_agent import BaseAgent, MessageBroker, AgentOrchestrator
        logger.info("✓ Base Agent components imported")
        
        from src.agents.specialized_agents import ValidationAgent, CulturalPerspectiveAgent, ContradictionDetectionAgent
        logger.info("✓ Specialized Agents imported")
        
        # Tools
        from src.tools.enhanced_theme_analysis_tool import EnhancedThemeAnalysisTool, EnhancedAnalyzeThemesFromEvidenceTool
        logger.info("✓ Enhanced Theme Analysis Tool imported")
        
        from src.tools.enhanced_database_tools import StoreEnhancedDestinationInsightsTool
        logger.info("✓ Enhanced Database Tools imported")
        
        return True
        
    except Exception as e:
        logger.error(f"✗ Import failed: {e}")
        return False

async def test_agent_system():
    """Test multi-agent system initialization"""
    logger.info("\nTesting multi-agent system...")
    
    try:
        from src.agents.base_agent import MessageBroker, AgentOrchestrator
        from src.agents.specialized_agents import ValidationAgent, CulturalPerspectiveAgent, ContradictionDetectionAgent
        
        # Create broker
        broker = MessageBroker()
        logger.info("✓ Message Broker created")
        
        # Create agents
        validation_agent = ValidationAgent("test_validation")
        cultural_agent = CulturalPerspectiveAgent("test_cultural")
        contradiction_agent = ContradictionDetectionAgent("test_contradiction")
        logger.info("✓ Agents created")
        
        # Register agents
        broker.register_agent(validation_agent)
        broker.register_agent(cultural_agent)
        broker.register_agent(contradiction_agent)
        logger.info("✓ Agents registered")
        
        # Create orchestrator
        orchestrator = AgentOrchestrator(broker)
        logger.info("✓ Orchestrator created")
        
        return True
        
    except Exception as e:
        logger.error(f"✗ Agent system test failed: {e}")
        return False

def test_evidence_hierarchy():
    """Test evidence hierarchy functionality"""
    logger.info("\nTesting evidence hierarchy...")
    
    try:
        from src.core.evidence_hierarchy import EvidenceHierarchy
        
        # Test source classification
        test_urls = [
            ("https://travel.state.gov/content/travel/en/traveladvisories/traveladvisories/france-travel-advisory.html", "government"),
            ("https://whc.unesco.org/en/list/1234", "unesco"),
            ("https://www.nature.com/articles/travel-study", "academic"),
            ("https://www.lonelyplanet.com/france/paris", "guidebook"),
            ("https://www.tripadvisor.com/Hotel_Review-g187147-d234567", "review"),
            ("https://www.travelblog.com/my-paris-trip", "blog")
        ]
        
        for url, expected_category in test_urls:
            category = EvidenceHierarchy.classify_source(url)
            logger.info(f"  {url[:50]}... → {category.value}")
            
        logger.info("✓ Evidence hierarchy working")
        return True
        
    except Exception as e:
        logger.error(f"✗ Evidence hierarchy test failed: {e}")
        return False

def test_confidence_scoring():
    """Test confidence scoring functionality"""
    logger.info("\nTesting confidence scoring...")
    
    try:
        from src.core.confidence_scoring import ConfidenceScorer
        
        # Test confidence calculation
        test_sources = [
            "https://travel.state.gov/content/travel/en/traveladvisories/france.html",
            "https://www.tripadvisor.com/Hotel_Review-g187147",
            "https://www.travelblog.com/paris"
        ]
        
        confidence = ConfidenceScorer.calculate_confidence(
            evidence_sources=test_sources,
            evidence_texts=["Safe destination", "Great hotel", "Amazing trip"],
            destination_country_code="FR"
        )
        
        logger.info(f"  Total confidence: {confidence.total_confidence:.2f}")
        logger.info(f"  Level: {confidence.confidence_level.value}")
        logger.info("✓ Confidence scoring working")
        return True
        
    except Exception as e:
        logger.error(f"✗ Confidence scoring test failed: {e}")
        return False

def test_enhanced_tools():
    """Test enhanced tools initialization"""
    logger.info("\nTesting enhanced tools...")
    
    try:
        from src.core.enhanced_database_manager import EnhancedDatabaseManager
        from src.tools.enhanced_database_tools import StoreEnhancedDestinationInsightsTool
        from src.tools.enhanced_theme_analysis_tool import EnhancedAnalyzeThemesFromEvidenceTool
        from src.core.content_intelligence_logic import ContentIntelligenceLogic
        
        # Test database manager
        db_manager = EnhancedDatabaseManager(
            db_path=":memory:",
            enable_json_export=False
        )
        logger.info("✓ Enhanced Database Manager initialized")
        
        # Test database tool
        db_tool = StoreEnhancedDestinationInsightsTool(db_manager=db_manager)
        logger.info("✓ Enhanced Database Tool created")
        
        # Test theme analysis tool
        content_logic = ContentIntelligenceLogic(config={})
        theme_tool = EnhancedAnalyzeThemesFromEvidenceTool(
            content_intelligence_logic=content_logic,
            agent_orchestrator=None
        )
        logger.info("✓ Enhanced Theme Analysis Tool created")
        
        return True
        
    except Exception as e:
        logger.error(f"✗ Enhanced tools test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all integration tests"""
    logger.info("=" * 50)
    logger.info("Enhanced Architecture Integration Test")
    logger.info("=" * 50)
    
    tests = [
        ("Imports", test_imports),
        ("Evidence Hierarchy", test_evidence_hierarchy),
        ("Confidence Scoring", test_confidence_scoring),
        ("Enhanced Tools", test_enhanced_tools)
    ]
    
    # Run sync tests
    results = {}
    for test_name, test_func in tests:
        if not asyncio.iscoroutinefunction(test_func):
            results[test_name] = test_func()
    
    # Run async tests
    async def run_async_tests():
        results["Agent System"] = await test_agent_system()
        return results
    
    loop = asyncio.get_event_loop()
    results = loop.run_until_complete(run_async_tests())
    
    # Summary
    logger.info("\n" + "=" * 50)
    logger.info("Test Summary:")
    logger.info("=" * 50)
    
    passed = sum(1 for result in results.values() if result)
    total = len(results)
    
    for test_name, result in results.items():
        status = "✓ PASSED" if result else "✗ FAILED"
        logger.info(f"{test_name:.<30} {status}")
    
    logger.info(f"\nTotal: {passed}/{total} tests passed")
    
    if passed == total:
        logger.info("\n🎉 All tests passed! Enhanced architecture is properly integrated.")
        return 0
    else:
        logger.error(f"\n❌ {total - passed} tests failed. Please check the errors above.")
        return 1

if __name__ == "__main__":
    sys.exit(main()) 