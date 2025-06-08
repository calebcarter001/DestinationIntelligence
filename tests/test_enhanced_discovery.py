import asyncio
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.core.web_discovery_logic import WebDiscoveryLogic
from src.config_loader import load_app_config

async def test_enhanced_discovery():
    """Test enhanced discovery with thematic content prioritization"""
    print("🧪 Testing Enhanced Discovery System")
    print("=" * 60)
    
    try:
        config = load_app_config()
        api_key = config.get('processing_settings', {}).get('web_discovery', {}).get('brave_search_api_key')
        if not api_key:
            api_key = os.getenv('BRAVE_SEARCH_API_KEY')
        
        if not api_key:
            print("❌ No Brave Search API key found. Please set BRAVE_SEARCH_API_KEY environment variable.")
            return
        
        wd = WebDiscoveryLogic(api_key=api_key, config=config.get('processing_settings', {}))
        
        async with wd as logic:
            destination = "Sydney, Australia"
            print(f"\n🎯 Testing enhanced discovery for: {destination}")
            
            # Test 1: Thematic discovery
            print("\n📝 Test 1: Thematic Content Discovery")
            print("-" * 40)
            thematic_results = await logic.discover_thematic_content(destination)
            
            total_thematic_sources = 0
            for theme, sources in thematic_results.items():
                print(f"  {theme}: {len(sources)} sources")
                total_thematic_sources += len(sources)
            
            print(f"Total thematic sources: {total_thematic_sources}")
            
            # Test 2: Content relevance filtering
            print("\n📝 Test 2: Content Relevance Filtering")
            print("-" * 40)
            test_content_good = f"Sydney is a beautiful city with amazing harbors. The Sydney Opera House is iconic. Great restaurants in Circular Quay."
            test_content_bad = "This website uses cookies. Accept all cookies. Privacy policy terms of service."
            
            relevant_good = logic._validate_content_relevance(test_content_good, destination, "test-url-good")
            relevant_bad = logic._validate_content_relevance(test_content_bad, destination, "test-url-bad")
            
            print(f"  Good content relevance: {relevant_good} ✅" if relevant_good else f"  Good content relevance: {relevant_good} ❌")
            print(f"  Bad content relevance: {not relevant_bad} ✅" if not relevant_bad else f"  Bad content relevance: {not relevant_bad} ❌")
            
            # Test 3: Source quality scoring
            print("\n📝 Test 3: Source Quality Scoring")
            print("-" * 40)
            test_sources = [
                {"url": "https://www.tripadvisor.com/sydney-guide", "title": "Sydney Travel Guide"},
                {"url": "https://example.com/generic", "title": "Generic Content"},
                {"url": "https://visitnsw.com/sydney-attractions", "title": "Official Sydney Attractions"},
            ]
            
            for source in test_sources:
                quality_score = logic._calculate_source_quality_score(source)
                print(f"  {source['url']}: Quality Score = {quality_score:.2f}")
            
            # Test 4: Full discovery with new prioritization
            print("\n📝 Test 4: Full Enhanced Discovery")
            print("-" * 40)
            all_sources = await logic.discover_real_content(destination)
            
            # Analyze distribution
            content_types = {}
            theme_categories = {}
            quality_scores = []
            
            for source in all_sources:
                content_type = source.get('content_type', 'unknown')
                content_types[content_type] = content_types.get(content_type, 0) + 1
                
                if content_type == 'thematic':
                    theme_cat = source.get('theme_category', 'unknown')
                    theme_categories[theme_cat] = theme_categories.get(theme_cat, 0) + 1
                
                quality_scores.append(source.get('quality_score', 1.0))
            
            avg_quality = sum(quality_scores) / len(quality_scores) if quality_scores else 0
            
            print(f"Total sources discovered: {len(all_sources)}")
            print(f"Content type distribution: {content_types}")
            print(f"Thematic categories: {theme_categories}")
            print(f"Average quality score: {avg_quality:.2f}")
            
            # Test 5: Content quality validation
            print("\n📝 Test 5: Content Quality Validation")
            print("-" * 40)
            content_lengths = [len(s.get('content', '')) for s in all_sources if s.get('content')]
            avg_content_length = sum(content_lengths) / len(content_lengths) if content_lengths else 0
            
            boilerplate_count = 0
            for source in all_sources:
                content = source.get('content', '').lower()
                if 'cookie' in content or 'privacy policy' in content:
                    boilerplate_count += 1
            
            print(f"Average content length: {avg_content_length:.0f} characters")
            print(f"Sources with boilerplate content: {boilerplate_count}/{len(all_sources)}")
            print(f"Content quality: {'✅ Good' if boilerplate_count < len(all_sources) * 0.2 else '⚠️  Needs improvement'}")
            
            # Summary
            print("\n🎉 Test Summary")
            print("=" * 60)
            success_criteria = [
                ("Thematic discovery working", total_thematic_sources > 0),
                ("Content filtering working", relevant_good and not relevant_bad),
                ("Quality scoring working", avg_quality > 1.0),
                ("Balanced distribution", content_types.get('thematic', 0) >= content_types.get('priority', 0)),
                ("Good content quality", boilerplate_count < len(all_sources) * 0.2)
            ]
            
            passed = sum(1 for _, result in success_criteria if result)
            total = len(success_criteria)
            
            for criteria, result in success_criteria:
                status = "✅" if result else "❌"
                print(f"  {status} {criteria}")
            
            print(f"\n🏆 Overall Result: {passed}/{total} tests passed")
            if passed == total:
                print("🎉 All enhanced discovery features working correctly!")
            else:
                print("⚠️  Some features need attention.")
            
    except Exception as e:
        print(f"❌ Error during testing: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(test_enhanced_discovery()) 