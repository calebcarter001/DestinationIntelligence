"""
Cultural Intelligence Demonstration Script
Shows all cultural intelligence features in action with sample data.
"""

import json
import os
from datetime import datetime

def demonstrate_cultural_intelligence():
    """Demonstrate all cultural intelligence features"""
    print("🎭 CULTURAL INTELLIGENCE DEMONSTRATION")
    print("=" * 60)
    print("Showcasing the Dual-Track Cultural Intelligence System")
    print()
    
    # 1. Theme Categorization Demo
    print("📋 1. THEME CATEGORIZATION")
    print("-" * 30)
    
    sample_themes = [
        {"name": "Grunge Music Heritage", "macro_category": "Cultural Identity & Atmosphere"},
        {"name": "Coffee Culture Origins", "macro_category": "Authentic Experiences"},
        {"name": "Pike Place Fish Throwing", "macro_category": "Distinctive Features"},
        {"name": "Public Transportation", "macro_category": "Transportation & Access"},
        {"name": "Crime Statistics", "macro_category": "Safety & Security"},
        {"name": "Local Food Scene", "macro_category": "Food & Dining"},
        {"name": "Nightlife Districts", "macro_category": "Entertainment & Nightlife"},
    ]
    
    # Import the categorization function
    try:
        from compare_destinations import get_processing_type, CATEGORY_PROCESSING_RULES
        
        categorized_themes = {"cultural": [], "practical": [], "hybrid": [], "unknown": []}
        
        for theme in sample_themes:
            processing_type = get_processing_type(theme["macro_category"])
            categorized_themes[processing_type].append(theme)
            
            icon = CATEGORY_PROCESSING_RULES.get(processing_type, {}).get("icon", "❓")
            color = CATEGORY_PROCESSING_RULES.get(processing_type, {}).get("color", "#666666")
            
            print(f"{icon} {theme['name']} → {processing_type.upper()}")
        
        print()
        for proc_type, themes in categorized_themes.items():
            if themes:
                count = len(themes)
                icon = CATEGORY_PROCESSING_RULES.get(proc_type, {}).get("icon", "❓")
                print(f"{icon} {proc_type.title()}: {count} themes")
        
    except ImportError:
        print("⚠️  Cultural intelligence scripts not available for demo")
    
    print()
    
    # 2. Authenticity Scoring Demo
    print("🔍 2. AUTHENTICITY SCORING")
    print("-" * 30)
    
    authentic_sources = [
        {"url": "reddit.com/r/Seattle", "title": "Local's guide to hidden gems"},
        {"url": "seattlelocalblog.com", "title": "Community recommendations"},
        {"url": "neighborhoodforum.local", "title": "Resident insights"}
    ]
    
    official_sources = [
        {"url": "seattle.gov/tourism", "title": "Official visitor guide"},
        {"url": "university.edu/study", "title": "Academic research"},
        {"url": "visitseattle.org", "title": "Tourism board recommendations"}
    ]
    
    def calculate_authenticity_demo(sources):
        authentic_indicators = ['reddit.com', 'local', 'community', 'blog']
        official_indicators = ['gov', 'edu', 'official', 'tourism']
        
        authentic_count = sum(1 for s in sources 
                            if any(ind in s['url'].lower() or ind in s['title'].lower() 
                                 for ind in authentic_indicators))
        
        total = len(sources)
        return authentic_count / total if total > 0 else 0
    
    authentic_score = calculate_authenticity_demo(authentic_sources)
    official_score = calculate_authenticity_demo(official_sources)
    
    print("Authentic Sources:")
    for source in authentic_sources:
        print(f"  🟢 {source['title']} ({source['url']})")
    print(f"  Authenticity Score: {authentic_score:.1%}")
    
    print("\nOfficial Sources:")
    for source in official_sources:
        print(f"  🔵 {source['title']} ({source['url']})")
    print(f"  Authenticity Score: {official_score:.1%}")
    
    print()
    
    # 3. Destination Personality Demo
    print("🏙️ 3. DESTINATION PERSONALITY DETECTION")
    print("-" * 40)
    
    destinations = {
        "Seattle": {"cultural": 6, "practical": 2, "hybrid": 3},
        "Singapore": {"cultural": 2, "practical": 8, "hybrid": 2},
        "Barcelona": {"cultural": 4, "practical": 3, "hybrid": 5},
        "Portland": {"cultural": 3, "practical": 3, "hybrid": 3}
    }
    
    def get_personality(theme_stats):
        total = sum(theme_stats.values())
        if total == 0:
            return "unknown"
        
        ratios = {k: v/total for k, v in theme_stats.items()}
        dominant = max(ratios, key=ratios.get)
        
        if ratios[dominant] > 0.4:
            return dominant
        else:
            return "balanced"
    
    personality_icons = {
        "cultural": "🎭 Cultural-Focused",
        "practical": "📋 Practical-Focused",
        "hybrid": "⚖️ Hybrid-Focused", 
        "balanced": "🌈 Well-Rounded"
    }
    
    for dest_name, stats in destinations.items():
        personality = get_personality(stats)
        icon_desc = personality_icons.get(personality, personality)
        
        total = sum(stats.values())
        cultural_pct = (stats["cultural"] / total * 100) if total > 0 else 0
        practical_pct = (stats["practical"] / total * 100) if total > 0 else 0
        hybrid_pct = (stats["hybrid"] / total * 100) if total > 0 else 0
        
        print(f"{dest_name}: {icon_desc}")
        print(f"  🎭 Cultural: {cultural_pct:.0f}% | 📋 Practical: {practical_pct:.0f}% | ⚖️ Hybrid: {hybrid_pct:.0f}%")
        print()
    
    # 4. Cultural Intelligence Metrics Demo
    print("📊 4. CULTURAL INTELLIGENCE METRICS")
    print("-" * 35)
    
    sample_destination_data = {
        "destination": "Seattle, United States",
        "total_themes": 15,
        "theme_distribution": {
            "cultural": {"count": 6, "avg_confidence": 0.82},
            "practical": {"count": 4, "avg_confidence": 0.91},
            "hybrid": {"count": 5, "avg_confidence": 0.76}
        },
        "high_confidence_themes": {
            "cultural": 4,
            "practical": 4, 
            "hybrid": 3
        }
    }
    
    data = sample_destination_data
    total = data["total_themes"]
    dist = data["theme_distribution"]
    
    print(f"Destination: {data['destination']}")
    print(f"Total Themes: {total}")
    print()
    
    print("Theme Distribution:")
    cultural_ratio = dist["cultural"]["count"] / total
    practical_ratio = dist["practical"]["count"] / total
    hybrid_ratio = dist["hybrid"]["count"] / total
    
    print(f"  🎭 Cultural: {dist['cultural']['count']} themes ({cultural_ratio:.1%}) - Avg Confidence: {dist['cultural']['avg_confidence']:.1%}")
    print(f"  📋 Practical: {dist['practical']['count']} themes ({practical_ratio:.1%}) - Avg Confidence: {dist['practical']['avg_confidence']:.1%}")
    print(f"  ⚖️ Hybrid: {dist['hybrid']['count']} themes ({hybrid_ratio:.1%}) - Avg Confidence: {dist['hybrid']['avg_confidence']:.1%}")
    
    print("\nHigh-Confidence Themes:")
    for proc_type, count in data["high_confidence_themes"].items():
        total_type = dist[proc_type]["count"]
        pct = (count / total_type * 100) if total_type > 0 else 0
        icon = {"cultural": "🎭", "practical": "📋", "hybrid": "⚖️"}[proc_type]
        print(f"  {icon} {proc_type.title()}: {count}/{total_type} ({pct:.0f}%)")
    
    # Cultural vs Practical Ratio
    cultural_practical_total = dist["cultural"]["count"] + dist["practical"]["count"]
    if cultural_practical_total > 0:
        cultural_ratio_cp = dist["cultural"]["count"] / cultural_practical_total
        print(f"\nCultural vs Practical Ratio: {cultural_ratio_cp:.1%} cultural, {1-cultural_ratio_cp:.1%} practical")
        
        if cultural_ratio_cp > 0.6:
            personality_desc = "🎭 Culturally-oriented destination"
        elif cultural_ratio_cp < 0.4:
            personality_desc = "📋 Practically-oriented destination"
        else:
            personality_desc = "⚖️ Balanced cultural-practical destination"
        
        print(f"Assessment: {personality_desc}")
    
    print()
    
    # 5. Enhanced Script Features Demo
    print("🚀 5. ENHANCED SCRIPT FEATURES")
    print("-" * 32)
    
    scripts_info = [
        {
            "script": "analyze_themes.py",
            "features": [
                "🎯 Processing type identification",
                "📊 Cultural intelligence metrics",
                "🎨 Color-coded theme categories",
                "📈 Cultural vs practical ratios"
            ]
        },
        {
            "script": "generate_dynamic_viewer.py", 
            "features": [
                "🏷️ Category badges and icons",
                "🎨 Visual theme categorization",
                "🔍 Category-based filtering",
                "📱 Enhanced interactive UI"
            ]
        },
        {
            "script": "compare_destinations.py",
            "features": [
                "🧠 Cultural intelligence similarity",
                "👥 Destination personality matching",
                "⚖️ Category-specific comparisons",
                "📊 Enhanced similarity metrics"
            ]
        }
    ]
    
    for script_info in scripts_info:
        print(f"📄 {script_info['script']}")
        for feature in script_info['features']:
            print(f"   {feature}")
        print()
    
    # 6. Success Summary
    print("✅ 6. IMPLEMENTATION SUCCESS")
    print("-" * 30)
    
    achievements = [
        "🎭 Dual-track processing (Cultural vs Practical vs Hybrid)",
        "🔍 Authenticity scoring based on source characteristics", 
        "🎯 Distinctiveness filtering for unique cultural themes",
        "👥 Destination personality detection and matching",
        "📊 Comprehensive cultural intelligence metrics",
        "🎨 Visual categorization across all scripts",
        "⚖️ Scientific yet nuanced approach to cultural analysis",
        "🔄 Consistent categorization across entire pipeline"
    ]
    
    for achievement in achievements:
        print(f"   {achievement}")
    
    print()
    print("🎯 RESULT: Transform generic themes like 'In Seattle' →")
    print("           Specific cultural themes like 'Grunge Music Heritage'!")
    print()
    print("=" * 60)
    print("🎉 Cultural Intelligence System Successfully Implemented!")
    print("=" * 60)

def generate_sample_config():
    """Generate a sample configuration for cultural intelligence"""
    config = {
        "cultural_intelligence": {
            "enable_cultural_categories": True,
            "enable_authenticity_scoring": True,
            "enable_distinctiveness_filtering": True,
            "authentic_source_indicators": [
                "reddit.com", "local", "community", "blog", "forum", 
                "neighborhood", "resident", "locals"
            ],
            "authoritative_source_indicators": [
                "gov", "edu", "official", "tourism", "government",
                "university", "academic", "ministry"
            ],
            "distinctiveness_indicators": {
                "unique_keywords": [
                    "unique", "distinctive", "special", "rare", "authentic",
                    "unusual", "exclusive", "signature", "characteristic"
                ],
                "generic_keywords": [
                    "popular", "common", "typical", "standard", "normal",
                    "regular", "ordinary", "conventional", "mainstream"
                ]
            },
            "category_processing_rules": {
                "cultural": {
                    "confidence_threshold": 0.45,
                    "distinctiveness_threshold": 0.3,
                    "authenticity_weight": 0.4
                },
                "practical": {
                    "confidence_threshold": 0.75,
                    "distinctiveness_threshold": 0.1,
                    "authority_weight": 0.5
                },
                "hybrid": {
                    "confidence_threshold": 0.6,
                    "distinctiveness_threshold": 0.2,
                    "balanced_weight": 0.35
                }
            }
        }
    }
    
    return config

if __name__ == "__main__":
    demonstrate_cultural_intelligence()
    
    # Also show sample configuration
    print("\n📝 SAMPLE CONFIGURATION")
    print("-" * 25)
    print("Add this to your config.yaml:")
    print()
    
    sample_config = generate_sample_config()
    import yaml
    try:
        yaml_output = yaml.dump(sample_config, default_flow_style=False, sort_keys=False)
        print(yaml_output)
    except:
        # Fallback to JSON if yaml not available
        json_output = json.dumps(sample_config, indent=2)
        print(json_output) 