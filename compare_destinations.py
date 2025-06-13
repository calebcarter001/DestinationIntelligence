import argparse
import logging
import sqlite3
import json
from typing import Dict, Any, Optional, List
from collections import defaultdict
from src.core.enhanced_database_manager import EnhancedDatabaseManager
from src.core.enhanced_data_models import Destination

logging.basicConfig(level=logging.INFO, format='%(asctime)s - [%(levelname)s] - %(message)s')

# CULTURAL INTELLIGENCE: Define category processing rules
CATEGORY_PROCESSING_RULES = {
    "cultural": {
        "categories": [
            "Cultural Identity & Atmosphere", "Authentic Experiences", "Distinctive Features",
            "Local Character & Vibe", "Artistic & Creative Scene",
            "Cultural", "Local Culture", "Heritage", "Arts", "Music", "Festivals"
        ],
        "color": "#9C27B0",
        "icon": "🎭",
        "weight": 0.4  # Higher weight for cultural comparison
    },
    "practical": {
        "categories": [
            "Safety & Security", "Transportation & Access", "Budget & Costs", 
            "Health & Medical", "Logistics & Planning", "Visa & Documentation"
        ],
        "color": "#2196F3",
        "icon": "📋",
        "weight": 0.3  # Moderate weight for practical comparison
    },
    "hybrid": {
        "categories": [
            "Food & Dining", "Entertainment & Nightlife", "Nature & Outdoor",
            "Shopping & Local Craft", "Family & Education", "Health & Wellness",
            "Popular", "POI", "Entertainment", "Food", "Dining", "Shopping"
        ],
        "color": "#4CAF50",
        "icon": "⚖️",
        "weight": 0.3  # Moderate weight for hybrid comparison
    }
}

def get_processing_type(macro_category):
    """Determine if theme is cultural, practical, or hybrid"""
    if not macro_category:
        return "unknown"
    
    for proc_type, rules in CATEGORY_PROCESSING_RULES.items():
        if macro_category in rules["categories"]:
            return proc_type
    return "unknown"

def load_destination_themes(db_path: str, destination_name: str) -> Dict[str, Any]:
    """Load all themes for a destination with cultural intelligence categorization"""
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # Convert destination name to database ID format
        dest_id = f"dest_{destination_name.replace(', ', '_').replace(' ', '_').lower()}"
        
        cursor.execute("""
            SELECT theme_id, name, macro_category, micro_category, description,
                   fit_score, confidence_level, confidence_breakdown,
                   tags, adjusted_overall_confidence, traveler_relevance_factor
            FROM themes
            WHERE destination_id = ?
            ORDER BY adjusted_overall_confidence DESC, fit_score DESC
        """, (dest_id,))
        
        themes_data = cursor.fetchall()
        conn.close()
        
        # Organize themes by processing type
        themes_by_type = {"cultural": [], "practical": [], "hybrid": [], "unknown": []}
        theme_stats = {"cultural": 0, "practical": 0, "hybrid": 0, "unknown": 0}
        total_confidence = {"cultural": 0.0, "practical": 0.0, "hybrid": 0.0}
        high_conf_themes = {"cultural": 0, "practical": 0, "hybrid": 0}
        
        for theme in themes_data:
            theme_id, name, macro_cat, micro_cat, description, fit_score, conf_level, conf_breakdown, tags, adj_conf, trav_relevance = theme
            
            proc_type = get_processing_type(macro_cat)
            confidence = adj_conf if adj_conf is not None else 0.0
            
            theme_obj = {
                "id": theme_id,
                "name": name,
                "macro_category": macro_cat,
                "micro_category": micro_cat,
                "description": description,
                "fit_score": fit_score or 0.0,
                "confidence": confidence,
                "confidence_level": conf_level,
                "processing_type": proc_type,
                "traveler_relevance": trav_relevance or 0.0
            }
            
            themes_by_type[proc_type].append(theme_obj)
            theme_stats[proc_type] += 1
            
            if proc_type in total_confidence:
                total_confidence[proc_type] += confidence
                if confidence > 0.7:
                    high_conf_themes[proc_type] += 1
        
        # Calculate averages
        avg_confidence = {}
        for proc_type in total_confidence:
            count = theme_stats[proc_type]
            avg_confidence[proc_type] = total_confidence[proc_type] / count if count > 0 else 0.0
        
        return {
            "themes_by_type": themes_by_type,
            "theme_stats": theme_stats,
            "avg_confidence": avg_confidence,
            "high_confidence_themes": high_conf_themes,
            "total_themes": sum(theme_stats.values())
        }
        
    except Exception as e:
        logging.error(f"Error loading themes for {destination_name}: {e}")
        return {
            "themes_by_type": {"cultural": [], "practical": [], "hybrid": [], "unknown": []},
            "theme_stats": {"cultural": 0, "practical": 0, "hybrid": 0, "unknown": 0},
            "avg_confidence": {"cultural": 0.0, "practical": 0.0, "hybrid": 0.0},
            "high_confidence_themes": {"cultural": 0, "practical": 0, "hybrid": 0},
            "total_themes": 0
        }

def calculate_cultural_intelligence_similarity(dest1_themes: Dict, dest2_themes: Dict) -> Dict[str, Any]:
    """Calculate cultural intelligence specific similarity metrics"""
    results = {}
    
    # 1. Theme Distribution Similarity
    theme_dist_sim = {}
    for proc_type in ["cultural", "practical", "hybrid"]:
        count1 = dest1_themes["theme_stats"][proc_type]
        count2 = dest2_themes["theme_stats"][proc_type]
        total1 = dest1_themes["total_themes"]
        total2 = dest2_themes["total_themes"]
        
        if total1 > 0 and total2 > 0:
            ratio1 = count1 / total1
            ratio2 = count2 / total2
            theme_dist_sim[proc_type] = 1 - abs(ratio1 - ratio2)
        else:
            theme_dist_sim[proc_type] = 0.0
    
    results["theme_distribution_similarity"] = theme_dist_sim
    
    # 2. Confidence Level Similarity by Type
    conf_sim = {}
    for proc_type in ["cultural", "practical", "hybrid"]:
        conf1 = dest1_themes["avg_confidence"][proc_type]
        conf2 = dest2_themes["avg_confidence"][proc_type]
        conf_sim[proc_type] = calculate_similarity(conf1, conf2)
    
    results["confidence_similarity"] = conf_sim
    
    # 3. Theme Name/Content Similarity (simplified)
    content_sim = {}
    for proc_type in ["cultural", "practical", "hybrid"]:
        themes1 = dest1_themes["themes_by_type"][proc_type]
        themes2 = dest2_themes["themes_by_type"][proc_type]
        
        if not themes1 or not themes2:
            content_sim[proc_type] = 0.0
            continue
        
        # Simple name-based similarity
        names1 = {theme["name"].lower() for theme in themes1}
        names2 = {theme["name"].lower() for theme in themes2}
        
        if names1 or names2:
            intersection = len(names1 & names2)
            union = len(names1 | names2)
            content_sim[proc_type] = intersection / union if union > 0 else 0.0
        else:
            content_sim[proc_type] = 0.0
    
    results["content_similarity"] = content_sim
    
    # 4. Cultural Character Analysis
    cultural_character = {}
    
    # Cultural vs Practical ratio comparison
    def get_cultural_ratio(themes_data):
        cultural_count = themes_data["theme_stats"]["cultural"]
        practical_count = themes_data["theme_stats"]["practical"]
        total = cultural_count + practical_count
        return cultural_count / total if total > 0 else 0.0
    
    ratio1 = get_cultural_ratio(dest1_themes)
    ratio2 = get_cultural_ratio(dest2_themes)
    cultural_character["cultural_practical_ratio_similarity"] = 1 - abs(ratio1 - ratio2)
    cultural_character["dest1_cultural_ratio"] = ratio1
    cultural_character["dest2_cultural_ratio"] = ratio2
    
    # Determine destination personalities
    def get_destination_personality(themes_data):
        ratios = {}
        total = themes_data["total_themes"]
        if total == 0:
            return "unknown"
        
        for proc_type in ["cultural", "practical", "hybrid"]:
            ratios[proc_type] = themes_data["theme_stats"][proc_type] / total
        
        dominant_type = max(ratios, key=ratios.get)
        # Increased threshold to 50% for more balanced detection
        if ratios[dominant_type] >= 0.5:
            return dominant_type
        else:
            return "balanced"
    
    cultural_character["dest1_personality"] = get_destination_personality(dest1_themes)
    cultural_character["dest2_personality"] = get_destination_personality(dest2_themes)
    
    results["cultural_character"] = cultural_character
    
    return results

def calculate_similarity(val1, val2, is_numeric=True):
    """Calculates a normalized similarity score between two values."""
    if val1 is None or val2 is None:
        return 0.0
    
    if not is_numeric:
        # For categorical or string data, similarity is 1 if they match, 0 otherwise
        return 1.0 if str(val1).lower() == str(val2).lower() else 0.0

    try:
        num1, num2 = float(val1), float(val2)
        # Avoid division by zero if both are zero
        if num1 == 0 and num2 == 0:
            return 1.0
        # Normalized difference
        return 1 - abs(num1 - num2) / (abs(num1) + abs(num2))
    except (ValueError, TypeError):
        return 0.0

def compare_destinations(dest1: Destination, dest2: Destination, dest1_themes: Dict, dest2_themes: Dict) -> Dict[str, Any]:
    """
    Enhanced destination comparison with cultural intelligence insights.
    """
    if not dest1 or not dest2:
        raise ValueError("Cannot compare None Destination objects.")

    results = {"scores": {}, "drivers": {}, "cultural_intelligence": {}}
    
    # CULTURAL INTELLIGENCE: Enhanced weights emphasizing cultural aspects
    weights = {
        "cultural_intelligence": 0.40,  # NEW: High weight for cultural intelligence
        "experiential": 0.20,           # Reduced from 0.25
        "geography": 0.15,              # Reduced from 0.20
        "cultural_traditional": 0.15,   # Reduced from 0.20
        "logistics": 0.10,              # Reduced from 0.25
    }
    
    # CULTURAL INTELLIGENCE: New comprehensive analysis
    cultural_intel_results = calculate_cultural_intelligence_similarity(dest1_themes, dest2_themes)
    results["cultural_intelligence"] = cultural_intel_results
    
    # Calculate weighted cultural intelligence score
    cultural_intel_score = 0.0
    ci_weights = {
        "theme_distribution": 0.3,
        "confidence_level": 0.2,
        "content_similarity": 0.3,
        "cultural_character": 0.2
    }
    
    # Theme distribution similarity
    theme_dist_scores = cultural_intel_results["theme_distribution_similarity"]
    avg_theme_dist = sum(theme_dist_scores.values()) / len(theme_dist_scores)
    cultural_intel_score += avg_theme_dist * ci_weights["theme_distribution"]
    
    # Confidence similarity
    conf_scores = cultural_intel_results["confidence_similarity"]
    avg_conf_sim = sum(conf_scores.values()) / len(conf_scores)
    cultural_intel_score += avg_conf_sim * ci_weights["confidence_level"]
    
    # Content similarity
    content_scores = cultural_intel_results["content_similarity"]
    avg_content_sim = sum(content_scores.values()) / len(content_scores)
    cultural_intel_score += avg_content_sim * ci_weights["content_similarity"]
    
    # Cultural character similarity
    char_sim = cultural_intel_results["cultural_character"]["cultural_practical_ratio_similarity"]
    cultural_intel_score += char_sim * ci_weights["cultural_character"]
    
    results["scores"]["cultural_intelligence"] = cultural_intel_score

    # 1. Traditional Experiential & Activity Alignment
    vibe_sim = len(set(dest1.vibe_descriptors) & set(dest2.vibe_descriptors)) / len(set(dest1.vibe_descriptors) | set(dest2.vibe_descriptors)) if dest1.vibe_descriptors or dest2.vibe_descriptors else 0
    results["scores"]["experiential"] = vibe_sim

    # 2. Geographic & Environmental Fit
    geo_scores = {
        "area": calculate_similarity(dest1.area_km2, dest2.area_km2),
    }
    results["scores"]["geography"] = sum(geo_scores.values()) / len(geo_scores) if geo_scores else 0
    
    # 3. Traditional Cultural & Social Context
    cultural_scores = {
        "language": calculate_similarity(dest1.primary_language, dest2.primary_language, is_numeric=False),
        "population": calculate_similarity(dest1.population, dest2.population),
        "religion": len(set(dest1.dominant_religions) & set(dest2.dominant_religions)) / len(set(dest1.dominant_religions) | set(dest2.dominant_religions)) if dest1.dominant_religions or dest2.dominant_religions else 0,
    }
    results["scores"]["cultural_traditional"] = sum(cultural_scores.values()) / len(cultural_scores) if cultural_scores else 0

    # 4. Practicality & Logistics
    logistics_scores = {
        "gdp": calculate_similarity(dest1.gdp_per_capita_usd, dest2.gdp_per_capita_usd),
        "hdi": calculate_similarity(dest1.hdi, dest2.hdi),
    }
    results["scores"]["logistics"] = sum(logistics_scores.values()) / len(logistics_scores) if logistics_scores else 0

    # Calculate overall weighted score with cultural intelligence emphasis
    overall_score = (
        results["scores"]["cultural_intelligence"] * weights["cultural_intelligence"] +
        results["scores"]["experiential"] * weights["experiential"] +
        results["scores"]["geography"] * weights["geography"] +
        results["scores"]["cultural_traditional"] * weights["cultural_traditional"] +
        results["scores"]["logistics"] * weights["logistics"]
    )
    results["overall_similarity_score"] = overall_score
    
    # Enhanced drivers analysis
    results["drivers"]["top_similarity"] = max(results["scores"], key=results["scores"].get)
    results["drivers"]["top_difference"] = min(results["scores"], key=results["scores"].get)
    
    # Cultural intelligence specific insights
    results["drivers"]["cultural_insights"] = {
        "most_similar_category": max(cultural_intel_results["content_similarity"], key=cultural_intel_results["content_similarity"].get),
        "most_different_category": min(cultural_intel_results["content_similarity"], key=cultural_intel_results["content_similarity"].get),
        "personality_match": cultural_intel_results["cultural_character"]["dest1_personality"] == cultural_intel_results["cultural_character"]["dest2_personality"]
    }

    return results

def print_enhanced_report(dest1_name, dest2_name, comparison_results, dest1_themes, dest2_themes):
    """Print an enhanced comparison report with cultural intelligence insights."""
    logging.info("\n" + "="*70)
    logging.info(f"🎭 CULTURAL INTELLIGENCE DESTINATION COMPARISON")
    logging.info(f"📍 {dest1_name} vs. {dest2_name}")
    logging.info("="*70)
    
    score = comparison_results.get("overall_similarity_score", 0)
    if score > 0.8:
        similarity_desc = "🔥 Extremely Similar"
    elif score > 0.6:
        similarity_desc = "✅ Highly Similar"
    elif score > 0.4:
        similarity_desc = "🤝 Moderately Similar"
    elif score > 0.2:
        similarity_desc = "🔄 Somewhat Different"
    else:
        similarity_desc = "🌍 Very Different"
    
    logging.info(f"Overall Similarity Score: {score:.1%} ({similarity_desc})")
    
    # Cultural Intelligence Analysis
    ci_results = comparison_results.get("cultural_intelligence", {})
    logging.info(f"\n🎭 CULTURAL INTELLIGENCE ANALYSIS")
    logging.info(f"Cultural Intelligence Score: {comparison_results['scores'].get('cultural_intelligence', 0):.1%}")
    
    # Destination personalities
    char_analysis = ci_results.get("cultural_character", {})
    dest1_personality = char_analysis.get("dest1_personality", "unknown")
    dest2_personality = char_analysis.get("dest2_personality", "unknown")
    
    personality_icons = {
        "cultural": "🎭 Cultural-Focused",
        "practical": "📋 Practical-Focused", 
        "hybrid": "⚖️ Balanced",
        "balanced": "🌈 Well-Rounded",
        "unknown": "❓ Unknown"
    }
    
    logging.info(f"{dest1_name}: {personality_icons.get(dest1_personality, dest1_personality)}")
    logging.info(f"{dest2_name}: {personality_icons.get(dest2_personality, dest2_personality)}")
    
    # Theme distribution comparison
    logging.info(f"\n📊 THEME DISTRIBUTION COMPARISON")
    for proc_type in ["cultural", "practical", "hybrid"]:
        count1 = dest1_themes["theme_stats"][proc_type]
        count2 = dest2_themes["theme_stats"][proc_type]
        total1 = dest1_themes["total_themes"]
        total2 = dest2_themes["total_themes"]
        
        pct1 = (count1 / total1 * 100) if total1 > 0 else 0
        pct2 = (count2 / total2 * 100) if total2 > 0 else 0
        
        icon = CATEGORY_PROCESSING_RULES[proc_type]["icon"]
        logging.info(f"{icon} {proc_type.title()}: {dest1_name} {pct1:.1f}% ({count1}) vs {dest2_name} {pct2:.1f}% ({count2})")
    
    # Content similarity by category
    content_sim = ci_results.get("content_similarity", {})
    logging.info(f"\n🔍 CONTENT SIMILARITY BY CATEGORY")
    for proc_type in ["cultural", "practical", "hybrid"]:
        sim_score = content_sim.get(proc_type, 0)
        icon = CATEGORY_PROCESSING_RULES[proc_type]["icon"]
        logging.info(f"{icon} {proc_type.title()} Themes: {sim_score:.1%} similar")
    
    logging.info(f"\n📈 DETAILED SCORE BREAKDOWN")
    scores = comparison_results.get("scores", {})
    for category, value in scores.items():
        if category == "cultural_intelligence":
            icon = "🎭"
        elif category == "experiential":
            icon = "🎪"
        elif category == "geography":
            icon = "🗺️"
        elif category == "cultural_traditional":
            icon = "🏛️"
        elif category == "logistics":
            icon = "🚀"
        else:
            icon = "📊"
        
        logging.info(f"{icon} {category.replace('_', ' ').title()}: {value:.1%}")
        
    # Key insights
    drivers = comparison_results.get("drivers", {})
    cultural_insights = drivers.get("cultural_insights", {})
    
    logging.info(f"\n🔑 KEY INSIGHTS")
    logging.info(f"Most Similar On: {drivers.get('top_similarity', 'N/A').replace('_', ' ').title()}")
    logging.info(f"Most Different On: {drivers.get('top_difference', 'N/A').replace('_', ' ').title()}")
    logging.info(f"Most Similar Category: {cultural_insights.get('most_similar_category', 'N/A').title()}")
    logging.info(f"Most Different Category: {cultural_insights.get('most_different_category', 'N/A').title()}")
    logging.info(f"Personality Match: {'✅ Yes' if cultural_insights.get('personality_match') else '❌ No'}")
    
    logging.info("\n" + "="*70)

def main():
    parser = argparse.ArgumentParser(description="Compare two destinations with Cultural Intelligence analysis.")
    parser.add_argument("destination1", type=str, help="Name of the first destination (e.g., 'Chicago, United States')")
    parser.add_argument("destination2", type=str, help="Name of the second destination (e.g., 'London, United Kingdom')")
    parser.add_argument("--db_path", type=str, default="enhanced_destination_intelligence.db", help="Path to the database file.")
    args = parser.parse_args()

    try:
        # Load destination data
        db_manager = EnhancedDatabaseManager(db_path=args.db_path)
        
        logging.info(f"🎭 Loading Cultural Intelligence data for {args.destination1}...")
        dest1 = db_manager.get_destination_by_name(args.destination1)
        dest1_themes = load_destination_themes(args.db_path, args.destination1)
        
        logging.info(f"🎭 Loading Cultural Intelligence data for {args.destination2}...")
        dest2 = db_manager.get_destination_by_name(args.destination2)
        dest2_themes = load_destination_themes(args.db_path, args.destination2)
        
        if not dest1 or not dest2:
            logging.error("Could not load one or both destinations from the database. Ensure they have been processed.")
            return

        if dest1_themes["total_themes"] == 0 or dest2_themes["total_themes"] == 0:
            logging.error("One or both destinations have no themes. Run the main analysis first.")
            return

        # Perform enhanced comparison
        comparison_results = compare_destinations(dest1, dest2, dest1_themes, dest2_themes)
        print_enhanced_report(args.destination1, args.destination2, comparison_results, dest1_themes, dest2_themes)

    except Exception as e:
        logging.error(f"An error occurred: {e}", exc_info=True)
    finally:
        if 'db_manager' in locals() and db_manager.conn:
            db_manager.close_db()

if __name__ == "__main__":
    main() 