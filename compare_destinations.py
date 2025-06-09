import argparse
import logging
import sqlite3
import json
from typing import Dict, Any, Optional
from src.core.enhanced_database_manager import EnhancedDatabaseManager
from src.core.enhanced_data_models import Destination

logging.basicConfig(level=logging.INFO, format='%(asctime)s - [%(levelname)s] - %(message)s')

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

def compare_destinations(dest1: Destination, dest2: Destination) -> Dict[str, Any]:
    """
    Compares two Destination objects based on the consolidated heuristics.
    """
    if not dest1 or not dest2:
        raise ValueError("Cannot compare None Destination objects.")

    results = {"scores": {}, "drivers": {}}
    
    # Heuristic weights (can be customized for traveler profiles)
    weights = {
        "experiential": 0.25,
        "geography": 0.20,
        "cultural": 0.20,
        "logistics": 0.25,
        "perception": 0.10,
    }

    # 1. Experiential & Activity Alignment
    vibe_sim = len(set(dest1.vibe_descriptors) & set(dest2.vibe_descriptors)) / len(set(dest1.vibe_descriptors) | set(dest2.vibe_descriptors)) if dest1.vibe_descriptors or dest2.vibe_descriptors else 0
    results["scores"]["vibe_similarity"] = vibe_sim
    # (Other experiential heuristics like POI overlap would require theme loading)

    # 2. Geographic & Environmental Fit
    geo_scores = {
        "area": calculate_similarity(dest1.area_km2, dest2.area_km2),
        # Climate/Topography would need parsing from core_geo
    }
    results["scores"]["geographic_fit"] = sum(geo_scores.values()) / len(geo_scores) if geo_scores else 0
    
    # 3. Cultural & Social Context
    cultural_scores = {
        "language": calculate_similarity(dest1.primary_language, dest2.primary_language, is_numeric=False),
        "population": calculate_similarity(dest1.population, dest2.population),
        "religion": len(set(dest1.dominant_religions) & set(dest2.dominant_religions)) / len(set(dest1.dominant_religions) | set(dest2.dominant_religions)) if dest1.dominant_religions or dest2.dominant_religions else 0,
    }
    results["scores"]["cultural_context"] = sum(cultural_scores.values()) / len(cultural_scores) if cultural_scores else 0

    # 4. Practicality & Logistics
    logistics_scores = {
        "gdp": calculate_similarity(dest1.gdp_per_capita_usd, dest2.gdp_per_capita_usd),
        "hdi": calculate_similarity(dest1.hdi, dest2.hdi),
    }
    results["scores"]["logistics_parity"] = sum(logistics_scores.values()) / len(logistics_scores) if logistics_scores else 0

    # 5. Perception & Trend Signals
    perception_scores = {
        "tourist_arrivals": calculate_similarity(dest1.annual_tourist_arrivals, dest2.annual_tourist_arrivals),
        "popularity": calculate_similarity(dest1.popularity_stage, dest2.popularity_stage, is_numeric=False),
    }
    results["scores"]["perception_signals"] = sum(perception_scores.values()) / len(perception_scores) if perception_scores else 0

    # Calculate overall weighted score
    overall_score = (
        results["scores"]["vibe_similarity"] * weights["experiential"] +
        results["scores"]["geographic_fit"] * weights["geography"] +
        results["scores"]["cultural_context"] * weights["cultural"] +
        results["scores"]["logistics_parity"] * weights["logistics"] +
        results["scores"]["perception_signals"] * weights["perception"]
    )
    results["overall_similarity_score"] = overall_score
    
    # Identify top drivers
    # (simplified for now)
    results["drivers"]["top_similarity"] = max(results["scores"], key=results["scores"].get)
    results["drivers"]["top_difference"] = min(results["scores"], key=results["scores"].get)

    return results

def print_report(dest1_name, dest2_name, comparison_results):
    """Prints a formatted comparison report."""
    logging.info("\n" + "="*50)
    logging.info(f"Destination Similarity Report: {dest1_name} vs. {dest2_name}")
    logging.info("="*50)
    
    score = comparison_results.get("overall_similarity_score", 0)
    logging.info(f"Overall Similarity Score: {score:.2%} ({'Highly Similar' if score > 0.7 else 'Moderately Similar' if score > 0.4 else 'Dissimilar'})")
    
    logging.info("\n--- Score Breakdown ---")
    for category, value in comparison_results.get("scores", {}).items():
        logging.info(f"- {category.replace('_', ' ').title()}: {value:.2%}")
        
    logging.info("\n--- Key Drivers ---")
    top_sim = comparison_results.get("drivers", {}).get("top_similarity", "N/A").replace('_', ' ').title()
    top_diff = comparison_results.get("drivers", {}).get("top_difference", "N/A").replace('_', ' ').title()
    logging.info(f"Most Aligned On: {top_sim}")
    logging.info(f"Most Different On: {top_diff}")
    
    logging.info("\n" + "="*50)

def main():
    parser = argparse.ArgumentParser(description="Compare two destinations based on stored heuristic data.")
    parser.add_argument("destination1", type=str, help="Name of the first destination (e.g., 'Chicago, United States')")
    parser.add_argument("destination2", type=str, help="Name of the second destination (e.g., 'London, United Kingdom')")
    parser.add_argument("--db_path", type=str, default="enhanced_destination_intelligence.db", help="Path to the database file.")
    args = parser.parse_args()

    try:
        db_manager = EnhancedDatabaseManager(db_path=args.db_path)
        
        logging.info(f"Loading data for {args.destination1}...")
        dest1 = db_manager.get_destination_by_name(args.destination1)
        
        logging.info(f"Loading data for {args.destination2}...")
        dest2 = db_manager.get_destination_by_name(args.destination2)
        
        if not dest1 or not dest2:
            logging.error("Could not load one or both destinations from the database. Ensure they have been processed.")
            return

        comparison_results = compare_destinations(dest1, dest2)
        print_report(args.destination1, args.destination2, comparison_results)

    except Exception as e:
        logging.error(f"An error occurred: {e}", exc_info=True)
    finally:
        if 'db_manager' in locals() and db_manager.conn:
            db_manager.close_db()

if __name__ == "__main__":
    main() 