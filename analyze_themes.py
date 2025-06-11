import json
import os
import sqlite3
import yaml
from datetime import datetime
from collections import defaultdict

# Determine the project root and construct the database path
DB_NAME = "enhanced_destination_intelligence.db"
PROJECT_ROOT = os.getcwd()
DB_PATH = os.path.join(PROJECT_ROOT, DB_NAME)
OUTPUTS_DIR = os.path.join(PROJECT_ROOT, "outputs", "theme_reports")
os.makedirs(OUTPUTS_DIR, exist_ok=True)

# CULTURAL INTELLIGENCE: Define category processing rules
CATEGORY_PROCESSING_RULES = {
    "cultural": {
        "categories": [
            "Cultural Identity & Atmosphere", "Authentic Experiences", "Distinctive Features",
            "Local Character & Vibe", "Artistic & Creative Scene"
        ],
        "color": "#9C27B0",  # Purple for cultural
        "icon": "🎭"
    },
    "practical": {
        "categories": [
            "Safety & Security", "Transportation & Access", "Budget & Costs", 
            "Health & Medical", "Logistics & Planning", "Visa & Documentation"
        ],
        "color": "#2196F3",  # Blue for practical
        "icon": "📋"
    },
    "hybrid": {
        "categories": [
            "Food & Dining", "Entertainment & Nightlife", "Nature & Outdoor",
            "Shopping & Local Craft", "Family & Education", "Health & Wellness"
        ],
        "color": "#4CAF50",  # Green for hybrid
        "icon": "⚖️"
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

def load_config():
    """Loads the application configuration from config.yaml."""
    config_path = os.path.join(PROJECT_ROOT, "config", "config.yaml")
    if not os.path.exists(config_path):
        print(f"Error: config.yaml not found at {config_path}")
        return None
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def get_db_connection():
    """Establishes a connection to the SQLite database."""
    try:
        conn = sqlite3.connect(DB_PATH)
        conn.row_factory = sqlite3.Row # Access columns by name
        return conn
    except sqlite3.Error as e:
        print(f"Error connecting to database {DB_PATH}: {e}")
        raise

def load_themes_from_db(db_path, destination_name):
    """Load themes from database for a specific destination"""
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # Convert destination name to destination ID format
        dest_id = f"dest_{destination_name.replace(', ', '_').replace(' ', '_').lower()}"
        
        cursor.execute("""
            SELECT 
                theme_id, name, macro_category, micro_category, description,
                fit_score, confidence_level, adjusted_overall_confidence,
                confidence_breakdown, tags
            FROM themes
            WHERE destination_id = ?
        """, (dest_id,))
        
        rows = cursor.fetchall()
        themes = []
        
        for row in rows:
            theme_dict = {
                "theme_id": row[0],
                "name": row[1],
                "macro_category": row[2],
                "micro_category": row[3],
                "description": row[4],
                "fit_score": row[5] or 0.0,
                "confidence_level": row[6],
                "overall_confidence": row[7] or 0.0,
                "confidence_breakdown": row[8],
                "tags": row[9]
            }
            
            # Add processing type based on macro category
            theme_dict["processing_type"] = get_processing_type(theme_dict["macro_category"])
            
            themes.append(theme_dict)
        
        conn.close()
        return themes
        
    except sqlite3.Error as e:
        print(f"Database error loading themes: {e}")
        return []
    except Exception as e:
        print(f"Error loading themes: {e}")
        return []

def calculate_cultural_intelligence_metrics(themes_data):
    """Calculate cultural intelligence specific metrics"""
    metrics = {
        "total_themes": len(themes_data),
        "category_breakdown": {"cultural": 0, "practical": 0, "hybrid": 0, "unknown": 0},
        "theme_distribution": {"cultural": 0, "practical": 0, "hybrid": 0, "unknown": 0},  # Add alias for tests
        "avg_confidence_by_type": {"cultural": 0, "practical": 0, "hybrid": 0},
        "high_confidence_themes": {"cultural": 0, "practical": 0, "hybrid": 0},
        "distinctiveness_analysis": {},
        "authenticity_indicators": {},
        "top_cultural_themes": [],
        "top_practical_themes": [],
        "cultural_practical_ratio": 0.0  # Add the missing key that tests expect
    }
    
    confidence_sums = {"cultural": 0, "practical": 0, "hybrid": 0}
    confidence_counts = {"cultural": 0, "practical": 0, "hybrid": 0}
    
    for theme in themes_data:
        proc_type = get_processing_type(theme.get('metadata', {}).get('macro_category') or theme.get('macro_category'))
        metrics["category_breakdown"][proc_type] += 1
        metrics["theme_distribution"][proc_type] += 1  # Update both keys
        
        confidence = theme.get('overall_confidence', 0)
        if proc_type in confidence_sums:
            confidence_sums[proc_type] += confidence
            confidence_counts[proc_type] += 1
            
            # Count high confidence themes (>0.7)
            if confidence > 0.7:
                metrics["high_confidence_themes"][proc_type] += 1
            
            # Track top themes for each type
            if proc_type == "cultural":
                metrics["top_cultural_themes"].append({
                    "name": theme.get('name', ''),
                    "confidence": confidence,
                    "category": theme.get('macro_category', '')
                })
            elif proc_type == "practical":
                metrics["top_practical_themes"].append({
                    "name": theme.get('name', ''),
                    "confidence": confidence,
                    "category": theme.get('macro_category', '')
                })
    
    # Calculate averages
    for proc_type in confidence_sums:
        if confidence_counts[proc_type] > 0:
            metrics["avg_confidence_by_type"][proc_type] = confidence_sums[proc_type] / confidence_counts[proc_type]
    
    # Calculate cultural to practical ratio
    cultural_count = metrics["category_breakdown"]["cultural"]
    practical_count = metrics["category_breakdown"]["practical"]
    
    if practical_count > 0:
        metrics["cultural_practical_ratio"] = cultural_count / practical_count
    elif cultural_count > 0:
        metrics["cultural_practical_ratio"] = float('inf')  # All cultural, no practical
    else:
        metrics["cultural_practical_ratio"] = 0.0  # No themes at all
    
    # Sort top themes by confidence
    metrics["top_cultural_themes"] = sorted(metrics["top_cultural_themes"], key=lambda x: x["confidence"], reverse=True)[:5]
    metrics["top_practical_themes"] = sorted(metrics["top_practical_themes"], key=lambda x: x["confidence"], reverse=True)[:5]
    
    # Add distinctiveness and authenticity analysis
    metrics["distinctiveness_analysis"] = {
        "unique_themes": len([t for t in themes_data if "unique" in t.get("name", "").lower()]),
        "signature_themes": len([t for t in themes_data if "signature" in t.get("name", "").lower()]),
        "common_themes": len([t for t in themes_data if any(word in t.get("name", "").lower() for word in ["popular", "common", "typical"])])
    }
    
    metrics["authenticity_indicators"] = {
        "local_source_themes": len([t for t in themes_data if "local" in t.get("description", "").lower()]),
        "insider_themes": len([t for t in themes_data if any(word in t.get("description", "").lower() for word in ["insider", "secret", "hidden"])]),
        "authentic_experiences": len([t for t in themes_data if "authentic" in t.get("description", "").lower()])
    }
    
    return metrics

def generate_cultural_report(destination_name, themes_data=None, cultural_metrics=None):
    """Generate a cultural intelligence report for a destination"""
    try:
        if themes_data is None:
            # Load themes from database if not provided
            db_path = "enhanced_destination_intelligence.db"
            themes_data = load_themes_from_db(db_path, destination_name)
        
        # Calculate cultural intelligence metrics if not provided
        if cultural_metrics is None:
            cultural_metrics = calculate_cultural_intelligence_metrics(themes_data)
        
        # Generate formatted report lines
        report_lines = []
        
        # Header
        report_lines.append(f"Cultural Intelligence Report for {destination_name}")
        report_lines.append("=" * 60)
        report_lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report_lines.append("")
        
        # Summary metrics
        total_themes = len(themes_data) if isinstance(themes_data, list) else cultural_metrics.get("total_themes", 0)
        cultural_count = cultural_metrics.get("category_breakdown", {}).get("cultural", 0)
        practical_count = cultural_metrics.get("category_breakdown", {}).get("practical", 0)
        hybrid_count = cultural_metrics.get("category_breakdown", {}).get("hybrid", 0)
        
        report_lines.append("📊 THEME DISTRIBUTION:")
        report_lines.append(f"  Total Themes: {total_themes}")
        report_lines.append(f"  🎭 Cultural: {cultural_count} ({cultural_count/max(total_themes,1)*100:.1f}%)")
        report_lines.append(f"  📋 Practical: {practical_count} ({practical_count/max(total_themes,1)*100:.1f}%)")
        report_lines.append(f"  ⚖️ Hybrid: {hybrid_count} ({hybrid_count/max(total_themes,1)*100:.1f}%)")
        report_lines.append("")
        
        # Cultural vs Practical ratio
        ratio = cultural_metrics.get("cultural_practical_ratio", 0.0)
        if ratio == float('inf'):
            ratio_text = "All Cultural (∞)"
        elif ratio == 0.0:
            ratio_text = "No themes"
        else:
            ratio_text = f"{ratio:.2f}"
        
        report_lines.append(f"🎯 CULTURAL/PRACTICAL RATIO: {ratio_text}")
        report_lines.append("")
        
        # Confidence analysis
        avg_conf = cultural_metrics.get("avg_confidence_by_type", {})
        report_lines.append("🔍 AVERAGE CONFIDENCE BY TYPE:")
        for proc_type in ["cultural", "practical", "hybrid"]:
            conf = avg_conf.get(proc_type, 0.0)
            report_lines.append(f"  {proc_type.title()}: {conf:.2f}")
        report_lines.append("")
        
        # Top themes
        top_cultural = cultural_metrics.get("top_cultural_themes", [])[:3]
        if top_cultural:
            report_lines.append("🎭 TOP CULTURAL THEMES:")
            for i, theme in enumerate(top_cultural, 1):
                name = theme.get("name", "Unknown")
                conf = theme.get("confidence", 0.0)
                report_lines.append(f"  {i}. {name} (confidence: {conf:.2f})")
            report_lines.append("")
        
        top_practical = cultural_metrics.get("top_practical_themes", [])[:3]
        if top_practical:
            report_lines.append("📋 TOP PRACTICAL THEMES:")
            for i, theme in enumerate(top_practical, 1):
                name = theme.get("name", "Unknown")
                conf = theme.get("confidence", 0.0)
                report_lines.append(f"  {i}. {name} (confidence: {conf:.2f})")
            report_lines.append("")
        
        # Distinctiveness analysis
        distinct = cultural_metrics.get("distinctiveness_analysis", {})
        report_lines.append("✨ DISTINCTIVENESS ANALYSIS:")
        report_lines.append(f"  Unique themes: {distinct.get('unique_themes', 0)}")
        report_lines.append(f"  Signature themes: {distinct.get('signature_themes', 0)}")
        report_lines.append(f"  Common themes: {distinct.get('common_themes', 0)}")
        report_lines.append("")
        
        # Authenticity indicators
        auth = cultural_metrics.get("authenticity_indicators", {})
        report_lines.append("🔐 AUTHENTICITY INDICATORS:")
        report_lines.append(f"  Local source themes: {auth.get('local_source_themes', 0)}")
        report_lines.append(f"  Insider themes: {auth.get('insider_themes', 0)}")
        report_lines.append(f"  Authentic experiences: {auth.get('authentic_experiences', 0)}")
        
        return report_lines
        
    except Exception as e:
        return [
            f"Error generating cultural report for {destination_name}",
            f"Error: {str(e)}",
            f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
        ]

def fetch_and_save_comprehensive_report(destination_id: str):
    """Connects to the database, retrieves all theme details for a specific destination, and saves them to a JSON file."""
    conn = None
    try:
        conn = get_db_connection()
        cursor = conn.cursor()

        # Sanitize destination_id for use in filename
        safe_filename_dest = destination_id.replace(",", "").replace(" ", "_").lower()
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        report_filename = f"cultural_intelligence_report_{safe_filename_dest}_{timestamp}.json"
        report_path = os.path.join(OUTPUTS_DIR, report_filename)

        print(f"🎭 CULTURAL INTELLIGENCE: Querying all themes and details for destination_id: {destination_id}")
        cursor.execute("""
            SELECT 
                theme_id, 
                name, 
                fit_score, 
                confidence_breakdown, 
                confidence_level,
                macro_category, 
                micro_category, 
                description, 
                tags, 
                sentiment_analysis, 
                temporal_analysis, 
                authentic_insights, 
                local_authorities,
                source_evidence_ids,
                destination_id,
                traveler_relevance_factor,
                adjusted_overall_confidence
            FROM themes
            WHERE destination_id = ?
        """, (destination_id,))
        
        rows = cursor.fetchall()
        print(f"📊 Found {len(rows)} themes for {destination_id}.")

        if not rows:
            print("No themes found for this destination. JSON report will be an empty list.")
            processed_themes = []
        else:
            processed_themes = []
            for row in rows:
                theme_dict = dict(row) # Convert sqlite3.Row to dict
                fit_score = theme_dict.get('fit_score') if theme_dict.get('fit_score') is not None else 0.0
                
                # overall_confidence is now directly from adjusted_overall_confidence if available
                adjusted_confidence = theme_dict.get('adjusted_overall_confidence')
                primary_confidence_for_sorting = adjusted_confidence if adjusted_confidence is not None else 0.0
                
                # Keep original confidence_breakdown as is from DB
                confidence_breakdown_full = None
                confidence_breakdown_str = theme_dict.get('confidence_breakdown')
                if confidence_breakdown_str:
                    try:
                        confidence_breakdown_full = json.loads(confidence_breakdown_str)
                    except json.JSONDecodeError:
                        print(f"Warning: Could not parse confidence_breakdown for theme '{theme_dict.get('name')}': {confidence_breakdown_str[:100]}...")
                        confidence_breakdown_full = {"error": "Failed to parse JSON from DB", "original_value": confidence_breakdown_str}
                
                # CULTURAL INTELLIGENCE: Determine processing type and add category info
                proc_type = get_processing_type(theme_dict.get('macro_category'))
                category_info = CATEGORY_PROCESSING_RULES.get(proc_type, {})
                
                # Consolidate metadata fields
                metadata_fields = [
                    'theme_id', 'destination_id', 'macro_category', 'micro_category', 
                    'description', 'tags', 'sentiment_analysis', 'temporal_analysis', 
                    'authentic_insights', 'local_authorities', 'source_evidence_ids'
                ]
                theme_metadata = {}
                for field in metadata_fields:
                    raw_value = theme_dict.get(field)
                    # Attempt to parse if it's a string that looks like valid JSON
                    if isinstance(raw_value, str):
                        if raw_value.strip().startswith(("{", "[")) and raw_value.strip().endswith(("}", "]")):
                            try:
                                theme_metadata[field] = json.loads(raw_value)
                            except json.JSONDecodeError:
                                theme_metadata[field] = raw_value
                        else:
                            theme_metadata[field] = raw_value
                    else:
                        theme_metadata[field] = raw_value

                processed_themes.append({
                    'name': theme_dict.get('name'),
                    'fit_score': float(fit_score),
                    'overall_confidence': float(primary_confidence_for_sorting), 
                    'confidence_level': theme_dict.get('confidence_level', 'N/A'),
                    'confidence_breakdown': confidence_breakdown_full,
                    'traveler_relevance_factor': theme_dict.get('traveler_relevance_factor'),
                    # CULTURAL INTELLIGENCE: Add processing type and category information
                    'processing_type': proc_type,
                    'category_color': category_info.get('color', '#666666'),
                    'category_icon': category_info.get('icon', '📌'),
                    'metadata': theme_metadata
                })

        # CULTURAL INTELLIGENCE: Calculate enhanced metrics
        cultural_metrics = calculate_cultural_intelligence_metrics(processed_themes)
        
        # Sort by the new primary confidence (adjusted), then fit_score
        sorted_themes = sorted(processed_themes, key=lambda x: (x['overall_confidence'], x['fit_score']), reverse=True)
        
        # Create comprehensive report with cultural intelligence analysis
        comprehensive_report = {
            "destination_id": destination_id,
            "analysis_timestamp": datetime.now().isoformat(),
            "cultural_intelligence_metrics": cultural_metrics,
            "themes": sorted_themes,
            "summary": {
                "total_themes": len(sorted_themes),
                "avg_overall_confidence": sum(t['overall_confidence'] for t in sorted_themes) / len(sorted_themes) if sorted_themes else 0,
                "cultural_vs_practical_ratio": cultural_metrics["cultural_practical_ratio"],
                "high_confidence_percentage": sum(cultural_metrics["high_confidence_themes"].values()) / max(len(sorted_themes), 1) * 100
            }
        }
        
        # Save to JSON file
        try:
            with open(report_path, 'w', encoding='utf-8') as f:
                json.dump(comprehensive_report, f, indent=4)
            print(f"\n✅ Successfully generated CULTURAL INTELLIGENCE report: {report_path}")
            
            # Print summary statistics
            print(f"\n📊 CULTURAL INTELLIGENCE ANALYSIS SUMMARY:")
            print(f"   🎭 Cultural themes: {cultural_metrics['category_breakdown']['cultural']}")
            print(f"   📋 Practical themes: {cultural_metrics['category_breakdown']['practical']}")
            print(f"   ⚖️ Hybrid themes: {cultural_metrics['category_breakdown']['hybrid']}")
            print(f"   🎯 Cultural vs Practical ratio: {comprehensive_report['summary']['cultural_vs_practical_ratio']:.2f}")
            print(f"   🏆 High confidence themes: {sum(cultural_metrics['high_confidence_themes'].values())}/{len(sorted_themes)}")
            
        except IOError as e:
            print(f"\nError writing JSON report: {e}")
        except TypeError as e:
            print(f"\nError serializing data to JSON: {e}. Check for non-serializable data types.")

    except sqlite3.Error as e:
        print(f"Database error: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
    finally:
        if conn:
            conn.close()
            print("\nDatabase connection closed.")

if __name__ == "__main__":
    config = load_config()
    if config:
        destinations = config.get("destinations", [])
        if not destinations:
            print("No destinations found in config.yaml.")
        else:
            print(f"🚀 Starting CULTURAL INTELLIGENCE analysis for {len(destinations)} destinations...")
            for dest_name in destinations:
                # Convert "City, Country" to "dest_city_country" format
                dest_id = f"dest_{dest_name.replace(', ', '_').replace(' ', '_').lower()}"
                fetch_and_save_comprehensive_report(dest_id) 