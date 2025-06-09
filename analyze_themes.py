import json
import os
import sqlite3

# Determine the project root and construct the database path
DB_NAME = "enhanced_destination_intelligence.db"
DB_PATH = os.path.join(os.getcwd(), DB_NAME)
# New JSON output path
JSON_REPORT_PATH = os.path.join(os.getcwd(), "comprehensive_database_report.json") 

DESTINATION_ID_TO_QUERY = "dest_chicago_united_states"

def get_db_connection():
    """Establishes a connection to the SQLite database."""
    try:
        conn = sqlite3.connect(DB_PATH)
        conn.row_factory = sqlite3.Row # Access columns by name
        return conn
    except sqlite3.Error as e:
        print(f"Error connecting to database {DB_PATH}: {e}")
        raise

def fetch_and_save_comprehensive_report():
    """Connects to the database, retrieves all theme details, and saves them to a JSON file."""
    conn = None
    try:
        conn = get_db_connection()
        cursor = conn.cursor()

        print(f"Querying all themes and details for destination_id: {DESTINATION_ID_TO_QUERY}")
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
        """, (DESTINATION_ID_TO_QUERY,))
        
        rows = cursor.fetchall()
        print(f"Found {len(rows)} themes for {DESTINATION_ID_TO_QUERY}.")

        if not rows:
            print("No themes found for this destination. JSON report will be an empty list.")
            processed_themes = []
        else:
            processed_themes = []
            for row in rows:
                theme_dict = dict(row) # Convert sqlite3.Row to dict
                fit_score = theme_dict.get('fit_score') if theme_dict.get('fit_score') is not None else 0.0
                
                # overall_confidence is now directly from adjusted_overall_confidence if available
                # The original overall_confidence is still in confidence_breakdown
                adjusted_confidence = theme_dict.get('adjusted_overall_confidence')
                # Use original from breakdown if adjusted is missing (e.g. older data not yet reprocessed)
                # but for new data, adjusted_overall_confidence should be the primary one to use for sorting.
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
                
                # Consolidate all other fields into a 'metadata' sub-dictionary for the JSON output
                # This helps keep the top-level theme object cleaner
                metadata_fields = [
                    'theme_id', 'destination_id', 'macro_category', 'micro_category', 
                    'description', 'tags', 'sentiment_analysis', 'temporal_analysis', 
                    'authentic_insights', 'local_authorities', 'source_evidence_ids'
                ]
                theme_metadata = {}
                for field in metadata_fields:
                    raw_value = theme_dict.get(field)
                    # Attempt to parse if it's a string that looks like valid JSON, otherwise keep as is
                    if isinstance(raw_value, str):
                        if raw_value.strip().startswith(("{", "[")) and raw_value.strip().endswith(("}", "]")):
                            try:
                                theme_metadata[field] = json.loads(raw_value)
                            except json.JSONDecodeError:
                                theme_metadata[field] = raw_value # Store as string if parsing fails
                        else:
                            theme_metadata[field] = raw_value # Store non-JSON string as is
                    else:
                        theme_metadata[field] = raw_value # Store non-string types as is

                processed_themes.append({
                    'name': theme_dict.get('name'),
                    'fit_score': float(fit_score),
                    # IMPORTANT: For sorting and primary display, use adjusted_overall_confidence
                    'overall_confidence': float(primary_confidence_for_sorting), 
                    'confidence_level': theme_dict.get('confidence_level', 'N/A'),
                    'confidence_breakdown': confidence_breakdown_full, # Original breakdown for details
                    'traveler_relevance_factor': theme_dict.get('traveler_relevance_factor'),
                    # adjusted_overall_confidence is now the main 'overall_confidence' for the JSON output
                    'metadata': theme_metadata
                })

        # Sort by the new primary confidence (adjusted), then fit_score
        sorted_themes = sorted(processed_themes, key=lambda x: (x['overall_confidence'], x['fit_score']), reverse=True)
        
        # Save to JSON file
        try:
            with open(JSON_REPORT_PATH, 'w', encoding='utf-8') as f:
                json.dump(sorted_themes, f, indent=4)
            print(f"\nSuccessfully generated comprehensive JSON report: {JSON_REPORT_PATH}")
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
    fetch_and_save_comprehensive_report() 