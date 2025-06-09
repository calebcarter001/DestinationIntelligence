import sqlite3
import os

DB_NAME = "enhanced_destination_intelligence.db"
DB_PATH = os.path.join(os.getcwd(), DB_NAME)
DESTINATION_ID = "dest_chicago,_united_states"

def get_top_10_themes():
    """Connects to the database and prints the top 10 themes based on score."""
    conn = None
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()

        cursor.execute("""
            SELECT name, adjusted_overall_confidence, fit_score, confidence_level
            FROM themes
            WHERE destination_id = ?
            ORDER BY adjusted_overall_confidence DESC, fit_score DESC
            LIMIT 10
        """, (DESTINATION_ID,))
        
        top_themes = cursor.fetchall()

        if not top_themes:
            print(f"No themes found for destination_id: {DESTINATION_ID}")
            return

        print(f"--- Top 10 Themes for {DESTINATION_ID.replace('dest_', '').replace(',_united_states', '').replace('_', ' ').title()} ---")
        for i, theme in enumerate(top_themes):
            name, confidence, fit, level = theme
            print(f"{i+1}. {name}")
            print(f"   - Adjusted Confidence: {confidence:.4f} ({level})")
            print(f"   - Fit Score: {fit:.4f}")
        print("-------------------------------------------------")

    except sqlite3.Error as e:
        print(f"Database error: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
    finally:
        if conn:
            conn.close()

if __name__ == "__main__":
    get_top_10_themes() 