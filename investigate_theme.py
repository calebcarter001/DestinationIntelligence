import sqlite3
import os
import json
import sys

DB_NAME = "enhanced_destination_intelligence.db"
DB_PATH = os.path.join(os.getcwd(), DB_NAME)
DESTINATION_ID = "dest_chicago,_united_states"

def investigate_theme(theme_name):
    """Fetches and displays the evidence for a specific theme."""
    conn = None
    try:
        conn = sqlite3.connect(DB_PATH)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()

        print(f"--- Investigating Theme: '{theme_name}' for {DESTINATION_ID.replace('dest_', '').replace(',_united_states', '').replace('_', ' ').title()} ---")

        # First, get the theme's source_evidence_ids
        cursor.execute("""
            SELECT source_evidence_ids
            FROM themes
            WHERE name = ? AND destination_id = ?
        """, (theme_name, DESTINATION_ID))
        
        theme_row = cursor.fetchone()

        if not theme_row:
            print(f"Theme '{theme_name}' not found for this destination.")
            return

        evidence_ids_str = theme_row['source_evidence_ids']
        if not evidence_ids_str:
            print("No source evidence IDs found for this theme.")
            return

        try:
            evidence_ids = json.loads(evidence_ids_str)
            if not evidence_ids:
                print("Evidence ID list is empty.")
                return
        except json.JSONDecodeError:
            print("Error: Could not parse source_evidence_ids JSON.")
            return

        print(f"Found {len(evidence_ids)} evidence snippet(s) for this theme. Fetching details...")
        print("-" * 20)

        # Create a placeholder string for the IN clause
        placeholders = ','.join('?' for _ in evidence_ids)
        
        # Fetch details for each piece of evidence
        cursor.execute(f"""
            SELECT id, source_url, text_snippet AS content_snippet, confidence AS relevance_score
            FROM evidence
            WHERE id IN ({placeholders})
        """, evidence_ids)

        evidence_rows = cursor.fetchall()

        for evidence in evidence_rows:
            print(f"Evidence ID: {evidence['id']}")
            print(f"  URL: {evidence['source_url']}")
            print(f"  Relevance Score: {evidence['relevance_score']:.4f}")
            print(f"  Snippet: \"{evidence['content_snippet']}\"")
            print("-" * 20)

    except sqlite3.Error as e:
        print(f"Database error: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}", file=sys.stderr)
    finally:
        if conn:
            conn.close()

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python investigate_theme.py \"<Theme Name>\"")
        sys.exit(1)
    
    theme_to_investigate = sys.argv[1]
    investigate_theme(theme_to_investigate) 