#!/usr/bin/env python3
"""
Query Enhanced Destination Data from the new schema
"""

import sqlite3
import json
import sys
from datetime import datetime

def query_destinations():
    conn = sqlite3.connect('real_destination_intelligence.db')
    cursor = conn.cursor()
    
    print("\n🌍 DESTINATIONS IN DATABASE")
    print("=" * 60)
    
    cursor.execute("""
        SELECT id, names, country_code, timezone, last_updated, destination_revision
        FROM destinations
        ORDER BY last_updated DESC
    """)
    
    destinations = cursor.fetchall()
    for dest in destinations:
        names = json.loads(dest[1]) if dest[1] else []
        print(f"\n📍 {', '.join(names)} ({dest[0]})")
        print(f"   Country: {dest[2]} | Timezone: {dest[3]}")
        print(f"   Last Updated: {dest[4]} | Revision: {dest[5]}")
        
        # Count themes
        cursor.execute("SELECT COUNT(*) FROM themes WHERE destination_id = ?", (dest[0],))
        theme_count = cursor.fetchone()[0]
        
        # Count evidence
        cursor.execute("SELECT COUNT(*) FROM evidence WHERE destination_id = ?", (dest[0],))
        evidence_count = cursor.fetchone()[0]
        
        # Count dimensions
        cursor.execute("SELECT COUNT(*) FROM dimensions WHERE destination_id = ?", (dest[0],))
        dimension_count = cursor.fetchone()[0]
        
        print(f"   📊 Themes: {theme_count} | Evidence: {evidence_count} | Dimensions: {dimension_count}")
    
    conn.close()

def query_themes(destination_id=None):
    conn = sqlite3.connect('real_destination_intelligence.db')
    cursor = conn.cursor()
    
    print("\n🎭 THEMES ANALYSIS")
    print("=" * 60)
    
    if destination_id:
        cursor.execute("""
            SELECT theme_id, name, macro_category, micro_category, 
                   description, confidence_level, fit_score
            FROM themes
            WHERE destination_id = ?
            ORDER BY fit_score DESC
        """, (destination_id,))
    else:
        cursor.execute("""
            SELECT t.theme_id, t.name, t.macro_category, t.micro_category, 
                   t.description, t.confidence_level, t.fit_score, d.names
            FROM themes t
            JOIN destinations d ON t.destination_id = d.id
            ORDER BY t.fit_score DESC
            LIMIT 20
        """)
    
    themes = cursor.fetchall()
    for theme in themes:
        print(f"\n🏷️  {theme[1]}")
        print(f"   Category: {theme[2]} > {theme[3]}")
        print(f"   Description: {theme[4][:100]}...")
        print(f"   Confidence: {theme[5]} | Fit Score: {theme[6]:.3f}")
        
        if not destination_id:
            names = json.loads(theme[7]) if theme[7] else []
            print(f"   Destination: {', '.join(names)}")
        
        # Count evidence for this theme
        cursor.execute("""
            SELECT COUNT(*) FROM theme_evidence WHERE theme_id = ?
        """, (theme[0],))
        evidence_count = cursor.fetchone()[0]
        print(f"   Evidence pieces: {evidence_count}")
    
    conn.close()

def query_evidence(destination_id=None, limit=10):
    conn = sqlite3.connect('real_destination_intelligence.db')
    cursor = conn.cursor()
    
    print("\n📚 EVIDENCE SOURCES")
    print("=" * 60)
    
    if destination_id:
        cursor.execute("""
            SELECT source_url, source_category, authority_weight, 
                   text_snippet, confidence, published_date
            FROM evidence
            WHERE destination_id = ?
            ORDER BY authority_weight DESC, confidence DESC
            LIMIT ?
        """, (destination_id, limit))
    else:
        cursor.execute("""
            SELECT e.source_url, e.source_category, e.authority_weight, 
                   e.text_snippet, e.confidence, e.published_date, d.names
            FROM evidence e
            JOIN destinations d ON e.destination_id = d.id
            ORDER BY e.authority_weight DESC, e.confidence DESC
            LIMIT ?
        """, (limit,))
    
    evidence = cursor.fetchall()
    for ev in evidence:
        print(f"\n🔗 {ev[0][:60]}...")
        print(f"   Category: {ev[1]} | Authority: {ev[2]:.2f} | Confidence: {ev[4]:.2f}")
        print(f"   Published: {ev[5] or 'Unknown'}")
        print(f"   Snippet: {ev[3][:150]}...")
        
        if not destination_id:
            names = json.loads(ev[6]) if ev[6] else []
            print(f"   Destination: {', '.join(names)}")
    
    conn.close()

def query_dimensions(destination_id=None):
    conn = sqlite3.connect('real_destination_intelligence.db')
    cursor = conn.cursor()
    
    print("\n📏 DESTINATION DIMENSIONS")
    print("=" * 60)
    
    if destination_id:
        cursor.execute("""
            SELECT dimension_name, value, unit, confidence
            FROM dimensions
            WHERE destination_id = ?
            ORDER BY dimension_name
        """, (destination_id,))
    else:
        cursor.execute("""
            SELECT dim.dimension_name, dim.value, dim.unit, dim.confidence, d.names
            FROM dimensions dim
            JOIN destinations d ON dim.destination_id = d.id
            ORDER BY dim.dimension_name
        """)
    
    dimensions = cursor.fetchall()
    current_category = None
    
    for dim in dimensions:
        # Group by category (before underscore)
        category = dim[0].split('_')[0]
        if category != current_category:
            current_category = category
            print(f"\n{category.upper()}")
        
        print(f"  {dim[0].replace('_', ' ').title()}: {dim[1]:.2f}{dim[2] or ''} (confidence: {dim[3]:.2f})")
        
        if not destination_id:
            names = json.loads(dim[4]) if dim[4] else []
            print(f"    → {', '.join(names)}")
    
    conn.close()

if __name__ == "__main__":
    if len(sys.argv) > 1:
        command = sys.argv[1].lower()
        
        if command == "destinations":
            query_destinations()
        elif command == "themes":
            dest_id = sys.argv[2] if len(sys.argv) > 2 else None
            query_themes(dest_id)
        elif command == "evidence":
            dest_id = sys.argv[2] if len(sys.argv) > 2 else None
            query_evidence(dest_id)
        elif command == "dimensions":
            dest_id = sys.argv[2] if len(sys.argv) > 2 else None
            query_dimensions(dest_id)
        elif command == "all":
            dest_id = sys.argv[2] if len(sys.argv) > 2 else None
            if dest_id:
                print(f"\n📍 Analysis for destination: {dest_id}")
            query_destinations()
            query_themes(dest_id)
            query_evidence(dest_id, 5)
            query_dimensions(dest_id)
    else:
        print("🔍 Enhanced Destination Data Query Tool")
        print("\nUsage:")
        print("  python query_enhanced_data.py destinations")
        print("  python query_enhanced_data.py themes [destination_id]")
        print("  python query_enhanced_data.py evidence [destination_id]")
        print("  python query_enhanced_data.py dimensions [destination_id]")
        print("  python query_enhanced_data.py all [destination_id]")
        print("\nExamples:")
        print("  python query_enhanced_data.py destinations")
        print("  python query_enhanced_data.py all legacy_bend_oregon") 