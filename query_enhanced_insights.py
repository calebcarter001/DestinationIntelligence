#!/usr/bin/env python3
"""
Query Enhanced Destination Insights

This script demonstrates how to retrieve and display the rich, structured
destination insights stored by the enhanced analysis system.
"""

import sys
import os
import json
from typing import List, Dict, Any

# Add project root to path
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)

from src.core.database_manager import DatabaseManager
from src.tools.enhanced_database_storage_tool import EnhancedDatabaseStorageTool

def display_insights_summary(destination_name: str, db_path: str = "real_destination_intelligence.db"):
    """Display a summary of enhanced insights for a destination"""
    
    db_manager = DatabaseManager(db_path=db_path)
    enhanced_storage = EnhancedDatabaseStorageTool(db_manager=db_manager)
    
    print(f"\n🌟 Enhanced Destination Insights for: {destination_name}")
    print("=" * 60)
    
    categories = ["attractions", "hotels", "restaurants", "activities", "neighborhoods", "practical_info"]
    
    for category in categories:
        insights = enhanced_storage.get_stored_insights(
            destination_name=destination_name, 
            category=category,
            min_confidence=0.5
        )
        
        if insights:
            category_emoji = {
                "attractions": "🏛️",
                "hotels": "🏨", 
                "restaurants": "🍽️",
                "activities": "🎯",
                "neighborhoods": "🏘️",
                "practical_info": "ℹ️"
            }
            
            print(f"\n{category_emoji.get(category, '📍')} {category.upper()} ({len(insights)} found)")
            print("-" * 40)
            
            for i, insight in enumerate(insights[:3], 1):  # Show top 3
                print(f"{i}. {insight['name']}")
                print(f"   Description: {insight['description'][:150]}...")
                print(f"   Confidence: {insight['confidence_score']:.2f}")
                
                if insight['highlights']:
                    print(f"   Highlights: {', '.join(insight['highlights'][:2])}")
                
                if insight['practical_info']:
                    practical_items = []
                    for key, value in insight['practical_info'].items():
                        practical_items.append(f"{key}: {value}")
                    if practical_items:
                        print(f"   Practical: {', '.join(practical_items[:2])}")
                
                print()
    
    # Get summary
    try:
        if not db_manager.conn:
            print("Error: Database connection is None")
            return
            
        cursor = db_manager.conn.cursor()
        cursor.execute("""
            SELECT summary, total_attractions, total_hotels, total_restaurants, 
                   total_activities, total_neighborhoods, total_practical_info,
                   analysis_timestamp
            FROM enhanced_destination_analysis 
            WHERE destination_name = ?
            ORDER BY analysis_timestamp DESC LIMIT 1
        """, (destination_name,))
        
        row = cursor.fetchone()
        if row:
            print(f"\n📝 DESTINATION SUMMARY")
            print("-" * 40)
            print(f"{row[0]}")
            print(f"\n📊 ANALYSIS STATS")
            print(f"   Attractions: {row[1]} | Hotels: {row[2]} | Restaurants: {row[3]}")
            print(f"   Activities: {row[4]} | Neighborhoods: {row[5]} | Practical Info: {row[6]}")
            print(f"   Last Updated: {row[7]}")
        
    except Exception as e:
        print(f"Error retrieving summary: {e}")
    
    db_manager.close_db()

def list_analyzed_destinations(db_path: str = "real_destination_intelligence.db"):
    """List all destinations that have been analyzed"""
    
    db_manager = DatabaseManager(db_path=db_path)
    
    try:
        if not db_manager.conn:
            print("Error: Database connection is None")
            return
            
        cursor = db_manager.conn.cursor()
        cursor.execute("""
            SELECT destination_name, analysis_timestamp, 
                   total_attractions + total_hotels + total_restaurants + 
                   total_activities + total_neighborhoods + total_practical_info as total_insights
            FROM enhanced_destination_analysis 
            ORDER BY analysis_timestamp DESC
        """)
        
        rows = cursor.fetchall()
        
        if rows:
            print("\n🗺️  Analyzed Destinations")
            print("=" * 50)
            for i, row in enumerate(rows, 1):
                print(f"{i}. {row[0]} - {row[2]} insights ({row[1]})")
        else:
            print("\n⚠️  No enhanced destination analyses found in database.")
            print("Run the enhanced analysis system to generate detailed insights!")
        
    except Exception as e:
        print(f"Error listing destinations: {e}")
    
    db_manager.close_db()

def export_insights_to_json(destination_name: str, output_file: str = None, db_path: str = "real_destination_intelligence.db"):
    """Export enhanced insights to JSON file"""
    
    if not output_file:
        safe_name = destination_name.replace(" ", "_").replace(",", "").lower()
        output_file = f"enhanced_insights_{safe_name}.json"
    
    db_manager = DatabaseManager(db_path=db_path)
    enhanced_storage = EnhancedDatabaseStorageTool(db_manager=db_manager)
    
    all_insights = enhanced_storage.get_stored_insights(destination_name=destination_name)
    
    # Group by category
    export_data = {
        "destination": destination_name,
        "export_timestamp": "",
        "insights_by_category": {},
        "total_insights": len(all_insights)
    }
    
    from datetime import datetime
    export_data["export_timestamp"] = datetime.now().isoformat()
    
    for insight in all_insights:
        category = insight["category"]
        if category not in export_data["insights_by_category"]:
            export_data["insights_by_category"][category] = []
        
        export_data["insights_by_category"][category].append(insight)
    
    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(export_data, f, indent=2, ensure_ascii=False)
        
        print(f"✅ Enhanced insights exported to: {output_file}")
        print(f"📊 Total insights: {len(all_insights)} across {len(export_data['insights_by_category'])} categories")
        
    except Exception as e:
        print(f"❌ Error exporting insights: {e}")
    
    db_manager.close_db()

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("🔍 Enhanced Destination Insights Query Tool")
        print("\nUsage:")
        print("  python query_enhanced_insights.py list")
        print("  python query_enhanced_insights.py show \"Destination Name\"")
        print("  python query_enhanced_insights.py export \"Destination Name\" [output_file.json]")
        print("\nExamples:")
        print("  python query_enhanced_insights.py list")
        print("  python query_enhanced_insights.py show \"Bend, Oregon\"")
        print("  python query_enhanced_insights.py export \"Seattle, Washington\"")
        sys.exit(1)
    
    command = sys.argv[1].lower()
    
    if command == "list":
        list_analyzed_destinations()
    
    elif command == "show" and len(sys.argv) >= 3:
        destination = sys.argv[2]
        display_insights_summary(destination)
    
    elif command == "export" and len(sys.argv) >= 3:
        destination = sys.argv[2]
        output_file = sys.argv[3] if len(sys.argv) >= 4 else None
        export_insights_to_json(destination, output_file)
    
    else:
        print("❌ Invalid command or missing arguments")
        print("Use: list, show \"destination\", or export \"destination\" [file]") 