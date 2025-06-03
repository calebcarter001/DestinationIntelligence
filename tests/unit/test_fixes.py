#!/usr/bin/env python3
"""
Test script to verify that the enhanced theme analysis fixes work correctly.
"""

import sys
import os
import sqlite3
import json
import logging
from datetime import datetime
from typing import Dict, List, Any, Optional

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Add src to Python path for imports
src_path = os.path.join(os.path.dirname(__file__), 'src')
sys.path.insert(0, src_path)

# Now import with absolute path
try:
    from src.tools.enhanced_theme_analysis_tool import EnhancedThemeAnalysisTool
except ImportError:
    # Alternative import method
    sys.path.insert(0, os.path.dirname(__file__))
    from src.tools.enhanced_theme_analysis_tool import EnhancedThemeAnalysisTool

class ExportFixer:
    def __init__(self, db_path: str = "enhanced_destination_intelligence.db"):
        self.db_path = db_path
        self.conn = None
        self.destination_id = None
        self.destination_name = None
        
    def connect(self):
        """Connect to the database"""
        try:
            self.conn = sqlite3.connect(self.db_path)
            logger.info(f"Connected to database: {self.db_path}")
        except sqlite3.Error as e:
            logger.error(f"Failed to connect to database: {e}")
            raise

    def get_destinations(self) -> List[Dict[str, Any]]:
        """Get list of destinations from database"""
        cursor = self.conn.cursor()
        cursor.execute("SELECT id, names FROM destinations")
        rows = cursor.fetchall()
        
        destinations = []
        for row in rows:
            destinations.append({
                'id': row[0],
                'name': json.loads(row[1])[0] if row[1] else None
            })
        return destinations

    def get_evidence_for_destination(self, destination_id: str) -> List[Dict[str, Any]]:
        """Get all evidence for a destination"""
        cursor = self.conn.cursor()
        cursor.execute("""
            SELECT id, source_url, source_category, evidence_type, authority_weight,
                   text_snippet, timestamp, confidence, sentiment, cultural_context,
                   relationships, agent_id, published_date
            FROM evidence 
            WHERE destination_id = ?
        """, (destination_id,))
        
        evidence_list = []
        for row in cursor.fetchall():
            evidence = {
                'id': row[0],
                'source_url': row[1],
                'source_category': row[2],
                'evidence_type': row[3],
                'authority_weight': row[4],
                'text_snippet': row[5],
                'timestamp': row[6],
                'confidence': row[7],
                'sentiment': row[8],
                'cultural_context': json.loads(row[9]) if row[9] else {},
                'relationships': json.loads(row[10]) if row[10] else [],
                'agent_id': row[11],
                'published_date': row[12]
            }
            evidence_list.append(evidence)
        
        return evidence_list

    def get_themes_for_destination(self, destination_id: str) -> List[Dict[str, Any]]:
        """Get all themes for a destination"""
        cursor = self.conn.cursor()
        cursor.execute("""
            SELECT theme_id, macro_category, micro_category, name, description,
                   fit_score, confidence_breakdown, confidence_level, tags,
                   factors, cultural_summary, sentiment_analysis, temporal_analysis
            FROM themes 
            WHERE destination_id = ?
        """, (destination_id,))
        
        themes_list = []
        for row in cursor.fetchall():
            theme = {
                'theme_id': row[0],
                'macro_category': row[1],
                'micro_category': row[2],
                'name': row[3],
                'description': row[4],
                'fit_score': row[5],
                'confidence_breakdown': json.loads(row[6]) if row[6] else {},
                'confidence_level': row[7],
                'tags': json.loads(row[8]) if row[8] else [],
                'factors': json.loads(row[9]) if row[9] else {},
                'cultural_summary': json.loads(row[10]) if row[10] else {},
                'sentiment_analysis': json.loads(row[11]) if row[11] else {},
                'temporal_analysis': json.loads(row[12]) if row[12] else {}
            }
            themes_list.append(theme)
        
        return themes_list

    def create_evidence_export(self, destination_id: str, evidence_list: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Create evidence export structure"""
        export = {
            'export_metadata': {
                'destination_id': destination_id,
                'timestamp': datetime.utcnow().isoformat(),
                'version': '2.0'
            },
            'evidence_summary': {
                'total_count': len(evidence_list),
                'source_distribution': self._calculate_source_distribution(evidence_list)
            },
            'evidence_by_source': self._group_evidence_by_source(evidence_list),
            'all_evidence': evidence_list
        }
        return export

    def create_themes_export(self, destination_id: str, themes_list: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Create themes export structure"""
        export = {
            'export_metadata': {
                'destination_id': destination_id,
                'timestamp': datetime.utcnow().isoformat(),
                'version': '2.0'
            },
            'themes_by_category': self._group_themes_by_category(themes_list),
            'theme_statistics': self._calculate_theme_statistics(themes_list)
        }
        return export

    def _calculate_source_distribution(self, evidence_list: List[Dict[str, Any]]) -> Dict[str, int]:
        """Calculate distribution of evidence sources"""
        distribution = {}
        for evidence in evidence_list:
            source_type = evidence.get('source_category', 'unknown')
            distribution[source_type] = distribution.get(source_type, 0) + 1
        return distribution

    def _group_evidence_by_source(self, evidence_list: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
        """Group evidence by source category"""
        grouped = {}
        for evidence in evidence_list:
            source_type = evidence.get('source_category', 'unknown')
            if source_type not in grouped:
                grouped[source_type] = []
            grouped[source_type].append(evidence)
        return grouped

    def _group_themes_by_category(self, themes_list: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
        """Group themes by macro category"""
        grouped = {}
        for theme in themes_list:
            category = theme.get('macro_category', 'uncategorized')
            if category not in grouped:
                grouped[category] = []
            grouped[category].append(theme)
        return grouped

    def _calculate_theme_statistics(self, themes_list: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate theme statistics"""
        stats = {
            'total_themes': len(themes_list),
            'category_distribution': {},
            'confidence_levels': {
                'high': 0,
                'medium': 0,
                'low': 0
            }
        }
        
        for theme in themes_list:
            # Category distribution
            category = theme.get('macro_category', 'uncategorized')
            stats['category_distribution'][category] = stats['category_distribution'].get(category, 0) + 1
            
            # Confidence levels
            confidence = theme.get('confidence_level', 'low').lower()
            if confidence in stats['confidence_levels']:
                stats['confidence_levels'][confidence] += 1
        
        return stats

    def save_json_export(self, data: Dict[str, Any], filename: str):
        """Save data to JSON file"""
        os.makedirs('destination_insights', exist_ok=True)
        filepath = os.path.join('destination_insights', filename)
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
        logger.info(f"Saved export to: {filepath}")

    def close(self):
        """Close database connection"""
        if self.conn:
            self.conn.close()
            logger.info("Closed database connection")

def main():
    fixer = ExportFixer()
    try:
        fixer.connect()
        
        # Get all destinations
        destinations = fixer.get_destinations()
        
        for destination in destinations:
            destination_id = destination['id']
            
            # Get evidence and themes
            evidence_list = fixer.get_evidence_for_destination(destination_id)
            themes_list = fixer.get_themes_for_destination(destination_id)
            
            # Create exports
            evidence_export = fixer.create_evidence_export(destination_id, evidence_list)
            themes_export = fixer.create_themes_export(destination_id, themes_list)
            
            # Save exports
            fixer.save_json_export(
                evidence_export,
                f"{destination['name'].lower().replace(' ', '_')}_evidence_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            )
            fixer.save_json_export(
                themes_export,
                f"{destination['name'].lower().replace(' ', '_')}_themes_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            )
            
    finally:
        fixer.close()

if __name__ == "__main__":
    main() 