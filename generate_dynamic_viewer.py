#!/usr/bin/env python3
"""
Generate Dynamic Database Viewer HTML with Cultural Intelligence
Creates enhanced dynamic viewers with cultural intelligence categorization
"""

import json
import sqlite3
from pathlib import Path
import os
import argparse
import yaml

# CULTURAL INTELLIGENCE: Define category processing rules
CATEGORY_PROCESSING_RULES = {
    "cultural": {
        "categories": [
            "Cultural Identity & Atmosphere", "Authentic Experiences", "Distinctive Features",
            "Local Character & Vibe", "Artistic & Creative Scene"
        ],
        "color": "#9C27B0",  # Purple for cultural
        "border_color": "#7B1FA2",
        "icon": "🎭",
        "description": "Cultural themes focus on authenticity, local character, and distinctive experiences"
    },
    "practical": {
        "categories": [
            "Safety & Security", "Transportation & Access", "Budget & Costs", 
            "Health & Medical", "Logistics & Planning", "Visa & Documentation"
        ],
        "color": "#2196F3",  # Blue for practical
        "border_color": "#1976D2",
        "icon": "📋",
        "description": "Practical themes provide essential travel information and logistics"
    },
    "hybrid": {
        "categories": [
            "Food & Dining", "Entertainment & Nightlife", "Nature & Outdoor",
            "Shopping & Local Craft", "Family & Education", "Health & Wellness"
        ],
        "color": "#4CAF50",  # Green for hybrid
        "border_color": "#388E3C",
        "icon": "⚖️",
        "description": "Hybrid themes balance practical information with cultural authenticity"
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
    config_path = "config/config.yaml"
    if not os.path.exists(config_path):
        print(f"❌ config.yaml not found!")
        return None
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def generate_dynamic_database_viewer(destination_name=None, check_exists=False):
    """Generate the dynamic database viewer HTML file with cultural intelligence features"""
    
    output_filename = "dynamic_database_viewer.html"
    if destination_name:
        # Create a URL-safe version of the destination name for the filename
        safe_dest_name = destination_name.replace(', ', '_').replace(',', '_').replace(' ', '_').lower()
        output_filename = f"dynamic_viewer_{safe_dest_name}.html"

    if check_exists and os.path.exists(output_filename):
        print(f"✅ Viewer for {destination_name} already exists. Skipping.")
        return True, None
    
    # Check for database file
    db_path = "enhanced_destination_intelligence.db"
    if not os.path.exists(db_path):
        print(f"❌ Database file {db_path} not found!")
        return False, None
    
    # Connect to database and get all theme data
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # Get all themes with full details
        query = """
            SELECT theme_id, name, macro_category, micro_category, description,
                   fit_score, confidence_level, confidence_breakdown,
                   tags, sentiment_analysis, temporal_analysis,
                   traveler_relevance_factor, adjusted_overall_confidence
            FROM themes 
        """
        params = []
        if destination_name:
            query += " WHERE destination_id = ?"
            # The database ID has a specific format we need to match
            dest_id = f"dest_{destination_name.replace(', ', '_').replace(' ', '_').lower()}"
            params.append(dest_id)

        query += " ORDER BY fit_score DESC"
        
        cursor.execute(query, params)
        themes = cursor.fetchall()
        
        # Get total count
        theme_count = len(themes)
        
        conn.close()
        
    except Exception as e:
        print(f"❌ Error reading database: {e}")
        return False, None
    
    # Build themes data for JavaScript with cultural intelligence
    themes_js_data = []
    category_stats = {"cultural": 0, "practical": 0, "hybrid": 0, "unknown": 0}
    
    for i, theme in enumerate(themes, 1):
        theme_id, name, macro_cat, micro_cat, description, fit_score, confidence_level, confidence_breakdown, tags, sentiment_analysis, temporal_analysis, traveler_relevance_factor, adjusted_overall_confidence = theme
        
        # Parse JSON fields safely
        try:
            confidence_breakdown_data = json.loads(confidence_breakdown) if confidence_breakdown else {}
        except:
            confidence_breakdown_data = {}
            
        try:
            tags_data = json.loads(tags) if tags else []
        except:
            tags_data = []
            
        # Get overall confidence from breakdown
        overall_confidence = confidence_breakdown_data.get('overall_confidence', adjusted_overall_confidence or 0.0)
        if isinstance(overall_confidence, str):
            try:
                overall_confidence = float(overall_confidence)
            except:
                overall_confidence = 0.0
        
        # CULTURAL INTELLIGENCE: Determine processing type and styling
        proc_type = get_processing_type(macro_cat)
        category_info = CATEGORY_PROCESSING_RULES.get(proc_type, {
            "color": "#666666", "border_color": "#444444", "icon": "📌", "description": "Unknown category"
        })
        category_stats[proc_type] += 1
                
        theme_data = {
            "id": theme_id,
            "name": name or f"Theme {i}",
            "macro_category": macro_cat or "General",
            "micro_category": micro_cat or "",
            "description": description or f"Theme about {name or 'Unknown'}",
            "fit_score": fit_score or 0.0,
            "confidence_level": confidence_level or "unknown",
            "overall_confidence": overall_confidence,
            "confidence_breakdown": confidence_breakdown_data,
            "created_date": None,  # Column doesn't exist
            "tags": tags_data,
            "traveler_relevance_factor": traveler_relevance_factor,
            # CULTURAL INTELLIGENCE: Add processing type and styling
            "processing_type": proc_type,
            "category_color": category_info["color"],
            "category_border_color": category_info["border_color"],
            "category_icon": category_info["icon"],
            "category_description": category_info["description"],
            "metadata": {
                "theme_id": theme_id,
                "description": description,
                "sentiment_analysis": sentiment_analysis,
                "temporal_analysis": temporal_analysis
            }
        }
        themes_js_data.append(theme_data)
    
    # Create the HTML content with cultural intelligence features
    title = "Cultural Intelligence Theme Report"
    if destination_name:
        title = f"Cultural Intelligence Report - {destination_name}"

    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{title}</title>
    <style>
        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}
        
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, Cantarell, sans-serif;
            line-height: 1.6;
            color: #333;
            background-color: #f8f9fa;
        }}
        
        .header {{
            background: linear-gradient(135deg, #9C27B0 0%, #2196F3 50%, #4CAF50 100%);
            color: white;
            padding: 2rem;
            text-align: center;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }}
        
        .header h1 {{
            font-size: 2.5rem;
            margin-bottom: 0.5rem;
            font-weight: 600;
        }}
        
        .category-legend {{
            background: white;
            margin: 1rem 2rem;
            padding: 1rem;
            border-radius: 12px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.1);
            display: flex;
            gap: 2rem;
            justify-content: center;
            flex-wrap: wrap;
        }}
        
        .legend-item {{
            display: flex;
            align-items: center;
            gap: 0.5rem;
            padding: 0.5rem 1rem;
            border-radius: 8px;
            font-weight: 500;
        }}
        
        .legend-cultural {{
            background: rgba(156, 39, 176, 0.1);
            border-left: 4px solid #9C27B0;
        }}
        
        .legend-practical {{
            background: rgba(33, 150, 243, 0.1);
            border-left: 4px solid #2196F3;
        }}
        
        .legend-hybrid {{
            background: rgba(76, 175, 80, 0.1);
            border-left: 4px solid #4CAF50;
        }}
        
        .controls {{
            background: white;
            padding: 1.5rem;
            margin: 2rem;
            border-radius: 12px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.1);
            display: flex;
            gap: 1rem;
            align-items: center;
            flex-wrap: wrap;
        }}
        
        .control-group {{
            display: flex;
            align-items: center;
            gap: 0.5rem;
        }}
        
        .control-group label {{
            font-weight: 500;
            color: #555;
        }}
        
        .control-group input, .control-group select {{
            padding: 0.5rem;
            border: 1px solid #ddd;
            border-radius: 6px;
            font-size: 14px;
        }}
        
        .apply-btn {{
            background: #4285f4;
            color: white;
            border: none;
            padding: 0.6rem 1.2rem;
            border-radius: 6px;
            cursor: pointer;
            font-weight: 500;
            transition: background 0.2s;
        }}
        
        .apply-btn:hover {{
            background: #3367d6;
        }}
        
        .themes-container {{
            margin: 0 2rem 2rem 2rem;
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(500px, 1fr));
            gap: 1.5rem;
        }}
        
        .theme-card {{
            background: white;
            border-radius: 12px;
            padding: 1.5rem;
            box-shadow: 0 2px 8px rgba(0,0,0,0.1);
            transition: transform 0.2s, box-shadow 0.2s;
            border-left: 4px solid var(--theme-color);
            position: relative;
        }}
        
        .theme-card:hover {{
            transform: translateY(-2px);
            box-shadow: 0 4px 16px rgba(0,0,0,0.15);
        }}
        
        .theme-header {{
            font-size: 1.4rem;
            font-weight: 600;
            color: var(--theme-color);
            margin-bottom: 1rem;
            display: flex;
            align-items: center;
            gap: 0.5rem;
        }}
        
        .category-badge {{
            position: absolute;
            top: 1rem;
            right: 1rem;
            padding: 0.25rem 0.5rem;
            border-radius: 6px;
            font-size: 0.75rem;
            font-weight: 600;
            color: white;
            background: var(--theme-color);
        }}
        
        .theme-meta {{
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 0.5rem;
            margin-bottom: 1rem;
        }}
        
        .meta-item {{
            display: flex;
            justify-content: space-between;
            padding: 0.3rem 0;
        }}
        
        .meta-label {{
            font-weight: 500;
            color: #666;
        }}
        
        .meta-value {{
            font-weight: 600;
        }}
        
        .confidence-high {{ color: #34a853; }}
        .confidence-moderate {{ color: #fbbc04; }}
        .confidence-low {{ color: #ea4335; }}
        .confidence-unknown {{ color: #9aa0a6; }}
        
        .description {{
            color: #666;
            font-style: italic;
            margin-bottom: 1rem;
            padding: 0.8rem;
            background: #f8f9fa;
            border-radius: 6px;
        }}
        
        .expandable {{
            margin-top: 1rem;
        }}
        
        .expand-btn {{
            background: #f1f3f4;
            border: none;
            padding: 0.5rem 1rem;
            border-radius: 6px;
            cursor: pointer;
            font-weight: 500;
            color: var(--theme-color);
            width: 100%;
            text-align: left;
            display: flex;
            align-items: center;
            justify-content: space-between;
            margin-bottom: 0.5rem;
        }}
        
        .expand-btn:hover {{
            background: #e8f0fe;
        }}
        
        .expand-content {{
            display: none;
            background: #f8f9fa;
            padding: 1rem;
            border-radius: 6px;
            border: 1px solid #e0e0e0;
        }}
        
        .expand-content.show {{
            display: block;
        }}
        
        .no-results {{
            text-align: center;
            padding: 4rem;
            color: #666;
            font-size: 1.2rem;
        }}
        
        .pagination {{
            text-align: center;
            margin: 2rem;
        }}
        
        .pagination button {{
            background: white;
            border: 1px solid #ddd;
            padding: 0.5rem 1rem;
            margin: 0 0.25rem;
            border-radius: 6px;
            cursor: pointer;
        }}
        
        .pagination button.active {{
            background: #4285f4;
            color: white;
            border-color: #4285f4;
        }}
        
        .pagination button:hover:not(.active) {{
            background: #f1f3f4;
        }}
        
        @media (max-width: 768px) {{
            .themes-container {{
                grid-template-columns: 1fr;
                margin: 0 1rem 2rem 1rem;
            }}
            
            .controls {{
                margin: 1rem;
                flex-direction: column;
                align-items: stretch;
            }}
            
            .control-group {{
                justify-content: space-between;
            }}
            
            .category-legend {{
                flex-direction: column;
                gap: 1rem;
            }}
        }}
    </style>
</head>
<body>
    <div class="header">
        <h1>{title}</h1>
        <p>🎭 Cultural Intelligence analysis of {theme_count:,} destination themes</p>
        <p>🎯 Cultural: {category_stats['cultural']} | 📋 Practical: {category_stats['practical']} | ⚖️ Hybrid: {category_stats['hybrid']}</p>
    </div>
    
    <div class="category-legend">
        <div class="legend-item legend-cultural">
            <span>🎭</span>
            <span><strong>Cultural:</strong> Authenticity & Local Character</span>
        </div>
        <div class="legend-item legend-practical">
            <span>📋</span>
            <span><strong>Practical:</strong> Essential Travel Information</span>
        </div>
        <div class="legend-item legend-hybrid">
            <span>⚖️</span>
            <span><strong>Hybrid:</strong> Balanced Practical & Cultural</span>
        </div>
    </div>
    
    <div class="controls">
        <div class="control-group">
            <label for="filterInput">Filter by Name/Desc:</label>
            <input type="text" id="filterInput" placeholder="Enter keyword...">
        </div>
        <div class="control-group">
            <label for="categoryFilter">Category:</label>
            <select id="categoryFilter">
                <option value="all">All Categories</option>
                <option value="cultural">🎭 Cultural</option>
                <option value="practical">📋 Practical</option>
                <option value="hybrid">⚖️ Hybrid</option>
            </select>
        </div>
        <div class="control-group">
            <label for="sortSelect">Sort by:</label>
            <select id="sortSelect">
                <option value="confidence_desc" selected>Tourist Relevance (High-Low)</option>
                <option value="fit_score_desc">Evidence Quality (High-Low)</option>
                <option value="fit_score_asc">Evidence Quality (Low-High)</option>
                <option value="confidence_asc">Tourist Relevance (Low-High)</option>
                <option value="name_asc">Name (A-Z)</option>
                <option value="name_desc">Name (Z-A)</option>
                <option value="category_asc">Category (Cultural First)</option>
            </select>
        </div>
        <div class="control-group">
            <label for="itemsPerPage">Items per page:</label>
            <select id="itemsPerPage">
                <option value="50">50</option>
                <option value="100" selected>100</option>
                <option value="200">200</option>
                <option value="500">500</option>
            </select>
        </div>
        <button class="apply-btn" onclick="applyFilters()">Apply</button>
    </div>
    
    <div id="themesContainer" class="themes-container">
        <!-- Themes will be populated by JavaScript -->
    </div>
    
    <div id="pagination" class="pagination">
        <!-- Pagination will be populated by JavaScript -->
    </div>
    
    <script>
        // Theme data from database with cultural intelligence
        const allThemes = {json.dumps(themes_js_data, indent=2)};
        
        let filteredThemes = [...allThemes];
        let currentPage = 1;
        let itemsPerPage = 100;
        
        function formatConfidenceLevelCssClass(level) {{
            switch(level) {{
                case 'high': return 'confidence-high';
                case 'moderate': return 'confidence-moderate';
                case 'low': return 'confidence-low';
                default: return 'confidence-unknown';
            }}
        }}
        
        function formatJsonForDisplay(obj) {{
            if (!obj) return 'No data available';
            return '<pre>' + JSON.stringify(obj, null, 2) + '</pre>';
        }}
        
        function escapeHtml(text) {{
            const div = document.createElement('div');
            div.textContent = text;
            return div.innerHTML;
        }}
        
        function renderThemes() {{
            const container = document.getElementById('themesContainer');
            const startIndex = (currentPage - 1) * itemsPerPage;
            const endIndex = startIndex + itemsPerPage;
            const themesToShow = filteredThemes.slice(startIndex, endIndex);
            
            if (themesToShow.length === 0) {{
                container.innerHTML = '<div class="no-results">No themes found matching your criteria.</div>';
                return;
            }}
            
            let html = '';
            themesToShow.forEach((theme, index) => {{
                const globalIndex = startIndex + index + 1;
                const name = escapeHtml(theme.name || 'Unnamed Theme');
                const fitScore = typeof theme.fit_score === 'number' ? theme.fit_score.toFixed(4) : 'N/A';
                const adjustedConfidence = typeof theme.overall_confidence === 'number' ? theme.overall_confidence.toFixed(4) : 'N/A';
                const confidenceLevelDisplay = escapeHtml(theme.confidence_level || 'N/A');
                const confidenceClass = formatConfidenceLevelCssClass(theme.confidence_level);
                const relevanceFactor = typeof theme.traveler_relevance_factor === 'number' ? theme.traveler_relevance_factor.toFixed(2) : 'N/A';
                
                // CULTURAL INTELLIGENCE: Theme styling based on processing type
                const themeColor = theme.category_color || '#4285f4';
                const categoryIcon = theme.category_icon || '📌';
                const processingType = theme.processing_type || 'unknown';
                const categoryDescription = theme.category_description || 'Unknown category';
                
                html += `
                <div class="theme-card" style="--theme-color: ${{themeColor}}">
                    <div class="category-badge">${{categoryIcon}} ${{processingType.toUpperCase()}}</div>
                    <div class="theme-header">
                        <span>${{categoryIcon}}</span>
                        <span>${{globalIndex}}. ${{name}}</span>
                    </div>
                    
                    <div class="theme-meta">
                        <div class="meta-item">
                            <span class="meta-label">Fit Score:</span>
                            <span class="meta-value">${{fitScore}}</span>
                        </div>
                        <div class="meta-item">
                            <span class="meta-label">Overall Confidence:</span>
                            <span class="meta-value ${{confidenceClass}}">${{adjustedConfidence}} (Level: ${{confidenceLevelDisplay}})</span>
                        </div>
                        <div class="meta-item">
                            <span class="meta-label">Category:</span>
                            <span class="meta-value">${{escapeHtml(theme.macro_category)}}</span>
                        </div>
                        <div class="meta-item">
                            <span class="meta-label">Processing Type:</span>
                            <span class="meta-value" style="color: ${{themeColor}}">${{processingType.toUpperCase()}}</span>
                        </div>
                    </div>
                    
                    <div class="description">
                        <strong>Description:</strong> ${{escapeHtml(theme.description || 'No description available')}}
                        <br><small style="color: #888; margin-top: 0.5rem; display: block;"><em>${{categoryDescription}}</em></small>
                    </div>
                    
                    <div class="expandable">
                        <button class="expand-btn" onclick="toggleExpand(this)">
                            <span>▶ Full Confidence Breakdown</span>
                        </button>
                        <div class="expand-content">
                            ${{formatJsonForDisplay(theme.confidence_breakdown)}}
                        </div>
                    </div>
                    
                    <div class="expandable">
                        <button class="expand-btn" onclick="toggleExpand(this)">
                            <span>▶ Additional Metadata</span>
                        </button>
                        <div class="expand-content">
                            <p><strong>Theme ID:</strong> ${{theme.id}}</p>
                            <p><strong>Category:</strong> ${{escapeHtml(theme.macro_category)}}</p>
                            <p><strong>Subcategory:</strong> ${{escapeHtml(theme.micro_category)}}</p>
                            <p><strong>Processing Type:</strong> <span style="color: ${{themeColor}}">${{processingType.toUpperCase()}}</span></p>
                            <p><strong>Traveler Relevance:</strong> ${{relevanceFactor}}</p>
                            ${{theme.tags && theme.tags.length > 0 ? '<p><strong>Tags:</strong> ' + theme.tags.map(tag => escapeHtml(tag)).join(', ') + '</p>' : ''}}
                        </div>
                    </div>
                </div>
                `;
            }});
            
            container.innerHTML = html;
            renderPagination();
        }}
        
        function renderPagination() {{
            const totalPages = Math.ceil(filteredThemes.length / itemsPerPage);
            const pagination = document.getElementById('pagination');
            
            if (totalPages <= 1) {{
                pagination.innerHTML = '';
                return;
            }}
            
            let html = '';
            
            // Previous button
            if (currentPage > 1) {{
                html += `<button onclick="goToPage(${{currentPage - 1}})">Previous</button>`;
            }}
            
            // Page numbers
            const startPage = Math.max(1, currentPage - 2);
            const endPage = Math.min(totalPages, currentPage + 2);
            
            for (let i = startPage; i <= endPage; i++) {{
                const activeClass = i === currentPage ? 'active' : '';
                html += `<button class="${{activeClass}}" onclick="goToPage(${{i}})">${{i}}</button>`;
            }}
            
            // Next button
            if (currentPage < totalPages) {{
                html += `<button onclick="goToPage(${{currentPage + 1}})">Next</button>`;
            }}
            
            pagination.innerHTML = html;
        }}
        
        function goToPage(page) {{
            currentPage = page;
            renderThemes();
        }}
        
        function toggleExpand(button) {{
            const content = button.nextElementSibling;
            const icon = button.querySelector('span');
            
            if (content.classList.contains('show')) {{
                content.classList.remove('show');
                icon.textContent = icon.textContent.replace('▼', '▶');
            }} else {{
                content.classList.add('show');
                icon.textContent = icon.textContent.replace('▶', '▼');
            }}
        }}
        
        function applyFilters() {{
            const filterText = document.getElementById('filterInput').value.toLowerCase();
            const categoryFilter = document.getElementById('categoryFilter').value;
            const sortBy = document.getElementById('sortSelect').value;
            itemsPerPage = parseInt(document.getElementById('itemsPerPage').value);
            
            // Filter themes
            filteredThemes = allThemes.filter(theme => {{
                const nameMatch = theme.name && theme.name.toLowerCase().includes(filterText);
                const descMatch = theme.description && theme.description.toLowerCase().includes(filterText);
                const textMatch = nameMatch || descMatch;
                
                // CULTURAL INTELLIGENCE: Category filtering
                const categoryMatch = categoryFilter === 'all' || theme.processing_type === categoryFilter;
                
                return textMatch && categoryMatch;
            }});
            
            // Sort themes
            filteredThemes.sort((a, b) => {{
                switch(sortBy) {{
                    case 'fit_score_desc':
                        return (b.fit_score || 0) - (a.fit_score || 0);
                    case 'fit_score_asc':
                        return (a.fit_score || 0) - (b.fit_score || 0);
                    case 'confidence_desc':
                        return (b.overall_confidence || 0) - (a.overall_confidence || 0);
                    case 'confidence_asc':
                        return (a.overall_confidence || 0) - (b.overall_confidence || 0);
                    case 'name_asc':
                        return (a.name || '').localeCompare(b.name || '');
                    case 'name_desc':
                        return (b.name || '').localeCompare(a.name || '');
                    case 'category_asc':
                        // Sort by processing type: cultural, practical, hybrid
                        const order = {{'cultural': 0, 'practical': 1, 'hybrid': 2, 'unknown': 3}};
                        return (order[a.processing_type] || 3) - (order[b.processing_type] || 3);
                    default:
                        // DEFAULT: Sort by tourist relevance (overall_confidence which includes traveler_relevance_factor)
                        return (b.overall_confidence || 0) - (a.overall_confidence || 0);
                }}
            }});
            
            currentPage = 1;
            renderThemes();
        }}
        
        // Initialize
        document.addEventListener('DOMContentLoaded', function() {{
            renderThemes();
            
            // Add enter key support for filter
            document.getElementById('filterInput').addEventListener('keypress', function(e) {{
                if (e.key === 'Enter') {{
                    applyFilters();
                }}
            }});
        }});
    </script>
</body>
</html>"""
    
    # Write the HTML file
    with open(output_filename, "w", encoding="utf-8") as f:
        f.write(html_content)
    
    print(f"✅ Cultural Intelligence dynamic viewer created: {output_filename}")
    print(f"📊 Generated with {theme_count:,} themes")
    print(f"🎭 Cultural: {category_stats['cultural']} | 📋 Practical: {category_stats['practical']} | ⚖️ Hybrid: {category_stats['hybrid']}")
    print(f"🌐 Open http://localhost:8000/{output_filename} to view")
    
    return True, output_filename

def load_and_categorize_themes(db_path, destination_name):
    """Load themes from database and categorize them with cultural intelligence processing"""
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
            theme_id, name, macro_cat, micro_cat, description, fit_score, confidence_level, overall_confidence, confidence_breakdown, tags = row
            
            # Determine processing type and get category info
            proc_type = get_processing_type(macro_cat)
            category_info = CATEGORY_PROCESSING_RULES.get(proc_type, {
                "color": "#666666", "border_color": "#444444", "icon": "📌", "description": "Unknown category"
            })
            
            theme_data = {
                "theme_id": theme_id,
                "name": name,
                "macro_category": macro_cat,
                "micro_category": micro_cat,
                "description": description,
                "fit_score": fit_score or 0.0,
                "confidence_level": confidence_level,
                "overall_confidence": overall_confidence or 0.0,
                "confidence_breakdown": confidence_breakdown,
                "tags": tags,
                "processing_type": proc_type,
                "category_color": category_info["color"],
                "category_border_color": category_info["border_color"],
                "category_icon": category_info["icon"],
                "category_description": category_info["description"]
            }
            
            themes.append(theme_data)
        
        conn.close()
        return themes
        
    except sqlite3.Error as e:
        print(f"Database error loading and categorizing themes: {e}")
        return []
    except Exception as e:
        print(f"Error loading and categorizing themes: {e}")
        return []

def apply_category_styling(theme_data, processing_rules=None):
    """Apply category-based styling to a single theme or themes data for dynamic viewer generation"""
    if processing_rules is None:
        processing_rules = {
            "cultural": {"color": "#9C27B0", "icon": "🎭"},
            "practical": {"color": "#2196F3", "icon": "🛠️"},
            "hybrid": {"color": "#FF9800", "icon": "🌟"},
            "unknown": {"color": "#757575", "icon": "❓"}
        }
    
    # Handle single theme vs dictionary of themes
    if isinstance(theme_data, dict) and ("name" in theme_data or "processing_type" in theme_data):
        # Single theme object
        processing_type = theme_data.get("processing_type") or get_theme_processing_type(theme_data.get("macro_category", ""))
        styling = processing_rules.get(processing_type, processing_rules["unknown"])
        
        styled_theme = {
            **theme_data,
            "processing_type": processing_type,
            "category_color": styling["color"],
            "category_icon": styling["icon"],
            "category_class": f"theme-{processing_type}"
        }
        
        return styled_theme
    else:
        # Dictionary of themes (original behavior)
        styled_themes = []
        
        for theme_id, theme_data in theme_data.items():
            # Determine processing type for the theme
            processing_type = get_theme_processing_type(theme_data.get("macro_category", ""))
            
            # Apply styling based on processing type
            styling = processing_rules.get(processing_type, processing_rules["unknown"])
            
            styled_theme = {
                **theme_data,
                "processing_type": processing_type,
                "style": {
                    "color": styling["color"],
                    "icon": styling["icon"],
                    "category_class": f"theme-{processing_type}"
                }
            }
            
            styled_themes.append((theme_id, styled_theme))
        
        return styled_themes

def get_theme_processing_type(macro_category):
    """Determine processing type for a theme category"""
    if not macro_category:
        return "unknown"
        
    cultural_categories = [
        "Cultural Identity & Atmosphere", "Authentic Experiences", "Distinctive Features",
        "Local Character & Vibe", "Artistic & Creative Scene", "Cultural & Arts", "Heritage & History"
    ]
    
    practical_categories = [
        "Safety & Security", "Transportation & Access", "Health & Medical", "Budget & Costs",
        "Logistics & Planning", "Communication & Language", "Legal & Documentation"
    ]
    
    hybrid_categories = [
        "Food & Dining", "Entertainment & Nightlife", "Nature & Outdoor", "Shopping & Commerce",
        "Accommodations", "Climate & Weather", "Adventure & Sports"
    ]
    
    if macro_category in cultural_categories:
        return "cultural"
    elif macro_category in practical_categories:
        return "practical"
    elif macro_category in hybrid_categories:
        return "hybrid"
    else:
        return "unknown"

def generate_theme_cards_html(themes_data):
    """Generate HTML cards for themes with cultural intelligence styling"""
    if not themes_data:
        return "<p>No themes available for display.</p>"
    
    html_parts = []
    html_parts.append('<div class="themes-container">')
    
    for theme in themes_data:
        # Get theme data
        name = theme.get("name", "Unknown Theme")
        description = theme.get("description", "No description available")
        confidence = theme.get("confidence", 0.0)
        processing_type = theme.get("processing_type", "unknown")
        category_color = theme.get("category_color", "#757575")
        category_icon = theme.get("category_icon", "❓")
        
        # Generate theme card HTML
        html_parts.append(f'''
        <div class="theme-card {processing_type}-theme" style="border-left: 4px solid {category_color};">
            <div class="theme-header">
                <span class="category-badge" style="background-color: {category_color};">
                    {category_icon} {processing_type.title()}
                </span>
                <span class="processing-type">{processing_type}</span>
            </div>
            <h3 class="theme-title">{name}</h3>
            <p class="theme-description">{description}</p>
            <div class="theme-metrics">
                <span class="confidence">Confidence: {confidence:.2f}</span>
            </div>
        </div>
        ''')
    
    html_parts.append('</div>')
    
    return '\n'.join(html_parts)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate a dynamic HTML viewer for destination themes.")
    parser.add_argument("-d", "--destination", type=str, help="The name of the destination to generate the viewer for (e.g., 'Chicago, United States').")
    parser.add_argument("--all", action="store_true", help="Generate viewers for all destinations in config.yaml.")
    args = parser.parse_args()

    config = load_config()
    if not config:
        exit()

    destinations_to_process = []
    if args.destination:
        destinations_to_process.append(args.destination)
    elif args.all:
        destinations_to_process = config.get("destinations", [])
    else:
        # Default to the first destination in the config
        first_dest = config.get("destinations", [None])[0]
        if first_dest:
            destinations_to_process.append(first_dest)

    if not destinations_to_process:
        print("❌ No destinations specified or found in config.yaml.")
        exit()

    print(f"🔧 Generating dynamic database viewer(s) for: {', '.join(destinations_to_process)}")
    
    generated_files = []
    for dest in destinations_to_process:
        success, filename = generate_dynamic_database_viewer(dest, check_exists=True)
        if success and filename:
            generated_files.append(filename)

    if generated_files:
        print("\n🎉 Success! Dynamic viewer(s) created with full functionality!")
        print("   - Interactive filtering and sorting")
        print("   - Card-based layout matching your original")
        print("   - Expandable confidence breakdowns")
        print("   - Responsive design")
        print("\n📝 To view:")
        print("   1. python -m http.server 8000")
        for i, f in enumerate(generated_files):
            print(f"   {i+2}. Open: http://localhost:8000/{f}")
    else:
        print("\n🤷 No new viewers were generated. They may already exist or there was an error.") 