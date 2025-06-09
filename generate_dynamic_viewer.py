#!/usr/bin/env python3
"""
Generate Dynamic Database Viewer HTML
Creates the dynamic_database_viewer.html file with interactive features
"""

import json
import sqlite3
from pathlib import Path
import os

def generate_dynamic_database_viewer():
    """Generate the dynamic database viewer HTML file"""
    
    # Check for database file
    db_path = "enhanced_destination_intelligence.db"
    if not os.path.exists(db_path):
        print(f"❌ Database file {db_path} not found!")
        return False
    
    # Connect to database and get all theme data
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # Get all themes with full details
        cursor.execute("""
            SELECT theme_id, name, macro_category, micro_category, description,
                   fit_score, confidence_level, confidence_breakdown,
                   tags, sentiment_analysis, temporal_analysis,
                   traveler_relevance_factor, adjusted_overall_confidence
            FROM themes 
            ORDER BY fit_score DESC
        """)
        themes = cursor.fetchall()
        
        # Get total count
        theme_count = len(themes)
        
        conn.close()
        
    except Exception as e:
        print(f"❌ Error reading database: {e}")
        return False
    
    # Build themes data for JavaScript
    themes_js_data = []
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
            "metadata": {
                "theme_id": theme_id,
                "description": description,
                "sentiment_analysis": sentiment_analysis,
                "temporal_analysis": temporal_analysis
            }
        }
        themes_js_data.append(theme_data)
    
    # Create the HTML content
    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Comprehensive Database Theme Report</title>
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
            background: linear-gradient(135deg, #4285f4 0%, #34a853 100%);
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
            border-left: 4px solid #4285f4;
        }}
        
        .theme-card:hover {{
            transform: translateY(-2px);
            box-shadow: 0 4px 16px rgba(0,0,0,0.15);
        }}
        
        .theme-header {{
            font-size: 1.4rem;
            font-weight: 600;
            color: #4285f4;
            margin-bottom: 1rem;
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
            color: #4285f4;
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
        }}
    </style>
</head>
<body>
    <div class="header">
        <h1>Comprehensive Database Theme Report</h1>
        <p>Interactive analysis of {theme_count:,} destination intelligence themes</p>
    </div>
    
    <div class="controls">
        <div class="control-group">
            <label for="filterInput">Filter by Name/Desc:</label>
            <input type="text" id="filterInput" placeholder="Enter keyword...">
        </div>
        <div class="control-group">
            <label for="sortSelect">Sort by:</label>
            <select id="sortSelect">
                <option value="fit_score_desc">Fit Score (High-Low)</option>
                <option value="fit_score_asc">Fit Score (Low-High)</option>
                <option value="confidence_desc">Confidence (High-Low)</option>
                <option value="confidence_asc">Confidence (Low-High)</option>
                <option value="name_asc">Name (A-Z)</option>
                <option value="name_desc">Name (Z-A)</option>
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
        // Theme data from database
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
                
                html += `
                <div class="theme-card">
                    <div class="theme-header">${{globalIndex}}. ${{name}}</div>
                    
                    <div class="theme-meta">
                        <div class="meta-item">
                            <span class="meta-label">Fit Score:</span>
                            <span class="meta-value">${{fitScore}}</span>
                        </div>
                        <div class="meta-item">
                            <span class="meta-label">Overall Confidence:</span>
                            <span class="meta-value ${{confidenceClass}}">${{adjustedConfidence}} (Level: ${{confidenceLevelDisplay}})</span>
                        </div>
                    </div>
                    
                    <div class="description">
                        <strong>Description:</strong> ${{escapeHtml(theme.description || 'No description available')}}
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
                            <p><strong>Created:</strong> Unknown</p>
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
            const sortBy = document.getElementById('sortSelect').value;
            itemsPerPage = parseInt(document.getElementById('itemsPerPage').value);
            
            // Filter themes
            filteredThemes = allThemes.filter(theme => {{
                const nameMatch = theme.name && theme.name.toLowerCase().includes(filterText);
                const descMatch = theme.description && theme.description.toLowerCase().includes(filterText);
                return nameMatch || descMatch;
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
                    default:
                        return 0;
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
    with open("dynamic_database_viewer.html", "w", encoding="utf-8") as f:
        f.write(html_content)
    
    print(f"✅ Dynamic database viewer created: dynamic_database_viewer.html")
    print(f"📊 Generated with {theme_count:,} themes")
    print(f"🌐 Open http://localhost:8000/dynamic_database_viewer.html to view")
    
    return True

if __name__ == "__main__":
    print("🔧 Generating dynamic database viewer...")
    success = generate_dynamic_database_viewer()
    
    if success:
        print("\n🎉 Success! Dynamic viewer recreated with full functionality!")
        print("   - Interactive filtering and sorting")
        print("   - Card-based layout matching your original")
        print("   - Expandable confidence breakdowns")
        print("   - Responsive design")
        print("\n📝 To view:")
        print("   1. python -m http.server 8000")
        print("   2. Open: http://localhost:8000/dynamic_database_viewer.html")
    else:
        print("\n❌ Could not generate viewer. Check that the database exists and contains data.") 