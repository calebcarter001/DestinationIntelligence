#!/usr/bin/env python3
"""
Comprehensive fix for all attribute access errors and chained .get() method calls
"""

import os
import re
from pathlib import Path

def fix_consolidated_json_export_manager():
    """Fix the main attribute access error in consolidated_json_export_manager.py"""
    file_path = "src/core/consolidated_json_export_manager.py"
    
    with open(file_path, 'r') as f:
        content = f.read()
    
    # Fix destination.id access
    content = content.replace(
        "destination_id_safe = destination.id.replace(' ', '_').replace(',', '_').lower()",
        """# Handle both objects and dictionaries for destination ID
            if hasattr(destination, 'id'):
                destination_id = destination.id
            elif isinstance(destination, dict):
                destination_id = destination.get('id', 'unknown_destination')
            else:
                destination_id = 'unknown_destination'
            
            destination_id_safe = destination_id.replace(' ', '_').replace(',', '_').lower()"""
    )
    
    # Fix destination.themes access
    content = content.replace(
        "discovered_themes_count = len(destination.themes)",
        """# Handle both objects and dictionaries for themes
            if hasattr(destination, 'themes'):
                discovered_themes_count = len(destination.themes)
            elif isinstance(destination, dict):
                discovered_themes_count = len(destination.get('themes', []))
            else:
                discovered_themes_count = 0"""
    )
    
    # Fix destination.id in data_quality_result call
    content = content.replace(
        "destination_name=destination.id,",
        "destination_name=destination_id,"
    )
    
    with open(file_path, 'w') as f:
        f.write(content)
    
    print(f"✅ Fixed attribute access errors in {file_path}")

def fix_chained_get_calls():
    """Fix all chained .get() method calls throughout the codebase"""
    
    # Files with chained .get() calls that need fixing
    files_to_fix = [
        "src/tools/priority_aggregation_tool.py",
        "src/tools/web_discovery_tools.py", 
        "src/agents/enrichment_agents.py",
        "demo_cultural_intelligence.py",
        "analyze_themes.py",
        "run_enhanced_agent_app.py"
    ]
    
    for file_path in files_to_fix:
        if not os.path.exists(file_path):
            print(f"⚠️  File not found: {file_path}")
            continue
            
        with open(file_path, 'r') as f:
            content = f.read()
        
        original_content = content
        
        # Add safe_get_nested import if not present
        if "from src.core.safe_dict_utils import" in content:
            # Update existing import
            content = re.sub(
                r"from src\.core\.safe_dict_utils import ([^\n]+)",
                r"from src.core.safe_dict_utils import \1, safe_get_nested",
                content
            )
        elif "from ..core.safe_dict_utils import" in content:
            # Update existing relative import
            content = re.sub(
                r"from \.\.core\.safe_dict_utils import ([^\n]+)",
                r"from ..core.safe_dict_utils import \1, safe_get_nested",
                content
            )
        else:
            # Add new import at the top
            if file_path.startswith("src/"):
                import_line = "from ..core.safe_dict_utils import safe_get_nested\n"
            else:
                import_line = "from src.core.safe_dict_utils import safe_get_nested\n"
            
            # Find a good place to insert the import
            lines = content.split('\n')
            insert_index = 0
            for i, line in enumerate(lines):
                if line.startswith('import ') or line.startswith('from '):
                    insert_index = i + 1
            
            lines.insert(insert_index, import_line.strip())
            content = '\n'.join(lines)
        
        # Fix common chained .get() patterns
        patterns_to_fix = [
            # config.get("llm_settings", {}).get("provider", "gemini")
            (r'config\.get\("([^"]+)", \{\}\)\.get\("([^"]+)", "([^"]+)"\)',
             r'safe_get_nested(config, ["\1", "\2"], "\3")'),
            
            # config.get("llm_settings", {}).get("provider")
            (r'config\.get\("([^"]+)", \{\}\)\.get\("([^"]+)"\)',
             r'safe_get_nested(config, ["\1", "\2"])'),
            
            # entry.get("safety", {}).get("crime_index")
            (r'entry\.get\("([^"]+)", \{\}\)\.get\("([^"]+)"\)',
             r'safe_get_nested(entry, ["\1", "\2"])'),
            
            # obj.get("key", {}).get("nested_key", default)
            (r'(\w+)\.get\("([^"]+)", \{\}\)\.get\("([^"]+)", ([^)]+)\)',
             r'safe_get_nested(\1, ["\2", "\3"], \4)'),
            
            # obj.get("key", {}).get("nested_key")
            (r'(\w+)\.get\("([^"]+)", \{\}\)\.get\("([^"]+)"\)',
             r'safe_get_nested(\1, ["\2", "\3"])'),
            
            # self.config.get("web_discovery", {}).get("min_content_length_chars", 200)
            (r'self\.config\.get\("([^"]+)", \{\}\)\.get\("([^"]+)", ([^)]+)\)',
             r'safe_get_nested(self.config, ["\1", "\2"], \3)'),
            
            # self.app_config.get('api_keys', {}).get('brave_search')
            (r"self\.app_config\.get\('([^']+)', \{\}\)\.get\('([^']+)'\)",
             r"safe_get_nested(self.app_config, ['\1', '\2'])"),
        ]
        
        for pattern, replacement in patterns_to_fix:
            content = re.sub(pattern, replacement, content)
        
        # Only write if content changed
        if content != original_content:
            with open(file_path, 'w') as f:
                f.write(content)
            print(f"✅ Fixed chained .get() calls in {file_path}")
        else:
            print(f"ℹ️  No changes needed in {file_path}")

def main():
    """Run all fixes"""
    print("🔧 Starting comprehensive attribute access error fixes...")
    
    # Fix the main JSON export error
    fix_consolidated_json_export_manager()
    
    # Fix all chained .get() calls
    fix_chained_get_calls()
    
    print("\n✅ All fixes completed!")
    print("\n🚀 You can now run the enhanced agent app again to test the fixes.")

if __name__ == "__main__":
    main() 