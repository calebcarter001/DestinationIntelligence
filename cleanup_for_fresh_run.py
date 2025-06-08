#!/usr/bin/env python3
"""
Comprehensive cleanup script for fresh runs of the Destination Intelligence system.
Removes all cached data, logs, databases, and previous outputs.
"""

import os
import shutil
import glob
from pathlib import Path

def clean_directory(dir_path, description):
    """Clean a directory and report results"""
    if os.path.exists(dir_path):
        try:
            if os.path.isfile(dir_path):
                os.remove(dir_path)
                print(f"✅ Removed file: {description}")
            else:
                files_removed = 0
                for item in os.listdir(dir_path):
                    item_path = os.path.join(dir_path, item)
                    if os.path.isfile(item_path):
                        os.remove(item_path)
                        files_removed += 1
                    elif os.path.isdir(item_path):
                        shutil.rmtree(item_path)
                        files_removed += 1
                
                if files_removed > 0:
                    print(f"✅ Cleaned {description}: removed {files_removed} items")
                else:
                    print(f"✅ {description}: already clean")
        except Exception as e:
            print(f"⚠️  Error cleaning {description}: {e}")
    else:
        print(f"✅ {description}: directory doesn't exist (clean)")

def main():
    """Run comprehensive cleanup"""
    print("🧹 Starting comprehensive cleanup for fresh run...")
    print("=" * 60)
    
    # Define cleanup targets
    cleanup_targets = [
        ("logs", "Application logs"),
        ("cache", "File-based cache"),
        ("chroma_db", "ChromaDB vector database"),
        ("outputs", "Previous output files"),
        ("destination_insights", "Previous destination insights"),
        ("test_destination_insights", "Test destination insights"),
    ]
    
    # Database files
    db_files = [
        "enhanced_destination_intelligence.db",
        "test_enhanced_destination_intelligence.db"
    ]
    
    # Clean directories
    for dir_path, description in cleanup_targets:
        clean_directory(dir_path, description)
    
    # Clean database files
    for db_file in db_files:
        if os.path.exists(db_file):
            try:
                os.remove(db_file)
                print(f"✅ Removed database: {db_file}")
            except Exception as e:
                print(f"⚠️  Error removing {db_file}: {e}")
        else:
            print(f"✅ Database file {db_file}: doesn't exist (clean)")
    
    # Clean any stray log files in root
    log_files = glob.glob("*.log")
    if log_files:
        for log_file in log_files:
            try:
                os.remove(log_file)
                print(f"✅ Removed stray log file: {log_file}")
            except Exception as e:
                print(f"⚠️  Error removing {log_file}: {e}")
    
    # Clean any temporary test files
    temp_files = glob.glob("debug_*.py") + glob.glob("temp_*.py") + glob.glob("test_*.py")
    temp_files = [f for f in temp_files if f not in ["test_storage_theme_conversion.py"]]  # Keep legitimate test files
    
    if temp_files:
        for temp_file in temp_files:
            try:
                os.remove(temp_file)
                print(f"✅ Removed temporary file: {temp_file}")
            except Exception as e:
                print(f"⚠️  Error removing {temp_file}: {e}")
    
    print("\n" + "=" * 60)
    print("🎉 Cleanup completed! System ready for fresh run.")
    print("\n📊 Post-cleanup status:")
    
    # Verify cleanup
    verification_dirs = ["logs", "cache", "chroma_db", "outputs", "destination_insights"]
    for dir_name in verification_dirs:
        if os.path.exists(dir_name):
            item_count = len(os.listdir(dir_name))
            status = "✅ Empty" if item_count == 0 else f"⚠️  {item_count} items remaining"
            print(f"   {dir_name}/: {status}")
        else:
            print(f"   {dir_name}/: ✅ Directory doesn't exist")
    
    # Check database files
    for db_file in db_files:
        status = "✅ Removed" if not os.path.exists(db_file) else "⚠️  Still exists"
        print(f"   {db_file}: {status}")
    
    print("\n🚀 Ready for fresh run with: python run_enhanced_agent_app.py")

if __name__ == "__main__":
    main() 