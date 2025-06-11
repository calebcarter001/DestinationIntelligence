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
        (".pytest_cache", "Pytest cache directory"),
    ]
    
    # Database files and large data files
    db_files = [
        "enhanced_destination_intelligence.db",
        "test_enhanced_destination_intelligence.db", 
        ":memory:",  # SQLite memory database file
        "comprehensive_database_report.json",  # Large 2GB+ report file
    ]
    
    # Generated output files that should be cleaned
    generated_files = [
        "dynamic_database_viewer.html",
        "results_viewer.html", 
        "run_enhanced_agent_app.py.lprof",  # Profiling output
    ]
    
    # IDE and editor artifacts (optional cleanup)
    ide_dirs = ['.vscode', '.idea', '.sublime-project', '.sublime-workspace']
    
    # Clean directories
    for dir_path, description in cleanup_targets:
        clean_directory(dir_path, description)
    
    # Clean IDE directories (if they exist)
    for ide_dir in ide_dirs:
        if os.path.exists(ide_dir):
            clean_directory(ide_dir, f"IDE artifacts ({ide_dir})")
    
    # Clean database files and large data files
    for db_file in db_files:
        if os.path.exists(db_file):
            try:
                file_size = os.path.getsize(db_file) / (1024*1024)  # Size in MB
                os.remove(db_file)
                if file_size > 100:  # Show size for large files
                    print(f"✅ Removed large file: {db_file} ({file_size:.1f}MB)")
                else:
                    print(f"✅ Removed file: {db_file}")
            except Exception as e:
                print(f"⚠️  Error removing {db_file}: {e}")
        else:
            print(f"✅ File {db_file}: doesn't exist (clean)")
    
    # Clean generated output files
    for gen_file in generated_files:
        if os.path.exists(gen_file):
            try:
                os.remove(gen_file)
                print(f"✅ Removed generated file: {gen_file}")
            except Exception as e:
                print(f"⚠️  Error removing {gen_file}: {e}")
    
    # NEW: Clean dynamic viewer HTML files (pattern-based cleanup)
    print("🧹 Cleaning dynamic viewer HTML files...")
    dynamic_viewer_files = glob.glob("dynamic_viewer_*.html")
    if dynamic_viewer_files:
        for dv_file in dynamic_viewer_files:
            try:
                file_size = os.path.getsize(dv_file) / (1024*1024)  # Size in MB
                os.remove(dv_file)
                if file_size > 0.1:  # Show size for files > 100KB
                    print(f"✅ Removed dynamic viewer: {dv_file} ({file_size:.1f}MB)")
                else:
                    print(f"✅ Removed dynamic viewer: {dv_file}")
            except Exception as e:
                print(f"⚠️  Error removing {dv_file}: {e}")
    else:
        print("✅ No dynamic viewer HTML files found")
    
    # Clean Python cache files recursively
    print("🧹 Cleaning Python cache files...")
    cache_cleaned = 0
    
    # Remove __pycache__ directories recursively
    for root, dirs, files in os.walk('.'):
        if '__pycache__' in dirs:
            pycache_path = os.path.join(root, '__pycache__')
            try:
                shutil.rmtree(pycache_path)
                cache_cleaned += 1
            except Exception as e:
                print(f"⚠️  Error removing {pycache_path}: {e}")
    
    # Remove .pyc and .pyo files recursively
    pyc_files = glob.glob('**/*.pyc', recursive=True) + glob.glob('**/*.pyo', recursive=True)
    for pyc_file in pyc_files:
        try:
            os.remove(pyc_file)
            cache_cleaned += 1
        except Exception as e:
            print(f"⚠️  Error removing {pyc_file}: {e}")
    
    if cache_cleaned > 0:
        print(f"✅ Cleaned Python cache: removed {cache_cleaned} items")
    else:
        print("✅ Python cache: already clean")
    
    # Clean system files and development artifacts
    system_files = glob.glob('.DS_Store') + glob.glob('**/.DS_Store', recursive=True)
    system_files += glob.glob('.coverage') + glob.glob('**/*.db-journal', recursive=True)
    system_files += glob.glob('**/*.sqlite', recursive=True) + glob.glob('**/*.sqlite3', recursive=True)
    
    # NEW: Additional development artifacts
    system_files += glob.glob('.env.local') + glob.glob('.env.*.local')  # Local env files
    system_files += glob.glob('**/*.prof', recursive=True)  # Profiling files
    system_files += glob.glob('**/*.pstats', recursive=True)  # Performance stats
    system_files += glob.glob('.coverage.*')  # Coverage data files
    
    if system_files:
        for sys_file in system_files:
            try:
                # Skip files in venv directory
                if 'venv/' in sys_file or sys_file.startswith('venv/'):
                    continue
                os.remove(sys_file)
                print(f"✅ Removed system file: {sys_file}")
            except Exception as e:
                print(f"⚠️  Error removing {sys_file}: {e}")
    
    # Clean coverage HTML reports
    if os.path.exists('htmlcov'):
        clean_directory('htmlcov', "Coverage HTML reports")
    
    # Clean any stray log files in root
    log_files = glob.glob("*.log")
    if log_files:
        for log_file in log_files:
            try:
                os.remove(log_file)
                print(f"✅ Removed stray log file: {log_file}")
            except Exception as e:
                print(f"⚠️  Error removing {log_file}: {e}")
    
    # Clean any temporary test files (but preserve legitimate ones)
    temp_files = glob.glob("debug_*.py") + glob.glob("temp_*.py")
    # Be more selective with test files - only remove obvious temporary ones
    temp_test_files = [f for f in glob.glob("test_*.py") if any(pattern in f for pattern in ["temp_", "debug_", "_tmp", "_test_run"])]
    temp_files.extend(temp_test_files)
    
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
    
    # Verify cleanup of directories
    verification_dirs = ["logs", "cache", "chroma_db", "outputs", "destination_insights", ".pytest_cache", "htmlcov"]
    for dir_name in verification_dirs:
        if os.path.exists(dir_name):
            item_count = len(os.listdir(dir_name))
            status = "✅ Empty" if item_count == 0 else f"⚠️  {item_count} items remaining"
            print(f"   {dir_name}/: {status}")
        else:
            print(f"   {dir_name}/: ✅ Directory doesn't exist")
    
    # Check database and large files
    for db_file in db_files:
        status = "✅ Removed" if not os.path.exists(db_file) else "⚠️  Still exists"
        print(f"   {db_file}: {status}")
    
    # Check generated files
    for gen_file in generated_files:
        status = "✅ Removed" if not os.path.exists(gen_file) else "⚠️  Still exists"
        print(f"   {gen_file}: {status}")
    
    # NEW: Check dynamic viewer files
    remaining_dynamic_viewers = glob.glob("dynamic_viewer_*.html")
    if remaining_dynamic_viewers:
        print(f"   Dynamic viewer HTMLs: ⚠️  {len(remaining_dynamic_viewers)} remaining")
        for dv_file in remaining_dynamic_viewers[:3]:  # Show first 3
            print(f"      - {dv_file}")
        if len(remaining_dynamic_viewers) > 3:
            print(f"      - ... and {len(remaining_dynamic_viewers) - 3} more")
    else:
        print("   Dynamic viewer HTMLs: ✅ All removed")
    
    # Check for remaining cache files
    remaining_pycache = len(glob.glob('**/__pycache__', recursive=True))
    remaining_pyc = len(glob.glob('**/*.pyc', recursive=True))
    remaining_ds_store = len(glob.glob('**/.DS_Store', recursive=True))
    
    cache_status = "✅ Clean" if (remaining_pycache + remaining_pyc + remaining_ds_store) == 0 else f"⚠️  {remaining_pycache + remaining_pyc + remaining_ds_store} items remaining"
    print(f"   Python cache files: {cache_status}")
    
    if remaining_ds_store > 0:
        print(f"   .DS_Store files: ⚠️  {remaining_ds_store} remaining")
    
    # NEW: Check for development artifacts
    remaining_prof = len(glob.glob('**/*.prof', recursive=True))
    remaining_coverage = len(glob.glob('.coverage*'))
    if remaining_prof > 0 or remaining_coverage > 0:
        print(f"   Development artifacts: ⚠️  {remaining_prof + remaining_coverage} files remaining")
    else:
        print("   Development artifacts: ✅ Clean")
    
    print("\n🚀 Ready for fresh run with: python run_enhanced_agent_app.py")

if __name__ == "__main__":
    main() 