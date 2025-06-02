import os
import json
import hashlib
import logging
from datetime import datetime, timedelta
from typing import List, Any, Optional

CACHE_DIR = "cache" # This will be in the root, alongside src/
os.makedirs(CACHE_DIR, exist_ok=True)

CACHE_EXPIRY_DAYS = 7 
PAGE_CONTENT_CACHE_EXPIRY_DAYS = 30

def get_cache_path(key_parts: List[str]) -> str:
    """Generate a cache file path based on key parts."""
    project_root = os.path.join(os.path.dirname(__file__), '..')
    cache_dir_abs = os.path.join(project_root, CACHE_DIR)
    os.makedirs(cache_dir_abs, exist_ok=True)

    key = "_".join(key_parts).replace("https://", "").replace("http://", "").replace("/", "_").replace(":", "_")
    hashed_key = hashlib.md5(key.encode('utf-8')).hexdigest()
    return os.path.join(cache_dir_abs, f"{hashed_key}.json")

def read_from_cache(key_parts: List[str], expiry_days: int) -> Optional[Any]:
    """Read data from cache if it exists and is not expired."""
    cache_file = get_cache_path(key_parts)
    if os.path.exists(cache_file):
        try:
            with open(cache_file, 'r') as f:
                cached_data = json.load(f)
            
            timestamp_str = cached_data.get("timestamp")
            if timestamp_str:
                timestamp = datetime.fromisoformat(timestamp_str)
                if datetime.now() - timestamp < timedelta(days=expiry_days):
                    logging.info(f"CACHE HIT: Using cached data for key {'_'.join(key_parts)}")
                    return cached_data.get("data")
                else:
                    logging.info(f"CACHE EXPIRED: Cached data for key {'_'.join(key_parts)} is too old.")
            else: 
                 return cached_data 
        except (json.JSONDecodeError, IOError) as e:
            logging.warning(f"CACHE ERROR: Could not read or decode cache file {cache_file}: {e}")
    logging.info(f"CACHE MISS: No valid cache found for key {'_'.join(key_parts)}")
    return None

def write_to_cache(key_parts: List[str], data: Any):
    """Write data to cache with a timestamp."""
    cache_file = get_cache_path(key_parts)
    try:
        with open(cache_file, 'w') as f:
            json.dump({"timestamp": datetime.now().isoformat(), "data": data}, f, indent=4)
        logging.info(f"CACHE WRITE: Saved data to cache for key {'_'.join(key_parts)}")
    except IOError as e:
        logging.warning(f"CACHE ERROR: Could not write to cache file {cache_file}: {e}") 