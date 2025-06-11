#!/usr/bin/env python3
"""
Safe Dictionary Utilities
Comprehensive utilities to handle variables that could be dicts, objects, JSON strings, or None
"""

import json
import logging
from typing import Any, Union, Dict, Optional

logger = logging.getLogger(__name__)

def safe_get(obj: Any, key: str, default: Any = None) -> Any:
    """
    Safely get a value from an object that could be:
    - Dict
    - Object with attributes  
    - JSON string
    - None
    - Any other type
    """
    if obj is None:
        return default
    
    # Handle dictionary
    if isinstance(obj, dict):
        return obj.get(key, default)
    
    # Handle object with attributes
    if hasattr(obj, key):
        return getattr(obj, key, default)
    
    # Handle JSON string
    if isinstance(obj, str):
        try:
            parsed = json.loads(obj)
            if isinstance(parsed, dict):
                return parsed.get(key, default)
        except (json.JSONDecodeError, TypeError):
            pass
    
    # Fallback for any other type
    return default

def safe_get_confidence_value(confidence_breakdown: Any, key: str = 'overall_confidence', default: float = 0.0) -> float:
    """
    Safely extract confidence values from confidence_breakdown that could be:
    - Dict
    - Object with attributes
    - JSON string
    - None
    """
    result = safe_get(confidence_breakdown, key, default)
    
    # Ensure we return a float
    try:
        return float(result)
    except (ValueError, TypeError):
        return default


def safe_get_nested(obj: Any, keys: list, default: Any = None) -> Any:
    """
    Safely get nested values like obj.get('level1', {}).get('level2', default)
    Usage: safe_get_nested(obj, ['level1', 'level2'], default)
    """
    current = obj
    
    for key in keys:
        current = safe_get(current, key)
        if current is None:
            return default
    
    return current if current is not None else default


def safe_get_list(obj: Any, key: str, default: list = None) -> list:
    """
    Safely get a list value, ensuring it's always a list
    """
    if default is None:
        default = []
    
    result = safe_get(obj, key, default)
    
    # Ensure we return a list
    if isinstance(result, list):
        return result
    elif isinstance(result, (str, dict)) and result:
        return [result]  # Convert single item to list
    else:
        return default


def safe_get_dict(obj: Any, key: str, default: dict = None) -> dict:
    """
    Safely get a dict value, ensuring it's always a dict
    """
    if default is None:
        default = {}
    
    result = safe_get(obj, key, default)
    
    # Ensure we return a dict
    if isinstance(result, dict):
        return result
    else:
        return default 