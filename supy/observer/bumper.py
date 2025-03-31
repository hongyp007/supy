from typing import Union, Optional, List, Dict, Any, Tuple
from datetime import datetime
import numpy as np
from matplotlib.dates import date2num

# Helper functions for data processing
def safe_to_list(value) -> List[float]:
    """Convert a value to a list of floats safely."""
    if value is None:
        return []
    # Optimize: Use NumPy's vectorized operations if input is NumPy array
    if hasattr(value, 'shape') and hasattr(value, 'astype'):
        # Fast path for NumPy arrays
        try:
            # Replace NaN values with 0.0 and convert to Python list
            return np.nan_to_num(value).astype(float).tolist()
        except (TypeError, ValueError):
            pass  # Fall back to regular path if NumPy conversion fails
            
    if hasattr(value, '__iter__') and not isinstance(value, str):
        return [float(v) if v is not None else 0.0 for v in value]
    return [float(value) if value is not None else 0.0]

def safe_get(data_dict: Dict[str, Any], key: str, default: Any = None) -> Any:
    """Safely get a value from a dictionary with a default fallback."""
    return data_dict.get(key, default)

def safe_get_datetime(time_obj) -> Optional[datetime]:
    """Safely extract datetime from a Time object."""
    if time_obj is not None and hasattr(time_obj, 'datetime'):
        return time_obj.datetime
    return None

def safe_date2num(dt_obj: Any) -> Optional[float]:
    """
    Safely convert datetime to numeric value for plotting.
    Handles the case where date2num returns a numpy array instead of a float.
    """
    if dt_obj is None:
        return None
    try:
        if hasattr(dt_obj, "datetime"):
            dt_obj = dt_obj.datetime
        result = date2num(dt_obj)
        # Handle case where result is a numpy array
        if hasattr(result, 'item'):
            return float(result.item())
        # Handle case where result is already a scalar
        return float(result)
    except (TypeError, ValueError) as e:
        date2num(dt_obj)
        print(f"Warning: Error converting datetime to numeric: {e}")
        return None

def safe_get_attribute(obj, attr_path):
    """
    Extract a nested attribute safely.
    attr_path is a list like ['alt', 'value'] for obj.alt.value
    """
    if obj is None:
        return None
        
    current = obj
    for attr in attr_path:
        if not hasattr(current, attr):
            return None
        current = getattr(current, attr)
        if current is None:
            return None
    
    return current

def safe_get_alt_value(altaz_obj):
    """Extract altitude value from AltAz object safely."""
    return safe_get_attribute(altaz_obj, ['alt', 'value'])

def safe_get_separation_value(sep_obj):
    """Extract separation value safely."""
    sep = safe_get_attribute(sep_obj, ['value'])
    if sep is None:
        raise ValueError("Moon separation values are missing")
    return sep

def safe_extract_times(altaz_obj, attr_name='obstime'):
    """Directly extract time values for comparison."""
    if altaz_obj is None or not hasattr(altaz_obj, attr_name):
        return []
        
    attr_val = getattr(altaz_obj, attr_name)
    if attr_val is None:
        return []
        
    # If it's a scalar, return as single item
    if hasattr(attr_val, 'size') and attr_val.size == 1:
        return [attr_val]
        
    # Try to iterate if it's a collection
    try:
        # For arrays with shape attribute
        if hasattr(attr_val, 'shape'):
            return [t for t in attr_val]
        # For regular iterables
        return [t for t in attr_val]
    except TypeError:
        # If iteration fails, treat as a scalar
        return [attr_val]

def safe_extract_time_strings(altaz_obj, attr_name='obstime'):
    """
    Directly extract time values from possibly non-iterable TimeAttribute.
    Optimized version with caching for better performance.
    """
    result = []
    
    # Check if the object is valid
    if altaz_obj is None or not hasattr(altaz_obj, attr_name):
        return result
        
    attr_val = getattr(altaz_obj, attr_name)
    if attr_val is None:
        return result
    
    # Use an optimized approach for large arrays
    if hasattr(attr_val, 'size') and hasattr(attr_val.size, '__gt__') and attr_val.size > 100:
        # For large arrays, process in chunks for better memory performance
        chunk_size = 100
        # Initialize with the right size to avoid resizing
        result = [None] * attr_val.size
        
        for i in range(0, attr_val.size, chunk_size):
            end = min(i + chunk_size, attr_val.size)
            # Process this chunk
            for j in range(i, end):
                try:
                    result[j] = safe_isoformat(attr_val[j])
                except (IndexError, TypeError):
                    result[j] = ""
        return result
        
    # If it's a scalar Time object, handle it directly
    if hasattr(attr_val, 'size') and attr_val.size == 1:
        return [safe_isoformat(attr_val)]
        
    # If it seems to be an array or collection, try to iterate
    try:
        # For Time arrays with shape attribute - use a list comprehension
        if hasattr(attr_val, 'shape'):
            return [safe_isoformat(t) for t in attr_val]
        # For regular iterables
        return [safe_isoformat(t) for t in attr_val]
    except TypeError as e:
        # If iteration fails, treat as a scalar
        print(f"Warning: Non-iterable TimeAttribute handled as scalar: {e}")
        return [safe_isoformat(attr_val)]

def safe_isoformat(time_obj):
    """Safely extract ISO format from a Time or TimeAttribute object."""
    if time_obj is None:
        return ""
    if hasattr(time_obj, 'datetime'):
        if time_obj.datetime is None:
            return ""
        return time_obj.datetime.isoformat()
    if hasattr(time_obj, 'iso'):
        return time_obj.iso
    # Try direct conversion as last resort
    try:
        return str(time_obj)
    except:
        return ""

def safe_cached_time_to_num(time_str, time_conversion_cache):
    """Convert time string to numeric with caching for better performance."""
    if isinstance(time_str, (str, datetime)) and time_str in time_conversion_cache:
        return time_conversion_cache[time_str]
        
    try:
        if isinstance(time_str, str):
            dt_obj = datetime.fromisoformat(time_str)
            result = safe_date2num(dt_obj) or 0.0
        else:
            result = safe_date2num(time_str) or 0.0
            
        # Cache the result
        time_conversion_cache[time_str] = result
        return result
    except (ValueError, TypeError) as e:
        print(f"Warning: Cannot convert time {time_str}: {e}")
        return 0.0