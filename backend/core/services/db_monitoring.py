from typing import Dict, List, Optional, Any, Union, Callable, Type, TypeVar
import functools
import time
import inspect
import asyncio
from loguru import logger

from .monitoring_service import get_monitoring_service

# Type variable for generic function return type
T = TypeVar('T')

def monitor_db_operation(operation: str, table: str = "unknown"):
    """
    Decorator to monitor database operations
    
    Args:
        operation: Name of the database operation (e.g., "query", "insert", "update")
        table: Name of the table being operated on
    """
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Get monitoring service
            monitoring_service = get_monitoring_service()
            
            # Start timing
            start_time = time.time()
            
            # Record cache operations
            if "use_cache" in kwargs and kwargs["use_cache"]:
                cache_key = None
                if hasattr(args[0], "_get_cache_key"):
                    # Try to extract cache key from the function
                    try:
                        # Get the first parameter name from the function signature
                        sig = inspect.signature(func)
                        params = list(sig.parameters.keys())
                        if len(params) > 1:  # Skip 'self'
                            first_param = params[1]
                            if first_param in kwargs:
                                cache_key = args[0]._get_cache_key(
                                    func.__name__, **{first_param: kwargs[first_param]}
                                )
                    except Exception as e:
                        logger.debug(f"Error extracting cache key: {str(e)}")
                
                # Check if we got a cache hit from the function
                cache_hit = False
                try:
                    # Call the function
                    result = func(*args, **kwargs)
                    
                    # Try to determine if this was a cache hit
                    # This is a heuristic and may not be accurate for all functions
                    end_time = time.time()
                    duration = end_time - start_time
                    
                    # If the function completed very quickly, it might be a cache hit
                    if duration < 0.005:  # 5ms threshold
                        cache_hit = True
                        if cache_key:
                            monitoring_service.record_cache_hit(cache_key)
                    else:
                        if cache_key:
                            monitoring_service.record_cache_miss(cache_key)
                    
                    return result
                except Exception as e:
                    # Record error
                    monitoring_service.record_error(
                        error_type=type(e).__name__,
                        source=f"db:{operation}:{table}"
                    )
                    # Re-raise the exception
                    raise
            else:
                try:
                    # Call the function without cache
                    result = func(*args, **kwargs)
                    return result
                except Exception as e:
                    # Record error
                    monitoring_service.record_error(
                        error_type=type(e).__name__,
                        source=f"db:{operation}:{table}"
                    )
                    # Re-raise the exception
                    raise
                finally:
                    # Record timing
                    end_time = time.time()
                    duration = end_time - start_time
                    monitoring_service.record_db_latency(operation, table, duration)
        
        return wrapper
    
    return decorator


def monitor_db_operation_async(operation: str, table: str = "unknown"):
    """
    Decorator to monitor async database operations
    
    Args:
        operation: Name of the database operation (e.g., "query", "insert", "update")
        table: Name of the table being operated on
    """
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            # Get monitoring service
            monitoring_service = get_monitoring_service()
            
            # Start timing
            start_time = time.time()
            
            # Record cache operations
            if "use_cache" in kwargs and kwargs["use_cache"]:
                cache_key = None
                if hasattr(args[0], "_get_cache_key"):
                    # Try to extract cache key from the function
                    try:
                        # Get the first parameter name from the function signature
                        sig = inspect.signature(func)
                        params = list(sig.parameters.keys())
                        if len(params) > 1:  # Skip 'self'
                            first_param = params[1]
                            if first_param in kwargs:
                                cache_key = args[0]._get_cache_key(
                                    func.__name__, **{first_param: kwargs[first_param]}
                                )
                    except Exception as e:
                        logger.debug(f"Error extracting cache key: {str(e)}")
                
                # Check if we got a cache hit from the function
                cache_hit = False
                try:
                    # Call the function
                    result = await func(*args, **kwargs)
                    
                    # Try to determine if this was a cache hit
                    # This is a heuristic and may not be accurate for all functions
                    end_time = time.time()
                    duration = end_time - start_time
                    
                    # If the function completed very quickly, it might be a cache hit
                    if duration < 0.005:  # 5ms threshold
                        cache_hit = True
                        if cache_key:
                            monitoring_service.record_cache_hit(cache_key)
                    else:
                        if cache_key:
                            monitoring_service.record_cache_miss(cache_key)
                    
                    return result
                except Exception as e:
                    # Record error
                    monitoring_service.record_error(
                        error_type=type(e).__name__,
                        source=f"db:{operation}:{table}"
                    )
                    # Re-raise the exception
                    raise
            else:
                try:
                    # Call the function without cache
                    result = await func(*args, **kwargs)
                    return result
                except Exception as e:
                    # Record error
                    monitoring_service.record_error(
                        error_type=type(e).__name__,
                        source=f"db:{operation}:{table}"
                    )
                    # Re-raise the exception
                    raise
                finally:
                    # Record timing
                    end_time = time.time()
                    duration = end_time - start_time
                    monitoring_service.record_db_latency(operation, table, duration)
        
        return wrapper
    
    return decorator


def apply_db_monitoring(db_manager_class: Type) -> Type:
    """
    Apply monitoring decorators to all methods of a DBManager class
    
    Args:
        db_manager_class: The DBManager class to decorate
        
    Returns:
        The decorated class
    """
    # Map of method name patterns to operation types
    operation_map = {
        "get_logs": ("query", "logs"),
        "get_log_by_id": ("query", "logs"),
        "get_logs_summary": ("query", "logs"),
        "get_annotation": ("query", "annotations"),
        "save_annotation": ("update", "annotations"),
        "delete_annotation": ("delete", "annotations"),
        "get_dpo_ready_annotations": ("query", "annotations"),
        "export_dpo_data": ("query", "annotations"),
        "batch_save_annotations": ("batch_update", "annotations"),
        "batch_save_logs": ("batch_insert", "logs"),
    }
    
    # Get all methods from the class
    for name, method in inspect.getmembers(db_manager_class, inspect.isfunction):
        # Skip private methods
        if name.startswith('_'):
            continue
        
        # Determine operation type and table
        operation = "query"  # Default
        table = "unknown"
        
        # Check if method name matches any patterns
        for pattern, (op, tbl) in operation_map.items():
            if pattern in name:
                operation = op
                table = tbl
                break
        
        # Apply appropriate decorator based on whether the method is async
        if asyncio.iscoroutinefunction(method):
            setattr(
                db_manager_class, 
                name, 
                monitor_db_operation_async(operation, table)(method)
            )
        else:
            setattr(
                db_manager_class, 
                name, 
                monitor_db_operation(operation, table)(method)
            )
    
    return db_manager_class