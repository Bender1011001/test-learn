from typing import Dict, List, Optional, Any, Callable, Union
import time
from datetime import datetime
import functools
import asyncio
from loguru import logger
import json
import os
import threading
from contextlib import contextmanager
import statistics

# Try to import prometheus client if available
try:
    import prometheus_client as prom
    PROMETHEUS_AVAILABLE = True
except ImportError:
    PROMETHEUS_AVAILABLE = False
    logger.warning("Prometheus client not available. Metrics will not be exported.")


class MonitoringService:
    """Service for monitoring application performance and health"""
    
    def __init__(self, app_name: str = "camel_ext", enable_prometheus: bool = True):
        """Initialize the monitoring service"""
        self.app_name = app_name
        self.enable_prometheus = enable_prometheus and PROMETHEUS_AVAILABLE
        
        # Initialize metrics storage
        self._metrics: Dict[str, Dict[str, List[float]]] = {
            "api_latency": {},      # API endpoint latency
            "db_latency": {},       # Database operation latency
            "cache_hits": {},       # Cache hit counts
            "cache_misses": {},     # Cache miss counts
            "error_counts": {},     # Error counts by type
            "active_connections": {},  # Active WebSocket connections
            "workflow_durations": {},  # Workflow execution durations
        }
        
        # Thread lock for metrics access
        self._lock = threading.Lock()
        
        # Initialize Prometheus metrics if available
        if self.enable_prometheus:
            self._init_prometheus_metrics()
            
        logger.info(f"Monitoring service initialized for {app_name}")
    
    def _init_prometheus_metrics(self):
        """Initialize Prometheus metrics"""
        if not PROMETHEUS_AVAILABLE:
            return
            
        # API latency histogram
        self.prom_api_latency = prom.Histogram(
            name=f"{self.app_name}_api_latency_seconds",
            documentation="API endpoint latency in seconds",
            labelnames=["endpoint", "method"]
        )
        
        # Database latency histogram
        self.prom_db_latency = prom.Histogram(
            name=f"{self.app_name}_db_latency_seconds",
            documentation="Database operation latency in seconds",
            labelnames=["operation", "table"]
        )
        
        # Cache hit/miss counter
        self.prom_cache_hits = prom.Counter(
            name=f"{self.app_name}_cache_hits_total",
            documentation="Cache hit count",
            labelnames=["cache_key"]
        )
        
        self.prom_cache_misses = prom.Counter(
            name=f"{self.app_name}_cache_misses_total",
            documentation="Cache miss count",
            labelnames=["cache_key"]
        )
        
        # Error counter
        self.prom_errors = prom.Counter(
            name=f"{self.app_name}_errors_total",
            documentation="Error count by type",
            labelnames=["error_type", "source"]
        )
        
        # Active connections gauge
        self.prom_active_connections = prom.Gauge(
            name=f"{self.app_name}_active_connections",
            documentation="Number of active WebSocket connections",
            labelnames=["connection_type"]
        )
        
        # Workflow duration histogram
        self.prom_workflow_duration = prom.Histogram(
            name=f"{self.app_name}_workflow_duration_seconds",
            documentation="Workflow execution duration in seconds",
            labelnames=["workflow_id"]
        )
        
        # Start Prometheus HTTP server if enabled via environment variable
        prometheus_port = os.environ.get("PROMETHEUS_PORT")
        if prometheus_port:
            try:
                port = int(prometheus_port)
                prom.start_http_server(port)
                logger.info(f"Prometheus metrics server started on port {port}")
            except (ValueError, OSError) as e:
                logger.error(f"Failed to start Prometheus server: {str(e)}")
    
    def record_api_latency(self, endpoint: str, method: str, latency: float):
        """Record API endpoint latency"""
        key = f"{method}:{endpoint}"
        with self._lock:
            if key not in self._metrics["api_latency"]:
                self._metrics["api_latency"][key] = []
            self._metrics["api_latency"][key].append(latency)
        
        # Record in Prometheus if enabled
        if self.enable_prometheus:
            self.prom_api_latency.labels(endpoint=endpoint, method=method).observe(latency)
    
    def record_db_latency(self, operation: str, table: str, latency: float):
        """Record database operation latency"""
        key = f"{operation}:{table}"
        with self._lock:
            if key not in self._metrics["db_latency"]:
                self._metrics["db_latency"][key] = []
            self._metrics["db_latency"][key].append(latency)
        
        # Record in Prometheus if enabled
        if self.enable_prometheus:
            self.prom_db_latency.labels(operation=operation, table=table).observe(latency)
    
    def record_cache_hit(self, cache_key: str):
        """Record cache hit"""
        with self._lock:
            if cache_key not in self._metrics["cache_hits"]:
                self._metrics["cache_hits"][cache_key] = []
            # Use 1.0 as a counter increment
            self._metrics["cache_hits"][cache_key].append(1.0)
        
        # Record in Prometheus if enabled
        if self.enable_prometheus:
            self.prom_cache_hits.labels(cache_key=cache_key).inc()
    
    def record_cache_miss(self, cache_key: str):
        """Record cache miss"""
        with self._lock:
            if cache_key not in self._metrics["cache_misses"]:
                self._metrics["cache_misses"][cache_key] = []
            # Use 1.0 as a counter increment
            self._metrics["cache_misses"][cache_key].append(1.0)
        
        # Record in Prometheus if enabled
        if self.enable_prometheus:
            self.prom_cache_misses.labels(cache_key=cache_key).inc()
    
    def record_error(self, error_type: str, source: str):
        """Record error occurrence"""
        key = f"{error_type}:{source}"
        with self._lock:
            if key not in self._metrics["error_counts"]:
                self._metrics["error_counts"][key] = []
            # Use 1.0 as a counter increment
            self._metrics["error_counts"][key].append(1.0)
        
        # Record in Prometheus if enabled
        if self.enable_prometheus:
            self.prom_errors.labels(error_type=error_type, source=source).inc()
    
    def update_active_connections(self, connection_type: str, count: int):
        """Update active connection count"""
        with self._lock:
            self._metrics["active_connections"][connection_type] = [float(count)]
        
        # Record in Prometheus if enabled
        if self.enable_prometheus:
            self.prom_active_connections.labels(connection_type=connection_type).set(count)
    
    def record_workflow_duration(self, workflow_id: str, duration: float):
        """Record workflow execution duration"""
        with self._lock:
            if workflow_id not in self._metrics["workflow_durations"]:
                self._metrics["workflow_durations"][workflow_id] = []
            self._metrics["workflow_durations"][workflow_id].append(duration)
        
        # Record in Prometheus if enabled
        if self.enable_prometheus:
            self.prom_workflow_duration.labels(workflow_id=workflow_id).observe(duration)
    
    def get_metrics_summary(self) -> Dict[str, Any]:
        """Get summary of all metrics"""
        with self._lock:
            summary = {}
            
            # Process API latency metrics
            api_latency = {}
            for key, values in self._metrics["api_latency"].items():
                if values:
                    api_latency[key] = {
                        "count": len(values),
                        "min": min(values),
                        "max": max(values),
                        "avg": sum(values) / len(values),
                        "p95": self._percentile(values, 95),
                        "p99": self._percentile(values, 99)
                    }
            summary["api_latency"] = api_latency
            
            # Process DB latency metrics
            db_latency = {}
            for key, values in self._metrics["db_latency"].items():
                if values:
                    db_latency[key] = {
                        "count": len(values),
                        "min": min(values),
                        "max": max(values),
                        "avg": sum(values) / len(values),
                        "p95": self._percentile(values, 95),
                        "p99": self._percentile(values, 99)
                    }
            summary["db_latency"] = db_latency
            
            # Process cache metrics
            cache_hits = {k: sum(v) for k, v in self._metrics["cache_hits"].items()}
            cache_misses = {k: sum(v) for k, v in self._metrics["cache_misses"].items()}
            
            # Calculate hit rates
            cache_hit_rates = {}
            for key in set(cache_hits.keys()).union(cache_misses.keys()):
                hits = cache_hits.get(key, 0)
                misses = cache_misses.get(key, 0)
                total = hits + misses
                rate = hits / total if total > 0 else 0
                cache_hit_rates[key] = rate
            
            summary["cache"] = {
                "hits": cache_hits,
                "misses": cache_misses,
                "hit_rates": cache_hit_rates
            }
            
            # Process error counts
            summary["errors"] = {k: sum(v) for k, v in self._metrics["error_counts"].items()}
            
            # Process active connections
            summary["active_connections"] = {k: v[0] if v else 0 for k, v in self._metrics["active_connections"].items()}
            
            # Process workflow durations
            workflow_durations = {}
            for key, values in self._metrics["workflow_durations"].items():
                if values:
                    workflow_durations[key] = {
                        "count": len(values),
                        "min": min(values),
                        "max": max(values),
                        "avg": sum(values) / len(values)
                    }
            summary["workflow_durations"] = workflow_durations
            
            return summary
    
    def _percentile(self, values: List[float], percentile: int) -> float:
        """Calculate percentile value from a list"""
        if not values:
            return 0.0
        
        try:
            return statistics.quantiles(sorted(values), n=100)[percentile-1]
        except (IndexError, ValueError):
            # Fallback for small datasets
            sorted_values = sorted(values)
            k = (len(sorted_values) - 1) * percentile / 100
            f = int(k)
            c = int(k) + 1 if k > f else f
            if f >= len(sorted_values):
                return sorted_values[-1]
            if c >= len(sorted_values):
                return sorted_values[-1]
            return sorted_values[f] + (sorted_values[c] - sorted_values[f]) * (k - f)
    
    def reset_metrics(self):
        """Reset all metrics"""
        with self._lock:
            for category in self._metrics:
                self._metrics[category] = {}
    
    @contextmanager
    def measure_time(self, metric_type: str, **labels):
        """Context manager to measure execution time"""
        start_time = time.time()
        try:
            yield
        finally:
            end_time = time.time()
            duration = end_time - start_time
            
            if metric_type == "api":
                self.record_api_latency(
                    endpoint=labels.get("endpoint", "unknown"),
                    method=labels.get("method", "unknown"),
                    latency=duration
                )
            elif metric_type == "db":
                self.record_db_latency(
                    operation=labels.get("operation", "unknown"),
                    table=labels.get("table", "unknown"),
                    latency=duration
                )
            elif metric_type == "workflow":
                self.record_workflow_duration(
                    workflow_id=labels.get("workflow_id", "unknown"),
                    duration=duration
                )
    
    async def measure_time_async(self, metric_type: str, **labels):
        """Async context manager to measure execution time"""
        start_time = time.time()
        try:
            yield
        finally:
            end_time = time.time()
            duration = end_time - start_time
            
            if metric_type == "api":
                self.record_api_latency(
                    endpoint=labels.get("endpoint", "unknown"),
                    method=labels.get("method", "unknown"),
                    latency=duration
                )
            elif metric_type == "db":
                self.record_db_latency(
                    operation=labels.get("operation", "unknown"),
                    table=labels.get("table", "unknown"),
                    latency=duration
                )
            elif metric_type == "workflow":
                self.record_workflow_duration(
                    workflow_id=labels.get("workflow_id", "unknown"),
                    duration=duration
                )
    
    def time_function(self, metric_type: str, **labels):
        """Decorator to measure function execution time"""
        def decorator(func):
            @functools.wraps(func)
            def wrapper(*args, **kwargs):
                with self.measure_time(metric_type, **labels):
                    return func(*args, **kwargs)
            return wrapper
        return decorator
    
    def time_async_function(self, metric_type: str, **labels):
        """Decorator to measure async function execution time"""
        def decorator(func):
            @functools.wraps(func)
            async def wrapper(*args, **kwargs):
                start_time = time.time()
                try:
                    return await func(*args, **kwargs)
                finally:
                    end_time = time.time()
                    duration = end_time - start_time
                    
                    if metric_type == "api":
                        self.record_api_latency(
                            endpoint=labels.get("endpoint", "unknown"),
                            method=labels.get("method", "unknown"),
                            latency=duration
                        )
                    elif metric_type == "db":
                        self.record_db_latency(
                            operation=labels.get("operation", "unknown"),
                            table=labels.get("table", "unknown"),
                            latency=duration
                        )
                    elif metric_type == "workflow":
                        self.record_workflow_duration(
                            workflow_id=labels.get("workflow_id", "unknown"),
                            duration=duration
                        )
            return wrapper
        return decorator


# Singleton instance
_instance = None

def get_monitoring_service() -> MonitoringService:
    """Get the singleton monitoring service instance"""
    global _instance
    if _instance is None:
        app_name = os.environ.get("APP_NAME", "camel_ext")
        enable_prometheus = os.environ.get("ENABLE_PROMETHEUS", "true").lower() == "true"
        _instance = MonitoringService(app_name=app_name, enable_prometheus=enable_prometheus)
    return _instance