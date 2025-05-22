from fastapi import APIRouter, Depends, HTTPException, Response, Query
from typing import Dict, List, Optional, Any
from loguru import logger
import json
import os
from datetime import datetime, timedelta

from ...core.services.monitoring_service import get_monitoring_service
from ...db.base import get_db

# Create router
router = APIRouter()

@router.get("", response_model=Dict[str, Any])
async def get_metrics_summary():
    """Get summary of all metrics"""
    try:
        monitoring_service = get_monitoring_service()
        metrics = monitoring_service.get_metrics_summary()
        return metrics
    except Exception as e:
        logger.error(f"Error getting metrics summary: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error retrieving metrics: {str(e)}")


@router.get("/api", response_model=Dict[str, Any])
async def get_api_metrics():
    """Get API endpoint metrics"""
    try:
        monitoring_service = get_monitoring_service()
        metrics = monitoring_service.get_metrics_summary()
        return {"api_latency": metrics.get("api_latency", {})}
    except Exception as e:
        logger.error(f"Error getting API metrics: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error retrieving API metrics: {str(e)}")


@router.get("/db", response_model=Dict[str, Any])
async def get_db_metrics():
    """Get database metrics"""
    try:
        monitoring_service = get_monitoring_service()
        metrics = monitoring_service.get_metrics_summary()
        return {"db_latency": metrics.get("db_latency", {})}
    except Exception as e:
        logger.error(f"Error getting DB metrics: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error retrieving DB metrics: {str(e)}")


@router.get("/cache", response_model=Dict[str, Any])
async def get_cache_metrics():
    """Get cache metrics"""
    try:
        monitoring_service = get_monitoring_service()
        metrics = monitoring_service.get_metrics_summary()
        return {"cache": metrics.get("cache", {})}
    except Exception as e:
        logger.error(f"Error getting cache metrics: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error retrieving cache metrics: {str(e)}")


@router.get("/errors", response_model=Dict[str, Any])
async def get_error_metrics():
    """Get error metrics"""
    try:
        monitoring_service = get_monitoring_service()
        metrics = monitoring_service.get_metrics_summary()
        return {"errors": metrics.get("errors", {})}
    except Exception as e:
        logger.error(f"Error getting error metrics: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error retrieving error metrics: {str(e)}")


@router.get("/workflows", response_model=Dict[str, Any])
async def get_workflow_metrics():
    """Get workflow metrics"""
    try:
        monitoring_service = get_monitoring_service()
        metrics = monitoring_service.get_metrics_summary()
        return {"workflow_durations": metrics.get("workflow_durations", {})}
    except Exception as e:
        logger.error(f"Error getting workflow metrics: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error retrieving workflow metrics: {str(e)}")


@router.get("/connections", response_model=Dict[str, Any])
async def get_connection_metrics():
    """Get connection metrics"""
    try:
        monitoring_service = get_monitoring_service()
        metrics = monitoring_service.get_metrics_summary()
        return {"active_connections": metrics.get("active_connections", {})}
    except Exception as e:
        logger.error(f"Error getting connection metrics: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error retrieving connection metrics: {str(e)}")


@router.post("/reset", response_model=Dict[str, bool])
async def reset_metrics():
    """Reset all metrics"""
    try:
        monitoring_service = get_monitoring_service()
        monitoring_service.reset_metrics()
        return {"success": True}
    except Exception as e:
        logger.error(f"Error resetting metrics: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error resetting metrics: {str(e)}")


@router.get("/prometheus", response_class=Response)
async def get_prometheus_metrics():
    """Get Prometheus metrics"""
    try:
        # Check if prometheus_client is available
        try:
            import prometheus_client
            from prometheus_client import generate_latest, CONTENT_TYPE_LATEST
            
            # Generate metrics
            metrics_data = generate_latest()
            
            return Response(
                content=metrics_data,
                media_type=CONTENT_TYPE_LATEST
            )
        except ImportError:
            return Response(
                content="Prometheus client not available",
                media_type="text/plain",
                status_code=501
            )
    except Exception as e:
        logger.error(f"Error generating Prometheus metrics: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error generating Prometheus metrics: {str(e)}")


@router.get("/health", response_model=Dict[str, Any])
async def get_health_metrics(db=Depends(get_db)):
    """Get system health metrics"""
    try:
        # Get monitoring service
        monitoring_service = get_monitoring_service()
        
        # Get metrics summary
        metrics = monitoring_service.get_metrics_summary()
        
        # Check database connection
        db_healthy = True
        db_error = None
        try:
            # Simple query to check DB connection
            result = db.execute("SELECT 1").fetchone()
            if result[0] != 1:
                db_healthy = False
                db_error = "Database query returned unexpected result"
        except Exception as e:
            db_healthy = False
            db_error = str(e)
        
        # Check Redis connection if available
        redis_healthy = True
        redis_error = None
        try:
            from ..dependencies import get_redis_service
            redis_service = await get_redis_service()
            await redis_service.ping()
        except Exception as e:
            redis_healthy = False
            redis_error = str(e)
        
        # Calculate error rate
        total_errors = sum(metrics.get("errors", {}).values())
        
        # Get API request count
        api_request_count = 0
        for endpoint_metrics in metrics.get("api_latency", {}).values():
            api_request_count += endpoint_metrics.get("count", 0)
        
        # Calculate error rate
        error_rate = total_errors / api_request_count if api_request_count > 0 else 0
        
        # Get system info
        import psutil
        import platform
        
        # Get memory usage
        memory = psutil.virtual_memory()
        memory_usage = {
            "total": memory.total,
            "available": memory.available,
            "used": memory.used,
            "percent": memory.percent
        }
        
        # Get CPU usage
        cpu_usage = psutil.cpu_percent(interval=0.1)
        
        # Get disk usage
        disk = psutil.disk_usage('/')
        disk_usage = {
            "total": disk.total,
            "used": disk.used,
            "free": disk.free,
            "percent": disk.percent
        }
        
        # Get process info
        process = psutil.Process(os.getpid())
        process_info = {
            "cpu_percent": process.cpu_percent(interval=0.1),
            "memory_percent": process.memory_percent(),
            "threads": process.num_threads(),
            "create_time": datetime.fromtimestamp(process.create_time()).isoformat()
        }
        
        # Get uptime
        uptime = datetime.now() - datetime.fromtimestamp(process.create_time())
        
        # Construct health response
        health_data = {
            "status": "healthy" if db_healthy and redis_healthy and error_rate < 0.05 else "degraded",
            "timestamp": datetime.now().isoformat(),
            "uptime": str(uptime),
            "version": os.environ.get("APP_VERSION", "unknown"),
            "components": {
                "database": {
                    "status": "healthy" if db_healthy else "unhealthy",
                    "error": db_error
                },
                "redis": {
                    "status": "healthy" if redis_healthy else "unhealthy",
                    "error": redis_error
                }
            },
            "metrics": {
                "error_rate": error_rate,
                "request_count": api_request_count,
                "error_count": total_errors
            },
            "system": {
                "platform": platform.platform(),
                "python_version": platform.python_version(),
                "cpu_usage": cpu_usage,
                "memory_usage": memory_usage,
                "disk_usage": disk_usage,
                "process": process_info
            }
        }
        
        return health_data
    except Exception as e:
        logger.error(f"Error getting health metrics: {str(e)}")
        # Return a minimal health response
        return {
            "status": "degraded",
            "timestamp": datetime.now().isoformat(),
            "error": str(e)
        }