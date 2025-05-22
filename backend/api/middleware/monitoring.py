from fastapi import Request, Response
from fastapi.routing import APIRoute
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.types import ASGIApp
import time
from typing import Callable, Dict, Any
import json
from loguru import logger
import traceback
import uuid
import os

from ...core.services.monitoring_service import get_monitoring_service


class MonitoringMiddleware(BaseHTTPMiddleware):
    """Middleware for monitoring API requests and responses"""
    
    def __init__(
        self, 
        app: ASGIApp, 
        exclude_paths: list = None,
        log_request_body: bool = False,
        log_response_body: bool = False,
        log_headers: bool = False
    ):
        super().__init__(app)
        self.exclude_paths = exclude_paths or ["/health", "/metrics", "/favicon.ico"]
        self.log_request_body = log_request_body
        self.log_response_body = log_response_body
        self.log_headers = log_headers
        self.monitoring_service = get_monitoring_service()
        
        # Get environment settings
        self.debug_mode = os.environ.get("DEBUG", "false").lower() == "true"
        
        logger.info("Monitoring middleware initialized")
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        # Skip monitoring for excluded paths
        if any(request.url.path.startswith(path) for path in self.exclude_paths):
            return await call_next(request)
        
        # Generate request ID for tracking
        request_id = str(uuid.uuid4())
        request.state.request_id = request_id
        
        # Start timing
        start_time = time.time()
        
        # Extract request details
        method = request.method
        path = request.url.path
        client_host = request.client.host if request.client else "unknown"
        
        # Log request
        log_data = {
            "request_id": request_id,
            "method": method,
            "path": path,
            "client_ip": client_host,
            "query_params": dict(request.query_params),
        }
        
        if self.log_headers:
            log_data["headers"] = dict(request.headers)
        
        if self.log_request_body and request.method in ["POST", "PUT", "PATCH"]:
            try:
                # Clone the request body
                body = await request.body()
                # Create a new request with the same body for call_next
                request = Request(request.scope, request.receive)
                
                # Try to parse as JSON
                try:
                    body_str = body.decode("utf-8")
                    if body_str:
                        try:
                            body_json = json.loads(body_str)
                            log_data["body"] = body_json
                        except json.JSONDecodeError:
                            # If not JSON, log as string (truncated if too long)
                            if len(body_str) > 1000:
                                log_data["body"] = body_str[:1000] + "... [truncated]"
                            else:
                                log_data["body"] = body_str
                except UnicodeDecodeError:
                    log_data["body"] = "[binary data]"
            except Exception as e:
                log_data["body_error"] = str(e)
        
        logger.info(f"API Request: {json.dumps(log_data)}")
        
        # Process the request and catch any exceptions
        status_code = 500
        error_detail = None
        try:
            response = await call_next(request)
            status_code = response.status_code
            
            # Log response
            response_log = {
                "request_id": request_id,
                "status_code": status_code,
                "duration_ms": round((time.time() - start_time) * 1000, 2)
            }
            
            if self.log_response_body and status_code != 204:  # Don't try to read body for 204 No Content
                try:
                    # Read response body
                    response_body = b""
                    async for chunk in response.body_iterator:
                        response_body += chunk
                    
                    # Create a new response with the same body
                    response = Response(
                        content=response_body,
                        status_code=response.status_code,
                        headers=dict(response.headers),
                        media_type=response.media_type
                    )
                    
                    # Try to parse as JSON
                    try:
                        body_str = response_body.decode("utf-8")
                        if body_str:
                            try:
                                body_json = json.loads(body_str)
                                # Truncate large responses
                                if self.debug_mode:
                                    response_log["body"] = body_json
                                else:
                                    response_log["body"] = "[response body logged only in debug mode]"
                            except json.JSONDecodeError:
                                # If not JSON, log as string (truncated if too long)
                                if len(body_str) > 1000:
                                    response_log["body"] = body_str[:1000] + "... [truncated]"
                                else:
                                    response_log["body"] = body_str
                    except UnicodeDecodeError:
                        response_log["body"] = "[binary data]"
                except Exception as e:
                    response_log["body_error"] = str(e)
            
            # Log at appropriate level based on status code
            if status_code >= 500:
                logger.error(f"API Response: {json.dumps(response_log)}")
            elif status_code >= 400:
                logger.warning(f"API Response: {json.dumps(response_log)}")
            else:
                logger.info(f"API Response: {json.dumps(response_log)}")
            
        except Exception as e:
            # Log the exception
            end_time = time.time()
            duration_ms = round((end_time - start_time) * 1000, 2)
            
            error_detail = str(e)
            logger.error(
                f"API Error: {json.dumps({
                    'request_id': request_id,
                    'method': method,
                    'path': path,
                    'error': error_detail,
                    'traceback': traceback.format_exc(),
                    'duration_ms': duration_ms
                })}"
            )
            
            # Record error in monitoring service
            self.monitoring_service.record_error(
                error_type=type(e).__name__,
                source=f"{method}:{path}"
            )
            
            # Re-raise the exception
            raise
        finally:
            # Record metrics
            end_time = time.time()
            duration = end_time - start_time
            
            # Record API latency
            self.monitoring_service.record_api_latency(
                endpoint=path,
                method=method,
                latency=duration
            )
            
            # Record error if status code indicates an error
            if status_code >= 400:
                error_type = f"HTTP{status_code}"
                self.monitoring_service.record_error(
                    error_type=error_type,
                    source=f"{method}:{path}"
                )
        
        # Add custom headers to response
        response.headers["X-Request-ID"] = request_id
        response.headers["X-Response-Time"] = str(round(duration * 1000, 2))
        
        return response


class TimedRoute(APIRoute):
    """Custom API route that times requests"""
    
    def get_route_handler(self) -> Callable:
        original_route_handler = super().get_route_handler()
        monitoring_service = get_monitoring_service()
        
        async def timed_route_handler(request: Request) -> Response:
            # Get endpoint path for metrics
            endpoint = self.path_format
            method = request.method
            
            # Use the monitoring service's async context manager
            async with monitoring_service.measure_time_async("api", endpoint=endpoint, method=method):
                return await original_route_handler(request)
        
        return timed_route_handler