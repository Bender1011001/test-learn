"""
Health checking endpoint for the CAMEL Extensions GUI.

This module provides a simple FastAPI server that runs alongside Streamlit
and provides a health check endpoint for monitoring and orchestration.
"""
from fastapi import FastAPI, Response
from pydantic import BaseModel
import os
import sys
import psutil
import datetime

app = FastAPI(title="CAMEL GUI Health API")


class HealthStatus(BaseModel):
    """Model for health check response."""
    status: str
    uptime: str
    memory_usage: dict
    timestamp: str


@app.get("/healthz", response_model=HealthStatus)
def health_check():
    """
    Health check endpoint.
    
    Returns:
        HealthStatus object with system information.
    """
    # Get process info
    process = psutil.Process(os.getpid())
    
    # Calculate memory usage
    memory_info = process.memory_info()
    memory_usage = {
        "rss_mb": memory_info.rss / (1024 * 1024),  # RSS in MB
        "vms_mb": memory_info.vms / (1024 * 1024),  # VMS in MB
        "percent": process.memory_percent()
    }
    
    # Calculate uptime
    start_time = datetime.datetime.fromtimestamp(process.create_time())
    uptime = datetime.datetime.now() - start_time
    uptime_str = str(uptime).split(".")[0]  # Remove microseconds
    
    return HealthStatus(
        status="ok",
        uptime=uptime_str,
        memory_usage=memory_usage,
        timestamp=datetime.datetime.now().isoformat()
    )


@app.get("/")
def root():
    """Redirect root to health check endpoint."""
    return Response(
        status_code=302,
        headers={"Location": "/healthz"}
    )


if __name__ == "__main__":
    # This file is not meant to be run directly
    print("This health endpoint is meant to be run alongside Streamlit.")
    print("Use the provided Dockerfile or docker-compose.yml to run the application.")
    sys.exit(1)