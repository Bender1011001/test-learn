from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Depends, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse
import uvicorn
import os
from pathlib import Path
from dotenv import load_dotenv
from loguru import logger
import json
import asyncio

# Load environment variables
load_dotenv()

# Import dependencies and settings
from .dependencies import get_settings, get_services
from ..db.base import Base, engine, get_db
from ..core.services.workflow_manager import WorkflowManager

# Import routers
from .routers import workflows, configs, logs, dpo_training

# Create database tables
Base.metadata.create_all(bind=engine)

# Get application settings
settings = get_settings()

# Create FastAPI app
app = FastAPI(
    title=settings["app_name"],
    description="API for CAMEL Extensions GUI",
    version=settings["version"],
    docs_url=settings["docs_url"],
    openapi_url=settings["openapi_url"],
    redoc_url=settings["redoc_url"],
)

# Configure CORS
# Get allowed origins from environment or use default for development
allowed_origins = os.getenv("ALLOWED_ORIGINS", "http://localhost:8501,http://localhost").split(",")
if settings["debug"]:
    # In debug mode, allow all origins with a warning
    logger.warning("Running in debug mode: CORS is configured to allow all origins")
    allowed_origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["Authorization", "Content-Type", "X-Trace-ID"],
)

# WebSocket connections store
active_connections: dict[str, WebSocket] = {}

# Include routers with prefix
api_prefix = settings["api_prefix"]
app.include_router(workflows.router, prefix=f"{api_prefix}/workflows", tags=["workflows"])
app.include_router(configs.router, prefix=f"{api_prefix}/configs", tags=["configs"])
app.include_router(logs.router, prefix=f"{api_prefix}/logs", tags=["logs"])
app.include_router(dpo_training.router, prefix=f"{api_prefix}/dpo", tags=["dpo"])


@app.get("/", include_in_schema=False)
async def redirect_to_docs():
    """Redirect root to API docs"""
    return RedirectResponse(url=settings["docs_url"])


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    # Simple health check
    return {"status": "healthy", "version": app.version}


@app.websocket("/ws/workflow/{workflow_run_id}")
async def workflow_websocket(websocket: WebSocket, workflow_run_id: str, db=Depends(get_db)):
    """WebSocket endpoint for streaming workflow execution data"""
    await websocket.accept()
    active_connections[workflow_run_id] = websocket
    
    # Get workflow manager
    config_manager, workflow_manager, db_manager, dpo_trainer = get_services(db)
    
    # Check if workflow exists
    status = workflow_manager.get_workflow_status(workflow_run_id)
    if not status:
        await websocket.send_text(json.dumps({
            "type": "error",
            "message": f"Workflow {workflow_run_id} not found"
        }))
        await websocket.close()
        return
    
    # Send initial status
    await websocket.send_text(json.dumps({
        "type": "status",
        "data": status
    }))
    
    try:
        # Stream logs
        async for log in workflow_manager.stream_workflow_logs(workflow_run_id):
            await websocket.send_text(json.dumps({
                "type": "log",
                "data": log
            }))
            
            # Check if workflow status has changed
            new_status = workflow_manager.get_workflow_status(workflow_run_id)
            if not new_status:
                # Workflow might have been deleted
                break
            
            if new_status.get("status") != status.get("status"):
                status = new_status
                await websocket.send_text(json.dumps({
                    "type": "status",
                    "data": status
                }))
    
    except WebSocketDisconnect:
        if workflow_run_id in active_connections:
            del active_connections[workflow_run_id]
            logger.info(f"Client disconnected from workflow {workflow_run_id}")
    except Exception as e:
        logger.error(f"Error in workflow websocket: {str(e)}")
        try:
            await websocket.send_text(json.dumps({
                "type": "error",
                "message": str(e)
            }))
        except:
            pass


@app.on_event("startup")
async def startup_event():
    """Startup event handler"""
    logger.info("Starting CAMEL Extensions API")
    
    # Initialize service singletons
    db = next(get_db())
    try:
        config_manager, workflow_manager, db_manager, dpo_trainer = get_services(db)
        logger.info("Services initialized")
    finally:
        db.close()


@app.on_event("shutdown")
async def shutdown_event():
    """Shutdown event handler"""
    logger.info("Shutting down CAMEL Extensions API")
    
    # Close all WebSocket connections
    for workflow_id, connection in list(active_connections.items()):
        try:
            await connection.close()
            logger.debug(f"Closed WebSocket connection for workflow {workflow_id}")
        except:
            pass
    
    active_connections.clear()


if __name__ == "__main__":
    # Get port from environment or use default
    port = int(os.getenv("API_PORT", 8000))
    
    # Start the server using Uvicorn
    uvicorn.run(
        "backend.api.main:app",
        host="0.0.0.0",
        port=port,
        reload=settings["debug"]
    )