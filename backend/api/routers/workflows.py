from fastapi import APIRouter, Depends, HTTPException, WebSocket, WebSocketDisconnect, Query
from fastapi.responses import StreamingResponse
from typing import Dict, List, Optional, Any
from pydantic import BaseModel
import asyncio
from loguru import logger
import json

from ...core.services.workflow_manager import WorkflowManager
from ...core.services.config_manager import ConfigManager
from ...db.base import get_db

# Pydantic models for request/response validation

class InitialGoalRequest(BaseModel):
    """Request model for starting a workflow"""
    workflow_id: str
    initial_goal: str


class WorkflowStatusResponse(BaseModel):
    """Response model for workflow status"""
    run_id: str
    workflow_id: str
    status: str
    start_time: str
    end_time: Optional[str] = None
    initial_goal: str
    current_step: int
    total_steps: int
    error: Optional[str] = None


class WorkflowLogEntry(BaseModel):
    """Model for workflow log entry"""
    id: int
    run_id: str
    timestamp: str
    agent_name: str
    agent_type: str
    input: Dict[str, Any]
    output: Dict[str, Any]


# Create router
router = APIRouter()


# Dependency to get workflow manager
def get_workflow_manager(db=Depends(get_db)):
    from ..dependencies import get_services
    config_manager, workflow_manager, db_manager, dpo_trainer = get_services(db)
    return workflow_manager


# Dependency to get config manager
def get_config_manager(db=Depends(get_db)):
    from ..dependencies import get_services
    config_manager, workflow_manager, db_manager, dpo_trainer = get_services(db)
    return config_manager


@router.post("/start", response_model=Dict[str, str])
async def start_workflow(
    request: InitialGoalRequest,
    workflow_manager: WorkflowManager = Depends(get_workflow_manager)
):
    """Start a new workflow execution"""
    try:
        run_id = await workflow_manager.start_workflow(
            workflow_id=request.workflow_id,
            initial_goal=request.initial_goal
        )
        return {"run_id": run_id}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Error starting workflow: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to start workflow")


@router.get("/status/{run_id}", response_model=Optional[WorkflowStatusResponse])
async def get_workflow_status(
    run_id: str,
    workflow_manager: WorkflowManager = Depends(get_workflow_manager)
):
    """Get status of a workflow execution"""
    status = workflow_manager.get_workflow_status(run_id)
    
    if not status:
        raise HTTPException(status_code=404, detail=f"Workflow run {run_id} not found")
    
    return status


@router.post("/stop/{run_id}", response_model=Dict[str, bool])
async def stop_workflow(
    run_id: str,
    workflow_manager: WorkflowManager = Depends(get_workflow_manager)
):
    """Stop a running workflow execution"""
    success = workflow_manager.stop_workflow(run_id)
    
    if not success:
        raise HTTPException(
            status_code=400, 
            detail=f"Failed to stop workflow run {run_id}"
        )
    
    return {"success": True}


@router.get("/active", response_model=List[WorkflowStatusResponse])
async def get_active_workflows(
    workflow_manager: WorkflowManager = Depends(get_workflow_manager)
):
    """Get all active workflow executions"""
    return workflow_manager.get_active_workflows()


@router.get("/available", response_model=List[Dict[str, Any]])
async def get_available_workflows(
    config_manager: ConfigManager = Depends(get_config_manager)
):
    """Get all available workflow configurations"""
    workflow_ids = config_manager.get_all_workflow_ids()
    
    workflows = []
    for wf_id in workflow_ids:
        workflow = config_manager.get_workflow_config(wf_id)
        if workflow:
            workflows.append({
                "id": wf_id,
                "description": workflow.description,
                "agent_sequence": workflow.agent_sequence
            })
    
    return workflows


@router.get("/logs/{run_id}", response_model=List[WorkflowLogEntry])
async def get_workflow_logs(
    run_id: str,
    workflow_manager: WorkflowManager = Depends(get_workflow_manager)
):
    """Get logs for a specific workflow execution"""
    # This would normally query the database
    # For this implementation, we'll use the stream_workflow_logs generator
    logs = []
    async for log in workflow_manager.stream_workflow_logs(run_id):
        logs.append(log)
    
    return logs


@router.get("/stream-logs/{run_id}")
async def stream_workflow_logs(
    run_id: str,
    workflow_manager: WorkflowManager = Depends(get_workflow_manager)
):
    """Stream logs for a specific workflow execution as server-sent events"""
    
    async def event_generator():
        try:
            async for log in workflow_manager.stream_workflow_logs(run_id):
                # Format as SSE
                yield f"data: {json.dumps(log)}\n\n"
        except Exception as e:
            logger.error(f"Error streaming logs: {str(e)}")
            yield f"data: {json.dumps({'error': str(e)})}\n\n"
    
    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream"
    )