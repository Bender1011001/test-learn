from fastapi import APIRouter, Depends, HTTPException, WebSocket, WebSocketDisconnect, status
from typing import Dict, Any, List, Optional
from ..dependencies import get_services
from ...db.base import get_db
from loguru import logger
import json
import asyncio
import uuid
from datetime import datetime

router = APIRouter()

@router.post("/train", response_model=Dict[str, Any])
async def start_dpo_training(
    request: Dict[str, Any],
    db = Depends(get_db)
):
    """
    Start a new DPO training job.
    
    Args:
        request: Training configuration with agent_type, base_model_id, and adapter_name
    """
    # Get DPO trainer
    config_manager, workflow_manager, db_manager, dpo_trainer = get_services(db)
    
    agent_type = request.get("agent_type")
    base_model_id = request.get("base_model_id")
    adapter_name = request.get("adapter_name")
    training_params = request.get("training_params", {})
    
    # Validate request
    if not all([agent_type, base_model_id, adapter_name]):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="agent_type, base_model_id, and adapter_name are required"
        )
    
    # Start training
    try:
        job_id = dpo_trainer.start_training_job(
            agent_type=agent_type,
            base_model_id=base_model_id,
            adapter_name=adapter_name,
            training_params=training_params
        )
        
        return {
            "job_id": job_id,
            "status": "started",
            "message": f"DPO training job for {agent_type} started"
        }
    except Exception as e:
        logger.error(f"Error starting DPO training: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to start DPO training: {str(e)}"
        )


@router.get("/jobs/{job_id}", response_model=Dict[str, Any])
async def get_dpo_job_status(
    job_id: str,
    db = Depends(get_db)
):
    """
    Get the status of a DPO training job.
    
    Args:
        job_id: ID of the DPO training job
    """
    # Get DPO trainer
    config_manager, workflow_manager, db_manager, dpo_trainer = get_services(db)
    
    # Get job status
    status = dpo_trainer.get_training_job_status(job_id)
    if not status:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"DPO training job {job_id} not found"
        )
    
    return status


@router.post("/jobs/{job_id}/cancel", response_model=Dict[str, Any])
async def cancel_dpo_job(
    job_id: str,
    db = Depends(get_db)
):
    """
    Cancel a DPO training job.
    
    Args:
        job_id: ID of the DPO training job
    """
    # Get DPO trainer
    config_manager, workflow_manager, db_manager, dpo_trainer = get_services(db)
    
    # Cancel job
    try:
        success = dpo_trainer.cancel_training_job(job_id)
        if not success:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"DPO training job {job_id} not found or already completed"
            )
        
        return {
            "job_id": job_id,
            "status": "canceling",
            "message": f"DPO training job {job_id} is being canceled"
        }
    except Exception as e:
        logger.error(f"Error canceling DPO training: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to cancel DPO training: {str(e)}"
        )


@router.get("/annotations/{agent_type}", response_model=Dict[str, Any])
async def get_dpo_ready_annotations(
    agent_type: str,
    db = Depends(get_db)
):
    """
    Get information about DPO-ready annotations for an agent type.
    
    Args:
        agent_type: Type of agent (e.g., "proposer")
    """
    # Get DPO trainer
    config_manager, workflow_manager, db_manager, dpo_trainer = get_services(db)
    
    # Get annotation info
    try:
        # Use db_manager to get DPO-ready annotations asynchronously
        annotations = await db_manager.get_dpo_ready_annotations(agent_type, use_cache=True)
        count = len(annotations)
        samples = annotations[:5] if annotations else []
        
        return {
            "agent_type": agent_type,
            "dpo_ready_count": count,
            "sample_annotations": samples
        }
    except Exception as e:
        logger.error(f"Error getting annotation stats: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get annotation stats: {str(e)}"
        )


@router.websocket("/ws/{job_id}")
async def websocket_dpo_endpoint(
    websocket: WebSocket,
    job_id: str,
    db = Depends(get_db)
):
    """WebSocket endpoint for streaming DPO training status and output using Redis PubSub."""
    await websocket.accept()
    
    # Get DPO trainer and Redis service
    config_manager, workflow_manager, db_manager, dpo_trainer = get_services(db)
    from ..dependencies import get_redis_service
    redis_service = await get_redis_service()
    
    # Check if job exists
    status = dpo_trainer.get_training_job_status(job_id)
    if not status:
        await websocket.send_text(json.dumps({
            "type": "error",
            "message": f"Training job {job_id} not found"
        }))
        await websocket.close()
        return
    
    # Send initial status and output
    await websocket.send_text(json.dumps({
        "type": "status",
        "data": status
    }))
    
    output = dpo_trainer.get_training_job_output(job_id)
    await websocket.send_text(json.dumps({
        "type": "output",
        "data": output
    }))
    
    # Define callback for status updates
    async def on_status_update(data):
        await websocket.send_text(json.dumps({
            "type": "status",
            "data": data
        }))
    
    # Define callback for output updates
    async def on_output_update(data):
        # Check if we received new output
        if "output" in data:
            new_lines = [data["output"]]
            await websocket.send_text(json.dumps({
                "type": "output_append",
                "data": new_lines
            }))
    
    # Subscribe to status and output events
    await redis_service.subscribe(EventChannel.DPO_STATUS, job_id, on_status_update)
    await redis_service.subscribe(EventChannel.DPO_OUTPUT, job_id, on_output_update)
    
    try:
        # Keep the WebSocket connection open until a disconnect occurs
        # or until the job is marked as completed or failed
        while True:
            await asyncio.sleep(1)
            
            # Check if job is still active
            new_status = dpo_trainer.get_training_job_status(job_id)
            if not new_status:
                # Job might have been deleted
                break
                
            # If job is done, stop after giving events a chance to be processed
            if new_status.get("status") in ["completed", "failed"]:
                # Wait a bit to ensure final events are processed
                await asyncio.sleep(2)
                break
    
    except WebSocketDisconnect:
        logger.info(f"Client disconnected from training job {job_id}")
    except Exception as e:
        logger.error(f"Error in training websocket: {str(e)}")
        try:
            await websocket.send_text(json.dumps({
                "type": "error",
                "message": str(e)
            }))
        except:
            pass