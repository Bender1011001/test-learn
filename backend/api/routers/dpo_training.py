from fastapi import APIRouter, Depends, HTTPException, WebSocket, WebSocketDisconnect
from typing import Dict, List, Optional, Any
from pydantic import BaseModel
import asyncio
from loguru import logger
import json

from ...core.services.dpo_trainer import DPOTrainer
from ...core.services.db_manager import DBManager
from ...db.base import get_db

# Pydantic models for request/response validation

class StartTrainingRequest(BaseModel):
    """Request model for starting a DPO training job"""
    agent_type: str
    base_model_id: str
    adapter_name: str
    training_args: Optional[Dict[str, Any]] = None


class TrainingJobStatusResponse(BaseModel):
    """Response model for training job status"""
    job_id: str
    status: str
    start_time: str
    end_time: Optional[str] = None
    agent_type: str
    base_model_id: str
    adapter_name: str
    data_samples: int
    progress: float
    adapter_id: Optional[str] = None
    error: Optional[str] = None
    error_code: Optional[int] = None


class TrainingSummaryResponse(BaseModel):
    """Response model for training summary statistics"""
    total_jobs: int
    active_jobs: int
    completed_jobs: int
    failed_jobs: int


# Create router
router = APIRouter()


# Dependency to get DPO trainer
def get_dpo_trainer(db=Depends(get_db)):
    from ..dependencies import get_services
    config_manager, workflow_manager, db_manager, dpo_trainer = get_services(db)
    return dpo_trainer


# Dependency to get DB manager
def get_db_manager(db=Depends(get_db)):
    from ..dependencies import get_services
    config_manager, workflow_manager, db_manager, dpo_trainer = get_services(db)
    return db_manager


@router.post("/start", response_model=Dict[str, str])
async def start_training_job(
    request: StartTrainingRequest,
    dpo_trainer: DPOTrainer = Depends(get_dpo_trainer),
    db_manager: DBManager = Depends(get_db_manager)
):
    """Start a new DPO training job"""
    try:
        # Check if we have enough data
        annotations = db_manager.get_dpo_ready_annotations(request.agent_type)
        if not annotations:
            raise HTTPException(
                status_code=400,
                detail=f"No DPO-ready annotations found for agent type {request.agent_type}"
            )
        
        job_id = await dpo_trainer.start_training_job(
            agent_type=request.agent_type,
            base_model_id=request.base_model_id,
            adapter_name=request.adapter_name,
            training_args=request.training_args
        )
        
        return {"job_id": job_id}
    
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Error starting training job: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to start training job: {str(e)}")


@router.get("/status/{job_id}", response_model=Optional[TrainingJobStatusResponse])
async def get_training_job_status(
    job_id: str,
    dpo_trainer: DPOTrainer = Depends(get_dpo_trainer)
):
    """Get the status of a DPO training job"""
    status = dpo_trainer.get_training_job_status(job_id)
    
    if not status:
        raise HTTPException(status_code=404, detail=f"Training job {job_id} not found")
    
    return status


@router.get("/output/{job_id}")
async def get_training_job_output(
    job_id: str,
    max_lines: int = 100,
    dpo_trainer: DPOTrainer = Depends(get_dpo_trainer)
):
    """Get the output logs of a DPO training job"""
    status = dpo_trainer.get_training_job_status(job_id)
    
    if not status:
        raise HTTPException(status_code=404, detail=f"Training job {job_id} not found")
    
    output = dpo_trainer.get_training_job_output(job_id, max_lines)
    
    return {"output": output}


@router.post("/cancel/{job_id}", response_model=Dict[str, bool])
async def cancel_training_job(
    job_id: str,
    dpo_trainer: DPOTrainer = Depends(get_dpo_trainer)
):
    """Cancel a running DPO training job"""
    success = dpo_trainer.cancel_training_job(job_id)
    
    if not success:
        raise HTTPException(
            status_code=400,
            detail=f"Failed to cancel training job {job_id}"
        )
    
    return {"success": True}


@router.get("/active", response_model=List[TrainingJobStatusResponse])
async def get_active_training_jobs(
    dpo_trainer: DPOTrainer = Depends(get_dpo_trainer)
):
    """Get all active DPO training jobs"""
    return dpo_trainer.get_active_training_jobs()


@router.get("/summary", response_model=TrainingSummaryResponse)
async def get_training_summary(
    dpo_trainer: DPOTrainer = Depends(get_dpo_trainer)
):
    """Get summary statistics about DPO training jobs"""
    return dpo_trainer.get_training_summary_stats()


@router.websocket("/ws/{job_id}")
async def training_websocket(
    websocket: WebSocket,
    job_id: str,
    dpo_trainer: DPOTrainer = Depends(get_dpo_trainer)
):
    """WebSocket endpoint for streaming training job output"""
    await websocket.accept()
    
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
    
    # Simple polling mechanism for updates
    # In a real implementation, this would use a proper event-driven approach
    try:
        while True:
            # Check for new status every second
            await asyncio.sleep(1)
            
            new_status = dpo_trainer.get_training_job_status(job_id)
            if not new_status:
                # Job might have been deleted
                break
            
            # Check if status has changed
            if new_status.get("status") != status.get("status") or new_status.get("progress") != status.get("progress"):
                status = new_status
                await websocket.send_text(json.dumps({
                    "type": "status",
                    "data": status
                }))
            
            # Check for new output
            new_output = dpo_trainer.get_training_job_output(job_id, max_lines=1)
            if new_output and (not output or new_output[-1] != output[-1]):
                # Send only the new lines
                if output and len(output) > 0:
                    output_set = set(output)
                    new_lines = [line for line in new_output if line not in output_set]
                else:
                    new_lines = new_output
                
                if new_lines:
                    await websocket.send_text(json.dumps({
                        "type": "output_append",
                        "data": new_lines
                    }))
                    output = new_output
            
            # If job is done, stop polling
            if status.get("status") in ["completed", "failed"]:
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