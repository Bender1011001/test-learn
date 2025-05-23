from fastapi import APIRouter, Depends, HTTPException, Query
from typing import Dict, List, Optional, Any
from pydantic import BaseModel
from datetime import datetime
from loguru import logger

from ...core.services.db_manager import DBManager
from ...db.base import get_db

# Pydantic models for request/response validation

class LogEntry(BaseModel):
    """Model for log entry response"""
    id: int
    workflow_run_id: str
    timestamp: str
    agent_name: str
    agent_type: str
    input_data: Dict[str, Any]
    output_data: Dict[str, Any]
    has_annotation: bool
    annotation_count: Optional[int] = None


class LogSummary(BaseModel):
    """Model for log summary response"""
    total_logs: int
    logs_with_annotations: int
    logs_by_agent: Dict[str, int]
    recent_logs: int


class AnnotationResponse(BaseModel):
    """Model for annotation response"""
    id: int
    log_entry_id: int
    rating: float
    rationale: Optional[str] = None
    chosen_prompt: str
    rejected_prompt: str
    dpo_context: str
    user_id: Optional[str] = None
    timestamp: str


class AnnotationRequest(BaseModel):
    """Model for annotation request"""
    log_entry_id: int
    rating: float
    rationale: Optional[str] = None
    chosen_prompt: str
    rejected_prompt: str
    dpo_context: str
    user_id: Optional[str] = None


# Create router
router = APIRouter()


# Dependency to get db manager
def get_db_manager(db=Depends(get_db)):
    from ..dependencies import get_services
    config_manager, workflow_manager, db_manager, dpo_trainer = get_services(db)
    return db_manager


@router.get("", response_model=List[LogEntry])
async def get_logs(
    workflow_id: Optional[str] = None,
    agent_name: Optional[str] = None,
    agent_type: Optional[str] = None,
    start_date: Optional[datetime] = None,
    end_date: Optional[datetime] = None,
    has_annotation: Optional[bool] = None,
    keyword: Optional[str] = None,
    offset: int = Query(0, ge=0),
    limit: int = Query(100, ge=1, le=500),
    sort_by: str = "timestamp",
    sort_desc: bool = True,
    db_manager: DBManager = Depends(get_db_manager)
):
    """
    Get logs with various filters
    
    All filter parameters are optional
    """
    try:
        logs = await db_manager.get_logs_async(
            workflow_id=workflow_id,
            agent_name=agent_name,
            agent_type=agent_type,
            start_date=start_date,
            end_date=end_date,
            has_annotation=has_annotation,
            keyword=keyword,
            offset=offset,
            limit=limit,
            sort_by=sort_by,
            sort_desc=sort_desc
        )
        return logs
    except Exception as e:
        logger.error(f"Error getting logs: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error retrieving logs: {str(e)}")


@router.get("/summary", response_model=LogSummary)
async def get_logs_summary(
    db_manager: DBManager = Depends(get_db_manager)
):
    """Get summary statistics about logs"""
    try:
        summary = await db_manager.get_logs_summary_async()
        return summary
    except Exception as e:
        logger.error(f"Error getting log summary: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error retrieving log summary: {str(e)}")


@router.get("/{log_id}", response_model=LogEntry)
async def get_log_by_id(
    log_id: int,
    db_manager: DBManager = Depends(get_db_manager)
):
    """Get a specific log entry by ID"""
    log = await db_manager.get_log_by_id_async(log_id)
    
    if not log:
        raise HTTPException(status_code=404, detail=f"Log entry {log_id} not found")
    
    return log


@router.get("/{log_id}/annotation", response_model=Optional[AnnotationResponse])
async def get_annotation(
    log_id: int,
    db_manager: DBManager = Depends(get_db_manager)
):
    """Get annotation for a specific log entry"""
    annotation = await db_manager.get_annotation_async(log_id)
    return annotation  # Can be None


@router.post("/annotations", response_model=Dict[str, Any])
async def save_annotation(
    request: AnnotationRequest,
    db_manager: DBManager = Depends(get_db_manager)
):
    """Save or update an annotation for a log entry"""
    # Convert request model to dict
    annotation_data = request.model_dump()
    
    # Use async method for better performance
    annotation_id = await db_manager.save_annotation_async(annotation_data)
    
    if not annotation_id:
        raise HTTPException(
            status_code=400,
            detail=f"Failed to save annotation for log entry {request.log_entry_id}"
        )
    
    return {"id": annotation_id, "success": True}


class BatchAnnotationRequest(BaseModel):
    """Model for batch annotation request"""
    annotations: List[AnnotationRequest]


@router.post("/annotations/batch", response_model=Dict[str, Any])
async def batch_save_annotations(
    request: BatchAnnotationRequest,
    db_manager: DBManager = Depends(get_db_manager)
):
    """Save multiple annotations in a single transaction"""
    # Convert request models to dicts
    annotations_data = [ann.model_dump() for ann in request.annotations]
    
    # Use async batch method for better performance
    result_ids = await db_manager.batch_save_annotations_async(annotations_data)
    
    # Count successful saves
    success_count = sum(1 for id in result_ids if id is not None)
    
    return {
        "success": success_count > 0,
        "total": len(annotations_data),
        "successful": success_count,
        "failed": len(annotations_data) - success_count,
        "ids": [id for id in result_ids if id is not None]
    }


@router.delete("/annotations/{annotation_id}", response_model=Dict[str, bool])
async def delete_annotation(
    annotation_id: int,
    db_manager: DBManager = Depends(get_db_manager)
):
    """Delete an annotation"""
    # We don't have an async version of delete_annotation yet, so we'll use the sync version
    # In a real implementation, we would add an async version to DBManager
    success = db_manager.delete_annotation(annotation_id)
    
    if not success:
        raise HTTPException(
            status_code=404,
            detail=f"Annotation {annotation_id} not found or could not be deleted"
        )
    
    return {"success": True}


@router.delete("/annotations/batch", response_model=Dict[str, Any])
async def batch_delete_annotations(
    annotation_ids: List[int],
    db_manager: DBManager = Depends(get_db_manager)
):
    """Delete multiple annotations in a single request"""
    results = []
    
    # Process each deletion individually
    # In a real implementation, we would add a batch_delete_annotations method to DBManager
    for annotation_id in annotation_ids:
        success = db_manager.delete_annotation(annotation_id)
        results.append(success)
    
    # Count successful deletions
    success_count = sum(1 for result in results if result)
    
    return {
        "success": success_count > 0,
        "total": len(annotation_ids),
        "successful": success_count,
        "failed": len(annotation_ids) - success_count
    }


@router.get("/annotations/dpo", response_model=List[Dict[str, Any]])
async def get_dpo_annotations(
    agent_type: str,
    db_manager: DBManager = Depends(get_db_manager)
):
    """Get all annotations ready for DPO training for a specific agent type"""
    annotations = await db_manager.get_dpo_ready_annotations(agent_type, use_cache=True)
    return annotations


@router.get("/annotations/dpo/export")
async def export_dpo_data(
    agent_type: str,
    format: str = "jsonl",
    db_manager: DBManager = Depends(get_db_manager)
):
    """
    Export DPO training data for a specific agent type
    
    Returns a text file with the specified format (currently only 'jsonl' is supported)
    """
    try:
        data = await db_manager.export_dpo_data(agent_type, format, use_cache=True)
        
        # Create a response with the appropriate content type
        from fastapi.responses import Response
        return Response(
            content=data,
            media_type="application/json-lines",
            headers={"Content-Disposition": f"attachment; filename=dpo_data_{agent_type}.jsonl"}
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Error exporting DPO data: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error exporting DPO data: {str(e)}")