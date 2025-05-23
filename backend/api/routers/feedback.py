"""
API endpoints for bidirectional feedback management.

This module provides REST API endpoints for managing the bidirectional feedback
system, including collecting feedback, retrieving performance summaries, and
exporting data for training.
"""

from typing import Dict, List, Optional, Any
from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks
from pydantic import BaseModel, Field
from datetime import datetime
import asyncio

from ..dependencies import get_workflow_manager, get_db
from ...core.services.workflow_manager import WorkflowManager
from ...agents.bidirectional_feedback import FeedbackType, AgentRole
from sqlalchemy.orm import Session

router = APIRouter(prefix="/feedback", tags=["feedback"])


# Pydantic models for request/response
class FeedbackCollectionRequest(BaseModel):
    """Request model for collecting feedback"""
    interaction_context: Dict[str, Any] = Field(..., description="Context of the interaction to get feedback on")
    agents_involved: Optional[List[str]] = Field(default=None, description="List of agent names (defaults to all)")
    feedback_types: Optional[List[str]] = Field(default=None, description="Specific feedback types to collect")


class LearningLoopRequest(BaseModel):
    """Request model for starting autonomous learning loop"""
    generation_rate: Optional[int] = Field(default=3, description="Tasks per minute to generate")
    max_concurrent_tasks: Optional[int] = Field(default=5, description="Maximum concurrent task executions")
    feedback_frequency: Optional[str] = Field(default="after_each_task", description="When to collect feedback")
    auto_dpo_training: Optional[bool] = Field(default=True, description="Enable automatic DPO training")
    performance_threshold: Optional[float] = Field(default=7.0, description="Minimum performance to continue")
    max_iterations: Optional[int] = Field(default=100, description="Maximum learning iterations")
    continuous_operation: Optional[bool] = Field(default=True, description="Run continuously")


class FeedbackSummaryResponse(BaseModel):
    """Response model for feedback summary"""
    agent_name: Optional[str]
    overall_performance: Optional[Dict[str, Any]]
    performance_by_type: Optional[Dict[str, float]]
    feedback_activity: Optional[Dict[str, Any]]
    recent_feedback: Optional[List[Dict[str, Any]]]
    last_updated: Optional[str]


class SystemInsightsResponse(BaseModel):
    """Response model for system-wide feedback insights"""
    system_overview: Dict[str, Any]
    agent_pair_performance: Dict[str, Dict[str, Any]]
    common_strengths: List[List[Any]]
    common_improvements: List[List[Any]]
    agent_rankings: List[Dict[str, Any]]


@router.post("/collect", response_model=Dict[str, str])
async def collect_bidirectional_feedback(
    request: FeedbackCollectionRequest,
    background_tasks: BackgroundTasks,
    workflow_manager: WorkflowManager = Depends(get_workflow_manager)
):
    """
    Start bidirectional feedback collection for a specific interaction.
    
    This endpoint initiates a workflow that collects feedback from all agents
    about each other's performance in a given interaction context.
    """
    try:
        run_id = await workflow_manager.start_bidirectional_feedback_workflow(
            interaction_context=request.interaction_context,
            agents_involved=request.agents_involved
        )
        
        return {
            "run_id": run_id,
            "status": "started",
            "message": "Bidirectional feedback collection started successfully"
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to start feedback collection: {str(e)}")


@router.post("/learning-loop", response_model=Dict[str, str])
async def start_autonomous_learning_loop(
    request: LearningLoopRequest,
    workflow_manager: WorkflowManager = Depends(get_workflow_manager)
):
    """
    Start the complete autonomous learning loop.
    
    This endpoint starts a continuous process that:
    1. Generates tasks autonomously
    2. Executes tasks with agents
    3. Collects bidirectional feedback
    4. Triggers training when performance drops
    """
    try:
        loop_settings = request.dict(exclude_none=True)
        
        run_id = await workflow_manager.start_autonomous_learning_loop(
            loop_settings=loop_settings
        )
        
        return {
            "run_id": run_id,
            "status": "started",
            "message": "Autonomous learning loop started successfully"
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to start learning loop: {str(e)}")


@router.get("/summary/{agent_name}", response_model=FeedbackSummaryResponse)
async def get_agent_feedback_summary(
    agent_name: str,
    workflow_manager: WorkflowManager = Depends(get_workflow_manager)
):
    """
    Get feedback summary for a specific agent.
    
    Returns performance metrics, recent feedback, and trends for the specified agent.
    """
    try:
        summary = workflow_manager.get_feedback_summary(agent_name=agent_name)
        
        if not summary:
            raise HTTPException(status_code=404, detail=f"No feedback data found for agent: {agent_name}")
        
        return FeedbackSummaryResponse(**summary)
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get feedback summary: {str(e)}")


@router.get("/summary", response_model=SystemInsightsResponse)
async def get_system_feedback_insights(
    workflow_manager: WorkflowManager = Depends(get_workflow_manager)
):
    """
    Get system-wide feedback insights.
    
    Returns overall system performance, agent rankings, common patterns,
    and cross-agent performance metrics.
    """
    try:
        insights = workflow_manager.get_feedback_summary()
        
        return SystemInsightsResponse(**insights)
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get system insights: {str(e)}")


@router.get("/export/dpo", response_model=List[Dict[str, Any]])
async def export_feedback_for_dpo_training(
    min_rating: Optional[float] = 6.0,
    workflow_manager: WorkflowManager = Depends(get_workflow_manager)
):
    """
    Export feedback data in format suitable for DPO training.
    
    Returns structured feedback data that can be used directly for
    Direct Preference Optimization training of the agents.
    """
    try:
        dpo_data = workflow_manager.export_feedback_for_training(min_rating=min_rating)
        
        return dpo_data
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to export DPO data: {str(e)}")


@router.get("/workflow/{run_id}/status")
async def get_feedback_workflow_status(
    run_id: str,
    workflow_manager: WorkflowManager = Depends(get_workflow_manager)
):
    """
    Get the status of a feedback collection workflow.
    
    Returns current status, progress, and results of the feedback workflow.
    """
    try:
        status = await workflow_manager.get_workflow_status_async(run_id)
        
        if not status:
            raise HTTPException(status_code=404, detail=f"Workflow not found: {run_id}")
        
        return status
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get workflow status: {str(e)}")


@router.post("/workflow/{run_id}/stop")
async def stop_feedback_workflow(
    run_id: str,
    workflow_manager: WorkflowManager = Depends(get_workflow_manager)
):
    """
    Stop a running feedback workflow.
    
    Gracefully stops the specified feedback collection or learning loop workflow.
    """
    try:
        success = await workflow_manager.stop_workflow_async(run_id)
        
        if not success:
            raise HTTPException(status_code=404, detail=f"Workflow not found or not running: {run_id}")
        
        return {
            "run_id": run_id,
            "status": "stopping",
            "message": "Workflow stop initiated successfully"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to stop workflow: {str(e)}")


@router.get("/agents/{agent_name}/performance-trend")
async def get_agent_performance_trend(
    agent_name: str,
    days: Optional[int] = 7,
    workflow_manager: WorkflowManager = Depends(get_workflow_manager),
    db: Session = Depends(get_db)
):
    """
    Get performance trend for a specific agent over time.
    
    Returns historical performance data showing how the agent's
    performance has changed over the specified time period.
    """
    try:
        from ...db.models.logs import InteractionLog
        from datetime import datetime, timedelta
        
        # Calculate date range
        end_date = datetime.utcnow()
        start_date = end_date - timedelta(days=days)
        
        # Query feedback logs for the agent
        feedback_logs = db.query(InteractionLog).filter(
            InteractionLog.agent_name == agent_name,
            InteractionLog.agent_type == "feedback_provider",
            InteractionLog.timestamp >= start_date,
            InteractionLog.timestamp <= end_date
        ).order_by(InteractionLog.timestamp).all()
        
        # Process the logs to extract performance trends
        trend_data = []
        for log in feedback_logs:
            if "overall_rating" in log.output_data:
                trend_data.append({
                    "timestamp": log.timestamp.isoformat(),
                    "rating": log.output_data["overall_rating"],
                    "confidence": log.output_data.get("confidence_score", 0.8),
                    "evaluated_agent": log.input_data.get("evaluated_agent"),
                    "feedback_type": log.input_data.get("feedback_type")
                })
        
        # Calculate trend statistics
        if trend_data:
            ratings = [item["rating"] for item in trend_data]
            avg_rating = sum(ratings) / len(ratings)
            trend_direction = "improving" if len(ratings) > 1 and ratings[-1] > ratings[0] else "stable"
            if len(ratings) > 1 and ratings[-1] < ratings[0]:
                trend_direction = "declining"
        else:
            avg_rating = 0.0
            trend_direction = "no_data"
        
        return {
            "agent_name": agent_name,
            "period_days": days,
            "data_points": len(trend_data),
            "average_rating": round(avg_rating, 2),
            "trend_direction": trend_direction,
            "trend_data": trend_data[-50:]  # Return last 50 data points
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get performance trend: {str(e)}")


@router.post("/simulate-interaction")
async def simulate_agent_interaction(
    task_description: str,
    workflow_manager: WorkflowManager = Depends(get_workflow_manager)
):
    """
    Simulate a complete agent interaction and collect feedback.
    
    This endpoint creates a simulated interaction between all agents
    and then collects bidirectional feedback on their performance.
    Useful for testing and demonstration purposes.
    """
    try:
        # Start a regular workflow to simulate the interaction
        workflow_run_id = await workflow_manager.start_workflow(
            workflow_id="default_workflow",
            initial_goal=task_description
        )
        
        # Wait for the workflow to complete (with timeout)
        max_wait = 300  # 5 minutes
        start_time = datetime.now()
        
        while (datetime.now() - start_time).seconds < max_wait:
            status = await workflow_manager.get_workflow_status_async(workflow_run_id)
            if status and status["status"] not in ["running", "starting"]:
                break
            await asyncio.sleep(5)
        
        # Get the final status
        final_status = await workflow_manager.get_workflow_status_async(workflow_run_id)
        
        if final_status and final_status["status"] == "completed":
            # Start feedback collection
            interaction_context = {
                "workflow_run_id": workflow_run_id,
                "task_description": task_description,
                "simulation": True,
                "timestamp": datetime.utcnow().isoformat()
            }
            
            feedback_run_id = await workflow_manager.start_bidirectional_feedback_workflow(
                interaction_context=interaction_context,
                agents_involved=["Proposer", "Executor", "PeerReviewer"]
            )
            
            return {
                "workflow_run_id": workflow_run_id,
                "feedback_run_id": feedback_run_id,
                "status": "completed",
                "message": "Interaction simulated and feedback collection started"
            }
        else:
            return {
                "workflow_run_id": workflow_run_id,
                "status": "failed",
                "message": "Workflow did not complete successfully",
                "final_status": final_status
            }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to simulate interaction: {str(e)}")


@router.get("/health")
async def feedback_system_health(
    workflow_manager: WorkflowManager = Depends(get_workflow_manager)
):
    """
    Get health status of the feedback system.
    
    Returns information about active feedback workflows,
    system performance, and any issues.
    """
    try:
        # Get active workflows
        active_workflows = await workflow_manager.get_active_workflows_async()
        
        # Filter feedback-related workflows
        feedback_workflows = [
            wf for wf in active_workflows 
            if wf.get("workflow_id") in ["bidirectional_feedback", "autonomous_learning_loop"]
        ]
        
        # Get system insights
        insights = workflow_manager.get_feedback_summary()
        
        return {
            "status": "healthy",
            "active_feedback_workflows": len(feedback_workflows),
            "total_feedback_entries": insights.get("system_overview", {}).get("total_feedback_entries", 0),
            "average_system_rating": insights.get("system_overview", {}).get("average_rating", 0.0),
            "active_agents": insights.get("system_overview", {}).get("active_agents", 0),
            "feedback_workflows": feedback_workflows
        }
        
    except Exception as e:
        return {
            "status": "unhealthy",
            "error": str(e),
            "message": "Feedback system health check failed"
        }