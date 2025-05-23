"""
API endpoints for task generation and management.

This module provides REST API endpoints for:
- Autonomous task generation
- Task queue management
- Task execution tracking
- Task feedback and evaluation
- Task generation settings
"""

from typing import List, Optional, Dict, Any
from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session
from datetime import datetime, timedelta
import json

from ..dependencies import get_db
from ...db.models.tasks import (
    Task, TaskExecution, TaskFeedback, TaskQueue, TaskGenerationSettings,
    TaskDifficultyEnum, TaskCategoryEnum, TaskPriorityEnum, TaskStatusEnum
)
from ...core.services.config_manager import ConfigManager
from ...core.services.workflow_manager import WorkflowManager

router = APIRouter(prefix="/tasks", tags=["tasks"])


# Pydantic models for request/response
from pydantic import BaseModel, Field


class TaskGenerationRequest(BaseModel):
    """Request model for task generation"""
    category: Optional[str] = None
    difficulty: Optional[str] = None
    priority: Optional[str] = "medium"
    requirements: Optional[Dict[str, Any]] = None
    count: int = Field(default=1, ge=1, le=10)


class TaskResponse(BaseModel):
    """Response model for task data"""
    id: int
    task_id: str
    title: str
    description: str
    category: str
    difficulty: str
    priority: str
    status: str
    complexity_score: float
    estimated_duration: int
    success_criteria: List[str]
    evaluation_metrics: Dict[str, Any]
    prerequisites: List[str]
    tags: List[str]
    created_at: str
    updated_at: str


class TaskExecutionRequest(BaseModel):
    """Request model for task execution"""
    task_id: str
    executor_agent: Optional[str] = "ExecutorAgent"


class TaskExecutionResponse(BaseModel):
    """Response model for task execution"""
    id: int
    execution_id: str
    task_id: str
    executor_agent: str
    status: str
    execution_output: Optional[Dict[str, Any]]
    success_criteria_met: List[str]
    execution_time: Optional[float]
    error_message: Optional[str]
    quality_score: Optional[float]
    efficiency_score: Optional[float]
    completeness_score: Optional[float]
    started_at: str
    completed_at: Optional[str]


class TaskFeedbackRequest(BaseModel):
    """Request model for task feedback"""
    task_id: str
    execution_id: Optional[str] = None
    overall_rating: float = Field(ge=1, le=10)
    strengths: List[str] = []
    areas_for_improvement: List[str] = []
    effectiveness_score: Optional[float] = Field(default=None, ge=1, le=10)
    correctness_score: Optional[float] = Field(default=None, ge=1, le=10)
    detailed_feedback: Optional[str] = None
    task_difficulty_assessment: Optional[str] = None
    reviewer_agent: Optional[str] = "PeerReviewer"


class TaskQueueStatus(BaseModel):
    """Response model for task queue status"""
    total_tasks: int
    pending_tasks: int
    in_progress_tasks: int
    completed_tasks: int
    failed_tasks: int
    difficulty_distribution: Dict[str, int]
    category_distribution: Dict[str, int]
    average_complexity: float
    average_completion_time: Optional[float]


@router.post("/generate", response_model=List[TaskResponse])
async def generate_tasks(
    request: TaskGenerationRequest,
    db: Session = Depends(get_db)
):
    """
    Generate new tasks autonomously based on requirements.
    
    Args:
        request: Task generation parameters
        db: Database session
        
    Returns:
        List of generated tasks
    """
    try:
        # Import here to avoid circular imports
        from camel.agents.proposer import ProposerAgent, TaskCategory, TaskDifficulty, TaskPriority
        from camel.messages import SystemMessage
        
        # Create proposer agent
        system_message = SystemMessage(
            role_name="Proposer",
            content="You are an autonomous task generation agent. Generate diverse, challenging tasks across multiple domains."
        )
        proposer = ProposerAgent(system_message=system_message)
        
        generated_tasks = []
        
        for _ in range(request.count):
            # Prepare requirements
            requirements = request.requirements or {}
            
            if request.category:
                try:
                    requirements["category"] = TaskCategory(request.category.lower())
                except ValueError:
                    pass
            
            if request.difficulty:
                try:
                    requirements["difficulty"] = TaskDifficulty(request.difficulty.lower())
                except ValueError:
                    pass
            
            if request.priority:
                try:
                    requirements["priority"] = TaskPriority(request.priority.upper())
                except ValueError:
                    pass
            
            # Generate task
            task_obj = proposer._generate_autonomous_task(requirements)
            
            if task_obj and proposer._validate_task_quality(task_obj):
                # Convert to database model
                db_task = Task(
                    task_id=task_obj.task_id,
                    title=task_obj.title,
                    description=task_obj.description,
                    category=TaskCategoryEnum(task_obj.category.value),
                    difficulty=TaskDifficultyEnum(task_obj.difficulty.value),
                    priority=TaskPriorityEnum(task_obj.priority.value.lower()),
                    complexity_score=task_obj.complexity_score,
                    estimated_duration=task_obj.estimated_duration,
                    success_criteria=task_obj.success_criteria,
                    evaluation_metrics=task_obj.evaluation_metrics,
                    prerequisites=task_obj.prerequisites,
                    tags=task_obj.tags,
                    generation_context=requirements
                )
                
                db.add(db_task)
                db.commit()
                db.refresh(db_task)
                
                generated_tasks.append(TaskResponse(**db_task.to_dict()))
        
        return generated_tasks
        
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=f"Failed to generate tasks: {str(e)}")


@router.get("/", response_model=List[TaskResponse])
async def get_tasks(
    skip: int = Query(0, ge=0),
    limit: int = Query(100, ge=1, le=1000),
    category: Optional[str] = None,
    difficulty: Optional[str] = None,
    status: Optional[str] = None,
    db: Session = Depends(get_db)
):
    """
    Get tasks with optional filtering.
    
    Args:
        skip: Number of tasks to skip
        limit: Maximum number of tasks to return
        category: Filter by category
        difficulty: Filter by difficulty
        status: Filter by status
        db: Database session
        
    Returns:
        List of tasks
    """
    query = db.query(Task)
    
    if category:
        try:
            category_enum = TaskCategoryEnum(category.lower())
            query = query.filter(Task.category == category_enum)
        except ValueError:
            raise HTTPException(status_code=400, detail=f"Invalid category: {category}")
    
    if difficulty:
        try:
            difficulty_enum = TaskDifficultyEnum(difficulty.lower())
            query = query.filter(Task.difficulty == difficulty_enum)
        except ValueError:
            raise HTTPException(status_code=400, detail=f"Invalid difficulty: {difficulty}")
    
    if status:
        try:
            status_enum = TaskStatusEnum(status.lower())
            query = query.filter(Task.status == status_enum)
        except ValueError:
            raise HTTPException(status_code=400, detail=f"Invalid status: {status}")
    
    tasks = query.order_by(Task.created_at.desc()).offset(skip).limit(limit).all()
    
    return [TaskResponse(**task.to_dict()) for task in tasks]


@router.get("/{task_id}", response_model=TaskResponse)
async def get_task(task_id: str, db: Session = Depends(get_db)):
    """
    Get a specific task by ID.
    
    Args:
        task_id: Task identifier
        db: Database session
        
    Returns:
        Task details
    """
    task = db.query(Task).filter(Task.task_id == task_id).first()
    
    if not task:
        raise HTTPException(status_code=404, detail="Task not found")
    
    return TaskResponse(**task.to_dict())


@router.post("/{task_id}/execute", response_model=TaskExecutionResponse)
async def execute_task(
    task_id: str,
    request: TaskExecutionRequest,
    db: Session = Depends(get_db)
):
    """
    Execute a specific task.
    
    Args:
        task_id: Task identifier
        request: Execution parameters
        db: Database session
        
    Returns:
        Execution results
    """
    # Get the task
    task = db.query(Task).filter(Task.task_id == task_id).first()
    
    if not task:
        raise HTTPException(status_code=404, detail="Task not found")
    
    if task.status != TaskStatusEnum.PENDING:
        raise HTTPException(status_code=400, detail="Task is not in pending status")
    
    try:
        # Import here to avoid circular imports
        from camel.agents.executor import ExecutorAgent
        from camel.messages import SystemMessage, UserMessage
        
        # Create executor agent
        system_message = SystemMessage(
            role_name="Executor",
            content="You are a task execution agent. Execute tasks according to their specifications and provide detailed results."
        )
        executor = ExecutorAgent(system_message=system_message)
        
        # Update task status
        task.status = TaskStatusEnum.IN_PROGRESS
        task.started_at = datetime.utcnow()
        db.commit()
        
        # Create execution record
        execution = TaskExecution(
            task_id=task_id,
            executor_agent=request.executor_agent,
            status=TaskStatusEnum.IN_PROGRESS,
            started_at=datetime.utcnow()
        )
        db.add(execution)
        db.commit()
        db.refresh(execution)
        
        # Format task as message for executor
        task_message = f"# {task.title}\n\n"
        task_message += f"**Category:** {task.category.value.title()}\n"
        task_message += f"**Difficulty:** {task.difficulty.value.title()}\n"
        task_message += f"**Task ID:** {task.task_id}\n\n"
        task_message += f"## Description\n{task.description}\n\n"
        
        if task.success_criteria:
            task_message += f"## Success Criteria\n"
            for criteria in task.success_criteria:
                task_message += f"- {criteria}\n"
        
        user_message = UserMessage(role_name="user", content=task_message)
        
        # Execute the task
        start_time = datetime.utcnow()
        response_messages = executor.step(user_message)
        end_time = datetime.utcnow()
        
        execution_time = (end_time - start_time).total_seconds()
        
        # Process execution results
        if response_messages:
            execution_output = {
                "response": response_messages[0].content,
                "agent_response": response_messages[0].to_dict() if hasattr(response_messages[0], 'to_dict') else str(response_messages[0])
            }
            
            # Extract success criteria met (simplified)
            success_criteria_met = []
            if task.success_criteria:
                response_content = response_messages[0].content.lower()
                for criteria in task.success_criteria:
                    # Simple keyword matching - in production, use more sophisticated analysis
                    if any(word in response_content for word in criteria.lower().split()[:3]):
                        success_criteria_met.append(criteria)
            
            # Update execution record
            execution.status = TaskStatusEnum.COMPLETED
            execution.completed_at = end_time
            execution.execution_output = execution_output
            execution.success_criteria_met = success_criteria_met
            execution.execution_time = execution_time
            
            # Calculate basic quality scores (simplified)
            execution.quality_score = min(10.0, 5.0 + len(success_criteria_met) * 1.5)
            execution.efficiency_score = max(1.0, 10.0 - (execution_time / 60))  # Penalize long execution times
            execution.completeness_score = (len(success_criteria_met) / max(1, len(task.success_criteria or []))) * 10
            
            # Update task status
            task.status = TaskStatusEnum.COMPLETED
            task.completed_at = end_time
            
        else:
            execution.status = TaskStatusEnum.FAILED
            execution.error_message = "No response from executor agent"
            task.status = TaskStatusEnum.FAILED
        
        db.commit()
        db.refresh(execution)
        
        return TaskExecutionResponse(**execution.to_dict())
        
    except Exception as e:
        # Update status to failed
        execution.status = TaskStatusEnum.FAILED
        execution.error_message = str(e)
        execution.completed_at = datetime.utcnow()
        task.status = TaskStatusEnum.FAILED
        db.commit()
        
        raise HTTPException(status_code=500, detail=f"Task execution failed: {str(e)}")


@router.get("/{task_id}/executions", response_model=List[TaskExecutionResponse])
async def get_task_executions(task_id: str, db: Session = Depends(get_db)):
    """
    Get all executions for a specific task.
    
    Args:
        task_id: Task identifier
        db: Database session
        
    Returns:
        List of task executions
    """
    executions = db.query(TaskExecution).filter(TaskExecution.task_id == task_id).order_by(TaskExecution.started_at.desc()).all()
    
    return [TaskExecutionResponse(**execution.to_dict()) for execution in executions]


@router.post("/{task_id}/feedback")
async def submit_task_feedback(
    task_id: str,
    request: TaskFeedbackRequest,
    db: Session = Depends(get_db)
):
    """
    Submit feedback for a task execution.
    
    Args:
        task_id: Task identifier
        request: Feedback data
        db: Database session
        
    Returns:
        Success message
    """
    # Verify task exists
    task = db.query(Task).filter(Task.task_id == task_id).first()
    if not task:
        raise HTTPException(status_code=404, detail="Task not found")
    
    # Verify execution exists if specified
    if request.execution_id:
        execution = db.query(TaskExecution).filter(TaskExecution.execution_id == request.execution_id).first()
        if not execution:
            raise HTTPException(status_code=404, detail="Execution not found")
    
    try:
        # Create feedback record
        feedback = TaskFeedback(
            task_id=task_id,
            execution_id=request.execution_id,
            reviewer_agent=request.reviewer_agent,
            overall_rating=request.overall_rating,
            strengths=request.strengths,
            areas_for_improvement=request.areas_for_improvement,
            effectiveness_score=request.effectiveness_score,
            correctness_score=request.correctness_score,
            detailed_feedback=request.detailed_feedback,
            task_difficulty_assessment=request.task_difficulty_assessment
        )
        
        db.add(feedback)
        db.commit()
        
        return {"message": "Feedback submitted successfully", "feedback_id": feedback.feedback_id}
        
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=f"Failed to submit feedback: {str(e)}")


@router.get("/queue/status", response_model=TaskQueueStatus)
async def get_queue_status(db: Session = Depends(get_db)):
    """
    Get current task queue status and statistics.
    
    Args:
        db: Database session
        
    Returns:
        Queue status information
    """
    try:
        # Get task counts by status
        total_tasks = db.query(Task).count()
        pending_tasks = db.query(Task).filter(Task.status == TaskStatusEnum.PENDING).count()
        in_progress_tasks = db.query(Task).filter(Task.status == TaskStatusEnum.IN_PROGRESS).count()
        completed_tasks = db.query(Task).filter(Task.status == TaskStatusEnum.COMPLETED).count()
        failed_tasks = db.query(Task).filter(Task.status == TaskStatusEnum.FAILED).count()
        
        # Get difficulty distribution
        difficulty_dist = {}
        for difficulty in TaskDifficultyEnum:
            count = db.query(Task).filter(Task.difficulty == difficulty).count()
            difficulty_dist[difficulty.value] = count
        
        # Get category distribution
        category_dist = {}
        for category in TaskCategoryEnum:
            count = db.query(Task).filter(Task.category == category).count()
            category_dist[category.value] = count
        
        # Calculate average complexity
        avg_complexity_result = db.query(db.func.avg(Task.complexity_score)).scalar()
        avg_complexity = float(avg_complexity_result) if avg_complexity_result else 0.0
        
        # Calculate average completion time
        completed_executions = db.query(TaskExecution).filter(
            TaskExecution.status == TaskStatusEnum.COMPLETED,
            TaskExecution.execution_time.isnot(None)
        ).all()
        
        avg_completion_time = None
        if completed_executions:
            total_time = sum(exec.execution_time for exec in completed_executions)
            avg_completion_time = total_time / len(completed_executions)
        
        return TaskQueueStatus(
            total_tasks=total_tasks,
            pending_tasks=pending_tasks,
            in_progress_tasks=in_progress_tasks,
            completed_tasks=completed_tasks,
            failed_tasks=failed_tasks,
            difficulty_distribution=difficulty_dist,
            category_distribution=category_dist,
            average_complexity=avg_complexity,
            average_completion_time=avg_completion_time
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get queue status: {str(e)}")


@router.delete("/{task_id}")
async def delete_task(task_id: str, db: Session = Depends(get_db)):
    """
    Delete a specific task and all related data.
    
    Args:
        task_id: Task identifier
        db: Database session
        
    Returns:
        Success message
    """
    task = db.query(Task).filter(Task.task_id == task_id).first()
    
    if not task:
        raise HTTPException(status_code=404, detail="Task not found")
    
    try:
        db.delete(task)
        db.commit()
        
        return {"message": f"Task {task_id} deleted successfully"}
        
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=f"Failed to delete task: {str(e)}")


@router.post("/queue/clear")
async def clear_task_queue(
    status_filter: Optional[str] = None,
    db: Session = Depends(get_db)
):
    """
    Clear tasks from the queue with optional status filtering.
    
    Args:
        status_filter: Only clear tasks with this status (optional)
        db: Database session
        
    Returns:
        Number of tasks cleared
    """
    try:
        query = db.query(Task)
        
        if status_filter:
            try:
                status_enum = TaskStatusEnum(status_filter.lower())
                query = query.filter(Task.status == status_enum)
            except ValueError:
                raise HTTPException(status_code=400, detail=f"Invalid status: {status_filter}")
        
        tasks_to_delete = query.all()
        count = len(tasks_to_delete)
        
        for task in tasks_to_delete:
            db.delete(task)
        
        db.commit()
        
        return {"message": f"Cleared {count} tasks from queue"}
        
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=f"Failed to clear queue: {str(e)}")