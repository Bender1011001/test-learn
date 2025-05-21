from typing import Dict, List, Optional, Any, Callable, AsyncGenerator
import asyncio
import threading
from datetime import datetime
import uuid
from loguru import logger
import json

# This will be replaced with actual CAMEL imports when integrating
# from camel import Agent, Workflow

from .config_manager import ConfigManager
from ...db.models.logs import InteractionLog


class WorkflowManager:
    """Manager for executing and controlling CAMEL AI workflows"""
    
    def __init__(self, config_manager: ConfigManager, db_session_factory):
        """Initialize with config manager and db session factory"""
        self.config_manager = config_manager
        self.db_session_factory = db_session_factory
        self.active_workflows: Dict[str, Dict[str, Any]] = {}
        self._lock = threading.Lock()
    
    async def start_workflow(
        self, 
        workflow_id: str, 
        initial_goal: str,
        callback: Optional[Callable[[str, Dict[str, Any]], None]] = None
    ) -> str:
        """
        Start a new workflow execution
        
        Args:
            workflow_id: ID of workflow configuration to use
            initial_goal: Initial goal/objective for the workflow
            callback: Optional callback for receiving interaction updates
            
        Returns:
            Workflow run ID
        """
        # Get workflow configuration
        workflow_config = self.config_manager.get_workflow_config(workflow_id)
        if not workflow_config:
            raise ValueError(f"Workflow {workflow_id} not found")
        
        # Generate a unique run ID
        run_id = str(uuid.uuid4())
        
        # Create log entry for workflow start
        db = next(self.db_session_factory())
        try:
            start_log = InteractionLog(
                workflow_run_id=run_id,
                agent_name="system",
                agent_type="system",
                input_data={"workflow_id": workflow_id},
                output_data={"status": "started", "initial_goal": initial_goal}
            )
            db.add(start_log)
            db.commit()
        finally:
            db.close()
        
        # Start workflow in a separate thread
        thread = threading.Thread(
            target=self._run_workflow_thread,
            args=(run_id, workflow_id, initial_goal, callback)
        )
        thread.daemon = True
        
        # Register the active workflow
        with self._lock:
            self.active_workflows[run_id] = {
                "workflow_id": workflow_id,
                "status": "starting",
                "start_time": datetime.utcnow(),
                "initial_goal": initial_goal,
                "thread": thread,
                "current_step": 0,
                "total_steps": len(workflow_config.agent_sequence),
            }
        
        # Start the thread
        thread.start()
        
        logger.info(f"Started workflow {workflow_id} with run ID {run_id}")
        return run_id
    
    def _run_workflow_thread(
        self, 
        run_id: str, 
        workflow_id: str, 
        initial_goal: str,
        callback: Optional[Callable]
    ):
        """Execute workflow in a separate thread (blocking)"""
        try:
            # Update status
            with self._lock:
                self.active_workflows[run_id]["status"] = "running"
            
            # Get workflow configuration
            workflow_config = self.config_manager.get_workflow_config(workflow_id)
            agent_sequence = workflow_config.agent_sequence
            
            # This is a placeholder for actual CAMEL workflow execution
            # In a real implementation, this would use the CAMEL library
            
            # Simulate workflow execution for each agent in sequence
            for i, agent_id in enumerate(agent_sequence):
                # Update current step
                with self._lock:
                    self.active_workflows[run_id]["current_step"] = i
                
                # Get agent configuration
                agent_config = self.config_manager.get_agent_config(agent_id)
                
                # Simulate agent processing
                logger.info(f"Workflow {run_id}: Agent {agent_id} processing")
                
                # In a real implementation, this would create and run the agent
                # agent = Agent.from_config(agent_config)
                # result = agent.process(input_data)
                
                # Simulate input/output data
                input_data = {
                    "goal": initial_goal,
                    "step": i,
                    "agent": agent_id
                }
                
                # Simulate processing delay
                asyncio.run(asyncio.sleep(2))
                
                output_data = {
                    "response": f"Simulated response from {agent_id} for goal: {initial_goal}",
                    "status": "success",
                    "timestamp": datetime.utcnow().isoformat()
                }
                
                # Log the interaction
                self._log_interaction(
                    run_id=run_id,
                    agent_name=agent_id,
                    agent_type=agent_config.class_path.split('.')[-1],
                    input_data=input_data,
                    output_data=output_data
                )
                
                # Trigger callback if provided
                if callback:
                    interaction_data = {
                        "run_id": run_id,
                        "agent": agent_id,
                        "step": i,
                        "input": input_data,
                        "output": output_data,
                        "timestamp": datetime.utcnow().isoformat()
                    }
                    callback(run_id, interaction_data)
            
            # Update status to completed
            with self._lock:
                self.active_workflows[run_id]["status"] = "completed"
                self.active_workflows[run_id]["end_time"] = datetime.utcnow()
            
            # Log workflow completion
            db = next(self.db_session_factory())
            try:
                completion_log = InteractionLog(
                    workflow_run_id=run_id,
                    agent_name="system",
                    agent_type="system",
                    input_data={"workflow_id": workflow_id},
                    output_data={"status": "completed"}
                )
                db.add(completion_log)
                db.commit()
            finally:
                db.close()
            
            logger.info(f"Workflow {run_id} completed successfully")
            
        except Exception as e:
            # Handle exceptions
            logger.error(f"Error in workflow {run_id}: {str(e)}")
            
            # Update status to failed
            with self._lock:
                self.active_workflows[run_id]["status"] = "failed"
                self.active_workflows[run_id]["error"] = str(e)
                self.active_workflows[run_id]["end_time"] = datetime.utcnow()
            
            # Log the error
            db = next(self.db_session_factory())
            try:
                error_log = InteractionLog(
                    workflow_run_id=run_id,
                    agent_name="system",
                    agent_type="system",
                    input_data={"workflow_id": workflow_id},
                    output_data={"status": "failed", "error": str(e)}
                )
                db.add(error_log)
                db.commit()
            finally:
                db.close()
    
    def _log_interaction(
        self, 
        run_id: str, 
        agent_name: str, 
        agent_type: str, 
        input_data: Dict[str, Any], 
        output_data: Dict[str, Any]
    ):
        """Log an agent interaction to the database"""
        db = next(self.db_session_factory())
        try:
            log_entry = InteractionLog(
                workflow_run_id=run_id,
                agent_name=agent_name,
                agent_type=agent_type,
                input_data=input_data,
                output_data=output_data
            )
            db.add(log_entry)
            db.commit()
            logger.debug(f"Logged interaction for {agent_name} in workflow {run_id}")
        except Exception as e:
            logger.error(f"Error logging interaction: {str(e)}")
        finally:
            db.close()
    
    def stop_workflow(self, run_id: str) -> bool:
        """Stop a running workflow"""
        with self._lock:
            if run_id not in self.active_workflows:
                logger.warning(f"Workflow {run_id} not found")
                return False
            
            workflow = self.active_workflows[run_id]
            if workflow["status"] not in ["running", "starting"]:
                logger.warning(f"Workflow {run_id} is not running (status: {workflow['status']})")
                return False
            
            # In a real implementation, there would be a mechanism to signal
            # the workflow execution to stop gracefully
            workflow["status"] = "stopping"
        
        # Log workflow stopping
        db = next(self.db_session_factory())
        try:
            log_entry = InteractionLog(
                workflow_run_id=run_id,
                agent_name="system",
                agent_type="system",
                input_data={"action": "stop"},
                output_data={"status": "stopping"}
            )
            db.add(log_entry)
            db.commit()
        finally:
            db.close()
        
        logger.info(f"Stopping workflow {run_id}")
        return True
    
    def get_workflow_status(self, run_id: str) -> Optional[Dict[str, Any]]:
        """Get the status of a workflow"""
        with self._lock:
            if run_id not in self.active_workflows:
                return None
            
            workflow = self.active_workflows[run_id].copy()
            # Remove the thread object from the copy
            if "thread" in workflow:
                del workflow["thread"]
            
            return workflow
    
    def get_active_workflows(self) -> List[Dict[str, Any]]:
        """Get all active workflows"""
        with self._lock:
            return [
                {k: v for k, v in workflow.items() if k != "thread"}
                for workflow in self.active_workflows.values()
            ]
    
    async def stream_workflow_logs(self, run_id: str) -> AsyncGenerator[Dict[str, Any], None]:
        """Stream logs for a specific workflow run"""
        # This is a placeholder implementation
        # In a real implementation, this would query the database for existing logs
        # and then subscribe to new logs as they are created
        
        db = next(self.db_session_factory())
        try:
            # Get existing logs
            logs = db.query(InteractionLog).filter(
                InteractionLog.workflow_run_id == run_id
            ).order_by(InteractionLog.timestamp).all()
            
            # Yield existing logs
            for log in logs:
                yield {
                    "id": log.id,
                    "run_id": log.workflow_run_id,
                    "timestamp": log.timestamp.isoformat(),
                    "agent_name": log.agent_name,
                    "agent_type": log.agent_type,
                    "input": log.input_data,
                    "output": log.output_data
                }
            
            # Check if workflow is still active
            is_active = False
            with self._lock:
                if run_id in self.active_workflows:
                    status = self.active_workflows[run_id]["status"]
                    is_active = status in ["starting", "running"]
            
            # If workflow is still active, periodically check for new logs
            latest_id = logs[-1].id if logs else 0
            
            while is_active:
                await asyncio.sleep(1)  # Poll every second
                
                # Get new logs
                new_logs = db.query(InteractionLog).filter(
                    InteractionLog.workflow_run_id == run_id,
                    InteractionLog.id > latest_id
                ).order_by(InteractionLog.timestamp).all()
                
                # Yield new logs
                for log in new_logs:
                    latest_id = log.id
                    yield {
                        "id": log.id,
                        "run_id": log.workflow_run_id,
                        "timestamp": log.timestamp.isoformat(),
                        "agent_name": log.agent_name,
                        "agent_type": log.agent_type,
                        "input": log.input_data,
                        "output": log.output_data
                    }
                
                # Check if workflow is still active
                with self._lock:
                    if run_id in self.active_workflows:
                        status = self.active_workflows[run_id]["status"]
                        is_active = status in ["starting", "running"]
                    else:
                        is_active = False
        finally:
            db.close()