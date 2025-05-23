from typing import Dict, List, Optional, Any, Callable, AsyncGenerator
import asyncio
import threading
import time
from datetime import datetime
import uuid
from loguru import logger
import json
import importlib

# CAMEL library imports
from camel.agents import BaseAgent
from camel.configs import AgentConfig as CamelAgentConfig
from camel.types import AgentType
from camel.messages import BaseMessage, SystemMessage, AssistantMessage, UserMessage

from .config_manager import ConfigManager
from .redis_service import RedisService, EventChannel
from ...db.models.logs import InteractionLog
from ...agents.bidirectional_feedback import BidirectionalFeedbackManager, BidirectionalFeedbackAgent, FeedbackType


class WorkflowManager:
    """Manager for executing and controlling CAMEL AI workflows"""
    
    def __init__(self, config_manager: ConfigManager, db_session_factory):
        """Initialize with config manager and db session factory"""
        self.config_manager = config_manager
        self.db_session_factory = db_session_factory
        self.active_workflows: Dict[str, Dict[str, Any]] = {}
        self._lock = threading.Lock()
        self.feedback_manager = BidirectionalFeedbackManager()
    
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
        status_data = {
            "workflow_id": workflow_id,
            "status": "starting",
            "start_time": datetime.utcnow(),
            "initial_goal": initial_goal,
            "current_step": 0,
            "total_steps": len(workflow_config.agent_sequence),
        }
        
        with self._lock:
            self.active_workflows[run_id] = {
                **status_data,
                "thread": thread,
            }
        
        # Publish initial status to Redis
        asyncio.run_coroutine_threadsafe(
            self._publish_workflow_status(run_id, status_data),
            asyncio.get_event_loop()
        )
        
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
        db = None  # Initialize db session variable outside the try block
        try:
            # Update status
            with self._lock:
                self.active_workflows[run_id]["status"] = "running"
            
            # Get workflow configuration
            workflow_config = self.config_manager.get_workflow_config(workflow_id)
            agent_sequence = workflow_config.agent_sequence
            
            # Initialize the context that will be passed between agents
            # History will store CAMEL message dictionaries
            workflow_context = {
                "goal": initial_goal,
                "history": [],
                "current_step": 0,
                "run_id": run_id
            }
            
            # Execute workflow for each agent in sequence
            for i, agent_id in enumerate(agent_sequence):
                # Check if workflow is stopping
                with self._lock:
                    if run_id in self.active_workflows and self.active_workflows[run_id]["status"] == "stopping":
                        logger.info(f"Workflow {run_id} received stop signal. Exiting.")
                        break  # Exit the loop if stopping

                # Update current step
                with self._lock:
                    if run_id in self.active_workflows:
                        self.active_workflows[run_id]["current_step"] = i
                        
                        # Create status data for publishing
                        status_data = {k: v for k, v in self.active_workflows[run_id].items()
                                      if k != "thread"}
                
                # Publish updated status to Redis
                asyncio.run_coroutine_threadsafe(
                    self._publish_workflow_status(run_id, status_data),
                    asyncio.get_event_loop()
                )
                
                # Get agent configuration
                agent_config = self.config_manager.get_agent_config(agent_id)
                if not agent_config:
                    raise ValueError(f"Agent configuration {agent_id} not found")
                
                logger.info(f"Workflow {run_id}: Agent {agent_id} processing")
                
                # Dynamically import the agent class using the class path
                try:
                    module_path, class_name = agent_config.class_path.rsplit('.', 1)
                    agent_module = importlib.import_module(module_path)
                    agent_class = getattr(agent_module, class_name)
                except (ImportError, AttributeError) as e:
                    logger.error(f"Failed to import agent class {agent_config.class_path}: {str(e)}")
                    # Log the error and break the workflow
                    output_data = {
                        "content": f"Error importing agent class: {str(e)}",
                        "role": "system",
                        "status": "error",
                        "timestamp": datetime.utcnow().isoformat(),
                        "error": str(e)
                    }
                    self._log_interaction(
                        run_id=run_id,
                        agent_name=agent_id,
                        agent_type="system",  # Use system type for import errors
                        input_data={"agent_id": agent_id, "class_path": agent_config.class_path},
                        output_data=output_data
                    )
                    with self._lock:
                        if run_id in self.active_workflows:
                            self.active_workflows[run_id]["status"] = "failed"
                            self.active_workflows[run_id]["error"] = f"Failed to import agent class {agent_config.class_path}: {str(e)}"
                            self.active_workflows[run_id]["end_time"] = datetime.utcnow()
                    break  # Stop the workflow on import error

                # Convert our config format to CAMEL's AgentConfig format
                camel_config = CamelAgentConfig(
                    model=agent_config.model_id if agent_config.model_id else None,
                    adapter_path=agent_config.adapter_id if agent_config.adapter_id else None,
                    temperature=agent_config.parameters.get("temperature", 0.7),
                    max_tokens=agent_config.parameters.get("max_tokens", 2048)
                )

                # Create and initialize the agent
                # CAMEL agents often need a system message to define their role
                system_message_content = agent_config.parameters.get("system_message", f"You are a {agent_id}.")
                system_message = SystemMessage(role_name=agent_id, content=system_message_content)

                try:
                    agent = agent_class(system_message=system_message, config=camel_config)  # Pass system message and config
                except Exception as e:
                    logger.error(f"Failed to initialize agent {agent_id}: {str(e)}")
                    # Log the error and break the workflow
                    output_data = {
                        "content": f"Error initializing agent: {str(e)}",
                        "role": "system",
                        "status": "error",
                        "timestamp": datetime.utcnow().isoformat(),
                        "error": str(e)
                    }
                    self._log_interaction(
                        run_id=run_id,
                        agent_name=agent_id,
                        agent_type="system",  # Use system type for initialization errors
                        input_data={"agent_id": agent_id, "config": camel_config.to_dict()},
                        output_data=output_data
                    )
                    with self._lock:
                        if run_id in self.active_workflows:
                            self.active_workflows[run_id]["status"] = "failed"
                            self.active_workflows[run_id]["error"] = f"Failed to initialize agent {agent_id}: {str(e)}"
                            self.active_workflows[run_id]["end_time"] = datetime.utcnow()
                    break  # Stop the workflow on initialization error

                # Prepare input message for the agent
                if i == 0:
                    # First agent receives the initial goal as a UserMessage
                    input_message = UserMessage(role_name="user", content=initial_goal)  # Assuming a 'user' role for initial input
                else:
                    # Subsequent agents receive the last message from the history
                    if not workflow_context["history"]:
                        logger.error(f"Workflow {run_id}: History is empty for agent {agent_id} at step {i}")
                        # Log the error and break the workflow
                        output_data = {
                            "content": "Workflow history is empty for subsequent agent.",
                            "role": "system",
                            "status": "error",
                            "timestamp": datetime.utcnow().isoformat(),
                            "error": "Empty history for subsequent agent"
                        }
                        self._log_interaction(
                            run_id=run_id,
                            agent_name=agent_id,
                            agent_type="system",
                            input_data={"agent_id": agent_id, "step": i},
                            output_data=output_data
                        )
                        with self._lock:
                            if run_id in self.active_workflows:
                                self.active_workflows[run_id]["status"] = "failed"
                                self.active_workflows[run_id]["error"] = "Empty history for subsequent agent"
                                self.active_workflows[run_id]["end_time"] = datetime.utcnow()
                        break  # Stop the workflow on empty history

                    last_interaction = workflow_context["history"][-1]
                    last_output = last_interaction["output"]

                    # Create a message from the previous agent's output
                    # Assuming the output_data from the previous step has 'content' and 'role' keys
                    if not isinstance(last_output, dict) or "content" not in last_output or "role" not in last_output:
                        logger.error(f"Workflow {run_id}: Invalid output format from previous agent for agent {agent_id} at step {i}")
                        # Log the error and break the workflow
                        output_data = {
                            "content": "Invalid output format from previous agent.",
                            "role": "system",
                            "status": "error",
                            "timestamp": datetime.utcnow().isoformat(),
                            "error": "Invalid output format from previous agent"
                        }
                        self._log_interaction(
                            run_id=run_id,
                            agent_name=agent_id,
                            agent_type="system",
                            input_data={"agent_id": agent_id, "step": i, "last_output": last_output},
                            output_data=output_data
                        )
                        with self._lock:
                            if run_id in self.active_workflows:
                                self.active_workflows[run_id]["status"] = "failed"
                                self.active_workflows[run_id]["error"] = "Invalid output format from previous agent"
                                self.active_workflows[run_id]["end_time"] = datetime.utcnow()
                        break  # Stop the workflow on invalid output format

                    # The role should be the role of the previous agent's output message
                    # The content is the content of the previous agent's output message
                    input_message = BaseMessage(role_name=last_output["role"], content=last_output["content"])

                # Process input and get response using CAMEL agent's step method
                try:
                    # CAMEL agents typically take a message and return a response message
                    # Need to convert history to CAMEL message objects if they aren't already
                    camel_history = []
                    for item in workflow_context["history"]:
                        try:
                            camel_history.append(BaseMessage.from_dict(item["output"]))
                        except (KeyError, ValueError) as e:
                            logger.warning(f"Could not convert history item to CAMEL message: {e}")
                            # Skip invalid history items rather than failing the whole workflow

                    # The step method signature might vary. Let's assume it takes the input message and history.
                    # This is a simplification and might need adjustment based on the specific CAMEL agent implementation
                    response_messages = agent.step(input_message, chat_history=camel_history)  # Pass input message and history

                    if not response_messages:
                        logger.warning(f"Agent {agent_id} returned no response messages")
                        # Treat no response as a potential issue, but don't necessarily fail the workflow
                        output_data = {
                            "content": "Agent returned no response messages.",
                            "role": "system",
                            "status": "warning",  # Use warning status
                            "timestamp": datetime.utcnow().isoformat(),
                            "details": "Agent step method returned an empty list of messages."
                        }
                        # Do not update workflow_context history with this warning, as it's not an agent interaction message
                    else:
                        # Assuming the agent returns a list of messages, take the last one as the primary output
                        output_message = response_messages[-1]

                        # Convert the output message to a dictionary for logging and context
                        output_data = output_message.to_dict()
                        output_data["status"] = "success"  # Assuming success if a message is returned
                        output_data["timestamp"] = datetime.utcnow().isoformat()

                        # Update workflow context with this agent's results
                        # Store the output message (or its dict representation) in the history
                        workflow_context["history"].append({
                            "agent": agent_id,
                            "input": input_message.to_dict(),  # Store input message as dict
                            "output": output_data  # Store output message as dict
                        })
                        workflow_context["current_step"] = i + 1

                except Exception as e:
                    logger.error(f"Error in agent {agent_id} processing: {str(e)}")
                    output_data = {
                        "content": f"Error in agent processing: {str(e)}",  # Use 'content' key for consistency
                        "role": "system",  # Assign a role for error messages
                        "status": "error",
                        "timestamp": datetime.utcnow().isoformat(),
                        "error": str(e)
                    }
                    # Append error to history as a system message
                    workflow_context["history"].append({
                        "agent": agent_id,
                        "input": input_message.to_dict() if 'input_message' in locals() else {"content": initial_goal, "role": "user"},  # Store input message as dict if created
                        "output": output_data
                    })

                # Log the interaction if output_data was generated
                if 'output_data' in locals():
                    self._log_interaction(
                        run_id=run_id,
                        agent_name=agent_id,
                        agent_type=agent_config.class_path.split('.')[-1],
                        input_data=input_message.to_dict() if 'input_message' in locals() else {"content": initial_goal, "role": "user"},  # Log input message
                        output_data=output_data
                    )

                # Trigger callback if provided and output_data was generated
                if callback and 'output_data' in locals():
                    interaction_data = {
                        "run_id": run_id,
                        "agent": agent_id,
                        "step": i,
                        "input": input_message.to_dict() if 'input_message' in locals() else {"content": initial_goal, "role": "user"},  # Log input message
                        "output": output_data,
                        "timestamp": datetime.utcnow().isoformat()
                    }
                    callback(run_id, interaction_data)

                # Check for errors or specific status responses
                if 'output_data' in locals() and output_data.get("status") == "error":
                    logger.warning(f"Agent {agent_id} reported an error: {output_data.get('error', '')}")
                    # Break the loop on agent error as a default robust behavior
                    with self._lock:
                        if run_id in self.active_workflows:
                            self.active_workflows[run_id]["status"] = "failed"
                            self.active_workflows[run_id]["error"] = output_data.get('error', 'Unknown error')
                            self.active_workflows[run_id]["end_time"] = datetime.utcnow()
                    break  # Stop the workflow on agent error

            # After the loop, check if the workflow completed or failed or stopped
            with self._lock:
                if run_id in self.active_workflows:
                    current_status = self.active_workflows[run_id]["status"]
                else:
                    current_status = "unknown"  # Fallback if workflow was somehow removed

            if current_status == "running":  # If the loop completed without breaking due to error or stopping
                # Update status to completed
                with self._lock:
                    if run_id in self.active_workflows:
                        self.active_workflows[run_id]["status"] = "completed"
                        self.active_workflows[run_id]["end_time"] = datetime.utcnow()
                        
                        # Create status data for publishing
                        status_data = {k: v for k, v in self.active_workflows[run_id].items()
                                      if k != "thread"}
                
                # Publish final status to Redis
                asyncio.run_coroutine_threadsafe(
                    self._publish_workflow_status(run_id, status_data),
                    asyncio.get_event_loop()
                )

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
            
            elif current_status == "stopping":
                # Handle graceful stopping if implemented
                logger.info(f"Workflow {run_id} was stopped")
                # Log workflow stopped
                db = next(self.db_session_factory())
                try:
                    stopped_log = InteractionLog(
                        workflow_run_id=run_id,
                        agent_name="system",
                        agent_type="system",
                        input_data={"workflow_id": workflow_id},
                        output_data={"status": "stopped"}
                    )
                    db.add(stopped_log)
                    db.commit()
                finally:
                    db.close()
            # Note: If status is "failed", the error logging is already done inside the loop

        except Exception as e:
            # Handle unexpected exceptions during workflow execution
            logger.error(f"Unexpected error in workflow {run_id}: {str(e)}")

            # Update status to failed if not already set
            with self._lock:
                if run_id in self.active_workflows and self.active_workflows[run_id]["status"] not in ["failed", "completed", "stopped"]:
                    self.active_workflows[run_id]["status"] = "failed"
                    self.active_workflows[run_id]["error"] = str(e)
                    self.active_workflows[run_id]["end_time"] = datetime.utcnow()

            # Log the error if not already logged
            # Check if the last log entry for this run_id is already this error
            db = next(self.db_session_factory())
            try:
                last_log = db.query(InteractionLog).filter(
                    InteractionLog.workflow_run_id == run_id
                ).order_by(InteractionLog.timestamp.desc()).first()

                error_already_logged = False
                if last_log and last_log.agent_name == "system" and \
                   last_log.output_data.get("status") == "failed" and \
                   last_log.output_data.get("error") == str(e):
                    error_already_logged = True

                if not error_already_logged:
                    error_log = InteractionLog(
                        workflow_run_id=run_id,
                        agent_name="system",
                        agent_type="system",
                        input_data={"workflow_id": workflow_id},
                        output_data={"status": "failed", "error": f"Unexpected workflow error: {str(e)}"}
                    )
                    db.add(error_log)
                    db.commit()
            except Exception as log_e:
                logger.error(f"Error logging unexpected workflow error: {str(log_e)}")
            finally:
                if db:
                    db.close()

    async def _publish_workflow_status(self, run_id: str, status_data: Dict[str, Any]):
        """Publish workflow status update to Redis"""
        try:
            from ..api.dependencies import get_redis_service
            redis_service = await get_redis_service()
            await redis_service.publish_event(EventChannel.WORKFLOW_STATUS, run_id, status_data)
            logger.debug(f"Published workflow status update for {run_id}")
        except Exception as e:
            logger.error(f"Error publishing workflow status: {str(e)}")
    
    async def _publish_workflow_log(self, log_data: Dict[str, Any]):
        """Publish workflow log to Redis"""
        try:
            from ..api.dependencies import get_redis_service
            run_id = log_data["run_id"]
            redis_service = await get_redis_service()
            await redis_service.publish_event(EventChannel.WORKFLOW_LOG, run_id, log_data)
            logger.debug(f"Published workflow log for {run_id}")
        except Exception as e:
            logger.error(f"Error publishing workflow log: {str(e)}")
    
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
            
            # Get the ID of the new log entry
            log_id = log_entry.id
            
            # Create a log data dictionary for publishing
            log_data = {
                "id": log_id,
                "run_id": run_id,
                "timestamp": log_entry.timestamp.isoformat(),
                "agent_name": agent_name,
                "agent_type": agent_type,
                "input": input_data,
                "output": output_data
            }
            
            # Publish the log entry to Redis asynchronously
            # We need to run this in a new thread since we're in a synchronous method
            asyncio.run_coroutine_threadsafe(
                self._publish_workflow_log(log_data),
                asyncio.get_event_loop()
            )
            
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
            
            # Set status to stopping - the workflow thread will check this and stop gracefully
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
    
    async def stop_workflow_async(self, run_id: str) -> bool:
        """
        Async version of stop_workflow
        
        Args:
            run_id: The workflow run ID
            
        Returns:
            True if workflow was stopped, False otherwise
        """
        # Call the sync method since it's mostly in-memory operations
        result = self.stop_workflow(run_id)
        
        # If the workflow was stopped, publish the status update asynchronously
        if result:
            with self._lock:
                if run_id in self.active_workflows:
                    status_data = {k: v for k, v in self.active_workflows[run_id].items()
                                  if k != "thread"}
                    
                    # Publish status update directly (no need for run_coroutine_threadsafe)
                    await self._publish_workflow_status(run_id, status_data)
        
        return result
    
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
    
    async def get_workflow_status_async(self, run_id: str) -> Optional[Dict[str, Any]]:
        """
        Async version of get_workflow_status
        
        Args:
            run_id: The workflow run ID
            
        Returns:
            Workflow status dictionary or None if not found
        """
        # This is a simple wrapper around the sync method since it's just accessing in-memory data
        # In a real implementation with database access, this would be a true async method
        return self.get_workflow_status(run_id)
    
    def get_active_workflows(self) -> List[Dict[str, Any]]:
        """Get all active workflows"""
        with self._lock:
            return [
                {k: v for k, v in workflow.items() if k != "thread"}
                for workflow in self.active_workflows.values()
            ]
    
    async def get_active_workflows_async(self) -> List[Dict[str, Any]]:
        """
        Async version of get_active_workflows
        
        Returns:
            List of active workflow status dictionaries
        """
        # This is a simple wrapper around the sync method since it's just accessing in-memory data
        # In a real implementation with database access, this would be a true async method
        return self.get_active_workflows()
    
    async def stream_workflow_logs(self, run_id: str) -> AsyncGenerator[Dict[str, Any], None]:
        """
        Stream logs for a specific workflow run
        
        This implementation queries existing logs from the database and then
        subscribes to Redis PubSub for new logs while the workflow is active.
        
        Args:
            run_id: The workflow run ID
            
        Yields:
            Log entries as dictionaries
        """
        from ..api.dependencies import get_redis_service
        
        db = next(self.db_session_factory())
        try:
            # Get existing logs
            logs = db.query(InteractionLog).filter(
                InteractionLog.workflow_run_id == run_id
            ).order_by(InteractionLog.timestamp).all()
            
            # Yield existing logs
            for log in logs:
                log_data = {
                    "id": log.id,
                    "run_id": log.workflow_run_id,
                    "timestamp": log.timestamp.isoformat(),
                    "agent_name": log.agent_name,
                    "agent_type": log.agent_type,
                    "input": log.input_data,
                    "output": log.output_data
                }
                yield log_data
            
            # Check if workflow is still active
            is_active = False
            with self._lock:
                if run_id in self.active_workflows:
                    status = self.active_workflows[run_id]["status"]
                    is_active = status in ["starting", "running"]
            
            if not is_active:
                return
            
            # Create a queue for receiving Redis events
            log_queue = asyncio.Queue()
            
            # Define callback for Redis events
            async def on_log_event(data):
                await log_queue.put(data)
            
            # Get Redis service and subscribe to log events
            redis_service = await get_redis_service()
            await redis_service.subscribe(EventChannel.WORKFLOW_LOG, run_id, on_log_event)
            
            try:
                # Loop until workflow is no longer active
                while True:
                    # Check if workflow is still active
                    with self._lock:
                        if run_id in self.active_workflows:
                            status = self.active_workflows[run_id]["status"]
                            is_active = status in ["starting", "running"]
                        else:
                            is_active = False
                    
                    if not is_active and log_queue.empty():
                        break
                    
                    # Wait for a log event or timeout
                    try:
                        log_data = await asyncio.wait_for(log_queue.get(), timeout=1.0)
                        yield log_data
                    except asyncio.TimeoutError:
                        # No new logs received, continue checking if still active
                        continue
            finally:
                # Unsubscribe from Redis events
                await redis_service.unsubscribe(EventChannel.WORKFLOW_LOG, run_id, on_log_event)
        finally:
            db.close()

    # Autonomous Task Generation Methods
    
    async def start_autonomous_task_generation(
        self,
        generation_settings: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Start autonomous task generation workflow.
        
        Args:
            generation_settings: Optional settings for task generation
            
        Returns:
            Workflow run ID for the autonomous generation process
        """
        # Default generation settings
        default_settings = {
            "categories": ["coding", "reasoning", "creative", "analytical"],
            "difficulties": ["beginner", "intermediate", "advanced"],
            "generation_rate": 5,  # tasks per minute
            "max_queue_size": 50,
            "quality_threshold": 6.0
        }
        
        settings = {**default_settings, **(generation_settings or {})}
        
        # Create a special workflow for autonomous task generation
        run_id = str(uuid.uuid4())
        
        # Log the start of autonomous generation
        db = next(self.db_session_factory())
        try:
            start_log = InteractionLog(
                workflow_run_id=run_id,
                agent_name="ProposerAgent",
                agent_type="autonomous_generator",
                input_data={"action": "start_autonomous_generation", "settings": settings},
                output_data={"status": "started", "run_id": run_id}
            )
            db.add(start_log)
            db.commit()
        finally:
            db.close()
        
        # Start autonomous generation in a separate thread
        thread = threading.Thread(
            target=self._run_autonomous_generation_thread,
            args=(run_id, settings)
        )
        thread.daemon = True
        
        # Register the autonomous generation workflow
        status_data = {
            "workflow_id": "autonomous_task_generation",
            "status": "running",
            "start_time": datetime.utcnow(),
            "generation_settings": settings,
            "tasks_generated": 0,
            "current_step": 0,
            "total_steps": -1,  # Continuous generation
        }
        
        with self._lock:
            self.active_workflows[run_id] = {
                **status_data,
                "thread": thread,
            }
        
        # Start the thread
        thread.start()
        
        # Publish initial status
        await self._publish_workflow_status(run_id, status_data)
        
        logger.info(f"Started autonomous task generation with run_id: {run_id}")
        return run_id
    
    def _run_autonomous_generation_thread(self, run_id: str, settings: Dict[str, Any]):
        """
        Run autonomous task generation in a separate thread.
        
        Args:
            run_id: Workflow run ID
            settings: Generation settings
        """
        try:
            from camel.agents.proposer import ProposerAgent, TaskCategory, TaskDifficulty
            from camel.messages import SystemMessage, UserMessage
            
            # Create proposer agent
            system_message = SystemMessage(
                role_name="Proposer",
                content="You are an autonomous task generation agent. Generate diverse, challenging tasks continuously based on the provided settings."
            )
            proposer = ProposerAgent(system_message=system_message)
            
            generation_rate = settings.get("generation_rate", 5)
            max_queue_size = settings.get("max_queue_size", 50)
            categories = settings.get("categories", ["coding", "reasoning"])
            difficulties = settings.get("difficulties", ["intermediate"])
            
            tasks_generated = 0
            last_generation_time = time.time()
            
            logger.info(f"Starting autonomous generation loop for run_id: {run_id}")
            
            while True:
                # Check if workflow should continue
                with self._lock:
                    if run_id not in self.active_workflows:
                        break
                    if self.active_workflows[run_id]["status"] == "stopping":
                        break
                
                # Rate limiting
                current_time = time.time()
                time_since_last = current_time - last_generation_time
                min_interval = 60.0 / generation_rate  # seconds between generations
                
                if time_since_last < min_interval:
                    time.sleep(min_interval - time_since_last)
                
                # Check current queue size
                db = next(self.db_session_factory())
                try:
                    from ...db.models.tasks import Task, TaskStatusEnum
                    pending_count = db.query(Task).filter(Task.status == TaskStatusEnum.PENDING).count()
                    
                    if pending_count >= max_queue_size:
                        logger.debug(f"Queue full ({pending_count}/{max_queue_size}), skipping generation")
                        time.sleep(10)  # Wait before checking again
                        continue
                    
                    # Generate a new task
                    category = random.choice(categories)
                    difficulty = random.choice(difficulties)
                    
                    requirements = {
                        "category": category,
                        "difficulty": difficulty
                    }
                    
                    # Create generation request message
                    request_message = UserMessage(
                        role_name="user",
                        content=f"Generate a {difficulty} {category} task autonomously"
                    )
                    
                    # Generate task using proposer agent
                    response_messages = proposer.step(request_message)
                    
                    if response_messages:
                        # Extract task from response (simplified)
                        task_content = response_messages[0].content
                        
                        # Try to generate task object
                        task_obj = proposer._generate_autonomous_task(requirements)
                        
                        if task_obj and proposer._validate_task_quality(task_obj):
                            # Save to database
                            from ...db.models.tasks import Task, TaskCategoryEnum, TaskDifficultyEnum, TaskPriorityEnum
                            
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
                                generated_by_agent="ProposerAgent_Autonomous",
                                generation_context={"run_id": run_id, "settings": settings}
                            )
                            
                            db.add(db_task)
                            db.commit()
                            
                            tasks_generated += 1
                            last_generation_time = time.time()
                            
                            # Log the generation
                            generation_log = InteractionLog(
                                workflow_run_id=run_id,
                                agent_name="ProposerAgent",
                                agent_type="autonomous_generator",
                                input_data={"requirements": requirements},
                                output_data={
                                    "task_id": task_obj.task_id,
                                    "title": task_obj.title,
                                    "category": category,
                                    "difficulty": difficulty,
                                    "complexity_score": task_obj.complexity_score
                                }
                            )
                            db.add(generation_log)
                            db.commit()
                            
                            # Update workflow status
                            with self._lock:
                                if run_id in self.active_workflows:
                                    self.active_workflows[run_id]["tasks_generated"] = tasks_generated
                                    self.active_workflows[run_id]["last_generation"] = datetime.utcnow()
                            
                            logger.debug(f"Generated task {task_obj.task_id} for run_id: {run_id}")
                        else:
                            logger.warning(f"Generated task failed validation for run_id: {run_id}")
                    else:
                        logger.warning(f"No response from proposer agent for run_id: {run_id}")
                
                except Exception as e:
                    logger.error(f"Error in autonomous generation loop: {str(e)}")
                    # Log the error
                    error_log = InteractionLog(
                        workflow_run_id=run_id,
                        agent_name="ProposerAgent",
                        agent_type="autonomous_generator",
                        input_data={"action": "generate_task"},
                        output_data={"status": "error", "error": str(e)}
                    )
                    db.add(error_log)
                    db.commit()
                finally:
                    db.close()
                
                # Small delay to prevent tight loop
                time.sleep(1)
            
            # Mark workflow as completed
            with self._lock:
                if run_id in self.active_workflows:
                    self.active_workflows[run_id]["status"] = "completed"
                    self.active_workflows[run_id]["end_time"] = datetime.utcnow()
            
            logger.info(f"Autonomous generation completed for run_id: {run_id}, generated {tasks_generated} tasks")
            
        except Exception as e:
            logger.error(f"Error in autonomous generation thread: {str(e)}")
            
            # Mark workflow as failed
            with self._lock:
                if run_id in self.active_workflows:
                    self.active_workflows[run_id]["status"] = "failed"
                    self.active_workflows[run_id]["error"] = str(e)
                    self.active_workflows[run_id]["end_time"] = datetime.utcnow()
            
            # Log the error
            db = next(self.db_session_factory())
            try:
                error_log = InteractionLog(
                    workflow_run_id=run_id,
                    agent_name="ProposerAgent",
                    agent_type="autonomous_generator",
                    input_data={"action": "autonomous_generation"},
                    output_data={"status": "failed", "error": str(e)}
                )
                db.add(error_log)
                db.commit()
            finally:
                db.close()
    
    async def stop_autonomous_task_generation(self, run_id: str) -> bool:
        """
        Stop autonomous task generation workflow.
        
        Args:
            run_id: Workflow run ID
            
        Returns:
            True if stopped successfully, False otherwise
        """
        return await self.stop_workflow_async(run_id)
    
    def get_autonomous_generation_status(self, run_id: str) -> Optional[Dict[str, Any]]:
        """
        Get status of autonomous task generation.
        
        Args:
            run_id: Workflow run ID
            
        Returns:
            Status information or None if not found
        """
        return self.get_workflow_status(run_id)
    
    async def execute_task_workflow(
        self,
        task_id: str,
        executor_settings: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Execute a specific task using the workflow system.
        
        Args:
            task_id: Task identifier to execute
            executor_settings: Optional settings for task execution
            
        Returns:
            Workflow run ID for the task execution
        """
        # Get the task from database
        db = next(self.db_session_factory())
        try:
            from ...db.models.tasks import Task, TaskStatusEnum
            task = db.query(Task).filter(Task.task_id == task_id).first()
            
            if not task:
                raise ValueError(f"Task {task_id} not found")
            
            if task.status != TaskStatusEnum.PENDING:
                raise ValueError(f"Task {task_id} is not in pending status")
            
            # Create workflow for task execution
            run_id = str(uuid.uuid4())
            
            # Update task status
            task.status = TaskStatusEnum.IN_PROGRESS
            task.started_at = datetime.utcnow()
            db.commit()
            
            # Log the start of task execution
            start_log = InteractionLog(
                workflow_run_id=run_id,
                agent_name="ExecutorAgent",
                agent_type="task_executor",
                input_data={"task_id": task_id, "action": "start_execution"},
                output_data={"status": "started", "run_id": run_id}
            )
            db.add(start_log)
            db.commit()
            
        finally:
            db.close()
        
        # Start task execution in a separate thread
        thread = threading.Thread(
            target=self._run_task_execution_thread,
            args=(run_id, task_id, executor_settings or {})
        )
        thread.daemon = True
        
        # Register the task execution workflow
        status_data = {
            "workflow_id": "task_execution",
            "status": "running",
            "start_time": datetime.utcnow(),
            "task_id": task_id,
            "executor_settings": executor_settings or {},
            "current_step": 1,
            "total_steps": 3,  # Execute, Review, Complete
        }
        
        with self._lock:
            self.active_workflows[run_id] = {
                **status_data,
                "thread": thread,
            }
        
        # Start the thread
        thread.start()
        
        # Publish initial status
        await self._publish_workflow_status(run_id, status_data)
        
        logger.info(f"Started task execution workflow for task {task_id} with run_id: {run_id}")
        return run_id
    
    def _run_task_execution_thread(self, run_id: str, task_id: str, executor_settings: Dict[str, Any]):
        """
        Run task execution in a separate thread.
        
        Args:
            run_id: Workflow run ID
            task_id: Task identifier
            executor_settings: Executor settings
        """
        try:
            from camel.agents.executor import ExecutorAgent
            from camel.agents.peer_reviewer import PeerReviewer
            from camel.messages import SystemMessage, UserMessage
            
            # Get the task
            db = next(self.db_session_factory())
            try:
                from ...db.models.tasks import Task, TaskExecution, TaskStatusEnum
                task = db.query(Task).filter(Task.task_id == task_id).first()
                
                if not task:
                    raise ValueError(f"Task {task_id} not found")
                
                # Create executor agent
                executor_system_message = SystemMessage(
                    role_name="Executor",
                    content="You are a task execution agent. Execute tasks according to their specifications and provide detailed results."
                )
                executor = ExecutorAgent(system_message=executor_system_message)
                
                # Create execution record
                execution = TaskExecution(
                    task_id=task_id,
                    executor_agent="ExecutorAgent",
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
                            if any(word in response_content for word in criteria.lower().split()[:3]):
                                success_criteria_met.append(criteria)
                    
                    # Update execution record
                    execution.status = TaskStatusEnum.COMPLETED
                    execution.completed_at = end_time
                    execution.execution_output = execution_output
                    execution.success_criteria_met = success_criteria_met
                    execution.execution_time = execution_time
                    
                    # Calculate basic quality scores
                    execution.quality_score = min(10.0, 5.0 + len(success_criteria_met) * 1.5)
                    execution.efficiency_score = max(1.0, 10.0 - (execution_time / 60))
                    execution.completeness_score = (len(success_criteria_met) / max(1, len(task.success_criteria or []))) * 10
                    
                    # Update task status
                    task.status = TaskStatusEnum.COMPLETED
                    task.completed_at = end_time
                    
                    # Log execution completion
                    completion_log = InteractionLog(
                        workflow_run_id=run_id,
                        agent_name="ExecutorAgent",
                        agent_type="task_executor",
                        input_data={"task_id": task_id, "action": "execute"},
                        output_data={
                            "status": "completed",
                            "execution_time": execution_time,
                            "quality_score": execution.quality_score,
                            "success_criteria_met": len(success_criteria_met)
                        }
                    )
                    db.add(completion_log)
                    
                else:
                    execution.status = TaskStatusEnum.FAILED
                    execution.error_message = "No response from executor agent"
                    task.status = TaskStatusEnum.FAILED
                    
                    # Log execution failure
                    failure_log = InteractionLog(
                        workflow_run_id=run_id,
                        agent_name="ExecutorAgent",
                        agent_type="task_executor",
                        input_data={"task_id": task_id, "action": "execute"},
                        output_data={"status": "failed", "error": "No response from executor agent"}
                    )
                    db.add(failure_log)
                
                db.commit()
                
                # Update workflow status
                with self._lock:
                    if run_id in self.active_workflows:
                        self.active_workflows[run_id]["status"] = "completed"
                        self.active_workflows[run_id]["end_time"] = datetime.utcnow()
                        self.active_workflows[run_id]["execution_result"] = {
                            "execution_id": execution.execution_id,
                            "status": execution.status.value,
                            "quality_score": execution.quality_score
                        }
                
                logger.info(f"Task execution completed for task {task_id}, run_id: {run_id}")
                
            finally:
                db.close()
                
        except Exception as e:
            logger.error(f"Error in task execution thread: {str(e)}")
            
            # Mark workflow and task as failed
            db = next(self.db_session_factory())
            try:
                from ...db.models.tasks import Task, TaskExecution, TaskStatusEnum
                
                task = db.query(Task).filter(Task.task_id == task_id).first()
                if task:
                    task.status = TaskStatusEnum.FAILED
                
                execution = db.query(TaskExecution).filter(TaskExecution.task_id == task_id).order_by(TaskExecution.started_at.desc()).first()
                if execution:
                    execution.status = TaskStatusEnum.FAILED
                    execution.error_message = str(e)
                    execution.completed_at = datetime.utcnow()
                
                # Log the error
                error_log = InteractionLog(
                    workflow_run_id=run_id,
                    agent_name="ExecutorAgent",
                    agent_type="task_executor",
                    input_data={"task_id": task_id, "action": "execute"},
                    output_data={"status": "failed", "error": str(e)}
                )
                db.add(error_log)
                db.commit()
                
            finally:
                db.close()
            
            # Mark workflow as failed
            with self._lock:
                if run_id in self.active_workflows:
                    self.active_workflows[run_id]["status"] = "failed"
                    self.active_workflows[run_id]["error"] = str(e)
                    self.active_workflows[run_id]["end_time"] = datetime.utcnow()
            db.close()
    
    # Bidirectional Feedback Methods
    
    async def start_bidirectional_feedback_workflow(
        self,
        interaction_context: Dict[str, Any],
        agents_involved: List[str] = None
    ) -> str:
        """
        Start a bidirectional feedback collection workflow.
        
        Args:
            interaction_context: Context of the interaction to get feedback on
            agents_involved: List of agent names involved (defaults to all three)
            
        Returns:
            Workflow run ID for the feedback collection process
        """
        if agents_involved is None:
            agents_involved = ["Proposer", "Executor", "PeerReviewer"]
        
        run_id = str(uuid.uuid4())
        
        # Log the start of feedback collection
        db = next(self.db_session_factory())
        try:
            start_log = InteractionLog(
                workflow_run_id=run_id,
                agent_name="FeedbackManager",
                agent_type="feedback_collector",
                input_data={"action": "start_feedback_collection", "agents": agents_involved},
                output_data={"status": "started", "run_id": run_id}
            )
            db.add(start_log)
            db.commit()
        finally:
            db.close()
        
        # Start feedback collection in a separate thread
        thread = threading.Thread(
            target=self._run_bidirectional_feedback_thread,
            args=(run_id, interaction_context, agents_involved)
        )
        thread.daemon = True
        
        # Register the feedback workflow
        status_data = {
            "workflow_id": "bidirectional_feedback",
            "status": "running",
            "start_time": datetime.utcnow(),
            "interaction_context": interaction_context,
            "agents_involved": agents_involved,
            "feedback_collected": 0,
            "current_step": 0,
            "total_steps": len(agents_involved) * (len(agents_involved) - 1),  # Each agent evaluates others
        }
        
        with self._lock:
            self.active_workflows[run_id] = {
                **status_data,
                "thread": thread,
            }
        
        # Start the thread
        thread.start()
        
        # Publish initial status
        await self._publish_workflow_status(run_id, status_data)
        
        logger.info(f"Started bidirectional feedback workflow with run_id: {run_id}")
        return run_id
    
    def _run_bidirectional_feedback_thread(
        self,
        run_id: str,
        interaction_context: Dict[str, Any],
        agents_involved: List[str]
    ):
        """
        Run bidirectional feedback collection in a separate thread.
        
        Args:
            run_id: Workflow run ID
            interaction_context: Context of the interaction
            agents_involved: List of agent names
        """
        # Run the async method in the thread
        asyncio.run(self._run_bidirectional_feedback_async(run_id, interaction_context, agents_involved))
    
    async def _run_bidirectional_feedback_async(
        self,
        run_id: str,
        interaction_context: Dict[str, Any],
        agents_involved: List[str]
    ):
        """
        Async implementation of bidirectional feedback collection.
        
        Args:
            run_id: Workflow run ID
            interaction_context: Context of the interaction
            agents_involved: List of agent names
        """
        try:
            from camel.agents.proposer import ProposerAgent
            from camel.agents.executor import ExecutorAgent
            from camel.agents.peer_reviewer import PeerReviewer
            from camel.messages import SystemMessage
            
            # Create agent instances
            agents = {}
            
            if "Proposer" in agents_involved:
                proposer_system = SystemMessage(
                    role_name="Proposer",
                    content="You are a task proposer agent. Evaluate other agents' performance and provide constructive feedback."
                )
                agents["Proposer"] = BidirectionalFeedbackAgent(
                    ProposerAgent(system_message=proposer_system),
                    self.feedback_manager
                )
            
            if "Executor" in agents_involved:
                executor_system = SystemMessage(
                    role_name="Executor",
                    content="You are a task executor agent. Evaluate other agents' performance and provide constructive feedback."
                )
                agents["Executor"] = BidirectionalFeedbackAgent(
                    ExecutorAgent(system_message=executor_system),
                    self.feedback_manager
                )
            
            if "PeerReviewer" in agents_involved:
                reviewer_system = SystemMessage(
                    role_name="PeerReviewer",
                    content="You are a peer reviewer agent. Evaluate other agents' performance and provide constructive feedback."
                )
                agents["PeerReviewer"] = BidirectionalFeedbackAgent(
                    PeerReviewer(system_message=reviewer_system),
                    self.feedback_manager
                )
            
            feedback_collected = 0
            
            # Collect feedback from each agent about each other agent
            for evaluator_name, evaluator_agent in agents.items():
                for evaluated_name in agents_involved:
                    if evaluator_name != evaluated_name:  # Don't evaluate self
                        try:
                            # Determine feedback type based on agent roles
                            feedback_type = self._determine_feedback_type(evaluator_name, evaluated_name)
                            
                            # Collect feedback
                            feedback_entry = await evaluator_agent.provide_feedback_on(
                                other_agent_name=evaluated_name,
                                interaction_context=interaction_context,
                                feedback_type=feedback_type
                            )
                            
                            if feedback_entry:
                                feedback_collected += 1
                                
                                # Log the feedback collection
                                db = next(self.db_session_factory())
                                try:
                                    feedback_log = InteractionLog(
                                        workflow_run_id=run_id,
                                        agent_name=evaluator_name,
                                        agent_type="feedback_provider",
                                        input_data={
                                            "evaluated_agent": evaluated_name,
                                            "feedback_type": feedback_type.value
                                        },
                                        output_data={
                                            "feedback_id": feedback_entry.feedback_id,
                                            "overall_rating": feedback_entry.overall_rating,
                                            "confidence_score": feedback_entry.confidence_score
                                        }
                                    )
                                    db.add(feedback_log)
                                    db.commit()
                                finally:
                                    db.close()
                                
                                # Update workflow status
                                with self._lock:
                                    if run_id in self.active_workflows:
                                        self.active_workflows[run_id]["feedback_collected"] = feedback_collected
                                        self.active_workflows[run_id]["current_step"] = feedback_collected
                                
                                logger.debug(f"Collected feedback from {evaluator_name} about {evaluated_name}")
                            else:
                                logger.warning(f"Failed to collect feedback from {evaluator_name} about {evaluated_name}")
                        
                        except Exception as e:
                            logger.error(f"Error collecting feedback from {evaluator_name} about {evaluated_name}: {str(e)}")
            
            # Mark workflow as completed
            with self._lock:
                if run_id in self.active_workflows:
                    self.active_workflows[run_id]["status"] = "completed"
                    self.active_workflows[run_id]["end_time"] = datetime.utcnow()
                    self.active_workflows[run_id]["total_feedback_collected"] = feedback_collected
            
            logger.info(f"Bidirectional feedback collection completed for run_id: {run_id}, collected {feedback_collected} feedback entries")
            
        except Exception as e:
            logger.error(f"Error in bidirectional feedback thread: {str(e)}")
            
            # Mark workflow as failed
            with self._lock:
                if run_id in self.active_workflows:
                    self.active_workflows[run_id]["status"] = "failed"
                    self.active_workflows[run_id]["error"] = str(e)
                    self.active_workflows[run_id]["end_time"] = datetime.utcnow()
            
            # Log the error
            db = next(self.db_session_factory())
            try:
                error_log = InteractionLog(
                    workflow_run_id=run_id,
                    agent_name="FeedbackManager",
                    agent_type="feedback_collector",
                    input_data={"action": "bidirectional_feedback"},
                    output_data={"status": "failed", "error": str(e)}
                )
                db.add(error_log)
                db.commit()
            finally:
                db.close()
    
    def _determine_feedback_type(self, evaluator: str, evaluated: str) -> FeedbackType:
        """Determine the appropriate feedback type based on agent roles"""
        feedback_mapping = {
            ("Proposer", "Executor"): FeedbackType.EXECUTION_QUALITY,
            ("Proposer", "PeerReviewer"): FeedbackType.REVIEW_QUALITY,
            ("Executor", "Proposer"): FeedbackType.TASK_QUALITY,
            ("Executor", "PeerReviewer"): FeedbackType.REVIEW_QUALITY,
            ("PeerReviewer", "Proposer"): FeedbackType.TASK_QUALITY,
            ("PeerReviewer", "Executor"): FeedbackType.EXECUTION_QUALITY,
        }
        
        return feedback_mapping.get((evaluator, evaluated), FeedbackType.TASK_QUALITY)
    
    async def start_autonomous_learning_loop(
        self,
        loop_settings: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Start the complete autonomous learning loop: generation → execution → feedback → improvement.
        
        Args:
            loop_settings: Optional settings for the learning loop
            
        Returns:
            Workflow run ID for the learning loop
        """
        # Default loop settings
        default_settings = {
            "generation_rate": 3,  # tasks per minute
            "max_concurrent_tasks": 5,
            "feedback_frequency": "after_each_task",  # or "batch"
            "auto_dpo_training": True,
            "performance_threshold": 7.0,  # minimum performance to continue
            "max_iterations": 100,
            "continuous_operation": True
        }
        
        settings = {**default_settings, **(loop_settings or {})}
        
        run_id = str(uuid.uuid4())
        
        # Log the start of autonomous learning loop
        db = next(self.db_session_factory())
        try:
            start_log = InteractionLog(
                workflow_run_id=run_id,
                agent_name="LearningLoop",
                agent_type="autonomous_learner",
                input_data={"action": "start_learning_loop", "settings": settings},
                output_data={"status": "started", "run_id": run_id}
            )
            db.add(start_log)
            db.commit()
        finally:
            db.close()
        
        # Start learning loop in a separate thread
        thread = threading.Thread(
            target=self._run_autonomous_learning_loop_thread,
            args=(run_id, settings)
        )
        thread.daemon = True
        
        # Register the learning loop workflow
        status_data = {
            "workflow_id": "autonomous_learning_loop",
            "status": "running",
            "start_time": datetime.utcnow(),
            "loop_settings": settings,
            "iterations_completed": 0,
            "tasks_generated": 0,
            "tasks_executed": 0,
            "feedback_sessions": 0,
            "current_performance": 0.0,
            "current_step": 0,
            "total_steps": settings.get("max_iterations", -1),
        }
        
        with self._lock:
            self.active_workflows[run_id] = {
                **status_data,
                "thread": thread,
            }
        
        # Start the thread
        thread.start()
        
        # Publish initial status
        await self._publish_workflow_status(run_id, status_data)
        
        logger.info(f"Started autonomous learning loop with run_id: {run_id}")
        return run_id
    
    def _run_autonomous_learning_loop_thread(self, run_id: str, settings: Dict[str, Any]):
        """
        Run the autonomous learning loop in a separate thread.
        
        Args:
            run_id: Workflow run ID
            settings: Learning loop settings
        """
        try:
            max_iterations = settings.get("max_iterations", 100)
            continuous_operation = settings.get("continuous_operation", True)
            performance_threshold = settings.get("performance_threshold", 7.0)
            
            iterations_completed = 0
            
            while (continuous_operation or iterations_completed < max_iterations):
                # Check if workflow should stop
                with self._lock:
                    if run_id in self.active_workflows:
                        status = self.active_workflows[run_id]["status"]
                        if status == "stopping":
                            break
                    else:
                        break
                
                try:
                    # Step 1: Generate tasks
                    generation_run_id = asyncio.run(self.start_autonomous_task_generation({
                        "generation_rate": settings.get("generation_rate", 3),
                        "max_queue_size": 10,
                        "categories": ["coding", "reasoning", "creative"],
                        "difficulties": ["beginner", "intermediate", "advanced"]
                    }))
                    
                    # Wait for some tasks to be generated
                    time.sleep(30)  # Generate for 30 seconds
                    
                    # Stop generation
                    asyncio.run(self.stop_autonomous_task_generation(generation_run_id))
                    
                    # Step 2: Execute generated tasks
                    db = next(self.db_session_factory())
                    try:
                        from ...db.models.tasks import Task, TaskStatusEnum
                        pending_tasks = db.query(Task).filter(
                            Task.status == TaskStatusEnum.PENDING
                        ).limit(settings.get("max_concurrent_tasks", 5)).all()
                        
                        execution_run_ids = []
                        for task in pending_tasks:
                            execution_run_id = asyncio.run(self.execute_task_workflow(task.task_id))
                            execution_run_ids.append(execution_run_id)
                        
                        # Wait for executions to complete
                        max_wait_time = 300  # 5 minutes
                        start_wait = time.time()
                        
                        while time.time() - start_wait < max_wait_time:
                            all_completed = True
                            for exec_run_id in execution_run_ids:
                                status = self.get_workflow_status(exec_run_id)
                                if status and status["status"] in ["running", "starting"]:
                                    all_completed = False
                                    break
                            
                            if all_completed:
                                break
                            
                            time.sleep(5)
                        
                    finally:
                        db.close()
                    
                    # Step 3: Collect bidirectional feedback
                    feedback_context = {
                        "iteration": iterations_completed,
                        "tasks_in_iteration": len(pending_tasks) if 'pending_tasks' in locals() else 0,
                        "generation_run_id": generation_run_id,
                        "execution_run_ids": execution_run_ids if 'execution_run_ids' in locals() else []
                    }
                    
                    feedback_run_id = asyncio.run(self.start_bidirectional_feedback_workflow(
                        interaction_context=feedback_context,
                        agents_involved=["Proposer", "Executor", "PeerReviewer"]
                    ))
                    
                    # Wait for feedback collection to complete
                    max_feedback_wait = 120  # 2 minutes
                    start_feedback_wait = time.time()
                    
                    while time.time() - start_feedback_wait < max_feedback_wait:
                        feedback_status = self.get_workflow_status(feedback_run_id)
                        if feedback_status and feedback_status["status"] not in ["running", "starting"]:
                            break
                        time.sleep(5)
                    
                    # Step 4: Analyze performance and trigger DPO training if needed
                    current_performance = self._analyze_current_performance()
                    
                    if settings.get("auto_dpo_training", True) and current_performance < performance_threshold:
                        logger.info(f"Performance below threshold ({current_performance} < {performance_threshold}), triggering DPO training")
                        # Trigger DPO training (this would integrate with the existing DPO trainer)
                        # For now, just log the intent
                        db = next(self.db_session_factory())
                        try:
                            training_log = InteractionLog(
                                workflow_run_id=run_id,
                                agent_name="LearningLoop",
                                agent_type="autonomous_learner",
                                input_data={"action": "trigger_dpo_training", "performance": current_performance},
                                output_data={"status": "training_triggered", "threshold": performance_threshold}
                            )
                            db.add(training_log)
                            db.commit()
                        finally:
                            db.close()
                    
                    iterations_completed += 1
                    
                    # Update workflow status
                    with self._lock:
                        if run_id in self.active_workflows:
                            self.active_workflows[run_id]["iterations_completed"] = iterations_completed
                            self.active_workflows[run_id]["current_performance"] = current_performance
                            self.active_workflows[run_id]["current_step"] = iterations_completed
                    
                    logger.info(f"Completed learning loop iteration {iterations_completed} for run_id: {run_id}")
                    
                    # Small delay between iterations
                    time.sleep(10)
                
                except Exception as e:
                    logger.error(f"Error in learning loop iteration {iterations_completed}: {str(e)}")
                    # Continue to next iteration unless it's a critical error
                    time.sleep(30)
            
            # Mark workflow as completed
            with self._lock:
                if run_id in self.active_workflows:
                    self.active_workflows[run_id]["status"] = "completed"
                    self.active_workflows[run_id]["end_time"] = datetime.utcnow()
            
            logger.info(f"Autonomous learning loop completed for run_id: {run_id}, completed {iterations_completed} iterations")
            
        except Exception as e:
            logger.error(f"Error in autonomous learning loop thread: {str(e)}")
            
            # Mark workflow as failed
            with self._lock:
                if run_id in self.active_workflows:
                    self.active_workflows[run_id]["status"] = "failed"
                    self.active_workflows[run_id]["error"] = str(e)
                    self.active_workflows[run_id]["end_time"] = datetime.utcnow()
    
    def _analyze_current_performance(self) -> float:
        """Analyze current system performance based on recent feedback"""
        try:
            # Get recent feedback insights
            insights = self.feedback_manager.get_feedback_insights()
            
            if "system_overview" in insights:
                return insights["system_overview"].get("average_rating", 5.0)
            
            return 5.0  # Default neutral performance
            
        except Exception as e:
            logger.error(f"Error analyzing performance: {str(e)}")
            return 5.0
    
    def get_feedback_summary(self, agent_name: Optional[str] = None) -> Dict[str, Any]:
        """Get feedback summary for a specific agent or all agents"""
        if agent_name:
            return self.feedback_manager.get_agent_performance_summary(agent_name)
        else:
            return self.feedback_manager.get_feedback_insights()
    
    def export_feedback_for_training(self, min_rating: float = 6.0) -> List[Dict[str, Any]]:
        """Export feedback data for DPO training"""
        return self.feedback_manager.export_feedback_for_dpo(min_rating)