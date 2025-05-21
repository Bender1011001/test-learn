import requests
import json
import logging
import uuid
from typing import Dict, List, Optional, Any, Union
from tenacity import retry, stop_after_attempt, wait_exponential

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("api_client")

# Default timeout values (connect_timeout, read_timeout)
DEFAULT_TIMEOUT = (3, 10)

class APIClient:
    """Client for interacting with the CAMEL Extensions API."""
    
    def __init__(self, base_url: str = "http://localhost:8000/api"):
        """Initialize API client with base URL."""
        self.base_url = base_url
        self.session = requests.Session()
        self._active_websockets = []
        self.logger = logger
        self.logger.info(f"API client initialized with base URL: {base_url}")
    
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, max=8))
    def _request(self, method: str, path: str, **kwargs):
        """Make HTTP request with automatic retries and logging."""
        # Ensure headers exist
        headers = kwargs.get("headers", {})
        
        # Add trace ID for request tracing
        if "X-Trace-ID" not in headers:
            headers["X-Trace-ID"] = str(uuid.uuid4())
        
        kwargs["headers"] = headers
        url = f"{self.base_url}{path}"
        
        self.logger.debug(f"API Request: {method} {url} (Trace: {headers['X-Trace-ID']})")
        
        try:
            response = self.session.request(
                method,
                url,
                timeout=DEFAULT_TIMEOUT,
                **kwargs
            )
            response.raise_for_status()
            self.logger.debug(f"API Response: {response.status_code}")
            return response.json()
        except requests.exceptions.HTTPError as e:
            self.logger.error(f"HTTP error: {e}")
            if e.response is not None:
                try:
                    error_msg = e.response.json().get("detail", str(e))
                    self.logger.error(f"API error detail: {error_msg}")
                except:
                    pass
            raise
        except requests.exceptions.RequestException as e:
            self.logger.error(f"Request error: {e}")
            raise
    
    def get(self, path: str, params: Optional[Dict[str, Any]] = None, **kwargs):
        """Make GET request to API."""
        return self._request("GET", path, params=params, **kwargs)
    
    def post(self, path: str, json: Optional[Dict[str, Any]] = None, **kwargs):
        """Make POST request to API."""
        return self._request("POST", path, json=json, **kwargs)
    
    def put(self, path: str, json: Optional[Dict[str, Any]] = None, **kwargs):
        """Make PUT request to API."""
        return self._request("PUT", path, json=json, **kwargs)
    
    def delete(self, path: str, **kwargs):
        """Make DELETE request to API."""
        return self._request("DELETE", path, **kwargs)
    
    # --- Workflow Endpoints ---
    
    def get_available_workflows(self) -> List[Dict[str, Any]]:
        """Get all available workflow configurations."""
        return self.get("/workflows/available")
    
    def start_workflow(self, workflow_id: str, initial_goal: str) -> str:
        """
        Start a new workflow execution.
        
        Args:
            workflow_id: ID of workflow configuration to use
            initial_goal: Initial goal/objective for the workflow
            
        Returns:
            Workflow run ID
        """
        payload = {"workflow_id": workflow_id, "initial_goal": initial_goal}
        response = self.post("/workflows/start", json=payload)
        return response["run_id"]
    
    def get_workflow_status(self, run_id: str) -> Dict[str, Any]:
        """Get status of a workflow execution."""
        return self.get(f"/workflows/status/{run_id}")
    
    def stop_workflow(self, run_id: str) -> bool:
        """Stop a running workflow execution."""
        response = self.post(f"/workflows/stop/{run_id}")
        return response.get("success", False)
    
    def get_active_workflows(self) -> List[Dict[str, Any]]:
        """Get all active workflow executions."""
        return self.get("/workflows/active")
    
    # --- Config Endpoints ---
    
    def get_all_agent_configs(self) -> Dict[str, Any]:
        """Get all agent configurations."""
        response = self.get("/configs/agents")
        return response.get("agents", {})
    
    def get_agent_config(self, agent_id: str) -> Dict[str, Any]:
        """Get configuration for a specific agent."""
        return self.get(f"/configs/agents/{agent_id}")
    
    def update_agent_config(self, agent_id: str, config_updates: Dict[str, Any]) -> bool:
        """Update configuration for a specific agent."""
        response = self.put(f"/configs/agents/{agent_id}", json=config_updates)
        return response.get("success", False)
    
    def get_workflow_settings(self) -> Dict[str, Any]:
        """Get global workflow settings."""
        return self.get("/configs/settings")
    
    def get_all_adapters(self, agent_type: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Get all saved DPO adapters.
        
        Args:
            agent_type: Optional filter by agent type
        """
        params = {}
        if agent_type:
            params["agent_type"] = agent_type
        return self.get("/configs/adapters", params=params)
    
    def reload_config(self) -> bool:
        """Reload configuration from file."""
        response = self.post("/configs/reload")
        return response.get("success", False)
    
    # --- Log Endpoints ---
    
    def get_logs(
        self, 
        workflow_id: Optional[str] = None,
        agent_name: Optional[str] = None,
        agent_type: Optional[str] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        has_annotation: Optional[bool] = None,
        keyword: Optional[str] = None,
        offset: int = 0,
        limit: int = 100,
        sort_by: str = "timestamp",
        sort_desc: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Get logs with various filters.
        
        Args:
            workflow_id: Filter by workflow run ID
            agent_name: Filter by agent name
            agent_type: Filter by agent type
            start_date: Filter by timestamp >= start_date (ISO format)
            end_date: Filter by timestamp <= end_date (ISO format)
            has_annotation: Filter for logs with/without annotations
            keyword: Search for keyword in input/output data
            offset: Query offset for pagination
            limit: Query limit for pagination
            sort_by: Field to sort by
            sort_desc: Whether to sort descending
        """
        params = {
            "offset": offset,
            "limit": limit,
            "sort_by": sort_by,
            "sort_desc": str(sort_desc).lower()
        }
        
        if workflow_id:
            params["workflow_id"] = workflow_id
        if agent_name:
            params["agent_name"] = agent_name
        if agent_type:
            params["agent_type"] = agent_type
        if start_date:
            params["start_date"] = start_date
        if end_date:
            params["end_date"] = end_date
        if has_annotation is not None:
            params["has_annotation"] = str(has_annotation).lower()
        if keyword:
            params["keyword"] = keyword
        
        return self.get("/logs", params=params)
    
    def get_logs_summary(self) -> Dict[str, Any]:
        """Get summary statistics about logs."""
        return self.get("/logs/summary")
    
    def get_log_by_id(self, log_id: int) -> Dict[str, Any]:
        """Get a specific log entry by ID."""
        return self.get(f"/logs/{log_id}")
    
    def get_annotation(self, log_id: int) -> Optional[Dict[str, Any]]:
        """Get annotation for a specific log entry."""
        try:
            return self.get(f"/logs/{log_id}/annotation")
        except requests.exceptions.HTTPError as e:
            if e.response and e.response.status_code == 404:
                return None
            raise
    
    def save_annotation(self, annotation_data: Dict[str, Any]) -> int:
        """
        Save or update an annotation for a log entry.
        
        Args:
            annotation_data: Dictionary with annotation data, must include log_entry_id
        
        Returns:
            Annotation ID
        """
        response = self.post("/logs/annotations", json=annotation_data)
        return response.get("id")
    
    def delete_annotation(self, annotation_id: int) -> bool:
        """Delete an annotation."""
        response = self.delete(f"/logs/annotations/{annotation_id}")
        return response.get("success", False)
    
    # --- DPO Training Endpoints ---
    
    def start_dpo_training(
        self, 
        agent_type: str, 
        base_model_id: str, 
        adapter_name: str, 
        training_args: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Start a new DPO training job.
        
        Args:
            agent_type: Type of agent to train (e.g., 'proposer')
            base_model_id: ID of base model to fine-tune
            adapter_name: Name for the new adapter
            training_args: Optional override for default training arguments
            
        Returns:
            Job ID
        """
        payload = {
            "agent_type": agent_type,
            "base_model_id": base_model_id,
            "adapter_name": adapter_name,
        }
        
        if training_args:
            payload["training_args"] = training_args
        
        response = self.post("/dpo/start", json=payload)
        return response["job_id"]
    
    def get_training_job_status(self, job_id: str) -> Dict[str, Any]:
        """Get the status of a DPO training job."""
        return self.get(f"/dpo/status/{job_id}")
    
    def get_training_job_output(self, job_id: str, max_lines: int = 100) -> List[str]:
        """Get the output logs of a DPO training job."""
        response = self.get(f"/dpo/output/{job_id}", params={"max_lines": max_lines})
        return response.get("output", [])
    
    def cancel_training_job(self, job_id: str) -> bool:
        """Cancel a running DPO training job."""
        response = self.post(f"/dpo/cancel/{job_id}")
        return response.get("success", False)
    
    def get_active_training_jobs(self) -> List[Dict[str, Any]]:
        """Get all active DPO training jobs."""
        return self.get("/dpo/active")
    
    def get_training_summary(self) -> Dict[str, Any]:
        """Get summary statistics about DPO training jobs."""
        return self.get("/dpo/summary")
    
    def close(self):
        """Close the API client and any associated resources."""
        # Close any active WebSocket connections
        for ws in self._active_websockets:
            try:
                ws.close()
            except Exception as e:
                self.logger.error(f"Error closing WebSocket: {e}")
        
        self.session.close()
        self.logger.info("API client closed")