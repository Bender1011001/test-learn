import requests
import logging
import time
from typing import Dict, Any, Optional, List, Union
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
import streamlit as st

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("api_client")

class APIClient:
    """Client for interfacing with the CAMEL Extensions API."""
    
    def __init__(self, base_url: str = "http://localhost:8000"):
        """
        Initialize API client.
        
        Args:
            base_url: Base URL of the API server
        """
        self.base_url = base_url.rstrip('/')
        self.logger = logger
        self.session = requests.Session()
        self.logger.info(f"API client initialized with base URL: {base_url}")
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        retry=retry_if_exception_type(requests.exceptions.RequestException)
    )
    def _request(self, method: str, path: str, **kwargs) -> Dict[str, Any]:
        """
        Make a request to the API with retry logic.
        
        Args:
            method: HTTP method
            path: API endpoint path
            **kwargs: Additional arguments to pass to requests
            
        Returns:
            JSON response from API
        
        Raises:
            requests.exceptions.RequestException: If the request fails after retries
        """
        # Apply default timeouts if not specified
        if 'timeout' not in kwargs:
            kwargs['timeout'] = (3, 10)  # (connect timeout, read timeout) in seconds
            
        url = f"{self.base_url}/{path.lstrip('/')}"
        self.logger.debug(f"Making {method} request to {url}")
        
        try:
            response = self.session.request(method, url, **kwargs)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.HTTPError as e:
            self.logger.error(f"HTTP error: {e}")
            if e.response is not None:
                try:
                    # Try to get JSON error details
                    error_msg = e.response.json().get("detail", str(e))
                    self.logger.error(f"API error detail: {error_msg}")
                except (ValueError, KeyError) as json_err:
                    # Handle non-JSON error responses
                    error_text = e.response.text[:500]  # Limit to 500 chars
                    self.logger.error(f"Non-JSON error response: {error_text}")
                    error_msg = f"API error (status {e.response.status_code}): {error_text}"
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
    
    def close(self):
        """Close the requests session when done."""
        if self.session:
            try:
                self.session.close()
                self.logger.debug("API client session closed")
            except Exception as e:
                self.logger.error(f"Error closing API client session: {e}")
    
    def get_health(self) -> Dict[str, Any]:
        """Get API health status."""
        try:
            return self.get("health")
        except Exception as e:
            self.logger.error(f"Error checking API health: {e}")
            return {"status": "error", "message": str(e)}
    
    # Workflow Management
    
    def get_available_workflows(self) -> List[Dict[str, Any]]:
        """Get available workflow configurations."""
        return self.get("api/v1/workflows/available")
    
    def start_workflow(self, workflow_id: str, initial_goal: str) -> Dict[str, Any]:
        """Start a new workflow execution."""
        return self.post("api/v1/workflows/start", json={
            "workflow_id": workflow_id,
            "initial_goal": initial_goal
        })
    
    def get_workflow_status(self, run_id: str) -> Dict[str, Any]:
        """Get status of a workflow execution."""
        return self.get(f"api/v1/workflows/{run_id}/status")
    
    def stop_workflow(self, run_id: str) -> Dict[str, Any]:
        """Stop a running workflow."""
        return self.post(f"api/v1/workflows/{run_id}/stop")
    
    def get_active_workflows(self) -> List[Dict[str, Any]]:
        """Get all active workflows."""
        return self.get("api/v1/workflows/active")
    
    # Configuration Management
    
    def get_config_info(self) -> Dict[str, Any]:
        """Get configuration information."""
        return self.get("api/v1/configs/info")
    
    def get_agent_configs(self) -> Dict[str, Any]:
        """Get all agent configurations."""
        return self.get("api/v1/configs/agents")
    
    def update_agent_config(self, agent_id: str, config_update: Dict[str, Any]) -> Dict[str, Any]:
        """Update a specific agent configuration."""
        return self.put(f"api/v1/configs/agents/{agent_id}", json=config_update)
    
    def get_saved_adapters(self, agent_type: Optional[str] = None) -> List[Dict[str, Any]]:
        """Get saved DPO adapters, optionally filtered by agent type."""
        params = {"agent_type": agent_type} if agent_type else None
        return self.get("api/v1/configs/adapters", params=params)
    
    def download_config(self) -> str:
        """Download the agents.yaml configuration file as a string."""
        response = self.session.get(f"{self.base_url}/api/v1/configs/download", 
                                   timeout=(3, 10),
                                   headers={"Accept": "text/plain"})
        response.raise_for_status()
        return response.text
    
    def upload_config(self, config_content: str) -> Dict[str, Any]:
        """Upload a new agents.yaml configuration file."""
        files = {"file": ("agents.yaml", config_content, "text/plain")}
        response = self.session.post(f"{self.base_url}/api/v1/configs/upload", 
                                    files=files,
                                    timeout=(3, 10))
        response.raise_for_status()
        return response.json()
    
    # Log Management
    
    def get_logs(self, 
                 workflow_run_id: Optional[str] = None,
                 agent_name: Optional[str] = None,
                 agent_type: Optional[str] = None,
                 start_time: Optional[str] = None,
                 end_time: Optional[str] = None,
                 limit: int = 100,
                 offset: int = 0) -> Dict[str, Any]:
        """
        Get logs with optional filtering.
        
        Args:
            workflow_run_id: Optional workflow run ID filter
            agent_name: Optional agent name filter
            agent_type: Optional agent type filter
            start_time: Optional start time filter (ISO format)
            end_time: Optional end time filter (ISO format)
            limit: Maximum number of logs to return
            offset: Offset for pagination
            
        Returns:
            Dictionary with logs and total count
        """
        params = {
            "limit": limit,
            "offset": offset
        }
        if workflow_run_id:
            params["workflow_run_id"] = workflow_run_id
        if agent_name:
            params["agent_name"] = agent_name
        if agent_type:
            params["agent_type"] = agent_type
        if start_time:
            params["start_time"] = start_time
        if end_time:
            params["end_time"] = end_time
            
        return self.get("api/v1/logs", params=params)
    
    def get_log_entry(self, log_id: int) -> Dict[str, Any]:
        """Get a specific log entry by ID."""
        return self.get(f"api/v1/logs/{log_id}")
    
    def add_dpo_annotation(self, 
                          log_id: int, 
                          rating: int,
                          rationale: str, 
                          chosen_prompt: str, 
                          rejected_prompt: str,
                          dpo_context: str) -> Dict[str, Any]:
        """Add a DPO annotation to a log entry."""
        return self.post(f"api/v1/logs/{log_id}/annotate", json={
            "rating": rating,
            "rationale": rationale,
            "chosen_prompt": chosen_prompt,
            "rejected_prompt": rejected_prompt,
            "dpo_context": dpo_context
        })
    
    # DPO Training
    
    def get_dpo_ready_annotations(self, agent_type: str) -> Dict[str, Any]:
        """Get information about DPO-ready annotations for an agent type."""
        return self.get(f"api/v1/dpo/annotations/{agent_type}")
    
    def start_dpo_training(self, 
                          agent_type: str,
                          base_model_id: str,
                          adapter_name: str,
                          training_params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Start a DPO training job."""
        json_data = {
            "agent_type": agent_type,
            "base_model_id": base_model_id,
            "adapter_name": adapter_name
        }
        if training_params:
            json_data["training_params"] = training_params
            
        return self.post("api/v1/dpo/train", json=json_data)
    
    def get_dpo_job_status(self, job_id: str) -> Dict[str, Any]:
        """Get the status of a DPO training job."""
        return self.get(f"api/v1/dpo/jobs/{job_id}")
    
    def cancel_dpo_job(self, job_id: str) -> Dict[str, Any]:
        """Cancel a DPO training job."""
        return self.post(f"api/v1/dpo/jobs/{job_id}/cancel")