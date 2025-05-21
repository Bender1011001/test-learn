from fastapi import APIRouter, Depends, HTTPException, UploadFile, File
from fastapi.responses import FileResponse
from typing import Dict, List, Optional, Any
from pydantic import BaseModel, Field
import tempfile
import os
from loguru import logger

from ...core.services.config_manager import ConfigManager, AgentConfig, WorkflowConfig
from ...db.base import get_db

# Pydantic models for request/response validation

class UpdateAgentConfigRequest(BaseModel):
    """Request model for updating agent configuration"""
    model_id: Optional[str] = None
    adapter_id: Optional[str] = None
    parameters: Optional[Dict[str, Any]] = None


class SavedAdapterResponse(BaseModel):
    """Response model for saved adapter"""
    id: str
    name: str
    base_model_id: str
    creation_date: str
    path: str
    agent_type: str
    description: Optional[str] = None


# Create router
router = APIRouter()


# Dependency to get config manager
def get_config_manager(db=Depends(get_db)):
    from ..dependencies import get_services
    config_manager, workflow_manager, db_manager, dpo_trainer = get_services(db)
    return config_manager


@router.get("/agents", response_model=Dict[str, Any])
async def get_all_agent_configs(
    config_manager: ConfigManager = Depends(get_config_manager)
):
    """Get all agent configurations"""
    config = config_manager.config
    
    # Convert to dict for JSON serialization
    agents = {
        agent_id: agent.model_dump() 
        for agent_id, agent in config.agents.items()
    }
    
    return {"agents": agents}


@router.get("/agents/{agent_id}", response_model=Dict[str, Any])
async def get_agent_config(
    agent_id: str,
    config_manager: ConfigManager = Depends(get_config_manager)
):
    """Get configuration for a specific agent"""
    agent = config_manager.get_agent_config(agent_id)
    
    if not agent:
        raise HTTPException(status_code=404, detail=f"Agent {agent_id} not found")
    
    return agent.model_dump()


@router.put("/agents/{agent_id}", response_model=Dict[str, bool])
async def update_agent_config(
    agent_id: str,
    request: UpdateAgentConfigRequest,
    config_manager: ConfigManager = Depends(get_config_manager)
):
    """Update configuration for a specific agent"""
    # Convert request model to dict
    updated_config = request.model_dump(exclude_none=True)
    
    success = config_manager.update_agent_config(agent_id, updated_config)
    
    if not success:
        raise HTTPException(
            status_code=400, 
            detail=f"Failed to update agent {agent_id}"
        )
    
    # Save the configuration to file
    success = config_manager.save_config()
    
    if not success:
        raise HTTPException(
            status_code=500, 
            detail=f"Failed to save configuration to file"
        )
    
    return {"success": True}


@router.get("/workflows", response_model=Dict[str, Any])
async def get_all_workflow_configs(
    config_manager: ConfigManager = Depends(get_config_manager)
):
    """Get all workflow configurations"""
    config = config_manager.config
    
    # Convert to dict for JSON serialization
    workflows = {
        workflow_id: workflow.model_dump() 
        for workflow_id, workflow in config.workflows.items()
    }
    
    return {"workflows": workflows}


@router.get("/workflows/{workflow_id}", response_model=Dict[str, Any])
async def get_workflow_config(
    workflow_id: str,
    config_manager: ConfigManager = Depends(get_config_manager)
):
    """Get configuration for a specific workflow"""
    workflow = config_manager.get_workflow_config(workflow_id)
    
    if not workflow:
        raise HTTPException(status_code=404, detail=f"Workflow {workflow_id} not found")
    
    return workflow.model_dump()


@router.get("/settings", response_model=Dict[str, Any])
async def get_workflow_settings(
    config_manager: ConfigManager = Depends(get_config_manager)
):
    """Get global workflow settings"""
    config = config_manager.config
    
    return config.workflow_settings.model_dump()


@router.get("/adapters", response_model=List[SavedAdapterResponse])
async def get_all_adapters(
    agent_type: Optional[str] = None,
    config_manager: ConfigManager = Depends(get_config_manager)
):
    """
    Get all saved DPO adapters
    
    Optionally filter by agent_type
    """
    config = config_manager.config
    
    if not hasattr(config, 'saved_adapters'):
        return []
    
    adapters = list(config.saved_adapters.values())
    
    if agent_type:
        adapters = [
            adapter for adapter in adapters
            if adapter.agent_type.lower() == agent_type.lower()
        ]
    
    return adapters


@router.post("/reload", response_model=Dict[str, bool])
async def reload_config(
    config_manager: ConfigManager = Depends(get_config_manager)
):
    """Reload configuration from file"""
    try:
        config_manager.reload_config()
        return {"success": True}
    except Exception as e:
        logger.error(f"Error reloading configuration: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to reload configuration: {str(e)}"
        )


@router.get("/download")
async def download_config(
    config_manager: ConfigManager = Depends(get_config_manager)
):
    """Download the agents.yaml file"""
    return FileResponse(
        config_manager.config_path,
        media_type="application/x-yaml",
        filename="agents.yaml"
    )


@router.post("/upload", response_model=Dict[str, bool])
async def upload_config(
    file: UploadFile = File(...),
    config_manager: ConfigManager = Depends(get_config_manager)
):
    """Upload and replace the agents.yaml file"""
    if not file.filename.endswith(('.yaml', '.yml')):
        raise HTTPException(status_code=400, detail="File must be a YAML file")
    
    # Create a temporary file
    with tempfile.NamedTemporaryFile(delete=False, suffix='.yaml') as temp_file:
        temp_path = temp_file.name
        # Write uploaded content to temp file
        content = await file.read()
        temp_file.write(content)
    
    try:
        # Validate by trying to load the config
        from ruamel.yaml import YAML
        yaml = YAML()
        with open(temp_path, 'r') as f:
            yaml_content = yaml.load(f)
        
        # Temporary instance to validate
        from ...core.services.config_manager import AgentsConfig
        config = AgentsConfig(**yaml_content)
        
        # If validation passes, copy to the real config path
        import shutil
        shutil.copy(temp_path, config_manager.config_path)
        
        # Reload the configuration
        config_manager.reload_config()
        
        return {"success": True}
    
    except Exception as e:
        logger.error(f"Error uploading configuration: {str(e)}")
        raise HTTPException(
            status_code=400,
            detail=f"Invalid configuration file: {str(e)}"
        )
    
    finally:
        # Clean up the temp file
        if os.path.exists(temp_path):
            os.unlink(temp_path)