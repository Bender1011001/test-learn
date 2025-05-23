from ruamel.yaml import YAML
from pathlib import Path
from pydantic import BaseModel, Field, ValidationError
from typing import Dict, List, Optional, Any, Union
import os
from loguru import logger

# Define Pydantic models for config validation

class AgentConfig(BaseModel):
    """Pydantic model for agent configuration"""
    name: str
    class_path: str
    model_id: Optional[str] = None
    adapter_id: Optional[str] = None
    parameters: Optional[Dict[str, Any]] = Field(default_factory=dict)


class WorkflowSettings(BaseModel):
    """Pydantic model for workflow settings"""
    default_proposer_model_id: Optional[str] = None
    default_reviewer_model_id: Optional[str] = None
    max_iterations: int = 10


class WorkflowConfig(BaseModel):
    """Pydantic model for workflow configuration"""
    description: str
    agent_sequence: List[str]
    settings: Optional[WorkflowSettings] = Field(default_factory=WorkflowSettings)


class SavedAdapter(BaseModel):
    """Pydantic model for saved DPO adapter"""
    id: str
    name: str
    base_model_id: str
    creation_date: str
    path: str
    agent_type: str
    description: Optional[str] = None


class AgentsConfig(BaseModel):
    """Root Pydantic model for agents.yaml"""
    workflow_settings: WorkflowSettings
    agents: Dict[str, AgentConfig]
    workflows: Dict[str, WorkflowConfig]
    saved_adapters: Optional[Dict[str, SavedAdapter]] = Field(default_factory=dict)


class ConfigManager:
    """Manager for loading and saving agent configurations"""
    
    def __init__(self, config_path: str = "configs/agents.yaml"):
        """Initialize with path to agents.yaml"""
        self.config_path = Path(config_path)
        self.yaml = YAML()
        self.yaml.preserve_quotes = True
        self.yaml.indent(mapping=2, sequence=4, offset=2)
        
        # Load initial config
        self.reload_config()

    def reload_config(self) -> AgentsConfig:
        """Reload configuration from file"""
        try:
            if not self.config_path.exists():
                logger.warning(f"Config file {self.config_path} not found")
                # Create a minimal default config
                self.config = AgentsConfig(
                    workflow_settings=WorkflowSettings(),
                    agents={},
                    workflows={},
                )
                return self.config
            
            # Load YAML file
            with open(self.config_path, 'r') as f:
                config_dict = self.yaml.load(f)
            
            # Validate with Pydantic
            self.config = AgentsConfig(**config_dict)
            logger.info(f"Loaded configuration from {self.config_path}")
            return self.config
            
        except ValidationError as e:
            logger.error(f"Configuration validation error: {e}")
            raise
        except Exception as e:
            logger.error(f"Error loading configuration: {e}")
            raise
    
    async def reload_config_async(self) -> AgentsConfig:
        """
        Async version of reload_config
        
        Returns:
            Loaded configuration
        """
        # This is a simple wrapper around the sync method since it's just file I/O
        # In a real implementation, this would use aiofiles for true async file I/O
        return self.reload_config()

    def save_config(self) -> bool:
        """Save configuration to file"""
        try:
            # Create parent directories if they don't exist
            os.makedirs(self.config_path.parent, exist_ok=True)
            
            # Convert Pydantic model to dict
            config_dict = self.config.model_dump(exclude_none=True)
            
            # Save to YAML file
            with open(self.config_path, 'w') as f:
                self.yaml.dump(config_dict, f)
            
            logger.info(f"Saved configuration to {self.config_path}")
            return True
        except Exception as e:
            logger.error(f"Error saving configuration: {e}")
            return False
    
    async def save_config_async(self) -> bool:
        """
        Async version of save_config
        
        Returns:
            True if configuration was saved successfully, False otherwise
        """
        # This is a simple wrapper around the sync method since it's just file I/O
        # In a real implementation, this would use aiofiles for true async file I/O
        return self.save_config()

    def get_agent_config(self, agent_id: str) -> Optional[AgentConfig]:
        """Get configuration for a specific agent"""
        return self.config.agents.get(agent_id)

    def update_agent_config(self, agent_id: str, updated_config: Dict[str, Any]) -> bool:
        """Update configuration for a specific agent"""
        if agent_id not in self.config.agents:
            logger.error(f"Agent {agent_id} not found in configuration")
            return False
    
    async def add_saved_adapter_async(self, adapter: SavedAdapter) -> bool:
        """
        Async version of add_saved_adapter
        
        Args:
            adapter: SavedAdapter object to add
            
        Returns:
            True if adapter was added successfully, False otherwise
        """
        # This is a simple wrapper around the sync method since it's just in-memory operations
        return self.add_saved_adapter(adapter)
    
    async def update_agent_config_async(self, agent_id: str, updated_config: Dict[str, Any]) -> bool:
        """
        Async version of update_agent_config
        
        Args:
            agent_id: ID of the agent to update
            updated_config: Dictionary with updated configuration values
            
        Returns:
            True if agent configuration was updated successfully, False otherwise
        """
        # This is a simple wrapper around the sync method since it's just in-memory operations
        return self.update_agent_config(agent_id, updated_config)
        
        try:
            # Get existing config
            existing_config = self.config.agents[agent_id].model_dump()
            
            # Update with new values
            existing_config.update(updated_config)
            
            # Validate and update
            self.config.agents[agent_id] = AgentConfig(**existing_config)
            return True
        except ValidationError as e:
            logger.error(f"Invalid agent configuration: {e}")
            return False

    def get_workflow_config(self, workflow_id: str) -> Optional[WorkflowConfig]:
        """Get configuration for a specific workflow"""
        return self.config.workflows.get(workflow_id)
    
    def get_all_workflow_ids(self) -> List[str]:
        """Get list of all workflow IDs"""
        return list(self.config.workflows.keys())
    
    def add_saved_adapter(self, adapter: SavedAdapter) -> bool:
        """Add a new DPO adapter to the configuration"""
        try:
            self.config.saved_adapters[adapter.id] = adapter
            return True
        except Exception as e:
            logger.error(f"Error adding adapter: {e}")
            return False

    def get_adapters_for_agent_type(self, agent_type: str) -> List[SavedAdapter]:
        """Get all adapters for a specific agent type"""
        return [
            adapter for adapter in self.config.saved_adapters.values()
            if adapter.agent_type.lower() == agent_type.lower()
        ]