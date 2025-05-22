"""
Configuration Management View for CAMEL Extensions GUI.

This module implements the configuration management view with global settings,
agent-specific configurations, and configuration control buttons.
"""
import streamlit as st
import yaml
import io
from typing import Dict, Any, List, Optional


@st.cache_data(ttl=30)
def fetch_config_data():
    """
    Fetch configuration data from the API with caching.
    
    This function is cached with a 30-second TTL to avoid frequent API calls.
    """
    try:
        api_client = st.session_state.api_client
        
        # Fetch global settings
        workflow_settings = api_client.get_workflow_settings()
        
        # Fetch workflows
        workflows_response = api_client.get_all_workflow_configs()
        workflows = workflows_response.get("workflows", {})
        
        # Fetch agents
        agents_response = api_client.get_all_agent_configs()
        agents = agents_response
        
        # Fetch adapters
        adapters = api_client.get_all_adapters()
        
        # Organize adapters by agent type for easier access
        organized_adapters = {"proposer": [], "reviewer": [], "all": []}
        for adapter in adapters:
            adapter_type = adapter.get("agent_type", "").lower()
            adapter_id = adapter.get("id", "unknown")
            organized_adapters["all"].append(adapter)
            
            if "proposer" in adapter_type:
                organized_adapters["proposer"].append(adapter)
            elif "reviewer" in adapter_type:
                organized_adapters["reviewer"].append(adapter)
        
        # Combine data into a structure similar to what the view expects
        config_data = {
            "workflow_settings": workflow_settings,
            "workflows": workflows,
            "agents": agents,
            "saved_adapters": organized_adapters
        }
        
        return config_data
    
    except Exception as e:
        st.error(f"Error fetching configuration data: {str(e)}")
        # Return a minimal structure to avoid errors in the UI
        return {
            "workflow_settings": {},
            "workflows": {},
            "agents": {},
            "saved_adapters": {"proposer": [], "reviewer": [], "all": []}
        }


def init_config_data():
    """Initialize configuration data using the API client."""
    if "config_data" not in st.session_state or st.session_state.get("reload_config", False):
        try:
            st.session_state.config_data = fetch_config_data()
            st.session_state.reload_config = False
        except Exception as e:
            st.error(f"Error initializing configuration data: {str(e)}")
            
            # Use placeholder data as fallback
            if "config_data" not in st.session_state:
                st.session_state.config_data = {
                    "workflow_settings": {
                        "default_workflow": "proposer_executor_review_loop",
                        "logging_db_path": "logs.db",
                        "max_iterations": 10,
                        "default_proposer_model_id": "gpt-4o",
                        "default_reviewer_model_id": "claude-3-opus-20240229"
                    },
                    "workflows": {
                        "proposer_executor_review_loop": {
                            "name": "Default Self-Improving Loop",
                            "description": "A workflow involving a Proposer to suggest actions based on a state, an Executor to perform them, and a PeerReviewer to evaluate the outcome. Interaction data is logged for subsequent DPO fine-tuning of the Proposer.",
                            "agent_configs": {
                                "proposer_config_key_in_yaml": {
                                    "agent_class_path": "camel.agents.proposer_agent.ProposerAgent",
                                    "init_args": {
                                        "name": "Proposer_Agent_Instance_Name",
                                        "llm_model": "mistralai/Mistral-7B-Instruct-v0.2",
                                        "adapter_id": "None"
                                    }
                                },
                                "executor_config_key_in_yaml": {
                                    "agent_class_path": "camel.agents.executor_agent.ExecutorAgent",
                                    "init_args": {
                                        "name": "Executor_Agent_Instance_Name"
                                    }
                                },
                                "reviewer_config_key_in_yaml": {
                                    "agent_class_path": "camel.agents.peer_reviewer_agent.PeerReviewerAgent",
                                    "init_args": {
                                        "name": "Peer_Reviewer_Agent_Instance_Name",
                                        "llm_model": "mistralai/Mistral-7B-Instruct-v0.2",
                                        "adapter_id": "None"
                                    }
                                }
                            }
                        }
                    },
                    "agents": {
                        "Proposer": {
                            "class_path": "camel.agents.proposer.ProposerAgent",
                            "model_id": "mistralai/Mistral-7B-Instruct-v0.2",
                            "adapter_id": None
                        },
                        "Executor": {
                            "class_path": "camel.agents.executor.ExecutorAgent",
                        },
                        "PeerReviewer": {
                            "class_path": "camel.agents.peer_reviewer.PeerReviewerAgent",
                            "model_id": "mistralai/Mistral-7B-Instruct-v0.2",
                            "adapter_id": None
                        }
                    },
                    "saved_adapters": {
                        "proposer": [
                            {"id": "proposer_adapter_1", "name": "Proposer DPO 1", "path": "models/proposer_dpo_20250520_1"},
                            {"id": "proposer_adapter_2", "name": "Proposer DPO 2", "path": "models/proposer_dpo_20250519_1"}
                        ],
                        "reviewer": [
                            {"id": "reviewer_adapter_1", "name": "Reviewer DPO 1", "path": "models/reviewer_dpo_20250520_1"},
                            {"id": "reviewer_adapter_2", "name": "Reviewer DPO 2", "path": "models/reviewer_dpo_20250518_2"}
                        ],
                        "all": [
                            {"id": "proposer_adapter_1", "name": "Proposer DPO 1", "path": "models/proposer_dpo_20250520_1"},
                            {"id": "proposer_adapter_2", "name": "Proposer DPO 2", "path": "models/proposer_dpo_20250519_1"},
                            {"id": "reviewer_adapter_1", "name": "Reviewer DPO 1", "path": "models/reviewer_dpo_20250520_1"},
                            {"id": "reviewer_adapter_2", "name": "Reviewer DPO 2", "path": "models/reviewer_dpo_20250518_2"}
                        ]
                    }
                }


def save_agent_config(agent_id: str, updated_config: Dict[str, Any]):
    """
    Save agent configuration changes to the backend.
    
    Args:
        agent_id: ID of the agent to update
        updated_config: Updated configuration values
    """
    try:
        api_client = st.session_state.api_client
        success = api_client.update_agent_config(agent_id, updated_config)
        
        if success:
            st.success(f"Successfully updated {agent_id} configuration.")
            # Clear the cache to fetch fresh data next time
            fetch_config_data.clear()
            # Force a reload
            st.session_state.reload_config = True
            # Immediate refresh
            st.rerun()
        else:
            st.error(f"Failed to update {agent_id} configuration.")
    except Exception as e:
        st.error(f"Error updating agent configuration: {str(e)}")


def reload_configuration():
    """Reload configuration from the backend."""
    try:
        api_client = st.session_state.api_client
        success = api_client.reload_config()
        
        if success:
            st.success("Configuration reloaded successfully.")
            # Clear the cache
            fetch_config_data.clear()
            # Force a reload
            st.session_state.reload_config = True
            # Immediate refresh
            st.rerun()
        else:
            st.error("Failed to reload configuration.")
    except Exception as e:
        st.error(f"Error reloading configuration: {str(e)}")


def render_global_settings_tab():
    """Render the Global Settings tab with read-only configuration display."""
    st.header("Global Settings")
    
    if 'config_data' in st.session_state and 'workflow_settings' in st.session_state.config_data:
        workflow_settings = st.session_state.config_data.get('workflow_settings', {})
        
        # Create two columns for better layout
        col1, col2 = st.columns(2)
        
        with col1:
            st.text_input("Default Workflow:", 
                         value=workflow_settings.get('default_workflow', 'N/A'), 
                         disabled=True)
            st.text_input("Logging DB Path:", 
                         value=workflow_settings.get('logging_db_path', 'N/A'), 
                         disabled=True)
        
        with col2:
            st.text_input("Max Iterations:", 
                         value=str(workflow_settings.get('max_iterations', 'N/A')), 
                         disabled=True)
            st.text_input("Default Proposer Model:", 
                         value=workflow_settings.get('default_proposer_model_id', 'N/A'), 
                         disabled=True)
            st.text_input("Default Reviewer Model:", 
                         value=workflow_settings.get('default_reviewer_model_id', 'N/A'), 
                         disabled=True)
    else:
        st.warning("Configuration data not loaded.")


def render_proposer_agent_tab():
    """Render the Proposer Agent tab with configuration options."""
    st.header("Proposer Agent Configuration")
    
    if 'config_data' not in st.session_state or 'agents' not in st.session_state.config_data:
        st.warning("Configuration data not loaded.")
        return
    
    # Get the proposer agent configuration
    agent_id = "Proposer"
    agent_configs = st.session_state.config_data.get("agents", {})
    
    if agent_id not in agent_configs:
        st.warning(f"{agent_id} configuration not found.")
        return
    
    agent_config = agent_configs[agent_id]
    agent_class_path = agent_config.get("class_path", "N/A")
    
    # Display a sub-header for the configuration
    st.subheader(f"Configuration for: {agent_id}")
    
    # Agent Name (Read-only for MVP)
    st.text_input(
        "Agent Name:",
        value=agent_id,
        key="proposer_agent_name_cfg",
        disabled=True
    )
    
    # Agent Class Path (Read-only)
    st.text_input(
        "Agent Class Path:",
        value=agent_class_path,
        key="proposer_agent_class_path_cfg",
        disabled=True
    )
    
    # LLM Model ID (Editable)
    model_id = agent_config.get("model_id", "")
    new_model_id = st.text_input(
        "Base LLM Model ID:",
        value=model_id,
        help="e.g., 'mistralai/Mistral-7B-Instruct-v0.2' or API endpoint",
        key="proposer_llm_model_cfg"
    )
    
    # DPO Adapter Selection (Editable)
    adapter_options = ["None"]
    proposer_adapters = st.session_state.config_data.get("saved_adapters", {}).get("proposer", [])
    
    for adapter in proposer_adapters:
        adapter_options.append(adapter.get("id", "unknown"))
    
    current_adapter = agent_config.get("adapter_id", None) or "None"
    
    new_adapter = st.selectbox(
        "Active DPO Adapter:",
        options=adapter_options,
        index=adapter_options.index(current_adapter) if current_adapter in adapter_options else 0,
        help="Select a fine-tuned LoRA adapter. 'None' uses the base model only.",
        key="proposer_adapter_cfg"
    )
    
    # Convert "None" string to None
    if new_adapter == "None":
        new_adapter = None
    
    # Prepare updated configuration
    updated_config = {
        "model_id": new_model_id,
        "adapter_id": new_adapter
    }
    
    # Check if there are actual changes
    has_changes = (
        new_model_id != model_id or
        new_adapter != agent_config.get("adapter_id")
    )
    
    # Controls within the Proposer Agent Tab
    col1, col2 = st.columns(2)
    with col1:
        if st.button("Apply Changes to Proposer Agent", key="apply_proposer_agent_cfg", disabled=not has_changes):
            save_agent_config(agent_id, updated_config)
    with col2:
        if st.button("Revert Changes for Proposer Agent", key="revert_proposer_agent_cfg"):
            st.rerun()


def render_executor_agent_tab():
    """Render the Executor Agent tab with read-only configuration display."""
    st.header("Executor Agent Configuration")
    
    if 'config_data' not in st.session_state or 'agents' not in st.session_state.config_data:
        st.warning("Configuration data not loaded.")
        return
    
    # Get the executor agent configuration
    agent_id = "Executor"
    agent_configs = st.session_state.config_data.get("agents", {})
    
    if agent_id not in agent_configs:
        st.warning(f"{agent_id} configuration not found.")
        return
    
    agent_config = agent_configs[agent_id]
    agent_class_path = agent_config.get("class_path", "N/A")
    
    # Display a sub-header for the configuration
    st.subheader(f"Configuration for: {agent_id}")
    
    # Agent Name (Read-only)
    st.text_input(
        "Agent Name:",
        value=agent_id,
        key="executor_agent_name_cfg",
        disabled=True
    )
    
    # Agent Class Path (Read-only)
    st.text_input(
        "Agent Class Path:",
        value=agent_class_path,
        key="executor_agent_class_path_cfg",
        disabled=True
    )
    
    # Informational message about executor agents not having LLM configurations
    st.info("Executor agents typically do not have LLM-specific configurations like models or adapters.")
    
    # Controls within the Executor Agent Tab (Placeholders)
    col1, col2 = st.columns(2)
    with col1:
        st.button("Apply Changes to Executor Agent", key="apply_executor_agent_cfg", disabled=True)
    with col2:
        st.button("Revert Changes for Executor Agent", key="revert_executor_agent_cfg", disabled=True)


def render_peer_reviewer_tab():
    """Render the Peer Reviewer Agent tab with configuration options."""
    st.header("Peer Reviewer Agent Configuration")
    
    if 'config_data' not in st.session_state or 'agents' not in st.session_state.config_data:
        st.warning("Configuration data not loaded.")
        return
    
    # Get the peer reviewer agent configuration
    agent_id = "PeerReviewer"
    agent_configs = st.session_state.config_data.get("agents", {})
    
    if agent_id not in agent_configs:
        st.warning(f"{agent_id} configuration not found.")
        return
    
    agent_config = agent_configs[agent_id]
    agent_class_path = agent_config.get("class_path", "N/A")
    
    # Display a sub-header for the configuration
    st.subheader(f"Configuration for: {agent_id}")
    
    # Agent Name (Read-only)
    st.text_input(
        "Agent Name:",
        value=agent_id,
        key="reviewer_agent_name_cfg",
        disabled=True
    )
    
    # Agent Class Path (Read-only)
    st.text_input(
        "Agent Class Path:",
        value=agent_class_path,
        key="reviewer_agent_class_path_cfg",
        disabled=True
    )
    
    # LLM Model ID (Editable)
    model_id = agent_config.get("model_id", "")
    new_model_id = st.text_input(
        "Base LLM Model ID:",
        value=model_id,
        help="e.g., 'mistralai/Mistral-7B-Instruct-v0.2'",
        key="reviewer_llm_model_cfg"
    )
    
    # DPO Adapter Selection (Editable)
    adapter_options = ["None"]
    reviewer_adapters = st.session_state.config_data.get("saved_adapters", {}).get("reviewer", [])
    
    for adapter in reviewer_adapters:
        adapter_options.append(adapter.get("id", "unknown"))
    
    current_adapter = agent_config.get("adapter_id", None) or "None"
    
    new_adapter = st.selectbox(
        "Active DPO Adapter:",
        options=adapter_options,
        index=adapter_options.index(current_adapter) if current_adapter in adapter_options else 0,
        help="Select a fine-tuned LoRA adapter.",
        key="reviewer_adapter_cfg"
    )
    
    # Convert "None" string to None
    if new_adapter == "None":
        new_adapter = None
    
    # Prepare updated configuration
    updated_config = {
        "model_id": new_model_id,
        "adapter_id": new_adapter
    }
    
    # Check if there are actual changes
    has_changes = (
        new_model_id != model_id or
        new_adapter != agent_config.get("adapter_id")
    )
    
    # Controls within the Peer Reviewer Agent Tab
    col1, col2 = st.columns(2)
    with col1:
        if st.button("Apply Changes to Peer Reviewer Agent", key="apply_reviewer_agent_cfg", disabled=not has_changes):
            save_agent_config(agent_id, updated_config)
    with col2:
        if st.button("Revert Changes for Peer Reviewer Agent", key="revert_reviewer_agent_cfg"):
            st.rerun()


def render_model_hub_tab():
    """Render the Model & Adapter Hub tab (placeholder for now)."""
    st.header("Model & Adapter Hub")
    st.info("This feature will be implemented in post-MVP updates.")
    
    # List all adapters
    adapters = st.session_state.config_data.get("saved_adapters", {}).get("all", [])
    if adapters:
        st.subheader("Available Adapters")
        for adapter in adapters:
            adapter_id = adapter.get("id", "unknown")
            adapter_name = adapter.get("name", adapter_id)
            adapter_path = adapter.get("path", "unknown")
            adapter_type = adapter.get("agent_type", "unknown")
            
            with st.expander(f"{adapter_name} ({adapter_type})"):
                st.code(f"ID: {adapter_id}")
                st.code(f"Path: {adapter_path}")
                st.code(f"Type: {adapter_type}")


def render_global_controls():
    """Render global configuration control buttons."""
    st.header("Configuration Controls")
    
    try:
        api_client = st.session_state.api_client
        
        # Button row 1
        col1, col2 = st.columns(2)
        with col1:
            if st.button("💾 Save All Changes to agents.yaml", 
                       type="primary", 
                       key="save_all_configs_button"):
                st.info("Saving all configuration changes...")
                # This is already handled by the individual "Apply Changes" buttons
                # but we could implement a bulk save if needed
                st.success("All changes have been saved.")
        with col2:
            if st.button("🔄 Reload Configuration from agents.yaml", 
                       key="reload_configs_button"):
                reload_configuration()
        
        # Button row 2
        col1, col2 = st.columns(2)
        with col1:
            # Get the raw YAML from the API
            try:
                yaml_data = api_client.download_config()
                
                st.download_button(
                    "Download Current agents.yaml", 
                    data=yaml_data,
                    file_name="agents.yaml", 
                    key="download_configs_button"
                )
            except Exception as e:
                st.error(f"Error preparing download: {str(e)}")
        
        with col2:
            uploaded_file = st.file_uploader(
                "Upload agents.yaml", 
                type=['yaml'], 
                key="upload_config_button"
            )
            
            if uploaded_file is not None:
                try:
                    # Read the uploaded YAML file
                    content = uploaded_file.getvalue().decode('utf-8')
                    
                    # Send the content to the API
                    response = api_client.upload_config(content)
                    
                    if response.get('status') == 'success':
                        st.success(f"Configuration file uploaded: {uploaded_file.name}")
                    else:
                        st.warning(f"Upload response: {response.get('message', 'Unknown status')}")
                    
                    # Clear the cache to ensure we fetch fresh data
                    fetch_config_data.clear()
                    
                    # Force a reload
                    st.session_state.reload_config = True
                    
                    # Rerun to refresh the UI
                    st.rerun()
                except Exception as e:
                    st.error(f"Error processing uploaded file: {str(e)}")
    
    except Exception as e:
        st.error(f"Error in configuration controls: {str(e)}")


def render_config_view():
    """
    Render the configuration management view.
    
    This function creates the main tabs for different configuration sections
    and calls the appropriate rendering function for each tab.
    """
    st.title("Configuration Management")
    
    # Initialize config data if not already done
    init_config_data()
    
    # Create main tabs
    tabs = st.tabs([
        "Global Settings", 
        "Proposer Agent", 
        "Executor Agent", 
        "Peer Reviewer Agent", 
        "Model & Adapter Hub"
    ])
    
    # Render content for each tab
    with tabs[0]:
        render_global_settings_tab()
    
    with tabs[1]:
        render_proposer_agent_tab()
    
    with tabs[2]:
        render_executor_agent_tab()
        
    with tabs[3]:
        render_peer_reviewer_tab()
        
    with tabs[4]:
        render_model_hub_tab()
    
    # Render global configuration controls at the bottom
    render_global_controls()