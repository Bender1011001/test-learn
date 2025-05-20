"""
Configuration Management View for camel_ext GUI.

This module implements the configuration management view with global settings,
agent-specific configurations, and configuration control buttons.
"""
import streamlit as st


def init_config_data():
    """Initialize placeholder configuration data if not already in session state."""
    if "config_data" not in st.session_state:
        # This is a placeholder for the data that would normally be loaded from configs/agents.yaml
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
                                # Other non-LLM specific args not needed for MVP UI
                            }
                        },
                        "reviewer_config_key_in_yaml": {
                            "agent_class_path": "camel.agents.peer_reviewer_agent.PeerReviewerAgent",
                            "init_args": {
                                "name": "Peer_Reviewer_Agent_Instance_Name",
                                "llm_model": "mistralai/Mistral-7B-Instruct-v0.2",
                                "adapter_id": "None"
                                # Other args not needed for MVP UI
                            }
                        }
                    }
                }
            },
            "saved_adapters": {
                "proposer_adapter_1": "models/proposer_dpo_20250520_1",
                "proposer_adapter_2": "models/proposer_dpo_20250519_1",
                "proposer_dpo_model_1": "models/proposer_full_model_20250515",
                "reviewer_adapter_1": "models/reviewer_dpo_20250520_1",
                "reviewer_adapter_2": "models/reviewer_dpo_20250518_2"
            }
        }


def render_global_settings_tab():
    """Render the Global Settings tab with read-only configuration display."""
    st.header("Global Settings")
    
    if 'config_data' in st.session_state and 'workflow_settings' in st.session_state.config_data:
        workflow_settings = st.session_state.config_data['workflow_settings']
        
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
    
    if 'config_data' not in st.session_state or 'workflows' not in st.session_state.config_data:
        st.warning("Configuration data not loaded.")
        return
        
    # For MVP, directly access a predefined proposer agent config
    workflow_key = "proposer_executor_review_loop"
    agent_config_key = "proposer_config_key_in_yaml"
    
    if (workflow_key not in st.session_state.config_data["workflows"] or
        "agent_configs" not in st.session_state.config_data["workflows"][workflow_key] or
        agent_config_key not in st.session_state.config_data["workflows"][workflow_key]["agent_configs"]):
        st.warning(f"Proposer agent configuration not found for workflow: {workflow_key}")
        return
        
    # Get the proposer agent configuration
    proposer_config = st.session_state.config_data["workflows"][workflow_key]["agent_configs"][agent_config_key]
    proposer_agent_class_path = proposer_config.get("agent_class_path", "N/A")
    proposer_init_args = proposer_config.get("init_args", {})
    
    # Display a sub-header for the configuration
    st.subheader(f"Configuration for: Proposer")
    
    # Agent Name (Read-only for MVP)
    st.text_input(
        "Agent Name (in workflow):",
        value=proposer_init_args.get('name', 'N/A'),
        key="proposer_agent_name_cfg",
        disabled=True
    )
    
    # Agent Class Path (Read-only)
    st.text_input(
        "Agent Class Path:",
        value=proposer_agent_class_path,
        key="proposer_agent_class_path_cfg",
        disabled=True
    )
    
    # LLM Model ID (Editable)
    st.text_input(
        "Base LLM Model ID:",
        value=proposer_init_args.get('llm_model', ''),
        help="e.g., 'mistralai/Mistral-7B-Instruct-v0.2' or API endpoint",
        key="proposer_llm_model_cfg"
    )
    
    # DPO Adapter Selection (Editable)
    adapter_options = ["None"] + list(st.session_state.config_data.get('saved_adapters', {}).keys())
    current_adapter = proposer_init_args.get('adapter_id', "None")
    
    st.selectbox(
        "Active DPO Adapter:",
        options=adapter_options,
        index=adapter_options.index(current_adapter) if current_adapter in adapter_options else 0,
        help="Select a fine-tuned LoRA adapter. 'None' uses the base model only.",
        key="proposer_adapter_cfg"
    )
    
    # Controls within the Proposer Agent Tab (Placeholders)
    col1, col2 = st.columns(2)
    with col1:
        st.button("Apply Changes to Proposer Agent", key="apply_proposer_agent_cfg")
    with col2:
        st.button("Revert Changes for Proposer Agent", key="revert_proposer_agent_cfg")



def render_executor_agent_tab():
    """Render the Executor Agent tab with read-only configuration display."""
    st.header("Executor Agent Configuration")
    
    if 'config_data' not in st.session_state or 'workflows' not in st.session_state.config_data:
        st.warning("Configuration data not loaded.")
        return
        
    # For MVP, directly access a predefined executor agent config
    workflow_key = "proposer_executor_review_loop"
    agent_config_key = "executor_config_key_in_yaml"
    
    if (workflow_key not in st.session_state.config_data["workflows"] or
        "agent_configs" not in st.session_state.config_data["workflows"][workflow_key] or
        agent_config_key not in st.session_state.config_data["workflows"][workflow_key]["agent_configs"]):
        st.warning(f"Executor agent configuration not found for workflow: {workflow_key}")
        return
        
    # Get the executor agent configuration
    executor_config = st.session_state.config_data["workflows"][workflow_key]["agent_configs"][agent_config_key]
    executor_agent_class_path = executor_config.get("agent_class_path", "N/A")
    executor_init_args = executor_config.get("init_args", {})
    
    # Display a sub-header for the configuration
    st.subheader(f"Configuration for: Executor")
    
    # Agent Name (Read-only)
    st.text_input(
        "Agent Name (in workflow):",
        value=executor_init_args.get('name', 'N/A'),
        key="executor_agent_name_cfg",
        disabled=True
    )
    
    # Agent Class Path (Read-only)
    st.text_input(
        "Agent Class Path:",
        value=executor_agent_class_path,
        key="executor_agent_class_path_cfg",
        disabled=True
    )
    
    # Informational message about executor agents not having LLM configurations
    st.info("Executor agents typically do not have LLM-specific configurations like models or adapters.")
    
    # Controls within the Executor Agent Tab (Placeholders)
    col1, col2 = st.columns(2)
    with col1:
        st.button("Apply Changes to Executor Agent", key="apply_executor_agent_cfg")
    with col2:
        st.button("Revert Changes for Executor Agent", key="revert_executor_agent_cfg")

def render_peer_reviewer_tab():
    """Render the Peer Reviewer Agent tab with configuration options."""
    st.header("Peer Reviewer Agent Configuration")
    
    if 'config_data' not in st.session_state or 'workflows' not in st.session_state.config_data:
        st.warning("Configuration data not loaded.")
        return
        
    # For MVP, directly access a predefined peer reviewer agent config
    workflow_key = "proposer_executor_review_loop"
    agent_config_key = "reviewer_config_key_in_yaml"
    
    if (workflow_key not in st.session_state.config_data["workflows"] or
        "agent_configs" not in st.session_state.config_data["workflows"][workflow_key] or
        agent_config_key not in st.session_state.config_data["workflows"][workflow_key]["agent_configs"]):
        st.warning(f"Peer Reviewer agent configuration not found for workflow: {workflow_key}")
        return
        
    # Get the peer reviewer agent configuration
    reviewer_config = st.session_state.config_data["workflows"][workflow_key]["agent_configs"][agent_config_key]
    reviewer_agent_class_path = reviewer_config.get("agent_class_path", "N/A")
    reviewer_init_args = reviewer_config.get("init_args", {})
    
    # Display a sub-header for the configuration
    st.subheader(f"Configuration for: Peer Reviewer")
    
    # Agent Name (Read-only)
    st.text_input(
        "Agent Name (in workflow):",
        value=reviewer_init_args.get('name', 'N/A'),
        key="reviewer_agent_name_cfg",
        disabled=True
    )
    
    # Agent Class Path (Read-only)
    st.text_input(
        "Agent Class Path:",
        value=reviewer_agent_class_path,
        key="reviewer_agent_class_path_cfg",
        disabled=True
    )
    
    # LLM Model ID (Editable)
    st.text_input(
        "Base LLM Model ID:",
        value=reviewer_init_args.get('llm_model', ''),
        help="e.g., 'mistralai/Mistral-7B-Instruct-v0.2'",
        key="reviewer_llm_model_cfg"
    )
    
    # DPO Adapter Selection (Editable)
    adapter_options = ["None"] + list(st.session_state.config_data.get('saved_adapters', {}).keys())
    current_adapter = reviewer_init_args.get('adapter_id', "None")
    
    st.selectbox(
        "Active DPO Adapter:",
        options=adapter_options,
        index=adapter_options.index(current_adapter) if current_adapter in adapter_options else 0,
        help="Select a fine-tuned LoRA adapter.",
        key="reviewer_adapter_cfg"
    )
    
    # Controls within the Peer Reviewer Agent Tab (Placeholders)
    col1, col2 = st.columns(2)
    with col1:
        st.button("Apply Changes to Peer Reviewer Agent", key="apply_reviewer_agent_cfg")
    with col2:
        st.button("Revert Changes for Peer Reviewer Agent", key="revert_reviewer_agent_cfg")

def render_model_hub_tab():
    """Render the Model & Adapter Hub tab (placeholder for now)."""
    st.header("Model & Adapter Hub")
    st.info("This feature will be implemented in post-MVP updates.")


def render_global_controls():
    """Render global configuration control buttons."""
    st.header("Configuration Controls")
    
    # Button row 1
    col1, col2 = st.columns(2)
    with col1:
        st.button("💾 Save All Changes to agents.yaml", 
                 type="primary", 
                 key="save_all_configs_button")
    with col2:
        st.button("🔄 Reload Configuration from agents.yaml", 
                 key="reload_configs_button")
    
    # Button row 2
    col1, col2 = st.columns(2)
    with col1:
        st.download_button("Download Current agents.yaml", 
                          data="""
workflow_settings:
 default_workflow: proposer_executor_review_loop
 logging_db_path: logs.db
 max_iterations: 10
 default_proposer_model_id: gpt-4o
 default_reviewer_model_id: claude-3-opus-20240229

workflows:
 proposer_executor_review_loop:
   name: Default Self-Improving Loop
   description: A workflow involving a Proposer to suggest actions based on a state, an Executor to perform them, and a PeerReviewer to evaluate the outcome.
   agent_configs:
     proposer_config_key_in_yaml:
       agent_class_path: camel.agents.proposer_agent.ProposerAgent
       init_args:
         name: Proposer_Agent_Instance_Name
         llm_model: mistralai/Mistral-7B-Instruct-v0.2
         adapter_id: None
     executor_config_key_in_yaml:
       agent_class_path: camel.agents.executor_agent.ExecutorAgent
       init_args:
         name: Executor_Agent_Instance_Name
     reviewer_config_key_in_yaml:
       agent_class_path: camel.agents.peer_reviewer_agent.PeerReviewerAgent
       init_args:
         name: Peer_Reviewer_Agent_Instance_Name
         llm_model: mistralai/Mistral-7B-Instruct-v0.2
         adapter_id: None

saved_adapters:
 proposer_adapter_1: models/proposer_dpo_20250520_1
 proposer_adapter_2: models/proposer_dpo_20250519_1
 proposer_dpo_model_1: models/proposer_full_model_20250515
 reviewer_adapter_1: models/reviewer_dpo_20250520_1
 reviewer_adapter_2: models/reviewer_dpo_20250518_2
""",
                          file_name="agents_custom.yaml", 
                          key="download_configs_button")
    with col2:
        st.file_uploader("Upload agents.yaml", 
                        type=['yaml'], 
                        key="upload_config_button")


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
        "Model & Adapter Hub (Post-MVP)"
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