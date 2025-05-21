"""
DPO Training Control Panel View for CAMEL Extensions GUI.

This module implements the DPO Training Control Panel view with a setup form for
configuring and launching DPO training runs. It uses the API client for backend interaction.
"""
import streamlit as st
from datetime import datetime
from typing import Dict, Any, List, Optional


def handle_dpo_websocket_message(data: Dict[str, Any]):
    """
    Handle incoming WebSocket messages for DPO training.
    
    Args:
        data: Message data from WebSocket
    """
    message_type = data.get("type")
    
    if message_type == "status":
        # Update training status
        status_data = data.get("data", {})
        
        # Update progress information
        st.session_state.dpo_status_state = status_data.get("status", "running")
        if status_data.get("progress") is not None:
            st.session_state.dpo_progress_percent = int(status_data.get("progress", 0) * 100)
        
        # Check if training is complete
        if status_data.get("status") in ["completed", "failed"]:
            st.session_state.dpo_training_active = False
            st.session_state.dpo_active_job_id = None
            
            # Close WebSocket connection
            if "ws_manager" in st.session_state:
                ws_manager = st.session_state.ws_manager
                ws_manager.close_connection(f"dpo_{st.session_state.dpo_active_job_id}")
            
            # Signal completion 
            if status_data.get("status") == "completed":
                adapter_id = status_data.get("adapter_id")
                st.session_state.dpo_console_output.append(f"Training completed successfully. New adapter ID: {adapter_id}")
            else:
                st.session_state.dpo_console_output.append(f"Training failed: {status_data.get('error', 'Unknown error')}")
            
            # Trigger a rerun to update the UI
            st.rerun()
    
    elif message_type == "output":
        # Add output line to console
        output_line = data.get("data", "")
        if output_line:
            st.session_state.dpo_console_output.append(output_line)
            
            # Update progress text if it contains progress information
            if "progress" in output_line.lower():
                st.session_state.dpo_progress_text = output_line
            
            # Keep a reasonable cap on the number of lines
            if len(st.session_state.dpo_console_output) > 1000:
                st.session_state.dpo_console_output = st.session_state.dpo_console_output[-1000:]
            
            # Trigger a rerun to update the UI
            st.rerun()
    
    elif message_type == "error":
        # Handle error
        error_message = data.get("message", "Unknown error occurred")
        st.session_state.dpo_console_output.append(f"Error: {error_message}")
        st.session_state.dpo_status_state = "error"
        st.rerun()


def get_configurable_llm_models():
    """
    Fetch available LLM models from the API.
    
    Returns:
        list: A list of model ID strings.
    """
    try:
        api_client = st.session_state.api_client
        
        # Fetch agent configurations to extract model IDs
        agent_configs = api_client.get_all_agent_configs()
        
        # Extract unique model IDs
        model_ids = set()
        for agent_config in agent_configs.values():
            if "model_id" in agent_config and agent_config["model_id"]:
                model_ids.add(agent_config["model_id"])
        
        # Add some common models if the list is empty
        if not model_ids:
            model_ids = {
                "mistralai/Mistral-7B-Instruct-v0.2",
                "meta-llama/Llama-2-7b-chat-hf",
                "meta-llama/Llama-3-8b-instruct",
                "stabilityai/StableBeluga2",
                "google/gemma-7b-it"
            }
        
        return sorted(list(model_ids))
    
    except Exception as e:
        st.error(f"Error fetching configurable models: {str(e)}")
        return [
            "mistralai/Mistral-7B-Instruct-v0.2",
            "meta-llama/Llama-2-7b-chat-hf",
            "meta-llama/Llama-3-8b-instruct",
            "stabilityai/StableBeluga2",
            "google/gemma-7b-it"
        ]


def count_dpo_ready_annotations(agent_type: str) -> int:
    """
    Count the number of DPO-ready annotations for a specific agent type.
    
    Args:
        agent_type: Type of agent to count annotations for
    
    Returns:
        Number of available preference pairs for DPO training
    """
    try:
        api_client = st.session_state.api_client
        annotations = api_client.get_dpo_annotations(agent_type)
        return len(annotations)
    except Exception as e:
        st.error(f"Error fetching DPO annotations: {str(e)}")
        return 0


def start_dpo_training():
    """Start a new DPO training job using the API client."""
    try:
        api_client = st.session_state.api_client
        
        # Get form values
        agent_type = st.session_state.dpo_agent_to_train
        base_model_id = st.session_state.dpo_base_model_id
        adapter_name = st.session_state.dpo_new_adapter_name
        
        # Validate inputs
        if not agent_type:
            st.error("Agent type is required.")
            return
        
        if not base_model_id:
            st.error("Base model ID is required.")
            return
        
        if not adapter_name:
            st.error("Adapter name is required.")
            return
        
        # Set up training parameters (optimized for 8GB RTX 4060 Ti)
        training_args = {
            "learning_rate": 5e-5,
            "batch_size": 1,
            "gradient_accumulation_steps": 4,
            "epochs": 1,
            "quantization": "4bit",
            "target_modules": "q_proj,k_proj,v_proj,o_proj",
            "lora_rank": 16,
            "lora_alpha": 32,
            "lora_dropout": 0.05
        }
        
        # Start the training job
        job_id = api_client.start_dpo_training(agent_type, base_model_id, adapter_name, training_args)
        
        if job_id:
            # Update session state
            st.session_state.dpo_active_job_id = job_id
            st.session_state.dpo_training_active = True
            st.session_state.dpo_status_state = "running"
            st.session_state.dpo_progress_percent = 0
            st.session_state.dpo_progress_text = "Starting training..."
            st.session_state.dpo_console_output = ["Training job started..."]
            
            # Set up WebSocket connection for real-time updates
            ws_manager = st.session_state.ws_manager
            ws = ws_manager.create_dpo_training_connection(job_id, handle_dpo_websocket_message)
            
            st.success(f"DPO training job started with ID: {job_id}")
            st.rerun()
        else:
            st.error("Failed to start DPO training job.")
    
    except Exception as e:
        st.error(f"Error starting DPO training: {str(e)}")


def cancel_dpo_training():
    """Cancel the active DPO training job."""
    try:
        if not st.session_state.get("dpo_active_job_id"):
            st.warning("No active training job to cancel.")
            return
        
        api_client = st.session_state.api_client
        job_id = st.session_state.dpo_active_job_id
        
        success = api_client.cancel_training_job(job_id)
        
        if success:
            st.success("Training job cancelled successfully.")
            
            # Close WebSocket connection
            ws_manager = st.session_state.ws_manager
            ws_manager.close_connection(f"dpo_{job_id}")
            
            # Update session state
            st.session_state.dpo_training_active = False
            st.session_state.dpo_active_job_id = None
            st.session_state.dpo_status_state = "canceled"
            
            st.rerun()
        else:
            st.error("Failed to cancel training job.")
    
    except Exception as e:
        st.error(f"Error cancelling training job: {str(e)}")


def init_dpo_training_data():
    """Initialize session state variables for DPO training if they don't exist."""
    # Agent to train selection (MVP: Fixed to Proposer)
    if "dpo_agent_to_train" not in st.session_state:
        st.session_state.dpo_agent_to_train = "Proposer"
    
    # Base model selection
    if "dpo_base_model_id" not in st.session_state:
        models = get_configurable_llm_models()
        st.session_state.dpo_base_model_id = models[0] if models else "mistralai/Mistral-7B-Instruct-v0.2"
    
    # New adapter name
    if "dpo_new_adapter_name" not in st.session_state:
        current_date_time_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        st.session_state.dpo_new_adapter_name = f"proposer_dpo_{current_date_time_str}"
    
    # Training status variables
    if "dpo_active_job_id" not in st.session_state:
        st.session_state.dpo_active_job_id = None
    
    if "dpo_console_output" not in st.session_state:
        st.session_state.dpo_console_output = []


def render_dpo_training_setup_form():
    """
    Render the DPO training setup form with configuration options.
    
    This includes agent selection, model selection, adapter naming,
    and data source information.
    """
    st.header("DPO Training Setup")
    
    with st.form("dpo_training_form"):
        # Agent to Train (MVP: Fixed to Proposer)
        st.selectbox(
            "Select Agent to Train:", 
            options=["Proposer"], 
            disabled=True, 
            key="dpo_agent_to_train_form"
        )
        
        # Base Model Selection
        st.selectbox(
            "Select Base LLM Model ID:", 
            options=get_configurable_llm_models(), 
            help="The base model to fine-tune.",
            key="dpo_base_model_id_form"
        )
        
        # Adapter Naming
        current_date_time_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        st.text_input(
            "Name for New DPO Adapter:", 
            value=f"proposer_dpo_{current_date_time_str}", 
            help="e.g., 'proposer_goal_fix_v1'. Will be saved in models/adapters/",
            key="dpo_new_adapter_name_form"
        )
        
        # Data Source (Count DPO-ready annotations)
        agent_type = "proposer"  # Lowercase to match backend expectations
        annotation_count = count_dpo_ready_annotations(agent_type)
        
        st.info(f"Training will use all annotations currently marked for DPO for the '{st.session_state.dpo_agent_to_train}'.")
        st.markdown(f"Number of available preference pairs: **{annotation_count}**")
        
        if annotation_count == 0:
            st.warning("No DPO-ready annotations found. Please create some annotations in the Log Explorer before training.")
        
        # Training Parameters (Optimized defaults)
        st.markdown("#### Optimized Training Parameters (8GB 4060 Ti):")
        st.text("LoRA Rank (r): 16")
        st.text("Max Sequence Length: 1024")
        st.text("Number of Epochs: 1")
        st.text("Batch Size: 1, Grad Accumulation: 4")
        st.text("Quantization: 4-bit (Enabled)")
        st.text("Optimizer: PagedAdamW8bit (Enabled)")
        
        # Submit Button
        submit_disabled = annotation_count == 0 or st.session_state.get("dpo_training_active", False)
        submit_button = st.form_submit_button(
            "🚀 Start DPO Training Run",
            disabled=submit_disabled
        )
        
        if submit_button and not submit_disabled:
            # Update session state from form values
            st.session_state.dpo_agent_to_train = st.session_state.dpo_agent_to_train_form
            st.session_state.dpo_base_model_id = st.session_state.dpo_base_model_id_form
            st.session_state.dpo_new_adapter_name = st.session_state.dpo_new_adapter_name_form
            
            # Start training
            start_dpo_training()


def render_training_execution_monitoring():
    """
    Render the Training Execution & Monitoring Area UI.
    
    This area includes:
    - Status indicator with progress bar
    - Console output stream
    - Control buttons (e.g., cancel)
    """
    # Only render when dpo_training_active is True
    if not st.session_state.get('dpo_training_active', False):
        return
    
    st.header("Training Execution & Monitoring")
    
    # Display active job ID
    st.info(f"Active Job ID: {st.session_state.dpo_active_job_id}")
    
    # Status Indicator with Progress Bar
    status_state = st.session_state.get('dpo_status_state', 'running')
    progress_text = st.session_state.get('dpo_progress_text', 'Starting...')
    
    with st.status(
        f"Training Progress: {progress_text}",
        state=status_state,
        expanded=True
    ):
        st.progress(
            st.session_state.get('dpo_progress_percent', 0) / 100,
            text=progress_text
        )
    
    # Console Output
    st.subheader("Console Output")
    
    console_output = "\n".join(st.session_state.get('dpo_console_output', ["No output yet."]))
    
    st.code(
        console_output,
        language="bash",
        line_numbers=True
    )
    
    # Control Buttons
    if st.button(
        "🛑 Cancel Training Run",
        key="cancel_dpo_button_form",
        disabled=not st.session_state.get('dpo_training_active', False)
    ):
        cancel_dpo_training()


def render_dpo_training_view():
    """
    Render the DPO training control panel view.
    
    This function implements UI elements for DPO training configuration
    and execution monitoring. It uses the API client for backend interaction.
    """
    st.title("DPO Training Control Panel")
    
    # Initialize DPO training data if not already done
    init_dpo_training_data()
    
    # Render the DPO training setup form
    render_dpo_training_setup_form()
    
    # Render the Training Execution & Monitoring Area
    render_training_execution_monitoring()