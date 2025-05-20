"""
DPO Training Control Panel View for camel_ext GUI.

This module implements the DPO Training Control Panel view with a setup form for
configuring and launching DPO training runs. For MVP, this uses placeholder data
instead of actual training execution.
"""
import streamlit as st
from datetime import datetime


def get_configurable_llm_models_placeholder():
    """
    Placeholder function to simulate fetching available LLM models from agents.yaml.
    
    Returns:
        list: A list of model ID strings.
    """
    return [
        "mistralai/Mistral-7B-Instruct-v0.2",
        "meta-llama/Llama-2-7b-chat-hf",
        "meta-llama/Llama-3-8b-instruct",
        "stabilityai/StableBeluga2",
        "google/gemma-7b-it"
    ]


def count_dpo_ready_annotations_placeholder():
    """
    Placeholder function to simulate counting DPO-ready annotations.
    
    Returns:
        int: Number of available preference pairs for DPO training.
    """
    return 10


def init_dpo_training_data():
    """Initialize session state variables for DPO training if they don't exist."""
    # Agent to train selection (MVP: Fixed to Proposer)
    if "dpo_agent_to_train" not in st.session_state:
        st.session_state.dpo_agent_to_train = "Proposer"
    
    # Base model selection
    if "dpo_base_model_id" not in st.session_state:
        st.session_state.dpo_base_model_id = "mistralai/Mistral-7B-Instruct-v0.2"
    
    # New adapter name
    if "dpo_new_adapter_name" not in st.session_state:
        current_date_time_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        st.session_state.dpo_new_adapter_name = f"proposer_dpo_{current_date_time_str}"


def render_dpo_training_setup_form():
    """
    Render the DPO training setup form with configuration options.
    
    For MVP, this includes fixed training parameters and placeholder data.
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
            options=get_configurable_llm_models_placeholder(), 
            help="The base model to fine-tune.",
            key="dpo_base_model_id_form"
        )
        
        # Adapter Naming
        current_date_time_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        st.text_input(
            "Name for New DPO Adapter:", 
            value=f"proposer_dpo_{current_date_time_str}", 
            help="e.g., 'proposer_goal_fix_v1'. Will be saved in 'camel_ext/models/'",
            key="dpo_new_adapter_name_form"
        )
        
        # Data Source (MVP: Simplified Info)
        st.info(f"Training will use all annotations currently 'Marked for DPO' for the '{st.session_state.dpo_agent_to_train}'.")
        st.markdown(f"Number of available preference pairs: {count_dpo_ready_annotations_placeholder()}")
        
        # Training Parameters (MVP: Displayed as read-only, optimized defaults)
        st.markdown("#### Optimized Training Parameters (8GB 4060 Ti):")
        st.text("LoRA Rank (r): 16 (Fixed for MVP)")
        st.text("Max Sequence Length: 1024 (Fixed for MVP)")
        st.text("Number of Epochs: 1 (Fixed for MVP)")
        st.text("Batch Size: 1, Grad Accumulation: 4 (Fixed for MVP)")
        st.text("Quantization: 4-bit (Enabled)")
        st.text("Optimizer: PagedAdamW8bit (Enabled)")
        
        # Submit Button
        st.form_submit_button("🚀 Start DPO Training Run")


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
    
    # Status Indicator with Progress Bar
    with st.status(
        f"Training Progress: {st.session_state.get('dpo_progress_text', 'Starting...')}",
        state=st.session_state.get('dpo_status_state', 'running'),
        expanded=True
    ):
        st.progress(
            st.session_state.get('dpo_progress_percent', 0) / 100,
            text=st.session_state.get('dpo_progress_text', 'Waiting...')
        )
    
    # Console Output
    st.subheader("Console Output")
    st.code(
        st.session_state.get('dpo_console_output', "No output yet."),
        language="bash",
        line_numbers=True
    )
    
    # Control Buttons
    st.button(
        "🛑 Cancel Training Run",
        key="cancel_dpo_button_form",
        disabled=False  # For MVP, this doesn't actually do anything
    )


def render_dpo_training_view():
    """
    Render the DPO training control panel view.
    
    This function implements UI elements for DPO training configuration.
    For MVP, this uses placeholder data instead of actual training execution.
    """
    st.title("DPO Training Control Panel")
    
    # Initialize DPO training data if not already done
    init_dpo_training_data()
    
    # Render the DPO training setup form
    render_dpo_training_setup_form()
    
    # Render the Training Execution & Monitoring Area
    render_training_execution_monitoring()

    # For development only - toggle to show/hide the monitoring UI
    # Comment out before final deployment
    with st.expander("Development Controls (Remove in Production)"):
        if st.checkbox("Simulate DPO Training Active (for UI dev)",
                     key="dev_toggle_dpo_active",
                     value=st.session_state.get('dpo_training_active', False)):
            st.session_state.dpo_training_active = True
        else:
            st.session_state.dpo_training_active = False