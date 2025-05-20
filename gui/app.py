"""
Main application file for the camel_ext GUI.

This is the entry point for the Streamlit application that renders the camel_ext GUI.
"""
import streamlit as st
from views.dashboard_view import render_dashboard_view, render_sidebar
from views.config_view import render_config_view, init_config_data
from views.log_explorer_view import render_log_explorer_view
from views.dpo_training_view import render_dpo_training_view
from views.settings_view import render_settings_view


def init_session_state():
    """Initialize session state variables if they don't exist."""
    if "workflow_initial_goal" not in st.session_state:
        st.session_state.workflow_initial_goal = ""
    
    if "selected_workflow_name" not in st.session_state:
        st.session_state.selected_workflow_name = "proposer_executor_review_loop"
    
    # Initialize live log entries for interaction feed
    if "live_log_entries" not in st.session_state:
        st.session_state.live_log_entries = []
        
    # Initialize navigation state
    if "main_view_selection" not in st.session_state:
        st.session_state.main_view_selection = "Dashboard"
    
    # Initialize DPO Training Execution & Monitoring variables
    if "dpo_training_active" not in st.session_state:
        st.session_state.dpo_training_active = False  # Set to True for development visibility
    
    if "dpo_status_state" not in st.session_state:
        st.session_state.dpo_status_state = "running"  # Can be "running", "complete", "error"
    
    if "dpo_progress_percent" not in st.session_state:
        st.session_state.dpo_progress_percent = 50  # 0-100
    
    if "dpo_progress_text" not in st.session_state:
        st.session_state.dpo_progress_text = "Epoch 1/1, Step 50/100"
    
    if "dpo_console_output" not in st.session_state:
        st.session_state.dpo_console_output = "Initializing DPO trainer...\nLoading dataset...\nStarting training epoch 1/1...\nProcessing batch 1/25: loss=0.342\nProcessing batch 2/25: loss=0.318"


def main():
    """Main entry point for the Streamlit application."""
    # Configure the page
    st.set_page_config(
        page_title="Camel Workflow Dashboard",
        page_icon="🐪",
        layout="wide",
    )
    
    # Initialize session state
    init_session_state()
    
    # Render the sidebar with navigation
    render_sidebar()
    
    # Add navigation in sidebar
    st.sidebar.title("Navigation")
    view_selection = st.sidebar.radio("Go to", ["Dashboard", "Configuration", "Log Explorer", "DPO Training", "Settings"], key="main_view_selection")
    
    # Render the selected view
    if st.session_state.main_view_selection == "Dashboard":
        render_dashboard_view()
    elif st.session_state.main_view_selection == "Configuration":
        render_config_view()
    elif st.session_state.main_view_selection == "Log Explorer":
        render_log_explorer_view()
    elif st.session_state.main_view_selection == "DPO Training":
        render_dpo_training_view()
    elif st.session_state.main_view_selection == "Settings":
        render_settings_view()


if __name__ == "__main__":
    main()