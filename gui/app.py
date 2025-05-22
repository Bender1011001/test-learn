"""
Main application file for the camel_ext GUI.

This is the entry point for the Streamlit application that renders the camel_ext GUI.
"""
import streamlit as st
import os
from views.dashboard_view import render_dashboard_view, render_sidebar
from views.config_view import render_config_view, init_config_data
from views.log_explorer_view import render_log_explorer_view
from views.dpo_training_view import render_dpo_training_view
from views.settings_view import render_settings_view
from views.monitoring_view import render_monitoring_view
from api_client import APIClient
from websocket_client import WebSocketManager


def get_api_base_url():
    """Get the API base URL from environment variable or use default."""
    return os.environ.get("API_BASE_URL", "http://localhost:8000/api")


def init_api_client():
    """Initialize the API client if it doesn't exist."""
    if "api_client" not in st.session_state:
        base_url = get_api_base_url()
        st.session_state.api_client = APIClient(base_url=base_url)
        st.info(f"Connected to API at {base_url}")


def init_websocket_manager():
    """Initialize the WebSocket manager if it doesn't exist."""
    if "ws_manager" not in st.session_state:
        # Convert API URL to WebSocket URL
        base_url = get_api_base_url()
        ws_base_url = base_url.replace("http://", "ws://").replace("https://", "wss://")
        ws_base_url = ws_base_url.replace("/api", "")
        st.session_state.ws_manager = WebSocketManager(base_url=ws_base_url)


def init_session_state():
    """Initialize session state variables if they don't exist."""
    # Initialize API client and WebSocket manager
    init_api_client()
    init_websocket_manager()
    
    # Initialize workflow state
    if "workflow_initial_goal" not in st.session_state:
        st.session_state.workflow_initial_goal = ""
    
    if "selected_workflow_name" not in st.session_state:
        st.session_state.selected_workflow_name = "proposer_executor_review_loop"
    
    if "active_workflow_run_id" not in st.session_state:
        st.session_state.active_workflow_run_id = None
        
    # Initialize live log entries for interaction feed
    if "live_log_entries" not in st.session_state:
        st.session_state.live_log_entries = []
        
    # Initialize navigation state
    if "main_view_selection" not in st.session_state:
        st.session_state.main_view_selection = "Dashboard"
    
    # Initialize DPO Training Execution & Monitoring variables
    if "dpo_training_active" not in st.session_state:
        st.session_state.dpo_training_active = False
    
    if "dpo_active_job_id" not in st.session_state:
        st.session_state.dpo_active_job_id = None
        
    if "dpo_status_state" not in st.session_state:
        st.session_state.dpo_status_state = "idle"  # Can be "idle", "running", "complete", "error"
    
    if "dpo_progress_percent" not in st.session_state:
        st.session_state.dpo_progress_percent = 0  # 0-100
    
    if "dpo_progress_text" not in st.session_state:
        st.session_state.dpo_progress_text = ""
    
    if "dpo_console_output" not in st.session_state:
        st.session_state.dpo_console_output = []
    
    # Initialize monitoring state
    if "api_base_url" not in st.session_state:
        st.session_state.api_base_url = get_api_base_url()
    
    if "monitoring_auto_refresh" not in st.session_state:
        st.session_state.monitoring_auto_refresh = False


def cleanup_resources():
    """Clean up resources when the app exits."""
    if "ws_manager" in st.session_state:
        st.session_state.ws_manager.close_all()
    
    if "api_client" in st.session_state:
        st.session_state.api_client.close()


def main():
    """Main entry point for the Streamlit application."""
    # Configure the page
    st.set_page_config(
        page_title="CAMEL Extensions Dashboard",
        page_icon="🐪",
        layout="wide",
    )
    
    # Initialize session state
    init_session_state()
    
    # Register cleanup handler
    st.session_state["_cleanup_registered"] = True
    
    # Render the sidebar with navigation
    render_sidebar()
    
    # Add navigation in sidebar
    st.sidebar.title("Navigation")
    view_selection = st.sidebar.radio("Go to", ["Dashboard", "Configuration", "Log Explorer", "DPO Training", "Monitoring", "Settings"], key="main_view_selection")
    
    # Render the selected view
    if st.session_state.main_view_selection == "Dashboard":
        render_dashboard_view()
    elif st.session_state.main_view_selection == "Configuration":
        render_config_view()
    elif st.session_state.main_view_selection == "Log Explorer":
        render_log_explorer_view()
    elif st.session_state.main_view_selection == "DPO Training":
        render_dpo_training_view()
    elif st.session_state.main_view_selection == "Monitoring":
        render_monitoring_view()
    elif st.session_state.main_view_selection == "Settings":
        render_settings_view()


if __name__ == "__main__":
    try:
        main()
    finally:
        # Ensure resources are cleaned up
        cleanup_resources()