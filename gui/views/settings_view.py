"""
Settings View for CAMEL Extensions GUI.

This module implements the Settings View for displaying read-only critical paths 
and basic application information.
"""
import streamlit as st
import os
from datetime import date


def render_settings_view():
    """
    Render the Settings view with path configuration and application information.
    """
    st.title("Settings")
    
    # Get paths from environment or use defaults
    project_root = os.environ.get("PROJECT_ROOT", os.getcwd())
    db_path = os.environ.get("DATABASE_URL", "sqlite:///./camel_extensions.db")
    agents_yaml_path = os.path.join(project_root, "configs/agents.yaml")
    models_dir_path = os.path.join(project_root, "models/")
    
    # Application information
    app_version = "0.2.0"
    last_updated_date = "2025-05-21"
    project_github_url = "https://github.com/camel-ai/camel"
    
    # Path Configuration Section (Read-only)
    st.subheader("Path Configuration")
    st.text_input("Project Root Directory:", value=project_root, disabled=True)
    st.text_input("Database URL:", value=db_path, disabled=True)
    st.text_input("Path to agents.yaml:", value=agents_yaml_path, disabled=True)
    st.text_input("Path to Saved Model Adapters:", value=models_dir_path, disabled=True)
    
    # Check if API is connected
    api_status = "Connected"
    api_url = "Not configured"
    
    if "api_client" in st.session_state:
        api_url = st.session_state.api_client.base_url
        try:
            # Try a simple API call
            st.session_state.api_client.get_workflow_settings()
        except Exception as e:
            api_status = f"Error: {str(e)}"
    else:
        api_status = "Not initialized"
    
    st.text_input("API Endpoint:", value=api_url, disabled=True)
    st.text_input("API Status:", value=api_status, disabled=True)
    
    # WebSocket status
    ws_status = "Connected"
    ws_url = "Not configured"
    
    if "ws_manager" in st.session_state:
        ws_url = st.session_state.ws_manager.base_url
        if not st.session_state.ws_manager.active_connections:
            ws_status = "No active connections"
    else:
        ws_status = "Not initialized"
    
    st.text_input("WebSocket Base URL:", value=ws_url, disabled=True)
    st.text_input("WebSocket Status:", value=ws_status, disabled=True)
    
    # Application Information Section
    st.subheader("Application Information")
    
    # Check if logo file exists and display it
    logo_path = os.path.join(project_root, "misc/logo_light.png")
    if os.path.exists(logo_path):
        st.image(logo_path, width=100)
    
    st.header("CAMEL Extensions GUI")
    st.markdown(f"Version: {app_version} (Updated: {last_updated_date})")
    st.markdown("Developed for managing and improving CAMEL project agents.")
    st.link_button("View Project on GitHub", url=project_github_url)
    
    # System Information
    st.subheader("System Information")
    st.text_input("Python Version:", value=os.environ.get("PYTHON_VERSION", "3.10+"), disabled=True)
    st.text_input("Streamlit Version:", value=st.__version__, disabled=True)
    
    # Show active WebSocket connections
    if "ws_manager" in st.session_state and st.session_state.ws_manager.active_connections:
        st.subheader("Active WebSocket Connections")
        
        for key, ws in st.session_state.ws_manager.active_connections.items():
            st.text(f"Connection: {key} -> {ws.url}")