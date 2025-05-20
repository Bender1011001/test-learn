"""
Settings View for camel_ext GUI.

This module implements the Settings View for displaying read-only critical paths 
and basic application information. For MVP, this displays placeholder paths and 
application information.
"""
import streamlit as st
import os


def render_settings_view():
    """
    Render the Settings view with path configuration and application information.
    
    For MVP, this uses placeholder paths and application information.
    """
    st.title("Settings")

    # Define placeholder values
    PROJECT_ROOT_PATH_PLACEHOLDER = os.getcwd()  # Dynamic example using current working directory
    DB_PATH_PLACEHOLDER = "logs/camel_logs.db"  # Relative to project root
    AGENTS_YAML_PATH_PLACEHOLDER = "configs/agents.yaml"  # Relative to project root
    MODELS_DIR_PATH_PLACEHOLDER = "models/"  # Relative to project root
    APP_VERSION_PLACEHOLDER = "0.1.0-mvp"
    LAST_UPDATED_DATE_PLACEHOLDER = "2025-05-20"
    PROJECT_GITHUB_URL_PLACEHOLDER = "https://github.com/camel-ai/camel"

    # Path Configuration Section (Read-only for MVP)
    st.subheader("Path Configuration (Read-only for MVP)")
    st.text_input("Project Root Directory:", value=PROJECT_ROOT_PATH_PLACEHOLDER, disabled=True)
    st.text_input(
        "Path to logs.db:", 
        value=os.path.join(PROJECT_ROOT_PATH_PLACEHOLDER, DB_PATH_PLACEHOLDER) if PROJECT_ROOT_PATH_PLACEHOLDER else DB_PATH_PLACEHOLDER, 
        disabled=True
    )
    st.text_input(
        "Path to agents.yaml:", 
        value=os.path.join(PROJECT_ROOT_PATH_PLACEHOLDER, AGENTS_YAML_PATH_PLACEHOLDER) if PROJECT_ROOT_PATH_PLACEHOLDER else AGENTS_YAML_PATH_PLACEHOLDER, 
        disabled=True
    )
    st.text_input(
        "Path to Saved Model Adapters:", 
        value=os.path.join(PROJECT_ROOT_PATH_PLACEHOLDER, MODELS_DIR_PATH_PLACEHOLDER) if PROJECT_ROOT_PATH_PLACEHOLDER else MODELS_DIR_PATH_PLACEHOLDER, 
        disabled=True
    )

    # Application Information Section
    st.subheader("Application Information")
    
    # Check if logo file exists and display it (optional)
    logo_path = "misc/logo_light.png"
    if os.path.exists(logo_path):
        st.image(logo_path, width=100)
    
    st.header("CAMEL Extensions GUI")
    st.markdown(f"Version: {APP_VERSION_PLACEHOLDER} (Updated: {LAST_UPDATED_DATE_PLACEHOLDER})")
    st.markdown("Developed for managing and improving CAMEL project agents.")
    st.link_button("View Project on GitHub", url=PROJECT_GITHUB_URL_PLACEHOLDER)