"""
Dashboard / Workflow Execution View for CAMEL Extensions GUI.

This module implements the dashboard view with workflow selection, goal input,
execution controls, model display, and progress visualization.
"""
import streamlit as st
import json
import datetime
from typing import Dict, Any, List, Optional


def handle_workflow_websocket_message(data: Dict[str, Any]):
    """
    Handle incoming WebSocket messages for workflow execution.
    
    Args:
        data: Message data from WebSocket
    """
    message_type = data.get("type")
    
    if message_type == "log":
        # Add new log entry to the feed
        log_data = data.get("data", {})
        st.session_state.live_log_entries.append(log_data)
        # Keep a reasonable cap on the number of entries
        if len(st.session_state.live_log_entries) > 100:
            st.session_state.live_log_entries = st.session_state.live_log_entries[-100:]
        # Trigger a rerun to display the new log entry
        st.rerun()
    elif message_type == "status":
        # Update workflow status
        status_data = data.get("data", {})
        st.session_state.workflow_status = status_data
        
        # Check if the workflow has completed
        if status_data.get("status") in ["completed", "failed"]:
            st.session_state.workflow_running = False
            # Signal completion
            st.rerun()
    elif message_type == "error":
        # Handle error
        error_message = data.get("message", "Unknown error occurred")
        st.error(f"Workflow error: {error_message}")


def start_workflow():
    """Start a new workflow execution."""
    if not st.session_state.workflow_initial_goal:
        st.error("Please enter an initial goal before starting the workflow.")
        return
    
    if st.session_state.get("workflow_running", False):
        st.warning("A workflow is already running. Please stop it first.")
        return
    
    try:
        # Get the API client
        api_client = st.session_state.api_client
        
        # Start the workflow
        run_id = api_client.start_workflow(
            workflow_id=st.session_state.selected_workflow_name,
            initial_goal=st.session_state.workflow_initial_goal
        )
        
        # Store the run ID
        st.session_state.active_workflow_run_id = run_id
        st.session_state.workflow_running = True
        st.session_state.workflow_status = {"status": "starting"}
        
        # Clear previous log entries
        st.session_state.live_log_entries = []
        
        # Set up WebSocket connection for real-time updates
        ws_manager = st.session_state.ws_manager
        ws = ws_manager.create_workflow_connection(run_id, handle_workflow_websocket_message)
        
        st.success(f"Workflow started with ID: {run_id}")
        st.rerun()
    except Exception as e:
        st.error(f"Error starting workflow: {str(e)}")


def stop_workflow():
    """Stop the current workflow execution."""
    if not st.session_state.get("workflow_running", False):
        st.warning("No workflow is currently running.")
        return
    
    try:
        # Get the API client
        api_client = st.session_state.api_client
        
        # Stop the workflow
        if st.session_state.active_workflow_run_id:
            success = api_client.stop_workflow(st.session_state.active_workflow_run_id)
            
            if success:
                st.session_state.workflow_running = False
                st.success("Workflow stopped successfully.")
                
                # Close WebSocket connection
                ws_manager = st.session_state.ws_manager
                ws_manager.close_connection(st.session_state.active_workflow_run_id)
                
                st.rerun()
            else:
                st.error("Failed to stop workflow.")
    except Exception as e:
        st.error(f"Error stopping workflow: {str(e)}")


def fetch_available_workflows() -> List[Dict[str, Any]]:
    """
    Fetch available workflows from the API.
    
    Returns:
        List of available workflows
    """
    try:
        api_client = st.session_state.api_client
        return api_client.get_available_workflows()
    except Exception as e:
        st.error(f"Error fetching available workflows: {str(e)}")
        return []


def fetch_active_models() -> Dict[str, Dict[str, str]]:
    """
    Fetch information about active models and adapters.
    
    Returns:
        Dictionary with agent names as keys and model info as values
    """
    try:
        api_client = st.session_state.api_client
        
        # Get configurations for the agents in the workflow
        agent_configs = api_client.get_all_agent_configs()
        
        # Organize by agent name
        active_models = {}
        for agent_id, config in agent_configs.items():
            active_models[agent_id] = {
                "model_id": config.get("model_id", "Unknown"),
                "adapter_id": config.get("adapter_id", "None")
            }
        
        return active_models
    except Exception as e:
        st.error(f"Error fetching active models: {str(e)}")
        return {}


def clear_live_feed():
    """Clear the live interaction feed."""
    st.session_state.live_log_entries = []


def render_dashboard_view():
    """
    Render the main dashboard/workflow execution view.
    
    This function implements UI elements for workflow selection, goal input,
    execution controls, active model display, and workflow progress visualization.
    """
    st.title("CAMEL Workflow Dashboard")
    
    # 1. Workflow Selection & Description
    st.header("Workflow Selection")
    
    # Fetch available workflows from the API
    workflows = fetch_available_workflows()
    workflow_options = [w["id"] for w in workflows] if workflows else ["proposer_executor_review_loop"]
    
    # Default to the first workflow or the remembered one
    default_index = 0
    if st.session_state.selected_workflow_name in workflow_options:
        default_index = workflow_options.index(st.session_state.selected_workflow_name)
    
    selected_workflow = st.selectbox(
        "Select Workflow:",
        options=workflow_options,
        index=default_index,
        key="selected_workflow_name"
    )
    
    # Display workflow description
    workflow_description = "No description available."
    for workflow in workflows:
        if workflow["id"] == selected_workflow:
            workflow_description = workflow.get("description", "No description available.")
    
    st.markdown(f"**Description:** {workflow_description}")
    
    # 2. Initial Goal Input
    st.header("Initial Goal")
    goal_input = st.text_area(
        "Enter Initial Goal for Proposer:",
        height=100,
        key="workflow_initial_goal",
        help="The initial objective for the Proposer agent."
    )
    
    # Clear goal button
    if st.button("Clear Goal", key="clear_goal_button"):
        st.session_state.workflow_initial_goal = ""
        st.rerun()
    
    # 3. Execution Controls
    st.header("Execution Controls")
    
    # Use columns for horizontal layout of buttons
    col1, col2, col3, col4 = st.columns(4)
    
    workflow_running = st.session_state.get("workflow_running", False)
    
    with col1:
        if not workflow_running:
            if st.button("🚀 Start Workflow", key="run_workflow_button"):
                start_workflow()
        else:
            if st.button("⏹️ Stop Workflow", key="stop_workflow_button"):
                stop_workflow()
    
    with col2:
        pause_button = st.button(
            "⏸️ Pause Workflow" if not st.session_state.get("workflow_paused", False) else "▶️ Resume Workflow", 
            key="pause_resume_button",
            disabled=not workflow_running
        )
        if pause_button and workflow_running:
            # Toggle pause state
            st.session_state.workflow_paused = not st.session_state.get("workflow_paused", False)
            # API call would go here in a real implementation
    
    with col3:
        step_button = st.button("➡️ Step", key="step_button", disabled=not workflow_running or not st.session_state.get("workflow_paused", False))
        if step_button and workflow_running and st.session_state.get("workflow_paused", False):
            # Step logic would go here in a real implementation
            pass
    
    with col4:
        if workflow_running:
            status = st.session_state.get("workflow_status", {}).get("status", "starting")
            st.info(f"Status: {status.capitalize()}")
    
    # 5. Workflow Progress Visualization
    st.header("Workflow Progress")
    
    if workflow_running:
        # Get current step and total steps from workflow status
        status = st.session_state.get("workflow_status", {})
        current_step = status.get("current_step", 0)
        total_steps = status.get("total_steps", 10)
        
        st.markdown(f"Current Step: **{current_step}** of {total_steps}")
        # Add a progress bar
        st.progress(current_step / total_steps if total_steps > 0 else 0)
    else:
        st.markdown("Current Step: **Waiting to start**")
    
    # 6. Live Interaction Feed
    st.header("Live Interaction Feed")
    
    # Controls for feed
    col1, col2 = st.columns([3, 1])
    with col1:
        st.checkbox("Auto-scroll to latest", value=True, key="log_autoscroll")
    with col2:
        if st.button("Clear Feed", key="clear_live_feed_button"):
            clear_live_feed()
            st.rerun()
    
    # Custom CSS for the feed container
    st.markdown("""
        <style>
        .interaction-feed-container {
            height: 400px;
            overflow-y: auto;
            border: 1px solid #e0e0e0;
            padding: 10px;
            border-radius: 5px;
        }
        </style>
    """, unsafe_allow_html=True)
    
    # Create container for the interaction feed
    with st.container():
        st.markdown('<div class="interaction-feed-container">', unsafe_allow_html=True)
        
        # Display live log entries
        if st.session_state.live_log_entries:
            for log in st.session_state.live_log_entries:
                # Format timestamp
                timestamp = log.get("timestamp", "Unknown time")
                agent_name = log.get("agent_name", "Unknown")
                agent_type = log.get("agent_type", "")
                
                # Create a unique key for the expander
                expander_key = f"{timestamp}-{agent_name}-{log.get('id', '')}"
                
                with st.expander(f"{timestamp} - Agent: {agent_name} ({agent_type})", expanded=True):
                    st.json(log.get("input", {}), expanded=False)
                    st.json(log.get("output", {}), expanded=False)
        else:
            # If no logs yet, show a message
            if workflow_running:
                st.info("Waiting for interaction logs...")
            else:
                # Show placeholder logs if no workflow is running
                now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                
                with st.expander(f"{now} - Agent: Proposer (LLM) - (Example)", expanded=True):
                    st.json({"input": "Initial goal: Create a comprehensive report on renewable energy trends"}, expanded=False)
                    st.json({"output": "I'll start by researching the latest developments in solar, wind, and hydroelectric power sources."}, expanded=False)
                    
                with st.expander(f"{now} - Agent: Executor (LLM) - (Example)", expanded=True):
                    st.json({"input": "Examine renewable energy trends focusing on solar, wind, and hydroelectric sources"}, expanded=False)
                    st.json({"output": "Analysis complete. Solar adoption increased 25% in 2022, wind energy capacity grew by 15%, and new hydroelectric developments have slowed to 5% growth."}, expanded=False)
                    
                with st.expander(f"{now} - Agent: PeerReviewer (LLM) - (Example)", expanded=True):
                    st.json({"input": "Review the analysis on renewable energy trends"}, expanded=False)
                    st.json({"output": "The analysis is generally good but lacks context on regional variations and government policies. Suggest expanding these areas."}, expanded=False)
        
        st.markdown('</div>', unsafe_allow_html=True)


def render_sidebar():
    """Render the sidebar with active model/adapter information."""
    
    st.sidebar.title("Active Models")
    
    try:
        # Fetch active models from the API or use cached data
        active_models = fetch_active_models()
        
        # Display each agent's model information
        for agent_id, model_info in active_models.items():
            model_id = model_info.get("model_id", "Unknown")
            adapter_id = model_info.get("adapter_id", "None")
            adapter_display = adapter_id if adapter_id else "None"
            
            st.sidebar.markdown(f"**{agent_id}:** `{model_id}`\nAdapter: `{adapter_display}`")
            
    except Exception as e:
        # Fallback to placeholder data if the API call fails
        st.sidebar.markdown("**Proposer:** `mistralai/Mistral-7B-Instruct-v0.2`\nAdapter: `None`")
        st.sidebar.markdown("**PeerReviewer:** `claude-3-opus-20240229`\nAdapter: `None`")