"""
Log Explorer & Annotation View for CAMEL Extensions GUI.

This module implements the log explorer view with log display, filtering UI,
and annotation capabilities. It uses the API client to fetch and save data.
"""
import streamlit as st
import pandas as pd
import json
from datetime import datetime, date, timedelta
from typing import Dict, Any, List, Optional


def fetch_logs(filters: Optional[Dict[str, Any]] = None):
    """
    Fetch logs from the API with optional filters.
    
    Args:
        filters: Optional dictionary of filter parameters
    
    Returns:
        List of log entries
    """
    try:
        api_client = st.session_state.api_client
        
        # Build filter parameters
        params = {}
        if filters:
            # Apply workflow ID filter
            if filters.get("workflow_id"):
                params["workflow_id"] = filters["workflow_id"]
                
            # Apply date range filter
            if filters.get("start_date"):
                params["start_date"] = filters["start_date"].isoformat()
            if filters.get("end_date"):
                params["end_date"] = filters["end_date"].isoformat()
            
            # Apply agent name filter
            if filters.get("agent_names"):
                params["agent_name"] = filters["agent_names"][0]  # API currently only supports one
                
            # Apply agent type filter
            if filters.get("agent_types"):
                params["agent_type"] = filters["agent_types"][0]  # API currently only supports one
                
            # Apply annotation status filter
            if filters.get("has_annotation") is not None:
                params["has_annotation"] = filters["has_annotation"]
                
            # Apply keyword filter
            if filters.get("keyword"):
                params["keyword"] = filters["keyword"]
        
        # Add pagination parameters
        params["offset"] = filters.get("offset", 0) if filters else 0
        params["limit"] = filters.get("limit", 100) if filters else 100
        params["sort_by"] = filters.get("sort_by", "timestamp") if filters else "timestamp"
        params["sort_desc"] = filters.get("sort_desc", True) if filters else True
        
        # Fetch logs
        logs = api_client.get_logs(**params)
        
        # Process logs for display
        processed_logs = []
        for log in logs:
            # Format the log for display
            processed_log = {
                "ID": log.get("id"),
                "Timestamp": log.get("timestamp"),
                "WorkflowID": log.get("workflow_run_id"),
                "Step #": len(processed_logs) + 1,  # Temporary, would be from sequence in real data
                "Agent Name": log.get("agent_name"),
                "Agent Type": log.get("agent_type"),
                "Input Summary": str(log.get("input_data", {})),
                "Output Summary": str(log.get("output_data", {})),
                "Annotation Status": "Annotated" if log.get("has_annotation") else "Unannotated",
                "Raw": log  # Keep the raw log data for detail view
            }
            processed_logs.append(processed_log)
        
        return processed_logs
    
    except Exception as e:
        st.error(f"Error fetching logs: {str(e)}")
        return []


def init_log_explorer_data():
    """Initialize log data using the API client if not already in session state."""
    if "log_explorer_data" not in st.session_state or st.session_state.get("reload_logs", False):
        try:
            # Get filters from session state
            filters = get_current_filters()
            
            # Fetch logs with filters
            st.session_state.log_explorer_data = fetch_logs(filters)
            st.session_state.reload_logs = False
        except Exception as e:
            st.error(f"Error initializing log data: {str(e)}")
            
            # Use placeholder data as fallback
            if "log_explorer_data" not in st.session_state:
                st.session_state.log_explorer_data = []


def get_current_filters() -> Dict[str, Any]:
    """Get current filter values from session state."""
    filters = {}
    
    # WorkflowID filter
    workflow_id = st.session_state.get("log_filter_workflow_id")
    if workflow_id:
        filters["workflow_id"] = workflow_id
    
    # Date range filter
    date_range = st.session_state.get("log_filter_date_range")
    if date_range and len(date_range) == 2:
        filters["start_date"] = date_range[0]
        filters["end_date"] = date_range[1]
    
    # Agent name filter
    agent_names = st.session_state.get("log_filter_agent_name")
    if agent_names:
        filters["agent_names"] = agent_names
    
    # Agent type filter
    agent_types = st.session_state.get("log_filter_agent_type")
    if agent_types:
        filters["agent_types"] = agent_types
    
    # Annotation status filter
    annotation_status = st.session_state.get("log_filter_annotation_status")
    if annotation_status and annotation_status != "All":
        if annotation_status == "Unannotated":
            filters["has_annotation"] = False
        else:  # All annotated cases
            filters["has_annotation"] = True
    
    # Keyword search
    keyword = st.session_state.get("log_filter_keyword")
    if keyword:
        filters["keyword"] = keyword
    
    return filters


def apply_filters():
    """Apply filters and reload logs."""
    st.session_state.reload_logs = True
    st.rerun()


def reset_filters():
    """Reset all filters to default values."""
    # Reset WorkflowID filter
    st.session_state.log_filter_workflow_id = ""
    
    # Reset date range filter to last 7 days
    st.session_state.log_filter_date_range = (date.today() - timedelta(days=7), date.today())
    
    # Reset agent name filter
    st.session_state.log_filter_agent_name = []
    
    # Reset agent type filter
    st.session_state.log_filter_agent_type = []
    
    # Reset annotation status filter
    st.session_state.log_filter_annotation_status = "All"
    
    # Reset keyword search
    st.session_state.log_filter_keyword = ""
    
    # Force reload of logs
    st.session_state.reload_logs = True
    st.rerun()


def fetch_log_annotation(log_id: int) -> Optional[Dict[str, Any]]:
    """
    Fetch annotation for a specific log entry.
    
    Args:
        log_id: Log entry ID
    
    Returns:
        Annotation data if available, otherwise None
    """
    try:
        api_client = st.session_state.api_client
        return api_client.get_annotation(log_id)
    except Exception as e:
        st.error(f"Error fetching annotation: {str(e)}")
        return None


def save_annotation(annotation_data: Dict[str, Any]) -> bool:
    """
    Save annotation to the backend.
    
    Args:
        annotation_data: Annotation data to save
    
    Returns:
        Success or failure
    """
    try:
        api_client = st.session_state.api_client
        annotation_id = api_client.save_annotation(annotation_data)
        
        if annotation_id:
            st.success(f"Annotation saved successfully.")
            # Force reload of logs to reflect new annotation status
            st.session_state.reload_logs = True
            return True
        else:
            st.error("Failed to save annotation.")
            return False
    except Exception as e:
        st.error(f"Error saving annotation: {str(e)}")
        return False


def render_log_filtering_controls():
    """Render the controls for filtering logs."""
    with st.expander("Filter Logs", expanded=True):
        # Search by WorkflowID
        st.text_input("Search WorkflowID:", key="log_filter_workflow_id")
        
        # Date Range Filter
        # Default to last 7 days
        default_start_date = date.today() - timedelta(days=7)
        default_end_date = date.today()
        
        # Initialize the date range in session state if not already there
        if "log_filter_date_range" not in st.session_state:
            st.session_state.log_filter_date_range = (default_start_date, default_end_date)
        
        st.date_input("Date Range:", value=st.session_state.log_filter_date_range, key="log_filter_date_range")
        
        # Get agent names and types from logs for filter options
        agent_names = set()
        agent_types = set()
        
        if st.session_state.get("log_explorer_data"):
            for log in st.session_state.log_explorer_data:
                if "Agent Name" in log and log["Agent Name"]:
                    agent_names.add(log["Agent Name"])
                if "Agent Type" in log and log["Agent Type"]:
                    agent_types.add(log["Agent Type"])
        
        # Default options if no logs
        if not agent_names:
            agent_names = ["Proposer", "Executor", "PeerReviewer"]
        if not agent_types:
            agent_types = ["Proposer", "Executor", "PeerReviewer"]
        
        # Agent Name Filter
        st.multiselect(
            "Filter by Agent Name(s):", 
            options=list(agent_names),
            key="log_filter_agent_name"
        )
        
        # Agent Type Filter
        st.multiselect(
            "Filter by Agent Type(s):", 
            options=list(agent_types),
            key="log_filter_agent_type"
        )
        
        # Annotation Status Filter
        st.selectbox(
            "Filter by Annotation Status:",
            options=["All", "Unannotated", "Annotated"],
            key="log_filter_annotation_status"
        )
        
        # Keyword Search
        st.text_input("Keyword Search (in Input/Output):", key="log_filter_keyword")
        
        # Filter Action Buttons
        col1, col2 = st.columns(2)
        with col1:
            if st.button("Apply Filters", key="apply_log_filters_button"):
                apply_filters()
        with col2:
            if st.button("Reset Filters", key="reset_log_filters_button"):
                reset_filters()


def render_log_display():
    """Render the log data display with configurable columns."""
    st.header("Log Entries")
    
    if not st.session_state.log_explorer_data:
        st.info("No log data to display. Try adjusting the filters or check API connectivity.")
        return
    
    # Get all possible columns from the log data
    all_columns = [col for col in list(st.session_state.log_explorer_data[0].keys()) if col != "Raw"]
    
    # Default visible columns
    default_visible_columns = ["ID", "Timestamp", "Agent Name", "Agent Type", 
                              "Input Summary", "Output Summary", "Annotation Status"]
    
    # Column selection multiselect
    visible_columns = st.multiselect(
        "Visible Columns:",
        options=all_columns,
        default=[col for col in default_visible_columns if col in all_columns],
        key="log_visible_columns"
    )
    
    # Display the log data as a dataframe with selected columns
    if visible_columns:
        # Create a new DataFrame with only the visible columns
        display_data = [{col: log[col] for col in visible_columns if col in log} 
                        for log in st.session_state.log_explorer_data]
        display_df = pd.DataFrame(display_data)
        st.dataframe(display_df, use_container_width=True)
    else:
        st.info("Select columns to display.")


def render_log_detail_view():
    """
    Render the detailed view for a selected log entry with annotation capabilities.
    
    This includes:
    - A log entry selection mechanism via dropdown
    - Display of full interaction details in a tab
    - A form for annotating proposer actions in another tab
    """
    st.header("Log Detail View & Annotation")
    
    # Log entry selection mechanism
    if st.session_state.log_explorer_data:
        log_ids = [log['ID'] for log in st.session_state.log_explorer_data]
        if log_ids:
            selected_log_id = st.selectbox(
                "Select Log ID for Details/Annotation:",
                options=log_ids,
                key="selected_log_id_for_detail"
            )
            # Find the selected log entry
            selected_log_entry = next(
                (log for log in st.session_state.log_explorer_data if log['ID'] == selected_log_id),
                None
            )
        else:
            selected_log_entry = None
            st.info("No logs available to select.")
    else:
        selected_log_entry = None
        st.info("No log data available.")
    
    # Initialize selected_log_entry in session state if not already there
    if 'selected_log_entry' not in st.session_state:
        st.session_state.selected_log_entry = None
    
    if selected_log_entry:
        # Update session state with selected log entry
        st.session_state.selected_log_entry = selected_log_entry
        
        # Create tabs for different views of the selected log
        tabs = st.tabs(["Interaction Details", "Annotate Proposer Action"])
        
        # Interaction Details Tab
        with tabs[0]:
            # Get raw log data from the entry
            raw_log = selected_log_entry.get("Raw", {})
            
            st.subheader("Full Input Data")
            input_data = raw_log.get("input_data", {})
            st.json(input_data)
            
            st.subheader("Full Output Data")
            output_data = raw_log.get("output_data", {})
            st.json(output_data)
        
        # Annotate Proposer Action Tab
        with tabs[1]:
            # Only show annotation form if the selected log is from a Proposer
            if selected_log_entry['Agent Type'] == 'Proposer':
                # Fetch existing annotation if any
                annotation = fetch_log_annotation(selected_log_entry['ID'])
                
                # Extract goal and proposer command from raw data
                raw_log = selected_log_entry.get("Raw", {})
                input_data = raw_log.get("input_data", {})
                output_data = raw_log.get("output_data", {})
                
                # Try to extract a goal from input data
                goal = input_data.get('goal', str(input_data))
                
                # Try to extract proposer command from output data
                original_proposer_cmd = str(output_data.get('proposal', str(output_data)))
                
                # Context Display (Read-only)
                st.markdown(f"**Goal:** {goal}")
                st.markdown(f"**Proposer's Original Command:** `{original_proposer_cmd}`")
                
                # Executor and Peer Review placeholders for context
                # In a real implementation, we'd fetch related logs from the same workflow
                st.markdown(f"**Executor's Result (if available):** `stdout: Command executed successfully.`, `stderr: `, `exit_code: 0`")
                st.markdown(f"**Peer Review (if available):** Score: 4/5, Critique: 'The proposed command is good but could be more efficient.'")
                
                # Annotation Form
                with st.form("proposer_annotation_form"):
                    # Pre-fill form with existing annotation if available
                    rating_options = ["⭐", "⭐⭐", "⭐⭐⭐", "⭐⭐⭐⭐", "⭐⭐⭐⭐⭐"]
                    rating_index = min(int(annotation.get("rating", 3)) - 1, 4) if annotation else 2
                    
                    rating = st.radio(
                        "Rate Original Action Quality:",
                        options=rating_options,
                        index=rating_index,
                        horizontal=True,
                        key="proposer_quality_rating_form"
                    )
                    
                    rationale = st.text_area(
                        "Rationale for Rating (Optional):",
                        value=annotation.get("rationale", "") if annotation else "",
                        key="proposer_rating_rationale_form"
                    )
                    
                    st.subheader("For DPO Training (Direct Preference Optimization):")
                    st.markdown("Provide a **Chosen** (better or ideal) command and a **Rejected** (clearly worse) command for the *same initial goal*.")
                    
                    chosen_cmd = st.text_area(
                        "✅ **Chosen Command** (Ideal):",
                        value=annotation.get("chosen_prompt", original_proposer_cmd) if annotation else original_proposer_cmd,
                        height=100,
                        key="chosen_proposer_command_form"
                    )
                    
                    rejected_cmd = st.text_area(
                        "❌ **Rejected Command** (Inferior/Incorrect):",
                        value=annotation.get("rejected_prompt", "") if annotation else "",
                        height=100,
                        key="rejected_proposer_command_form",
                        help="Example: A less efficient, incorrect, or unsafe command."
                    )
                    
                    # Auto-generate DPO prompt context from goal
                    auto_generated_dpo_prompt = f"Given the goal: '{goal}', propose a command."
                    
                    prompt_context = st.text_area(
                        "📝 **Prompt/Context for DPO** (Auto-generated, editable):",
                        value=annotation.get("dpo_context", auto_generated_dpo_prompt) if annotation else auto_generated_dpo_prompt,
                        height=150,
                        key="proposer_dpo_prompt_form",
                        help="This will be the 'prompt' for the DPO trainer..."
                    )
                    
                    # Submit annotation
                    if st.form_submit_button("💾 Save Proposer Annotation"):
                        # Convert rating to numeric value
                        numeric_rating = len(rating)
                        
                        # Prepare annotation data
                        annotation_data = {
                            "log_entry_id": selected_log_entry['ID'],
                            "rating": numeric_rating,
                            "rationale": rationale,
                            "chosen_prompt": chosen_cmd,
                            "rejected_prompt": rejected_cmd,
                            "dpo_context": prompt_context,
                            "user_id": "gui_user"  # In a real app, this would be the authenticated user
                        }
                        
                        # Save annotation
                        if save_annotation(annotation_data):
                            # Force reload of log data to reflect changes
                            st.session_state.reload_logs = True
                            st.rerun()
            else:
                st.info("Annotation is only available for logs from Proposer agents. The selected log is from a different agent type.")


def render_log_explorer_view():
    """
    Render the log explorer & annotation view.
    
    This function implements UI elements for log display, filtering, and annotation.
    It uses the API client to fetch and save data.
    """
    st.title("Log Explorer & Annotation View")
    
    # Initialize log data if not already done
    init_log_explorer_data()
    
    # Create a two-column layout
    col1, col2 = st.columns([3, 1])
    
    with col2:
        # Render filtering controls in the sidebar
        render_log_filtering_controls()
    
    with col1:
        # Render the log display with configurable columns
        render_log_display()
        
        # Add separator
        st.divider()
        
        # Render the detailed view for the selected log
        render_log_detail_view()