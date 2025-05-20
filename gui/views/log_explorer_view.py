"""
Log Explorer & Annotation View for camel_ext GUI.

This module implements the log explorer view with log display and filtering UI.
For MVP, this uses placeholder data instead of actual database interaction.
"""
import streamlit as st
import pandas as pd
import json
from datetime import date, timedelta

def init_log_explorer_data():
    """Initialize placeholder log data if not already in session state."""
    if "log_explorer_data" not in st.session_state:
        # This is placeholder data that would normally be loaded from logs.db
        placeholder_logs = [
            {
                "ID": 1, 
                "Timestamp": "2025-05-20 10:00:00", 
                "WorkflowID": "wf_abc", 
                "Step #": 1, 
                "Agent Name": "Proposer", 
                "Agent Type": "Proposer", 
                "Input Summary": "{'goal': 'Create a comprehensive report on renewable energy trends'}", 
                "Output Summary": "{'proposal': 'I'll start by researching the latest developments in solar, wind, and hydroelectric power sources.'}", 
                "Annotation Status": "Unannotated"
            },
            {
                "ID": 2, 
                "Timestamp": "2025-05-20 10:00:05", 
                "WorkflowID": "wf_abc", 
                "Step #": 2, 
                "Agent Name": "Executor", 
                "Agent Type": "Executor", 
                "Input Summary": "{'command': 'Research renewable energy trends focusing on solar, wind, and hydroelectric sources'}", 
                "Output Summary": "{'stdout': 'Analysis complete. Solar adoption increased 25% in 2022, wind energy capacity grew by 15%, and hydroelectric developments slowed to 5% growth.'}", 
                "Annotation Status": "N/A"
            },
            {
                "ID": 3, 
                "Timestamp": "2025-05-20 10:01:10", 
                "WorkflowID": "wf_abc", 
                "Step #": 3, 
                "Agent Name": "PeerReviewer", 
                "Agent Type": "PeerReviewer", 
                "Input Summary": "{'review_subject': 'Analysis on renewable energy trends'}", 
                "Output Summary": "{'feedback': 'The analysis is good but lacks context on regional variations and government policies.'}", 
                "Annotation Status": "Marked for DPO - Chosen"
            },
            {
                "ID": 4, 
                "Timestamp": "2025-05-20 11:15:00", 
                "WorkflowID": "wf_def", 
                "Step #": 1, 
                "Agent Name": "Proposer", 
                "Agent Type": "Proposer", 
                "Input Summary": "{'goal': 'Optimize the database query performance'}", 
                "Output Summary": "{'proposal': 'Let's analyze the current query structure and identify potential bottlenecks.'}", 
                "Annotation Status": "Rated"
            },
            {
                "ID": 5, 
                "Timestamp": "2025-05-20 11:15:30", 
                "WorkflowID": "wf_def", 
                "Step #": 2, 
                "Agent Name": "Executor", 
                "Agent Type": "Executor", 
                "Input Summary": "{'command': 'Profile the database query performance'}", 
                "Output Summary": "{'stdout': 'Identified three queries with suboptimal performance. The main bottleneck is in the join operation.'}", 
                "Annotation Status": "N/A"
            },
            {
                "ID": 6, 
                "Timestamp": "2025-05-21 09:30:00", 
                "WorkflowID": "wf_ghi", 
                "Step #": 1, 
                "Agent Name": "Proposer", 
                "Agent Type": "Proposer", 
                "Input Summary": "{'goal': 'Develop a testing strategy for the new authentication module'}", 
                "Output Summary": "{'proposal': 'We should implement unit tests for each function and integration tests for the full workflow.'}", 
                "Annotation Status": "Marked for DPO - Rejected"
            }
        ]
        st.session_state.log_explorer_data = placeholder_logs


def render_log_filtering_controls():
    """Render the controls for filtering logs."""
    with st.expander("Filter Logs", expanded=True):
        # Search by WorkflowID
        st.text_input("Search WorkflowID:", key="log_filter_workflow_id")
        
        # Date Range Filter
        # Default to last 7 days
        default_start_date = date.today() - timedelta(days=7)
        default_end_date = date.today()
        st.date_input("Date Range:", value=(default_start_date, default_end_date), key="log_filter_date_range")
        
        # Agent Name Filter
        st.multiselect(
            "Filter by Agent Name(s):", 
            options=["Proposer", "Executor", "PeerReviewer"],
            key="log_filter_agent_name"
        )
        
        # Agent Type Filter
        st.multiselect(
            "Filter by Agent Type(s):", 
            options=["Proposer", "Executor", "PeerReviewer"],
            key="log_filter_agent_type"
        )
        
        # Annotation Status Filter
        st.selectbox(
            "Filter by Annotation Status:",
            options=["All", "Unannotated", "Rated", "Marked for DPO - Chosen", "Marked for DPO - Rejected"],
            key="log_filter_annotation_status"
        )
        
        # Keyword Search
        st.text_input("Keyword Search (in Input/Output JSON):", key="log_filter_keyword")
        
        # Filter Action Buttons
        col1, col2 = st.columns(2)
        with col1:
            st.button("Apply Filters", key="apply_log_filters_button")
        with col2:
            st.button("Reset Filters", key="reset_log_filters_button")


def render_log_display():
    """Render the log data display with configurable columns."""
    st.header("Log Entries")
    
    if not st.session_state.log_explorer_data:
        st.info("No log data to display.")
        return
    
    # Get all possible columns from the log data
    all_columns = list(st.session_state.log_explorer_data[0].keys())
    
    # Default visible columns
    default_visible_columns = ["ID", "Timestamp", "Agent Name", "Agent Type", 
                               "Input Summary", "Output Summary", "Annotation Status"]
    
    # Column selection multiselect
    visible_columns = st.multiselect(
        "Visible Columns:",
        options=all_columns,
        default=[col for col in default_visible_columns if col in all_columns]
    )
    
    # Display the log data as a dataframe with selected columns
    if visible_columns:
        display_df = pd.DataFrame(st.session_state.log_explorer_data)[visible_columns]
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
            st.subheader("Full Input Data")
            try:
                input_data = json.loads(st.session_state.selected_log_entry.get("Input Summary", "{}"))
                st.json(input_data)
            except json.JSONDecodeError:
                st.text(st.session_state.selected_log_entry.get("Input Summary", "Invalid or no input data"))

            st.subheader("Full Output Data")
            try:
                output_data = json.loads(st.session_state.selected_log_entry.get("Output Summary", "{}"))
                st.json(output_data)
            except json.JSONDecodeError:
                st.text(st.session_state.selected_log_entry.get("Output Summary", "Invalid or no output data"))
        
        # Annotate Proposer Action Tab
        with tabs[1]:
            # Only show annotation form if the selected log is from a Proposer
            if st.session_state.selected_log_entry['Agent Type'] == 'Proposer':
                # Context Display (Read-only)
                st.markdown(f"**Goal:** {st.session_state.selected_log_entry.get('Input Summary', 'N/A')}")
                
                # Extract proposer command from output
                try:
                    output_data = json.loads(st.session_state.selected_log_entry.get("Output Summary", "{}"))
                    original_proposer_cmd = output_data.get('proposal', 'N/A')
                except json.JSONDecodeError:
                    original_proposer_cmd = st.session_state.selected_log_entry.get("Output Summary", "N/A")
                
                st.markdown(f"**Proposer's Original Command:** `{original_proposer_cmd}`")
                
                # Placeholder for Executor's Result and Peer Review
                st.markdown(f"**Executor's Result (if available):** `stdout: Command executed successfully.`, `stderr: `, `exit_code: 0`")
                st.markdown(f"**Peer Review (if available):** Score: 4/5, Critique: 'The proposed command is good but could be more efficient.'")
                
                # Annotation Form
                with st.form("proposer_annotation_form"):
                    st.radio(
                        "Rate Original Action Quality:",
                        options=["⭐", "⭐⭐", "⭐⭐⭐", "⭐⭐⭐⭐", "⭐⭐⭐⭐⭐"],
                        horizontal=True,
                        key="proposer_quality_rating_form"
                    )
                    
                    st.text_area(
                        "Rationale for Rating (Optional):",
                        key="proposer_rating_rationale_form"
                    )
                    
                    st.subheader("For DPO Training (Direct Preference Optimization):")
                    st.markdown("Provide a **Chosen** (better or ideal) command and a **Rejected** (clearly worse) command for the *same initial goal*.")
                    
                    chosen_cmd = st.text_area(
                        "✅ **Chosen Command** (Ideal):",
                        value=original_proposer_cmd,
                        height=100,
                        key="chosen_proposer_command_form"
                    )
                    
                    rejected_cmd = st.text_area(
                        "❌ **Rejected Command** (Inferior/Incorrect):",
                        height=100,
                        key="rejected_proposer_command_form",
                        help="Example: A less efficient, incorrect, or unsafe command."
                    )
                    
                    # Auto-generate DPO prompt from goal
                    try:
                        input_data = json.loads(st.session_state.selected_log_entry.get("Input Summary", "{}"))
                        goal = input_data.get('goal', 'N/A')
                    except json.JSONDecodeError:
                        goal = st.session_state.selected_log_entry.get("Input Summary", "N/A")
                    
                    auto_generated_dpo_prompt = f"Given the goal: '{goal}', propose a command."
                    
                    prompt_context = st.text_area(
                        "📝 **Prompt/Context for DPO** (Auto-generated, editable):",
                        value=auto_generated_dpo_prompt,
                        height=150,
                        key="proposer_dpo_prompt_form",
                        help="This will be the 'prompt' for the DPO trainer..."
                    )
                    
                    st.form_submit_button("💾 Save Proposer Annotation")
            else:
                st.info("Annotation is only available for logs from Proposer agents. The selected log is from a different agent type.")


def render_log_explorer_view():
    """
    Render the log explorer & annotation view.
    
    This function implements UI elements for log display and filtering.
    For MVP, this uses placeholder data instead of actual database interaction.
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