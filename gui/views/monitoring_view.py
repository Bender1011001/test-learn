import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import requests
import json
from datetime import datetime, timedelta
import time

def render_monitoring_view():
    """Render the monitoring dashboard view"""
    st.title("System Monitoring Dashboard")
    
    # Create tabs for different monitoring views
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "API Performance", 
        "Database Performance", 
        "Cache Performance", 
        "Error Tracking",
        "System Health"
    ])
    
    # Get API base URL from session state
    api_base_url = st.session_state.get("api_base_url", "http://localhost:8000/api")
    
    # Function to fetch metrics data
    @st.cache_data(ttl=10)  # Cache for 10 seconds
    def fetch_metrics(endpoint=""):
        try:
            url = f"{api_base_url}/metrics{endpoint}"
            response = requests.get(url)
            if response.status_code == 200:
                return response.json()
            else:
                st.error(f"Error fetching metrics: {response.status_code}")
                return {}
        except Exception as e:
            st.error(f"Error connecting to API: {str(e)}")
            return {}
    
    # Function to fetch health data
    @st.cache_data(ttl=10)  # Cache for 10 seconds
    def fetch_health():
        try:
            url = f"{api_base_url}/metrics/health"
            response = requests.get(url)
            if response.status_code == 200:
                return response.json()
            else:
                st.error(f"Error fetching health data: {response.status_code}")
                return {}
        except Exception as e:
            st.error(f"Error connecting to API: {str(e)}")
            return {}
    
    # API Performance Tab
    with tab1:
        st.header("API Endpoint Performance")
        
        # Fetch API metrics
        api_metrics = fetch_metrics("/api")
        
        if api_metrics and "api_latency" in api_metrics:
            # Create a DataFrame for the API metrics
            api_data = []
            for endpoint, metrics in api_metrics["api_latency"].items():
                api_data.append({
                    "Endpoint": endpoint,
                    "Count": metrics.get("count", 0),
                    "Min (ms)": round(metrics.get("min", 0) * 1000, 2),
                    "Max (ms)": round(metrics.get("max", 0) * 1000, 2),
                    "Avg (ms)": round(metrics.get("avg", 0) * 1000, 2),
                    "P95 (ms)": round(metrics.get("p95", 0) * 1000, 2),
                    "P99 (ms)": round(metrics.get("p99", 0) * 1000, 2)
                })
            
            if api_data:
                df_api = pd.DataFrame(api_data)
                
                # Sort by average latency
                df_api = df_api.sort_values("Avg (ms)", ascending=False)
                
                # Display the table
                st.dataframe(df_api, use_container_width=True)
                
                # Create a bar chart for average latency
                fig = px.bar(
                    df_api, 
                    x="Endpoint", 
                    y="Avg (ms)",
                    title="Average API Latency by Endpoint",
                    labels={"Endpoint": "Endpoint", "Avg (ms)": "Average Latency (ms)"},
                    color="Avg (ms)",
                    color_continuous_scale="Viridis"
                )
                st.plotly_chart(fig, use_container_width=True)
                
                # Create a bar chart for request count
                fig = px.bar(
                    df_api, 
                    x="Endpoint", 
                    y="Count",
                    title="API Request Count by Endpoint",
                    labels={"Endpoint": "Endpoint", "Count": "Request Count"},
                    color="Count",
                    color_continuous_scale="Viridis"
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No API metrics data available yet.")
        else:
            st.info("No API metrics data available yet.")
    
    # Database Performance Tab
    with tab2:
        st.header("Database Operation Performance")
        
        # Fetch DB metrics
        db_metrics = fetch_metrics("/db")
        
        if db_metrics and "db_latency" in db_metrics:
            # Create a DataFrame for the DB metrics
            db_data = []
            for operation, metrics in db_metrics["db_latency"].items():
                db_data.append({
                    "Operation": operation,
                    "Count": metrics.get("count", 0),
                    "Min (ms)": round(metrics.get("min", 0) * 1000, 2),
                    "Max (ms)": round(metrics.get("max", 0) * 1000, 2),
                    "Avg (ms)": round(metrics.get("avg", 0) * 1000, 2),
                    "P95 (ms)": round(metrics.get("p95", 0) * 1000, 2),
                    "P99 (ms)": round(metrics.get("p99", 0) * 1000, 2)
                })
            
            if db_data:
                df_db = pd.DataFrame(db_data)
                
                # Sort by average latency
                df_db = df_db.sort_values("Avg (ms)", ascending=False)
                
                # Display the table
                st.dataframe(df_db, use_container_width=True)
                
                # Create a bar chart for average latency
                fig = px.bar(
                    df_db, 
                    x="Operation", 
                    y="Avg (ms)",
                    title="Average DB Operation Latency",
                    labels={"Operation": "Operation", "Avg (ms)": "Average Latency (ms)"},
                    color="Avg (ms)",
                    color_continuous_scale="Viridis"
                )
                st.plotly_chart(fig, use_container_width=True)
                
                # Create a bar chart for operation count
                fig = px.bar(
                    df_db, 
                    x="Operation", 
                    y="Count",
                    title="DB Operation Count",
                    labels={"Operation": "Operation", "Count": "Operation Count"},
                    color="Count",
                    color_continuous_scale="Viridis"
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No database metrics data available yet.")
        else:
            st.info("No database metrics data available yet.")
    
    # Cache Performance Tab
    with tab3:
        st.header("Cache Performance")
        
        # Fetch cache metrics
        cache_metrics = fetch_metrics("/cache")
        
        if cache_metrics and "cache" in cache_metrics:
            cache_data = cache_metrics["cache"]
            
            # Create hit rate chart
            if "hit_rates" in cache_data:
                hit_rates = cache_data["hit_rates"]
                
                if hit_rates:
                    # Create a DataFrame for the hit rates
                    hit_rate_data = []
                    for key, rate in hit_rates.items():
                        hit_rate_data.append({
                            "Cache Key": key,
                            "Hit Rate": round(rate * 100, 2)
                        })
                    
                    df_hit_rates = pd.DataFrame(hit_rate_data)
                    
                    # Sort by hit rate
                    df_hit_rates = df_hit_rates.sort_values("Hit Rate", ascending=False)
                    
                    # Display the table
                    st.dataframe(df_hit_rates, use_container_width=True)
                    
                    # Create a bar chart for hit rates
                    fig = px.bar(
                        df_hit_rates, 
                        x="Cache Key", 
                        y="Hit Rate",
                        title="Cache Hit Rates by Key",
                        labels={"Cache Key": "Cache Key", "Hit Rate": "Hit Rate (%)"},
                        color="Hit Rate",
                        color_continuous_scale="Viridis"
                    )
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Create a gauge chart for overall hit rate
                    overall_hit_rate = sum(hit_rates.values()) / len(hit_rates) * 100 if hit_rates else 0
                    
                    fig = go.Figure(go.Indicator(
                        mode="gauge+number",
                        value=overall_hit_rate,
                        title={"text": "Overall Cache Hit Rate"},
                        gauge={
                            "axis": {"range": [0, 100]},
                            "bar": {"color": "darkblue"},
                            "steps": [
                                {"range": [0, 50], "color": "red"},
                                {"range": [50, 80], "color": "yellow"},
                                {"range": [80, 100], "color": "green"}
                            ],
                            "threshold": {
                                "line": {"color": "black", "width": 4},
                                "thickness": 0.75,
                                "value": 80
                            }
                        }
                    ))
                    
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info("No cache hit rate data available yet.")
            else:
                st.info("No cache hit rate data available yet.")
        else:
            st.info("No cache metrics data available yet.")
    
    # Error Tracking Tab
    with tab4:
        st.header("Error Tracking")
        
        # Fetch error metrics
        error_metrics = fetch_metrics("/errors")
        
        if error_metrics and "errors" in error_metrics:
            errors = error_metrics["errors"]
            
            if errors:
                # Create a DataFrame for the errors
                error_data = []
                for error_key, count in errors.items():
                    error_type, source = error_key.split(":", 1) if ":" in error_key else (error_key, "unknown")
                    error_data.append({
                        "Error Type": error_type,
                        "Source": source,
                        "Count": count
                    })
                
                df_errors = pd.DataFrame(error_data)
                
                # Sort by count
                df_errors = df_errors.sort_values("Count", ascending=False)
                
                # Display the table
                st.dataframe(df_errors, use_container_width=True)
                
                # Create a bar chart for error counts
                fig = px.bar(
                    df_errors, 
                    x="Error Type", 
                    y="Count",
                    title="Error Counts by Type",
                    labels={"Error Type": "Error Type", "Count": "Error Count"},
                    color="Source",
                    color_discrete_sequence=px.colors.qualitative.Plotly
                )
                st.plotly_chart(fig, use_container_width=True)
                
                # Create a pie chart for error distribution
                fig = px.pie(
                    df_errors, 
                    values="Count", 
                    names="Error Type",
                    title="Error Distribution by Type"
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.success("No errors recorded. The system is running smoothly!")
        else:
            st.success("No errors recorded. The system is running smoothly!")
    
    # System Health Tab
    with tab5:
        st.header("System Health")
        
        # Fetch health metrics
        health_data = fetch_health()
        
        if health_data:
            # Display overall status
            status = health_data.get("status", "unknown")
            if status == "healthy":
                st.success("System Status: Healthy")
            elif status == "degraded":
                st.warning("System Status: Degraded")
            else:
                st.error(f"System Status: {status}")
            
            # Display component status
            st.subheader("Component Status")
            components = health_data.get("components", {})
            
            for component, status_data in components.items():
                component_status = status_data.get("status", "unknown")
                if component_status == "healthy":
                    st.success(f"{component.capitalize()}: Healthy")
                elif component_status == "degraded":
                    st.warning(f"{component.capitalize()}: Degraded")
                else:
                    st.error(f"{component.capitalize()}: {component_status}")
                    if "error" in status_data and status_data["error"]:
                        st.error(f"Error: {status_data['error']}")
            
            # Display system metrics
            st.subheader("System Metrics")
            system_data = health_data.get("system", {})
            
            if system_data:
                # Create columns for system metrics
                col1, col2 = st.columns(2)
                
                with col1:
                    # CPU Usage
                    cpu_usage = system_data.get("cpu_usage", 0)
                    st.metric("CPU Usage", f"{cpu_usage}%")
                    
                    # Memory Usage
                    memory_usage = system_data.get("memory_usage", {})
                    if memory_usage:
                        memory_percent = memory_usage.get("percent", 0)
                        st.metric("Memory Usage", f"{memory_percent}%")
                        
                        # Convert bytes to GB for display
                        memory_total = memory_usage.get("total", 0) / (1024 ** 3)
                        memory_used = memory_usage.get("used", 0) / (1024 ** 3)
                        st.metric("Memory Used/Total", f"{memory_used:.2f} GB / {memory_total:.2f} GB")
                
                with col2:
                    # Disk Usage
                    disk_usage = system_data.get("disk_usage", {})
                    if disk_usage:
                        disk_percent = disk_usage.get("percent", 0)
                        st.metric("Disk Usage", f"{disk_percent}%")
                        
                        # Convert bytes to GB for display
                        disk_total = disk_usage.get("total", 0) / (1024 ** 3)
                        disk_used = disk_usage.get("used", 0) / (1024 ** 3)
                        st.metric("Disk Used/Total", f"{disk_used:.2f} GB / {disk_total:.2f} GB")
                    
                    # Process Info
                    process_info = system_data.get("process", {})
                    if process_info:
                        process_cpu = process_info.get("cpu_percent", 0)
                        st.metric("Process CPU Usage", f"{process_cpu}%")
                        
                        process_memory = process_info.get("memory_percent", 0)
                        st.metric("Process Memory Usage", f"{process_memory}%")
                
                # Platform info
                st.text(f"Platform: {system_data.get('platform', 'unknown')}")
                st.text(f"Python Version: {system_data.get('python_version', 'unknown')}")
            
            # Display metrics
            st.subheader("Application Metrics")
            metrics_data = health_data.get("metrics", {})
            
            if metrics_data:
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    request_count = metrics_data.get("request_count", 0)
                    st.metric("Total Requests", request_count)
                
                with col2:
                    error_count = metrics_data.get("error_count", 0)
                    st.metric("Total Errors", error_count)
                
                with col3:
                    error_rate = metrics_data.get("error_rate", 0) * 100
                    st.metric("Error Rate", f"{error_rate:.2f}%")
            
            # Display uptime
            uptime = health_data.get("uptime", "unknown")
            st.metric("Uptime", uptime)
            
            # Display timestamp
            timestamp = health_data.get("timestamp", "unknown")
            st.text(f"Last Updated: {timestamp}")
        else:
            st.error("Unable to fetch health data")
    
    # Add a refresh button
    if st.button("Refresh Metrics"):
        st.experimental_rerun()
    
    # Add auto-refresh option
    auto_refresh = st.checkbox("Auto-refresh (10s)")
    if auto_refresh:
        time.sleep(10)
        st.experimental_rerun()