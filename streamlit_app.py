import streamlit as st
import pandas as pd
import numpy as np
import datetime
import requests
import plotly.graph_objects as go

# Set page configuration
st.set_page_config(
    page_title="Marine Performance Analysis",
    page_icon="🚢",
    layout="wide"
)

# Lambda URL
LAMBDA_URL = "https://crcgfvseuzhdqhhvan5gz2hr4e0kirfy.lambda-url.ap-south-1.on.aws/"

# Function to call Lambda API
def fetch_data_from_lambda(operation, params):
    try:
        payload = {
            "operation": operation,
            **params
        }
        
        response = requests.post(
            LAMBDA_URL,
            json=payload,
            headers={"Content-Type": "application/json"}
        )
        
        if response.status_code == 200:
            result = response.json()
            if result.get("success"):
                return result.get("data", [])
            else:
                st.error(f"API Error: {result.get('error', 'Unknown error')}")
                return None
        else:
            st.error(f"HTTP Error: {response.status_code}")
            return None
    except Exception as e:
        st.error(f"Error connecting to Lambda: {str(e)}")
        return None

# Function to create hull roughness chart
def create_hull_roughness_chart(data):
    if not data:
        return None
    
    # Prepare data
    dates = [pd.to_datetime(row['report_date']) for row in data]
    hull_roughness = [row.get('hull_roughness_power_loss', 0) for row in data]
    
    # Create figure
    fig = go.Figure()
    
    # Add scatter plot
    fig.add_trace(go.Scatter(
        x=dates,
        y=hull_roughness,
        mode='markers+lines',
        marker=dict(
            size=8,
            color=range(len(dates)),
            colorscale='Plasma'
        ),
        name='Hull Roughness'
    ))
    
    # Add threshold lines
    if dates:
        fig.add_shape(
            type="line",
            x0=min(dates),
            y0=15,
            x1=max(dates),
            y1=15,
            line=dict(color="yellow", width=2, dash="dash"),
        )
        
        fig.add_shape(
            type="line",
            x0=min(dates),
            y0=25,
            x1=max(dates),
            y1=25,
            line=dict(color="red", width=2, dash="dash"),
        )
        
        # Add annotations
        fig.add_annotation(
            x=max(dates),
            y=15,
            text="15% - Average condition",
            showarrow=False,
            yshift=10,
            font=dict(color="yellow")
        )
        
        fig.add_annotation(
            x=max(dates),
            y=25,
            text="25% - Poor condition",
            showarrow=False,
            yshift=10,
            font=dict(color="red")
        )
    
    # Update layout
    fig.update_layout(
        title="Hull Roughness Power Loss Trend",
        xaxis_title="Date",
        yaxis_title="Hull Roughness Power Loss (%)",
        template="plotly_dark",
        height=500
    )
    
    return fig

# Calculate hull condition status
def get_hull_condition(hull_roughness):
    if hull_roughness < 15:
        return "GOOD", "green"
    elif 15 <= hull_roughness < 25:
        return "AVERAGE", "orange"
    else:
        return "POOR", "red"

# App title
st.title("🚢 Marine Hull Performance Analysis")

# Sidebar for vessel input and date range
st.sidebar.header("Configuration")

# User input for vessel name
selected_vessel = st.sidebar.text_input("Enter Vessel Name", "")

# Date selectors
st.sidebar.subheader("Date Range")
end_date = datetime.datetime.now()
start_date = end_date - datetime.timedelta(days=180)  # Default to last 6 months

start_date = st.sidebar.date_input("Start Date", value=start_date)
end_date = st.sidebar.date_input("End Date", value=end_date)

# Check if dates are valid
if start_date > end_date:
    st.error("Error: Start date must be before end date")
elif not selected_vessel:
    st.info("Please enter a vessel name in the sidebar.")
else:
    # Button to fetch data
    if st.sidebar.button("Analyze Hull Performance"):
        with st.spinner(f"Fetching hull performance data for {selected_vessel}..."):
            # Call Lambda function to get hull performance data
            params = {
                "vesselName": selected_vessel,
                "startDate": start_date.isoformat(),
                "endDate": end_date.isoformat()
            }
            
            hull_data = fetch_data_from_lambda("getHullPerformance", params)
            
            if hull_data and len(hull_data) > 0:
                # Sort data by date
                hull_data.sort(key=lambda x: x['report_date'])
                
                # Create two columns for metrics
                col1, col2 = st.columns(2)
                
                # Calculate current hull condition
                # Get the latest hull roughness value
                latest_data = max(hull_data, key=lambda x: x['report_date'])
                latest_roughness = latest_data.get('hull_roughness_power_loss', 0)
                condition, color = get_hull_condition(latest_roughness)
                
                with col1:
                    st.metric("Latest Hull Roughness", f"{latest_roughness:.2f}%")
                    st.markdown(f"<h3 style='color:{color}'>Hull Condition: {condition}</h3>", unsafe_allow_html=True)
                
                with col2:
                    # Calculate any trends or recommendations
                    if condition == "GOOD":
                        recommendation = "No action required"
                    elif condition == "AVERAGE":
                        recommendation = "Consider hull cleaning at next convenient opportunity"
                    else:
                        recommendation = "Hull cleaning recommended as soon as possible"
                    
                    st.markdown(f"### Recommendation\n{recommendation}")
                
                # Display hull roughness chart
                st.subheader("Hull Roughness Trend")
                chart = create_hull_roughness_chart(hull_data)
                if chart:
                    st.plotly_chart(chart, use_container_width=True)
                
                # Display the raw data
                with st.expander("View Raw Data"):
                    # Convert to DataFrame for display
                    df = pd.DataFrame(hull_data)
                    if 'report_date' in df.columns:
                        df['report_date'] = pd.to_datetime(df['report_date'])
                    st.dataframe(df)
            else:
                st.error(f"No hull performance data available for {selected_vessel} in the selected date range.")
    else:
        # Initial instruction
        st.info("👈 Enter a vessel name and select a date range, then click 'Analyze Hull Performance' to view the analysis.")

# Footer
st.sidebar.markdown("---")
st.sidebar.info("Marine Performance Analysis System v1.0")
