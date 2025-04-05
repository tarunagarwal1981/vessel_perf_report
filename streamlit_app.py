import streamlit as st
import pandas as pd
import numpy as np
import datetime
import requests
import plotly.graph_objects as go
import plotly.express as px

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

# Function to create hull roughness chart with a neon color gradient
def create_hull_roughness_chart(data):
    if not data:
        return None, None
    
    # Prepare data - ensure dates are in datetime format
    dates = [pd.to_datetime(row['report_date']) for row in data]
    hull_roughness = [row.get('hull_roughness_power_loss', 0) for row in data]
    
    # Sort data by date for proper trend line
    sorted_indices = np.argsort(dates)
    dates_sorted = [dates[i] for i in sorted_indices]
    hull_roughness_sorted = [hull_roughness[i] for i in sorted_indices]
    
    # Create a color gradient based on dates
    # Convert dates to numeric values for the colorscale
    date_nums = [(d - min(dates)).total_seconds() / 86400 for d in dates]  # Days since earliest date
    
    # Create the figure
    fig = go.Figure()
    
    # Add scatter plot with neon color gradient (markers only, no lines)
    fig.add_trace(go.Scatter(
        x=dates,
        y=hull_roughness,
        mode='markers',  # Only markers, no lines
        marker=dict(
            size=10,
            color=date_nums,
            colorscale='Plasma',  # Neon-like colorscale
            showscale=True,
            colorbar=dict(
                title="Days",
                titleside="right"
            )
        ),
        name='Hull Roughness Data'
    ))
    
    # Calculate linear best fit if we have enough data points
    latest_hull_roughness = None
    if len(dates_sorted) > 1:
        # Convert dates to numeric for fitting
        x_numeric = np.array([(d - dates_sorted[0]).total_seconds() / 86400 for d in dates_sorted])
        y = np.array(hull_roughness_sorted)
        
        # Fit a line
        coeffs = np.polyfit(x_numeric, y, 1)
        slope = coeffs[0]
        intercept = coeffs[1]
        
        # Generate points for the trend line
        x_line = np.array([0, x_numeric[-1]])
        y_line = slope * x_line + intercept
        
        # Convert back to datetime for plotting
        x_line_dates = [dates_sorted[0] + datetime.timedelta(days=float(x)) for x in x_line]
        
        # Add the best fit line
        fig.add_trace(go.Scatter(
            x=x_line_dates,
            y=y_line,
            mode='lines',
            line=dict(color='#ff006e', width=3),  # Bright pink line
            name='Trend Line'
        ))
        
        # Get the latest hull roughness value from the trend line
        latest_hull_roughness = y_line[-1]
        
        # Add threshold lines for condition assessment
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
        
        # Add annotations for the thresholds
        fig.add_annotation(
            x=max(dates),
            y=15,
            text="15% - Average condition",
            showarrow=False,
            yshift=10,
            xshift=-100,
            font=dict(color="yellow")
        )
        
        fig.add_annotation(
            x=max(dates),
            y=25,
            text="25% - Poor condition",
            showarrow=False,
            yshift=10,
            xshift=-100,
            font=dict(color="red")
        )
        
        # Add annotation for the latest value
        fig.add_annotation(
            x=max(dates),
            y=latest_hull_roughness,
            text=f"Latest: {latest_hull_roughness:.2f}%",
            showarrow=True,
            arrowhead=1,
            arrowcolor="#ff006e",
            arrowsize=1,
            arrowwidth=2,
            ax=-40,
            ay=-40,
            font=dict(color="#ff006e", size=14)
        )
    
    # Update layout
    fig.update_layout(
        title="Hull Roughness Power Loss Trend",
        xaxis_title="Date",
        yaxis_title="Hull Roughness Power Loss (%)",
        template="plotly_dark",
        height=600,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )
    
    return fig, latest_hull_roughness

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
                # Ensure all entries have the necessary fields
                filtered_data = []
                for entry in hull_data:
                    if 'report_date' in entry and 'hull_roughness_power_loss' in entry:
                        if entry['hull_roughness_power_loss'] is not None:
                            filtered_data.append(entry)
                
                if not filtered_data:
                    st.error(f"No valid hull performance data found for {selected_vessel}.")
                    st.stop()
                
                # Sort data by date
                filtered_data.sort(key=lambda x: pd.to_datetime(x['report_date']))
                
                # Display hull roughness chart
                st.subheader("Hull Roughness Trend")
                
                try:
                    chart, latest_hull_roughness = create_hull_roughness_chart(filtered_data)
                    if chart:
                        st.plotly_chart(chart, use_container_width=True)
                        
                        # Create two columns for metrics
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            # Use the trend line's last value instead of the raw data point
                            if latest_hull_roughness is not None:
                                condition, color = get_hull_condition(latest_hull_roughness)
                                st.metric("Current Hull Roughness (from trend)", f"{latest_hull_roughness:.2f}%")
                                st.markdown(f"<h3 style='color:{color}'>Hull Condition: {condition}</h3>", unsafe_allow_html=True)
                        
                        with col2:
                            # Calculate any trends or recommendations
                            if latest_hull_roughness is not None:
                                if latest_hull_roughness < 15:
                                    recommendation = "No action required"
                                elif 15 <= latest_hull_roughness < 25:
                                    recommendation = "Consider hull cleaning at next convenient opportunity"
                                else:
                                    recommendation = "Hull cleaning recommended as soon as possible"
                                
                                st.markdown(f"### Recommendation\n{recommendation}")
                except Exception as e:
                    st.error(f"Error creating chart: {str(e)}")
                
                # Display the raw data
                with st.expander("View Raw Data"):
                    # Convert to DataFrame for display
                    df = pd.DataFrame(filtered_data)
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
