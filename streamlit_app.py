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
def create_performance_chart(data, metric_name, chart_title, y_axis_title):
    if not data:
        return None, None
    
    # Prepare data - ensure dates are in datetime format
    dates = [pd.to_datetime(row['report_date']) for row in data]
    metric_values = [row.get(metric_name, 0) for row in data]
    
    # Sort data by date for proper trend line
    sorted_indices = np.argsort(dates)
    dates_sorted = [dates[i] for i in sorted_indices]
    metric_values_sorted = [metric_values[i] for i in sorted_indices]
    
    # Create a color gradient based on dates
    # Convert dates to numeric values for the colorscale
    date_nums = [(d - min(dates)).total_seconds() / 86400 for d in dates]  # Days since earliest date
    
    # Create the figure
    fig = go.Figure()
    
    # Add scatter plot with neon color gradient (markers only, no lines)
    fig.add_trace(go.Scatter(
        x=dates,
        y=metric_values,
        mode='markers',  # Only markers, no lines
        marker=dict(
            size=10,
            color=date_nums,
            colorscale='Plasma',  # Neon-like colorscale
            showscale=True,
            colorbar=dict(
                title="Days Since First Data Point"
            )
        ),
        name='Performance Data'
    ))
    
    # Calculate linear best fit if we have enough data points
    latest_value = None
    if len(dates_sorted) > 1:
        # Convert dates to numeric for fitting
        x_numeric = np.array([(d - dates_sorted[0]).total_seconds() / 86400 for d in dates_sorted])
        y = np.array(metric_values_sorted)
        
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
        
        # Get the latest value from the trend line
        latest_value = y_line[-1]
        
        # For hull roughness power loss, add threshold lines
        if metric_name == 'hull_roughness_power_loss':
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
            y=latest_value,
            text=f"Latest: {latest_value:.2f}%",
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
        title=chart_title,
        xaxis_title="Date",
        yaxis_title=y_axis_title,
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
    
    return fig, latest_value

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

# Weather condition filter
st.sidebar.subheader("Weather Filter")
weather_option = st.sidebar.radio(
    "Select Weather Condition",
    options=["All Weather", "Good Weather (Wind Force ≤ 4)", "Bad Weather (Wind Force > 4)"],
    index=1  # Default to good weather
)

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
                # Filter data based on weather option
                if weather_option == "Good Weather (Wind Force ≤ 4)":
                    hull_data = [row for row in hull_data if row.get('windforce', 0) <= 4]
                elif weather_option == "Bad Weather (Wind Force > 4)":
                    hull_data = [row for row in hull_data if row.get('windforce', 0) > 4]
                
                # Ensure all entries have the necessary fields
                filtered_data_power = []
                filtered_data_speed = []
                
                for entry in hull_data:
                    if 'report_date' in entry and 'hull_roughness_power_loss' in entry:
                        if entry['hull_roughness_power_loss'] is not None:
                            filtered_data_power.append(entry)
                    
                    if 'report_date' in entry and 'hull_roughness_speed_loss' in entry:
                        if entry['hull_roughness_speed_loss'] is not None:
                            filtered_data_speed.append(entry)
                
                if not filtered_data_power and not filtered_data_speed:
                    st.error(f"No valid hull performance data found for {selected_vessel} with the selected filters.")
                    st.stop()
                
                # Create tabs for different charts
                tab1, tab2 = st.tabs(["Excess Power Chart", "Speed Loss Chart"])
                
                # Excess Power Tab
                with tab1:
                    if filtered_data_power:
                        # Sort data by date
                        filtered_data_power.sort(key=lambda x: pd.to_datetime(x['report_date']))
                        
                        # Display hull roughness power loss chart
                        st.subheader("Hull Roughness - Excess Power")
                        
                        try:
                            chart, latest_power_loss = create_performance_chart(
                                filtered_data_power, 
                                'hull_roughness_power_loss',
                                "Hull Roughness - Excess Power Trend",
                                "Excess Power (%)"
                            )
                            
                            if chart:
                                st.plotly_chart(chart, use_container_width=True)
                                
                                # Create two columns for metrics
                                col1, col2 = st.columns(2)
                                
                                with col1:
                                    # Use the trend line's last value instead of the raw data point
                                    if latest_power_loss is not None:
                                        condition, color = get_hull_condition(latest_power_loss)
                                        st.metric("Current Excess Power (from trend)", f"{latest_power_loss:.2f}%")
                                        st.markdown(f"<h3 style='color:{color}'>Hull Condition: {condition}</h3>", unsafe_allow_html=True)
                                
                                with col2:
                                    # Calculate any trends or recommendations
                                    if latest_power_loss is not None:
                                        if latest_power_loss < 15:
                                            recommendation = "No action required"
                                        elif 15 <= latest_power_loss < 25:
                                            recommendation = "Consider hull cleaning at next convenient opportunity"
                                        else:
                                            recommendation = "Hull cleaning recommended as soon as possible"
                                        
                                        st.markdown(f"### Recommendation\n{recommendation}")
                        except Exception as e:
                            st.error(f"Error creating Excess Power chart: {str(e)}")
                    else:
                        st.info("No valid hull roughness power loss data available for the selected filters.")
                
                # Speed Loss Tab
                with tab2:
                    if filtered_data_speed:
                        # Sort data by date
                        filtered_data_speed.sort(key=lambda x: pd.to_datetime(x['report_date']))
                        
                        # Display hull roughness speed loss chart
                        st.subheader("Hull Roughness - Speed Loss")
                        
                        try:
                            chart, latest_speed_loss = create_performance_chart(
                                filtered_data_speed, 
                                'hull_roughness_speed_loss',
                                "Hull Roughness - Speed Loss Trend",
                                "Speed Loss (%)"
                            )
                            
                            if chart:
                                st.plotly_chart(chart, use_container_width=True)
                                
                                # Show the latest value from trend
                                if latest_speed_loss is not None:
                                    st.metric("Current Speed Loss (from trend)", f"{latest_speed_loss:.2f}%")
                                    
                                    # Calculate impact on vessel performance
                                    st.markdown("### Performance Impact")
                                    st.markdown(f"""
                                    The current speed loss of {latest_speed_loss:.2f}% means that:
                                    
                                    - At the same power, the vessel's speed is reduced by {latest_speed_loss:.2f}%
                                    - To maintain the same speed, the vessel needs to increase power by approximately {latest_power_loss:.2f}% (from Excess Power tab)
                                    """)
                        except Exception as e:
                            st.error(f"Error creating Speed Loss chart: {str(e)}")
                    else:
                        st.info("No valid hull roughness speed loss data available for the selected filters.")
                
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
