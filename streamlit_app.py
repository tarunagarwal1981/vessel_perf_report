import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime, timedelta
import requests
import json
import io
import os
from PIL import Image
from io import BytesIO
import tempfile
import warnings
import plotly.express as px
import plotly.graph_objects as go

# Suppress warnings
warnings.filterwarnings("ignore", category=UserWarning, module="openpyxl")

# Set page configuration
st.set_page_config(
    page_title="Marine Performance Analysis",
    page_icon="🚢",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom styling
st.markdown("""
<style>
    .main {
        background-color: #0E1117;
    }
    .stApp {
        max-width: 1200px;
        margin: 0 auto;
    }
    h1, h2, h3 {
        color: #48CAE4;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 24px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: pre-wrap;
        background-color: #0E1117;
        border-radius: 4px 4px 0px 0px;
        gap: 1px;
        padding-top: 10px;
        padding-bottom: 10px;
    }
    .stTabs [aria-selected="true"] {
        background-color: #1E293B;
        color: #48CAE4;
    }
    .reportview-container .markdown-text-container {
        font-family: sans-serif;
    }
    .reportview-container .main .block-container {
        max-width: 1200px;
        padding-top: 2rem;
        padding-right: 2rem;
        padding-left: 2rem;
        padding-bottom: 2rem;
    }
    .metric-container {
        background-color: #1E293B;
        border-radius: 5px;
        padding: 15px;
        margin-bottom: 10px;
    }
    .metric-value {
        font-size: 24px;
        font-weight: bold;
    }
    .metric-label {
        font-size: 14px;
        color: #ADB5BD;
    }
</style>
""", unsafe_allow_html=True)

# Lambda connection configuration
LAMBDA_URL = "https://crcgfvseuzhdqhhvan5gz2hr4e0kirfy.lambda-url.ap-south-1.on.aws/"

# Function to call Lambda function via API URL
def call_lambda_api(operation, params=None):
    """Call Lambda function via API URL endpoint"""
    try:
        payload = {
            "operation": operation
        }
        
        # Add any additional parameters
        if params:
            payload.update(params)
            
        response = requests.post(
            LAMBDA_URL, 
            json=payload,
            headers={"Content-Type": "application/json"}
        )
        
        if response.status_code != 200:
            st.error(f"Error calling Lambda function: {response.status_code}")
            st.error(f"Response: {response.text}")
            return None
            
        result = response.json()
        
        if "error" in result:
            st.error(f"Error from Lambda: {result['error']}")
            return None
            
        return result
    except Exception as e:
        st.error(f"Error connecting to Lambda: {str(e)}")
        return None

# Function to fetch data - doesn't use caching with widgets inside
def fetch_data(operation, params=None):
    """Fetch data from Lambda function"""
    result = call_lambda_api(operation, params)
    
    if result is None or not result.get("success", False):
        st.error(f"Failed to fetch data for operation: {operation}")
        
        # Offer sample data option outside the cached function
        return None
    
    return result.get("data", [])

# Sample data loader for development/demo
def load_sample_data(data_type):
    """Load sample data when API is unavailable"""
    try:
        if data_type == "getVessels":
            # Return a list of vessel names
            return ["VESSEL_A", "VESSEL_B", "VESSEL_C"]
        
        elif data_type == "getHullPerformance":
            # Create sample data for hull performance
            dates = pd.date_range(end=datetime.now(), periods=100, freq='D')
            
            data = []
            for i, date in enumerate(dates):
                # Simulating increasing hull roughness
                hull_roughness = 10 + i * 0.15 + np.random.randint(-3, 3)
                excess_fuel = 2 + i * 0.05 + np.random.randint(-1, 1)
                
                # Alternate between ballast and laden
                loading_condition = "BALLAST" if i % 2 == 0 else "LADEN"
                
                # Random speed within reasonable range
                speed = 12 + np.random.randint(-2, 3)
                
                # Calculate consumption based on speed and some randomness
                consumption = 30 + speed * 1.5 + np.random.randint(-5, 5)
                
                data.append({
                    "vessel_name": "SAMPLE_VESSEL",
                    "report_date": date.isoformat(),
                    "hull_roughness_power_loss": hull_roughness,
                    "hull_excess_fuel_oil_mtd": excess_fuel,
                    "loading_condition": loading_condition,
                    "speed": speed,
                    "windforce": np.random.randint(0, 5),
                    "normalised_consumption": consumption
                })
            
            return data
        
        elif data_type == "getLastDrydockDate":
            # Return a date 2 years ago
            return (datetime.now() - timedelta(days=730)).isoformat()
            
        else:
            st.warning(f"Unknown sample data type requested: {data_type}")
            return None
    except Exception as e:
        st.error(f"Error loading sample data: {str(e)}")
        return None

# Create neon color gradient for charts
def create_neon_color_gradient(dates):
    """Create a neon color gradient based on date range"""
    # Convert string dates to datetime if needed
    if isinstance(dates[0], str):
        dates = [pd.to_datetime(d) for d in dates]
        
    norm = plt.Normalize(min(d.toordinal() for d in dates), max(d.toordinal() for d in dates))
    cmap = plt.get_cmap('plasma')
    return [cmap(norm(d.toordinal())) for d in dates]

# Polynomial fitting function
def polynomial_fit(x, y, degree):
    """Calculate polynomial fit for the given data"""
    coeffs = np.polyfit(x, y, degree)
    return np.poly1d(coeffs)

# Exponential fitting function
def exponential_fit(x, y):
    """Calculate exponential fit for the given data"""
    # Handle zero or negative values
    valid_indices = np.where(y > 0)[0]
    if len(valid_indices) < 2:
        # Fall back to polynomial if not enough valid points
        return polynomial_fit(x, y, 1)
        
    x_valid = np.array([x[i] for i in valid_indices])
    y_valid = np.array([y[i] for i in valid_indices])
    
    log_y = np.log(y_valid)
    coeffs = np.polyfit(x_valid, log_y, 1)
    return lambda x: np.exp(coeffs[1] + coeffs[0] * x)

# Calculate consumption function (placeholder - would connect to your API)
def calculate_consumption(speed, displacement, coefficients):
    """Calculate fuel consumption based on speed and displacement"""
    # This is a simplified placeholder - you should replace with your actual calculation
    consumption = (
        coefficients.get('speed_coeff', 0.5) * speed +
        coefficients.get('disp_coeff', 0.3) * displacement +
        coefficients.get('speed2_coeff', 0.05) * speed**2 +
        coefficients.get('disp2_coeff', 0.02) * displacement**2 +
        coefficients.get('intercept', 10)
    )
    return consumption

# Function to create interactive Plotly charts for speed vs consumption
def create_interactive_chart(data, condition, coefficients=None):
    """Create an interactive chart for speed vs consumption analysis"""
    # Parse dates if they're strings
    if isinstance(data[0].get('report_date'), str):
        for row in data:
            row['report_date'] = pd.to_datetime(row['report_date'])
    
    # Filter data for the condition
    condition_data = [row for row in data if row['loading_condition'].lower() == condition.lower()]
    
    if not condition_data:
        return go.Figure().update_layout(title=f"No data available for {condition} condition")
    
    # Create figure
    fig = go.Figure()
    
    # Add scatter plot for actual data
    speeds = [row['speed'] for row in condition_data]
    consumptions = [row['normalised_consumption'] for row in condition_data]
    dates = [row['report_date'] for row in condition_data]
    
    # Create color scale based on dates
    date_colors = [pd.to_datetime(d).toordinal() for d in dates]
    
    fig.add_trace(go.Scatter(
        x=speeds,
        y=consumptions,
        mode='markers',
        marker=dict(
            size=10,
            color=date_colors,
            colorscale='Plasma',
            showscale=True,
            colorbar=dict(title="Date")
        ),
        name='Actual Data'
    ))
    
    # Fit curve to data if we have enough points
    if len(speeds) > 2:
        try:
            # Exponential fit
            exp_func = exponential_fit(speeds, consumptions)
            x_smooth = np.linspace(min(speeds), max(speeds), 100)
            y_smooth = [exp_func(x) for x in x_smooth]
            
            fig.add_trace(go.Scatter(
                x=x_smooth,
                y=y_smooth,
                mode='lines',
                line=dict(color='#48CAE4', width=3),
                name='Performance Trend'
            ))
        except Exception as e:
            st.warning(f"Could not create trend line: {str(e)}")
    
    # Add baseline/reference data if coefficients are available
    if coefficients:
        # Generate baseline consumption values
        displacement = 8.0 if condition.lower() == 'ballast' else 12.0  # Example values
        baseline_speeds = np.arange(8, 16, 1)
        baseline_consumptions = [calculate_consumption(speed, displacement, coefficients) for speed in baseline_speeds]
        
        fig.add_trace(go.Scatter(
            x=baseline_speeds,
            y=baseline_consumptions,
            mode='markers',
            marker=dict(size=10, color='#00FFFF', symbol='diamond'),
            name='Baseline'
        ))
        
        # Trend line for baseline
        if len(baseline_speeds) > 2:
            try:
                p_baseline = polynomial_fit(baseline_speeds, baseline_consumptions, 2)
                x_smooth_baseline = np.linspace(min(baseline_speeds), max(baseline_speeds), 100)
                y_smooth_baseline = p_baseline(x_smooth_baseline)
                
                fig.add_trace(go.Scatter(
                    x=x_smooth_baseline,
                    y=y_smooth_baseline,
                    mode='lines',
                    line=dict(color='#d9d9d9', width=2, dash='dash'),
                    name='Baseline Trend'
                ))
            except Exception as e:
                st.warning(f"Could not create baseline trend: {str(e)}")
    
    # Update layout
    fig.update_layout(
        title=f"{condition.capitalize()} Condition Analysis",
        xaxis_title="Speed (knots)",
        yaxis_title="ME Consumption (mT/day)",
        template="plotly_dark",
        hovermode="closest",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        height=600
    )
    
    return fig

# Function to create hull roughness chart
def create_hull_roughness_chart(data):
    """Create an interactive chart for hull roughness performance"""
    # Parse dates if they're strings
    if isinstance(data[0].get('report_date'), str):
        for row in data:
            row['report_date'] = pd.to_datetime(row['report_date'])
    
    # Filter out entries without hull roughness data
    filtered_data = [row for row in data if 'hull_roughness_power_loss' in row and row['hull_roughness_power_loss'] is not None]
    
    if not filtered_data:
        return go.Figure().update_layout(title="No hull roughness data available")
    
    # Sort by date
    filtered_data.sort(key=lambda x: x['report_date'])
    
    # Extract data
    dates = [row['report_date'] for row in filtered_data]
    hull_roughness = [row['hull_roughness_power_loss'] for row in filtered_data]
    
    # Create figure
    fig = go.Figure()
    
    # Add scatter plot for power loss data
    date_colors = [pd.to_datetime(d).toordinal() for d in dates]
    
    fig.add_trace(go.Scatter(
        x=dates,
        y=hull_roughness,
        mode='markers',
        marker=dict(
            size=10,
            color=date_colors,
            colorscale='Plasma',
            showscale=True,
            colorbar=dict(title="Date")
        ),
        name='Daily Data'
    ))
    
    # Add trend line
    if len(filtered_data) > 2:
        try:
            x_numeric = [pd.to_datetime(d).toordinal() for d in dates]
            z_power = np.polyfit(x_numeric, hull_roughness, 1)
            p_power = np.poly1d(z_power)
            
            x_trend = [min(dates), max(dates)]
            y_trend = p_power([pd.to_datetime(d).toordinal() for d in x_trend])
            
            fig.add_trace(go.Scatter(
                x=x_trend,
                y=y_trend,
                mode='lines',
                line=dict(color='#ff006e', width=3),
                name='Trend Line'
            ))
            
            # Add lines at 15% and 25% thresholds
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
            
        except Exception as e:
            st.warning(f"Could not create trend line: {str(e)}")
    
    # Update layout
    fig.update_layout(
        title="Hull Roughness Power Loss Trend",
        xaxis_title="Date",
        yaxis_title="Hull Roughness Power Loss (%)",
        template="plotly_dark",
        hovermode="closest",
        height=600
    )
    
    return fig

# Function to forecast hull cleaning date
def forecast_hull_cleaning_date(data, last_drydock_date):
    """Forecast the date when hull cleaning will be needed"""
    # Parse dates if they're strings
    if isinstance(data[0].get('report_date'), str):
        for row in data:
            row['report_date'] = pd.to_datetime(row['report_date'])
            
    if isinstance(last_drydock_date, str):
        last_drydock_date = pd.to_datetime(last_drydock_date)
    
    # Filter out entries without hull roughness data and sort by date
    filtered_data = [row for row in data if 'hull_roughness_power_loss' in row and row['hull_roughness_power_loss'] is not None]
    filtered_data.sort(key=lambda x: x['report_date'])
    
    if not filtered_data or len(filtered_data) < 2:
        # Not enough data to forecast
        return datetime.now() + timedelta(days=180)  # Default to 6 months
    
    # Extract data for fitting
    dates = [row['report_date'] for row in filtered_data]
    hull_roughness = [row['hull_roughness_power_loss'] for row in filtered_data]
    
    # Convert dates to ordinal numbers for fitting
    x = [d.toordinal() for d in dates]
    y = hull_roughness
    
    try:
        # Fit a line to the hull roughness power loss data
        coefficients = np.polyfit(x, y, 1)
        linear_fit = np.poly1d(coefficients)
        slope = coefficients[0]
        
        # Check current condition
        current_date = datetime.now()
        current_ordinal = current_date.toordinal()
        last_y = linear_fit(current_ordinal)
        
        # If already exceeding 25%, recommend immediate hull cleaning
        if last_y > 25:
            return current_date
        
        # If slope is negative or flat, no forecast needed
        if slope <= 0:
            return current_date + timedelta(days=365*2)  # 2 years
        
        # Determine the extension period
        if last_drydock_date is not None:
            extension_date = last_drydock_date + timedelta(days=5*365)  # 5 years
        else:
            extension_date = current_date + timedelta(days=5*365)
        
        extension_ordinal = extension_date.toordinal()
        
        # Extend the curve and find when it crosses 25%
        for day in range(current_ordinal, extension_ordinal + 1):
            if linear_fit(day) >= 25:
                return datetime.fromordinal(day)
        
        # If it doesn't cross 25% within 5 years, return the extension date
        return extension_date
    
    except Exception as e:
        st.warning(f"Error forecasting hull cleaning date: {str(e)}")
        return current_date + timedelta(days=180)

# Function to calculate vessel performance metrics
def calculate_vessel_metrics(vessel_data, last_drydock_date):
    """Calculate key performance metrics for the vessel"""
    # Initialize metrics dictionary
    metrics = {}
    
    # Current date
    current_date = datetime.now()
    
    # Parse last drydock date if it's a string
    if isinstance(last_drydock_date, str):
        last_drydock_date = pd.to_datetime(last_drydock_date)
        metrics['last_drydock_date'] = last_drydock_date.strftime('%Y-%m-%d')
    elif last_drydock_date is None:
        metrics['last_drydock_date'] = "Unknown"
    else:
        metrics['last_drydock_date'] = last_drydock_date.strftime('%Y-%m-%d')
    
    # Parse dates in vessel data if they're strings
    if isinstance(vessel_data[0].get('report_date'), str):
        for row in vessel_data:
            row['report_date'] = pd.to_datetime(row['report_date'])
    
    # Hull roughness power loss calculation
    filtered_data = [row for row in vessel_data if 'hull_roughness_power_loss' in row 
                     and row['hull_roughness_power_loss'] is not None]
    
    if filtered_data and len(filtered_data) > 1:
        # Sort by date
        filtered_data.sort(key=lambda x: x['report_date'])
        
        # Extract data for fitting
        dates = [row['report_date'] for row in filtered_data]
        hull_roughness = [row['hull_roughness_power_loss'] for row in filtered_data]
        
        try:
            # Convert dates to ordinal numbers for fitting
            x = [d.toordinal() for d in dates]
            y = hull_roughness
            
            z_power = np.polyfit(x, y, 1)
            p_power = np.poly1d(z_power)
            hull_condition_power = p_power(current_date.toordinal())
            metrics['excess_power_percentage'] = hull_condition_power
            
            # Determine hull condition
            if hull_condition_power < 15:
                metrics['hull_condition'] = "GOOD"
                metrics['hull_recommendation'] = "NA"
            elif 15 <= hull_condition_power < 25:
                metrics['hull_condition'] = "AVERAGE"
                metrics['hull_recommendation'] = "Hull cleaning and propeller polishing recommended at next economical convenient opportunity"
            else:
                metrics['hull_condition'] = "POOR"
                metrics['hull_recommendation'] = "Hull cleaning and propeller polishing recommended at next convenient opportunity"
        except Exception as e:
            metrics['excess_power_percentage'] = "N/A"
            metrics['hull_condition'] = "Unknown"
            metrics['hull_recommendation'] = f"Unable to calculate hull condition: {str(e)}"
    else:
        metrics['excess_power_percentage'] = "N/A"
        metrics['hull_condition'] = "Unknown"
        metrics['hull_recommendation'] = "Insufficient data for hull condition analysis"
    
    # Fuel saving calculation
    fuel_data = [row for row in vessel_data if 'hull_excess_fuel_oil_mtd' in row 
                 and row['hull_excess_fuel_oil_mtd'] is not None]
    
    if fuel_data and len(fuel_data) > 1:
        # Sort by date
        fuel_data.sort(key=lambda x: x['report_date'])
        
        # Extract data for fitting
        dates = [row['report_date'] for row in fuel_data]
        excess_fuel = [row['hull_excess_fuel_oil_mtd'] for row in fuel_data]
        
        try:
            # Convert dates to ordinal numbers for fitting
            x = [d.toordinal() for d in dates]
            y = excess_fuel
            
            z_fuel = np.polyfit(x, y, 1)
            p_fuel = np.poly1d(z_fuel)
            potential_fuel_saving = p_fuel(current_date.toordinal())
            metrics['potential_fuel_saving_hull'] = potential_fuel_saving
        except Exception as e:
            metrics['potential_fuel_saving_hull'] = "N/A"
    else:
        metrics['potential_fuel_saving_hull'] = "N/A"
    
    # Forecast hull cleaning date
    forecasted_date = forecast_hull_cleaning_date(vessel_data, last_drydock_date)
    metrics['forecasted_hull_cleaning_date'] = forecasted_date.strftime('%Y-%m-%d')
    
    # Placeholder values for ME SFOC metrics (these would come from your API)
    metrics['me_sfoc_status'] = "180.5"
    metrics['me_recommendation'] = "ME performance is within acceptable range"
    metrics['potential_fuel_saving_me'] = 0.8
    
    # Placeholder values for auxiliary performance (these would come from your API)
    metrics['excess_boiler_consumption'] = 2.5
    metrics['redundant_ae_hrs'] = 12.0
    
    # Placeholder values for CII metrics (these would come from your API)
    metrics['cii_rating'] = "B"
    metrics['cii_impact'] = 0.0321
    
    return metrics

# Function to display metrics in a nice format
def display_metrics(metrics):
    """Display key metrics in a visually appealing way"""
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown(f"""
        <div class="metric-container">
            <div class="metric-label">Hull Condition</div>
            <div class="metric-value" style="color: {'#4dff4d' if metrics['hull_condition'] == 'GOOD' else '#ffcc00' if metrics['hull_condition'] == 'AVERAGE' else '#ff3333' if metrics['hull_condition'] == 'POOR' else '#ffffff'};">
                {metrics['hull_condition']}
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown(f"""
        <div class="metric-container">
            <div class="metric-label">Excess Power %</div>
            <div class="metric-value" style="color: {'#4dff4d' if metrics['excess_power_percentage'] != 'N/A' and float(metrics['excess_power_percentage']) < 15 else '#ffcc00' if metrics['excess_power_percentage'] != 'N/A' and 15 <= float(metrics['excess_power_percentage']) < 25 else '#ff3333' if metrics['excess_power_percentage'] != 'N/A' and float(metrics['excess_power_percentage']) >= 25 else '#ffffff'};">
                {f"{metrics['excess_power_percentage']:.2f}%" if metrics['excess_power_percentage'] != 'N/A' else 'N/A'}
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown(f"""
        <div class="metric-container">
            <div class="metric-label">Potential Fuel Saving (Hull)</div>
            <div class="metric-value">
                {f"{metrics['potential_fuel_saving_hull']:.2f} mt/day" if metrics['potential_fuel_saving_hull'] != 'N/A' else 'N/A'}
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="metric-container">
            <div class="metric-label">Main Engine SFOC</div>
            <div class="metric-value" style="color: {'#4dff4d' if metrics['me_sfoc_status'] != 'No data' and float(metrics['me_sfoc_status'].split(' ')[0]) <= 180 else '#ffcc00' if metrics['me_sfoc_status'] != 'No data' and 180 < float(metrics['me_sfoc_status'].split(' ')[0]) <= 190 else '#ff3333' if metrics['me_sfoc_status'] != 'No data' and float(metrics['me_sfoc_status'].split(' ')[0]) > 190 else '#ffffff'};">
                {metrics['me_sfoc_status']}
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown(f"""
        <div class="metric-container">
            <div class="metric-label">Potential Fuel Saving (ME)</div>
            <div class="metric-value">
                {f"{metrics['potential_fuel_saving_me']:.2f} mt/day"}
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown(f"""
        <div class="metric-container">
            <div class="metric-label">Last Drydock Date</div>
            <div class="metric-value">
                {metrics['last_drydock_date']}
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
        <div class="metric-container">
            <div class="metric-label">CII Rating</div>
            <div class="metric-value" style="color: {'#4dff4d' if metrics['cii_rating'] == 'A' else '#b3ff66' if metrics['cii_rating'] == 'B' else '#ffcc00' if metrics['cii_rating'] == 'C' else '#ff9933' if metrics['cii_rating'] == 'D' else '#ff3333' if metrics['cii_rating'] == 'E' else '#ffffff'};">
                {metrics['cii_rating']}
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown(f"""
        <div class="metric-container">
            <div class="metric-label">CII Impact</div>
            <div class="metric-value">
                {f"{metrics['cii_impact']:.4f}" if metrics['cii_impact'] != 'N/A' else 'N/A'}
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown(f"""
        <div class="metric-container">
            <div class="metric-label">Forecasted Hull Cleaning</div>
            <div class="metric-value">
                {metrics['forecasted_hull_cleaning_date']}
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    # Display recommendations
    st.markdown("### Recommendations")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown(f"""
        <div class="metric-container">
            <div class="metric-label">Hull Recommendation</div>
            <div style="font-size: 16px;">
                {metrics['hull_recommendation']}
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="metric-container">
            <div class="metric-label">ME Recommendation</div>
            <div style="font-size: 16px;">
                {metrics['me_recommendation']}
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    # Auxiliary performance
    st.markdown("### Auxiliary Performance")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown(f"""
        <div class="metric-container">
            <div class="metric-label">Excess Boiler Consumption</div>
            <div class="metric-value">
                {f"{metrics['excess_boiler_consumption']:.2f} mt"}
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="metric-container">
            <div class="metric-label">Redundant AE Hours</div>
            <div class="metric-value">
                {f"{metrics['redundant_ae_hrs']:.2f} hrs"}
            </div>
        </div>
        """, unsafe_allow_html=True)

# Function to generate a downloadable report
def generate_report(vessel_name, metrics, vessel_data):
    """Generate a formatted report of vessel performance"""
    # Create report content with Markdown
    report_md = f"""
    # Vessel Performance Report - {vessel_name.upper()}
    
    ## Performance Summary
    
    ### Hull Performance
    - **Vessel Name:** {vessel_name.upper()}
    - **Hull Condition:** {metrics['hull_condition']}
    - **Excess Power %:** {f"{metrics['excess_power_percentage']:.2f}%" if metrics['excess_power_percentage'] != 'N/A' else 'N/A'}
    - **Potential Fuel Saving (Hull):** {f"{metrics['potential_fuel_saving_hull']:.2f} mt/day" if metrics['potential_fuel_saving_hull'] != 'N/A' else 'N/A'}
    - **Last Drydock Date:** {metrics['last_drydock_date']}
    - **Forecasted Hull Cleaning Date:** {metrics['forecasted_hull_cleaning_date']}
    - **Hull Recommendation:** {metrics['hull_recommendation']}
    
    ### Engine Performance
    - **ME SFOC Status:** {metrics['me_sfoc_status']}
    - **Potential Fuel Saving (ME):** {f"{metrics['potential_fuel_saving_me']:.2f} mt/day"}
    - **ME Recommendation:** {metrics['me_recommendation']}
    
    ### Auxiliary Performance
    - **Excess Boiler Consumption:** {f"{metrics['excess_boiler_consumption']:.2f} mt"}
    - **Redundant AE Hours:** {f"{metrics['redundant_ae_hrs']:.2f} hrs"}
    
    ### Emissions
    - **CII Rating:** {metrics['cii_rating']}
    - **CII Impact:** {f"{metrics['cii_impact']:.4f}" if metrics['cii_impact'] != 'N/A' else 'N/A'}
    
    ## Analysis Date
    Report generated on {datetime.now().strftime('%Y-%m-%d')}
    """
    
    return report_md

# Main application interface
st.title("🚢 Marine Performance Analysis System")

# Sidebar with vessel selection
st.sidebar.image("https://img.freepik.com/premium-vector/ship-logo-design-vector-cruise-boat-ship-logo-vector-design_644408-263.jpg", width=200)
st.sidebar.title("Configuration")

# First, try to fetch vessel list from API
vessel_list = fetch_data("getVessels")

# If API call fails, offer to use sample data
if vessel_list is None:
    use_sample_data = st.sidebar.checkbox("Use sample data instead?", value=True)
    if use_sample_data:
        vessel_list = load_sample_data("getVessels")
    else:
        st.error("Unable to fetch vessel list from database. Please check connection settings.")
        st.stop()

# Vessel selection
selected_vessel = st.sidebar.selectbox("Select Vessel", vessel_list)

if selected_vessel:
    # Get all data for selected vessel
    st.sidebar.markdown("---")
    st.sidebar.markdown("### Data Filters")
    
    # Date range filter
    start_date = st.sidebar.date_input("Start Date", 
                                      value=datetime.now() - timedelta(days=180), 
                                      max_value=datetime.now())
    
    end_date = st.sidebar.date_input("End Date", 
                                    value=datetime.now(), 
                                    max_value=datetime.now(),
                                    min_value=start_date)
    
    # Wind force filter
    max_wind_force = st.sidebar.slider("Max Wind Force", 0, 12, 4)
    
    # Data fetching and processing
    with st.spinner(f"Loading data for {selected_vessel}..."):
        # Attempt to fetch data from API
        vessel_data = fetch_data("getHullPerformance", {
            "vesselName": selected_vessel,
            "startDate": start_date.isoformat(),
            "endDate": end_date.isoformat()
        })
        
        last_drydock_date = fetch_data("getLastDrydockDate", {"vesselName": selected_vessel})
        
        # If API calls fail, use sample data if enabled
        use_sample_data = st.sidebar.checkbox("Use sample data", value=vessel_data is None)
        
        if use_sample_data or vessel_data is None:
            vessel_data = load_sample_data("getHullPerformance")
            last_drydock_date = load_sample_data("getLastDrydockDate")
            st.warning("Using sample data for demonstration.")
    
    if not vessel_data:
        st.error(f"No data available for vessel: {selected_vessel}")
        st.stop()
    
    # Apply additional filters (date filter should already be applied by API, this is a safeguard)
    filtered_data = []
    for row in vessel_data:
        row_date = pd.to_datetime(row['report_date'])
        if (row_date >= pd.Timestamp(start_date) and 
            row_date <= pd.Timestamp(end_date) and 
            row.get('windforce', 0) <= max_wind_force):
            filtered_data.append(row)
    
    # Check if we have enough data after filtering
    if not filtered_data:
        st.warning("No data available after applying filters. Please adjust filter criteria.")
        st.stop()
    
    # Placeholder for coefficients (in a real app, these would come from your database)
    coefficients = {
        'speed_coeff': 0.5,
        'disp_coeff': 0.3,
        'speed2_coeff': 0.05,
        'disp2_coeff': 0.02,
        'intercept': 10
    }
    
    # Calculate vessel metrics
    metrics = calculate_vessel_metrics(filtered_data, last_drydock_date)
    
    # Main content area with tabs
    tab1, tab2, tab3, tab4 = st.tabs([
        "📊 Dashboard", 
        "🛥️ Speed-Consumption Analysis", 
        "🧠 Hull Performance", 
        "📝 Reports"
    ])
    
    with tab1:
        st.header(f"Performance Dashboard - {selected_vessel.upper()}")
        
        # Display key metrics
        display_metrics(metrics)
        
        # Hull roughness trend chart
        st.subheader("Hull Roughness Trend")
        hull_chart = create_hull_roughness_chart(filtered_data)
        st.plotly_chart(hull_chart, use_container_width=True)
        
        # Basic vessel stats
        st.subheader("Data Statistics")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric("Data Points", len(filtered_data))
            dates = [pd.to_datetime(row['report_date']) for row in filtered_data]
            st.metric("Date Range", f"{min(dates).strftime('%Y-%m-%d')} to {max(dates).strftime('%Y-%m-%d')}")
        
        with col2:
            speeds = [row.get('speed', 0) for row in filtered_data if 'speed' in row]
            consumptions = [row.get('normalised_consumption', 0) for row in filtered_data if 'normalised_consumption' in row]
            
            avg_speed = sum(speeds) / len(speeds) if speeds else 0
            avg_consumption = sum(consumptions) / len(consumptions) if consumptions else 0
            
            st.metric("Avg Speed", f"{avg_speed:.2f} knots")
            st.metric("Avg Fuel Consumption", f"{avg_consumption:.2f} mt/day")
    
    with tab2:
        st.header(f"Speed-Consumption Analysis - {selected_vessel.upper()}")
        
        # Create charts for both conditions
        ballast_chart = create_interactive_chart(filtered_data, "ballast", coefficients)
        laden_chart = create_interactive_chart(filtered_data, "laden", coefficients)
        
        # Display charts
        st.subheader("Ballast Condition")
        st.plotly_chart(ballast_chart, use_container_width=True)
        
        st.subheader("Laden Condition")
        st.plotly_chart(laden_chart, use_container_width=True)
        
        # Show raw data if desired
        if st.checkbox("Show raw data"):
            # Convert to DataFrame for easier display
            df = pd.DataFrame(filtered_data)
            if 'report_date' in df.columns:
                df['report_date'] = pd.to_datetime(df['report_date'])
            
            # Select relevant columns
            display_cols = ['report_date', 'loading_condition', 'speed', 'normalised_consumption', 'windforce']
            display_cols = [col for col in display_cols if col in df.columns]
            
            st.dataframe(df[display_cols])
    
    with tab3:
        st.header(f"Hull Performance Analysis - {selected_vessel.upper()}")
        
        # Hull roughness over time
        st.subheader("Hull Roughness Power Loss Trend")
        hull_chart = create_hull_roughness_chart(filtered_data)
        st.plotly_chart(hull_chart, use_container_width=True)
        
        # Hull excess fuel consumption over time
        fuel_data = [row for row in filtered_data if 'hull_excess_fuel_oil_mtd' in row and row['hull_excess_fuel_oil_mtd'] is not None]
        
        if fuel_data:
            st.subheader("Hull Excess Fuel Consumption Trend")
            
            # Parse dates if they're strings
            for row in fuel_data:
                if isinstance(row['report_date'], str):
                    row['report_date'] = pd.to_datetime(row['report_date'])
            
            # Sort by date
            fuel_data.sort(key=lambda x: x['report_date'])
            
            fig = go.Figure()
            
            dates = [row['report_date'] for row in fuel_data]
            excess_fuel = [row['hull_excess_fuel_oil_mtd'] for row in fuel_data]
            date_colors = [pd.to_datetime(d).toordinal() for d in dates]
            
            fig.add_trace(go.Scatter(
                x=dates,
                y=excess_fuel,
                mode='markers+lines',
                marker=dict(
                    size=8,
                    color=date_colors,
                    colorscale='Plasma',
                    showscale=True,
                    colorbar=dict(title="Date")
                ),
                name='Excess Fuel'
            ))
            
            # Add trend line
            if len(fuel_data) > 2:
                try:
                    x_numeric = [pd.to_datetime(d).toordinal() for d in dates]
                    z_fuel = np.polyfit(x_numeric, excess_fuel, 1)
                    p_fuel = np.poly1d(z_fuel)
                    
                    x_trend = [min(dates), max(dates)]
                    y_trend = p_fuel([pd.to_datetime(d).toordinal() for d in x_trend])
                    
                    fig.add_trace(go.Scatter(
                        x=x_trend,
                        y=y_trend,
                        mode='lines',
                        line=dict(color='#ff006e', width=3),
                        name='Trend Line'
                    ))
                except Exception as e:
                    st.warning(f"Could not create trend line: {str(e)}")
            
            fig.update_layout(
                title="Hull Excess Fuel Consumption Trend",
                xaxis_title="Date",
                yaxis_title="Excess Fuel (mt/day)",
                template="plotly_dark",
                height=500
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        # Hull cleaning forecast
        st.subheader("Hull Cleaning Forecast")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown(f"""
            <div class="metric-container">
                <div class="metric-label">Current Hull Condition</div>
                <div class="metric-value" style="color: {'#4dff4d' if metrics['hull_condition'] == 'GOOD' else '#ffcc00' if metrics['hull_condition'] == 'AVERAGE' else '#ff3333' if metrics['hull_condition'] == 'POOR' else '#ffffff'};">
                    {metrics['hull_condition']}
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown(f"""
            <div class="metric-container">
                <div class="metric-label">Last Drydock Date</div>
                <div class="metric-value">
                    {metrics['last_drydock_date']}
                </div>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown(f"""
            <div class="metric-container">
                <div class="metric-label">Forecasted Hull Cleaning Date</div>
                <div class="metric-value">
                    {metrics['forecasted_hull_cleaning_date']}
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            # Calculate days until hull cleaning
            today = datetime.now()
            forecasted_date = datetime.strptime(metrics['forecasted_hull_cleaning_date'], '%Y-%m-%d')
            days_until = (forecasted_date - today).days
            
            st.markdown(f"""
            <div class="metric-container">
                <div class="metric-label">Days Until Hull Cleaning</div>
                <div class="metric-value" style="color: {'#4dff4d' if days_until > 180 else '#ffcc00' if 90 < days_until <= 180 else '#ff3333' if days_until <= 90 else '#ffffff'};">
                    {days_until} days
                </div>
            </div>
            """, unsafe_allow_html=True)
    
    with tab4:
        st.header(f"Performance Reports - {selected_vessel.upper()}")
        
        # Generate report
        report_md = generate_report(selected_vessel, metrics, filtered_data)
        
        # Display report preview
        st.subheader("Report Preview")
        st.markdown(report_md)
        
        # Download options
        st.subheader("Download Options")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Download markdown report
            st.download_button(
                label="Download Report (Markdown)",
                data=report_md,
                file_name=f"{selected_vessel}_performance_report.md",
                mime="text/markdown",
            )
        
        with col2:
            # Generate PDF button (this would typically require additional libraries)
            if st.button("Generate PDF Report"):
                st.info("PDF generation requires additional setup. In a production environment, you would integrate with a PDF library.")
                st.code("""
# Example PDF generation code (not functional in Streamlit)
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter

def create_pdf(vessel_name, metrics):
    pdf_file = f"{vessel_name}_report.pdf"
    c = canvas.Canvas(pdf_file, pagesize=letter)
    c.drawString(72, 750, f"Vessel Performance Report - {vessel_name}")
    # ... Add more content ...
    c.save()
    return pdf_file
                """)
        
        # Additional export options
        st.subheader("Export Raw Data")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Export CSV
            df = pd.DataFrame(filtered_data)
            csv = df.to_csv(index=False)
            st.download_button(
                label="Export Data as CSV",
                data=csv,
                file_name=f"{selected_vessel}_performance_data.csv",
                mime="text/csv",
            )
        
        with col2:
            # Export Excel
            buffer = io.BytesIO()
            df.to_excel(buffer, index=False, engine='openpyxl')
            buffer.seek(0)
            
            st.download_button(
                label="Export Data as Excel",
                data=buffer,
                file_name=f"{selected_vessel}_performance_data.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            )

# Add info about the Lambda connection in the sidebar
st.sidebar.markdown("---")
st.sidebar.markdown("### API Connection")
st.sidebar.markdown(f"Using Lambda URL: `{LAMBDA_URL}`")

if st.sidebar.checkbox("Show Sample Lambda Request"):
    st.sidebar.code("""
# Sample request to Lambda URL
import requests
import json

def get_vessel_data(vessel_name):
    response = requests.post(
        "https://your-lambda-url.amazonaws.com/",
        json={
            "operation": "getHullPerformance",
            "vesselName": vessel_name,
            "startDate": "2023-01-01",
            "endDate": "2023-12-31"
        },
        headers={"Content-Type": "application/json"}
    )
    
    if response.status_code == 200:
        return response.json()
    else:
        print(f"Error: {response.status_code}")
        print(response.text)
        return None
    """, language="python")

# Footer with app info
st.sidebar.markdown("---")
st.sidebar.info("Marine Performance Analysis System v1.0")
st.sidebar.markdown("Developed using Streamlit and AWS Lambda")
