import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime, timedelta
import boto3
import json
import io
import os
import base64
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

# Lambda connection helper function
def invoke_lambda(function_name, payload):
    """Invoke AWS Lambda function and return the response"""
    lambda_client = boto3.client(
        'lambda',
        region_name=os.environ.get('AWS_REGION', 'us-east-1'),
        aws_access_key_id=os.environ.get('AWS_ACCESS_KEY_ID'),
        aws_secret_access_key=os.environ.get('AWS_SECRET_ACCESS_KEY')
    )
    
    response = lambda_client.invoke(
        FunctionName=function_name,
        InvocationType='RequestResponse',
        Payload=json.dumps(payload)
    )
    
    return json.loads(response['Payload'].read().decode('utf-8'))

# Function to fetch data from RDS via Lambda
@st.cache_data(ttl=3600)
def fetch_data(query_type, params=None):
    """Fetch data from RDS database through Lambda function"""
    try:
        payload = {
            "query_type": query_type,
            "params": params or {}
        }
        
        response = invoke_lambda("marine_performance_data_fetcher", payload)
        
        if "error" in response:
            st.error(f"Error fetching data: {response['error']}")
            return None
        
        if query_type in ["vessel_list", "vessel_details"]:
            return response["data"]
        else:
            return pd.DataFrame(response["data"])
    except Exception as e:
        st.error(f"Error connecting to database: {str(e)}")
        # For demo/development purposes, fallback to local files
        if st.checkbox("Use local sample data instead?"):
            return load_sample_data(query_type)
        return None

# Sample data loader for development/demo
def load_sample_data(data_type):
    """Load sample data from local files"""
    try:
        if data_type == "vessel_list":
            # Return a list of vessel names
            return ["VESSEL_A", "VESSEL_B", "VESSEL_C"]
        
        elif data_type == "performance_data":
            return pd.read_csv("sample_data/Hull Performance data.csv")
        
        elif data_type == "coeff_data":
            return pd.read_csv("sample_data/vessel performance coeff.csv")
        
        elif data_type == "sea_trial_data":
            return pd.read_excel("sample_data/Sea_Trial.xlsx", engine='openpyxl')
        
        elif data_type == "dd_dates_data":
            return pd.read_excel("sample_data/DD dates.xlsx", engine='openpyxl')
        
        elif data_type == "consumption_log_data":
            return pd.read_csv("sample_data/consumption_log.csv")
        
        elif data_type == "vessel_particulars_data":
            return pd.read_excel("sample_data/Vessel_Particulars.xlsx", engine='openpyxl')
        
        elif data_type == "perf_app_data":
            return pd.read_csv("sample_data/performance app.csv")
            
        else:
            st.warning(f"Unknown data type requested: {data_type}")
            return None
    except Exception as e:
        st.error(f"Error loading sample data: {str(e)}")
        return None

# Create neon color gradient for charts
def create_neon_color_gradient(dates):
    norm = plt.Normalize(dates.min().toordinal(), dates.max().toordinal())
    cmap = plt.get_cmap('plasma')
    return [cmap(norm(date.toordinal())) for date in dates]

# Polynomial fitting function
def polynomial_fit(x, y, degree):
    coeffs = np.polyfit(x, y, degree)
    return np.poly1d(coeffs)

# Exponential fitting function
def exponential_fit(x, y):
    log_y = np.log(y)
    coeffs = np.polyfit(x, log_y, 1)
    return lambda x: np.exp(coeffs[1] + coeffs[0] * x)

# Calculate consumption function
def calculate_consumption(vessel_data, speeds, displacements):
    results = []
    for speed in speeds:
        for displacement in displacements:
            consumption = (vessel_data['CONSP_SPEED1'] * speed +
                          vessel_data['CONSP_DISP1'] * displacement +
                          vessel_data['CONSP_SPEED2'] * speed**2 +
                          vessel_data['CONSP_DISP2'] * displacement**2 +
                          vessel_data['CONSP_INTERCEPT'])
            results.append({
                'Speed': speed,
                'Displacement': displacement,
                'Consumption (mt/day)': consumption
            })
    return results

# Function to create interactive Plotly charts for speed vs consumption
def create_interactive_chart(vessel_data, condition, event_date=None, event_name="Event"):
    # Filter data for the condition
    condition_data = vessel_data[vessel_data['LOADING_CONDITION'].str.lower() == condition.lower()]
    
    if condition_data.empty:
        return go.Figure().update_layout(title=f"No data available for {condition} condition")
    
    # Create figure
    fig = go.Figure()
    
    # Add scatter plot for actual data
    fig.add_trace(go.Scatter(
        x=condition_data['SPEED'],
        y=condition_data['NORMALISED_CONSUMPTION'],
        mode='markers',
        marker=dict(
            size=10,
            color=condition_data['REPORT_DATE'].apply(lambda x: x.toordinal()),
            colorscale='Plasma',
            showscale=True,
            colorbar=dict(title="Date")
        ),
        name='Actual Data'
    ))
    
    # Fit curve to data
    x_data = condition_data['SPEED'].values
    y_data = condition_data['NORMALISED_CONSUMPTION'].values
    
    if len(x_data) > 2:
        try:
            # Exponential fit
            exp_func = exponential_fit(x_data, y_data)
            x_smooth = np.linspace(min(x_data), max(x_data), 100)
            y_smooth = [exp_func(x) for x in x_smooth]
            
            fig.add_trace(go.Scatter(
                x=x_smooth,
                y=y_smooth,
                mode='lines',
                line=dict(color='#48CAE4', width=3),
                name='Exponential Fit'
            ))
        except Exception as e:
            st.warning(f"Could not create exponential fit: {str(e)}")
    
    # Add sea trial baseline data if available
    sea_trial_data = fetch_data("sea_trial_data", {"vessel_name": selected_vessel})
    coeff_data = fetch_data("coeff_data", {"vessel_name": selected_vessel})
    
    if sea_trial_data is not None and coeff_data is not None and not coeff_data.empty:
        # Get baseline consumption values
        displacements = sea_trial_data['DISPLACEMENT'].sort_values().values
        
        if len(displacements) >= 2:
            if condition.lower() == 'ballast':
                displacement = displacements[0] / 10000
            else:  # laden
                displacement = displacements[-1] / 10000
                
            speeds = np.arange(8, 16, 1)
            calculated_results = calculate_consumption(coeff_data.iloc[0], speeds, [displacement])
            
            baseline_speeds = [result['Speed'] for result in calculated_results]
            baseline_consumptions = [result['Consumption (mt/day)'] for result in calculated_results]
            
            fig.add_trace(go.Scatter(
                x=baseline_speeds,
                y=baseline_consumptions,
                mode='markers',
                marker=dict(size=10, color='#00FFFF', symbol='diamond'),
                name='Baseline'
            ))
            
            # Polynomial fit for baseline
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
                    st.warning(f"Could not create baseline fit: {str(e)}")
    
    # Update layout
    fig.update_layout(
        title=f"{condition.capitalize()} Condition - {selected_vessel.upper()}",
        xaxis_title="Speed (knots)",
        yaxis_title="ME Consumption (mT/day)",
        template="plotly_dark",
        hovermode="closest",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        height=600
    )
    
    return fig

# Function to create hull roughness chart
def create_hull_roughness_chart(vessel_data):
    # Drop missing values and filter for power loss data
    filtered_data = vessel_data.dropna(subset=['HULL_ROUGHNESS_POWER_LOSS'])
    
    if filtered_data.empty:
        return go.Figure().update_layout(title="No hull roughness data available")
    
    # Create figure
    fig = go.Figure()
    
    # Add scatter plot for power loss data
    fig.add_trace(go.Scatter(
        x=filtered_data['REPORT_DATE'],
        y=filtered_data['HULL_ROUGHNESS_POWER_LOSS'],
        mode='markers',
        marker=dict(
            size=10,
            color=filtered_data['REPORT_DATE'].apply(lambda x: x.toordinal()),
            colorscale='Plasma',
            showscale=True,
            colorbar=dict(title="Date")
        ),
        name='Daily Data'
    ))
    
    # Add trend line
    if len(filtered_data) > 2:
        x_power = filtered_data['REPORT_DATE'].map(datetime.toordinal)
        y_power = filtered_data['HULL_ROUGHNESS_POWER_LOSS']
        
        try:
            z_power = np.polyfit(x_power, y_power, 1)
            p_power = np.poly1d(z_power)
            
            x_dates = [filtered_data['REPORT_DATE'].min(), filtered_data['REPORT_DATE'].max()]
            y_trend = p_power([x.toordinal() for x in x_dates])
            
            fig.add_trace(go.Scatter(
                x=x_dates,
                y=y_trend,
                mode='lines',
                line=dict(color='#ff006e', width=3),
                name='Trend Line'
            ))
            
            # Add lines at 15% and 25% thresholds
            fig.add_shape(
                type="line",
                x0=filtered_data['REPORT_DATE'].min(),
                y0=15,
                x1=filtered_data['REPORT_DATE'].max(),
                y1=15,
                line=dict(color="yellow", width=2, dash="dash"),
            )
            
            fig.add_shape(
                type="line",
                x0=filtered_data['REPORT_DATE'].min(),
                y0=25,
                x1=filtered_data['REPORT_DATE'].max(),
                y1=25,
                line=dict(color="red", width=2, dash="dash"),
            )
            
            # Add annotations for the thresholds
            fig.add_annotation(
                x=filtered_data['REPORT_DATE'].max(),
                y=15,
                text="15% - Average condition",
                showarrow=False,
                yshift=10,
                xshift=-100,
                font=dict(color="yellow")
            )
            
            fig.add_annotation(
                x=filtered_data['REPORT_DATE'].max(),
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
        title=f"Hull Roughness Power Loss - {selected_vessel.upper()}",
        xaxis_title="Date",
        yaxis_title="Hull Roughness Power Loss (%)",
        template="plotly_dark",
        hovermode="closest",
        height=600
    )
    
    return fig

# Function to forecast hull cleaning date
def forecast_hull_cleaning_date(vessel_data, last_drydock_date, current_date):
    filtered_data = vessel_data.dropna(subset=['HULL_ROUGHNESS_POWER_LOSS'])
    
    if filtered_data.empty:
        return current_date + timedelta(days=180)  # Default to 6 months in the future
    
    # Convert dates to ordinal numbers for fitting
    filtered_data['DATE_ORDINAL'] = filtered_data['REPORT_DATE'].apply(lambda x: x.toordinal())
    
    # Fit a line to the hull roughness power loss data
    x = filtered_data['DATE_ORDINAL'].values
    y = filtered_data['HULL_ROUGHNESS_POWER_LOSS'].values
    
    if len(x) < 2:
        return current_date + timedelta(days=180)
    
    try:
        coefficients = np.polyfit(x, y, 1)
        linear_fit = np.poly1d(coefficients)
        slope = coefficients[0]
        last_y = linear_fit(x[-1])
        
        # If already exceeding 25%, recommend immediate hull cleaning
        if last_y > 25:
            return current_date
        
        # Determine the extension period
        if pd.notnull(last_drydock_date):
            extension_date = last_drydock_date + timedelta(days=5*365)  # 5 years
        else:
            extension_date = current_date + timedelta(days=5*365)
        
        extension_date_ordinal = extension_date.toordinal()
        current_date_ordinal = current_date.toordinal()
        
        # Extend the curve and find when it crosses 25%
        for day in range(current_date_ordinal, extension_date_ordinal + 1):
            if linear_fit(day) >= 25:
                return datetime.fromordinal(day)
        
        # If it doesn't cross 25% within 5 years, return the extension date
        return extension_date
    
    except Exception as e:
        st.warning(f"Error forecasting hull cleaning date: {str(e)}")
        return current_date + timedelta(days=180)

# Function to calculate vessel performance metrics
def calculate_vessel_metrics(vessel_data, coeff_data, sea_trial_data, dd_dates_data, vessel_particulars_data, consumption_log_data, perf_app_data):
    # Initialize results dictionary
    metrics = {}
    
    # Current date
    current_date = datetime.now()
    
    # Get last drydock date
    last_drydock_row = dd_dates_data[(dd_dates_data['VESSEL NAME'].str.lower() == selected_vessel.lower()) & 
                                    (dd_dates_data['EVENT NAME'].str.lower() == 'dd')]
    
    if not last_drydock_row.empty:
        last_drydock_date = pd.to_datetime(last_drydock_row['EVENT DATE']).max()
        metrics['last_drydock_date'] = last_drydock_date.strftime('%Y-%m-%d')
    else:
        last_drydock_date = None
        metrics['last_drydock_date'] = "Unknown"
    
    # Hull roughness power loss calculation
    filtered_data = vessel_data.dropna(subset=['HULL_ROUGHNESS_POWER_LOSS'])
    
    if not filtered_data.empty and len(filtered_data) > 1:
        x_power = filtered_data['REPORT_DATE'].map(datetime.toordinal)
        y_power = filtered_data['HULL_ROUGHNESS_POWER_LOSS']
        
        try:
            z_power = np.polyfit(x_power, y_power, 1)
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
            metrics['hull_recommendation'] = "Unable to calculate hull condition"
    else:
        metrics['excess_power_percentage'] = "N/A"
        metrics['hull_condition'] = "Unknown"
        metrics['hull_recommendation'] = "Insufficient data"
    
    # Fuel saving calculation
    fuel_data = vessel_data.dropna(subset=['HULL_EXCESS_FUEL_OIL_MTD'])
    
    if not fuel_data.empty and len(fuel_data) > 1:
        x_fuel = fuel_data['REPORT_DATE'].map(datetime.toordinal)
        y_fuel = fuel_data['HULL_EXCESS_FUEL_OIL_MTD']
        
        try:
            z_fuel = np.polyfit(x_fuel, y_fuel, 1)
            p_fuel = np.poly1d(z_fuel)
            potential_fuel_saving = p_fuel(current_date.toordinal())
            metrics['potential_fuel_saving_hull'] = potential_fuel_saving
        except Exception as e:
            metrics['potential_fuel_saving_hull'] = "N/A"
    else:
        metrics['potential_fuel_saving_hull'] = "N/A"
    
    # Forecast hull cleaning date
    forecasted_date = forecast_hull_cleaning_date(vessel_data, last_drydock_date, current_date)
    metrics['forecasted_hull_cleaning_date'] = forecasted_date.strftime('%Y-%m-%d')
    
    # ME SFOC metrics from performance app data
    if perf_app_data is not None:
        # Convert to DataFrame if it's a dictionary
        if isinstance(perf_app_data, dict):
            metrics.update(perf_app_data)
        else:
            # Filter for this vessel
            vessel_perf_data = perf_app_data[perf_app_data['VESSEL_NAME'].str.lower() == selected_vessel.lower()].copy()
            
            if not vessel_perf_data.empty:
                # Convert date and calculate 4-month window
                vessel_perf_data['REPORTDATE'] = pd.to_datetime(vessel_perf_data['REPORTDATE'])
                four_months_ago = vessel_perf_data['REPORTDATE'].max() - pd.Timedelta(days=120)
                recent_data = vessel_perf_data[vessel_perf_data['REPORTDATE'] >= four_months_ago]
                
                # Calculate ME SFOC metrics
                avg_me_sfoc = recent_data['ME_SFOC'].astype(float).mean()
                
                if pd.isna(avg_me_sfoc):
                    metrics['me_sfoc_status'] = "No data"
                    metrics['me_recommendation'] = "Insufficient data for ME SFOC analysis"
                    metrics['potential_fuel_saving_me'] = 0
                elif 160 <= avg_me_sfoc <= 240:
                    metrics['me_sfoc_status'] = f"{avg_me_sfoc:.2f}"
                    if avg_me_sfoc > 190:
                        metrics['me_recommendation'] = "Analyse ME performance and take action accordingly"
                        avg_normalised_consumption = recent_data['NORMALISED_ME_CONSUMPTION'].astype(float).mean()
                        metrics['potential_fuel_saving_me'] = (avg_me_sfoc - 180) * avg_normalised_consumption / 180
                    else:
                        metrics['me_recommendation'] = "ME performance is within acceptable range"
                        metrics['potential_fuel_saving_me'] = 0
                else:
                    metrics['me_sfoc_status'] = f"{avg_me_sfoc:.2f} (Anomalous)"
                    metrics['me_recommendation'] = "Investigate abnormal ME SFOC values"
                    metrics['potential_fuel_saving_me'] = 0
                
                # Auxiliary performance metrics
                metrics['excess_boiler_consumption'] = recent_data['BOILER_EXCESS_CONSUMPTION'].astype(float).sum()
                metrics['redundant_ae_hrs'] = recent_data['AE_REDUNDANT_HRS'].astype(float).sum()
            else:
                metrics['me_sfoc_status'] = "No data"
                metrics['me_recommendation'] = "No data available"
                metrics['potential_fuel_saving_me'] = 0
                metrics['excess_boiler_consumption'] = 0
                metrics['redundant_ae_hrs'] = 0
    else:
        metrics['me_sfoc_status'] = "No data"
        metrics['me_recommendation'] = "Performance app data not available"
        metrics['potential_fuel_saving_me'] = 0
        metrics['excess_boiler_consumption'] = 0
        metrics['redundant_ae_hrs'] = 0
    
    # CII metrics from consumption log
    if consumption_log_data is not None and vessel_particulars_data is not None:
        # Get vessel IMO
        vessel_info = vessel_particulars_data[vessel_particulars_data['Vessel_Name'].str.lower() == selected_vessel.lower()]
        
        if not vessel_info.empty:
            vessel_imo = vessel_info['IMO'].iloc[0]
            vessel_deadweight = vessel_info['Deadweight'].iloc[0]
            
            # Get CII rating
            cii_data = consumption_log_data[consumption_log_data['VESSEL_IMO'] == vessel_imo]
            
            if not cii_data.empty:
                metrics['cii_rating'] = cii_data['CII_RATING'].iloc[0]
                
                # Calculate CII impact if we have fuel saving data
                if metrics['potential_fuel_saving_hull'] != "N/A":
                    # Average distance travelled
                    avg_distance = vessel_data['DISTANCE_TRAVELLED_ACTUAL'].mean()
                    
                    if not pd.isna(avg_distance) and avg_distance > 0:
                        cii_impact = ((float(metrics['potential_fuel_saving_hull']) * (10**6)) * 3.114) / (avg_distance * vessel_deadweight)
                        metrics['cii_impact'] = cii_impact
                    else:
                        metrics['cii_impact'] = "N/A"
                else:
                    metrics['cii_impact'] = "N/A"
            else:
                metrics['cii_rating'] = "No data"
                metrics['cii_impact'] = "N/A"
        else:
            metrics['cii_rating'] = "Vessel not found"
            metrics['cii_impact'] = "N/A"
    else:
        metrics['cii_rating'] = "Data not available"
        metrics['cii_impact'] = "N/A"
    
    return metrics

# Function to display metrics in a nice format
def display_metrics(metrics):
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
st.sidebar.image("https://placeholder.pics/svg/150x80/3498DB/FFFFFF/Marine%20Analytics", width=200)
st.sidebar.title("Configuration")

# Get list of vessels
vessel_list = fetch_data("vessel_list")

if vessel_list is None:
    st.error("Unable to fetch vessel list from database. Check connection settings.")
    st.stop()

# Vessel selection
selected_vessel = st.sidebar.selectbox("Select Vessel", vessel_list)

if selected_vessel:
    # Get all data for selected vessel
    st.sidebar.markdown("---")
    st.sidebar.markdown("### Data Filters")
    
    # Date range filter
    start_date = st.sidebar.date_input("Start Date", 
                                      value=datetime.now() - timedelta(days=365), 
                                      max_value=datetime.now())
    
    end_date = st.sidebar.date_input("End Date", 
                                    value=datetime.now(), 
                                    max_value=datetime.now(),
                                    min_value=start_date)
    
    # Wind force filter
    max_wind_force = st.sidebar.slider("Max Wind Force", 0, 12, 4)
    
    # Fetch data with parameters
    with st.spinner(f"Loading data for {selected_vessel}..."):
        performance_data = fetch_data("performance_data", {"vessel_name": selected_vessel})
        coeff_data = fetch_data("coeff_data", {"vessel_name": selected_vessel})
        sea_trial_data = fetch_data("sea_trial_data", {"vessel_name": selected_vessel})
        dd_dates_data = fetch_data("dd_dates_data", {"vessel_name": selected_vessel})
        consumption_log_data = fetch_data("consumption_log_data", {"vessel_name": selected_vessel})
        vessel_particulars_data = fetch_data("vessel_particulars_data", {"vessel_name": selected_vessel})
        perf_app_data = fetch_data("perf_app_data", {"vessel_name": selected_vessel})
    
    if performance_data is None or performance_data.empty:
        st.error(f"No data available for vessel: {selected_vessel}")
        st.stop()
    
    # Pre-process data
    performance_data.columns = performance_data.columns.str.strip().str.upper()
    performance_data['REPORT_DATE'] = pd.to_datetime(performance_data['REPORT_DATE'])
    
    # Apply filters
    filtered_data = performance_data[
        (performance_data['REPORT_DATE'] >= pd.Timestamp(start_date)) &
        (performance_data['REPORT_DATE'] <= pd.Timestamp(end_date)) &
        (performance_data['WINDFORCE'] <= max_wind_force)
    ]
    
    if dd_dates_data is not None:
        dd_dates_data.columns = dd_dates_data.columns.str.strip().str.upper()
        dd_dates_data['EVENT DATE'] = pd.to_datetime(dd_dates_data['EVENT DATE'])
    
    # Check if we have enough data after filtering
    if filtered_data.empty:
        st.warning("No data available after applying filters. Please adjust filter criteria.")
        st.stop()
    
    # Calculate vessel metrics
    metrics = calculate_vessel_metrics(
        filtered_data, coeff_data, sea_trial_data, dd_dates_data, 
        vessel_particulars_data, consumption_log_data, perf_app_data
    )
    
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
            st.metric("Date Range", f"{filtered_data['REPORT_DATE'].min().strftime('%Y-%m-%d')} to {filtered_data['REPORT_DATE'].max().strftime('%Y-%m-%d')}")
        
        with col2:
            st.metric("Avg Speed", f"{filtered_data['SPEED'].mean():.2f} knots")
            st.metric("Avg Fuel Consumption", f"{filtered_data['NORMALISED_CONSUMPTION'].mean():.2f} mt/day")
    
    with tab2:
        st.header(f"Speed-Consumption Analysis - {selected_vessel.upper()}")
        
        # Create charts for both conditions
        ballast_chart = create_interactive_chart(filtered_data, "ballast")
        laden_chart = create_interactive_chart(filtered_data, "laden")
        
        # Display charts
        st.subheader("Ballast Condition")
        st.plotly_chart(ballast_chart, use_container_width=True)
        
        st.subheader("Laden Condition")
        st.plotly_chart(laden_chart, use_container_width=True)
        
        # Show raw data if desired
        if st.checkbox("Show raw data"):
            st.write(filtered_data[['REPORT_DATE', 'LOADING_CONDITION', 'SPEED', 'NORMALISED_CONSUMPTION', 'WINDFORCE']])
    
    with tab3:
        st.header(f"Hull Performance Analysis - {selected_vessel.upper()}")
        
        # Hull roughness over time
        st.subheader("Hull Roughness Power Loss Trend")
        hull_chart = create_hull_roughness_chart(filtered_data)
        st.plotly_chart(hull_chart, use_container_width=True)
        
        # Hull excess fuel consumption over time
        if 'HULL_EXCESS_FUEL_OIL_MTD' in filtered_data.columns:
            fuel_data = filtered_data.dropna(subset=['HULL_EXCESS_FUEL_OIL_MTD'])
            
            if not fuel_data.empty:
                st.subheader("Hull Excess Fuel Consumption Trend")
                
                fig = go.Figure()
                
                fig.add_trace(go.Scatter(
                    x=fuel_data['REPORT_DATE'],
                    y=fuel_data['HULL_EXCESS_FUEL_OIL_MTD'],
                    mode='markers+lines',
                    marker=dict(
                        size=8,
                        color=fuel_data['REPORT_DATE'].apply(lambda x: x.toordinal()),
                        colorscale='Plasma',
                        showscale=True,
                        colorbar=dict(title="Date")
                    ),
                    name='Excess Fuel'
                ))
                
                fig.update_layout(
                    title=f"Hull Excess Fuel Consumption - {selected_vessel.upper()}",
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
            # Generate PDF button
            st.warning("PDF generation would require additional libraries in a production environment")
            # Placeholder for PDF generation
            st.button("Generate PDF Report", disabled=True)
        
        # Additional export options
        st.subheader("Export Raw Data")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Export CSV
            csv = filtered_data.to_csv(index=False)
            st.download_button(
                label="Export Data as CSV",
                data=csv,
                file_name=f"{selected_vessel}_performance_data.csv",
                mime="text/csv",
            )
        
        with col2:
            # Export Excel
            buffer = io.BytesIO()
            filtered_data.to_excel(buffer, index=False, engine='openpyxl')
            buffer.seek(0)
            
            st.download_button(
                label="Export Data as Excel",
                data=buffer,
                file_name=f"{selected_vessel}_performance_data.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            )

# Add a simple AWS Lambda function template for data fetching
st.sidebar.markdown("---")
st.sidebar.markdown("### AWS Lambda Connection")

if st.sidebar.checkbox("Show Lambda Function Template"):
    st.sidebar.code("""
# Lambda function template for data fetching
import json
import boto3
import pandas as pd
import pymysql
import os

# RDS configuration
RDS_HOST = os.environ['RDS_HOST']
RDS_PORT = int(os.environ['RDS_PORT'])
RDS_USER = os.environ['RDS_USER']
RDS_PASSWORD = os.environ['RDS_PASSWORD']
RDS_DB = os.environ['RDS_DB']

def get_db_connection():
    conn = pymysql.connect(
        host=RDS_HOST,
        port=RDS_PORT,
        user=RDS_USER,
        password=RDS_PASSWORD,
        db=RDS_DB
    )
    return conn

def lambda_handler(event, context):
    try:
        query_type = event.get('query_type')
        params = event.get('params', {})
        
        if query_type == 'vessel_list':
            query = "SELECT DISTINCT vessel_name FROM vessel_performance"
            with get_db_connection() as conn:
                with conn.cursor() as cursor:
                    cursor.execute(query)
                    results = cursor.fetchall()
            return {
                'data': [row[0] for row in results]
            }
        
        elif query_type == 'performance_data':
            vessel_name = params.get('vessel_name')
            query = f"SELECT * FROM vessel_performance WHERE vessel_name = %s"
            with get_db_connection() as conn:
                df = pd.read_sql(query, conn, params=(vessel_name,))
            return {
                'data': df.to_dict(orient='records')
            }
        
        # Add other query types as needed
        
        else:
            return {
                'error': f'Unknown query type: {query_type}'
            }
    
    except Exception as e:
        return {
            'error': str(e)
        }
    """, language="python")

# Run the app
if __name__ == "__main__":
    st.sidebar.markdown("---")
    st.sidebar.info("Marine Performance Analysis System v1.0")
