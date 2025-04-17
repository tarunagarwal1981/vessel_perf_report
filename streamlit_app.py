import streamlit as st
import datetime
import pandas as pd
import traceback
import os

# For debug logging
os.environ['STREAMLIT_LOG_LEVEL'] = 'debug'

# Import the agents
from agents.hull_performance import HullPerformanceAgent
from agents.speed_consumption import SpeedConsumptionAgent
from agents.speed_consumption_table import SpeedConsumptionTableAgent  # Import the new Table Agent
from agents.advanced_report import AdvancedReportGenerator
from agents.cii_agent import CIIAgent
from utils.data_fetcher import fetch_data_from_lambda

# Initialize session state for persistence
if 'vessel_data' not in st.session_state:
    st.session_state.vessel_data = None
if 'cii_data' not in st.session_state:
    st.session_state.cii_data = None
if 'selected_vessel' not in st.session_state:
    st.session_state.selected_vessel = ""
if 'start_date' not in st.session_state:
    st.session_state.start_date = datetime.datetime.now() - datetime.timedelta(days=180)
if 'end_date' not in st.session_state:
    st.session_state.end_date = datetime.datetime.now()
if 'weather_option' not in st.session_state:
    st.session_state.weather_option = "Good Weather (Wind Force ≤ 4)"
if 'analysis_completed' not in st.session_state:
    st.session_state.analysis_completed = False
if 'error_message' not in st.session_state:
    st.session_state.error_message = None

# Set page configuration
st.set_page_config(
    page_title="Marine Performance Analysis",
    page_icon="🚢",
    layout="wide"
)

# App title
st.title("🚢 Marine Performance Analysis System")

# Sidebar for vessel input and date range
st.sidebar.header("Configuration")

# User input for vessel name
selected_vessel = st.sidebar.text_input("Enter Vessel Name", st.session_state.selected_vessel)
st.session_state.selected_vessel = selected_vessel

# Date selectors
st.sidebar.subheader("Date Range")
start_date = st.sidebar.date_input("Start Date", value=st.session_state.start_date)
end_date = st.sidebar.date_input("End Date", value=st.session_state.end_date)

st.session_state.start_date = start_date
st.session_state.end_date = end_date

# Weather condition filter
st.sidebar.subheader("Weather Filter")
weather_option = st.sidebar.radio(
    "Select Weather Condition",
    options=["All Weather", "Good Weather (Wind Force ≤ 4)", "Bad Weather (Wind Force > 4)"],
    index=1 if st.session_state.weather_option == "Good Weather (Wind Force ≤ 4)" else 
           (2 if st.session_state.weather_option == "Bad Weather (Wind Force > 4)" else 0)
)
st.session_state.weather_option = weather_option

# Add reset button if analysis has been completed
if st.session_state.analysis_completed:
    if st.sidebar.button("Reset / New Analysis"):
        # Clear the session state
        st.session_state.vessel_data = None
        st.session_state.cii_data = None
        st.session_state.selected_vessel = ""
        st.session_state.analysis_completed = False
        st.session_state.error_message = None
        # Force a page refresh
        st.rerun()

# Check if dates are valid
if start_date > end_date:
    st.error("Error: Start date must be before end date")
elif not selected_vessel:
    st.info("Please enter a vessel name in the sidebar.")
else:
    # Display error message if it exists in session state
    if st.session_state.error_message:
        st.error(st.session_state.error_message)
    
    # Button to fetch data
    if not st.session_state.analysis_completed and st.sidebar.button("Analyze Vessel Performance"):
        try:
            with st.spinner(f"Fetching performance data for {selected_vessel}..."):
                # Prepare parameters for the lambda function
                params = {
                    "vesselName": selected_vessel,
                    "startDate": start_date.isoformat(),
                    "endDate": end_date.isoformat()
                }
                
                # Fetch hull performance data from Lambda
                vessel_data = fetch_data_from_lambda("getHullPerformance", params)
                
                # Fetch CII data from Lambda
                cii_data = fetch_data_from_lambda("getVesselCIIData", params)
                
                # Store in session state
                st.session_state.vessel_data = vessel_data
                st.session_state.cii_data = cii_data
                
                if vessel_data and len(vessel_data) > 0:
                    # Filter hull performance data based on weather option
                    filtered_data = vessel_data
                    if weather_option == "Good Weather (Wind Force ≤ 4)":
                        filtered_data = [row for row in vessel_data if row.get('windforce', 0) <= 4]
                    elif weather_option == "Bad Weather (Wind Force > 4)":
                        filtered_data = [row for row in vessel_data if row.get('windforce', 0) > 4]
                    
                    # Update vessel_data in session state after filtering
                    st.session_state.vessel_data = filtered_data
                    
                    # Set analysis as completed
                    st.session_state.analysis_completed = True
                    st.session_state.error_message = None
                    
                    # Use st.rerun() to refresh the page
                    st.rerun()
                else:
                    st.session_state.error_message = f"No performance data available for {selected_vessel} in the selected date range."
                    st.error(st.session_state.error_message)
        except Exception as e:
            error_msg = f"Error fetching data: {str(e)}"
            st.session_state.error_message = error_msg
            st.error(error_msg)
            st.code(traceback.format_exc())
    
    # If analysis is completed and we have data, show the analysis tabs
    if st.session_state.analysis_completed and st.session_state.vessel_data:
        try:
            # Create main tabs for different analysis types
            main_tab1, main_tab2, main_tab3, main_tab4, main_tab5 = st.tabs([
                "Hull Performance", 
                "Speed-Consumption Analysis",
                "Speed-Consumption Table",  # New tab for the table agent
                "CII Analysis",
                "Generate Report"
            ])
            
            # Initialize the agents
            hull_agent = HullPerformanceAgent()
            speed_agent = SpeedConsumptionAgent()
            speed_table_agent = SpeedConsumptionTableAgent()  # Initialize the new table agent
            cii_agent = CIIAgent()
            report_agent = AdvancedReportGenerator()
            
            # Hull Performance Tab
            with main_tab1:
                hull_agent.run(st.session_state.vessel_data, st.session_state.selected_vessel)
            
            # Speed-Consumption Tab
            with main_tab2:
                speed_agent.run(st.session_state.vessel_data, st.session_state.selected_vessel)
            
            # Speed-Consumption Table Tab
            with main_tab3:
                speed_table_agent.run(st.session_state.vessel_data, st.session_state.selected_vessel)
            
            # CII Analysis Tab
            with main_tab4:
                # Use CII data if available, otherwise use hull performance data
                data_for_cii = st.session_state.cii_data if st.session_state.cii_data else st.session_state.vessel_data
                cii_agent.run(data_for_cii, st.session_state.selected_vessel)
            
            # Report Generator Tab
            with main_tab5:
                report_agent.run(
                    st.session_state.vessel_data, 
                    st.session_state.selected_vessel, 
                    hull_agent, 
                    speed_agent,
                    cii_agent  # Pass the CII agent to the report generator
                )
            
            # Display the raw data
            with st.expander("View Raw Data"):
                # Add tabs for different data types
                data_tab1, data_tab2 = st.tabs(["Hull Performance Data", "CII Data"])
                
                with data_tab1:
                    # Convert to DataFrame for display
                    df = pd.DataFrame(st.session_state.vessel_data)
                    if 'report_date' in df.columns:
                        df['report_date'] = pd.to_datetime(df['report_date'])
                    st.dataframe(df)
                
                with data_tab2:
                    if st.session_state.cii_data:
                        # Convert to DataFrame for display
                        cii_df = pd.DataFrame(st.session_state.cii_data)
                        if 'report_date' in cii_df.columns:
                            cii_df['report_date'] = pd.to_datetime(cii_df['report_date'])
                        st.dataframe(cii_df)
                    else:
                        st.info("No CII data available. Please check Lambda function.")
        
        except Exception as e:
            error_msg = f"Error during analysis: {str(e)}"
            st.error(error_msg)
            st.code(traceback.format_exc())
    
    elif not st.session_state.analysis_completed:
        # Initial instruction
        st.info("👈 Enter a vessel name, select date range and weather filter, then click 'Analyze Vessel Performance' to view the analysis.")

# Footer
st.sidebar.markdown("---")
st.sidebar.info("Marine Performance Analysis System v1.0")
