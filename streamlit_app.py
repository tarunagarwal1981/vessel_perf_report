import streamlit as st
import datetime
import pandas as pd
import os
os.environ['STREAMLIT_LOG_LEVEL'] = 'debug'

# Import the agents
from agents.hull_performance import HullPerformanceAgent
from agents.speed_consumption import SpeedConsumptionAgent
#from agents.report_generator import ReportGeneratorAgent  # Add the new agent
from utils.data_fetcher import fetch_data_from_lambda
from agents.minimal_report import MinimalReportGenerator

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
    if st.sidebar.button("Analyze Vessel Performance"):
        with st.spinner(f"Fetching performance data for {selected_vessel}..."):
            # Prepare parameters for the lambda function
            params = {
                "vesselName": selected_vessel,
                "startDate": start_date.isoformat(),
                "endDate": end_date.isoformat()
            }
            
            # Fetch data from Lambda
            vessel_data = fetch_data_from_lambda("getHullPerformance", params)
            
            if vessel_data and len(vessel_data) > 0:
                # Filter data based on weather option
                if weather_option == "Good Weather (Wind Force ≤ 4)":
                    vessel_data = [row for row in vessel_data if row.get('windforce', 0) <= 4]
                elif weather_option == "Bad Weather (Wind Force > 4)":
                    vessel_data = [row for row in vessel_data if row.get('windforce', 0) > 4]
                
                # Create main tabs for different analysis types
                main_tab1, main_tab2, main_tab3 = st.tabs(["Hull Performance", "Speed-Consumption Analysis", "Generate Report"])
                
                # Initialize the agents
                hull_agent = HullPerformanceAgent()
                speed_agent = SpeedConsumptionAgent()
                report_agent = MinimalReportGenerator()  # Initialize the report generator
                
                # Hull Performance Tab
                with main_tab1:
                    hull_agent.run(vessel_data, selected_vessel)
                
                # Speed-Consumption Tab
                with main_tab2:
                    speed_agent.run(vessel_data, selected_vessel)
                
                # Report Generator Tab
                with main_tab3:
                    report_agent.run(vessel_data, selected_vessel, hull_agent, speed_agent)
                
                # Display the raw data
                with st.expander("View Raw Data"):
                    # Convert to DataFrame for display
                    df = pd.DataFrame(vessel_data)
                    if 'report_date' in df.columns:
                        df['report_date'] = pd.to_datetime(df['report_date'])
                    st.dataframe(df)
            else:
                st.error(f"No performance data available for {selected_vessel} in the selected date range.")
    else:
        # Initial instruction
        st.info("👈 Enter a vessel name, select date range and weather filter, then click 'Analyze Vessel Performance' to view the analysis.")

# Footer
st.sidebar.markdown("---")
st.sidebar.info("Marine Performance Analysis System v1.0")
