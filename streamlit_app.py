import streamlit as st
import pandas as pd
from io import BytesIO
from docx import Document
import traceback

class MinimalReportGenerator:
    def __init__(self):
        self.report_generated = False
    
    def run(self, vessel_data, selected_vessel, hull_agent, speed_agent):
        st.header("Performance Report Generator")
        
        try:
            # Create tabs for report view and download
            report_tab1, report_tab2 = st.tabs(["Report Preview", "Download Report"])
            
            with report_tab1:
                # Display basic report preview
                st.subheader("Report Preview")
                
                # Extract basic metrics without creating charts
                hull_condition, power_loss, fuel_savings, recommendation = self._get_hull_metrics(vessel_data, hull_agent)
                ballast_avg, laden_avg = self._get_speed_metrics(vessel_data)
                
                # Display preview
                st.markdown("### Hull Performance")
                st.markdown(f"**Hull Condition:** {hull_condition}")
                st.markdown(f"**Power Loss:** {power_loss:.1f}%")
                st.markdown(f"**Potential Fuel Savings:** {fuel_savings:.1f} MT/D")
                st.markdown(f"**Recommendation:** {recommendation}")
                
                st.markdown("### Speed & Consumption")
                st.markdown(f"**Aver
