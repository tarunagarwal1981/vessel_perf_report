import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, date, timedelta
import plotly.graph_objects as go
import requests
import json
import traceback

class CIIAgent:
    def __init__(self):
        # Emission factors for different fuel types
        self.emission_factors = {
            'VLSFO': 3.151,
            'LSMGO': 3.206,
            'LNG': 2.75,
            'HFO': 3.114,
            'LFO': 3.151,
            'GO_DO': 3.206,
            'LPG': 3.00,
            'METHANOL': 1.375,
            'ETHANOL': 1.913
        }
        
        # Vessel type mapping for IMO CII calculations
        self.vessel_type_mapping = {
            'ASPHALT/BITUMEN TANKER': 'tanker',
            'BULK CARRIER': 'bulk_carrier',
            'CEMENT CARRIER': 'bulk_carrier',
            'CHEM/PROD TANKER': 'tanker',
            'CHEMICAL TANKER': 'tanker',
            'Chemical/Products Tanker': 'tanker',
            'Combination Carrier': 'combination_carrier',
            'CONTAINER': 'container_ship',
            'Container Ship': 'container_ship',
            'Container/Ro-Ro Ship': 'ro_ro_cargo_ship',
            'Crude Oil Tanker': 'tanker',
            'Gas Carrier': 'gas_carrier',
            'General Cargo Ship': 'general_cargo_ship',
            'LNG CARRIER': 'lng_carrier',
            'LPG CARRIER': 'gas_carrier',
            'LPG Tanker': 'gas_carrier',
            'OIL TANKER': 'tanker',
            'Products Tanker': 'tanker',
            'Refrigerated Cargo Ship': 'refrigerated_cargo_carrier',
            'Ro-Ro Ship': 'ro_ro_cargo_ship',
            'Vehicle Carrier': 'ro_ro_cargo_ship_vc'
        }
        
        # CII reference parameters by ship type
        self.cii_reference_params = {
            'bulk_carrier': [{'capacity_threshold': 279000, 'a': 4745, 'c': 0.622}],
            'gas_carrier': [{'capacity_threshold': 65000, 'a': 144050000000, 'c': 2.071}],
            'tanker': [{'capacity_threshold': float('inf'), 'a': 5247, 'c': 0.61}],
            'container_ship': [{'capacity_threshold': float('inf'), 'a': 1984, 'c': 0.489}],
            'general_cargo_ship': [{'capacity_threshold': float('inf'), 'a': 31948, 'c': 0.792}],
            'refrigerated_cargo_carrier': [{'capacity_threshold': float('inf'), 'a': 4600, 'c': 0.557}],
            'lng_carrier': [{'capacity_threshold': 100000, 'a': 144790000000000, 'c': 2.673}],
        }
        
        # Reduction factors by year
        self.reduction_factors = {
            2023: 0.95, 
            2024: 0.93, 
            2025: 0.91, 
            2026: 0.89
        }
    
    def run(self, vessel_data, selected_vessel):
        """
        Main method to run the CII agent analysis
        """
        st.header("Carbon Intensity Indicator (CII) Analysis")
        
        try:
            # Process data to calculate CII metrics
            cii_data = self._process_cii_data(vessel_data, selected_vessel)
            
            if cii_data:
                # Display CII summary
                self._display_cii_summary(cii_data)
                
                # Create and display CII trend chart
                cii_chart = self._create_cii_trend_chart(cii_data)
                if cii_chart:
                    st.plotly_chart(cii_chart, use_container_width=True)
                
            else:
                st.warning("No CII data available for the selected vessel. Please check if the vessel has reported consumption data.")
        
        except Exception as e:
            st.error(f"Error in CII analysis: {str(e)}")
            st.code(traceback.format_exc())
    
    def _process_cii_data(self, vessel_data, selected_vessel):
        """
        Process vessel data to calculate CII metrics
        """
        try:
            # Example of how to extract data for CII calculation
            # In a real scenario, you would fetch this from your Lambda function
            
            # Filter data for the current vessel
            vessel_specific_data = []
            
            for entry in vessel_data:
                if 'VESSEL_NAME' in entry and entry['VESSEL_NAME'] == selected_vessel:
                    vessel_specific_data.append(entry)
            
            if not vessel_specific_data:
                return None
            
            # Extract vessel particulars (assuming these are available in vessel_data)
            vessel_particulars = self._extract_vessel_particulars(vessel_specific_data)
            
            # Group data by month for trend analysis
            monthly_data = self._group_data_by_month(vessel_specific_data)
            
            # Calculate CII metrics for each month
            cii_metrics = self._calculate_monthly_cii(monthly_data, vessel_particulars)
            
            return {
                'vessel_name': selected_vessel,
                'vessel_particulars': vessel_particulars,
                'monthly_data': monthly_data,
                'cii_metrics': cii_metrics,
                'latest_cii': cii_metrics[-1] if cii_metrics else None
            }
        
        except Exception as e:
            st.error(f"Error processing CII data: {str(e)}")
            return None
    
    def _extract_vessel_particulars(self, vessel_data):
        """
        Extract vessel particulars from data
        """
        # In a real scenario, this would extract actual vessel particulars
        # For demonstration, we'll create sample data
        
        # Use the first entry to get vessel type and IMO
        first_entry = vessel_data[0]
        vessel_type = first_entry.get('VESSEL_TYPE', 'BULK CARRIER')
        vessel_imo = first_entry.get('VESSEL_IMO', '9999999')
        
        # Map to IMO vessel type
        imo_ship_type = self.vessel_type_mapping.get(vessel_type, 'bulk_carrier')
        
        # Assume a capacity based on vessel type
        capacity = 80000  # Default capacity (deadweight tons)
        
        return {
            'vessel_imo': vessel_imo,
            'vessel_type': vessel_type,
            'imo_ship_type': imo_ship_type,
            'capacity': capacity
        }
    
    def _group_data_by_month(self, vessel_data):
        """
        Group vessel data by month for trend analysis
        """
        monthly_data = {}
        
        # For demonstration, create sample monthly data
        # In a real scenario, this would group actual data
        
        current_year = date.today().year
        start_date = date(current_year, 1, 1)
        end_date = date.today()
        
        # Generate monthly data points
        current_date = start_date
        while current_date <= end_date:
            month_key = current_date.strftime('%Y-%m')
            
            # Generate sample data for this month
            # In reality, you would aggregate actual vessel data
            monthly_data[month_key] = {
                'month': current_date,
                'distance': np.random.uniform(8000, 12000),  # Sample distance in nautical miles
                'co2_emissions': np.random.uniform(1500, 2500),  # Sample CO2 emissions in metric tons
                'days_at_sea': np.random.randint(20, 30)  # Sample days at sea
            }
            
            # Move to next month
            if current_date.month == 12:
                current_date = date(current_date.year + 1, 1, 1)
            else:
                current_date = date(current_date.year, current_date.month + 1, 1)
        
        return monthly_data
    
    def _calculate_monthly_cii(self, monthly_data, vessel_particulars):
        """
        Calculate CII metrics for each month
        """
        cii_metrics = []
        capacity = vessel_particulars['capacity']
        imo_ship_type = vessel_particulars['imo_ship_type']
        
        # Calculate reference CII
        reference_cii = self._calculate_reference_cii(capacity, imo_ship_type)
        
        # Calculate required CII for the current year
        current_year = date.today().year
        required_cii = self._calculate_required_cii(reference_cii, current_year)
        
        cumulative_distance = 0
        cumulative_co2 = 0
        
        # Calculate metrics for each month
        for month_key, month_data in sorted(monthly_data.items()):
            month_date = month_data['month']
            distance = month_data['distance']
            co2 = month_data['co2_emissions']
            
            # Calculate monthly AER
            monthly_aer = (co2 * 1000000) / (distance * capacity)
            
            # Calculate cumulative values
            cumulative_distance += distance
            cumulative_co2 += co2
            
            # Calculate cumulative AER (YTD)
            cumulative_aer = (cumulative_co2 * 1000000) / (cumulative_distance * capacity)
            
            # Determine CII rating
            cii_rating = self._calculate_cii_rating(cumulative_aer, required_cii)
            
            cii_metrics.append({
                'date': month_date,
                'month': month_date.strftime('%b %Y'),
                'monthly_aer': monthly_aer,
                'cumulative_aer': cumulative_aer,
                'required_cii': required_cii,
                'cii_rating': cii_rating,
                'distance': distance,
                'co2_emissions': co2,
                'cumulative_distance': cumulative_distance,
                'cumulative_co2': cumulative_co2
            })
        
        return cii_metrics
    
    def _calculate_reference_cii(self, capacity, ship_type):
        """
        Calculate reference CII based on capacity and ship type
        """
        # Use the reference parameters for the ship type
        ship_params = self.cii_reference_params.get(ship_type.lower())
        if not ship_params:
            # Use bulk carrier as default if ship type is not found
            ship_params = self.cii_reference_params.get('bulk_carrier')
        
        # Use the first set of parameters (simplification)
        a, c = ship_params[0]['a'], ship_params[0]['c']
        return a * (capacity ** -c)
    
    def _calculate_required_cii(self, reference_cii, year):
        """
        Calculate required CII based on reference CII and year
        """
        # Use the reduction factor for the year or default to 1.0
        reduction_factor = self.reduction_factors.get(year, 1.0)
        return reference_cii * reduction_factor
    
    def _calculate_cii_rating(self, attained_cii, required_cii):
        """
        Calculate CII rating based on attained and required CII
        """
        if attained_cii <= required_cii * 0.95:
            return 'A'
        elif attained_cii <= required_cii:
            return 'B'
        elif attained_cii <= required_cii * 1.05:
            return 'C'
        elif attained_cii <= required_cii * 1.15:
            return 'D'
        else:
            return 'E'
    
    def _display_cii_summary(self, cii_data):
        """
        Display summary of CII metrics
        """
        latest_cii = cii_data['latest_cii']
        vessel_name = cii_data['vessel_name']
        vessel_type = cii_data['vessel_particulars']['vessel_type']
        
        # Create columns for metrics
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("### Current CII Rating")
            rating = latest_cii['cii_rating']
            rating_color = self._get_rating_color(rating)
            st.markdown(f"""
            <div style="background-color: {rating_color}; padding: 20px; border-radius: 5px; text-align: center;">
                <h1 style="color: white; margin: 0;">{rating}</h1>
                <p style="color: white; margin: 0;">CII Rating</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown("### Attained AER")
            st.metric(
                "Current AER",
                f"{latest_cii['cumulative_aer']:.2f} gCO₂/dwt-nm",
                delta=f"{latest_cii['cumulative_aer'] - latest_cii['required_cii']:.2f} vs Required",
                delta_color="inverse"
            )
        
        with col3:
            st.markdown("### Vessel Information")
            st.markdown(f"""
            **Vessel Name**: {vessel_name}  
            **Vessel Type**: {vessel_type}  
            **Required CII**: {latest_cii['required_cii']:.2f} gCO₂/dwt-nm
            """)
        
        # Add explanation of what CII means
        with st.expander("What is CII?"):
            st.markdown("""
            **Carbon Intensity Indicator (CII)** is an operational efficiency measure that applies to ships of 5,000 GT and above. 
            It determines the annual reduction factor needed to ensure continuous improvement of the ship's operational carbon intensity 
            within a specific rating level.

            The CII is based on the Annual Efficiency Ratio (AER), which is calculated as:
            
            AER = (CO₂ emissions in grams) / (deadweight tonnage × distance traveled in nautical miles)
            
            Ships are rated on a scale from A to E, where A is the best performance (lowest carbon intensity).
            """)
    
    def _get_rating_color(self, rating):
        """
        Get color code for CII rating
        """
        rating_colors = {
            'A': '#4CAF50',  # Green
            'B': '#8BC34A',  # Light Green
            'C': '#FFC107',  # Amber
            'D': '#FF9800',  # Orange
            'E': '#F44336'   # Red
        }
        return rating_colors.get(rating, '#757575')  # Default to gray
    
    def _create_cii_trend_chart(self, cii_data):
        """
        Create CII trend chart
        """
        try:
            # Extract metrics for plotting
            metrics = cii_data['cii_metrics']
            
            dates = [metric['date'] for metric in metrics]
            cumulative_aer = [metric['cumulative_aer'] for metric in metrics]
            monthly_aer = [metric['monthly_aer'] for metric in metrics]
            required_cii = [metric['required_cii'] for metric in metrics]
            
            # Create figure
            fig = go.Figure()
            
            # Add monthly AER data (individual points)
            fig.add_trace(go.Scatter(
                x=dates, 
                y=monthly_aer,
                mode='markers',
                name='Monthly AER',
                marker=dict(
                    size=10,
                    color='rgba(0, 170, 255, 0.7)',
                    line=dict(width=1, color='rgb(0, 120, 180)')
                )
            ))
            
            # Add cumulative AER data (trend line)
            fig.add_trace(go.Scatter(
                x=dates, 
                y=cumulative_aer,
                mode='lines+markers',
                name='Cumulative AER (YTD)',
                line=dict(color='rgb(0, 170, 255)', width=3),
                marker=dict(size=8)
            ))
            
            # Add required CII line
            fig.add_trace(go.Scatter(
                x=dates, 
                y=required_cii,
                mode='lines',
                name='Required CII',
                line=dict(color='rgb(255, 152, 0)', width=2, dash='dash')
            ))
            
            # Add reference lines for CII ratings
            # Assuming 'C' rating range is between required and required*1.05
            fig.add_trace(go.Scatter(
                x=dates,
                y=[r * 0.95 for r in required_cii],
                mode='lines',
                name='A-B Boundary',
                line=dict(color='rgb(139, 195, 74)', width=1, dash='dot')
            ))
            
            fig.add_trace(go.Scatter(
                x=dates,
                y=[r * 1.05 for r in required_cii],
                mode='lines',
                name='C-D Boundary',
                line=dict(color='rgb(255, 152, 0)', width=1, dash='dot')
            ))
            
            fig.add_trace(go.Scatter(
                x=dates,
                y=[r * 1.15 for r in required_cii],
                mode='lines',
                name='D-E Boundary',
                line=dict(color='rgb(244, 67, 54)', width=1, dash='dot')
            ))
            
            # Update layout
            fig.update_layout(
                title="Carbon Intensity Indicator (CII) Trend",
                xaxis_title="Month",
                yaxis_title="AER (gCO₂/dwt-nm)",
                template="plotly_dark",
                height=500,
                legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=1.02,
                    xanchor="center",
                    x=0.5
                )
            )
            
            # Customize x-axis to show month names
            fig.update_xaxes(
                tickformat="%b %Y",
                tickmode="array",
                tickvals=dates,
                ticktext=[d.strftime("%b %Y") for d in dates]
            )
            
            return fig
            
        except Exception as e:
            st.error(f"Error creating CII trend chart: {str(e)}")
            return None
    
    def get_cii_data_for_report(self, vessel_data, selected_vessel):
        """
        Public method to get CII data for report generation
        Returns CII metrics and chart for report
        """
        try:
            # Process data to calculate CII metrics
            cii_data = self._process_cii_data(vessel_data, selected_vessel)
            
            if not cii_data:
                return None, None
            
            # Create CII trend chart
            cii_chart = self._create_cii_trend_chart(cii_data)
            
            # Get the latest CII metrics
            latest_cii = cii_data['latest_cii']
            
            # Format metrics for report
            cii_metrics = {
                'rating': latest_cii['cii_rating'],
                'attained_aer': latest_cii['cumulative_aer'],
                'required_cii': latest_cii['required_cii'],
                'cumulative_distance': latest_cii['cumulative_distance'],
                'cumulative_co2': latest_cii['cumulative_co2']
            }
            
            return cii_metrics, cii_chart
            
        except Exception as e:
            print(f"Error getting CII data for report: {str(e)}")
            traceback.print_exc()
            return None, None
