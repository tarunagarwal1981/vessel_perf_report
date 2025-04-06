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
        
        # Lambda endpoint URL
        self.lambda_url = "https://crcgfvseuzhdqhhvan5gz2hr4e0kirfy.lambda-url.ap-south-1.on.aws/"
    
    def run(self, vessel_data, selected_vessel):
        """
        Main method to run the CII agent analysis
        """
        st.header("Carbon Intensity Indicator (CII) Analysis")
        
        try:
            # Get start and end dates for filtering
            end_date = date.today()
            start_date = date(end_date.year, 1, 1)  # Jan 1st of current year
            
            # Fetch CII data from Lambda
            cii_data = self._fetch_cii_data(selected_vessel, start_date, end_date)
            
            if cii_data:
                # Process the CII data
                processed_data = self._process_cii_data(cii_data, selected_vessel)
                
                # Display CII summary
                self._display_cii_summary(processed_data)
                
                # Create and display CII trend chart
                cii_chart = self._create_cii_trend_chart(processed_data)
                if cii_chart:
                    st.plotly_chart(cii_chart, use_container_width=True)
                
            else:
                st.warning("No CII data available for the selected vessel. Please check if the vessel has reported consumption data.")
        
        except Exception as e:
            st.error(f"Error in CII analysis: {str(e)}")
            st.code(traceback.format_exc())
    
    def _fetch_cii_data(self, vessel_name, start_date, end_date):
        """
        Fetch CII data from Lambda function
        """
        try:
            # Prepare request payload
            payload = {
                "operation": "getVesselCIIData",
                "vesselName": vessel_name,
                "startDate": start_date.isoformat(),
                "endDate": end_date.isoformat()
            }
            
            # Make request to Lambda
            response = requests.post(
                self.lambda_url,
                json=payload,
                headers={"Content-Type": "application/json"}
            )
            
            # Check if request was successful
            if response.status_code == 200:
                result = response.json()
                if result.get("success"):
                    return result.get("data", [])
                else:
                    st.error(f"API Error: {result.get('error', 'Unknown error')}")
                    return []
            else:
                st.error(f"HTTP Error: {response.status_code}")
                return []
                
        except Exception as e:
            st.error(f"Error fetching CII data from Lambda: {str(e)}")
            return []
    
    def _process_cii_data(self, vessel_data, selected_vessel):
        """
        Process vessel data to calculate CII metrics on a daily basis
        """
        try:
            if not vessel_data:
                return None
            
            # Extract vessel particulars from the first record
            first_record = vessel_data[0]
            vessel_particulars = {
                'vessel_imo': first_record.get('vessel_imo'),
                'vessel_type': first_record.get('vessel_type_particular'),
                'imo_ship_type': self.vessel_type_mapping.get(first_record.get('vessel_type_particular'), 'bulk_carrier'),
                'capacity': self._safe_float(first_record.get('deadweight', 0))
            }
            
            # Sort data by date
            sorted_data = sorted(vessel_data, key=lambda x: self._parse_date(x.get('report_date')))
            
            # Calculate daily cumulative CII metrics
            cii_metrics = self._calculate_daily_cii(sorted_data, vessel_particulars)
            
            return {
                'vessel_name': selected_vessel,
                'vessel_particulars': vessel_particulars,
                'cii_metrics': cii_metrics,
                'latest_cii': cii_metrics[-1] if cii_metrics else None
            }
        
        except Exception as e:
            st.error(f"Error processing CII data: {str(e)}")
            traceback.print_exc()
            return None
    
    def _parse_date(self, date_str):
        """
        Parse date string to datetime.date object
        """
        if isinstance(date_str, str):
            try:
                # Remove 'Z' and timezone info if present
                date_str = date_str.replace('Z', '')
                if '+' in date_str:
                    date_str = date_str.split('+')[0]
                return datetime.fromisoformat(date_str).date()
            except ValueError:
                # Try parsing with different format if ISO format fails
                try:
                    return datetime.strptime(date_str, '%Y-%m-%d').date()
                except ValueError:
                    return date.today()
        elif isinstance(date_str, datetime):
            return date_str.date()
        elif isinstance(date_str, date):
            return date_str
        else:
            return date.today()
    
    def _safe_float(self, value, default=0.0):
        """
        Safely convert value to float
        """
        if value is None:
            return default
        if isinstance(value, (int, float)):
            return float(value)
        if isinstance(value, str):
            try:
                return float(value)
            except (ValueError, TypeError):
                return default
        return default
    
    def _calculate_daily_cii(self, sorted_data, vessel_particulars):
        """
        Calculate CII metrics on a daily YTD basis
        Returns cumulative values for each day
        """
        cii_metrics = []
        capacity = vessel_particulars['capacity']
        imo_ship_type = vessel_particulars['imo_ship_type']
        
        # Calculate reference CII
        reference_cii = self._calculate_reference_cii(capacity, imo_ship_type)
        
        # Calculate required CII for the current year
        # Fix: Parse the date string from the first record
        if sorted_data:
            first_date = self._parse_date(sorted_data[0]['report_date'])
            current_year = first_date.year
        else:
            current_year = date.today().year
            
        required_cii = self._calculate_required_cii(reference_cii, current_year)
        
        cumulative_distance = 0
        cumulative_co2 = 0
        
        # Process each day's data
        for record in sorted_data:
            # Parse date
            report_date = self._parse_date(record.get('report_date'))
            
            # Get distance and CO2 for this day
            distance = self._safe_float(record.get('distance_travelled_actual', 0))
            co2 = self._calculate_co2_from_fuel(record)
            
            # Skip days with no distance
            if distance <= 0:
                continue
            
            # Update cumulative values
            cumulative_distance += distance
            cumulative_co2 += co2
            
            # Calculate AER (only if we have both distance and CO2)
            if cumulative_distance > 0 and capacity > 0:
                # YTD AER calculation
                ytd_aer = (cumulative_co2 * 1000000) / (cumulative_distance * capacity)
                
                # Determine CII rating
                cii_rating = self._calculate_cii_rating(ytd_aer, required_cii)
                
                # Store metrics for this day
                cii_metrics.append({
                    'date': report_date,
                    'daily_distance': distance,
                    'daily_co2': co2,
                    'cumulative_distance': cumulative_distance,
                    'cumulative_co2': cumulative_co2,
                    'attained_aer': ytd_aer,
                    'required_cii': required_cii,
                    'cii_rating': cii_rating
                })
        
        return cii_metrics
    
    def _calculate_co2_from_fuel(self, record):
        """
        Calculate CO2 emissions from fuel consumption
        """
        # Helper function to safely handle numeric values
        def safe_value(v):
            return self._safe_float(v)
        
        # Calculate CO2 for each fuel type
        # Subtracting fuel consumed at port (FC prefix) from total fuel
        hfo = safe_value(record.get('fuel_consumption_hfo')) - safe_value(record.get('fc_fuel_consumption_hfo'))
        lfo = safe_value(record.get('fuel_consumption_lfo')) - safe_value(record.get('fc_fuel_consumption_lfo'))
        go_do = safe_value(record.get('fuel_consumption_go_do')) - safe_value(record.get('fc_fuel_consumption_go_do'))
        lng = safe_value(record.get('fuel_consumption_lng')) - safe_value(record.get('fc_fuel_consumption_lng'))
        lpg = safe_value(record.get('fuel_consumption_lpg')) - safe_value(record.get('fc_fuel_consumption_lpg'))
        methanol = safe_value(record.get('fuel_consumption_methanol')) - safe_value(record.get('fc_fuel_consumption_methanol'))
        ethanol = safe_value(record.get('fuel_consumption_ethanol')) - safe_value(record.get('fc_fuel_consumption_ethanol'))
        
        # Apply emission factors
        co2_total = 0
        co2_total += max(0, hfo) * self.emission_factors['HFO']  # Ensure no negative values
        co2_total += max(0, lfo) * self.emission_factors['LFO']
        co2_total += max(0, go_do) * self.emission_factors['GO_DO']
        co2_total += max(0, lng) * self.emission_factors['LNG']
        co2_total += max(0, lpg) * self.emission_factors['LPG']
        co2_total += max(0, methanol) * self.emission_factors['METHANOL']
        co2_total += max(0, ethanol) * self.emission_factors['ETHANOL']
        
        return co2_total
    
    def _calculate_reference_cii(self, capacity, ship_type):
        """
        Calculate reference CII based on capacity and ship type
        """
        # Parameters with capacity thresholds
        params = {
            'bulk_carrier': [{'capacity_threshold': float('inf'), 'a': 4745, 'c': 0.622}],
            'gas_carrier': [
                {'capacity_threshold': 65000, 'a': 8104, 'c': 0.639},
                {'capacity_threshold': float('inf'), 'a': 144050000000, 'c': 2.071}
            ],
            'tanker': [{'capacity_threshold': float('inf'), 'a': 5247, 'c': 0.61}],
            'container_ship': [{'capacity_threshold': float('inf'), 'a': 1984, 'c': 0.489}],
            'general_cargo_ship': [{'capacity_threshold': float('inf'), 'a': 31948, 'c': 0.792}],
            'refrigerated_cargo_carrier': [{'capacity_threshold': float('inf'), 'a': 4600, 'c': 0.557}],
            'lng_carrier': [{'capacity_threshold': float('inf'), 'a': 144790000000000, 'c': 2.673}],
        }
        
        if capacity <= 0:
            st.warning(f"Invalid capacity: {capacity}. Using default value.")
            capacity = 10000  # Default fallback value
        
        ship_params = params.get(ship_type.lower())
        if not ship_params:
            st.warning(f"Unknown ship type: {ship_type}. Using bulk carrier as default.")
            ship_params = params.get('bulk_carrier')
        
        # Find the appropriate parameters based on capacity
        a, c = None, None
        
        for param in ship_params:
            if capacity < param['capacity_threshold']:
                a, c = param['a'], param['c']
                break
        
        # If no threshold was matched, use the last set of parameters
        if a is None or c is None:
            a, c = ship_params[-1]['a'], ship_params[-1]['c']
        
        return a * (capacity ** -c)
    
    def _calculate_required_cii(self, reference_cii, year):
        """
        Calculate required CII based on reference CII and year
        """
        # Use exactly the same reduction factors as in the provided code
        reduction_factors = {2023: 0.95, 2024: 0.93, 2025: 0.91, 2026: 0.89}
        return reference_cii * reduction_factors.get(year, 1.0)
    
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
        if not cii_data or not cii_data.get('latest_cii'):
            st.warning("No CII metrics available to display")
            return
            
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
                f"{latest_cii['attained_aer']:.2f} gCO₂/dwt-nm",
                delta=f"{latest_cii['attained_aer'] - latest_cii['required_cii']:.2f} vs Required",
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
        Create CII trend chart showing daily trend of AER values
        """
        try:
            if not cii_data or not cii_data.get('cii_metrics'):
                return None
                
            # Extract metrics for plotting
            metrics = cii_data['cii_metrics']
            
            # Select points for chart (to avoid overcrowding)
            # If more than 30 points, sample data to reduce density
            sampled_metrics = metrics
            if len(metrics) > 30:
                # Take every nth point to get about 30 points
                n = max(1, len(metrics) // 30)
                sampled_metrics = metrics[::n]
                # Always include the latest point
                if sampled_metrics[-1] != metrics[-1]:
                    sampled_metrics.append(metrics[-1])
            
            dates = [metric['date'] for metric in sampled_metrics]
            attained_aer = [metric['attained_aer'] for metric in sampled_metrics]
            required_cii = [metric['required_cii'] for metric in sampled_metrics]
            
            # Create figure
            fig = go.Figure()
            
            # Add AER trend line
            fig.add_trace(go.Scatter(
                x=dates, 
                y=attained_aer,
                mode='lines+markers',
                name='YTD Attained AER',
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
            # A-B boundary
            fig.add_trace(go.Scatter(
                x=dates,
                y=[r * 0.95 for r in required_cii],
                mode='lines',
                name='A-B Boundary',
                line=dict(color='rgb(139, 195, 74)', width=1, dash='dot')
            ))
            
            # C-D boundary
            fig.add_trace(go.Scatter(
                x=dates,
                y=[r * 1.05 for r in required_cii],
                mode='lines',
                name='C-D Boundary',
                line=dict(color='rgb(255, 152, 0)', width=1, dash='dot')
            ))
            
            # D-E boundary
            fig.add_trace(go.Scatter(
                x=dates,
                y=[r * 1.15 for r in required_cii],
                mode='lines',
                name='D-E Boundary',
                line=dict(color='rgb(244, 67, 54)', width=1, dash='dot')
            ))
            
            # Add annotations for the rating zones
            y_max = max(attained_aer) * 1.1
            y_min = min([min(attained_aer), min([r * 0.95 for r in required_cii])]) * 0.9
            
            # Update layout
            fig.update_layout(
                title="Carbon Intensity Indicator (CII) Trend (YTD)",
                xaxis_title="Date",
                yaxis_title="AER (gCO₂/dwt-nm)",
                template="plotly_dark",
                height=500,
                yaxis=dict(
                    range=[y_min, y_max]
                ),
                legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=1.02,
                    xanchor="center",
                    x=0.5
                ),
                annotations=[
                    dict(
                        x=dates[-1],
                        y=required_cii[-1] * 0.975,
                        xref="x",
                        yref="y",
                        text="A",
                        showarrow=False,
                        align="right",
                        font=dict(color="#4CAF50", size=14)
                    ),
                    dict(
                        x=dates[-1],
                        y=required_cii[-1] * 1.025,
                        xref="x",
                        yref="y",
                        text="B",
                        showarrow=False,
                        align="right",
                        font=dict(color="#8BC34A", size=14)
                    ),
                    dict(
                        x=dates[-1],
                        y=required_cii[-1] * 1.1,
                        xref="x",
                        yref="y",
                        text="C",
                        showarrow=False,
                        align="right",
                        font=dict(color="#FFC107", size=14)
                    ),
                    dict(
                        x=dates[-1],
                        y=required_cii[-1] * 1.2,
                        xref="x",
                        yref="y",
                        text="D",
                        showarrow=False,
                        align="right",
                        font=dict(color="#FF9800", size=14)
                    ),
                    dict(
                        x=dates[-1],
                        y=required_cii[-1] * 1.25,
                        xref="x",
                        yref="y",
                        text="E",
                        showarrow=False,
                        align="right",
                        font=dict(color="#F44336", size=14)
                    ),
                ]
            )
            
            # Format dates on x-axis
            fig.update_xaxes(
                tickformat="%d %b %Y",
                tickangle=-45
            )
            
            return fig
            
        except Exception as e:
            st.error(f"Error creating CII trend chart: {str(e)}")
            traceback.print_exc()
            return None
    
    def get_cii_data_for_report(self, vessel_data, selected_vessel):
        """
        Public method to get CII data for report generation
        Returns CII metrics and chart for report
        """
        try:
            # Determine date range for data (YTD)
            end_date = date.today()
            start_date = date(end_date.year, 1, 1)  # Jan 1st of current year
            
            # Fetch CII data from Lambda instead of using passed vessel_data
            cii_data_from_lambda = self._fetch_cii_data(selected_vessel, start_date, end_date)
            
            if not cii_data_from_lambda:
                return None, None
            
            # Process data to calculate CII metrics
            processed_data = self._process_cii_data(cii_data_from_lambda, selected_vessel)
            
            if not processed_data or not processed_data.get('latest_cii'):
                return None, None
            
            # Create CII trend chart
            cii_chart = self._create_cii_trend_chart(processed_data)
            
            # Get the latest CII metrics
            latest_cii = processed_data['latest_cii']
            
            # Format metrics for report
            cii_metrics = {
                'rating': latest_cii['cii_rating'],
                'attained_aer': latest_cii['attained_aer'],
                'required_cii': latest_cii['required_cii'],
                'cumulative_distance': latest_cii['cumulative_distance'],
                'cumulative_co2': latest_cii['cumulative_co2']
            }
            
            return cii_metrics, cii_chart
            
        except Exception as e:
            print(f"Error getting CII data for report: {str(e)}")
            traceback.print_exc()
            return None, None
