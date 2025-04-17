import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from typing import List, Dict, Any, Tuple, Optional

class SpeedConsumptionTableAgent:
    def __init__(self):
        pass
    
    def run(self, data: List[Dict[Any, Any]], vessel_name: str):
        """Main entry point for the agent"""
        # Filter data for speed-consumption analysis
        filtered_data_consumption = []
        
        for entry in data:
            if ('speed' in entry and entry['speed'] is not None and 
                'normalised_consumption' in entry and entry['normalised_consumption'] is not None and
                'loading_condition' in entry and entry['loading_condition'] is not None):
                filtered_data_consumption.append(entry)
        
        if not filtered_data_consumption:
            st.warning("No valid speed-consumption data found for the selected vessel and filters.")
            return
        
        # Create tabs for the tables and charts
        sc_tab1, sc_tab2 = st.tabs(["Consumption Table", "Consumption Charts"])
        
        # Process both conditions
        ballast_coeffs = self.process_consumption_data(filtered_data_consumption, "ballast")
        laden_coeffs = self.process_consumption_data(filtered_data_consumption, "laden")
        
        # Table Tab
        with sc_tab1:
            st.subheader(f"Speed vs. Consumption Table - {vessel_name.upper()}")
            
            # Generate consumption table for speeds 8-15
            if ballast_coeffs is not None or laden_coeffs is not None:
                table_data = self.generate_consumption_table(ballast_coeffs, laden_coeffs)
                
                # Display the table
                st.dataframe(table_data, use_container_width=True)
                
                # Provide download button for CSV
                csv = table_data.to_csv(index=False)
                st.download_button(
                    label="Download Table as CSV",
                    data=csv,
                    file_name=f"{vessel_name}_consumption_table.csv",
                    mime="text/csv"
                )
                
                # Add explanation
                st.markdown("""
                ### Table Information
                
                - The table shows predicted fuel consumption at different speeds based on the vessel's performance data
                - Values are calculated using quadratic regression models derived from filtered operational data
                - Speeds range from 8 to 15 knots in 0.5 knot increments
                - If a condition has insufficient data, those columns will show 'N/A'
                """)
            else:
                st.info("Insufficient data to generate consumption table.")
        
        # Charts Tab
        with sc_tab2:
            # Create columns for the two conditions
            col1, col2 = st.columns(2)
            
            with col1:
                if ballast_coeffs is not None:
                    ballast_chart = self.create_consumption_curve_chart(ballast_coeffs, "Ballast Condition")
                    st.plotly_chart(ballast_chart, use_container_width=True)
                else:
                    st.info("Insufficient data for ballast condition chart.")
            
            with col2:
                if laden_coeffs is not None:
                    laden_chart = self.create_consumption_curve_chart(laden_coeffs, "Laden Condition")
                    st.plotly_chart(laden_chart, use_container_width=True)
                else:
                    st.info("Insufficient data for laden condition chart.")
    
    def process_consumption_data(self, data: List[Dict[Any, Any]], condition: str) -> Optional[np.ndarray]:
        """Process data for a specific loading condition and return polynomial coefficients"""
        # Filter data for the specified loading condition
        condition_data = [row for row in data if row.get('loading_condition', '').lower() == condition.lower()]
        
        if not condition_data or len(condition_data) < 3:  # Need at least 3 points for a meaningful quadratic fit
            return None
        
        # Extract speed and consumption data
        speeds = [row.get('speed', 0) for row in condition_data]
        consumptions = [row.get('normalised_consumption', 0) for row in condition_data]
        
        # Remove outliers using IQR method
        cleaned_speeds, cleaned_consumptions = self.remove_outliers(speeds, consumptions)
        
        if len(cleaned_speeds) < 3:  # Still need enough data after outlier removal
            return None
        
        # Fit a 2nd order polynomial (quadratic)
        coeffs = np.polyfit(cleaned_speeds, cleaned_consumptions, 2)
        return coeffs
    
    def remove_outliers(self, speeds: List[float], consumptions: List[float]) -> Tuple[List[float], List[float]]:
        """Remove outliers using the IQR method"""
        # Convert to numpy arrays for easier calculation
        speed_array = np.array(speeds)
        consumption_array = np.array(consumptions)
        
        # Calculate consumption per speed (efficiency)
        efficiency = consumption_array / speed_array
        
        # Calculate Q1, Q3 and IQR for efficiency
        q1 = np.percentile(efficiency, 25)
        q3 = np.percentile(efficiency, 75)
        iqr = q3 - q1
        
        # Define bounds for outliers (1.5 is the standard multiplier for outliers)
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        
        # Find indices of non-outliers
        mask = (efficiency >= lower_bound) & (efficiency <= upper_bound)
        
        # Return cleaned data
        return list(speed_array[mask]), list(consumption_array[mask])
    
    def generate_consumption_table(self, ballast_coeffs: Optional[np.ndarray], 
                                 laden_coeffs: Optional[np.ndarray]) -> pd.DataFrame:
        """Generate a consumption table for speeds from 8 to 15 knots"""
        # Create speed range from 8 to 15 knots in 0.5 knot increments
        speeds = np.arange(8, 15.5, 0.5)
        
        # Initialize data dictionary
        data = {'Speed (knots)': speeds}
        
        # Calculate consumption for ballast condition
        if ballast_coeffs is not None:
            ballast_consumption = [self.calculate_consumption(speed, ballast_coeffs) for speed in speeds]
            data['Ballast Consumption (mt/day)'] = ballast_consumption
        else:
            data['Ballast Consumption (mt/day)'] = ['N/A'] * len(speeds)
        
        # Calculate consumption for laden condition
        if laden_coeffs is not None:
            laden_consumption = [self.calculate_consumption(speed, laden_coeffs) for speed in speeds]
            data['Laden Consumption (mt/day)'] = laden_consumption
        else:
            data['Laden Consumption (mt/day)'] = ['N/A'] * len(speeds)
        
        # Add efficiency columns (if data is available)
        if ballast_coeffs is not None:
            ballast_efficiency = [round(consumption / speed, 2) if isinstance(consumption, (int, float)) else 'N/A' 
                                 for speed, consumption in zip(speeds, data['Ballast Consumption (mt/day)'])]
            data['Ballast Efficiency (mt/knot)'] = ballast_efficiency
        
        if laden_coeffs is not None:
            laden_efficiency = [round(consumption / speed, 2) if isinstance(consumption, (int, float)) else 'N/A'
                               for speed, consumption in zip(speeds, data['Laden Consumption (mt/day)'])]
            data['Laden Efficiency (mt/knot)'] = laden_efficiency
        
        # Create DataFrame
        df = pd.DataFrame(data)
        
        # Format the table
        for col in df.columns:
            if col != 'Speed (knots)':
                df[col] = df[col].apply(lambda x: round(x, 2) if isinstance(x, (int, float)) else x)
        
        return df
    
    def calculate_consumption(self, speed: float, coeffs: np.ndarray) -> float:
        """Calculate consumption using the quadratic formula"""
        return coeffs[0] * speed**2 + coeffs[1] * speed + coeffs[2]
    
    def create_consumption_curve_chart(self, coeffs: np.ndarray, condition: str) -> go.Figure:
        """Create a chart showing the consumption curve"""
        # Create speed range from 7 to 16 knots (slightly wider than table for better visualization)
        speeds = np.linspace(7, 16, 100)
        consumptions = [self.calculate_consumption(speed, coeffs) for speed in speeds]
        
        # Calculate efficiency
        efficiency = [consumption / speed for speed, consumption in zip(speeds, consumptions)]
        
        # Find optimal speed (minimum efficiency)
        optimal_idx = np.argmin(efficiency)
        optimal_speed = speeds[optimal_idx]
        optimal_consumption = consumptions[optimal_idx]
        
        # Create figure
        fig = go.Figure()
        
        # Add consumption curve
        fig.add_trace(go.Scatter(
            x=speeds,
            y=consumptions,
            mode='lines',
            line=dict(color='#0077b6', width=3),
            name='Consumption Curve'
        ))
        
        # Add marker for optimal speed
        fig.add_trace(go.Scatter(
            x=[optimal_speed],
            y=[optimal_consumption],
            mode='markers',
            marker=dict(color='#e63946', size=12, symbol='star'),
            name=f'Optimal Speed: {optimal_speed:.2f} knots'
        ))
        
        # Highlight the region from 8-15 knots (table range)
        fig.add_vrect(
            x0=8, x1=15,
            fillcolor="#90e0ef",
            opacity=0.2,
            layer="below",
            line_width=0,
            annotation_text="Table Range",
            annotation_position="top left"
        )
        
        # Add equation annotation
        equation = f"y = {coeffs[0]:.4f}x² + {coeffs[1]:.4f}x + {coeffs[2]:.4f}"
        fig.add_annotation(
            x=0.05,
            y=0.95,
            xref="paper",
            yref="paper",
            text=equation,
            showarrow=False,
            bgcolor="rgba(255,255,255,0.8)",
            bordercolor="#0077b6",
            borderwidth=1,
            borderpad=4
        )
        
        # Update layout
        fig.update_layout(
            title=f"Speed vs. Consumption - {condition}",
            xaxis_title="Speed (knots)",
            yaxis_title="Fuel Consumption (mt/day)",
            template="plotly_white",
            height=400,
            margin=dict(l=20, r=20, t=50, b=20)
        )
        
        # Update axes to start at 0
        fig.update_yaxes(rangemode="tozero")
        
        return fig
