import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

class SpeedConsumptionAgent:
    def __init__(self):
        pass
    
    def run(self, data, vessel_name):
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
        
        # Create tabs for different loading conditions
        sc_tab1, sc_tab2 = st.tabs(["Ballast Condition", "Laden Condition"])
        
        # Ballast Condition Tab
        with sc_tab1:
            try:
                chart = self.create_speed_consumption_chart(
                    filtered_data_consumption,
                    "ballast",
                    f"Speed vs. Consumption - Ballast Condition ({vessel_name.upper()})"
                )
                
                if chart:
                    st.plotly_chart(chart, use_container_width=True)
                    
                    # Add descriptive text
                    st.markdown("""
                    ### Analysis
                    The chart above shows the relationship between vessel speed and fuel consumption in ballast condition.
                    
                    - Each point represents a daily report
                    - The blue curve is a 2nd order polynomial fit to the data
                    - Points are color-coded by date (newer points are brighter)
                    
                    This curve can be used to estimate the optimal speed for fuel efficiency.
                    """)
                else:
                    st.info("No ballast condition data available for the selected vessel and filters.")
            except Exception as e:
                st.error(f"Error creating Ballast Condition chart: {str(e)}")
        
        # Laden Condition Tab
        with sc_tab2:
            try:
                chart = self.create_speed_consumption_chart(
                    filtered_data_consumption,
                    "laden",
                    f"Speed vs. Consumption - Laden Condition ({vessel_name.upper()})"
                )
                
                if chart:
                    st.plotly_chart(chart, use_container_width=True)
                    
                    # Add descriptive text
                    st.markdown("""
                    ### Analysis
                    The chart above shows the relationship between vessel speed and fuel consumption in laden condition.
                    
                    - Each point represents a daily report
                    - The blue curve is a 2nd order polynomial fit to the data
                    - Points are color-coded by date (newer points are brighter)
                    
                    This curve can be used to estimate the optimal speed for fuel efficiency.
                    """)
                else:
                    st.info("No laden condition data available for the selected vessel and filters.")
            except Exception as e:
                st.error(f"Error creating Laden Condition chart: {str(e)}")
    
    def create_speed_consumption_chart(self, data, condition, chart_title):
        if not data:
            return None
        
        # Filter data for the specified loading condition
        condition_data = [row for row in data if row.get('loading_condition', '').lower() == condition.lower()]
        
        if not condition_data:
            return None
        
        # Extract speed and consumption data
        speeds = [row.get('speed', 0) for row in condition_data]
        consumptions = [row.get('normalised_consumption', 0) for row in condition_data]
        dates = [pd.to_datetime(row.get('report_date')) for row in condition_data]
        
        # Create a color gradient based on dates
        # Convert dates to numeric values for the colorscale
        date_nums = [(d - min(dates)).total_seconds() / 86400 for d in dates]  # Days since earliest date
        
        # Create the figure
        fig = go.Figure()
        
        # Add scatter plot with neon color gradient (markers only, no lines)
        fig.add_trace(go.Scatter(
            x=speeds,
            y=consumptions,
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
            name='Operational Data'
        ))
        
        # Calculate 2nd order polynomial fit if we have enough data points
        if len(speeds) > 2:
            # Fit a 2nd order polynomial
            coeffs = np.polyfit(speeds, consumptions, 2)
            poly = np.poly1d(coeffs)
            
            # Generate points for the curve
            x_smooth = np.linspace(min(speeds), max(speeds), 100)
            y_smooth = poly(x_smooth)
            
            # Add the polynomial curve
            fig.add_trace(go.Scatter(
                x=x_smooth,
                y=y_smooth,
                mode='lines',
                line=dict(color='#48cae4', width=3),  # Blue line
                name='Polynomial Fit (2nd Order)'
            ))
            
            # Add equation annotation
            equation = f"y = {coeffs[0]:.4f}x² + {coeffs[1]:.4f}x + {coeffs[2]:.4f}"
            fig.add_annotation(
                x=0.05,
                y=0.95,
                xref="paper",
                yref="paper",
                text=equation,
                showarrow=False,
                bgcolor="rgba(0,0,0,0.5)",
                bordercolor="#48cae4",
                borderwidth=1,
                borderpad=4,
                font=dict(color="#48cae4")
            )
        
        # Update layout
        fig.update_layout(
            title=chart_title,
            xaxis_title="Speed (knots)",
            yaxis_title="Fuel Consumption (mt/day)",
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
        
        return fig
