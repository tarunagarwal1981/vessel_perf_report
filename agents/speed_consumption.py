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
        """
        Create a publication-quality speed consumption chart with quadratic fit
        using pure Matplotlib for maximum compatibility
        """
        if not data:
            return None
        
        import matplotlib.pyplot as plt
        import pandas as pd
        import numpy as np
        
        # Filter data for the specified loading condition
        condition_data = [row for row in data if row.get('loading_condition', '').lower() == condition.lower()]
        
        if not condition_data:
            return None
        
        # Extract speed and consumption data
        speeds = [row.get('speed', 0) for row in condition_data]
        consumptions = [row.get('normalised_consumption', 0) for row in condition_data]
        dates = [pd.to_datetime(row.get('report_date')) for row in condition_data]
        
        # Create a simple figure
        plt.figure(figsize=(10, 6))
        
        # Create scatter plot 
        plt.scatter(speeds, consumptions, s=60, c='blue', alpha=0.7, label='Operational Data')
        
        # Calculate quadratic fit if we have enough data points
        if len(speeds) > 2:
            # Sort by speed for better curve rendering
            sorted_indices = np.argsort(speeds)
            speeds_sorted = [speeds[i] for i in sorted_indices]
            consumptions_sorted = [consumptions[i] for i in sorted_indices]
            
            # Fit a 2nd order polynomial (quadratic)
            coeffs = np.polyfit(speeds, consumptions, 2)
            
            # Generate points for smooth curve
            x_smooth = np.linspace(min(speeds), max(speeds), 100)
            y_smooth = coeffs[0] * x_smooth**2 + coeffs[1] * x_smooth + coeffs[2]
            
            # Add the fitted curve
            plt.plot(x_smooth, y_smooth, 'r-', linewidth=2, 
                    label=f'Quadratic Fit')
            
            # Add equation text
            equation = f"Consumption = {coeffs[0]:.4f}×Speed² + {coeffs[1]:.4f}×Speed + {coeffs[2]:.4f}"
            plt.text(0.05, 0.95, equation, transform=plt.gca().transAxes,
                    fontsize=10, verticalalignment='top', 
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
            
            # Find optimal efficiency point (minimum consumption/speed ratio)
            try:
                # Calculate consumption per unit speed
                efficiency = y_smooth / x_smooth
                optimal_idx = np.argmin(efficiency)
                optimal_speed = x_smooth[optimal_idx]
                optimal_consumption = y_smooth[optimal_idx]
                
                # Mark optimal point with a star
                plt.plot(optimal_speed, optimal_consumption, 'r*', markersize=15, 
                        label='Optimal Efficiency')
                
                # Add annotation for optimal point
                plt.annotate(
                    f'Optimal Speed: {optimal_speed:.2f} knots\nConsumption: {optimal_consumption:.2f} mt/day',
                    xy=(optimal_speed, optimal_consumption),
                    xytext=(20, -30),
                    textcoords='offset points',
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
                    fontsize=10,
                    arrowprops=dict(arrowstyle='->')
                )
            except Exception as e:
                print(f"Error finding optimal point: {str(e)}")
        
        # Basic styling
        plt.xlabel('Speed (knots)')
        plt.ylabel('Fuel Consumption (mt/day)')
        plt.title(chart_title)
        
        # Add a simple grid
        plt.grid(True, linestyle='--', alpha=0.5)
        
        # Set axis to start at 0
        plt.xlim(left=0)
        plt.ylim(bottom=0)
        
        # Add a legend
        plt.legend()
        
        # Tight layout
        plt.tight_layout()
        
        # Return the figure
        return plt.gcf()
