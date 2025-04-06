import streamlit as st
import pandas as pd
import numpy as np
import datetime
import plotly.graph_objects as go

class HullPerformanceAgent:
    def __init__(self):
        pass
    
    def run(self, data, vessel_name):
        # Ensure all entries have the necessary fields for hull performance
        filtered_data_power = []
        filtered_data_speed = []
        
        for entry in data:
            if 'report_date' in entry and 'hull_roughness_power_loss' in entry:
                if entry['hull_roughness_power_loss'] is not None:
                    filtered_data_power.append(entry)
            
            if 'report_date' in entry and 'hull_roughness_speed_loss' in entry:
                if entry['hull_roughness_speed_loss'] is not None:
                    filtered_data_speed.append(entry)
        
        if not filtered_data_power and not filtered_data_speed:
            st.warning("No valid hull performance data found. Please check the Speed-Consumption tab.")
            return
        
        # Create tabs for different hull performance charts
        hull_tab1, hull_tab2 = st.tabs(["Excess Power Chart", "Speed Loss Chart"])
        
        # Excess Power Tab
        with hull_tab1:
            if filtered_data_power:
                # Sort data by date
                filtered_data_power.sort(key=lambda x: pd.to_datetime(x['report_date']))
                
                # Display hull roughness power loss chart
                st.subheader("Hull Roughness - Excess Power")
                
                try:
                    chart, latest_power_loss = self.create_performance_chart(
                        filtered_data_power, 
                        'hull_roughness_power_loss',
                        "Hull Roughness - Excess Power Trend",
                        "Excess Power (%)"
                    )
                    
                    if chart:
                        st.plotly_chart(chart, use_container_width=True)
                        
                        # Create two columns for metrics
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            # Use the trend line's last value instead of the raw data point
                            if latest_power_loss is not None:
                                condition, color = self.get_hull_condition(latest_power_loss)
                                st.metric("Current Excess Power (from trend)", f"{latest_power_loss:.2f}%")
                                st.markdown(f"<h3 style='color:{color}'>Hull Condition: {condition}</h3>", unsafe_allow_html=True)
                        
                        with col2:
                            # Calculate any trends or recommendations
                            if latest_power_loss is not None:
                                if latest_power_loss < 15:
                                    recommendation = "No action required"
                                elif 15 <= latest_power_loss < 25:
                                    recommendation = "Consider hull cleaning at next convenient opportunity"
                                else:
                                    recommendation = "Hull cleaning recommended as soon as possible"
                                
                                st.markdown(f"### Recommendation\n{recommendation}")
                except Exception as e:
                    st.error(f"Error creating Excess Power chart: {str(e)}")
            else:
                st.info("No valid hull roughness power loss data available for the selected filters.")
        
        # Speed Loss Tab
        with hull_tab2:
            if filtered_data_speed:
                # Sort data by date
                filtered_data_speed.sort(key=lambda x: pd.to_datetime(x['report_date']))
                
                # Display hull roughness speed loss chart
                st.subheader("Hull Roughness - Speed Loss")
                
                try:
                    chart, latest_speed_loss = self.create_performance_chart(
                        filtered_data_speed, 
                        'hull_roughness_speed_loss',
                        "Hull Roughness - Speed Loss Trend",
                        "Speed Loss (%)"
                    )
                    
                    if chart:
                        st.plotly_chart(chart, use_container_width=True)
                        
                        # Show the latest value from trend
                        if latest_speed_loss is not None:
                            st.metric("Current Speed Loss (from trend)", f"{latest_speed_loss:.2f}%")
                            
                            # Calculate impact on vessel performance
                            st.markdown("### Performance Impact")
                            st.markdown(f"""
                            The current speed loss of {latest_speed_loss:.2f}% means that:
                            
                            - At the same power, the vessel's speed is reduced by {latest_speed_loss:.2f}%
                            - To maintain the same speed, the vessel needs to increase power
                            """)
                except Exception as e:
                    st.error(f"Error creating Speed Loss chart: {str(e)}")
            else:
                st.info("No valid hull roughness speed loss data available for the selected filters.")
    
    def create_performance_chart(self, data, metric_name, chart_title, y_axis_title):
        """
        Create a publication-quality hull performance chart with linear trend line
        using pure Matplotlib for maximum compatibility
        """
        if not data:
            return None, None
        
        import matplotlib.pyplot as plt
        import matplotlib.dates as mdates
        import pandas as pd
        import numpy as np
        import io
        import base64
        
        # Prepare data
        dates = [pd.to_datetime(row['report_date']) for row in data]
        metric_values = [row.get(metric_name, 0) for row in data]
        
        # Sort data for proper trend line
        sorted_indices = np.argsort(dates)
        dates_sorted = [dates[i] for i in sorted_indices]
        metric_values_sorted = [metric_values[i] for i in sorted_indices]
        
        # Create a plain figure with simple styling
        plt.figure(figsize=(10, 6))
        
        # Add colored background zones for hull roughness power loss
        if metric_name == 'hull_roughness_power_loss':
            y_max = max(max(metric_values) * 1.2, 40)
            
            # Good zone (0-15%)
            plt.axhspan(0, 15, color='lightgreen', alpha=0.3, label='Good Condition')
            
            # Average zone (15-25%)
            plt.axhspan(15, 25, color='lightyellow', alpha=0.3, label='Average Condition')
            
            # Poor zone (>25%)
            plt.axhspan(25, y_max, color='lightcoral', alpha=0.3, label='Poor Condition')
            
            # Add threshold lines
            plt.axhline(y=15, color='orange', linestyle='--', alpha=0.7, linewidth=1.5, 
                       label='Good/Average Threshold (15%)')
            plt.axhline(y=25, color='red', linestyle='--', alpha=0.7, linewidth=1.5, 
                       label='Average/Poor Threshold (25%)')
        
        # Create scatter plot with plain colored points
        plt.scatter(dates, metric_values, s=60, c='blue', alpha=0.7, label='Performance Data')
        
        # Calculate linear trend
        latest_value = None
        if len(dates_sorted) > 1:
            # Convert dates to numbers for linear regression
            x_numeric = mdates.date2num(dates_sorted)
            y = np.array(metric_values_sorted)
            
            # Calculate linear regression
            coeffs = np.polyfit(x_numeric, y, 1)
            slope = coeffs[0]
            intercept = coeffs[1]
            
            # Create line points
            x_line = np.array([min(x_numeric), max(x_numeric)])
            y_line = slope * x_line + intercept
            
            # Plot trend line
            plt.plot(mdates.num2date(x_line), y_line, 'r-', linewidth=2, 
                    label=f'Trend Line (Slope: {slope:.4f}% per day)')
            
            # Get the latest value from the trend line
            latest_value = y_line[-1]
            
            # Add annotation for latest value
            if latest_value is not None:
                # Determine condition based on value
                if metric_name == 'hull_roughness_power_loss':
                    if latest_value < 15:
                        condition = 'GOOD'
                    elif latest_value < 25:
                        condition = 'AVERAGE'
                    else:
                        condition = 'POOR'
                else:
                    condition = ''
                
                # Simple annotation without fancy styling
                plt.annotate(
                    f'Latest: {latest_value:.2f}%\nCondition: {condition}',
                    xy=(dates_sorted[-1], latest_value),
                    xytext=(20, 10),
                    textcoords='offset points',
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
                    fontsize=10
                )
        
        # Basic styling
        plt.xlabel('Date')
        plt.ylabel(y_axis_title)
        plt.title(chart_title)
        
        # Format x-axis dates with no overlapping
        plt.gcf().autofmt_xdate()
        
        # Add a simple grid
        plt.grid(True, linestyle='--', alpha=0.5)
        
        # Set y-axis to start at 0
        plt.ylim(bottom=0)
        
        # Add a legend
        plt.legend()
        
        # Tight layout
        plt.tight_layout()
        
        # For compatibility, return both the Matplotlib figure and the latest value
        # Your report generator will need to handle the Matplotlib figure directly
        fig = plt.gcf()  # Get current figure
        
        # Return the figure and latest value
        return fig, latest_value
    def get_hull_condition(self, hull_roughness):
        if hull_roughness < 15:
            return "GOOD", "green"
        elif 15 <= hull_roughness < 25:
            return "AVERAGE", "orange"
        else:
            return "POOR", "red"
