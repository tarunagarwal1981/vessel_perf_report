import streamlit as st
import pandas as pd
import numpy as np
import datetime
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import matplotlib.dates as mdates
from matplotlib.gridspec import GridSpec
import matplotlib.patheffects as path_effects
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.collections import LineCollection

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
        """Create chart using Altair which has first-class support in Streamlit"""
        if not data:
            return None, None
        
        import altair as alt
        import pandas as pd
        import numpy as np
        
        # Prepare data
        df = pd.DataFrame({
            'date': [pd.to_datetime(row['report_date']) for row in data],
            'value': [row.get(metric_name, 0) for row in data]
        })
        
        # Sort by date
        df = df.sort_values('date')
        
        # Calculate trend
        if len(df) > 1:
            # Add day number column for regression
            df['day'] = (df['date'] - df['date'].min()).dt.total_seconds() / (24 * 3600)
            
            # Calculate trend line
            x = df['day']
            y = df['value']
            coeffs = np.polyfit(x, y, 1)
            
            # Generate trend line points
            df_trend = pd.DataFrame({
                'date': [df['date'].min(), df['date'].max()],
                'value': [
                    coeffs[1] + coeffs[0] * df['day'].min(),
                    coeffs[1] + coeffs[0] * df['day'].max()
                ],
                'type': 'Trend Line'
            })
            
            # Get latest value from trend
            latest_value = df_trend['value'].iloc[-1]
        else:
            latest_value = df['value'].iloc[-1] if not df.empty else None
        
        # Create base chart
        base = alt.Chart(df).encode(
            x=alt.X('date:T', title='Date', axis=alt.Axis(format='%b %Y', labelAngle=-45))
        )
        
        # Create the main chart with points
        points = base.mark_circle(size=100).encode(
            y=alt.Y('value:Q', title=y_axis_title),
            color=alt.Color('date:T', 
                           scale=alt.Scale(scheme='viridis'),
                           legend=alt.Legend(title='Time Progression')),
            tooltip=['date:T', alt.Tooltip('value:Q', title='Value', format='.2f')]
        )
        
        if len(df) > 1:
            # Add trend line
            trend = alt.Chart(df_trend).mark_line(color='red', strokeWidth=3).encode(
                x='date:T',
                y='value:Q'
            )
            
            # Combine points and trend
            chart = points + trend
        else:
            chart = points
        
        # Apply styling
        chart = chart.properties(
            title=chart_title,
            width=800,
            height=500
        ).configure_view(
            stroke='lightgray'
        )
        
        # Return the chart and latest value
        # For Streamlit, we return the Altair chart directly
        # For report generator, we'll need to handle conversion separately
        return chart, latest_value
    def get_hull_condition(self, hull_roughness):
        if hull_roughness < 15:
            return "GOOD", "green"
        elif 15 <= hull_roughness < 25:
            return "AVERAGE", "orange"
        else:
            return "POOR", "red"
