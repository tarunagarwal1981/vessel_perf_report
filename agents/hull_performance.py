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
        if not data:
            return None, None
        
        # Prepare data - ensure dates are in datetime format
        dates = [pd.to_datetime(row['report_date']) for row in data]
        metric_values = [row.get(metric_name, 0) for row in data]
        
        # Sort data by date for proper trend line
        df = pd.DataFrame({'date': dates, 'value': metric_values})
        df = df.sort_values('date')
        dates_sorted = df['date'].tolist()
        metric_values_sorted = df['value'].tolist()
        
        # Create a color gradient based on time progression
        time_progress = [(d - min(dates_sorted)).total_seconds() / 86400 for d in dates_sorted]  # Days since first date
        normalized_progress = [t / max(time_progress) * 100 if max(time_progress) > 0 else 50 for t in time_progress]
        
        # Create the figure with dark theme
        fig = go.Figure()
        
        # Add background colored zones with defined boundaries
        fig.add_shape(type="rect", x0=min(dates_sorted), x1=max(dates_sorted), y0=0, y1=15,
                     fillcolor="rgba(0, 100, 0, 0.15)", line=dict(width=0), layer="below")
        fig.add_shape(type="rect", x0=min(dates_sorted), x1=max(dates_sorted), y0=15, y1=25,
                     fillcolor="rgba(255, 165, 0, 0.15)", line=dict(width=0), layer="below")
        fig.add_shape(type="rect", x0=min(dates_sorted), x1=max(dates_sorted), y0=25, y1=max(metric_values_sorted) * 1.1,
                     fillcolor="rgba(139, 0, 0, 0.15)", line=dict(width=0), layer="below")
        
        # Add threshold lines
        fig.add_shape(type="line", x0=min(dates_sorted), x1=max(dates_sorted), y0=15, y1=15,
                     line=dict(color="#00ff00", width=1.5))
        fig.add_shape(type="line", x0=min(dates_sorted), x1=max(dates_sorted), y0=25, y1=25,
                     line=dict(color="#ff0000", width=1.5))
        
        # Add scatter plot with dots colored by time progression
        fig.add_trace(go.Scatter(
            x=dates_sorted,
            y=metric_values_sorted,
            mode='markers',
            marker=dict(
                size=10,
                color=normalized_progress,
                colorscale='Viridis',  # This gives a nice blue->green->yellow gradient
                showscale=True,
                colorbar=dict(
                    title="Time Progression",
                    tickvals=[0, 50, 100],
                    ticktext=["Early", "Mid", "Recent"],
                    thickness=12,
                    len=0.6,
                    y=0.6,
                    title_side='right'
                ),
                line=dict(width=1, color='white')
            ),
            name='Performance Data'
        ))
        
        # Calculate linear best fit if we have enough data points
        latest_value = None
        if len(dates_sorted) > 1:
            # Convert dates to numeric for fitting
            x_numeric = np.array([(d - dates_sorted[0]).total_seconds() / 86400 for d in dates_sorted])
            y = np.array(metric_values_sorted)
            
            # Fit a line
            coeffs = np.polyfit(x_numeric, y, 1)
            slope = coeffs[0]
            intercept = coeffs[1]
            
            # Generate points for the trend line
            x_line = np.array([0, x_numeric[-1]])
            y_line = slope * x_line + intercept
            
            # Convert back to datetime for plotting
            x_line_dates = [dates_sorted[0] + datetime.timedelta(days=float(x)) for x in x_line]
            
            # Add the best fit line
            fig.add_trace(go.Scatter(
                x=x_line_dates,
                y=y_line,
                mode='lines',
                line=dict(color='white', width=2),
                name='Trend Line'
            ))
            
            # Get the latest value from the trend line
            latest_value = y_line[-1]
        
        # Add custom legend for zone colors at the top
        fig.add_trace(go.Scatter(
            x=[None], y=[None], mode='markers', marker=dict(size=0),
            name="Good Zone", line=dict(color="#00ff00", width=4)
        ))
        fig.add_trace(go.Scatter(
            x=[None], y=[None], mode='markers', marker=dict(size=0),
            name="Average Zone", line=dict(color="#ffa500", width=4)
        ))
        fig.add_trace(go.Scatter(
            x=[None], y=[None], mode='markers', marker=dict(size=0),
            name="Poor Zone", line=dict(color="#ff0000", width=4)
        ))
        
        # Update layout for a clean, crisp dark theme
        fig.update_layout(
            title=dict(
                text=chart_title,
                font=dict(size=22, color='white', family="Arial, sans-serif"),
                x=0.01,
                y=0.97,
                xanchor='left',
                yanchor='top'
            ),
            xaxis=dict(
                title="Date",
                showgrid=True,
                gridcolor="rgba(255, 255, 255, 0.1)",
                tickangle=-45,
                tickfont=dict(size=10, family="Arial, sans-serif"),
                zeroline=False,
                title_font=dict(size=12, family="Arial, sans-serif"),
                tickformat="%b-%Y"
            ),
            yaxis=dict(
                title=y_axis_title,
                showgrid=True,
                gridcolor="rgba(255, 255, 255, 0.1)",
                range=[0, max(max(metric_values_sorted) * 1.1, 25 * 1.5)],
                zeroline=False,
                title_font=dict(size=12, family="Arial, sans-serif"),
                tickfont=dict(size=10, family="Arial, sans-serif")
            ),
            plot_bgcolor="rgb(0, 0, 0)",
            paper_bgcolor="rgb(0, 0, 0)",
            font=dict(color="white", family="Arial, sans-serif"),
            height=550,
            margin=dict(t=80, b=60, l=80, r=40),
            legend=dict(
                orientation="h",
                yanchor="top",
                y=1.14,
                xanchor="right",
                x=1,
                bgcolor="rgba(0,0,0,0)",
                font=dict(size=10)
            ),
            hovermode="closest",
            hoverlabel=dict(
                bgcolor="rgba(0,0,0,0.8)",
                font_size=12,
                font_family="Arial, sans-serif"
            )
        )
        
        # Add hover template for better information display
        fig.update_traces(
            hovertemplate="<b>Date</b>: %{x|%d %b %Y}<br><b>" + y_axis_title + "</b>: %{y:.2f}%<br>",
            selector=dict(name='Performance Data')
        )
        
        # Custom axis and grid adjustments for a cleaner look
        fig.update_xaxes(
            mirror=False,
            ticks='outside',
            showline=True,
            linecolor='rgba(255, 255, 255, 0.3)',
            nticks=8
        )
        fig.update_yaxes(
            mirror=False,
            ticks='outside',
            showline=True,
            linecolor='rgba(255, 255, 255, 0.3)',
            nticks=10
        )
        
        return fig, latest_value
    def get_hull_condition(self, hull_roughness):
        if hull_roughness < 15:
            return "GOOD", "green"
        elif 15 <= hull_roughness < 25:
            return "AVERAGE", "orange"
        else:
            return "POOR", "red"
