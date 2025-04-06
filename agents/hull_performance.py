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
        if not data:
            return None, None
        
        # Prepare data
        dates = [pd.to_datetime(row['report_date']) for row in data]
        metric_values = [row.get(metric_name, 0) for row in data]
        
        # Sort data for trend line
        sorted_indices = np.argsort(dates)
        dates_sorted = [dates[i] for i in sorted_indices]
        metric_values_sorted = [metric_values[i] for i in sorted_indices]
        
        # Create the figure with enhanced appearance
        fig = go.Figure()
        
        # Add 3D-like scatter plot with elevated appearance
        fig.add_trace(go.Scatter(
            x=dates,
            y=metric_values,
            mode='markers',
            marker=dict(
                size=12,
                color=metric_values,  # Color by value instead of date for better meaning
                colorscale='Turbo',   # More vibrant colorscale
                showscale=True,
                colorbar=dict(
                    title="Power Loss (%)",
                    titleside="right",
                    titlefont=dict(size=14)
                ),
                line=dict(width=2, color='rgba(255, 255, 255, 0.8)'),  # Add white border for 3D effect
                symbol='circle',
            ),
            name='Performance Data',
            hoverlabel=dict(
                bgcolor="white",
                font_size=14,
                font_family="Arial"
            ),
        ))
        
        # Calculate trend line
        latest_value = None
        if len(dates_sorted) > 1:
            x_numeric = np.array([(d - dates_sorted[0]).total_seconds() / 86400 for d in dates_sorted])
            y = np.array(metric_values_sorted)
            
            # Fit a polynomial instead of just linear for more interesting visual
            if len(dates_sorted) >= 6:
                # Higher order polynomial if we have enough points
                coeffs = np.polyfit(x_numeric, y, 2)  # 2nd order polynomial
                p = np.poly1d(coeffs)
                
                # Generate smooth curve
                x_line = np.linspace(0, x_numeric[-1], 100)
                y_line = p(x_line)
            else:
                # Linear if we don't have enough points
                coeffs = np.polyfit(x_numeric, y, 1)
                slope = coeffs[0]
                intercept = coeffs[1]
                x_line = np.array([0, x_numeric[-1]])
                y_line = slope * x_line + intercept
            
            # Convert back to datetime for plotting
            x_line_dates = [dates_sorted[0] + datetime.timedelta(days=float(x)) for x in x_line]
            
            # Add gradient-filled trend area with dynamic transparency
            fig.add_trace(go.Scatter(
                x=x_line_dates,
                y=y_line,
                mode='lines',
                line=dict(
                    color='rgba(255, 0, 110, 0.9)',
                    width=4,
                    shape='spline',  # Smoother line
                    dash='solid'
                ),
                name='Trend',
                fill='tozeroy',
                fillcolor='rgba(255, 0, 110, 0.2)'
            ))
            
            # Get the latest value from the trend line
            latest_value = y_line[-1] if len(y_line) > 0 else None
            
            # Add thresholds with better visualization
            if metric_name == 'hull_roughness_power_loss':
                # Add semi-transparent areas for different zones
                # Good zone (green, 0-15%)
                fig.add_traces([
                    go.Scatter(
                        x=[min(dates), max(dates)], 
                        y=[0, 0],
                        mode='lines',
                        line=dict(width=0),
                        showlegend=False
                    ),
                    go.Scatter(
                        x=[min(dates), max(dates)], 
                        y=[15, 15],
                        mode='lines',
                        line=dict(width=0),
                        fill='tonexty',
                        fillcolor='rgba(68, 214, 44, 0.15)',
                        name='Good Zone'
                    )
                ])
                
                # Average zone (yellow, 15-25%)
                fig.add_traces([
                    go.Scatter(
                        x=[min(dates), max(dates)], 
                        y=[15, 15],
                        mode='lines',
                        line=dict(width=0),
                        showlegend=False
                    ),
                    go.Scatter(
                        x=[min(dates), max(dates)], 
                        y=[25, 25],
                        mode='lines',
                        line=dict(width=0),
                        fill='tonexty',
                        fillcolor='rgba(255, 214, 0, 0.15)',
                        name='Average Zone'
                    )
                ])
                
                # Poor zone (red, >25%)
                fig.add_traces([
                    go.Scatter(
                        x=[min(dates), max(dates)], 
                        y=[25, 25],
                        mode='lines',
                        line=dict(width=0),
                        showlegend=False
                    ),
                    go.Scatter(
                        x=[min(dates), max(dates)], 
                        y=[max(metric_values) * 1.2 if metric_values else 40, max(metric_values) * 1.2 if metric_values else 40],
                        mode='lines',
                        line=dict(width=0),
                        fill='tonexty',
                        fillcolor='rgba(255, 0, 0, 0.15)',
                        name='Poor Zone'
                    )
                ])
                
                # Add threshold lines
                fig.add_shape(
                    type="line",
                    x0=min(dates),
                    y0=15,
                    x1=max(dates),
                    y1=15,
                    line=dict(color="rgba(255, 214, 0, 0.8)", width=2, dash="dot"),
                )
                
                fig.add_shape(
                    type="line",
                    x0=min(dates),
                    y0=25,
                    x1=max(dates),
                    y1=25,
                    line=dict(color="rgba(255, 0, 0, 0.8)", width=2, dash="dot"),
                )
                
                # Add more professional annotations
                fig.add_annotation(
                    x=min(dates),
                    y=15,
                    text="Good/Average Threshold (15%)",
                    showarrow=False,
                    yshift=10,
                    xshift=100,
                    font=dict(color="rgba(255, 214, 0, 0.8)", size=14),
                    bgcolor="rgba(255, 255, 255, 0.7)",
                    bordercolor="rgba(255, 214, 0, 0.8)",
                    borderwidth=1,
                    borderpad=4,
                    align="left"
                )
                
                fig.add_annotation(
                    x=min(dates),
                    y=25,
                    text="Average/Poor Threshold (25%)",
                    showarrow=False,
                    yshift=10,
                    xshift=100,
                    font=dict(color="rgba(255, 0, 0, 0.8)", size=14),
                    bgcolor="rgba(255, 255, 255, 0.7)",
                    bordercolor="rgba(255, 0, 0, 0.8)",
                    borderwidth=1,
                    borderpad=4,
                    align="left"
                )
            
            # Add latest value annotation with more emphasis
            if latest_value is not None:
                # Determine color based on value
                if metric_name == 'hull_roughness_power_loss':
                    if latest_value < 15:
                        color = "rgba(68, 214, 44, 1)"  # Green
                    elif latest_value < 25:
                        color = "rgba(255, 214, 0, 1)"  # Yellow
                    else:
                        color = "rgba(255, 0, 0, 1)"    # Red
                else:
                    color = "rgba(255, 0, 110, 1)"      # Pink
                    
                fig.add_annotation(
                    x=max(dates),
                    y=latest_value,
                    text=f"Latest: {latest_value:.2f}%",
                    showarrow=True,
                    arrowhead=2,
                    arrowcolor=color,
                    arrowsize=1.5,
                    arrowwidth=2.5,
                    ax=-60,
                    ay=-40,
                    font=dict(color=color, size=16, family="Arial", weight="bold"),
                    bgcolor="rgba(255, 255, 255, 0.9)",
                    bordercolor=color,
                    borderwidth=2,
                    borderpad=4
                )
        
        # Update layout for a more professional appearance
        fig.update_layout(
            title={
                'text': chart_title,
                'y':0.95,
                'x':0.5,
                'xanchor': 'center',
                'yanchor': 'top',
                'font': dict(size=24, family="Arial", color="#333333")
            },
            xaxis_title={
                'text': "Date",
                'font': dict(size=18, family="Arial", color="#333333")
            },
            yaxis_title={
                'text': y_axis_title,
                'font': dict(size=18, family="Arial", color="#333333")
            },
            paper_bgcolor='rgb(255, 255, 255)',
            plot_bgcolor='rgb(245, 245, 245)',
            height=700,
            margin=dict(l=50, r=50, t=80, b=50),
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="center",
                x=0.5,
                font=dict(
                    family="Arial",
                    size=14,
                    color="#333333"
                ),
                bgcolor="rgba(255, 255, 255, 0.7)",
                bordercolor="#666666",
                borderwidth=1
            )
        )
        
        # Update axes for better appearance
        fig.update_xaxes(
            showgrid=True,
            gridwidth=1,
            gridcolor='rgba(211, 211, 211, 0.5)',
            zeroline=False,
            showline=True,
            linewidth=2,
            linecolor='#333333',
            tickfont=dict(family="Arial", size=12, color="#333333"),
            tickangle=-30
        )
        
        fig.update_yaxes(
            showgrid=True,
            gridwidth=1,
            gridcolor='rgba(211, 211, 211, 0.5)',
            zeroline=False,
            showline=True,
            linewidth=2,
            linecolor='#333333',
            tickfont=dict(family="Arial", size=12, color="#333333")
        )
        
        return fig, latest_value
    
    def get_hull_condition(self, hull_roughness):
        if hull_roughness < 15:
            return "GOOD", "green"
        elif 15 <= hull_roughness < 25:
            return "AVERAGE", "orange"
        else:
            return "POOR", "red"
