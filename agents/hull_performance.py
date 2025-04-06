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
        Create a professional hull performance chart with linear trend line
        """
        if not data:
            return None, None
        
        import matplotlib.pyplot as plt
        import seaborn as sns
        import pandas as pd
        import numpy as np
        from matplotlib.ticker import MaxNLocator
        import plotly.graph_objects as go
        
        # Prepare data - ensure dates are in datetime format
        dates = [pd.to_datetime(row['report_date']) for row in data]
        metric_values = [row.get(metric_name, 0) for row in data]
        
        # Sort data for proper trend line
        sorted_indices = np.argsort(dates)
        dates_sorted = [dates[i] for i in sorted_indices]
        metric_values_sorted = [metric_values[i] for i in sorted_indices]
        
        # Create a Plotly figure (maintaining compatibility with existing code)
        fig = go.Figure()
        
        # Add scatter points
        fig.add_trace(go.Scatter(
            x=dates,
            y=metric_values,
            mode='markers',
            marker=dict(
                size=12,
                color=metric_values,
                colorscale='Viridis',
                showscale=True,
                colorbar=dict(
                    title=dict(
                        text="Power Loss (%)",
                    )
                )
            ),
            name='Performance Data'
        ))
        
        # Calculate linear trend (explicitly linear as requested)
        latest_value = None
        if len(dates_sorted) > 1:
            # Convert dates to numeric for fitting
            x_numeric = np.array([(d - dates_sorted[0]).total_seconds() / 86400 for d in dates_sorted])
            y = np.array(metric_values_sorted)
            
            # Calculate linear regression
            coeffs = np.polyfit(x_numeric, y, 1)
            slope = coeffs[0]
            intercept = coeffs[1]
            
            # Generate points for the trend line
            x_line = np.array([0, x_numeric[-1]])
            y_line = slope * x_line + intercept
            
            # Convert back to datetime for plotting
            x_line_dates = [dates_sorted[0] + pd.Timedelta(days=float(x)) for x in x_line]
            
            # Add the trend line
            fig.add_trace(go.Scatter(
                x=x_line_dates,
                y=y_line,
                mode='lines',
                line=dict(color='#FF0066', width=3),
                name=f'Trend Line (Slope: {slope:.4f} % per day)',
                fill='tozeroy',
                fillcolor='rgba(255, 0, 102, 0.1)'
            ))
            
            # Save the latest value for reference
            latest_value = y_line[-1]
        
        # Add thresholds for hull roughness power loss
        if metric_name == 'hull_roughness_power_loss':
            # Add good zone (0-15%)
            fig.add_shape(
                type="rect",
                x0=min(dates),
                x1=max(dates),
                y0=0,
                y1=15,
                fillcolor="rgba(68, 214, 44, 0.15)",
                line_width=0,
                name="Good Zone"
            )
            
            # Add average zone (15-25%)
            fig.add_shape(
                type="rect",
                x0=min(dates),
                x1=max(dates),
                y0=15,
                y1=25,
                fillcolor="rgba(255, 214, 0, 0.15)",
                line_width=0,
                name="Average Zone"
            )
            
            # Add poor zone (>25%)
            fig.add_shape(
                type="rect",
                x0=min(dates),
                x1=max(dates),
                y0=25,
                y1=max(metric_values) * 1.2 if metric_values else 40,
                fillcolor="rgba(255, 0, 0, 0.15)",
                line_width=0,
                name="Poor Zone"
            )
            
            # Add threshold lines
            fig.add_shape(
                type="line",
                x0=min(dates),
                y0=15,
                x1=max(dates),
                y1=15,
                line=dict(color="rgba(255, 214, 0, 0.8)", width=2, dash="dash"),
            )
            
            fig.add_shape(
                type="line",
                x0=min(dates),
                y0=25,
                x1=max(dates),
                y1=25,
                line=dict(color="rgba(255, 0, 0, 0.8)", width=2, dash="dash"),
            )
            
            # Add threshold annotations
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
        
        # Add latest value annotation
        if latest_value is not None:
            # Determine color based on value
            if metric_name == 'hull_roughness_power_loss':
                if latest_value < 15:
                    color = "rgba(68, 214, 44, 1)"  # Green
                    condition = 'GOOD'
                elif latest_value < 25:
                    color = "rgba(255, 214, 0, 1)"  # Yellow
                    condition = 'AVERAGE'
                else:
                    color = "rgba(255, 0, 0, 1)"    # Red
                    condition = 'POOR'
            else:
                color = "rgba(255, 0, 110, 1)"      # Pink
                condition = ''
                
            fig.add_annotation(
                x=max(dates),
                y=latest_value,
                text=f"Latest: {latest_value:.2f}%<br>Condition: {condition}",
                showarrow=True,
                arrowhead=2,
                arrowcolor=color,
                arrowsize=1.5,
                arrowwidth=2.5,
                ax=-60,
                ay=-40,
                font=dict(color=color, size=16, weight="bold"),
                bgcolor="rgba(255, 255, 255, 0.9)",
                bordercolor=color,
                borderwidth=2,
                borderpad=4
            )
        
        # Update layout
        fig.update_layout(
            title={
                'text': chart_title,
                'y':0.95,
                'x':0.5,
                'xanchor': 'center',
                'yanchor': 'top'
            },
            xaxis_title="Date",
            yaxis_title=y_axis_title,
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
                bgcolor="rgba(255, 255, 255, 0.7)",
                bordercolor="#666666",
                borderwidth=1
            )
        )
        
        # Update axes
        fig.update_xaxes(
            showgrid=True,
            gridwidth=1,
            gridcolor='rgba(211, 211, 211, 0.5)',
            zeroline=False,
            showline=True,
            linewidth=2,
            linecolor='#333333',
            tickangle=-30
        )
        
        fig.update_yaxes(
            showgrid=True,
            gridwidth=1,
            gridcolor='rgba(211, 211, 211, 0.5)',
            zeroline=False,
            showline=True,
            linewidth=2,
            linecolor='#333333'
        )
        
        return fig, latest_value
    
    def get_hull_condition(self, hull_roughness):
        if hull_roughness < 15:
            return "GOOD", "green"
        elif 15 <= hull_roughness < 25:
            return "AVERAGE", "orange"
        else:
            return "POOR", "red"
