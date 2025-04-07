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
        Create a publication-quality hull performance chart with interactive elements
        using Plotly for superior visualization quality
        """
        if not data:
            return None, None

        # Prepare data
        df = pd.DataFrame(data)
        df['report_date'] = pd.to_datetime(df['report_date'])
        df = df.sort_values('report_date')
        
        # Create figure
        fig = go.Figure()

        # Add shaded regions for power loss only
        if metric_name == 'hull_roughness_power_loss':
            # Add colored background zones
            fig.add_shape(type="rect", x0=df['report_date'].min(), x1=df['report_date'].max(),
                          y0=0, y1=15, fillcolor="green", opacity=0.1, layer="below",
                          line=dict(width=0))
            fig.add_shape(type="rect", x0=df['report_date'].min(), x1=df['report_date'].max(),
                          y0=15, y1=25, fillcolor="orange", opacity=0.1, layer="below",
                          line=dict(width=0))
            fig.add_shape(type="rect", x0=df['report_date'].min(), x1=df['report_date'].max(),
                          y0=25, y1=df[metric_name].max()*1.2, fillcolor="red", opacity=0.1,
                          layer="below", line=dict(width=0))

        # Add raw data points
        fig.add_trace(go.Scatter(
            x=df['report_date'],
            y=df[metric_name],
            mode='markers+lines',
            name='Measured Data',
            line=dict(color='#1f77b4', width=2, dash='dot'),
            marker=dict(
                size=8,
                color='#1f77b4',
                line=dict(width=1, color='DarkSlateGrey')
            )     
        ))

        # Add trend line if enough data points
        latest_value = None
        if len(df) > 1:
            # Calculate linear trend
            x_numeric = df['report_date'].astype(np.int64) // 10**9  # Convert to seconds
            coeffs = np.polyfit(x_numeric, df[metric_name], 1)
            trend_line = np.poly1d(coeffs)(x_numeric)
            
            fig.add_trace(go.Scatter(
                x=df['report_date'],
                y=trend_line,
                mode='lines',
                name='Trend Line',
                line=dict(color='firebrick', width=3),
                hoverinfo='skip'
            ))

            # Calculate latest trend value
            latest_value = trend_line[-1]

            # Add trend annotation
            fig.add_annotation(
                x=df['report_date'].iloc[-1],
                y=latest_value,
                text=f"Current Trend: {latest_value:.1f}%",
                showarrow=True,
                arrowhead=4,
                ax=-50,
                ay=-40,
                font=dict(size=12, color='firebrick'),
                bordercolor='firebrick',
                borderwidth=1,
                borderpad=4,
                bgcolor='white'
            )

        # Add condition zones if power loss
        if metric_name == 'hull_roughness_power_loss':
            fig.add_annotation(
                x=0.5, y=0.95,
                xref='paper', yref='paper',
                text="Condition Zones:<br><span style='color:green'>Good</span> | "
                     "<span style='color:orange'>Average</span> | <span style='color:red'>Poor</span>",
                showarrow=False,
                font=dict(size=12),
                bordercolor='black',
                bgcolor='white'
            )

        # Update layout for publication quality
        fig.update_layout(
            title=dict(
                text=chart_title,
                font=dict(size=24, family='Arial', color='#2a2a2a'),
                x=0.03,
                y=0.93
            ),
            xaxis=dict(
                title=dict(
                    text='Date',
                    font=dict(size=18, family='Arial', color='#2a2a2a')
                ),
                tickfont=dict(size=14, color='#2a2a2a'),
                gridcolor='lightgrey',
                showgrid=True,
                linecolor='#2a2a2a',
                mirror=True
            ),
            yaxis=dict(
                title=dict(
                    text=y_axis_title,
                    font=dict(size=18, family='Arial', color='#2a2a2a')
                ),
                tickfont=dict(size=14, color='#2a2a2a'),
                gridcolor='lightgrey',
                showgrid=True,
                zeroline=False,
                linecolor='#2a2a2a',
                mirror=True,
                range=[0, max(df[metric_name].max() * 1.2, 40)]
            ),
            legend=dict(
                orientation='h',
                yanchor='bottom',
                y=1.02,
                xanchor='right',
                x=1,
                font=dict(size=14)
            ),
            plot_bgcolor='white',
            margin=dict(l=50, r=30, t=80, b=50),
            hovermode='x unified',
            hoverlabel=dict(
                bgcolor='white',
                font_size=14,
                font_family='Arial'
            )
        )

        # Add custom grid lines
        fig.update_yaxes(showline=True, linewidth=2, linecolor='gray')
        fig.update_xaxes(showline=True, linewidth=2, linecolor='gray')

        return fig, latest_value
    def get_hull_condition(self, hull_roughness):
        if hull_roughness < 15:
            return "GOOD", "green"
        elif 15 <= hull_roughness < 25:
            return "AVERAGE", "orange"
        else:
            return "POOR", "red"
