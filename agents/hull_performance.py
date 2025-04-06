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
        Create a publication-quality hull performance chart with linear trend line using Altair
        """
        if not data:
            return None, None
        
        import altair as alt
        import pandas as pd
        import numpy as np
        from vega_datasets import data as vega_data
        import plotly.graph_objects as go
        import io
        import base64
        
        # Set Altair renderer
        alt.renderers.enable('default')
        
        # Prepare data
        df = pd.DataFrame({
            'date': [pd.to_datetime(row['report_date']) for row in data],
            'value': [row.get(metric_name, 0) for row in data]
        })
        
        # Sort by date
        df = df.sort_values('date')
        
        # Calculate linear trend
        latest_value = None
        if len(df) > 1:
            # For the regression, add a numeric date column
            df['date_num'] = (df['date'] - df['date'].min()).dt.total_seconds() / (24 * 3600)
            
            # Fit the linear model
            coeffs = np.polyfit(df['date_num'], df['value'], 1)
            slope = coeffs[0]
            intercept = coeffs[1]
            
            # Create trend data
            trend_data = pd.DataFrame({
                'date_num': [df['date_num'].min(), df['date_num'].max()],
                'date': [df['date'].min(), df['date'].max()],
                'trend': [
                    intercept + slope * df['date_num'].min(),
                    intercept + slope * df['date_num'].max()
                ]
            })
            
            # Get the latest value from the trend line
            latest_value = trend_data['trend'].iloc[-1]
            
            # Determine condition based on latest value
            if metric_name == 'hull_roughness_power_loss':
                if latest_value < 15:
                    condition = 'GOOD'
                elif latest_value < 25:
                    condition = 'AVERAGE'
                else:
                    condition = 'POOR'
            else:
                condition = ''
            
            # Add information to trend data
            trend_data['info'] = f"Latest: {latest_value:.2f}%\nCondition: {condition}\nSlope: {slope:.4f}% per day"
        
        # Create the base chart
        base = alt.Chart(df).encode(
            x=alt.X('date:T', 
                    title='Date',
                    axis=alt.Axis(labelAngle=-45, format='%b %Y'))
        )
        
        # Create background zones for thresholds if it's hull performance
        if metric_name == 'hull_roughness_power_loss':
            # Calculate the max y value for the chart
            y_max = max(df['value'].max() * 1.2, 40)
            
            # Create data for the zones
            zones_data = pd.DataFrame([
                {'zone': 'Good', 'start': 0, 'end': 15, 'color': '#4CAF50'},
                {'zone': 'Average', 'start': 15, 'end': 25, 'color': '#FFC107'},
                {'zone': 'Poor', 'start': 25, 'end': y_max, 'color': '#F44336'}
            ])
            
            # Create background colored zones
            zones = alt.Chart(zones_data).mark_rect(opacity=0.15).encode(
                y=alt.Y('start:Q', title='', scale=alt.Scale(domain=[0, y_max])),
                y2=alt.Y2('end:Q'),
                color=alt.Color('zone:N', 
                              scale=alt.Scale(domain=['Good', 'Average', 'Poor'],
                                             range=['#4CAF50', '#FFC107', '#F44336']),
                              legend=alt.Legend(title="Hull Condition"))
            ).properties(
                width=800,
                height=500
            )
            
            # Create threshold lines
            threshold_data = pd.DataFrame([
                {'threshold': 'Good/Average (15%)', 'value': 15, 'color': '#FFC107'},
                {'threshold': 'Average/Poor (25%)', 'value': 25, 'color': '#F44336'}
            ])
            
            thresholds = alt.Chart(threshold_data).mark_rule(strokeDash=[5, 5]).encode(
                y='value:Q',
                color=alt.Color('threshold:N', 
                              scale=alt.Scale(domain=['Good/Average (15%)', 'Average/Poor (25%)'],
                                             range=['#FFC107', '#F44336']),
                              legend=alt.Legend(title="Thresholds"))
            )
        
        # Create main scatter plot
        points = base.mark_circle(size=100).encode(
            y=alt.Y('value:Q', 
                    title=y_axis_title,
                    scale=alt.Scale(domain=[0, max(df['value'].max() * 1.2, 40) if metric_name == 'hull_roughness_power_loss' else None])),
            color=alt.Color('value:Q', 
                          scale=alt.Scale(scheme='viridis'),
                          legend=alt.Legend(title="Power Loss (%)")),
            tooltip=['date:T', alt.Tooltip('value:Q', title='Power Loss', format='.2f')]
        )
        
        # Create trend line if we have enough data
        if len(df) > 1:
            trend_line = alt.Chart(trend_data).mark_line(
                color='#FF1493',
                strokeWidth=3
            ).encode(
                x='date:T',
                y='trend:Q'
            )
            
            # Text annotation for the latest value
            text = alt.Chart(trend_data.iloc[[-1]]).mark_text(
                align='left',
                baseline='middle',
                dx=30,
                fontSize=16,
                fontWeight='bold'
            ).encode(
                x='date:T',
                y='trend:Q',
                text='info:N'
            )
        
        # Combine all chart elements
        if metric_name == 'hull_roughness_power_loss':
            # Start with zones in the background
            chart = zones
            
            # Add threshold lines
            chart += thresholds
            
            # Add data points
            chart += points
            
            # Add trend line and annotation if available
            if len(df) > 1:
                chart += trend_line
                chart += text
        else:
            # Start with just the points
            chart = points
            
            # Add trend line and annotation if available
            if len(df) > 1:
                chart += trend_line
                chart += text
        
        # Configure the chart
        chart = chart.properties(
            title=chart_title,
            width=800,
            height=500
        ).configure_axis(
            grid=True,
            gridColor='#DDDDDD',
            titleFontSize=14,
            titleFontWeight='bold',
            labelFontSize=12
        ).configure_title(
            fontSize=18,
            fontWeight='bold'
        ).configure_legend(
            titleFontSize=12,
            labelFontSize=12
        ).configure_view(
            strokeWidth=1,
            stroke='#DDDDDD'
        )
        
        # Convert Altair chart to HTML
        html = chart.to_html()
        
        # Create a Plotly figure wrapper for compatibility
        plotly_fig = go.Figure()
        
        # Add the HTML as an iframe
        plotly_fig.add_layout_image(
            dict(
                source=f"data:text/html;base64,{base64.b64encode(html.encode()).decode()}",
                xref="paper",
                yref="paper",
                x=0,
                y=1,
                sizex=1,
                sizey=1,
                sizing="stretch",
                layer="below"
            )
        )
        
        # Set layout to match the Altair chart dimensions
        plotly_fig.update_layout(
            autosize=False,
            width=900,
            height=600,
            margin=dict(l=0, r=0, t=0, b=0)
        )
        
        return plotly_fig, latest_value 
    def get_hull_condition(self, hull_roughness):
        if hull_roughness < 15:
            return "GOOD", "green"
        elif 15 <= hull_roughness < 25:
            return "AVERAGE", "orange"
        else:
            return "POOR", "red"
