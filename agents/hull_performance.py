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
        Create a professional hull performance chart with linear trend line using Matplotlib
        Returns both the matplotlib figure and the latest value
        """
        if not data:
            return None, None
        
        import matplotlib.pyplot as plt
        import matplotlib.dates as mdates
        import pandas as pd
        import numpy as np
        from io import BytesIO
        import plotly.graph_objects as go
        
        # Prepare data
        df = pd.DataFrame({
            'date': [pd.to_datetime(row['report_date']) for row in data],
            'value': [row.get(metric_name, 0) for row in data]
        })
        
        # Sort by date
        df = df.sort_values('date')
        
        # Create a high-quality matplotlib figure
        plt.figure(figsize=(12, 8), dpi=100)
        
        # Set the style
        plt.style.use('ggplot')
        
        # Create color gradient based on values for scatter points
        scatter = plt.scatter(df['date'], df['value'], 
                             s=120,  # Large point size
                             c=df['value'],  # Color by value
                             cmap='viridis',  # Professional color map
                             alpha=0.8,
                             edgecolors='white',  # White edge for 3D effect
                             linewidths=2,
                             zorder=5)  # Ensure points are on top
        
        # Add a color bar
        cbar = plt.colorbar(scatter)
        cbar.set_label('Excess Power (%)', rotation=270, labelpad=20)
        
        # Calculate linear trend
        latest_value = None
        if len(df) > 1:
            # Convert dates to ordinal values for regression
            x_numeric = np.array(mdates.date2num(df['date']))
            y = df['value'].values
            
            # Calculate linear regression
            slope, intercept = np.polyfit(x_numeric, y, 1)
            
            # Generate line points
            x_line = np.array([x_numeric.min(), x_numeric.max()])
            y_line = slope * x_line + intercept
            
            # Calculate slope in readable terms (percent per day)
            slope_per_day = slope
            
            # Plot the trend line
            plt.plot(mdates.num2date(x_line), y_line, 
                    color='#FF0066',  # Vibrant pink
                    linestyle='-', 
                    linewidth=3, 
                    label=f'Trend Line (Slope: {slope_per_day:.4f}% per day)')
            
            # Save the latest value for reference
            latest_value = y_line[-1]
        
        # Add thresholds for hull power loss
        if metric_name == 'hull_roughness_power_loss':
            # Add colored background zones
            plt.axhspan(0, 15, color='#4CAF50', alpha=0.15, label='Good Condition')
            plt.axhspan(15, 25, color='#FFC107', alpha=0.15, label='Average Condition')
            plt.axhspan(25, max(df['value'].max() * 1.1, 30), color='#F44336', alpha=0.15, label='Poor Condition')
            
            # Add threshold lines
            plt.axhline(y=15, color='#FFC107', linestyle='--', alpha=0.7, linewidth=2, label='Good/Average Threshold (15%)')
            plt.axhline(y=25, color='#F44336', linestyle='--', alpha=0.7, linewidth=2, label='Average/Poor Threshold (25%)')
        
        # Add latest value annotation
        if latest_value is not None:
            # Determine color and condition based on value
            if metric_name == 'hull_roughness_power_loss':
                if latest_value < 15:
                    color = '#4CAF50'  # Green
                    condition = 'GOOD'
                elif latest_value < 25:
                    color = '#FFC107'  # Amber
                    condition = 'AVERAGE'
                else:
                    color = '#F44336'  # Red
                    condition = 'POOR'
            else:
                color = '#FF0066'  # Pink
                condition = ''
            
            # Add annotation with arrow pointing to the end of trend line
            plt.annotate(f'Latest: {latest_value:.2f}%\nCondition: {condition}',
                        xy=(df['date'].iloc[-1], latest_value),  # End of the trend line
                        xytext=(40, 0),  # Offset text to right
                        textcoords='offset points',
                        arrowprops=dict(arrowstyle='->', lw=2, color=color),
                        bbox=dict(boxstyle='round,pad=0.5', fc='white', ec=color, alpha=0.9),
                        fontsize=12,
                        color=color,
                        weight='bold',
                        ha='left',
                        va='center',
                        zorder=6)  # Ensure annotation is on top
        
        # Customize the axes
        plt.xlabel('Date', fontsize=14, fontweight='bold')
        plt.ylabel(y_axis_title, fontsize=14, fontweight='bold')
        plt.title(chart_title, fontsize=18, fontweight='bold', pad=20)
        
        # Format x-axis dates nicely
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
        plt.gca().xaxis.set_major_locator(mdates.MonthLocator(interval=1))
        plt.xticks(rotation=45)
        
        # Format y-axis
        plt.ylim(bottom=0)  # Start y-axis at 0
        
        # Add gridlines
        plt.grid(True, linestyle='--', alpha=0.7)
        
        # Add a legend
        plt.legend(loc='upper left', frameon=True, framealpha=0.9, edgecolor='gray')
        
        # Tight layout
        plt.tight_layout()
        
        # Create a Plotly figure wrapper for compatibility with the rest of your code
        # Convert matplotlib figure to image bytes
        buf = BytesIO()
        plt.savefig(buf, format='png', dpi=300, bbox_inches='tight')
        buf.seek(0)
        
        # Create a minimal plotly figure that wraps the matplotlib image
        fig = go.Figure()
        
        # Add the image as a layout image
        fig.add_layout_image(
            dict(
                source=f"data:image/png;base64,{BytesIO(buf.read()).getvalue().hex()}",
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
        
        # Update layout to match the image dimensions
        fig.update_layout(
            autosize=True,
            width=1200,
            height=800,
            margin=dict(l=0, r=0, b=0, t=0)
        )
        
        # Close the matplotlib figure to free memory
        plt.close()
        
        return fig, latest_value
    
    def get_hull_condition(self, hull_roughness):
        if hull_roughness < 15:
            return "GOOD", "green"
        elif 15 <= hull_roughness < 25:
            return "AVERAGE", "orange"
        else:
            return "POOR", "red"
