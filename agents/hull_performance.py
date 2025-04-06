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
        Create a publication-quality hull performance chart with linear trend line using Matplotlib
        """
        if not data:
            return None, None
        
        import matplotlib.pyplot as plt
        import matplotlib.dates as mdates
        import pandas as pd
        import numpy as np
        from matplotlib.ticker import MaxNLocator
        import matplotlib.patheffects as path_effects
        
        # Prepare data
        df = pd.DataFrame({
            'date': [pd.to_datetime(row['report_date']) for row in data],
            'value': [row.get(metric_name, 0) for row in data]
        })
        
        # Sort by date
        df = df.sort_values('date')
        
        # Create the figure with high resolution
        fig, ax = plt.subplots(figsize=(12, 8), dpi=300)
        
        # Set high-quality styles
        plt.rcParams['font.family'] = ['DejaVu Sans', 'Arial', 'Helvetica', 'sans-serif']
        plt.rcParams['font.size'] = 12
        plt.rcParams['axes.linewidth'] = 1.2
        plt.rcParams['axes.labelsize'] = 14
        plt.rcParams['axes.titlesize'] = 16
        plt.rcParams['xtick.labelsize'] = 12
        plt.rcParams['ytick.labelsize'] = 12
        
        # Add colored background zones for hull roughness power loss
        if metric_name == 'hull_roughness_power_loss':
            y_max = max(df['value'].max() * 1.2, 40)
            
            # Good zone (0-15%)
            ax.axhspan(0, 15, color='#4CAF50', alpha=0.15, label='Good Condition')
            
            # Average zone (15-25%)
            ax.axhspan(15, 25, color='#FFC107', alpha=0.15, label='Average Condition')
            
            # Poor zone (>25%)
            ax.axhspan(25, y_max, color='#F44336', alpha=0.15, label='Poor Condition')
            
            # Add threshold lines
            ax.axhline(y=15, color='#FFC107', linestyle='--', alpha=0.8, linewidth=2, 
                       label='Good/Average Threshold (15%)')
            ax.axhline(y=25, color='#F44336', linestyle='--', alpha=0.8, linewidth=2, 
                       label='Average/Poor Threshold (25%)')
        
        # Create a colormap based on values
        scatter = ax.scatter(
            df['date'], 
            df['value'],
            s=100,  # Size of points
            c=df['value'],  # Color by value
            cmap='viridis',  # Professional colormap
            alpha=0.8,
            edgecolor='white',  # White edge for 3D effect
            linewidth=1.5,
            zorder=5  # Ensure points are above background
        )
        
        # Add a color bar
        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label('Power Loss (%)', rotation=270, labelpad=20, fontsize=12)
        
        # Calculate and plot linear trend
        latest_value = None
        if len(df) > 1:
            # Convert dates to numbers for linear regression
            x_numeric = mdates.date2num(df['date'])
            y = df['value'].values
            
            # Calculate linear regression
            coeffs = np.polyfit(x_numeric, y, 1)
            slope = coeffs[0]
            intercept = coeffs[1]
            
            # Calculate slope in percent per day
            slope_per_day = slope
            
            # Create line points
            x_line = np.array([x_numeric.min(), x_numeric.max()])
            y_line = slope * x_line + intercept
            
            # Plot trend line
            line = ax.plot(
                mdates.num2date(x_line), 
                y_line, 
                color='#FF1493',  # Deep pink
                linestyle='-', 
                linewidth=3, 
                label=f'Trend Line (Slope: {slope_per_day:.4f}% per day)',
                zorder=4  # Ensure line is above background but below points
            )
            
            # Get the latest value from the trend line
            latest_value = y_line[-1]
            
            # Add annotation for latest value
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
                    color = '#FF1493'  # Pink
                    condition = ''
                
                # Add text annotation with custom styling and arrow
                annotation = ax.annotate(
                    f'Latest: {latest_value:.2f}%\nCondition: {condition}',
                    xy=(df['date'].iloc[-1], latest_value),  # Position at end of trend line
                    xytext=(30, 20),  # Offset text for clarity
                    textcoords='offset points',
                    bbox=dict(
                        boxstyle='round,pad=0.5',
                        fc='white',
                        ec=color,
                        alpha=0.9,
                        linewidth=2
                    ),
                    fontsize=12,
                    color=color,
                    weight='bold',
                    ha='left',
                    va='center',
                    arrowprops=dict(
                        arrowstyle='->',
                        color=color,
                        linewidth=2,
                        connectionstyle="arc3,rad=0.2"
                    ),
                    zorder=6  # Ensure annotation is on top
                )
                
                # Add subtle shadow effect to make text stand out
                annotation.set_path_effects([
                    path_effects.withStroke(linewidth=3, foreground='white')
                ])
        
        # Improve axis styling
        ax.set_xlabel('Date', fontsize=14, fontweight='bold')
        ax.set_ylabel(y_axis_title, fontsize=14, fontweight='bold')
        ax.set_title(chart_title, fontsize=18, fontweight='bold', pad=20)
        
        # Format x-axis dates with no overlapping
        plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=1))
        
        # Set y-axis to start at 0
        ax.set_ylim(bottom=0)
        
        # Add gridlines
        ax.grid(True, linestyle='--', alpha=0.4, color='gray')
        
        # Add frame around the plot area
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_linewidth(1.0)
            spine.set_color('#333333')
        
        # Position legend for optimal visibility
        legend = ax.legend(
            loc='upper left', 
            frameon=True, 
            framealpha=0.95,
            edgecolor='gray',
            fancybox=True,
            shadow=True,
            fontsize=10
        )
        
        # Set tight layout
        plt.tight_layout()
        
        # Instead of returning a Plotly figure, return the Matplotlib figure directly
        # and implement a compatible wrapper class for integration with your report generator
        
        class MatplotlibFigureWrapper:
            def __init__(self, fig):
                self.fig = fig
                # Add a dummy update_layout method for compatibility
                self.update_layout = lambda **kwargs: None
                
            def savefig(self, *args, **kwargs):
                return self.fig.savefig(*args, **kwargs)
    
        # Return the wrapped figure and latest value
        return MatplotlibFigureWrapper(fig), latest_value    
    def get_hull_condition(self, hull_roughness):
        if hull_roughness < 15:
            return "GOOD", "green"
        elif 15 <= hull_roughness < 25:
            return "AVERAGE", "orange"
        else:
            return "POOR", "red"
