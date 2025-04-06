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
        
        # Set up the styling
        sns.set(style="whitegrid", font="Arial")
        plt.rcParams["figure.figsize"] = (12, 8)
        plt.rcParams["font.family"] = "sans-serif"
        plt.rcParams["font.sans-serif"] = ["Arial"]
        
        # Prepare data - ensure dates are in datetime format
        df = pd.DataFrame({
            'date': [pd.to_datetime(row['report_date']) for row in data],
            'value': [row.get(metric_name, 0) for row in data]
        })
        
        # Sort by date for trend analysis
        df = df.sort_values('date')
        
        # Create figure and axis objects
        fig, ax = plt.subplots(1, 1, facecolor='white')
        
        # Plot the scatter points
        scatter = sns.scatterplot(
            x='date', 
            y='value', 
            data=df, 
            s=100,  # Larger point size
            alpha=0.7,
            palette="viridis",
            hue='value',  # Color by value
            legend=False,
            ax=ax
        )
        
        # Calculate linear trend (explicitly linear as requested)
        if len(df) > 1:
            # Convert dates to ordinal values for regression
            x_numeric = np.array([(d - df['date'].min()).total_seconds() / 86400 for d in df['date']])
            y = df['value'].values
            
            # Calculate linear regression
            slope, intercept = np.polyfit(x_numeric, y, 1)
            
            # Create date range for the trend line
            x_line = np.array([0, x_numeric.max()])
            y_line = slope * x_line + intercept
            
            # Convert back to datetime for plotting
            x_line_dates = [df['date'].min() + pd.Timedelta(days=int(x)) for x in x_line]
            
            # Add the trend line
            plt.plot(x_line_dates, y_line, color='#FF0066', linestyle='-', linewidth=3, 
                     label=f'Trend Line (Slope: {slope:.4f} % per day)')
            
            # Save the latest value for reference
            latest_value = y_line[-1]
        else:
            latest_value = df['value'].iloc[-1] if not df.empty else None
        
        # Add thresholds for hull roughness power loss
        if metric_name == 'hull_roughness_power_loss':
            # Add colored background zones
            plt.axhspan(0, 15, alpha=0.15, color='green', label='Good Condition')
            plt.axhspan(15, 25, alpha=0.15, color='orange', label='Average Condition')
            plt.axhspan(25, max(df['value'].max() * 1.2, 30), alpha=0.15, color='red', label='Poor Condition')
            
            # Add threshold lines
            plt.axhline(y=15, color='orange', linestyle='--', alpha=0.7, linewidth=2)
            plt.axhline(y=25, color='red', linestyle='--', alpha=0.7, linewidth=2)
            
            # Add annotations for thresholds - positioned to avoid overlap
            plt.text(df['date'].min(), 15.5, 'Good/Average Threshold (15%)', 
                     fontsize=12, ha='left', va='bottom', color='darkorange',
                     bbox=dict(facecolor='white', alpha=0.7, boxstyle='round,pad=0.5'))
            
            plt.text(df['date'].min(), 25.5, 'Average/Poor Threshold (25%)', 
                     fontsize=12, ha='left', va='bottom', color='darkred',
                     bbox=dict(facecolor='white', alpha=0.7, boxstyle='round,pad=0.5'))
        
        # Add latest value annotation if we have it
        if latest_value is not None:
            # Determine color and condition based on value
            if metric_name == 'hull_roughness_power_loss':
                if latest_value < 15:
                    color = 'green'
                    condition = 'GOOD'
                elif latest_value < 25:
                    color = 'orange'
                    condition = 'AVERAGE'
                else:
                    color = 'red'
                    condition = 'POOR'
            else:
                color = '#FF0066'
                condition = ''
                
            # Add a distinct callout for the latest value
            plt.annotate(
                f'Latest: {latest_value:.2f}%\nCondition: {condition}',
                xy=(df['date'].max(), latest_value),
                xytext=(30, 20),
                textcoords='offset points',
                arrowprops=dict(arrowstyle='->', lw=2, color=color),
                bbox=dict(boxstyle='round,pad=0.5', fc='white', alpha=0.8),
                fontsize=12,
                color=color,
                weight='bold'
            )
        
        # Improve axes and labels
        ax.set_title(chart_title, fontsize=20, pad=20, fontweight='bold')
        ax.set_xlabel('Date', fontsize=14, labelpad=15)
        ax.set_ylabel(y_axis_title, fontsize=14, labelpad=15)
        
        # Format x-axis dates nicely - with no overlapping
        fig.autofmt_xdate()
        
        # Format y-axis with percentage
        ax.yaxis.set_major_locator(MaxNLocator(nbins=10))
        ax.set_ylim(bottom=0)  # Start y-axis at 0
        
        # Add a legend with no overlap
        handles, labels = ax.get_legend_handles_labels()
        ax.legend(handles, labels, loc='upper left', fontsize=12, 
                  frameon=True, framealpha=0.9, edgecolor='gray')
        
        # Add a subtle border around the plot
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_color('lightgray')
        
        # Add a watermark-like vessel name in the background
        if metric_name == 'hull_roughness_power_loss':
            ax.text(0.5, 0.5, data[0].get('vessel_name', '').upper(),
                    transform=ax.transAxes, fontsize=60, color='gray',
                    ha='center', va='center', alpha=0.07)
        
        # Ensure tight layout
        plt.tight_layout()
        
        # For Word doc, we'll return the figure to save as image
        return fig, latest_value
    
    def get_hull_condition(self, hull_roughness):
        if hull_roughness < 15:
            return "GOOD", "green"
        elif 15 <= hull_roughness < 25:
            return "AVERAGE", "orange"
        else:
            return "POOR", "red"
