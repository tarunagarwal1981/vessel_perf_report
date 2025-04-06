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
        using Seaborn for enhanced aesthetics
        """
        if not data:
            return None, None
        
        import matplotlib.pyplot as plt
        import matplotlib.dates as mdates
        import pandas as pd
        import numpy as np
        import seaborn as sns
        
        # Prepare data
        df = pd.DataFrame({
            'date': [pd.to_datetime(row['report_date']) for row in data],
            'value': [row.get(metric_name, 0) for row in data]
        })
        
        # Sort by date
        df = df.sort_values('date')
        
        # Set a professional seaborn style
        sns.set_theme(style="whitegrid", context="paper")
        
        # Create figure with the right size
        fig, ax = plt.subplots(figsize=(12, 7))
        
        # Add colored background zones for hull roughness power loss
        if metric_name == 'hull_roughness_power_loss':
            y_max = max(df['value'].max() * 1.2, 40)
            
            # Good zone (0-15%)
            ax.axhspan(0, 15, color=sns.color_palette("Greens", n_colors=9)[3], alpha=0.2, label='Good Condition')
            
            # Average zone (15-25%)
            ax.axhspan(15, 25, color=sns.color_palette("Oranges", n_colors=9)[3], alpha=0.2, label='Average Condition')
            
            # Poor zone (>25%)
            ax.axhspan(25, y_max, color=sns.color_palette("Reds", n_colors=9)[3], alpha=0.2, label='Poor Condition')
            
            # Add threshold lines
            ax.axhline(y=15, color=sns.color_palette("Oranges", n_colors=9)[5], linestyle='--', 
                      linewidth=1.5, label='Good/Average Threshold (15%)')
            ax.axhline(y=25, color=sns.color_palette("Reds", n_colors=9)[5], linestyle='--', 
                      linewidth=1.5, label='Average/Poor Threshold (25%)')
        
        # Create scatter plot with enhanced aesthetics
        scatter = sns.scatterplot(
            data=df,
            x='date',
            y='value',
            hue='value',  # Color points by value
            palette='viridis',  # Professional color palette
            size='value',  # Vary size by value
            sizes=(50, 200),  # Min and max sizes
            alpha=0.7,
            legend=False,
            ax=ax
        )
        
        # Add a color bar for the values
        from matplotlib.cm import ScalarMappable
        from matplotlib.colors import Normalize
        
        norm = Normalize(vmin=df['value'].min(), vmax=df['value'].max())
        sm = ScalarMappable(cmap='viridis', norm=norm)
        sm.set_array([])
        
        cbar = plt.colorbar(sm, ax=ax)
        cbar.set_label('Excess Power (%)', rotation=270, labelpad=20)
        
        # Calculate linear trend
        latest_value = None
        if len(df) > 1:
            # Add regression line with confidence interval
            sns.regplot(
                x=mdates.date2num(df['date']),
                y=df['value'],
                scatter=False,
                color=sns.color_palette("rocket")[4],
                line_kws={"linewidth": 2.5},
                ax=ax
            )
            
            # Calculate the actual trend line for reporting the slope
            x_numeric = mdates.date2num(df['date'])
            slope, intercept = np.polyfit(x_numeric, df['value'], 1)
            
            # Get the latest value from the trend line
            latest_x = x_numeric[-1]
            latest_value = slope * latest_x + intercept
            
            # Add a custom label for the trend line
            ax.plot([], [], color=sns.color_palette("rocket")[4], linewidth=2.5, 
                   label=f'Trend Line (Slope: {slope:.4f}% per day)')
            
            # Add annotation for latest value
            if latest_value is not None:
                # Determine condition based on value
                if metric_name == 'hull_roughness_power_loss':
                    if latest_value < 15:
                        condition = 'GOOD'
                        color = sns.color_palette("Greens", n_colors=9)[6]
                    elif latest_value < 25:
                        condition = 'AVERAGE'
                        color = sns.color_palette("Oranges", n_colors=9)[6]
                    else:
                        condition = 'POOR'
                        color = sns.color_palette("Reds", n_colors=9)[6]
                else:
                    condition = ''
                    color = sns.color_palette("rocket")[4]
                
                # Add annotation with Seaborn styling
                bbox_props = dict(
                    boxstyle="round,pad=0.5", 
                    fc="white", 
                    ec=color, 
                    alpha=0.9,
                    lw=2
                )
                
                ax.annotate(
                    f'Latest: {latest_value:.2f}%\nCondition: {condition}',
                    xy=(df['date'].iloc[-1], latest_value),
                    xytext=(30, 15),
                    textcoords='offset points',
                    bbox=bbox_props,
                    fontsize=11,
                    color=color,
                    weight='bold',
                    arrowprops=dict(
                        arrowstyle="->",
                        color=color,
                        connectionstyle="arc3,rad=0.2"
                    )
                )
        
        # Improve styling
        ax.set_xlabel('Date', fontweight='bold')
        ax.set_ylabel(y_axis_title, fontweight='bold')
        plt.title(chart_title, fontweight='bold', pad=20, fontsize=16)
        
        # Format x-axis dates
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=1))
        plt.xticks(rotation=45, ha='right')
        
        # Set y-axis to start at 0
        ax.set_ylim(bottom=0)
        
        # Add a professional-looking legend
        ax.legend(loc='upper left', frameon=True, framealpha=0.95)
        
        # Set tight layout
        plt.tight_layout()
        
        return fig, latest_value
    def get_hull_condition(self, hull_roughness):
        if hull_roughness < 15:
            return "GOOD", "green"
        elif 15 <= hull_roughness < 25:
            return "AVERAGE", "orange"
        else:
            return "POOR", "red"
