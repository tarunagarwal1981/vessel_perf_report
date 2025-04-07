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
        """
        Create a hull performance chart using the provided styling
        """
        if not data:
            return None, None
        
        import matplotlib.pyplot as plt
        import numpy as np
        import pandas as pd
        import matplotlib.dates as mdates
        from matplotlib.patches import Rectangle
        from matplotlib.gridspec import GridSpec
        import matplotlib.patheffects as path_effects
        from matplotlib.colors import LinearSegmentedColormap
        from matplotlib.collections import LineCollection
        
        # Prepare data
        dates = [pd.to_datetime(row['report_date']) for row in data]
        metric_values = [row.get(metric_name, 0) for row in data]
        
        # Sort data by date
        sorted_indices = np.argsort(dates)
        dates = [dates[i] for i in sorted_indices]
        metric_values = [metric_values[i] for i in sorted_indices]
        
        # Set the style for publication quality
        plt.rcParams['figure.dpi'] = 300
        plt.rcParams['font.family'] = 'sans-serif'
        plt.rcParams['axes.linewidth'] = 1.2
        
        # Create figure with dark background
        fig = plt.figure(figsize=(12, 8))
        fig.patch.set_facecolor('#1a1a1a')
        gs = GridSpec(2, 1, height_ratios=[4, 1], hspace=0.3)
        
        # Main plot
        ax1 = plt.subplot(gs[0])
        ax1.set_facecolor('#1a1a1a')
        
        # Create subtle zone backgrounds
        zones = [
            (0, 15, '#004400', 'Good Zone'),
            (15, 25, '#443300', 'Average Zone'),
            (25, 40, '#440000', 'Poor Zone')
        ]
        
        for start, end, color, label in zones:
            ax1.axhspan(start, end, color=color, alpha=0.3, label=label)
        
        # Calculate trend line
        dates_num = mdates.date2num(dates)
        z = np.polyfit(dates_num, metric_values, 1)
        p = np.poly1d(z)
        trend_line = p(dates_num)
        
        # Get latest value from trend line
        latest_value = trend_line[-1] if len(trend_line) > 0 else None
        
        # Create neon color gradient for time-based coloring
        neon_colors = [
            '#FF00FF',  # Neon Pink
            '#00FF00',  # Neon Green
            '#00FFFF',  # Neon Cyan
            '#FF0099',  # Neon Magenta
            '#7F00FF'   # Neon Purple
        ]
        
        # Create custom colormap for the neon effect
        cmap = LinearSegmentedColormap.from_list('neon', neon_colors)
        
        # Add trend line with glow effect
        # Outer glow
        ax1.plot(dates, trend_line, 
                 color='white', 
                 linewidth=8, 
                 alpha=0.1, 
                 zorder=2)
        
        # Inner glow
        ax1.plot(dates, trend_line, 
                 color='white', 
                 linewidth=4, 
                 alpha=0.2, 
                 zorder=3)
        
        # Main line with gradient color
        points = np.array([dates_num, trend_line]).T.reshape(-1, 1, 2)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)
        lc = LineCollection(segments, cmap=cmap, linewidth=2)
        lc.set_array(np.linspace(0, 1, len(dates)))
        ax1.add_collection(lc)
        
        # Add threshold lines with neon glow
        for threshold, color, label in [(15, '#00FF00', 'Good/Average Threshold'),
                                       (25, '#FF0000', 'Average/Poor Threshold')]:
            # Outer glow
            ax1.axhline(y=threshold, color=color, linestyle='-',
                        alpha=0.2, linewidth=4, zorder=1)
            # Inner glow
            ax1.axhline(y=threshold, color=color, linestyle='-',
                        alpha=0.4, linewidth=2, zorder=2)
            # Main line
            line = ax1.axhline(y=threshold, color=color, linestyle='-',
                               alpha=0.8, linewidth=1, label=label,
                               path_effects=[path_effects.SimpleLineShadow(offset=(0, 0),
                                                                         alpha=0.2),
                                           path_effects.Normal()],
                               zorder=3)
        
        # Plot scatter points with enhanced neon glow effect
        # Outer glow
        ax1.scatter(dates, metric_values,
                   c=np.arange(len(dates)),
                   cmap=cmap,
                   s=200,
                   alpha=0.1,
                   zorder=4)
        
        # Middle glow
        ax1.scatter(dates, metric_values,
                   c=np.arange(len(dates)),
                   cmap=cmap,
                   s=150,
                   alpha=0.2,
                   zorder=5)
        
        # Main points
        scatter = ax1.scatter(dates, metric_values,
                             c=np.arange(len(dates)),
                             cmap=cmap,
                             s=100,
                             alpha=1,
                             edgecolor='white',
                             linewidth=1,
                             label='Performance Data',
                             zorder=6)
        
        # Inner highlight
        ax1.scatter(dates, metric_values,
                   c='white',
                   s=30,
                   alpha=0.5,
                   zorder=7)
        
        # Styling for dark theme
        title_text = chart_title or 'Hull Performance Analysis'
        title = ax1.set_title(title_text,
                              fontsize=16,
                              pad=20,
                              fontweight='bold',
                              color='white')
        title.set_path_effects([path_effects.SimpleLineShadow(offset=(2, 2),
                                                             alpha=0.3),
                               path_effects.Normal()])
        
        ax1.set_xlabel('Date', fontsize=12, labelpad=10, color='white')
        ax1.set_ylabel(y_axis_title or 'Performance Metric (%)', fontsize=12, labelpad=10, color='white')
        
        # Format axes
        ax1.tick_params(axis='both', which='major', labelsize=10, colors='white')
        ax1.xaxis.set_major_locator(mdates.MonthLocator())
        ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        plt.setp(ax1.get_xticklabels(), rotation=45, ha='right')
        
        # Customize grid for dark theme
        ax1.grid(True, linestyle='--', alpha=0.2, color='white', zorder=0)
        
        # Add colorbar for time progression
        cbar = plt.colorbar(scatter, ax=ax1, pad=0.02)
        cbar.set_label('Time Progression', fontsize=10, color='white')
        cbar.ax.yaxis.set_tick_params(color='white')
        cbar.ax.tick_params(labelcolor='white')
        
        # Update spines color
        for spine in ax1.spines.values():
            spine.set_color('white')
        
        # Add legend with dark theme styling
        legend = ax1.legend(bbox_to_anchor=(1.15, 1),
                           loc='upper left',
                           fontsize=10,
                           frameon=True,
                           facecolor='#1a1a1a',
                           edgecolor='white',
                           shadow=True)
        for text in legend.get_texts():
            text.set_color('white')
        
        # Add statistics subplot with dark theme styling but remove the stats as requested
        ax2 = plt.subplot(gs[1])
        ax2.set_facecolor('#1a1a1a')
        
        # Instead of stats, just add a note about the trend
        if z[0] < 0:
            trend_direction = "Improving (Decreasing)"
            trend_color = "#00FF00"  # Green for improvement
        else:
            trend_direction = "Degrading (Increasing)"
            trend_color = "#FF0000"  # Red for degradation
        
        trend_text = f"Trend Direction: {trend_direction}"
        
        # Add trend info box with dark theme
        trend_info = ax2.text(0.05, 0.5, trend_text,
                         transform=ax2.transAxes,
                         bbox=dict(facecolor='#1a1a1a',
                                  edgecolor=trend_color,
                                  alpha=0.8,
                                  boxstyle='round,pad=1'),
                         fontsize=12,
                         color=trend_color,
                         verticalalignment='center')
        ax2.axis('off')
        
        # Adjust layout
        plt.tight_layout()
        
        # Return the figure and latest value
        return fig, latest_value
    def get_hull_condition(self, hull_roughness):
        if hull_roughness < 15:
            return "GOOD", "green"
        elif 15 <= hull_roughness < 25:
            return "AVERAGE", "orange"
        else:
            return "POOR", "red"
