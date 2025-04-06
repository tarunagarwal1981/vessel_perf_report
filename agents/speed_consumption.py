import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

class SpeedConsumptionAgent:
    def __init__(self):
        pass
    
    def run(self, data, vessel_name):
        # Filter data for speed-consumption analysis
        filtered_data_consumption = []
        
        for entry in data:
            if ('speed' in entry and entry['speed'] is not None and 
                'normalised_consumption' in entry and entry['normalised_consumption'] is not None and
                'loading_condition' in entry and entry['loading_condition'] is not None):
                filtered_data_consumption.append(entry)
        
        if not filtered_data_consumption:
            st.warning("No valid speed-consumption data found for the selected vessel and filters.")
            return
        
        # Create tabs for different loading conditions
        sc_tab1, sc_tab2 = st.tabs(["Ballast Condition", "Laden Condition"])
        
        # Ballast Condition Tab
        with sc_tab1:
            try:
                chart = self.create_speed_consumption_chart(
                    filtered_data_consumption,
                    "ballast",
                    f"Speed vs. Consumption - Ballast Condition ({vessel_name.upper()})"
                )
                
                if chart:
                    st.plotly_chart(chart, use_container_width=True)
                    
                    # Add descriptive text
                    st.markdown("""
                    ### Analysis
                    The chart above shows the relationship between vessel speed and fuel consumption in ballast condition.
                    
                    - Each point represents a daily report
                    - The blue curve is a 2nd order polynomial fit to the data
                    - Points are color-coded by date (newer points are brighter)
                    
                    This curve can be used to estimate the optimal speed for fuel efficiency.
                    """)
                else:
                    st.info("No ballast condition data available for the selected vessel and filters.")
            except Exception as e:
                st.error(f"Error creating Ballast Condition chart: {str(e)}")
        
        # Laden Condition Tab
        with sc_tab2:
            try:
                chart = self.create_speed_consumption_chart(
                    filtered_data_consumption,
                    "laden",
                    f"Speed vs. Consumption - Laden Condition ({vessel_name.upper()})"
                )
                
                if chart:
                    st.plotly_chart(chart, use_container_width=True)
                    
                    # Add descriptive text
                    st.markdown("""
                    ### Analysis
                    The chart above shows the relationship between vessel speed and fuel consumption in laden condition.
                    
                    - Each point represents a daily report
                    - The blue curve is a 2nd order polynomial fit to the data
                    - Points are color-coded by date (newer points are brighter)
                    
                    This curve can be used to estimate the optimal speed for fuel efficiency.
                    """)
                else:
                    st.info("No laden condition data available for the selected vessel and filters.")
            except Exception as e:
                st.error(f"Error creating Laden Condition chart: {str(e)}")
    
    def create_speed_consumption_chart(self, data, condition, chart_title):
        if not data:
            return None
        
        # Filter data for the specified loading condition
        condition_data = [row for row in data if row.get('loading_condition', '').lower() == condition.lower()]
        
        if not condition_data:
            return None
        
        # Extract speed and consumption data
        speeds = [row.get('speed', 0) for row in condition_data]
        consumptions = [row.get('normalised_consumption', 0) for row in condition_data]
        dates = [pd.to_datetime(row.get('report_date')) for row in condition_data]
        
        # Create the figure with enhanced design
        fig = go.Figure()
        
        # Add a decorative background grid to suggest a 3D plane
        speeds_range = max(speeds) - min(speeds)
        consumption_range = max(consumptions) - min(consumptions)
        
        # Create the scatter plot with enhanced visualization
        fig.add_trace(go.Scatter(
            x=speeds,
            y=consumptions,
            mode='markers',
            marker=dict(
                size=14,
                color=speeds,  # Color by speed value
                colorscale='Viridis',
                showscale=True,
                colorbar=dict(
                    title="Speed (knots)",
                    titleside="right",
                    titlefont=dict(size=12, family="Arial")
                ),
                line=dict(width=2, color='rgba(255, 255, 255, 0.8)'),  # White border for 3D effect
                symbol='circle',
            ),
            name='Speed-Consumption Data',
            hoverlabel=dict(
                bgcolor="white",
                font_size=12,
                font_family="Arial"
            ),
        ))
        
        # Calculate curve fit if enough data points
        if len(speeds) > 2:
            # Sort values by speed for smooth curve
            sorted_indices = np.argsort(speeds)
            speeds_sorted = [speeds[i] for i in sorted_indices]
            consumptions_sorted = [consumptions[i] for i in sorted_indices]
            
            # Create ranges for plotting
            speeds_smooth = np.linspace(min(speeds), max(speeds), 100)
            
            # Try cubic fit if enough points, otherwise polynomial
            if len(speeds) >= 8:
                from scipy.interpolate import CubicSpline
                try:
                    cs = CubicSpline(speeds_sorted, consumptions_sorted)
                    consumptions_smooth = cs(speeds_smooth)
                    line_type = 'spline'
                except:
                    # Fallback to polynomial
                    coeffs = np.polyfit(speeds, consumptions, 3)
                    poly = np.poly1d(coeffs)
                    consumptions_smooth = poly(speeds_smooth)
                    line_type = 'polynomial'
            else:
                # Use polynomial fit
                coeffs = np.polyfit(speeds, consumptions, 2)
                poly = np.poly1d(coeffs)
                consumptions_smooth = poly(speeds_smooth)
                line_type = 'polynomial'
            
            # Add the optimized curve with gradient
            fig.add_trace(go.Scatter(
                x=speeds_smooth,
                y=consumptions_smooth,
                mode='lines',
                line=dict(
                    color='rgba(65, 105, 225, 0.9)',
                    width=3.5,
                    shape='spline'
                ),
                name='Optimized Curve',
                fill='tozeroy',
                fillcolor='rgba(65, 105, 225, 0.1)'
            ))
            
            # Add annotation with equation - more professional looking
            if line_type == 'polynomial':
                if len(coeffs) >= 3:
                    equation = f"y = {coeffs[0]:.4f}x² + {coeffs[1]:.4f}x + {coeffs[2]:.4f}"
                    fig.add_annotation(
                        x=0.5,
                        y=0.05,
                        xref="paper",
                        yref="paper",
                        text=equation,
                        showarrow=False,
                        font=dict(family="Arial", size=12, color="#333333"),
                        bgcolor="rgba(255, 255, 255, 0.9)",
                        bordercolor="rgba(65, 105, 225, 0.9)",
                        borderwidth=1,
                        borderpad=4
                    )
            
            # Find optimal speed (lowest consumption per distance)
            # This would be the minimum of consumption/speed
            efficiency = [c/s for c, s in zip(consumptions_smooth, speeds_smooth) if s > 0]
            if efficiency:
                optimal_index = np.argmin(efficiency)
                optimal_speed = speeds_smooth[optimal_index]
                optimal_consumption = consumptions_smooth[optimal_index]
                
                # Mark the optimal point
                fig.add_trace(go.Scatter(
                    x=[optimal_speed],
                    y=[optimal_consumption],
                    mode='markers',
                    marker=dict(
                        size=16,
                        color='rgba(255, 90, 0, 1)',
                        symbol='star',
                        line=dict(width=2, color='white')
                    ),
                    name='Optimal Efficiency Point'
                ))
                
                fig.add_annotation(
                    x=optimal_speed,
                    y=optimal_consumption,
                    text=f"Optimal: {optimal_speed:.1f} knots",
                    showarrow=True,
                    arrowhead=2,
                    arrowcolor="rgba(255, 90, 0, 1)",
                    arrowsize=1,
                    arrowwidth=2,
                    ax=40,
                    ay=-40,
                    font=dict(family="Arial", size=12, color="rgba(255, 90, 0, 1)"),
                    bgcolor="white",
                    bordercolor="rgba(255, 90, 0, 1)",
                    borderwidth=1,
                    borderpad=4
                )
        
        # Update layout for a more professional look
        fig.update_layout(
            title={
                'text': chart_title,
                'y':0.95,
                'x':0.5,
                'xanchor': 'center',
                'yanchor': 'top',
                'font': dict(size=18, family="Arial", color="#333333")
            },
            xaxis_title={
                'text': "Speed (knots)",
                'font': dict(size=14, family="Arial", color="#333333")
            },
            yaxis_title={
                'text': "Fuel Consumption (mt/day)",
                'font': dict(size=14, family="Arial", color="#333333")
            },
            paper_bgcolor='rgb(255, 255, 255)',
            plot_bgcolor='rgb(245, 245, 245)',
            height=500,
            margin=dict(l=40, r=50, t=60, b=40),
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1,
                font=dict(family="Arial", size=12, color="#333333"),
                bgcolor="rgba(255, 255, 255, 0.7)",
                bordercolor="#DDDDDD",
                borderwidth=1
            )
        )
        
        # Update axes styling for consistency
        fig.update_xaxes(
            showgrid=True,
            gridwidth=1,
            gridcolor='rgba(211, 211, 211, 0.5)',
            zeroline=False,
            showline=True,
            linewidth=2,
            linecolor='#333333',
            tickfont=dict(family="Arial", size=12, color="#333333")
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
        
        return fig
