import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from io import BytesIO
from docx import Document
from docx.shared import Inches, Pt, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
import base64
import tempfile
import os
import copy
import uuid
import traceback

class AdvancedReportGenerator:
    def __init__(self):
        # Define paths to icon images - update these paths to your actual icon locations
        self.icon_paths = {
            'hull_icon': 'icons/hull_icon.png',
            'machinery_icon': 'icons/machinery_icon.png',
            'emissions_icon': 'icons/emissions_icon.png'
        }
    
    def run(self, vessel_data, selected_vessel, hull_agent, speed_agent):
        st.header("Performance Report Generator")
        
        try:
            # Create tabs for report view and download
            report_tab1, report_tab2 = st.tabs(["Report Preview", "Download Report"])
            
            with report_tab1:
                # Display basic report preview
                st.subheader("Vessel Performance Summary")
                
                # Extract metrics for preview
                hull_condition, power_loss, fuel_savings, recommendation = self._get_hull_metrics(vessel_data, hull_agent)
                ballast_avg, laden_avg = self._get_speed_metrics(vessel_data)
                
                # Create preview layout similar to the document
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.markdown("### Hull & Propeller")
                    st.markdown(f"**Condition:** {hull_condition}")
                    st.markdown(f"**Power Loss:** {power_loss:.1f}%")
                    st.markdown(f"**Potential Savings:** {fuel_savings:.1f} MT/D")
                
                with col2:
                    st.markdown("### Speed & Consumption")
                    st.markdown(f"**Ballast:** {ballast_avg:.1f} MT/day")
                    st.markdown(f"**Laden:** {laden_avg:.1f} MT/day")
                
                with col3:
                    st.markdown("### Emissions")
                    st.markdown("**CII Rating:** A (Placeholder)")
                    st.markdown("**AER:** 2.9 (Placeholder)")
                
                # Display hull performance chart if available
                st.subheader("Hull & Propeller Performance Analysis")
                hull_chart = self._create_hull_performance_chart(vessel_data)
                if hull_chart:
                    st.plotly_chart(hull_chart, use_container_width=True, key=f"hull_preview_{str(uuid.uuid4())[:8]}")
                
                # Display speed-consumption charts
                st.subheader("Speed Consumption Profile")
                col1, col2 = st.columns(2)
                
                ballast_chart, laden_chart = self._create_speed_consumption_charts(vessel_data)
                
                with col1:
                    st.markdown("**Ballast Condition**")
                    if ballast_chart:
                        st.plotly_chart(ballast_chart, use_container_width=True, key=f"ballast_preview_{str(uuid.uuid4())[:8]}")
                    else:
                        st.info("No ballast data available")
                
                with col2:
                    st.markdown("**Laden Condition**")
                    if laden_chart:
                        st.plotly_chart(laden_chart, use_container_width=True, key=f"laden_preview_{str(uuid.uuid4())[:8]}")
                    else:
                        st.info("No laden data available")
            
            with report_tab2:
                st.markdown("### Generate Report")
                
                # Report inputs
                col1, col2 = st.columns(2)
                
                with col1:
                    report_date = st.date_input("Report Date")
                    analyst_name = st.text_input("Analyst Name", "Marine Performance Team")
                
                with col2:
                    # Include sections options
                    st.subheader("Report Sections")
                    include_hull = st.checkbox("Include Hull Performance", value=True)
                    include_speed = st.checkbox("Include Speed Consumption", value=True)
                    include_emissions = st.checkbox("Include Emissions (Placeholder)", value=False)
                    include_machinery = st.checkbox("Include Machinery (Placeholder)", value=False)
                
                # Report template options
                st.subheader("Report Template")
                template_option = st.radio(
                    "Select Template Style",
                    options=["Basic Template", "Advanced Template (With Icons)"],
                    index=1
                )
                
                # Generate report button
                if st.button("Generate Report"):
                    try:
                        with st.spinner("Generating report..."):
                            # Extract metrics for report
                            hull_condition, power_loss, fuel_savings, recommendation = self._get_hull_metrics(vessel_data, hull_agent)
                            ballast_avg, laden_avg = self._get_speed_metrics(vessel_data)
                            
                            # Generate the report document
                            docx_file = self._generate_formatted_report(
                                vessel_name=selected_vessel,
                                report_date=report_date,
                                analyst_name=analyst_name,
                                hull_metrics={
                                    'condition': hull_condition,
                                    'power_loss': power_loss,
                                    'fuel_savings': fuel_savings,
                                    'recommendation': recommendation
                                },
                                speed_metrics={
                                    'ballast_avg': ballast_avg,
                                    'laden_avg': laden_avg
                                },
                                options={
                                    'include_hull': include_hull,
                                    'include_speed': include_speed,
                                    'include_emissions': include_emissions,
                                    'include_machinery': include_machinery,
                                    'template_option': template_option
                                },
                                vessel_data=vessel_data
                            )
                            
                            # Create download button
                            st.download_button(
                                label="Download Report",
                                data=docx_file,
                                file_name=f"{selected_vessel}_Performance_Report_{report_date}.docx",
                                mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                                key="download_report_button"
                            )
                            
                            st.success("Report generated successfully!")
                    except Exception as e:
                        st.error(f"Error generating report: {str(e)}")
                        st.code(traceback.format_exc())
    def _get_hull_metrics(self, vessel_data, hull_agent):
        try:
            # Filter data for power loss
            filtered_data = []
            for entry in vessel_data:
                if 'report_date' in entry and 'hull_roughness_power_loss' in entry:
                    if entry['hull_roughness_power_loss'] is not None:
                        filtered_data.append(entry)
    
            if filtered_data:
                # Sort by date
                filtered_data.sort(key=lambda x: pd.to_datetime(x['report_date']))  # Remove extra ')'
    
                # Get latest power loss
                power_loss = filtered_data[-1].get('hull_roughness_power_loss', 0)
    
                # Get condition
                condition, _ = hull_agent.get_hull_condition(power_loss)
    
                # Calculate savings (formula can be adjusted)
                fuel_savings = (power_loss - 15) * 0.05 if power_loss > 15 else 0
    
                # Generate recommendation
                if power_loss < 15:
                    recommendation = "Hull and propeller performance is good. No action required."
                elif 15 <= power_loss < 25:
                    recommendation = "Consider hull cleaning at next convenient opportunity."
                else:
                    recommendation = "Hull cleaning recommended as soon as possible."
    
                return condition, power_loss, fuel_savings, recommendation
            else:
                return "Unknown", 0, 0, "Insufficient data to provide recommendation."
    
        except Exception as e:
            # print(f"Error extracting hull metrics: {str(e)}")
            # return "Error", 0, 0, f"Error analyzing hull performance: {str(e)}"
            return "Error", 0, 0, str(e)    
            
    def _get_speed_metrics(self, vessel_data):
        try:
            # Filter data for speed-consumption
            ballast_consumptions = []
            laden_consumptions = []
            
            for entry in vessel_data:
                if 'loading_condition' in entry and 'normalised_consumption' in entry:
                    if entry['normalised_consumption'] is not None:
                        if entry.get('loading_condition', '').lower() == 'ballast':
                            ballast_consumptions.append(entry['normalised_consumption'])
                        elif entry.get('loading_condition', '').lower() == 'laden':
                            laden_consumptions.append(entry['normalised_consumption'])
            
            # Calculate averages
            ballast_avg = sum(ballast_consumptions) / len(ballast_consumptions) if ballast_consumptions else 0
            laden_avg = sum(laden_consumptions) / len(laden_consumptions) if laden_consumptions else 0
            
            return ballast_avg, laden_avg
        except Exception as e:
            print(f"Error extracting speed metrics: {str(e)}")
            return 0, 0
    
    def _create_hull_performance_chart(self, vessel_data):
        try:
            # Filter data for hull performance
            filtered_data = []
            for entry in vessel_data:
                if 'report_date' in entry and 'hull_roughness_power_loss' in entry:
                    if entry['hull_roughness_power_loss'] is not None:
                        filtered_data.append({
                            'date': pd.to_datetime(entry['report_date']),
                            'power_loss': entry['hull_roughness_power_loss']
                        })
            
            if not filtered_data:
                return None
            
            # Sort by date
            filtered_data.sort(key=lambda x: x['date'])
            
            # Extract data for plotting
            dates = [entry['date'] for entry in filtered_data]
            power_loss = [entry['power_loss'] for entry in filtered_data]
            
            # Create figure
            fig = go.Figure()
            
            # Add scatter plot
            fig.add_trace(go.Scatter(
                x=dates,
                y=power_loss,
                mode='markers+lines',
                name='Hull Roughness - Power Loss',
                line=dict(color='#0077b6', width=3),
                marker=dict(
                    size=8,
                    color='#48cae4',
                    line=dict(width=1, color='#023e8a')
                )
            ))
            
            # Add threshold lines
            fig.add_shape(
                type="line",
                x0=min(dates),
                y0=15,
                x1=max(dates),
                y1=15,
                line=dict(color="orange", width=2, dash="dash"),
                name="Average Threshold"
            )
            
            fig.add_shape(
                type="line",
                x0=min(dates),
                y0=25,
                x1=max(dates),
                y1=25,
                line=dict(color="red", width=2, dash="dash"),
                name="Poor Threshold"
            )
            
            # Add annotations
            fig.add_annotation(
                x=max(dates),
                y=15,
                text="15% - Average",
                showarrow=False,
                yshift=10,
                font=dict(color="orange", size=12)
            )
            
            fig.add_annotation(
                x=max(dates),
                y=25,
                text="25% - Poor",
                showarrow=False,
                yshift=10,
                font=dict(color="red", size=12)
            )
            
            # Layout customization
            fig.update_layout(
                title="Hull Roughness - Excess Power Trend",
                xaxis_title="Date",
                yaxis_title="Excess Power (%)",
                template="plotly_dark",
                height=500,
                margin=dict(l=40, r=40, t=40, b=40),
                uirevision=str(uuid.uuid4())
            )
            
            return fig
        except Exception as e:
            print(f"Error creating hull performance chart: {str(e)}")
            return None
    
    def _create_speed_consumption_charts(self, vessel_data):
        try:
            # Filter data
            ballast_data = []
            laden_data = []
            
            for entry in vessel_data:
                if ('speed' in entry and entry['speed'] is not None and 
                    'normalised_consumption' in entry and entry['normalised_consumption'] is not None and
                    'loading_condition' in entry and entry['loading_condition'] is not None):
                    
                    if entry['loading_condition'].lower() == 'ballast':
                        ballast_data.append({
                            'speed': entry['speed'],
                            'consumption': entry['normalised_consumption']
                        })
                    elif entry['loading_condition'].lower() == 'laden':
                        laden_data.append({
                            'speed': entry['speed'],
                            'consumption': entry['normalised_consumption']
                        })
            
            # Create charts
            ballast_chart = None
            laden_chart = None
            
            if ballast_data:
                ballast_speeds = [entry['speed'] for entry in ballast_data]
                ballast_consumptions = [entry['consumption'] for entry in ballast_data]
                
                ballast_chart = go.Figure()
                
                # Add scatter points
                ballast_chart.add_trace(go.Scatter(
                    x=ballast_speeds,
                    y=ballast_consumptions,
                    mode='markers',
                    name='Data Points',
                    marker=dict(
                        size=10,
                        color='#48cae4',
                        line=dict(width=1, color='#023e8a')
                    )
                ))
                
                # Add polynomial fit if enough data
                if len(ballast_data) > 2:
                    try:
                        # Sort data for fitting
                        sorted_indices = sorted(range(len(ballast_speeds)), key=lambda i: ballast_speeds[i])
                        speeds_sorted = [ballast_speeds[i] for i in sorted_indices]
                        consumptions_sorted = [ballast_consumptions[i] for i in sorted_indices]
                        
                        # Fit polynomial
                        coeffs = np.polyfit(speeds_sorted, consumptions_sorted, 2)
                        poly = np.poly1d(coeffs)
                        
                        # Generate points for curve
                        x_smooth = np.linspace(min(speeds_sorted), max(speeds_sorted), 100)
                        y_smooth = poly(x_smooth)
                        
                        # Add fit line
                        ballast_chart.add_trace(go.Scatter(
                            x=x_smooth,
                            y=y_smooth,
                            mode='lines',
                            name='Trend Line',
                            line=dict(color='#ff006e', width=3)
                        ))
                    except Exception as e:
                        print(f"Error fitting ballast data: {str(e)}")
                
                # Layout
                ballast_chart.update_layout(
                    title="Speed vs. Consumption - Ballast",
                    xaxis_title="Speed (knots)",
                    yaxis_title="Fuel Consumption (mt/day)",
                    template="plotly_dark",
                    height=400,
                    margin=dict(l=40, r=40, t=40, b=40),
                    uirevision=str(uuid.uuid4())
                )
            
            if laden_data:
                laden_speeds = [entry['speed'] for entry in laden_data]
                laden_consumptions = [entry['consumption'] for entry in laden_data]
                
                laden_chart = go.Figure()
                
                # Add scatter points
                laden_chart.add_trace(go.Scatter(
                    x=laden_speeds,
                    y=laden_consumptions,
                    mode='markers',
                    name='Data Points',
                    marker=dict(
                        size=10,
                        color='#48cae4',
                        line=dict(width=1, color='#023e8a')
                    )
                ))
                
                # Add polynomial fit if enough data
                if len(laden_data) > 2:
                    try:
                        # Sort data for fitting
                        sorted_indices = sorted(range(len(laden_speeds)), key=lambda i: laden_speeds[i])
                        speeds_sorted = [laden_speeds[i] for i in sorted_indices]
                        consumptions_sorted = [laden_consumptions[i] for i in sorted_indices]
                        
                        # Fit polynomial
                        coeffs = np.polyfit(speeds_sorted, consumptions_sorted, 2)
                        poly = np.poly1d(coeffs)
                        
                        # Generate points for curve
                        x_smooth = np.linspace(min(speeds_sorted), max(speeds_sorted), 100)
                        y_smooth = poly(x_smooth)
                        
                        # Add fit line
                        laden_chart.add_trace(go.Scatter(
                            x=x_smooth,
                            y=y_smooth,
                            mode='lines',
                            name='Trend Line',
                            line=dict(color='#ff006e', width=3)
                        ))
                    except Exception as e:
                        print(f"Error fitting laden data: {str(e)}")
                
                # Layout
                laden_chart.update_layout(
                    title="Speed vs. Consumption - Laden",
                    xaxis_title="Speed (knots)",
                    yaxis_title="Fuel Consumption (mt/day)",
                    template="plotly_dark",
                    height=400,
                    margin=dict(l=40, r=40, t=40, b=40),
                    uirevision=str(uuid.uuid4())
                )
            
            return ballast_chart, laden_chart
        except Exception as e:
            print(f"Error creating speed charts: {str(e)}")
            return None, None
    
    def _save_chart_as_image(self, fig):
        if fig is None:
            return None
        
        try:
            # Create a deep copy to avoid modifying the original
            fig_copy = copy.deepcopy(fig)
            
            # Update layout for better image output
            fig_copy.update_layout(
                template="plotly_white",  # Better for print
                height=600,
                width=800,
                margin=dict(l=40, r=40, t=50, b=40),
                font=dict(color='black', size=14)
            )
            
            # Create temporary file
            with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as temp:
                fig_copy.write_image(temp.name, scale=2)
                return temp.name
        except Exception as e:
            print(f"Error saving chart image: {str(e)}")
            return None
    
    def _generate_formatted_report(self, vessel_name, report_date, analyst_name, hull_metrics, speed_metrics, options, vessel_data):
        try:
            # Create a new document
            doc = Document()
            
            # Add title and vessel info
            doc.add_heading('Vessel Performance Summary', 0)
            
            # Add horizontal line
            doc.add_paragraph('_' * 80)
            
            # Add vessel information
            p = doc.add_paragraph()
            p.add_run('Vessel Name: ').bold = True
            p.add_run(vessel_name.upper())
            
            p = doc.add_paragraph()
            p.add_run('Prepared In: ').bold = True
            p.add_run(str(report_date.strftime('%B %Y')))
            
            # Create the main comparison table (3x6)
            table = doc.add_table(rows=3, cols=6)
            table.style = 'Table Grid'
            
            # Set column widths
            for i, width in enumerate([1.5, 1.0, 1.5, 1.0, 1.5, 1.0]):
                table.columns[i].width = Inches(width)
            
            # Header row
            header_cells = table.rows[0].cells
            header_cells[0].text = "Hull & Propeller"
            header_cells[1].text = hull_metrics['condition']
            header_cells[2].text = "Machinery"
            header_cells[3].text = "Good"
            header_cells[4].text = "Emissions"
            header_cells[5].text = "CII Rating - A"
            
            # Apply bold formatting to headers
            for cell in header_cells:
                for paragraph in cell.paragraphs:
                    for run in paragraph.runs:
                        run.bold = True
            
            # Icons row - if using advanced template
            if options['template_option'] == "Advanced Template (With Icons)":
                try:
                    # Add placeholder for icons - in real implementation, add actual images
                    icon_cells = table.rows[1].cells
                    icon_cells[0].text = "[Hull Icon]"
                    icon_cells[2].text = "[Machinery Icon]"
                    icon_cells[4].text = "[Emissions Icon]"
                except Exception as e:
                    print(f"Error adding icons: {str(e)}")
            
            # Values row
            value_cells = table.rows[2 if options['template_option'] == "Advanced Template (With Icons)" else 1].cells
            value_cells[1].text = f"Potential Savings\n{hull_metrics['fuel_savings']:.1f} MT/D"
            value_cells[3].text = "Potential Savings\n-"
            value_cells[5].text = "Potential Improvement\n-"
            
            doc.add_paragraph()
            
            # Create detailed metrics table
            if options['include_hull']:
                doc.add_heading('Hull & Propeller Performance', 1)
                hull_table = doc.add_table(rows=5, cols=2)
                hull_table.style = 'Table Grid'
                
                rows = hull_table.rows
                rows[0].cells[0].text = 'Additional Power Consumption'
                rows[0].cells[1].text = f"{hull_metrics['power_loss']:.1f} %"
                
                rows[1].cells[0].text = 'Potential Fuel Savings from Hull Cleaning & Propeller Polishing'
                rows[1].cells[1].text = f"{hull_metrics['fuel_savings']:.1f} MT/D"
                
                rows[2].cells[0].text = 'Hull & Propeller Condition'
                rows[2].cells[1].text = hull_metrics['condition']
                
                rows[3].cells[0].text = 'Recommendation'
                rows[3].cells[1].text = hull_metrics['recommendation']
                
                rows[4].cells[0].text = 'Forecasted Date of Hull cleaning'
                rows[4].cells[1].text = '-'
                
                doc.add_paragraph()
                
                # Hull Performance Analysis section
                doc.add_heading('Hull & Propeller Performance Analysis', 1)
                
                # Generate and add hull performance chart
                hull_chart = self._create_hull_performance_chart(vessel_data)
                
                # Only try to save and add the chart if charts are enabled in template
                if hull_chart and options['template_option'] == "Advanced Template (With Icons)":
                    try:
                        chart_path = self._save_chart_as_image(hull_chart)
                        if chart_path:
                            doc.add_picture(chart_path, width=Inches(6.0))
                            try:
                                os.unlink(chart_path)  # Clean up the temporary file
                            except:
                                pass
                    except Exception as e:
                        print(f"Error adding hull chart to document: {str(e)}")
                        doc.add_paragraph("[Hull performance chart would appear here]")
                else:
                    doc.add_paragraph("[Hull performance chart would appear here]")
                
                doc.add_paragraph()
                
                # Add notes
                doc.add_heading('Notes:', 3)
                
                note1 = doc.add_paragraph()
                note1.add_run('- ').bold = True
                note1.add_run("The vessel tends to operate with a gradual change in added resistance over time.")
                
                note2 = doc.add_paragraph()
                note2.add_run('- ').bold = True
                note2.add_run("Regular monitoring helps optimize hull cleaning intervals.")
                
                doc.add_paragraph()
            
            # Speed Consumption Profile
            if options['include_speed']:
                doc.add_heading('Speed Consumption Profile', 1)
                
                # Add speed consumption charts
                ballast_chart, laden_chart = self._create_speed_consumption_charts(vessel_data)
                
                if options['template_option'] == "Advanced Template (With Icons)" and (ballast_chart or laden_chart):
                    # Create a table for the charts
                    chart_table = doc.add_table(rows=1, cols=2)
                    chart_table.style = 'Table Grid'
                    
                    # Left column - Ballast condition
                    if ballast_chart:
                        try:
                            ballast_chart_path = self._save_chart_as_image(ballast_chart)
                            if ballast_chart_path:
                                chart_table.cell(0, 0).paragraphs[0].add_run("Ballast Condition").bold = True
                                chart_table.cell(0, 0).add_paragraph().add_run().add_picture(
                                    ballast_chart_path, width=Inches(3.0)
                                )
                                try:
                                    os.unlink(ballast_chart_path)
                                except:
                                    pass
                        except Exception as e:
                            print(f"Error adding ballast chart: {str(e)}")
                            chart_table.cell(0, 0).add_paragraph("[Ballast condition chart]")
                    else:
                        chart_table.cell(0, 0).add_paragraph("[No ballast data available]")
                    
                    # Right column - Laden condition
                    if laden_chart:
                        try:
                            laden_chart_path = self._save_chart_as_image(laden_chart)
                            if laden_chart_path:
                                chart_table.cell(0, 1).paragraphs[0].add_run("Laden Condition").bold = True
                                chart_table.cell(0, 1).add_paragraph().add_run().add_picture(
                                    laden_chart_path, width=Inches(3.0)
                                )
                                try:
                                    os.unlink(laden_chart_path)
                                except:
                                    pass
                        except Exception as e:
                            print(f"Error adding laden chart: {str(e)}")
                            chart_table.cell(0, 1).add_paragraph("[Laden condition chart]")
                    else:
                        chart_table.cell(0, 1).add_paragraph("[No laden data available]")
                else:
                    doc.add_paragraph("[Speed consumption charts would appear here]")
                
                doc.add_paragraph()
                
                # Add notes
                doc.add_heading('Notes:', 3)
                
                note1 = doc.add_paragraph()
                note1.add_run('- ').bold = True
                
                if speed_metrics['ballast_avg'] > 0:
                    note1.add_run(f"In ballast condition, the vessel shows an average fuel consumption of {speed_metrics['ballast_avg']:.1f} MT/day.")
                else:
                    note1.add_run("Insufficient data to analyze ballast condition performance.")
                
                note2 = doc.add_paragraph()
                note2.add_run('- ').bold = True
                
                if speed_metrics['laden_avg'] > 0:
                    note2.add_run(f"In laden condition, the vessel shows an average fuel consumption of {speed_metrics['laden_avg']:.1f} MT/day.")
                else:
                    note2.add_run("Insufficient data to analyze laden condition performance.")
                
                doc.add_paragraph()
            
            # Emissions section (placeholder)
            if options['include_emissions']:
                doc.add_heading('Emissions Profile', 1)
                doc.add_paragraph()
                
                p = doc.add_paragraph()
                p.add_run('CII Rating:').bold = True
                p.add_run('\n\nCII rating for 2024 of the vessel is "A" (exclusions not included). CII rating for 2024 is provisional, as it is subject to further verification and adjustments based on exclusion data.')
                
                doc.add_paragraph("[Emissions chart would appear here]")
                doc.add_paragraph()
            
            # Machinery section (placeholder)
            if options['include_machinery']:
                doc.add_heading('Main Engine Performance', 1)
                machinery_table = doc.add_table(rows=3, cols=2)
                machinery_table.style = 'Table Grid'
                
                rows = machinery_table.rows
                rows[0].cells[0].text = 'Average ME SFOC'
                rows[0].cells[1].text = "167.12 g/KWhr at 81% Load (Placeholder)"
                
                rows[1].cells[0].text = 'ME Recommendation'
                rows[1].cells[1].text = "ME Performance is within Acceptable Range"
                
                rows[2].cells[0].text = 'Potential Fuel Saving from ME'
                rows[2].cells[1].text = "-"
                
                doc.add_paragraph()
                
                doc.add_heading('Auxiliaries Performance', 2)
                aux_table = doc.add_table(rows=2, cols=2)
                aux_table.style = 'Table Grid'
                
                rows = aux_table.rows
                rows[0].cells[0].text = 'Excess Boiler Consumption (last 6 month)'
                rows[0].cells[1].text = "16.7 MT (Placeholder)"
                
                rows[1].cells[0].text = 'Redundant AE Hours (last 6 month)'
                rows[1].cells[1].text = "-"
                
                doc.add_paragraph()
            
            # Appendix
            doc.add_heading('Appendix', 1)
            doc.add_heading('General Conditions', 2)
            
            appendix = doc.add_paragraph()
            appendix.add_run('- ').bold = True
            appendix.add_run("Analysis Period is Last Six Months or the after the Last Event which ever is later")
            
            appendix2 = doc.add_paragraph()
            appendix2.add_run('- ').bold = True
            appendix2.add_run("Days with Good Weather (BF<=4) are considered for analysis.")
            
            appendix3 = doc.add_paragraph()
            appendix3.add_run('- ').bold = True
            appendix3.add_run("Days with Steaming hrs greater than 17 considered for analysis.")
            
            appendix4 = doc.add_paragraph()
            appendix4.add_run('- ').bold = True
            appendix4.add_run("Data is compared with Original Sea Trial")
            
            # Hull Performance rating criteria
            doc.add_heading('Hull Performance', 2)
            hull_app1 = doc.add_paragraph()
            hull_app1.add_run('- ').bold = True
            hull_app1.add_run("Excess Power < 15 % -- Rating Good")
            
            hull_app2 = doc.add_paragraph()
            hull_app2.add_run('- ').bold = True
            hull_app2.add_run("15 < Excess Power < 25 % -- Rating Average")
            
            hull_app3 = doc.add_paragraph()
            hull_app3.add_run('- ').bold = True
            hull_app3.add_run("Excess Power > 25 % -- Rating Poor")
            
            # Machinery Performance rating criteria (if included)
            if options['include_machinery']:
                doc.add_heading('Machinery Performance', 2)
                mach_app1 = doc.add_paragraph()
                mach_app1.add_run('- ').bold = True
                mach_app1.add_run("SFOC(Grms/kW.hr) within +/- 10 from Shop test condition are considered as \"Good\"")
                
                mach_app2 = doc.add_paragraph()
                mach_app2.add_run('- ').bold = True
                mach_app2.add_run("SFOC(Grms/kW.hr) Greater than 10 and less than 20 are considered as \"Average\"")
                
                mach_app3 = doc.add_paragraph()
                mach_app3.add_run('- ').bold = True
                mach_app3.add_run("SFOC(Grms/kW.hr) Above 20 are considered as \"Poor\"")
            
            # Auxiliaries performance criteria (if included)
            if options['include_machinery']:
                doc.add_heading('Auxiliaries Performance', 2)
                aux_app = doc.add_paragraph()
                aux_app.add_run('- ').bold = True
                aux_app.add_run("Excess boiler consumption refers to the amount of fuel oil used by the boiler during the Noon at Sea. Any consideration to operational requirements not given, due to the lack of evidence of such requirement in Noon Reports.")
            
            # Speed consumption criteria (if included)
            if options['include_speed']:
                doc.add_heading('Speed Consumption Performance', 2)
                speed_app = doc.add_paragraph()
                speed_app.add_run('- ').bold = True
                speed_app.add_run("Analysis carried out using regression techniques.")
            
            # Convert the document to bytes
            docx_file = BytesIO()
            doc.save(docx_file)
            docx_file.seek(0)
            
            return docx_file.getvalue()
        except Exception as e:
            # Log the full error
            print(f"Error generating report: {str(e)}")
            print(traceback.format_exc())
            
            # Create a simple error report
            error_doc = Document()
            error_doc.add_heading('ERROR: Report Generation Failed', 0)
            error_doc.add_paragraph(f"Vessel Name: {vessel_name}")
            error_doc.add_paragraph(f"Date: {report_date}")
            
            error_doc.add_heading('Error Details', 1)
            error_doc.add_paragraph(f"An error occurred during report generation: {str(e)}")
            
            error_doc.add_heading('Basic Report Content (No Formatting)', 1)
            
            # Add basic content without fancy formatting
            error_doc.add_paragraph(f"Hull Condition: {hull_metrics['condition']}")
            error_doc.add_paragraph(f"Power Loss: {hull_metrics['power_loss']:.1f}%")
            error_doc.add_paragraph(f"Potential Savings: {hull_metrics['fuel_savings']:.1f} MT/D")
            
            if speed_metrics['ballast_avg'] > 0:
                error_doc.add_paragraph(f"Ballast Consumption: {speed_metrics['ballast_avg']:.1f} MT/day")
            
            if speed_metrics['laden_avg'] > 0:
                error_doc.add_paragraph(f"Laden Consumption: {speed_metrics['laden_avg']:.1f} MT/day")
            
            # Save the error document
            error_file = BytesIO()
            error_doc.save(error_file)
            error_file.seek(0)
            
            return error_file.getvalue()
