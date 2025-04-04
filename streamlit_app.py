
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime, timedelta
from docx import Document
from docx.shared import Inches
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="openpyxl")

def create_neon_color_gradient(dates):
    norm = plt.Normalize(dates.min().toordinal(), dates.max().toordinal())
    cmap = plt.get_cmap('plasma')
    return [cmap(norm(date.toordinal())) for date in dates]

def process_vessel_data(data, vessel_name):
    vessel_data = data[data['VESSEL_NAME'].str.lower() == vessel_name.lower()].copy()
    vessel_data['REPORT_DATE'] = pd.to_datetime(vessel_data['REPORT_DATE'])
    return vessel_data

def main():
    st.set_page_config(page_title="Vessel Performance Analysis", layout="wide")

    st.title("🚢 Vessel Performance Analysis Dashboard")

    # File uploaders in the sidebar
    st.sidebar.header("Data Input")
    hull_perf_file = st.sidebar.file_uploader("Hull Performance Data (CSV)", type="csv")
    vessel_coeff_file = st.sidebar.file_uploader("Vessel Performance Coefficients (CSV)", type="csv")
    sea_trial_file = st.sidebar.file_uploader("Sea Trial Data (Excel)", type="xlsx")
    dd_dates_file = st.sidebar.file_uploader("DD Dates (Excel)", type="xlsx")

    if all([hull_perf_file, vessel_coeff_file, sea_trial_file, dd_dates_file]):
        try:
            # Load data
            hull_data = pd.read_csv(hull_perf_file)
            coeff_data = pd.read_csv(vessel_coeff_file)
            sea_trial_data = pd.read_excel(sea_trial_file)
            dd_dates_data = pd.read_excel(dd_dates_file)

            # Get list of vessels
            vessels = sorted(hull_data['VESSEL_NAME'].unique())

            # Vessel selection
            selected_vessel = st.selectbox("Select Vessel for Analysis", vessels)

            if st.button("Analyze Vessel"):
                st.subheader(f"Analysis for {selected_vessel}")

                # Process vessel data
                vessel_data = process_vessel_data(hull_data, selected_vessel)

                # Create two columns for metrics
                col1, col2 = st.columns(2)

                with col1:
                    st.metric(
                        "Average Power Loss", 
                        f"{vessel_data['HULL_ROUGHNESS_POWER_LOSS'].mean():.2f}%"
                    )

                with col2:
                    st.metric(
                        "Average Fuel Consumption", 
                        f"{vessel_data['HULL_EXCESS_FUEL_OIL_MTD'].mean():.2f} MT/day"
                    )

                # Create performance trend chart
                st.subheader("Performance Trends")
                fig, ax = plt.subplots(figsize=(10, 6))

                # Plot power loss trend
                ax.scatter(
                    vessel_data['REPORT_DATE'],
                    vessel_data['HULL_ROUGHNESS_POWER_LOSS'],
                    c=create_neon_color_gradient(vessel_data['REPORT_DATE']),
                    alpha=0.6
                )

                ax.set_xlabel('Date')
                ax.set_ylabel('Hull Roughness Power Loss (%)')
                ax.set_title(f'{selected_vessel} - Hull Performance Trend')
                ax.grid(True, alpha=0.3)
                plt.xticks(rotation=45)

                st.pyplot(fig)
                plt.close()

                # Display recent data
                st.subheader("Recent Performance Data")
                recent_data = vessel_data.sort_values('REPORT_DATE', ascending=False).head(10)
                st.dataframe(
                    recent_data[['REPORT_DATE', 'HULL_ROUGHNESS_POWER_LOSS', 'HULL_EXCESS_FUEL_OIL_MTD']]
                )

        except Exception as e:
            st.error(f"An error occurred: {str(e)}")
    else:
        st.info("Please upload all required data files to begin analysis")

if __name__ == "__main__":
    main()
