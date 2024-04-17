import streamlit as st
import pandas as pd
import sqlalchemy
from datetime import datetime, timedelta

# Define a function to create a SQLAlchemy database connection
def create_engine():
    database_url = "postgresql://postgres.kfuizzxktmneperhsekb:RDSO_Analytics_Change@2015@aws-0-ap-southeast-1.pooler.supabase.com:5432/postgres"
    engine = sqlalchemy.create_engine(database_url)
    return engine

# Updated caching function using st.cache_data
@st.cache_data(ttl=600, suppress_st_warning=True)
def get_data(start_date, end_date, engine):
    # Formatted SQL query to fetch all relevant data between two dates
    query = """
    SELECT *
    FROM public.custom_report_rdso
    WHERE created_at BETWEEN %s AND %s;
    """
    df = pd.read_sql_query(query, engine, params=[start_date, end_date])
    return df

def process_data(df):
    df['timestamp'] = pd.to_datetime(df['created_at'])
    df = df.sort_values(by='timestamp')
    df['time_diff'] = df['timestamp'].diff().dt.total_seconds()

    # Identify discharge periods and positive pulses
    df['discharge'] = df['Battery_Pack_Current(A)'] < 0
    df['pos_current'] = df['Battery_Pack_Current(A)'] > 0
    df['pos_duration'] = df['pos_current'] * df['time_diff']
    df['pos_pulse'] = df['pos_duration'].rolling(window=3).sum()

    df['pos_pulse'] = (df['pos_pulse'] < 30) & df['pos_current']

    # Smoothing voltage to identify voltage drops
    df['smoothed_voltage'] = df['Battery_Pack_Voltage(V)'].rolling(window=5, center=True).mean()
    df['voltage_change'] = df['smoothed_voltage'].diff()
    df['discharge_start'] = (df['discharge'] == True) & (df['discharge'].shift(1) == False)
    df['discharge_end'] = (df['discharge'] == False) & (df['discharge'].shift(1) == True)

    # Tagging cycles
    df['cycle'] = df['discharge_start'].cumsum()

    return df

def main():
    st.set_page_config(layout="wide", page_title="Battery Discharge Analysis")

    # Sidebar for date input and cycle filtering
    with st.sidebar:
        st.title("Filter Settings")
        start_date = st.date_input("Start Date", datetime.now().date() - timedelta(days=7))
        end_date = st.date_input("End Date", datetime.now().date())

        if start_date > end_date:
            st.error("End date must be after start date.")
        fetch_button = st.button("Fetch Data")

    if fetch_button:
        engine = create_engine()
        df = get_data(start_date, end_date, engine)
        if not df.empty:
            processed_df = process_data(df)
            cycle_number = st.sidebar.selectbox("Select Discharge Cycle", processed_df['cycle'].unique())
            filtered_data = processed_df[processed_df['cycle'] == cycle_number]
            st.dataframe(filtered_data[['timestamp', 'Battery_Pack_Current(A)', 'Battery_Pack_Voltage(V)', 'smoothed_voltage', 'discharge', 'pos_pulse', 'cycle']])
        else:
            st.write("No data found for the selected date range.")

if __name__ == "__main__":
    main()
