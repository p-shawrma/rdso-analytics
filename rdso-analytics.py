import streamlit as st
import pandas as pd
import psycopg2
from datetime import datetime, timedelta

# Function to create a database connection using psycopg2
@st.cache(allow_output_mutation=True, ttl=600, show_spinner=False)
def get_data(start_date, end_date):
    # Connection parameters
    user = "postgres.kfuizzxktmneperhsekb"
    password = "RDSO_Analytics_Change@2015"
    host = "aws-0-ap-southeast-1.pooler.supabase.com"
    port = "5432"
    dbname = "postgres"
    
    # Connect to the database
    conn = psycopg2.connect(
        dbname=dbname,
        user=user,
        password=password,
        host=host,
        port=port
    )
    
    # Formatted SQL query to fetch all relevant data between two dates
    query = """
    SELECT *
    FROM public.custom_report_rdso
    WHERE created_at BETWEEN %s AND %s;
    """
    
    # Convert dates to strings to ensure compatibility with the SQL query
    start_str = start_date.strftime('%Y-%m-%d %H:%M:%S')
    end_str = end_date.strftime('%Y-%m-%d %H:%M:%S')
    
    # Execute the query and fetch data
    df = pd.read_sql_query(query, conn, params=(start_str, end_str))
    
    # Close the database connection
    conn.close()
    
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
        df = get_data(engine, start_date, end_date)
        if not df.empty:
            processed_df = process_data(df)
            cycle_number = st.sidebar.selectbox("Select Discharge Cycle", processed_df['cycle'].unique())
            filtered_data = processed_df[processed_df['cycle'] == cycle_number]
            st.dataframe(filtered_data[['timestamp', 'Battery_Pack_Current(A)', 'Battery_Pack_Voltage(V)', 'smoothed_voltage', 'discharge', 'pos_pulse', 'cycle']])
        else:
            st.write("No data found for the selected date range.")

if __name__ == "__main__":
    main()
