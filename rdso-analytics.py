import streamlit as st
import pandas as pd
import psycopg2
from datetime import datetime, timedelta

# Function to create a database connection using psycopg2
@st.cache(allow_output_mutation=True, ttl=600, show_spinner=False)
def get_data(start_date, end_date):
    user = "postgres.kfuizzxktmneperhsekb"
    password = "RDSO_Analytics_Change@2015"
    host = "aws-0-ap-southeast-1.pooler.supabase.com"
    port = "5432"
    dbname = "postgres"
    
    with psycopg2.connect(
        dbname=dbname,
        user=user,
        password=password,
        host=host,
        port=port
    ) as conn:
        cursor = conn.cursor()
        # Using direct string formatting to ensure full-day coverage
        query = f"""
        SELECT *
        FROM public.custom_report_rdso
        WHERE created_at BETWEEN '{start_date.strftime('%Y-%m-%d')} 00:00:00' AND '{end_date.strftime('%Y-%m-%d')} 23:59:59';
        """
        cursor.execute(query)
        records = cursor.fetchall()
        columns = [desc[0] for desc in cursor.description]
        df = pd.DataFrame(records, columns=columns)
        
        # Log the query and the number of rows fetched
        print(f"Executed query: {query}")
        print(f"Number of rows fetched: {len(df)}")

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

    # Sidebar for date input
    with st.sidebar:
        st.title("Filter Settings")
        start_date = st.date_input("Start Date", datetime.now().date() - timedelta(days=7))
        end_date = st.date_input("End Date", datetime.now().date())

        if start_date > end_date:
            st.error("End date must be after start date.")
        fetch_button = st.button("Fetch Data")

    # Display data without cycle filter
    if fetch_button:
        df = get_data(start_date, end_date)
        if not df.empty:
            processed_df = process_data(df)
            st.write("Data Overview:")
            st.dataframe(processed_df)  # Display the entire dataframe
        else:
            st.write("No data found for the selected date range.")

if __name__ == "__main__":
    main()
