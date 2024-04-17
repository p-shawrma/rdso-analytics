import streamlit as st
import pandas as pd
import psycopg2
from datetime import datetime

def create_connection():
    return psycopg2.connect(
        dbname="postgres",
        user="postgres.kfuizzxktmneperhsekb",  # Replace with your actual username
        password="RDSO_Analytics_Change@2015",  # Replace with your actual password
        host="aws-0-ap-southeast-1.pooler.supabase.com",  # Replace with your actual host
        port="5432"
    )

def fetch_data(start_date, end_date):
    with create_connection() as conn:
        query = """
        SELECT *, ("Cell_Voltage_1_(V)" + "Cell_Voltage_2_(V)" + ... + "Cell_Voltage_35_(V)") AS "Total_Voltage"
        FROM public.custom_report_rdso
        WHERE created_at BETWEEN %s AND %s;
        """
        df = pd.read_sql_query(query, conn, params=[start_date, end_date])
    return df

def main():
    st.set_page_config(layout="wide", page_title="Battery Discharge Analysis")

    with st.sidebar:
        st.title("Filter Settings")
        start_date = st.date_input("Start date", datetime.now().date())
        end_date = st.date_input("End date", datetime.now().date())
        if start_date > end_date:
            st.error("End date must be after start date.")
        fetch_button = st.button("Fetch Data")

    if fetch_button:
        df = fetch_data(start_date, end_date)
        df['timestamp'] = pd.to_datetime(df['created_at'])
        df['time_diff'] = df['timestamp'].diff().dt.total_seconds()
        df['discharge'] = df['Battery_Pack_Current(A)'] < 0
        df['cycle'] = df['discharge'].cumsum()  # Simple cycle counting
        df['smoothed_voltage'] = df['Total_Voltage'].rolling(window=5, center=True).mean()
        df['voltage_drop'] = df['smoothed_voltage'].diff() < 0

        st.dataframe(df[['timestamp', 'Battery_Pack_Current(A)', 'Total_Voltage', 'smoothed_voltage', 'discharge', 'cycle']])

if __name__ == "__main__":
    main()
