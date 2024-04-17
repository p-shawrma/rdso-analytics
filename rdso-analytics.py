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
        SELECT created_at, "Battery_Pack_Voltage(V)", "Battery_Pack_Current(A)",
               "Max_Cell_Voltage_(V)", "Min_Cell_Voltage_(V)",
               "Max_Cell_Temp_(C)", "Min_Cell_Temp_(C)", "SOC(%)"
        FROM public.custom_report_rdso
        WHERE created_at BETWEEN %s AND %s;
        """
        # Convert dates to datetime at the start of the start_date and the end of the end_date
        start_datetime = datetime.combine(start_date, datetime.min.time())
        end_datetime = datetime.combine(end_date, datetime.max.time())
        
        # Debug: print or log the datetime values
        print("Start datetime:", start_datetime)
        print("End datetime:", end_datetime)

        df = pd.read_sql_query(query, conn, params=[start_datetime, end_datetime])
    return df

# Example usage within a Streamlit app:
import streamlit as st

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
        if not df.empty:
            st.dataframe(df)
        else:
            st.write("No data found for the selected date range.")

if __name__ == "__main__":
    main()
