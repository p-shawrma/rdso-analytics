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

# Function to fetch data from the database
def fetch_data(start_date, end_date):
    with create_connection() as conn:
        query = """
        SELECT * FROM public.custom_report_rdso
        WHERE created_at BETWEEN %s AND %s;
        """
        # Convert dates to datetime at the start of the start_date and the end of the end_date
        start_datetime = datetime.combine(start_date, datetime.min.time())
        end_datetime = datetime.combine(end_date, datetime.max.time())

        # Print the datetime parameters to debug
        print("Fetching data between:", start_datetime, "and", end_datetime)

        # Fetch data using a DataFrame
        df = pd.read_sql_query(query, conn, params=[start_datetime, end_datetime])
    return df

# Main function to run the Streamlit app
def main():
    st.set_page_config(layout="wide", page_title="Battery Discharge Analysis Dashboard")

    # Sidebar for user inputs
    with st.sidebar:
        st.title("Data Filter Settings")
        start_date = st.date_input("Start date", datetime.now().date() - timedelta(days=7))
        end_date = st.date_input("End date", datetime.now().date())
        if start_date > end_date:
            st.error("End date must be after start date.")
        fetch_button = st.button("Fetch Data")

    # Data fetching and display
    if fetch_button:
        df = fetch_data(start_date, end_date)
        if not df.empty:
            st.write("Data fetched successfully.")
            st.dataframe(df)
        else:
            st.error("No data found for the selected date range.")

if __name__ == "__main__":
    main()
