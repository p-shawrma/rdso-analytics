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

    # Smoothing current and voltage with exponential moving average
    forgetting_factor = 3
    alpha = 2 / (forgetting_factor + 1)
    df['smoothed_current'] = df['Battery_Pack_Current(A)'].ewm(alpha=alpha).mean()
    df['smoothed_voltage'] = df['Battery_Pack_Voltage(V)'].ewm(alpha=alpha).mean()

    # Define conditions for charging, discharging, and idle states
    conditions = [
        df['smoothed_current'] > 0,  # Charging condition
        df['smoothed_current'] < 0,  # Discharging condition
        df['smoothed_current'] == 0  # Idle condition
    ]

    # Define the choice for each condition
    choices = ['charge', 'discharge', 'idle']

    # Use np.select to apply conditions and choices to the dataframe
    df['state'] = np.select(conditions, choices, default='idle')

    # Identifying continuous stints
    df['stint_id'] = (df['state'] != df['state'].shift()).cumsum()

    # Identifying the duration of each stint
    df['stint_duration'] = df.groupby('stint_id')['time_diff'].transform('sum')

    # Determining if the stint is valid based on its duration
    df['stint_valid'] = df['stint_duration'] > 60

    # Creating a filtered dataframe with valid stints only
    valid_stints_df = df[df['stint_valid']].copy()

    # Counting valid discharge and charge stints
    valid_stints_df['discharge_stint_count'] = (valid_stints_df['state'] == 'discharge').cumsum()
    valid_stints_df['charge_stint_count'] = (valid_stints_df['state'] == 'charge').cumsum()

    return valid_stints_df

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
