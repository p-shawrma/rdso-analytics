import streamlit as st
import pandas as pd
import psycopg2
import numpy as np
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import plotly.graph_objects as go


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
    
    # Handle the first time_diff NaN
    df['time_diff'].fillna(method='bfill', inplace=True)

    # Base alpha for an expected time difference, e.g., 10 seconds
    base_time_diff = 10  # Base time difference in seconds
    base_alpha = 0.1    # Base alpha for smoothing

    # Adjust alpha based on actual time difference
    # Protect against division by zero just in case
    df['alpha'] = df['time_diff'].apply(lambda x: base_alpha / x * base_time_diff if x > 0 else base_alpha)
    df['alpha'] = df['alpha'].clip(upper=0.3)  # Ensure alpha does not exceed 1

    # Initialize the first current to the first actual current reading
    ema_current = df['Battery_Pack_Current(A)'].iloc[0]
    smoothed_currents = [ema_current]

    # Apply dynamic EMA
    for i in range(1, len(df)):
        alpha = df['alpha'].iloc[i]
        current = df['Battery_Pack_Current(A)'].iloc[i]
        ema_current = ema_current * (1 - alpha) + current * alpha
        smoothed_currents.append(ema_current)

    df['smoothed_current'] = smoothed_currents

    # Static EMA for voltage
    df['smoothed_voltage'] = df['Battery_Pack_Voltage(V)'].ewm(alpha=base_alpha).mean()

    # Define conditions and choices for states
    epsilon = 0.2
    conditions = [
        df['smoothed_current'] > epsilon,   # Charging condition
        df['smoothed_current'] < -epsilon,  # Discharging condition
        abs(df['smoothed_current']) <= epsilon  # Idle condition
    ]
    choices = ['charge', 'discharge', 'idle']
    df['state'] = np.select(conditions, choices, default='idle')

    return df

def plot_data(df):
    # Create traces for the smoothed current and voltage
    trace1 = go.Scatter(
        x=df['timestamp'],
        y=df['smoothed_current'],
        mode='lines',
        name='Smoothed Current (A)',
        line=dict(color='red')
    )
    
    trace2 = go.Scatter(
        x=df['timestamp'],
        y=df['smoothed_voltage'],
        mode='lines',
        name='Smoothed Voltage (V)',
        line=dict(color='blue'),
        yaxis='y2'
    )
    
    # Layout with dual-axis configuration
    layout = go.Layout(
        title='Smoothed Current and Voltage Over Time',
        xaxis=dict(title='Timestamp'),
        yaxis=dict(title='Smoothed Current (A)'),
        yaxis2=dict(
            title='Smoothed Voltage (V)',
            overlaying='y',
            side='right'
        ),
        autosize=True,  # Enable autosize to fill the container width
        template='plotly_white'  # Optional: use a Plotly theme for nicer styling
    )
    
    # Combine traces and layout into a figure
    fig = go.Figure(data=[trace1, trace2], layout=layout)
    fig.update_layout(
        autosize=True,  # Ensures that plot size adjusts based on the container size
        margin=dict(l=20, r=20, t=40, b=20),  # Reduces margins to make use of available space
    )
    return fig

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

    if fetch_button:
        df = get_data(start_date, end_date)
        if not df.empty:
            processed_df = process_data(df)
            st.write("Data Overview:")
            st.dataframe(processed_df)  # Display the entire dataframe
            fig = plot_data(processed_df)
            st.plotly_chart(fig, use_container_width=True)  # Ensures that the plot stretches to the full container width
        else:
            st.write("No data found for the selected date range.")

if __name__ == "__main__":
    main()
