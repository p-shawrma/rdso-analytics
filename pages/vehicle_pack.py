import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import plotly.graph_objects as go
import clickhouse_connect
from pygwalker.api.streamlit import StreamlitRenderer

# ClickHouse connection details
ch_host = 'a84a1hn9ig.ap-south-1.aws.clickhouse.cloud'
ch_user = 'default'
ch_password = 'dKd.Y9kFMv06x'
ch_database = 'telematics'

# Function to create a ClickHouse client
def create_client():
    return clickhouse_connect.get_client(
        host=ch_host,
        user=ch_user,
        password=ch_password,
        database=ch_database,
        secure=True
    )

# Function to fetch all model numbers and device dates
@st.cache_data(ttl=6000)
def fetch_model_numbers_and_dates():
    client = create_client()
    query = "SELECT DISTINCT Model_Numb, toStartOfDay(DeviceDate) as DeviceDate FROM custom_tracking"
    result = client.query(query)
    df = pd.DataFrame(result.result_rows, columns=result.column_names)
    return df

# Function to fetch data based on filters
@st.cache_data(ttl=6000)
def fetch_data(model_numbers, start_date, end_date):
    client = create_client()
    model_numbers_str = ','.join([f"'{model}'" for model in model_numbers])
    query = f"""
    SELECT *
    FROM custom_tracking
    WHERE Model_Numb IN ({model_numbers_str}) AND DeviceDate BETWEEN '{start_date}' AND '{end_date}';
    """
    result = client.query(query)
    df = pd.DataFrame(result.result_rows, columns=result.column_names)
    return df

# Function to process data
def process_data(df):
    df['timestamp'] = pd.to_datetime(df['DeviceDate'])
    df = df.sort_values(by='timestamp')
    df['time_diff'] = df['timestamp'].diff().dt.total_seconds()
    
    df['time_diff'].fillna(method='bfill', inplace=True)

    base_time_diff = 10
    base_alpha = 0.33

    df['alpha'] = df['time_diff'].apply(lambda x: base_alpha / x * base_time_diff if x > 0 else base_alpha)
    df['alpha'] = df['alpha'].clip(upper=0.66)

    ema_current = df['BM_BattCurrent'].iloc[0]
    smoothed_currents = [ema_current]

    for i in range(1, len(df)):
        alpha = df['alpha'].iloc[i]
        current = df['BM_BattCurrent'].iloc[i]
        ema_current = ema_current * (1 - alpha) + current * alpha
        smoothed_currents.append(ema_current)

    df['Fitted_Current(A)'] = smoothed_currents
    df['Fitted_Voltage(V)'] = df['BM_BattVoltage'].ewm(alpha=base_alpha).mean()

    df['voltage_increase'] = df['Fitted_Voltage(V)'].diff() > 0.01
    df['soc_increase'] = df['BM_SocPercent'].diff() > 0.01

    cell_temp_columns = [col for col in df.columns if 'Cell_Temperature' in col]
    df['Pack_Temperature_(C)'] = df[cell_temp_columns].mean(axis=1)

    epsilon = 0.05
    conditions = [
        (df['voltage_increase'] | df['soc_increase']) & (df['Fitted_Current(A)'] > epsilon),
        (df['Fitted_Current(A)'] < -epsilon) & (~df['voltage_increase']),
        abs(df['Fitted_Current(A)']) <= epsilon
    ]
    choices = ['charge', 'discharge', 'idle']
    df['state'] = np.select(conditions, choices, default='idle')

    df['state_change'] = (df['state'] != df['state'].shift(1)).cumsum()
    grp = df.groupby('state_change')
    df['state_duration'] = grp['timestamp'].transform(lambda x: (x.max() - x.min()).total_seconds())
    df['filtered_state'] = np.where(df['state_duration'] > 5, df['state'], np.nan)
    df['filtered_state'].fillna(method='bfill', inplace=True)

    state_mapping = {'charge': 0, 'discharge': 1, 'idle': 2}
    df['step_type'] = df['filtered_state'].map(state_mapping)

    return df

def calculate_percentile(n):
    def percentile_(x):
        return np.percentile(x, n)
    percentile_.__name__ = 'percentile_%s' % n
    return percentile_

def process_grouped_data(df):
    grouped = df.groupby((df['filtered_state'] != df['filtered_state'].shift()).cumsum())
    result = grouped.agg(
        start_timestamp=('timestamp', 'min'),
        end_timestamp=('timestamp', 'max'),
        step_type=('filtered_state', 'first'),
        duration_minutes=('timestamp', lambda x: (x.max() - x.min()).total_seconds() / 60),
        soc_start=('BM_SocPercent', 'first'),
        soc_end=('BM_SocPercent', 'last'),
        voltage_start=('Fitted_Voltage(V)', 'first'),
        voltage_end=('Fitted_Voltage(V)', 'last'),
        average_current=('Fitted_Current(A)', 'mean'),
        median_current=('Fitted_Current(A)', 'median'),
        min_current=('Fitted_Current(A)', calculate_percentile(10)),
        max_current=('Fitted_Current(A)', calculate_percentile(90)),
        current_25th=('Fitted_Current(A)', calculate_percentile(25)),
        current_75th=('Fitted_Current(A)', calculate_percentile(75)),
        median_max_cell_temperature=('Max_monomer_temperature', 'median'),
        median_min_cell_temperature=('Min_monomer_temperature', 'median'),
        median_pack_temperature=('Pack_Temperature_(C)', 'median')
    )

    result['date'] = result['start_timestamp'].dt.date
    result['change_in_soc'] = result['soc_end'] - result['soc_start']

    columns_ordered = ['date', 'start_timestamp', 'end_timestamp', 'step_type', 'duration_minutes',
                       'soc_start', 'soc_end', 'change_in_soc', 'voltage_start', 'voltage_end',
                       'average_current', 'median_current', 'min_current', 'max_current', 'current_25th',
                       'current_75th', 'median_max_cell_temperature', 'median_min_cell_temperature', 'median_pack_temperature']

    result = result.reindex(columns=columns_ordered)
    
    return result

def apply_filters(df):
    step_types = df['step_type'].unique().tolist()
    selected_types = st.sidebar.multiselect('Select Step Types', step_types, default=step_types)
    filtered_df = df[df['step_type'].isin(selected_types)]
    return filtered_df

def plot_current_voltage(df):
    trace1 = go.Scatter(x=df['timestamp'], y=df['Fitted_Current(A)'], mode='lines', name='Current (A)', line=dict(color='red'))
    trace2 = go.Scatter(x=df['timestamp'], y=df['Fitted_Voltage(V)'], mode='lines', name='Voltage (V)', yaxis='y2', line=dict(color='blue'))
    layout = go.Layout(
        title='Current and Voltage Over Time',
        xaxis=dict(title='Timestamp'),
        yaxis=dict(title='Current (A)'),
        yaxis2=dict(title='Voltage (V)', overlaying='y', side='right'),
        autosize=True,
        template='plotly_white'
    )
    fig = go.Figure(data=[trace1, trace2], layout=layout)
    fig.update_layout(autosize=True, margin=dict(l=20, r=20, t=40, b=20))
    return fig

def plot_current_soc(df):
    trace1 = go.Scatter(x=df['timestamp'], y=df['Fitted_Current(A)'], mode='lines', name='Current (A)', line=dict(color='red'))
    trace2 = go.Scatter(x=df['timestamp'], y=df['BM_SocPercent'], mode='lines', name='SOC (%)', yaxis='y2', line=dict(color='green'))
    layout = go.Layout(
        title='Current and SOC Over Time',
        xaxis=dict(title='Timestamp'),
        yaxis=dict(title='Current (A)'),
        yaxis2=dict(title='SOC (%)', overlaying='y', side='right')
    )
    fig = go.Figure(data=[trace1, trace2], layout=layout)
    return fig

def plot_voltage_soc(df):
    trace1 = go.Scatter(x=df['timestamp'], y=df['Fitted_Voltage(V)'], mode='lines', name='Voltage (V)', line=dict(color='blue'))
    trace2 = go.Scatter(x=df['timestamp'], y=df['BM_SocPercent'], mode='lines', name='SOC (%)', yaxis='y2', line=dict(color='green'))
    layout = go.Layout(
        title='Voltage and SOC Over Time',
        xaxis=dict(title='Timestamp'),
        yaxis=dict(title='Voltage (V)'),
        yaxis2=dict(title='SOC (%)', overlaying='y', side='right')
    )
    fig = go.Figure(data=[trace1, trace2], layout=layout)
    return fig

def plot_temp(df):
    trace1 = go.Scatter(x=df['timestamp'], y=df['Max_monomer_temperature'], mode='lines', name='Max Cell Temp (C)', line=dict(color='orange'))
    trace2 = go.Scatter(x=df['timestamp'], y=df['Min_monomer_temperature'], mode='lines', name='Min Cell Temp (C)', line=dict(color='purple'))
    layout = go.Layout(title='Max and Min Cell Temperatures Over Time', xaxis=dict(title='Timestamp'), yaxis=dict(title='Temperature (C)'))
    fig = go.Figure(data=[trace1, trace2], layout=layout)
    return fig

def calculate_midpoint(row):
    return row['start_timestamp'] + (row['end_timestamp'] - row['start_timestamp']) / 2

def plot_discharge_currents(df):
    df['mid_timestamp'] = df.apply(calculate_midpoint, axis=1)
    discharge_df = df[df['step_type'] == 'discharge']
    fig = go.Figure(data=[go.Candlestick(
        x=discharge_df['mid_timestamp'],
        open=discharge_df['current_25th'],
        high=discharge_df['max_current'],
        low=discharge_df['min_current'],
        close=discharge_df['current_75th'],
        increasing_line_color='green', decreasing_line_color='red'
    )])
    fig.update_layout(title='Current Distribution in Discharge States', xaxis_title='Time', yaxis_title='Current (A)', xaxis_rangeslider_visible=False)
    return fig

def create_day_wise_summary(df):
    discharge = df[df['step_type'] == 'discharge']
    charge = df[df['step_type'] == 'charge']

    discharge_summary = discharge.groupby('date').agg({
        'change_in_soc': 'sum',
        'duration_minutes': ['sum', 'min', 'max', 'median', calculate_percentile(25), calculate_percentile(75)]
    })

    charge_summary = charge.groupby('date').agg({
        'change_in_soc': 'sum'
    })

    discharge_summary.columns = ['_'.join(col).strip() for col in discharge_summary.columns.values]
    charge_summary.columns = ['total_charge_soc']

    day_wise_summary = pd.merge(discharge_summary, charge_summary, on='date', how='outer')
    day_wise_summary.rename(columns={
        'change_in_soc_sum': 'total_discharge_soc',
        'duration_minutes_sum': 'total_discharge_time',
        'duration_minutes_min': 'discharge_time_min',
        'duration_minutes_max': 'discharge_time_max',
        'duration_minutes_median': 'discharge_time_median',
        'duration_minutes_percentile_25': 'discharge_time_25th',
        'duration_minutes_percentile_75': 'discharge_time_75th'
    }, inplace=True)

    return day_wise_summary

def plot_discharge_duration_candlestick(df):
    fig = go.Figure(data=[go.Candlestick(
        x=df.index,
        open=df['discharge_time_25th'],
        high=df['discharge_time_max'],
        low=df['discharge_time_min'],
        close=df['discharge_time_75th'],
        increasing_line_color='green', decreasing_line_color='red'
    )])
    fig.update_layout(title='Candlestick Plot of Discharge Durations', xaxis_title='Date', yaxis_title='Duration in Minutes', xaxis_rangeslider_visible=False)
    return fig

def main():
    st.set_page_config(layout="wide", page_title="Battery Discharge Analysis")

    with st.sidebar:
        st.title("Filter Settings")
        model_numbers_and_dates = fetch_model_numbers_and_dates()
        model_numbers = model_numbers_and_dates['Model_Numb'].unique().tolist()
        selected_model_numbers = st.multiselect('Select Model Numbers', model_numbers, default=model_numbers)

        date_range = model_numbers_and_dates['DeviceDate'].unique()
        start_date = st.date_input("Start Date", min(date_range), min_value=min(date_range), max_value=max(date_range))
        end_date = st.date_input("End Date", max(date_range), min_value=min(date_range), max_value=max(date_range))

        fetch_button = st.button("Fetch Data")

    if fetch_button or 'data_loaded' not in st.session_state:
        if start_date > end_date:
            st.error("End date must be after start date.")
            return

        df = fetch_data(selected_model_numbers, start_date, end_date)
        if not df.empty:
            processed_df = process_data(df)
            grouped_df = process_grouped_data(processed_df)
            st.session_state['processed_df'] = processed_df
            st.session_state['grouped_df'] = grouped_df
            st.session_state['data_loaded'] = True
        else:
            st.write("No data found for the selected date range.")
            st.session_state['data_loaded'] = False

    if st.session_state.get('data_loaded', False):
        all_step_types = st.session_state['grouped_df']['step_type'].unique().tolist()
        selected_step_types = st.multiselect('Select Step Type', all_step_types, default=all_step_types)

        filtered_df = st.session_state['grouped_df'][st.session_state['grouped_df']['step_type'].isin(selected_step_types)]

        if not filtered_df.empty:
            min_duration, max_duration = filtered_df['duration_minutes'].agg(['min', 'max'])
            min_duration, max_duration = int(min_duration), int(max_duration)
        else:
            min_duration, max_duration = 0, 0
        
        initial_min_duration = max(1, min_duration)
        
        duration_range = st.slider("Select Duration Range (minutes)", min_duration, max_duration, (initial_min_duration, max_duration))
        
        filtered_df = filtered_df[(filtered_df['duration_minutes'] >= duration_range[0]) & (filtered_df['duration_minutes'] <= duration_range[1])]

        display_data_and_plots(filtered_df, st.session_state['processed_df'])

def display_data_and_plots(filtered_df, processed_df):
    st.write("Data Overview:")
    st.dataframe(processed_df)
    
    # Add vis spec here
    
    fig = plot_current_voltage(processed_df)
    st.plotly_chart(fig, use_container_width=True)
    fig = plot_current_soc(processed_df)
    st.plotly_chart(fig, use_container_width=True)
    fig = plot_voltage_soc(processed_df)
    st.plotly_chart(fig, use_container_width=True)
    fig = plot_temp(processed_df)
    st.plotly_chart(fig, use_container_width=True)
    
    st.write("Filtered Grouped Data Overview:")
    st.dataframe(filtered_df)
    
    fig = plot_discharge_currents(filtered_df)
    st.plotly_chart(fig, use_container_width=True)
    
    summary_df = create_day_wise_summary(filtered_df)
    st.write("Day-wise Summary:")
    st.dataframe(summary_df)
    
    fig = plot_discharge_duration_candlestick(summary_df)
    st.plotly_chart(fig, use_container_width=True)

if __name__ == "__main__":
    main()

