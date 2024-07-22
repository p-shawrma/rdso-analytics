import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import plotly.graph_objects as go
import clickhouse_connect
from haversine import haversine, Unit
from geopy.geocoders import Nominatim
import psycopg2

# Function to fetch data from Supabase
@st.cache_data(ttl=6000)
def fetch_mapping_table():
    user = "postgres.gqmpfexjoachyjgzkhdf"
    password = "Change@2015Log9"
    host = "aws-0-ap-south-1.pooler.supabase.com"
    port = "6543"
    dbname = "postgres"
    
    with psycopg2.connect(
        dbname=dbname,
        user=user,
        password=password,
        host=host,
        port=port
    ) as conn:
        cursor = conn.cursor()
        query = "SELECT * FROM mapping_table"
        cursor.execute(query)
        records = cursor.fetchall()
        columns = [desc[0] for desc in cursor.description]
        df_mapping = pd.DataFrame(records, columns=columns)
        return df_mapping

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
    query = "SELECT DISTINCT Model_Number, toStartOfDay(DeviceDate) as DeviceDate FROM custom_tracking"
    result = client.query(query)
    df = pd.DataFrame(result.result_rows, columns=result.column_names)
    return df

# Function to fetch data based on filters
@st.cache_data(ttl=6000)
def fetch_data(model_numbers, start_date, end_date):
    client = create_client()
    model_numbers_str = ','.join([f"'{model}'" for model in model_numbers])
    start_date = start_date.strftime('%Y-%m-%d %H:%M:%S')
    end_date = end_date.strftime('%Y-%m-%d %H:%M:%S')
    query = f"""
    SELECT *
    FROM custom_tracking
    WHERE Model_Number IN ({model_numbers_str}) AND DeviceDate BETWEEN '{start_date}' AND '{end_date}'
    """
    result = client.query(query)
    df = pd.DataFrame(result.result_rows, columns=result.column_names)
    return df

def haversine_distance(lat1, lon1, lat2, lon2):
    loc1 = (lat1, lon1)
    loc2 = (lat2, lon2)
    return haversine(loc1, loc2, unit=Unit.KILOMETERS)

def get_location_description(lat, lon):
    geolocator = Nominatim(user_agent="geoapiExercises")
    location = geolocator.reverse((lat, lon), language='en')
    return location.address if location else "Unknown location"

def process_data(df):
    grouped = df.groupby('Model_Number')
    processed_dfs = []

    for name, group in grouped:
        group['timestamp'] = pd.to_datetime(group['DeviceDate'])
        group = group.sort_values(by='timestamp')
        group['time_diff'] = group['timestamp'].diff().dt.total_seconds()
        group['time_diff'].fillna(method='bfill', inplace=True)

        base_time_diff = 10
        base_alpha = 0.33

        group['alpha'] = group['time_diff'].apply(lambda x: base_alpha / x * base_time_diff if x > 0 else base_alpha)
        group['alpha'] = group['alpha'].clip(upper=0.66)

        ema_current = group['BM_BattCurrent'].iloc[0]
        smoothed_currents = [ema_current]

        for i in range(1, len(group)):
            alpha = group['alpha'].iloc[i]
            current = group['BM_BattCurrent'].iloc[i]
            ema_current = ema_current * (1 - alpha) + current * alpha
            smoothed_currents.append(ema_current)

        group['Fitted_Current(A)'] = smoothed_currents
        group['Fitted_Voltage(V)'] = group['BM_BattVoltage'].ewm(alpha=base_alpha).mean()

        group['voltage_increase'] = group['Fitted_Voltage(V)'].diff() >= 0.05
        group['soc_increase'] = group['BM_SocPercent'].diff() >= 0.05

        cell_temp_columns = [col for col in group.columns if 'Cell_Temperature' in col]
        group['Pack_Temperature_(C)'] = group[cell_temp_columns].mean(axis=1)

        epsilon = 0.5
        conditions = [
            (group['Fitted_Current(A)'] > epsilon) | (group['voltage_increase'] | group['soc_increase']),
            (group['Fitted_Current(A)'] < -epsilon) & ~((group['voltage_increase']) | (group['soc_increase'])),
            abs(group['Fitted_Current(A)']) <= epsilon
        ]
        choices = ['charge', 'discharge', 'idle']
        group['state'] = np.select(conditions, choices, default='idle')

        group['state_change'] = (group['state'] != group['state'].shift(1)).cumsum()
        grp = group.groupby('state_change')
        group['state_duration'] = grp['timestamp'].transform(lambda x: (x.max() - x.min()).total_seconds())
        group['soc_diff'] = grp['BM_SocPercent'].transform(lambda x: x.iloc[-1] - x.iloc[0])

        group['filtered_state'] = np.where(
            (group['state_duration'] > 60), 
            group['state'], 
            np.nan
        )
        group['filtered_state'].fillna(method='ffill', inplace=True)

        group['final_state'] = np.where(
            (group['soc_diff'].abs() < 1) & (group['state_duration'] < 600),
            np.nan,
            group['filtered_state']
        )
        group['final_state'].fillna(method='ffill', inplace=True)

        state_mapping = {'charge': 0, 'discharge': 1, 'idle': 2}
        group['step_type'] = group['final_state'].map(state_mapping)

        processed_dfs.append(group)

    return pd.concat(processed_dfs)

def calculate_percentile(n):
    def percentile_(x):
        return np.percentile(x, n)
    percentile_.__name__ = 'percentile_%s' % n
    return percentile_

def process_grouped_data(df):
    grouped = df.groupby(['Model_Number', (df['final_state'] != df['final_state'].shift()).cumsum()])
    result = grouped.agg(
        start_timestamp=('timestamp', 'min'),
        end_timestamp=('timestamp', 'max'),
        step_type=('final_state', 'first'),
        duration_minutes=('timestamp', lambda x: (x.max() - x.min()).total_seconds() / 60),
        soc_start=('BM_SocPercent', 'first'),
        soc_end=('BM_SocPercent', 'last'),
        voltage_start=('BM_BattVoltage', 'first'),
        voltage_end=('BM_BattVoltage', 'last'),
        average_current=('BM_BattCurrent', 'mean'),
        median_current=('BM_BattCurrent', 'median'),
        min_current=('BM_BattCurrent', calculate_percentile(5)),
        max_current=('BM_BattCurrent', calculate_percentile(95)),
        current_25th=('BM_BattCurrent', calculate_percentile(25)),
        current_75th=('BM_BattCurrent', calculate_percentile(75)),
        median_max_cell_temperature=('Max_monomer_temperature', 'median'),
        median_min_cell_temperature=('Min_monomer_temperature', 'median'),
        median_pack_temperature=('Pack_Temperature_(C)', 'median')
    )

    result['date'] = result['start_timestamp'].dt.date
    result['change_in_soc'] = result['soc_end'] - result['soc_start']

    columns_ordered = ['Model_Number', 'date', 'start_timestamp', 'end_timestamp', 'step_type', 'duration_minutes',
                       'soc_start', 'soc_end', 'change_in_soc', 'voltage_start', 'voltage_end',
                       'average_current', 'median_current', 'min_current', 'max_current', 'current_25th',
                       'current_75th', 'median_max_cell_temperature', 'median_min_cell_temperature', 'median_pack_temperature']
    result = result.reset_index()[columns_ordered]

    return result

def generate_soc_report(df):
    grouped = df.groupby(['Model_Number', (df['final_state'] != df['final_state'].shift()).cumsum()])
    
    result = grouped.agg(
        start_timestamp=('timestamp', 'min'),
        end_timestamp=('timestamp', 'max'),
        soc_type=('final_state', 'first'),
        duration_minutes=('timestamp', lambda x: (x.max() - x.min()).total_seconds() / 60),
        soc_start=('BM_SocPercent', 'first'),
        soc_end=('BM_SocPercent', 'last'),
        voltage_start=('BM_BattVoltage', 'first'),
        voltage_end=('BM_BattVoltage', 'last'),
        average_current=('BM_BattCurrent', 'mean'),
        median_current=('BM_BattCurrent', 'median'),
        min_current=('BM_BattCurrent', calculate_percentile(5)),
        max_current=('BM_BattCurrent', calculate_percentile(95)),
        current_25th=('BM_BattCurrent', calculate_percentile(25)),
        current_75th=('BM_BattCurrent', calculate_percentile(75)),
        median_max_cell_temperature=('Max_monomer_temperature', 'median'),
        median_min_cell_temperature=('Min_monomer_temperature', 'median'),
        median_pack_temperature=('Pack_Temperature_(C)', 'median')
    )

    result['start_date'] = result['start_timestamp'].dt.date
    result['change_in_soc'] = result['soc_end'] - result['soc_start']
    result['end_date'] = result['end_timestamp'].dt.date

    result.rename(columns={
        'Model_Number': 'vehicle_number',
        'start_timestamp': 'start_time',
        'end_timestamp': 'end_time',
        'soc_type': 'soc_type',
        'start_date': 'start_date',
        'end_date': 'end_date'
    }, inplace=True)

    result['primary_id'] = result.apply(lambda row: f"{row['vehicle_number']}-{row['start_time']}", axis=1)
    result['soc_range'] = result.apply(lambda row: f"{row['soc_start']}% - {row['soc_end']}%", axis=1)

    result['total_distance_km'] = None
    result['total_running_time_seconds'] = None
    result['energy_consumption'] = None
    result['total_halt_time_seconds'] = None
    result['charging_location'] = None
    result['charging_location_coordinates'] = None

    for name, group in grouped:
        if group['final_state'].iloc[0] == 'discharge':
            total_distance = 0
            for i in range(1, len(group)):
                coord1 = (group.iloc[i-1]['Latitude'], group.iloc[i-1]['Longitude'])
                coord2 = (group.iloc[i]['Latitude'], group.iloc[i]['Longitude'])
                total_distance += haversine(coord1, coord2)
            result.loc[result['primary_id'] == name, 'total_distance_km'] = total_distance
            total_running_time = group['duration_minutes'].sum() * 60
            result.loc[result['primary_id'] == name, 'total_running_time_seconds'] = total_running_time
        elif group['final_state'].iloc[0] == 'charge':
            lat, lon = group.iloc[0]['Latitude'], group.iloc[0]['Longitude']
            result.loc[result['primary_id'] == name, 'charging_location'] = fetch_location_description(lat, lon)
            result.loc[result['primary_id'] == name, 'charging_location_coordinates'] = f"{lat}, {lon}"
        elif group['final_state'].iloc[0] == 'idle':
            total_halt_time = group['duration_minutes'].sum() * 60
            result.loc[result['primary_id'] == name, 'total_halt_time_seconds'] = total_halt_time

    mapping_df = fetch_mapping_table()
    mapping_dict = mapping_df.set_index('telematics_number').T.to_dict()

    def fetch_mapping_info(telematics_number, key):
        return mapping_dict.get(telematics_number, {}).get(key, None)

    result['telematics_number'] = result['vehicle_number'].apply(lambda x: x.replace("it_", ""))
    result['partner_id'] = result['telematics_number'].apply(lambda x: fetch_mapping_info(x, 'partner_id'))
    result['product'] = result['telematics_number'].apply(lambda x: fetch_mapping_info(x, 'product'))
    result['deployed_city'] = result['telematics_number'].apply(lambda x: fetch_mapping_info(x, 'deployed_city'))
    result['reg_no'] = result['telematics_number'].apply(lambda x: fetch_mapping_info(x, 'reg_no'))
    result['chassis_number'] = result['telematics_number'].apply(lambda x: fetch_mapping_info(x, 'chassis_number'))

    return result

def apply_filters(df):
    step_types = df['step_type'].unique().tolist()
    selected_types = st.sidebar.multiselect('Select Step Types', step_types, default=step_types)
    filtered_df = df[df['step_type'].isin(selected_types)]
    return filtered_df

def create_day_wise_summary(df):
    discharge = df[df['step_type'] == 'discharge']
    charge = df[df['step_type'] == 'charge']

    discharge_summary = discharge.groupby(['Model_Number', 'date']).agg({
        'change_in_soc': 'sum',
        'duration_minutes': ['sum', 'min', 'max', 'median', calculate_percentile(25), calculate_percentile(75)]
    })

    charge_summary = charge.groupby(['Model_Number', 'date']).agg({
        'change_in_soc': 'sum'
    })

    discharge_summary.columns = ['_'.join(col).strip() for col in discharge_summary.columns.values]
    charge_summary.columns = ['total_charge_soc']

    day_wise_summary = pd.merge(discharge_summary, charge_summary, on=['Model_Number', 'date'], how='outer')
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

def main():
    st.set_page_config(layout="wide", page_title="Battery Discharge Analysis")

    with st.sidebar:
        st.title("Filter Settings")
        model_numbers_and_dates = fetch_model_numbers_and_dates()
        model_numbers = model_numbers_and_dates['Model_Number'].unique().tolist()
        selected_model_numbers = st.multiselect('Select Model Numbers', model_numbers)

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
            soc_report = generate_soc_report(processed_df)  # Generate the SOC report
            st.session_state['processed_df'] = processed_df
            st.session_state['grouped_df'] = grouped_df
            st.session_state['soc_report'] = soc_report  # Save the SOC report
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

        display_data_and_plots(filtered_df, st.session_state['processed_df'], st.session_state['soc_report'])  # Pass the SOC report

def display_data_and_plots(filtered_df, processed_df, soc_report):
    st.write("Data Overview:")
    st.dataframe(processed_df)
    
    # Add vis spec here
    
    st.write("Filtered Grouped Data Overview:")
    st.dataframe(filtered_df)
    
    st.write("SOC Report:")
    st.dataframe(soc_report)
    
    summary_df = create_day_wise_summary(filtered_df)
    st.write("Day-wise Summary:")
    st.dataframe(summary_df)

if __name__ == "__main__":
    main()

