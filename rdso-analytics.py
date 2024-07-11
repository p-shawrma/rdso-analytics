# import streamlit as st
# import pandas as pd
# import psycopg2
# import numpy as np
# from datetime import datetime, timedelta
# import plotly.graph_objects as go

# # Function to create a database connection using psycopg2
# @st.cache(allow_output_mutation=True, ttl=6000, show_spinner=False)
# def get_data(start_date, end_date):
#     user = "postgres.kfuizzxktmneperhsekb"
#     password = "RDSO_Analytics_Change@2015"
#     host = "aws-0-ap-southeast-1.pooler.supabase.com"
#     port = "5432"
#     dbname = "postgres"
    
#     with psycopg2.connect(
#         dbname=dbname,
#         user=user,
#         password=password,
#         host=host,
#         port=port
#     ) as conn:
#         cursor = conn.cursor()
#         # Using direct string formatting to ensure full-day coverage
#         query = f"""
#         SELECT *
#         FROM public.custom_report_rdso
#         WHERE created_at BETWEEN '{start_date.strftime('%Y-%m-%d')} 00:00:00' AND '{end_date.strftime('%Y-%m-%d')} 23:59:59';
#         """
#         cursor.execute(query)
#         records = cursor.fetchall()
#         columns = [desc[0] for desc in cursor.description]
#         df = pd.DataFrame(records, columns=columns)
        
#         # Log the query and the number of rows fetched
#         print(f"Executed query: {query}")
#         print(f"Number of rows fetched: {len(df)}")

#         return df

# def process_data(df):
#     df['timestamp'] = pd.to_datetime(df['created_at'])
#     df = df.sort_values(by='timestamp')
#     df['time_diff'] = df['timestamp'].diff().dt.total_seconds()
    
#     # Handle the first time_diff NaN
#     df['time_diff'].fillna(method='bfill', inplace=True)

#     # Base alpha for an expected time difference, e.g., 10 seconds
#     base_time_diff = 10  # Base time difference in seconds
#     base_alpha = 0.33    # Base alpha for smoothing

#     # Adjust alpha based on actual time difference
#     df['alpha'] = df['time_diff'].apply(lambda x: base_alpha / x * base_time_diff if x > 0 else base_alpha)
#     df['alpha'] = df['alpha'].clip(upper=0.66)  # Ensure alpha does not exceed 0.45

#     # Initialize the first current to the first actual current reading
#     ema_current = df['Battery_Pack_Current(A)'].iloc[0]
#     smoothed_currents = [ema_current]

#     # Apply dynamic EMA for current
#     for i in range(1, len(df)):
#         alpha = df['alpha'].iloc[i]
#         current = df['Battery_Pack_Current(A)'].iloc[i]
#         ema_current = ema_current * (1 - alpha) + current * alpha
#         smoothed_currents.append(ema_current)

#     df['Fitted_Current(A)'] = smoothed_currents

#     # Static EMA for voltage
#     df['Fitted_Voltage(V)'] = df['Battery_Pack_Voltage(V)'].ewm(alpha=base_alpha).mean()

#     # Calculate voltage increase
#     df['voltage_increase'] = df['Fitted_Voltage(V)'].diff() > 0.02

#     # Calculate average pack temperature from all cell temperature columns
#     cell_temp_columns = [col for col in df.columns if 'Cell_Temperature' in col]
#     df['Pack_Temperature_(C)'] = df[cell_temp_columns].mean(axis=1)

#     # Define conditions and choices for states
#     epsilon = 0.1
#     conditions = [
#         df['voltage_increase'],  # Charging condition based on voltage increase
#         (df['Fitted_Current(A)'] < -epsilon) & (~df['voltage_increase']),  # Discharging condition
#         abs(df['Fitted_Current(A)']) <= epsilon  # Idle condition
#     ]
#     choices = ['charge', 'discharge', 'idle']
#     df['state'] = np.select(conditions, choices, default='idle')

#     # Group by state changes and filter short durations
#     df['state_change'] = (df['state'] != df['state'].shift(1)).cumsum()
#     grp = df.groupby('state_change')
#     df['state_duration'] = grp['timestamp'].transform(lambda x: (x.max() - x.min()).total_seconds())
#     df['filtered_state'] = np.where(df['state_duration'] <= 30, np.nan, df['state'])
#     df['filtered_state'].fillna(method='bfill', inplace=True)

#     return df

# def calculate_percentile(n):
#     """Utility function to create a percentile function for aggregation."""
#     def percentile_(x):
#         return np.percentile(x, n)
#     percentile_.__name__ = 'percentile_%s' % n
#     return percentile_

# def process_grouped_data(df):
#     """Processes grouped data to extract meaningful statistics for each continuous state segment."""
#     # Group by continuous states and apply calculations
#     grouped = df.groupby((df['filtered_state'] != df['filtered_state'].shift()).cumsum())
#     result = grouped.agg(
#         start_timestamp=('timestamp', 'min'),
#         end_timestamp=('timestamp', 'max'),
#         step_type=('filtered_state', 'first'),
#         duration_minutes=('timestamp', lambda x: (x.max() - x.min()).total_seconds() / 60),
#         soc_start=('SOC(%)', 'first'),
#         soc_end=('SOC(%)', 'last'),
#         voltage_start=('Fitted_Voltage(V)', 'first'),
#         voltage_end=('Fitted_Voltage(V)', 'last'),
#         average_current=('Fitted_Current(A)', 'mean'),
#         median_current=('Fitted_Current(A)', 'median'),
#         min_current=('Fitted_Current(A)', calculate_percentile(10)),
#         max_current=('Fitted_Current(A)', calculate_percentile(90)),
#         current_25th=('Fitted_Current(A)', calculate_percentile(25)),
#         current_75th=('Fitted_Current(A)', calculate_percentile(75)),
#         median_max_cell_temperature=('Max_Cell_Temp_(C)', 'median'),
#         median_min_cell_temperature=('Min_Cell_Temp_(C)', 'median'),
#         median_pack_temperature=('Pack_Temperature_(C)', 'median')  # Assuming you calculate or have this column
#     )
    
#     # Calculate the 'date' from 'start_timestamp'
#     result['date'] = result['start_timestamp'].dt.date
    
#     # Calculate the change in SOC
#     result['change_in_soc'] = result['soc_end'] - result['soc_start']
    
#     # Reorder columns to place 'date' just before 'start_timestamp' and 'change_in_soc' just after 'soc_end'
#     columns_ordered = ['date', 'start_timestamp', 'end_timestamp', 'step_type', 'duration_minutes',
#                        'soc_start', 'soc_end', 'change_in_soc', 'voltage_start', 'voltage_end',
#                        'average_current', 'median_current', 'min_current', 'max_current', 'current_25th',
#                        'current_75th', 'median_max_cell_temperature', 'median_min_cell_temperature', 'median_pack_temperature']

#     result = result.reindex(columns=columns_ordered)
    
#     return result

# def apply_filters(df):
#     # Retrieve user-selected step types from the sidebar
#     step_types = df['step_type'].unique().tolist()
#     selected_types = st.sidebar.multiselect('Select Step Types', step_types, default=step_types)
#     filtered_df = df[df['step_type'].isin(selected_types)]
#     return filtered_df


# def plot_current_voltage(df):
#     trace1 = go.Scatter(x=df['timestamp'], y=df['Fitted_Current(A)'], mode='lines', name='Current (A)', line=dict(color='red'))
#     trace2 = go.Scatter(x=df['timestamp'], y=df['Fitted_Voltage(V)'], mode='lines', name='Voltage (V)', yaxis='y2', line=dict(color='blue'))
#     layout = go.Layout(
#         title='Current and Voltage Over Time',
#         xaxis=dict(title='Timestamp'),
#         yaxis=dict(title='Current (A)'),
#         yaxis2=dict(title='Voltage (V)', overlaying='y', side='right'),
#         autosize=True,
#         template='plotly_white'
#     )
#     fig = go.Figure(data=[trace1, trace2], layout=layout)
#     fig.update_layout(autosize=True, margin=dict(l=20, r=20, t=40, b=20))
#     return fig

# def plot_current_soc(df):
#     trace1 = go.Scatter(x=df['timestamp'], y=df['Fitted_Current(A)'], mode='lines', name='Current (A)', line=dict(color='red'))
#     trace2 = go.Scatter(x=df['timestamp'], y=df['SOC(%)'], mode='lines', name='SOC (%)', yaxis='y2', line=dict(color='green'))
#     layout = go.Layout(
#         title='Current and SOC Over Time',
#         xaxis=dict(title='Timestamp'),
#         yaxis=dict(title='Current (A)'),
#         yaxis2=dict(title='SOC (%)', overlaying='y', side='right')
#     )
#     fig = go.Figure(data=[trace1, trace2], layout=layout)
#     return fig

# def plot_voltage_soc(df):
#     trace1 = go.Scatter(x=df['timestamp'], y=df['Fitted_Voltage(V)'], mode='lines', name='Voltage (V)', line=dict(color='blue'))
#     trace2 = go.Scatter(x=df['timestamp'], y=df['SOC(%)'], mode='lines', name='SOC (%)', yaxis='y2', line=dict(color='green'))
#     layout = go.Layout(
#         title='Voltage and SOC Over Time',
#         xaxis=dict(title='Timestamp'),
#         yaxis=dict(title='Voltage (V)'),
#         yaxis2=dict(title='SOC (%)', overlaying='y', side='right')
#     )
#     fig = go.Figure(data=[trace1, trace2], layout=layout)
#     return fig

# def plot_temp(df):
#     trace1 = go.Scatter(x=df['timestamp'], y=df['Max_Cell_Temp_(C)'], mode='lines', name='Max Cell Temp (C)', line=dict(color='orange'))
#     trace2 = go.Scatter(x=df['timestamp'], y=df['Min_Cell_Temp_(C)'], mode='lines', name='Min Cell Temp (C)', line=dict(color='purple'))
#     layout = go.Layout(title='Max and Min Cell Temperatures Over Time', xaxis=dict(title='Timestamp'), yaxis=dict(title='Temperature (C)'))
#     fig = go.Figure(data=[trace1, trace2], layout=layout)
#     return fig

# def calculate_midpoint(row):
#     return row['start_timestamp'] + (row['end_timestamp'] - row['start_timestamp']) / 2

# def plot_discharge_currents(df):
#     df['mid_timestamp'] = df.apply(calculate_midpoint, axis=1)
#     discharge_df = df[df['step_type'] == 'discharge']
#     fig = go.Figure(data=[go.Candlestick(
#         x=discharge_df['mid_timestamp'],
#         open=discharge_df['current_25th'],
#         high=discharge_df['max_current'],
#         low=discharge_df['min_current'],
#         close=discharge_df['current_75th'],
#         increasing_line_color='green', decreasing_line_color='red'
#     )])
#     fig.update_layout(title='Current Distribution in Discharge States', xaxis_title='Time', yaxis_title='Current (A)', xaxis_rangeslider_visible=False)
#     return fig

# def create_day_wise_summary(df):
#     # Filter the DataFrame for discharge and charge states
#     discharge = df[df['step_type'] == 'discharge']
#     charge = df[df['step_type'] == 'charge']

#     # Aggregate the data by date for discharge
#     discharge_summary = discharge.groupby('date').agg({
#         'change_in_soc': 'sum',
#         'duration_minutes': ['sum', 'min', 'max', 'median', calculate_percentile(25), calculate_percentile(75)]
#     })

#     # Aggregate the data by date for charge
#     charge_summary = charge.groupby('date').agg({
#         'change_in_soc': 'sum'
#     })

#     # Flatten the MultiIndex columns created by aggregation
#     discharge_summary.columns = ['_'.join(col).strip() for col in discharge_summary.columns.values]
#     charge_summary.columns = ['total_charge_soc']

#     # Merge discharge and charge summaries into a single DataFrame
#     day_wise_summary = pd.merge(discharge_summary, charge_summary, on='date', how='outer')
#     day_wise_summary.rename(columns={
#         'change_in_soc_sum': 'total_discharge_soc',
#         'duration_minutes_sum': 'total_discharge_time',  # Added total discharge time
#         'duration_minutes_min': 'discharge_time_min',
#         'duration_minutes_max': 'discharge_time_max',
#         'duration_minutes_median': 'discharge_time_median',
#         'duration_minutes_percentile_25': 'discharge_time_25th',
#         'duration_minutes_percentile_75': 'discharge_time_75th'
#     }, inplace=True)

#     return day_wise_summary

# def plot_discharge_duration_candlestick(df):
#     fig = go.Figure(data=[go.Candlestick(
#         x=df.index,
#         open=df['discharge_time_25th'],
#         high=df['discharge_time_max'],
#         low=df['discharge_time_min'],
#         close=df['discharge_time_75th'],
#         increasing_line_color='green', decreasing_line_color='red'
#     )])
#     fig.update_layout(title='Candlestick Plot of Discharge Durations', xaxis_title='Date', yaxis_title='Duration in Minutes', xaxis_rangeslider_visible=False)
#     return fig

# def main():
#     st.set_page_config(layout="wide", page_title="Battery Discharge Analysis")

#     # Sidebar for date and filtering input
#     with st.sidebar:
#         st.title("Filter Settings")
#         start_date = st.date_input("Start Date", datetime.now().date() - timedelta(days=7))
#         end_date = st.date_input("End Date", datetime.now().date())
#         fetch_button = st.button("Fetch Data")

#     # Fetch data on button press
#     if fetch_button or 'data_loaded' not in st.session_state:
#         if start_date > end_date:
#             st.error("End date must be after start date.")
#             return

#         df = get_data(start_date, end_date)
#         if not df.empty:
#             processed_df = process_data(df)
#             grouped_df = process_grouped_data(processed_df)
#             st.session_state['processed_df'] = processed_df
#             st.session_state['grouped_df'] = grouped_df
#             st.session_state['data_loaded'] = True
#         else:
#             st.write("No data found for the selected date range.")
#             st.session_state['data_loaded'] = False

#     if st.session_state.get('data_loaded', False):
#         # Initialize filters after data is loaded
#         all_step_types = st.session_state['grouped_df']['step_type'].unique().tolist()
#         selected_step_types = st.multiselect('Select Step Type', all_step_types, default=all_step_types)

#         # Filter the data based on step type selection
#         filtered_df = st.session_state['grouped_df'][st.session_state['grouped_df']['step_type'].isin(selected_step_types)]

#         # Update the duration slider based on filtered data
#         if not filtered_df.empty:
#             min_duration, max_duration = filtered_df['duration_minutes'].agg(['min', 'max'])
#             min_duration, max_duration = int(min_duration), int(max_duration)
#         else:
#             min_duration, max_duration = 0, 0  # Defaults when no data is available
        
#         # Set the initial value of the slider to start at 10, or at the minimum value if it's higher than 10
#         initial_min_duration = max(10, min_duration)
        
#         duration_range = st.slider("Select Duration Range (minutes)", min_duration, max_duration, (initial_min_duration, max_duration))
        
#         # Apply the duration filter
#         filtered_df = filtered_df[(filtered_df['duration_minutes'] >= duration_range[0]) & (filtered_df['duration_minutes'] <= duration_range[1])]
#         # Display plots and data
#         display_data_and_plots(filtered_df, st.session_state['processed_df'])

# def display_data_and_plots(filtered_df, processed_df):
#     st.write("Data Overview:")
#     st.dataframe(processed_df)
#     fig = plot_current_voltage(processed_df)
#     st.plotly_chart(fig, use_container_width=True)
#     fig = plot_current_soc(processed_df)
#     st.plotly_chart(fig, use_container_width=True)
#     fig = plot_voltage_soc(processed_df)
#     st.plotly_chart(fig, use_container_width=True)
#     fig = plot_temp(processed_df)
#     st.plotly_chart(fig, use_container_width=True)
#     st.write("Filtered Grouped Data Overview:")
#     st.dataframe(filtered_df)
#     fig = plot_discharge_currents(filtered_df)
#     st.plotly_chart(fig, use_container_width=True)
#     summary_df = create_day_wise_summary(filtered_df)
#     st.write("Day-wise Summary:")
#     st.dataframe(summary_df)
#     fig = plot_discharge_duration_candlestick(summary_df)
#     st.plotly_chart(fig, use_container_width=True)

# if __name__ == "__main__":
#     main()


import streamlit as st
import pandas as pd
import psycopg2
import numpy as np
from datetime import datetime, timedelta
import plotly.graph_objects as go
from pygwalker.api.streamlit import StreamlitRenderer

# Function to create a database connection using psycopg2
@st.cache(allow_output_mutation=True, ttl=6000, show_spinner=False)
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
    base_alpha = 0.33    # Base alpha for smoothing

    # Adjust alpha based on actual time difference
    df['alpha'] = df['time_diff'].apply(lambda x: base_alpha / x * base_time_diff if x > 0 else base_alpha)
    df['alpha'] = df['alpha'].clip(upper=0.66)  # Ensure alpha does not exceed 0.45

    # Initialize the first current to the first actual current reading
    ema_current = df['Battery_Pack_Current(A)'].iloc[0]
    smoothed_currents = [ema_current]

    # Apply dynamic EMA for current
    for i in range(1, len(df)):
        alpha = df['alpha'].iloc[i]
        current = df['Battery_Pack_Current(A)'].iloc[i]
        ema_current = ema_current * (1 - alpha) + current * alpha
        smoothed_currents.append(ema_current)

    df['Fitted_Current(A)'] = smoothed_currents

    # Static EMA for voltage
    df['Fitted_Voltage(V)'] = df['Battery_Pack_Voltage(V)'].ewm(alpha=base_alpha).mean()

    # Calculate voltage increase
    df['voltage_increase'] = df['Fitted_Voltage(V)'].diff() > 0.02

    # Calculate average pack temperature from all cell temperature columns
    cell_temp_columns = [col for col in df.columns if 'Cell_Temperature' in col]
    df['Pack_Temperature_(C)'] = df[cell_temp_columns].mean(axis=1)

    # Define conditions and choices for states
    epsilon = 0.1
    conditions = [
        df['voltage_increase'],  # Charging condition based on voltage increase
        (df['Fitted_Current(A)'] < -epsilon) & (~df['voltage_increase']),  # Discharging condition
        abs(df['Fitted_Current(A)']) <= epsilon  # Idle condition
    ]
    choices = ['charge', 'discharge', 'idle']
    df['state'] = np.select(conditions, choices, default='idle')

    # Group by state changes and filter short durations
    df['state_change'] = (df['state'] != df['state'].shift(1)).cumsum()
    grp = df.groupby('state_change')
    df['state_duration'] = grp['timestamp'].transform(lambda x: (x.max() - x.min()).total_seconds())
    df['filtered_state'] = np.where(df['state_duration'] <= 30, np.nan, df['state'])
    df['filtered_state'].fillna(method='bfill', inplace=True)

    return df

def calculate_percentile(n):
    """Utility function to create a percentile function for aggregation."""
    def percentile_(x):
        return np.percentile(x, n)
    percentile_.__name__ = 'percentile_%s' % n
    return percentile_

def process_grouped_data(df):
    """Processes grouped data to extract meaningful statistics for each continuous state segment."""
    # Group by continuous states and apply calculations
    grouped = df.groupby((df['filtered_state'] != df['filtered_state'].shift()).cumsum())
    result = grouped.agg(
        start_timestamp=('timestamp', 'min'),
        end_timestamp=('timestamp', 'max'),
        step_type=('filtered_state', 'first'),
        duration_minutes=('timestamp', lambda x: (x.max() - x.min()).total_seconds() / 60),
        soc_start=('SOC(%)', 'first'),
        soc_end=('SOC(%)', 'last'),
        voltage_start=('Fitted_Voltage(V)', 'first'),
        voltage_end=('Fitted_Voltage(V)', 'last'),
        average_current=('Fitted_Current(A)', 'mean'),
        median_current=('Fitted_Current(A)', 'median'),
        min_current=('Fitted_Current(A)', calculate_percentile(10)),
        max_current=('Fitted_Current(A)', calculate_percentile(90)),
        current_25th=('Fitted_Current(A)', calculate_percentile(25)),
        current_75th=('Fitted_Current(A)', calculate_percentile(75)),
        median_max_cell_temperature=('Max_Cell_Temp_(C)', 'median'),
        median_min_cell_temperature=('Min_Cell_Temp_(C)', 'median'),
        median_pack_temperature=('Pack_Temperature_(C)', 'median')  # Assuming you calculate or have this column
    )
    
    # Calculate the 'date' from 'start_timestamp'
    result['date'] = result['start_timestamp'].dt.date
    
    # Calculate the change in SOC
    result['change_in_soc'] = result['soc_end'] - result['soc_start']
    
    # Reorder columns to place 'date' just before 'start_timestamp' and 'change_in_soc' just after 'soc_end'
    columns_ordered = ['date', 'start_timestamp', 'end_timestamp', 'step_type', 'duration_minutes',
                       'soc_start', 'soc_end', 'change_in_soc', 'voltage_start', 'voltage_end',
                       'average_current', 'median_current', 'min_current', 'max_current', 'current_25th',
                       'current_75th', 'median_max_cell_temperature', 'median_min_cell_temperature', 'median_pack_temperature']

    result = result.reindex(columns=columns_ordered)
    
    return result

def apply_filters(df):
    # Retrieve user-selected step types from the sidebar
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
    trace2 = go.Scatter(x=df['timestamp'], y=df['SOC(%)'], mode='lines', name='SOC (%)', yaxis='y2', line=dict(color='green'))
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
    trace2 = go.Scatter(x=df['timestamp'], y=df['SOC(%)'], mode='lines', name='SOC (%)', yaxis='y2', line=dict(color='green'))
    layout = go.Layout(
        title='Voltage and SOC Over Time',
        xaxis=dict(title='Timestamp'),
        yaxis=dict(title='Voltage (V)'),
        yaxis2=dict(title='SOC (%)', overlaying='y', side='right')
    )
    fig = go.Figure(data=[trace1, trace2], layout=layout)
    return fig

def plot_temp(df):
    trace1 = go.Scatter(x=df['timestamp'], y=df['Max_Cell_Temp_(C)'], mode='lines', name='Max Cell Temp (C)', line=dict(color='orange'))
    trace2 = go.Scatter(x=df['timestamp'], y=df['Min_Cell_Temp_(C)'], mode='lines', name='Min Cell Temp (C)', line=dict(color='purple'))
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
    # Filter the DataFrame for discharge and charge states
    discharge = df[df['step_type'] == 'discharge']
    charge = df[df['step_type'] == 'charge']

    # Aggregate the data by date for discharge
    discharge_summary = discharge.groupby('date').agg({
        'change_in_soc': 'sum',
        'duration_minutes': ['sum', 'min', 'max', 'median', calculate_percentile(25), calculate_percentile(75)]
    })

    # Aggregate the data by date for charge
    charge_summary = charge.groupby('date').agg({
        'change_in_soc': 'sum'
    })

    # Flatten the MultiIndex columns created by aggregation
    discharge_summary.columns = ['_'.join(col).strip() for col in discharge_summary.columns.values]
    charge_summary.columns = ['total_charge_soc']

    # Merge discharge and charge summaries into a single DataFrame
    day_wise_summary = pd.merge(discharge_summary, charge_summary, on='date', how='outer')
    day_wise_summary.rename(columns={
        'change_in_soc_sum': 'total_discharge_soc',
        'duration_minutes_sum': 'total_discharge_time',  # Added total discharge time
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

    # Sidebar for date and filtering input
    with st.sidebar:
        st.title("Filter Settings")
        start_date = st.date_input("Start Date", datetime.now().date() - timedelta(days=7))
        end_date = st.date_input("End Date", datetime.now().date())
        fetch_button = st.button("Fetch Data")

    # Fetch data on button press
    if fetch_button or 'data_loaded' not in st.session_state:
        if start_date > end_date:
            st.error("End date must be after start date.")
            return

        df = get_data(start_date, end_date)
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
        # Initialize filters after data is loaded
        all_step_types = st.session_state['grouped_df']['step_type'].unique().tolist()
        selected_step_types = st.multiselect('Select Step Type', all_step_types, default=all_step_types)

        # Filter the data based on step type selection
        filtered_df = st.session_state['grouped_df'][st.session_state['grouped_df']['step_type'].isin(selected_step_types)]

        # Update the duration slider based on filtered data
        if not filtered_df.empty:
            min_duration, max_duration = filtered_df['duration_minutes'].agg(['min', 'max'])
            min_duration, max_duration = int(min_duration), int(max_duration)
        else:
            min_duration, max_duration = 0, 0  # Defaults when no data is available
        
        # Set the initial value of the slider to start at 10, or at the minimum value if it's higher than 10
        initial_min_duration = max(10, min_duration)
        
        duration_range = st.slider("Select Duration Range (minutes)", min_duration, max_duration, (initial_min_duration, max_duration))
        
        # Apply the duration filter
        filtered_df = filtered_df[(filtered_df['duration_minutes'] >= duration_range[0]) & (filtered_df['duration_minutes'] <= duration_range[1])]
        # Display plots and data
        display_data_and_plots(filtered_df, st.session_state['processed_df'])

def display_data_and_plots(filtered_df, processed_df):
    # st.write("Data Overview:")
    # st.dataframe(processed_df)
    
    # PyG Walker for data exploration
    pyg_app = StreamlitRenderer(processed_df)
    pyg_app.explorer()

    fig = plot_current_voltage(processed_df)
    st.plotly_chart(fig, use_container_width=True)
    fig = plot_current_soc(processed_df)
    st.plotly_chart(fig, use_container_width=True)
    fig = plot_voltage_soc(processed_df)
    st.plotly_chart(fig, use_container_width=True)
    fig = plot_temp(processed_df)
    st.plotly_chart(fig, use_container_width=True)
    
    # st.write("Filtered Grouped Data Overview:")
    # st.dataframe(filtered_df)
    
    # fig = plot_discharge_currents(filtered_df)
    # st.plotly_chart(fig, use_container_width=True)
    
    # summary_df = create_day_wise_summary(filtered_df)
    # st.write("Day-wise Summary:")
    # st.dataframe(summary_df)
    
    # fig = plot_discharge_duration_candlestick(summary_df)
    # st.plotly_chart(fig, use_container_width=True)

if __name__ == "__main__":
    main()

