import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import mplfinance as mpf
import streamlit as st
import random

# Function to get the previous available date
def get_previous_available_date(date, date_list):
    date_list = pd.to_datetime(date_list, errors='coerce')
    date = pd.to_datetime(date)
    date_list = date_list.dropna()  # Remove NaT values
    date_list_sorted = date_list.sort_values().reset_index(drop=True)
    idx = date_list_sorted.searchsorted(date, side='right') - 1
    if idx >= 0 and idx < len(date_list_sorted):
        return date_list_sorted.iloc[idx]
    else:
        return None

# Functions for trendline analysis (same as before)
def check_trend_line(support: bool, pivot: int, slope: float, y: pd.Series):
    intercept = -slope * pivot + y.iloc[pivot]
    line_vals = slope * np.arange(len(y)) + intercept
    diffs = line_vals - y.values

    if support and diffs.max() > 1e-5:
        return -1.0
    elif not support and diffs.min() < -1e-5:
        return -1.0

    err = (diffs ** 2.0).sum()
    return err

def optimize_slope(support: bool, pivot: int, init_slope: float, y: pd.Series):
    slope_unit = (y.max() - y.min()) / len(y)
    opt_step = 1.0
    min_step = 0.0001
    curr_step = opt_step
    best_slope = init_slope

    best_err = check_trend_line(support, pivot, init_slope, y)
    if best_err < 0:
        best_err = 0  # Set error to zero if negative

    get_derivative = True
    derivative = None
    attempts = 0  # Track the number of iterations

    while curr_step > min_step:
        if get_derivative:
            slope_change = best_slope + slope_unit * min_step
            test_err = check_trend_line(support, pivot, slope_change, y)
            derivative = test_err - best_err

            if test_err < 0.0:
                slope_change = best_slope - slope_unit * min_step
                test_err = check_trend_line(support, pivot, slope_change, y)
                derivative = best_err - test_err

            if test_err < 0.0:
                break  # Exit the loop

            get_derivative = False

        if derivative > 0.0:
            test_slope = best_slope - slope_unit * curr_step
        else:
            test_slope = best_slope + slope_unit * curr_step

        test_err = check_trend_line(support, pivot, test_slope, y)
        if test_err < 0 or test_err >= best_err:
            curr_step *= 0.5
        else:
            best_err = test_err
            best_slope = test_slope
            get_derivative = True

        attempts += 1
        if attempts > 100:
            break

    return (best_slope, -best_slope * pivot + y.iloc[pivot])

def fit_trendlines(high: pd.Series, low: pd.Series, close: pd.Series):
    slope, intercept = np.polyfit(np.arange(len(close)), close, 1)

    upper_pivot = high.idxmax()
    lower_pivot = low.idxmin()

    upper_pivot_pos = high.index.get_loc(upper_pivot)
    lower_pivot_pos = low.index.get_loc(lower_pivot)

    support_coefs = optimize_slope(True, lower_pivot_pos, slope, low)
    resist_coefs = optimize_slope(False, upper_pivot_pos, slope, high)
    return support_coefs, resist_coefs

# Streamlit Title
st.title("Trading Strategy Optimization Using Technical Analysis")

# Read data.csv
data = pd.read_csv('data.csv', delimiter=';')
data.columns = ['date', 'price']
data['price'] = data['price'].str.replace(',', '.').astype(float)
data['date'] = pd.to_datetime(data['date'], format='%d/%m/%Y', errors='coerce')
data = data.dropna(subset=['date'])  # Remove rows with invalid dates

# Read data1.csv
data1 = pd.read_csv('data1.csv')
data1['date'] = pd.to_datetime(data1['date'], errors='coerce')
data1 = data1.dropna(subset=['date'])  # Remove rows with invalid dates

# Get min and max dates from both datasets
min_date = min(data['date'].min(), data1['date'].min())
max_date = max(data['date'].max(), data1['date'].max())

# User Input for Date
selected_date = st.date_input(
    "Select a date",
    value=max_date.date(),
    min_value=min_date.date(),
    max_value=max_date.date()
)

# Add a button to select a random date
if st.button("Select Random Date"):
    random_date = pd.to_datetime(np.random.choice(pd.date_range(min_date, max_date).date))
    selected_date = random_date
    st.write(f"Random date selected: {selected_date.date()}")

selected_date = pd.to_datetime(selected_date)

# Function to get the previous available date in datasets
dates_in_data = data['date']
dates_in_data1 = data1['date']

selected_date_in_data = get_previous_available_date(selected_date, dates_in_data)
selected_date_in_data1 = get_previous_available_date(selected_date, dates_in_data1)

# Analysis on data.csv
st.header("Market trend of the XAUUSD market using least square method")
if selected_date_in_data is None:
    st.warning(f"No available date before or on {selected_date.date()} in data.csv")
else:
    selected_date_data = selected_date_in_data
    # Always use a fixed lookback period of 31 days
    lookback_days_first_graph = 31
    start_date = selected_date_data - pd.Timedelta(days=lookback_days_first_graph-1)
    end_date = selected_date_data

    # Filter data between start_date and end_date
    filtered_data = data[(data['date'] >= start_date) & (data['date'] <= end_date)]

    if filtered_data.empty:
        st.warning(f"No data available from {start_date.date()} to {end_date.date()} in data.csv.")
    else:
        # Prepare data for regression
        filtered_data = filtered_data.sort_values('date')
        filtered_data['day_number'] = (filtered_data['date'] - filtered_data['date'].min()).dt.days

        x = filtered_data['day_number'].values
        y = filtered_data['price'].values

        # Calculate sums for linear regression
        n = len(x)
        sum_x = np.sum(x)
        sum_y = np.sum(y)
        sum_xy = np.sum(x * y)
        sum_x_squared = np.sum(x ** 2)
        denominator = n * sum_x_squared - sum_x ** 2

        if denominator == 0:
            st.error("Cannot compute linear regression due to zero denominator.")
        else:
            # Calculate slope (m) and intercept (b)
            m = (n * sum_xy - sum_x * sum_y) / denominator
            b = (sum_y * sum_x_squared - sum_x * sum_xy) / denominator

            # Display the equation of the line
            st.write(f"The equation of the line is: y = {m:.2f}x + {b:.2f}")

            # Generate predictions
            predicted_prices = m * x + b

            # Plot the results using Matplotlib
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.scatter(filtered_data['date'], y, color='blue', label='Original Data')
            ax.plot(filtered_data['date'], predicted_prices, color='red', label=f'Linear fit: y = {m:.2f}x + {b:.2f}')
            ax.set_xlabel('Date')
            ax.set_ylabel('Price')
            ax.set_title(f'Linear Regression on Prices from {start_date.date()} to {end_date.date()}')
            ax.legend()
            ax.grid(True)

            # Display the plot in Streamlit
            st.pyplot(fig)

            # Display the trend message
            if m > 1:
                st.success(f"The market is up trending ( Bullish ) from {start_date.date()} to {end_date.date()} priorities Buying at this moment.")
            elif m < -1:
                st.error(f"The market is down trending from {start_date.date()} to {end_date.date()} priorities Selling at this moment.")
            else:
                st.info(f"The market is ranging from {start_date.date()} to {end_date.date()} it is better to wait at this moment until the market will be trending either up or down.")


# Trendline Analysis (data1.csv)
st.header("Trendline analysis ")
lookback_days = st.number_input(
    "Enter the number of days to include before the chosen date (for Trendline Analysis)",
    min_value=1,
    max_value=365,
    value=61
)

if selected_date_in_data1 is None:
    st.warning(f"No available date before or on {selected_date.date()} in data1.csv")
else:
    selected_date_data1 = selected_date_in_data1
    start_date1 = selected_date_data1 - pd.Timedelta(days=lookback_days - 1)
    end_date1 = selected_date_data1

    # Filter data between start_date1 and end_date1
    data1_filtered = data1[(data1['date'] >= start_date1) & (data1['date'] <= end_date1)]

    if data1_filtered.empty:
        st.warning(f"No data available from {start_date1.date()} to {end_date1.date()} in data1.csv.")
    else:
        # Sort the data by date in ascending order
        data1_filtered = data1_filtered.sort_values('date')

        # Prepare data
        data1_filtered = data1_filtered[['date', 'open', 'high', 'low', 'close']]
        data1_filtered = data1_filtered.set_index('date')

        # Ensure numerical data types
        data1_filtered = data1_filtered.astype(float)

        # Trendline fitting and plotting
        candles = data1_filtered.copy()

        # Ensure there are enough data points for trendline calculation
        if len(candles) < 2:
            st.warning(f"Not enough data to compute trendlines from {start_date1.date()} to {end_date1.date()}.")
        else:
            support_coefs, resist_coefs = fit_trendlines(candles['high'], candles['low'], candles['close'])
            support_line = support_coefs[0] * np.arange(len(candles)) + support_coefs[1]
            resist_line = resist_coefs[0] * np.arange(len(candles)) + resist_coefs[1]

            # Prepare trendlines for plotting
            alines = [
                [(candles.index[i], support_line[i]) for i in range(len(candles))],
                [(candles.index[i], resist_line[i]) for i in range(len(candles))]
            ]

            # Plot the candlestick chart with trendlines
            fig, axlist = mpf.plot(
                candles,
                type='candle',
                alines=dict(alines=alines, colors=['green', 'red']),
                style='charles',
                title=f"Candlestick with Support and Resistance Trendlines from {start_date1.date()} to {end_date1.date()}",
                figsize=(20, 12),  # Increase figure size for better visibility
                returnfig=True
            )

            # Use Streamlit's columns to center the graph
            col1, col2, col3 = st.columns([0.5, 4, 0.5])  # Center the graph in the middle column
            with col2:
                st.pyplot(fig)
