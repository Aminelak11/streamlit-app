import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import mplfinance as mpf
import streamlit as st
import random

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


# Function to check the trend line
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


# Function to optimize the slope
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


# Function to fit trendlines (Support and Resistance)
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

# Read data1.csv (Use for both graphs)
data1 = pd.read_csv('data1.csv')
data1['date'] = pd.to_datetime(data1['date'], errors='coerce')
data1 = data1.dropna(subset=['date'])  # Remove rows with invalid dates

# Ensure numerical data types for high, low, close columns
data1[['high', 'low', 'close']] = data1[['high', 'low', 'close']].astype(float)

# Get min and max dates from data1.csv
min_date = data1['date'].min()
max_date = data1['date'].max()

# Initialize session state for storing the random date
if 'random_date' not in st.session_state:
    st.session_state.random_date = None

# User Input for Date
selected_date = st.date_input(
    "Select a date",
    value=st.session_state.random_date.date() if st.session_state.random_date else max_date.date(),
    min_value=min_date.date(),
    max_value=max_date.date()
)

# Add a button to select a random date
if st.button("Select Random Date"):
    st.session_state.random_date = pd.to_datetime(np.random.choice(pd.date_range(min_date, max_date).date))
    selected_date = st.session_state.random_date
    st.write(f"Random date selected: {selected_date.date()}")
else:
    if st.session_state.random_date:
        selected_date = st.session_state.random_date

selected_date = pd.to_datetime(selected_date)

# Function to get the previous available date in data1.csv
dates_in_data1 = data1['date']
selected_date_in_data1 = get_previous_available_date(selected_date, dates_in_data1)

# Analysis on data1.csv
st.header("Market trend of the XAUUSD market using least square method")
if selected_date_in_data1 is None:
    st.warning(f"No available date before or on {selected_date.date()} in data1.csv")
else:
    selected_date_data1 = selected_date_in_data1
    # Always use a fixed lookback period of 31 days
    lookback_days_first_graph = 31
    start_date = selected_date_data1 - pd.Timedelta(days=lookback_days_first_graph - 1)
    end_date = selected_date_data1

    # Filter data between start_date and end_date
    filtered_data1 = data1[(data1['date'] >= start_date) & (data1['date'] <= end_date)]

    if filtered_data1.empty:
        st.warning(f"No data available from {start_date.date()} to {end_date.date()} in data1.csv.")
    else:
        # Prepare data for regression
        filtered_data1 = filtered_data1.sort_values('date')
        filtered_data1['day_number'] = (filtered_data1['date'] - filtered_data1['date'].min()).dt.days

        # Select the price option
        price_option = st.selectbox("Select Price Type for Analysis",
                                    ["Closing Price", "Average Price (High + Low) / 2"])
        if price_option == "Closing Price":
            y = filtered_data1['close'].values
        else:
            y = ((filtered_data1['high'] + filtered_data1['low']) / 2).values

        x = filtered_data1['day_number'].values

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
            ax.scatter(filtered_data1['date'], y, color='blue', label='Original Data')
            ax.plot(filtered_data1['date'], predicted_prices, color='red', label=f'Linear fit: y = {m:.2f}x + {b:.2f}')
            ax.set_xlabel('Date')
            ax.set_ylabel('Price')
            ax.set_title(f'Linear Regression on {price_option} from {start_date.date()} to {end_date.date()}')
            ax.legend()
            ax.grid(True)

            # Display the plot in Streamlit
            st.pyplot(fig)

            # Display the trend message
            if m > 1:
                st.success(
                    f"The market is up trending (Bullish) from {start_date.date()} to {end_date.date()}. Prioritize buying at this moment.")
            elif m < -1:
                st.error(
                    f"The market is down trending from {start_date.date()} to {end_date.date()}. Prioritize selling at this moment.")
            else: 
                st.info(
                    f"The market is ranging from {start_date.date()} to {end_date.date()}. better to wait for a trend creation.")

# Trendline Analysis (data1.csv) with Linear Regression
st.header("Trendline Analysis with Linear Regression Line")
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
            # Fit Support and Resistance trendlines
            support_coefs, resist_coefs = fit_trendlines(candles['high'], candles['low'], candles['close'])
            support_line = support_coefs[0] * np.arange(len(candles)) + support_coefs[1]
            resist_line = resist_coefs[0] * np.arange(len(candles)) + resist_coefs[1]

            # Calculate Linear Regression Line
            x_vals = np.arange(len(candles))
            slope, intercept = np.polyfit(x_vals, candles['close'], 1)
            regression_line = slope * x_vals + intercept

            # Prepare trendlines for plotting
            alines = [
                [(candles.index[i], support_line[i]) for i in range(len(candles))],
                [(candles.index[i], resist_line[i]) for i in range(len(candles))],
                [(candles.index[i], regression_line[i]) for i in range(len(candles))],  # Regression Line
            ]

            # Plot the candlestick chart with trendlines and linear regression line
            fig, axlist = mpf.plot(
                candles,
                type='candle',
                alines=dict(alines=alines, colors=['green', 'red', 'blue']),  # Blue for Regression Line
                style='charles',
                title=f"Candlestick with Support, Resistance, and Regression Line from {start_date1.date()} to {end_date1.date()}",
                figsize=(20, 12),  # Increase figure size for better visibility
                returnfig=True
            )

            # Use Streamlit's columns to center the graph
            col1, col2, col3 = st.columns([0.5, 4, 0.5])  # Center the graph in the middle column
            with col2:
                st.pyplot(fig)
