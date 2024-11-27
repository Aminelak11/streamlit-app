import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import mplfinance as mpf
import streamlit as st
import random

# Function to compute linear regression manually
def compute_linear_regression(x, y):
    n = len(x)
    sum_x = np.sum(x)
    sum_y = np.sum(y)
    sum_xy = np.sum(x * y)
    sum_x_squared = np.sum(x ** 2)
    denominator = n * sum_x_squared - sum_x ** 2

    if denominator == 0:
        return None, None
    else:
        slope = (n * sum_xy - sum_x * sum_y) / denominator
        intercept = (sum_y * sum_x_squared - sum_x * sum_xy) / denominator
        return slope, intercept

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
    x = np.arange(len(close))
    y = close.values
    slope, intercept = compute_linear_regression(x, y)

    if slope is None or intercept is None:
        # Handle zero denominator case
        slope = 0
        intercept = y.mean()

    upper_pivot = high.idxmax()
    lower_pivot = low.idxmin()

    upper_pivot_pos = high.index.get_loc(upper_pivot)
    lower_pivot_pos = low.index.get_loc(lower_pivot)

    support_coefs = optimize_slope(True, lower_pivot_pos, slope, low)
    resist_coefs = optimize_slope(False, upper_pivot_pos, slope, high)
    return support_coefs, resist_coefs

# Streamlit Title
st.title("Trading Strategy Optimization Using Technical Analysis for the GOLD Market")
st.write("This tool provides automated trading analysis for gold specifically for day traders, leveraging three core algorithms to empower traders with actionable insights. It identifies the market trend, plots precise trendlines, and highlights critical key levels. With this analysis, traders can determine optimal positions (buy or sell) and pinpoint potential entry points at key levels, enabling informed and strategic decision-making")

# Read data1.csv
data1 = pd.read_csv('data1.csv')
data1['date'] = pd.to_datetime(data1['date'], errors='coerce')
data1 = data1.dropna(subset=['date'])

# Ensure numerical data types for high, low, close columns
data1[['open', 'high', 'low', 'close']] = data1[['open', 'high', 'low', 'close']].astype(float)

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
    max_value=max_date.date(),
    key="date_input_1"  # Add a unique key here
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
st.header("Market Trend Identification")
st.write("The first algorithm identifies the market's current trend, which is a crucial starting point for any analysis. By understanding the trend, you can determine whether to focus on buy trades or sell trades, setting a clear direction for your strategy")
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
                                    ["Closing Price", "Average Price (High + Low) / 2"],
                                    key="price_option_1")
        if price_option == "Closing Price":
            y = filtered_data1['close'].values
        else:
            y = ((filtered_data1['high'] + filtered_data1['low']) / 2).values

        x = filtered_data1['day_number'].values

        n = len(x)
        sum_x = np.sum(x)
        sum_y = np.sum(y)
        sum_xy = np.sum(x * y)
        sum_x_squared = np.sum(x ** 2)
        denominator = n * sum_x_squared - sum_x ** 2

        if denominator == 0:
            st.error("Cannot compute Least square method due to zero denominator.")
        else:
            slope = (n * sum_xy - sum_x * sum_y) / denominator
            intercept = (sum_y * sum_x_squared - sum_x * sum_xy) / denominator

            st.write(f"The equation of the line using Least square method is: y = {slope:.2f}x + {intercept:.2f}")

        
            predicted_prices = slope * x + intercept

            fig, ax = plt.subplots(figsize=(10, 6))
            ax.scatter(filtered_data1['date'], y, color='blue', label='Original Data')
            ax.plot(filtered_data1['date'], predicted_prices, color='red', label=f'Linear fit: y = {slope:.2f}x + {intercept:.2f}')
            ax.set_xlabel('Date')
            ax.set_ylabel('Price')
            ax.set_title(f'Trend identfication on {price_option} from {start_date.date()} to {end_date.date()}')
            ax.legend()
            ax.grid(True)

            # Display the plot in Streamlit
            st.pyplot(fig)

            # Display the trend message
            if slope > 1:
                st.success(
                    f"The market is up trending (Bullish) from {start_date.date()} to {end_date.date()}. Prioritize buying at this moment.")
            elif slope < -1:
                st.error(
                    f"The market is down trending from {start_date.date()} to {end_date.date()}. Prioritize selling at this moment.")
            else:
                st.info(
                    f"The market is ranging from {start_date.date()} to {end_date.date()}. Better to wait for a trend creation.")

# Trendline Analysis (data1.csv) with Linear Regression
st.header("Trendline Identification")
st.write("The second algorithm helps you identify potential trend continuations and reversals by accurately plotting trendlines based on market highs and lows. This insight allows you to anticipate market movements and adjust your strategy accordingly.")

lookback_days = st.number_input(
    "Enter the number of days to include before the chosen date (for Trendline Analysis)",
    min_value=1,
    max_value=365,
    value=61,
    key="lookback_days_trendline"
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
            support_slope, support_intercept = support_coefs
            resist_slope, resist_intercept = resist_coefs

            # Calculate the regression line manually
            x_vals = np.arange(len(candles))
            y_vals = candles['close'].values

            slope, intercept = compute_linear_regression(x_vals, y_vals)
            if slope is None or intercept is None:
                st.error("Cannot compute linear regression due to zero denominator.")
                regression_line = np.full_like(x_vals, y_vals.mean())  # Use mean as fallback
            else:
                regression_line = slope * x_vals + intercept

            # Display the calculated trendline equations
            st.write("### Calculated Trendline Equations:")
            st.latex(fr"y_\text{{support}} = {support_slope:.2f}x + {support_intercept:.2f}")
            st.latex(fr"y_\text{{resistance}} = {resist_slope:.2f}x + {resist_intercept:.2f}")

            # Prepare trendlines for plotting
            support_line = support_slope * np.arange(len(candles)) + support_intercept
            resist_line = resist_slope * np.arange(len(candles)) + resist_intercept

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
# Streamlit Title
st.title("Support and Resistance Levels")
st.write("The third and final algorithm draws potential buy or sell levels by mapping support and resistance zones leveraging the trend identified by the first algorithm. Using market volatility to refine these zones, offering a more accurate and actionable range for strategic decision-making.")

# User input for minimum distance between two support lines
min_distance_between_supports = st.number_input(
    "Enter the minimum distance between two support/resistance lines (e.g., 2% of price range):",
    min_value=0.01,  # minimum value is 0.01 to avoid too small numbers
    value=0.10,  # default value is 2%
    key="min_distance_supports"
)

# Input for standard deviation multiplier
multiplier = st.number_input(
    'Enter the multiplier for standard deviation (e.g., 1 or 2):',
    min_value=0.1,
    value=0.25,
    key='std_multiplier'
)

# Reuse selected_date or get a new date_input with a unique key
selected_date_candlestick = st.date_input(
    "Select a date for Candlestick Chart with Trend Analysis",
    value=st.session_state.random_date.date() if st.session_state.random_date else max_date.date(),
    min_value=min_date.date(),
    max_value=max_date.date(),
    key="date_input_candlestick"
)

# Function to get the previous available date in data1.csv
selected_date_in_data1_candlestick = get_previous_available_date(selected_date_candlestick, dates_in_data1)

if selected_date_in_data1_candlestick is None:
    st.warning(f"No available date before or on {selected_date_candlestick.date()} in data1.csv")
else:
    selected_date_data1 = selected_date_in_data1_candlestick
    lookback_days_candlestick = st.number_input(
        "Enter the number of days to include before the chosen date (for Candlestick Chart)",
        min_value=1,
        max_value=365,
        value=61,
        key="lookback_days_candlestick"
    )
    start_date = selected_date_data1 - pd.Timedelta(days=lookback_days_candlestick - 1)
    end_date = selected_date_data1

    # Filter data between start_date and end_date
    filtered_data1 = data1[(data1['date'] >= start_date) & (data1['date'] <= end_date)]

    if filtered_data1.empty:
        st.warning(f"No data available from {start_date.date()} to {end_date.date()} in data1.csv.")
    else:
        # Sort the filtered data by date
        filtered_data1 = filtered_data1.sort_values('date')

        # Prepare data for regression based on candlestick body (average of open and close)
        filtered_data1['body_avg'] = (filtered_data1['open'] + filtered_data1['close']) / 2

        # Prepare day number for x-axis (days since start of filtered data)
        filtered_data1['day_number'] = (filtered_data1['date'] - filtered_data1['date'].min()).dt.days
        x = filtered_data1['day_number'].values
        y = filtered_data1['body_avg'].values

        # Perform linear regression manually
        n = len(x)
        sum_x = np.sum(x)
        sum_y = np.sum(y)
        sum_xy = np.sum(x * y)
        sum_x_squared = np.sum(x ** 2)
        denominator = n * sum_x_squared - sum_x ** 2

        if denominator == 0:
            st.error("Cannot compute linear regression due to zero denominator.")
            slope = 0
            intercept = y.mean()
        else:
            # Calculate slope and intercept
            slope = (n * sum_xy - sum_x * sum_y) / denominator
            intercept = (sum_y * sum_x_squared - sum_x * sum_xy) / denominator

        # Plot the candlestick chart
        candles = filtered_data1[['date', 'open', 'high', 'low', 'close']].set_index('date')

        # Define buffer threshold (for example, 2% of the price range)
        buffer_threshold = (filtered_data1['high'].max() - filtered_data1['low'].min()) * min_distance_between_supports

        # Function to check if the line is too close to another line
        def is_line_too_close(new_line_price, existing_lines, buffer):
            return any(abs(new_line_price - line[0]) < buffer for line in existing_lines)

        # Prepare list to store lines for mplfinance and track horizontal lines
        lines = []
        horizontal_lines = []  # Keep track of the horizontal lines

        # If there is an uptrend, proceed to identify the special green line case
        if slope > 1:
           

            # Variables to track state
            red_candle_found = False
            red_candle_high = None
            red_candle_low = None
            last_red_candle_low = None
            last_red_candle_index = None

            # Iterate over the filtered data to find patterns
            for i in range(2, len(filtered_data1)):
                current_candle = filtered_data1.iloc[i]
                previous_candle_1 = filtered_data1.iloc[i - 1]
                previous_candle_2 = filtered_data1.iloc[i - 2]

                # Check for at least two consecutive green candles
                if (previous_candle_1['close'] > previous_candle_1['open'] and
                    previous_candle_2['close'] > previous_candle_2['open']):
                    # Find a valid red candle after the green sequence
                    if current_candle['close'] < current_candle['open']:
                        red_candle_found = True
                        red_candle_high = current_candle['open']
                        red_candle_low = current_candle['close']
                        last_red_candle_low = red_candle_low  # Store the low of this red candle
                        last_red_candle_index = i  # Store the index of the red candle
                        continue  # Move to the next candle to look for the green one

                # If another red candle is found after the first, update the red candle tracking
                if red_candle_found and current_candle['close'] < current_candle['open']:
                    red_candle_high = current_candle['open']
                    red_candle_low = current_candle['close']
                    last_red_candle_low = red_candle_low  # Update to the most recent red candle
                    last_red_candle_index = i  # Update the index

                # Once a red candle is found, look for the next green candle that breaks its high
                if red_candle_found:
                    if current_candle['close'] > current_candle['open'] and current_candle['close'] > red_candle_high:
                        # Check if the support line is too close to another horizontal line
                        if not is_line_too_close(last_red_candle_low, horizontal_lines, buffer_threshold):
                            # Add the green horizontal line at the low of the most recent red candle's body
                            lines.append(mpf.make_addplot([last_red_candle_low]*len(candles), color='green', linestyle='--'))
                            horizontal_lines.append( (last_red_candle_low, 'support') )  # Store the support line
                            st.write(f"Green line drawn at {last_red_candle_low} from the red candle on {filtered_data1.iloc[last_red_candle_index]['date'].date()}.")
                        red_candle_found = False  # Reset the red candle tracking

            # If no green candle breaks the red candle's high, draw the line on the low of the last red candle
            if red_candle_found and last_red_candle_low:
                if not is_line_too_close(last_red_candle_low, horizontal_lines, buffer_threshold):
                    lines.append(mpf.make_addplot([last_red_candle_low]*len(candles), color='green', linestyle='--'))
                    horizontal_lines.append( (last_red_candle_low, 'support') )  # Store the support line
                    st.write(f"Green line drawn at {last_red_candle_low} from the last red candle's low on {filtered_data1.iloc[last_red_candle_index]['date'].date()}.")

        elif slope < -1:
           
            # Variables to track state
            green_candle_found = False
            green_candle_high = None
            green_candle_low = None
            last_green_candle_high = None
            last_green_candle_index = None

            # Iterate over the filtered data to find patterns
            for i in range(2, len(filtered_data1)):
                current_candle = filtered_data1.iloc[i]
                previous_candle_1 = filtered_data1.iloc[i - 1]
                previous_candle_2 = filtered_data1.iloc[i - 2]

                # Check for at least two consecutive red candles
                if (previous_candle_1['close'] < previous_candle_1['open'] and
                    previous_candle_2['close'] < previous_candle_2['open']):
                    # Find a valid green candle after the red sequence
                    if current_candle['close'] > current_candle['open']:
                        green_candle_found = True
                        green_candle_high = current_candle['close']
                        green_candle_low = current_candle['open']
                        last_green_candle_high = green_candle_high  # Store the high of this green candle
                        last_green_candle_index = i  # Store the index of the green candle
                        continue  # Move to the next candle to look for the red one

                # If another green candle is found after the first, update the green candle tracking
                if green_candle_found and current_candle['close'] > current_candle['open']:
                    green_candle_high = current_candle['close']
                    green_candle_low = current_candle['open']
                    last_green_candle_high = green_candle_high  # Update to the most recent green candle
                    last_green_candle_index = i  # Update the index

                # Once a green candle is found, look for the next red candle that breaks its low
                if green_candle_found:
                    if current_candle['close'] < current_candle['open'] and current_candle['close'] < green_candle_low:
                        # Check if the resistance line is too close to another horizontal line
                        if not is_line_too_close(last_green_candle_high, horizontal_lines, buffer_threshold):
                            # Add the red horizontal line at the high of the most recent green candle's body
                            lines.append(mpf.make_addplot([last_green_candle_high]*len(candles), color='red', linestyle='--'))
                            horizontal_lines.append( (last_green_candle_high, 'resistance') )  # Store the resistance line
                            st.write(f"Red line drawn at {last_green_candle_high} from the green candle on {filtered_data1.iloc[last_green_candle_index]['date'].date()}.")
                        green_candle_found = False  # Reset the green candle tracking

            # If no red candle breaks the green candle's low, draw the line on the high of the last green candle
            if green_candle_found and last_green_candle_high:
                if not is_line_too_close(last_green_candle_high, horizontal_lines, buffer_threshold):
                    lines.append(mpf.make_addplot([last_green_candle_high]*len(candles), color='red', linestyle='--'))
                    horizontal_lines.append( (last_green_candle_high, 'resistance') )  # Store the resistance line
                    st.write(f"Red line drawn at {last_green_candle_high} from the last green candle's high on {filtered_data1.iloc[last_green_candle_index]['date'].date()}.")

        else:
           

        # Calculate mean price and standard deviation
        mean_price = filtered_data1['close'].mean()
        std_dev = filtered_data1['close'].std()

        # Display mean price and standard deviation
        st.write(f"Mean Price: {mean_price:.2f}")
        st.write(f"Standard Deviation: {std_dev:.2f}")

        # Calculate zone width
        zone_width = std_dev * multiplier

        # Prepare zones based on horizontal lines and zone width
        zones = []

        for line_price, line_type in horizontal_lines:
            upper_boundary = line_price + (zone_width / 2)
            lower_boundary = line_price - (zone_width / 2)
            zones.append((lower_boundary, upper_boundary, line_type))

        # Plot with horizontal lines if they exist
        if lines:
            fig, axlist = mpf.plot(
                candles,
                type='candle',
                style='charles',
                title=f"Candlestick Chart with Support/Resistance Zones from {start_date.date()} to {end_date.date()}",
                figsize=(10, 6),
                returnfig=True,
                addplot=lines  # Add the horizontal lines
            )
        else:
            fig, axlist = mpf.plot(
                candles,
                type='candle',
                style='charles',
                title=f"Candlestick Chart from {start_date.date()} to {end_date.date()}",
                figsize=(10, 6),
                returnfig=True,
            )

        ax = axlist[0]

        # Plot the zones as shaded areas
        for lower_boundary, upper_boundary, line_type in zones:
            if line_type == 'support':
                ax.axhspan(lower_boundary, upper_boundary, color='lightgreen', alpha=0.3)
            else:
                ax.axhspan(lower_boundary, upper_boundary, color='lightcoral', alpha=0.3)

        # Display the plot in Streamlit
        st.pyplot(fig)

        # Fourth Plot: Candlestick Chart with Trendlines and Support/Resistance Levels
    st.header("Visual Summary of the Trading strategy")
    st.write(
        "This final algorithm provides a comprehensive conclusion by synthesizing the results of the first three algorithms based on the specific date range selected by the user. It consolidates all key insights into a unified visualization, helping traders make informed decisions by presenting a clear summary of market trends and actionable insights tailored to their chosen timeframe.")

    # Ensure 'filtered_data1' is sorted and contains necessary columns
    filtered_data1 = filtered_data1.sort_values('date')
    candles = filtered_data1[['date', 'open', 'high', 'low', 'close']].set_index('date')
    candles = candles.astype(float)

    # Fit Support and Resistance trendlines
    support_coefs, resist_coefs = fit_trendlines(candles['high'], candles['low'], candles['close'])
    support_slope, support_intercept = support_coefs
    resist_slope, resist_intercept = resist_coefs

    # Calculate the regression line manually
    x_vals = np.arange(len(candles))
    y_vals = candles['close'].values

    slope, intercept = compute_linear_regression(x_vals, y_vals)
    if slope is None or intercept is None:
        st.error("Cannot compute linear regression due to zero denominator.")
        regression_line = np.full_like(x_vals, y_vals.mean())
    else:
        regression_line = slope * x_vals + intercept

    # Prepare trendlines as pandas Series
    support_line = support_slope * x_vals + support_intercept
    resist_line = resist_slope * x_vals + resist_intercept

    support_line_series = pd.Series(support_line, index=candles.index)
    resist_line_series = pd.Series(resist_line, index=candles.index)
    regression_line_series = pd.Series(regression_line, index=candles.index)

    # Create addplots for trendlines
    ap0 = mpf.make_addplot(support_line_series, color='green', width=1.5)
    ap1 = mpf.make_addplot(resist_line_series, color='red', width=1.5)
    ap2 = mpf.make_addplot(regression_line_series, color='blue', linestyle='--', width=1.5)

    # Combine all addplots (trendlines and horizontal support/resistance lines)
    all_addplots = [ap0, ap1, ap2] + lines  # 'lines' should be defined from the third plot

    # Plot the combined candlestick chart
    fig, axlist = mpf.plot(
        candles,
        type='candle',
        style='charles',
        title=f"Candlestick Chart with Trendlines and Support/Resistance Zones from {start_date.date()} to {end_date.date()}",
        figsize=(12, 8),
        returnfig=True,
        addplot=all_addplots
    )

    ax = axlist[0]

    # Plot the zones as shaded areas
    for lower_boundary, upper_boundary, line_type in zones:
        if line_type == 'support':
            ax.axhspan(lower_boundary, upper_boundary, color='lightgreen', alpha=0.3)
        else:
            ax.axhspan(lower_boundary, upper_boundary, color='lightcoral', alpha=0.3)

    # Display the plot in Streamlit
    st.pyplot(fig)
    # Display the trend message
    if slope > 1:
        st.success(
            f"The market is up trending (Bullish) from {start_date.date()} to {end_date.date()} use those key levels to find your best entry-level making sure to follow the trend given by the Identification algorithm .")
    elif slope < -1:
        st.error(
            f"The market is down trending from {start_date.date()} to {end_date.date()} use those key levels to find your best entry-level making sure to follow the trend given by the Identification algorithm.")
    else:
        st.info(
            f"The market is ranging from {start_date.date()} to {end_date.date()} better to one for a trend creation.")
