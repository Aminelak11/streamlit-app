import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

# Streamlit Title
st.title("Market Trend Using Least Squares Method for XAUUSD Market")

# Read the predefined CSV file (replace 'data.csv' with your filename)
data = pd.read_csv('data.csv', delimiter=';')

# Format the file
data.columns = ['date', 'price']
data['price'] = data['price'].str.replace(',', '.').astype(float)
data['date'] = pd.to_datetime(data['date'], format='%d/%m/%Y')

# Select Date
selected_date = st.date_input(
    "Select an end date",
    value=data['date'].max().date(),
    min_value=data['date'].min().date(),
    max_value=data['date'].max().date()
)

# Convert selected_date to Pandas Timestamp
selected_date = pd.to_datetime(selected_date)

# Create a date range from selected_date - 31 days to selected_date - 1 day
start_date = selected_date - pd.Timedelta(days=31)
end_date = selected_date - pd.Timedelta(days=1)

# Filter data between start_date and end_date
filtered_data = data[(data['date'] >= start_date) & (data['date'] <= end_date)]

# Check if data is available
if filtered_data.empty:
    st.warning(f"No data available from {start_date.date()} to {end_date.date()}.")
else:
    # Prepare data for regression
    # Use days since start_date as x-values
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
        ax.set_title(f'Linear Regression on Prices from {start_date.date()} to {end_date.date()}')  # Corrected syntax
        ax.legend()
        ax.grid(True)

        # Display the plot in Streamlit
        st.pyplot(fig)

        # Display the trend message with updated colors
        if m > 1:
            st.success(f"The trend is up from {start_date.date()} to {end_date.date()}.")
        elif m < -1:
            st.error(f"The trend is down from {start_date.date()} to {end_date.date()}.")
        else:
            st.info(f"The market is ranging from {start_date.date()} to {end_date.date()}.")
