import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

# Streamlit Title
st.title("Linear test Regression Analysis on Price Data")

# Read the predefined CSV file (replace 'data.csv' with your filename)
data = pd.read_csv('data.csv', delimiter=';')

# Format the file
data.columns = ['date', 'price']
data['price'] = data['price'].str.replace(',', '.').astype(float)
data['date'] = pd.to_datetime(data['date'], format='%d/%m/%Y')

# Select Year and Month
year = st.selectbox("Select the Year", data['date'].dt.year.unique())
month = st.selectbox("Select the Month", range(1, 13))

# Filter data for the specified month and year
filtered_data = data[(data['date'].dt.year == year) & (data['date'].dt.month == month)]

# Check if data is available for the selected month and year
if filtered_data.empty:
    st.warning(f"No data available for {year}-{month:02d}.")
else:
    # Perform linear regression analysis and display results
    days = filtered_data['date'].dt.day.values
    prices = filtered_data['price'].values

    # Calculate sums for linear regression
    n = len(days)
    sum_x = np.sum(days)
    sum_y = np.sum(prices)
    sum_xy = np.sum(days * prices)
    sum_x_squared = np.sum(days ** 2)
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
        predicted_prices = m * days + b

        # Plot the results using Matplotlib
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.scatter(days, prices, color='blue', label='Original Data')
        ax.plot(days, predicted_prices, color='red', label=f'Linear fit: y = {m:.2f}x + {b:.2f}')
        ax.set_xlabel('Day')
        ax.set_ylabel('Price')
        month_name = pd.to_datetime(f'{year}-{month:02d}-01').strftime('%B')
        ax.set_title(f'Linear Regression on Prices for {month_name} {year}')
        ax.legend()
        ax.grid(True)

        # Display the plot in Streamlit
        st.pyplot(fig)
