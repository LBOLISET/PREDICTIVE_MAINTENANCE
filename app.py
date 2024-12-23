import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

# Assuming your dataset is stored in a CSV file named 'predictive_maintenance_data.csv'
df = pd.read_csv('predictive_maintenance.csv')

# Page Title
st.title('Predictive Maintenance Data Analysis')

# Sidebar with dataset information
st.sidebar.header('Dataset Information')
st.sidebar.text(f"Rows: {df.shape[0]}")
st.sidebar.text(f"Columns: {df.shape[1]}")

# Display a subset of the data using st.dataframe
st.subheader('Subset of Data')
subset_size = st.slider("Number of Rows in Subset", 1, df.shape[0], 10)
subset = df.head(subset_size)
st.dataframe(subset)

# Extracting relevant columns
rotational_speed = df['Rotational speed [rpm]']
tool_wear = df['Tool wear [min]']

# Scatter plot in the main section
st.subheader('Scatter Plot: Impact of Rotational Speed on Tool Wear')

# Plotting the scatter plot
fig, ax = plt.subplots(figsize=(10, 6))
ax.scatter(rotational_speed, tool_wear, alpha=0.5, color='b')

# Adding labels and title
ax.set_title('Impact of Rotational Speed on Tool Wear')
ax.set_xlabel('Rotational Speed [rpm]')
ax.set_ylabel('Tool Wear [min]')

# Display the plot
st.pyplot(fig)

# Optionally, you can display descriptive statistics or additional analysis below the plot
# For example, you could calculate and display correlation
correlation = df['Rotational speed [rpm]'].corr(df['Tool wear [min]'])
st.text(f'Correlation between Rotational Speed and Tool Wear: {correlation}')
