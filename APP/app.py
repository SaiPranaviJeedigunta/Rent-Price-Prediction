import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import requests
import plotly.express as px
from io import StringIO, BytesIO
from urllib.error import URLError

# Set a custom color scheme
custom_theme = {
    "primaryColor": "#005599",
    "secondaryBackgroundColor": "#ff9900",
    "textColor": "#333333",
    "font": "sans-serif"
}

# Load the dataset
data_url = "https://github.com/SaiPranaviJeedigunta/capstone/raw/main/data/House_Rent_Dataset.csv"

try:
    response = requests.get(data_url)
    response.raise_for_status()  # Raise an exception for HTTP errors (e.g., 404, 500)
    data = pd.read_csv(BytesIO(response.content))
except requests.exceptions.HTTPError as errh:
    st.error(f"HTTP Error: {errh}")
except requests.exceptions.ConnectionError as errc:
    st.error(f"Error Connecting: {errc}")
except requests.exceptions.Timeout as errt:
    st.error(f"Timeout Error: {errt}")
except requests.exceptions.RequestException as err:
    st.error(f"Other Request Exception: {err}")
except pd.errors.ParserError as perr:
    st.error(f"Parser Error: {perr}")
else:
    # Set the custom theme
    st.set_page_config(layout="wide")
    st.markdown(f"""
    <style>
        .reportview-container .main .block-container {{
            max-width: 1200px;
            padding-top: 2rem;
            padding-right: 2rem;
            padding-left: 2rem;
            padding-bottom: 2rem;
        }}
    </style>
    """, unsafe_allow_html=True)

    # Sidebar title
    st.sidebar.title("Rent Insights Hub")

    # Sidebar options
    option = st.sidebar.selectbox(
        'Select an option:',
        ('Data Overview', 'Data Visualization', 'Filtering', 'Price Distribution',
         'Property Type Analysis', 'Location-based Analysis', 'Time Series Analysis',
         'Price Prediction', 'Comparative Analysis', 'Heatmap Visualization', 'Data Export')
    )
   # Main content title
    st.title("Rent Insights Hub")

    # Data Overview
    if option == 'Data Overview':
        st.subheader("Data Overview")
        st.write("Welcome to the Rent Insights Hub! Here, you can find various insights and visualizations related to rent prices.")

        st.write("Summary Statistics:")
        st.write(data.describe())

        with st.expander("View First 5 Rows"):
            st.write(data.head())

    # Data Visualization
    elif option == 'Data Visualization':
        st.subheader("Data Visualization")
        st.write("This scatter plot shows the relationship between Size and Rent.")
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.scatterplot(data=data, x='Size', y='Rent', ax=ax)
        st.pyplot(fig)

        st.write("This bar chart shows the count of each Furnishing Status.")
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.countplot(x='Furnishing Status', data=data, ax=ax)
        plt.xlabel('Furnishing Status')
        plt.ylabel('Count')
        plt.title('Furnishing Status Count')
        st.pyplot(fig)

        st.write("This line plot shows the trend of Rent over time.")
        data['Posted On'] = pd.to_datetime(data['Posted On'])
        fig = px.line(data, x='Posted On', y='Rent', title='Rent Over Time')
        st.plotly_chart(fig)

    # Filtering
    elif option == 'Filtering':
        st.subheader("Filtering")
        st.write("Filter the data based on the number of bathrooms and BHK.")
        bathroom_filter = st.slider('Select number of bathrooms:', min_value=1, max_value=5, value=1)
        bhk_filter = st.slider('Select number of BHK:', min_value=1, max_value=5, value=1)
        filtered_data = data[(data['Bathroom'] == bathroom_filter) & (data['BHK'] == bhk_filter)]
        st.write(filtered_data)

        st.write("Filter the data based on Furnishing Status and Tenant Preferred.")
        furnishing_status_filter = st.multiselect('Select Furnishing Status:', data['Furnishing Status'].unique())
        tenant_preferred_filter = st.multiselect('Select Tenant Preferred:', data['Tenant Preferred'].unique())
        filtered_data = data[(data['Furnishing Status'].isin(furnishing_status_filter)) & (data['Tenant Preferred'].isin(tenant_preferred_filter))]
        st.write(filtered_data)

        st.write("Filter the data based on Area Type and City.")
        area_type_filter = st.multiselect('Select Area Type:', data['Area Type'].unique())
        city_filter = st.multiselect('Select City:', data['City'].unique())
        filtered_data = data[(data['Area Type'].isin(area_type_filter)) & (data['City'].isin(city_filter))]
        st.write(filtered_data)

    # Price Distribution
    elif option == 'Price Distribution':
        st.subheader("Price Distribution")
        st.write("This box plot shows the distribution of Rent prices.")
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.boxplot(y='Rent', data=data, ax=ax)
        st.pyplot(fig)

    # Property Type Analysis
    elif option == 'Property Type Analysis':
        st.subheader("Property Type Analysis")
        property_type_counts = data['Area Type'].value_counts()
        st.write(property_type_counts)

    # Location-based Analysis
    elif option == 'Location-based Analysis':
        st.subheader("Location-based Analysis")
        location_counts = data['Area Locality'].value_counts()
        st.write(location_counts)

        # Time Series Analysis
    elif option == 'Time Series Analysis':
        st.subheader("Time Series Analysis")
        data['Posted On'] = pd.to_datetime(data['Posted On'])
        fig, ax = plt.subplots(figsize=(12, 6))
        for area_type in data['Area Type'].unique():
            subset = data[data['Area Type'] == area_type]
            sns.lineplot(x='Posted On', y='Rent', data=subset, label=area_type)
        plt.xlabel('Date')
        plt.ylabel('Rent')
        plt.title('Rent Over Time by Area Type')
        plt.legend(title='Area Type', loc='upper left', bbox_to_anchor=(1, 1))
        st.pyplot(fig)

    # Price Prediction
    elif option == 'Price Prediction':
        st.subheader("Price Prediction")
        st.write("Price prediction functionality goes here.")

    # Comparative Analysis
    elif option == 'Comparative Analysis':
        st.subheader("Comparative Analysis")
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.boxplot(x='Furnishing Status', y='Rent', data=data, ax=ax)
        plt.xlabel('Furnishing Status')
        plt.ylabel('Rent')
        plt.title('Rent by Furnishing Status')
        st.pyplot(fig)

    # Heatmap Visualization
    elif option == 'Heatmap Visualization':
        st.subheader("Heatmap Visualization")
        st.write("This heatmap shows the correlation matrix of the dataset.")
    
        # Handle missing values
        data_cleaned = data.dropna()
        if len(data_cleaned) > 0:
            # Compute the correlation matrix
            corr_matrix = data_cleaned.corr()
            fig, ax = plt.subplots(figsize=(10, 8))
            sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f", ax=ax)
            st.pyplot(fig)
        else:
            st.warning("No data available after removing missing values. Please check your data.")

    # Data Export
    elif option == 'Data Export':
        st.subheader("Data Export")
        st.write("Exporting data to CSV file...")
        data.to_csv('rent_data_export.csv', index=False)
        st.write("Data exported successfully.")

   
