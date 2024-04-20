import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import requests
import ssl
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error, r2_score
from io import BytesIO
import certifi

# Set the SSL certificate verification context
ssl_ctx = ssl.create_default_context(cafile=certifi.where())

# Set a custom color scheme
wellerman_palette = ['#393e46', '#00adb5', '#eeeeee', '#ffd369', '#f8b500']
st.set_page_config(page_title="Rent Insights Hub", page_icon="🏠", layout="wide", initial_sidebar_state="expanded")

# Load the dataset
data_url = "https://github.com/SaiPranaviJeedigunta/capstone/raw/main/data/House_Rent_Dataset.csv"

try:
    response = requests.get(data_url, verify=False)
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
    # Sidebar options
    option = st.sidebar.selectbox(
        'Select an option:',
        ('Data Overview', 'Data Visualization', 'Filtering', 'Analysis', 'Price Prediction', 'Data Export')
    )

    # Main content title
    st.title("Rent Insights Hub")

    # Data Overview
    if option == 'Data Overview':
        st.write("Welcome to the Rent Insights Hub! Here, you can find various insights and visualizations related to rent prices.")

        st.write("Summary Statistics:")
        st.write(data.describe())

        with st.expander("View First 5 Rows"):
            st.write(data.head())

        st.sidebar.markdown("This tab provides an overview of the property details dataset, including summary statistics and the option to view the first 5 rows of data.")

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

        st.sidebar.markdown("This tab provides visualizations of the property details dataset, including scatter plots, bar charts, and line plots to visualize various aspects of the data.")

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

        st.sidebar.markdown("This tab allows you to filter the property details dataset based on various criteria, including bathrooms, BHK, Furnishing Status, Tenant Preferred, Area Type, and City.")

    # Analysis
    elif option == 'Analysis':
        st.subheader("Analysis")
        st.write("Property Type Analysis:")
        property_type_counts = data['Area Type'].value_counts()
        st.write(property_type_counts)

        st.write("Location-based Analysis:")
        location_counts = data['Area Locality'].value_counts()
        st.write(location_counts)

        st.write("Price Distribution and Time Series Analysis:")
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        sns.boxplot(y='Rent', data=data, ax=ax1, palette=wellerman_palette)
        ax1.set_title('Price Distribution')
        data['Posted On'] = pd.to_datetime(data['Posted On'])
        for area_type in data['Area Type'].unique():
            subset = data[data['Area Type'] == area_type]
            sns.lineplot(x='Posted On', y='Rent', data=subset, label=area_type, ax=ax2)
        ax2.set_title('Rent Over Time by Area Type')
        ax2.legend(title='Area Type', loc='upper left', bbox_to_anchor=(1, 1))
        st.pyplot(fig)

        st.write("Comparative Analysis:")
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.boxplot(x='Furnishing Status', y='Rent', data=data, ax=ax, palette=wellerman_palette)
        ax.set_xlabel('Furnishing Status')
        ax.set_ylabel('Rent')
        ax.set_title('Rent by Furnishing Status')
        st.pyplot(fig)

        st.sidebar.markdown("This tab provides analysis of the property details dataset, including property type analysis, location-based analysis, price distribution, time series analysis, and comparative analysis.")

    # Price Prediction
    elif option == 'Price Prediction':
        st.subheader("Price Prediction")
        
        # Select a city
        city = st.selectbox('Select a city:', data['City'].unique())

        # Filter data for the selected city
        city_data = data[data['City'] == city]

        # Select a feature for prediction
        prediction_option = st.selectbox('Choose an option', city_data.drop(['Rent', 'City'], axis=1).columns)

        if prediction_option:
            selected_values = st.multiselect(f'Select value(s) for {prediction_option}', city_data[prediction_option].unique())
            
            if selected_values:
                selected_row = city_data[city_data[prediction_option].isin(selected_values)]
                
                # Prepare data for prediction
                X = selected_row.drop('Rent', axis=1)
                y = selected_row['Rent']

                # Encode categorical variables
                le = LabelEncoder()
                for col in X.select_dtypes(include='object'):
                    X[col] = le.fit_transform(X[col])

                # Split the data into training and testing sets
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

                # Train the model
                model = LinearRegression()
                model.fit(X_train, y_train)

                # Make predictions
                y_pred = model.predict(X_test)

                # Display prediction result
                st.write(f"Based on {prediction_option} being '{', '.join(map(str, selected_values))}', the predicted rent price is approximately: ₹{y_pred[0]:.2f}")

                st.write("Please note that this is an estimate and actual prices may vary.")
            else:
                st.info("Please select value(s) for prediction.")
        else:
            st.info("Please select an option for prediction.")

        st.sidebar.markdown("This tab allows you to predict rent prices based on selected features such as city, area locality, etc. The model used for prediction is a Linear Regression model.")
    # Data Export
    elif option == 'Data Export':
        st.subheader("Data Export")
        st.write("Exporting data to CSV file...")
        data.to_csv('rent_data_export.csv', index=False)
        st.write("Data exported successfully.")

        st.sidebar.markdown("This tab allows you to export the property details dataset to a CSV file.")
