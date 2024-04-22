import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import requests
import ssl
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error, r2_score
from io import BytesIO
import certifi
import statsmodels.api as sm
from sklearn.cluster import KMeans
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import NearestNeighbors
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from io import BytesIO
import certifi


# Set the SSL certificate verification context
ssl_ctx = ssl.create_default_context(cafile=certifi.where())

# Set a custom color scheme
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
    
     # Train a Random Forest Regressor model for property valuation
    X = data[['Size', 'BHK', 'Bathroom']]
    y = data['Rent']
    model = RandomForestRegressor(random_state=42)
    model.fit(X, y)

     # Get feature importances
    feature_importances = model.feature_importances_

    # Create a DataFrame to store feature importances
    importance_df = pd.DataFrame({'Feature': X.columns, 'Importance': feature_importances})
    importance_df = importance_df.sort_values(by='Importance', ascending=False)

    
    # Demand Prediction Logic
    def predict_demand(data):
      data['Posted On'] = pd.to_datetime(data['Posted On'])
      demand_data = data.groupby(['Area Type', pd.Grouper(key='Posted On', freq='M')])['Rent'].count().reset_index()
      avg_demand = demand_data.groupby('Area Type')['Rent'].mean().reset_index()
      avg_demand.columns = ['Area Type', 'Average Demand']
      return avg_demand

   # Property valuation logic
    def property_valuation(data, property_features):
        # Handling missing values
        property_features = [0 if val is None else val for val in property_features]

        # Make prediction
        predicted_rent = model.predict([property_features])[0]
        return predicted_rent



#analysis
    def calculate_price_index(data):
    # Calculate the mean rent for each area type and city
      mean_rent = data.groupby(['Area Type', 'City'])['Rent'].mean().reset_index()

    # Calculate the price index
      mean_rent['Price Index'] = mean_rent.groupby('Area Type')['Rent'].transform(lambda x: (x / x.mean()) * 100)

      return mean_rent[['Area Type', 'City', 'Price Index']]


    # Predict Rent Trend
    def predict_rent_trend(data):
    # Convert 'Posted On' to datetime and set it as the index
      data['Posted On'] = pd.to_datetime(data['Posted On'])
      data.set_index('Posted On', inplace=True)
    
    # Resample data by month and calculate the mean rent for each month
      monthly_rent_mean = data['Rent'].resample('M').mean()
    
    # Fit a SARIMA model to the monthly rent data
      sarima_model = sm.tsa.SARIMAX(monthly_rent_mean, order=(1, 1, 1), seasonal_order=(1, 1, 1, 12))
      sarima_result = sarima_model.fit()
    
    # Predict future rent trends
      forecast = sarima_result.predict(start=monthly_rent_mean.index[-1], end=monthly_rent_mean.index[-1] + pd.DateOffset(months=12))
    
      return forecast

# Analyze Tenant Preferences
    def analyze_tenant_preferences(data):
    # Encode categorical variables
      data_encoded = pd.get_dummies(data[['Furnishing Status', 'Area Type']])
    
    # Fit KMeans clustering model
      kmeans = KMeans(n_clusters=3, random_state=42)
      kmeans.fit(data_encoded)
    
    # Get cluster labels
      cluster_labels = kmeans.labels_
    
      return cluster_labels

# Detect Rent Outliers
    def detect_rent_outliers(data):
    # Fit Isolation Forest model
      isolation_forest = IsolationForest(contamination=0.05, random_state=42)
      outliers = isolation_forest.fit_predict(data[['Rent']])
    
      return outliers

# Segment Rental Market
    def segment_rental_market(data):
    # Apply KMeans clustering on relevant features
      kmeans = KMeans(n_clusters=4, random_state=42)
      data['Cluster'] = kmeans.fit_predict(data[['Size', 'BHK']])
    
      return data

# Recommend Properties
    def recommend_properties(data, user_preferences):
    # Fit Nearest Neighbors model
      nn_model = NearestNeighbors(n_neighbors=5, algorithm='ball_tree')
      nn_model.fit(data[['Size', 'BHK', 'Rent']])
    
    # Find nearest neighbors based on user preferences
      _, indices = nn_model.kneighbors([user_preferences])
    
    # Get recommended properties
      recommended_properties = data.iloc[indices[0]]
    
      return recommended_properties


     
     # Sidebar options
    option = st.sidebar.selectbox(
    'Select an option:',
    ('Data Overview', 'Data Visualization', 'Filtering', 'Analysis', 'Price Prediction', 'Rent Affordability Calculator', 'Demand Prediction', 'Property Valuation Tool', 'Predict Rent Trend', 'Analyze Tenant Preferences', 'Detect Rent Outliers', 'Segment Rental Market', 'Recommend Properties')
)

# Sidebar descriptions
    sidebar_descriptions = {
    "Data Overview": "Explore a summary of the dataset and visualizations to understand key insights at a glance.",
    "Data Visualization": "Explore visualizations of the dataset.",
    "Filtering": "Filter the dataset based on various criteria.",
    "Analysis": "Gain deeper insights into the rental market with detailed analysis.",
    "Price Prediction": "Predict the rental price of properties based on specific criteria, helping you make informed decisions about renting or leasing.",
    "Rent Affordability Calculator": "Determine rental options that fit within your budget based on your monthly income, helping you find affordable rental properties.",
    "Demand Prediction": "Predict the demand for rental properties in different area types to understand market trends and make informed decisions.",
    "Property Valuation Tool": "Estimate the rental value of a property based on its features to make informed decisions about pricing and investment.",
    "Predict Rent Trend": "Forecast the future trend of rental prices based on historical data to anticipate market changes.",
    "Analyze Tenant Preferences": "Gain insights into tenant preferences using clustering analysis to understand popular rental property features.",
    "Detect Rent Outliers": "Identify rental properties with unusual rental prices compared to the average, helping you make informed decisions.",
    "Segment Rental Market": "Explore how the rental market is segmented based on property size and number of bedrooms, providing insights into different rental property categories.",
    "Recommend Properties": "Get personalized recommendations for rental properties based on your preferences, helping you find the perfect rental property that matches your criteria.",
}

# Display description based on selected option
    if option:
      st.sidebar.write(sidebar_descriptions[option])
    else:
      st.sidebar.write("No description available for this option.")
  
    # Main content title
    st.title("Rent Insights Hub")

    # Data Overview
    if option == 'Data Overview':
        st.write("Welcome to the Rent Insights Hub! Here, you can find various insights and visualizations related to rent prices.")

        # Scatter plot
        st.write("3D Scatter Plot:")
        fig = px.scatter_3d(data, x='Size', y='BHK', z='Rent', color='Furnishing Status')
        st.plotly_chart(fig)
        
        st.write("Summary Statistics:")
        st.write(data.describe())


         # Animated plot
        st.write("Animated Plot:")
        data['Posted On'] = pd.to_datetime(data['Posted On'])
        data['Year'] = data['Posted On'].dt.year
        data['Month'] = data['Posted On'].dt.month
        fig = px.scatter(data, x='Size', y='BHK', animation_frame='Year', animation_group='Month', 
                     color='Area Type', range_x=[0, 5000], range_y=[0, 10])
        st.plotly_chart(fig)

        # Plot feature importance as a pie chart
        fig, ax = plt.subplots(figsize=(4, 4))
        ax.pie(importance_df['Importance'], labels=importance_df['Feature'], startangle=90, counterclock=False, autopct='%1.1f%%')
        ax.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle
        ax.set_title('Feature Importance')
        st.pyplot(fig)

        #st.sidebar.markdown("This tab provides an overview of the property details dataset, including summary statistics and the option to view the first 5 rows of data.")

      
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

    
        #st.sidebar.markdown("This tab provides visualizations of the property details dataset, including scatter plots, bar charts, and line plots to visualize various aspects of the data.")

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

        #st.sidebar.markdown("This tab allows you to filter the property details dataset based on various criteria, including bathrooms, BHK, Furnishing Status, Tenant Preferred, Area Type, and City.")

    # Analysis
    elif option == 'Analysis':
      st.subheader("Analysis")
      st.write("Property Type Analysis:")
      st.write("Perform detailed analysis of the dataset to extract meaningful insights.")
      property_type_counts = data['Area Type'].value_counts()
      st.write(property_type_counts)

      st.write("Location-based Analysis:")
      location_counts = data['Area Locality'].value_counts()
      st.write(location_counts)

      st.write("Price Distribution and Time Series Analysis:")
      fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
      sns.boxplot(y='Rent', data=data, ax=ax1)
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
      sns.boxplot(x='Furnishing Status', y='Rent', data=data, ax=ax)
      ax.set_xlabel('Furnishing Status')
      ax.set_ylabel('Rent')
      ax.set_title('Rent by Furnishing Status')
      st.pyplot(fig)

      st.write("Property Price Index per Area Type and City:")
      price_index = calculate_price_index(data)
      st.write(price_index)

      #st.sidebar.markdown("This tab provides analysis of the property details dataset, including property type analysis, location-based analysis, price distribution, time series analysis, and comparative analysis.")


    # Price Prediction
    elif option == 'Price Prediction':
      st.subheader("Price Prediction")
      st.write("Predict the rental price of a property in a selected city based on specific criteria. Choose a city and a feature to filter the data, and the model will predict the rent price, helping you make informed decisions about renting or leasing properties.")

       
      city = st.selectbox('Select a city:', data['City'].unique())
      city_data = data[data['City'] == city]
      prediction_option = st.selectbox('Choose an option', city_data.drop(['Rent', 'City'], axis=1).columns)
      if prediction_option:
          selected_values = st.multiselect(f'Select value(s) for {prediction_option}', city_data[prediction_option].unique())
          if selected_values:
              selected_row = city_data[city_data[prediction_option].isin(selected_values)]
              X = selected_row.drop('Rent', axis=1)
              y = selected_row['Rent']
              le = LabelEncoder()
              for col in X.select_dtypes(include='object'):
                  X[col] = le.fit_transform(X[col])
              X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
              model = LinearRegression()
              model.fit(X_train, y_train)
              y_pred = model.predict(X_test)

            # Use a Seaborn palette for the boxplot
              sns.set_palette("husl")
              st.write(f"Based on {prediction_option} being '{', '.join(map(str, selected_values))}', the predicted rent price is approximately: ₹{y_pred[0]:.2f} per month.")



# Rent Affordability Calculator
    elif option == 'Rent Affordability Calculator':
      st.subheader("Rent Affordability Calculator")
      st.write("This Rent Affordability Calculator helps you determine the rental options that fit within your budget based on your monthly income. Simply enter your monthly income and select a city, and the calculator will filter the available rental properties to show you those that you can afford.")

    # User input
      monthly_income = st.number_input("Enter your monthly income (₹):", value=0, step=1)
      city = st.selectbox('Select a city:', data['City'].unique())

    # Filter data based on income and city
      filtered_data = data[data['City'] == city]
      max_affordable_rent = monthly_income * 0.3  # Assuming 30% of income can be allocated to rent
      affordable_properties = filtered_data[filtered_data['Rent'] <= max_affordable_rent]

    # Display filtered options
      st.write(f"Based on your monthly income of ₹{monthly_income}, you can afford the following rental options in {city}:")
      st.write(affordable_properties[['Posted On', 'Rent', 'BHK', 'Size', 'Floor', 'Area Locality', 'Furnishing Status', 'Tenant Preferred', 'Bathroom']])


    # Demand Prediction
    elif option == 'Demand Prediction':
        st.write("Predict the demand for rental properties in different area types.")
        st.subheader("Demand Prediction")
        avg_demand = predict_demand(data)
        st.write("Average Demand per Area Type:")
        st.write(avg_demand)
        # Sample code for an interactive plot showing the predicted demand for rental properties
        demand_data = predict_demand(data)
        fig = px.bar(demand_data, x='Area Type', y='Average Demand', color='Area Type')
        st.plotly_chart(fig)

    # Property Valuation Tool
    elif option == 'Property Valuation Tool':
      st.write("Estimate the value of a property based on its features.")
      st.subheader("Property Valuation Tool")
      
      property_size = st.number_input("Enter property size:", step=1, value=1)
      property_bhk = st.number_input("Enter number of bedrooms (BHK):", step=1, value=1)
      property_bathroom = st.number_input("Enter number of bathrooms:", step=1, value=1)
    
    # Update property_features with the input values
      property_features = [property_size, property_bhk, property_bathroom]
    
      predicted_rent = property_valuation(data, property_features)
      st.write(f"Based on the provided property features, the predicted rent price is approximately: ₹{predicted_rent:.2f} per month")

    # Filter properties based on predicted rent price
      filtered_properties = data[data['Rent'] <= predicted_rent]
      st.write("Properties Available based on Property Valuation:")
      st.write(filtered_properties)

    # Update property_features with the input values
      property_features = [property_size, property_bhk, property_bathroom]
    
      predicted_rent = property_valuation(data, property_features)
    
# Predict Rent Trend
    elif option == 'Predict Rent Trend':
        st.write("Forecast the future trend of rental prices based on historical data.")
        st.subheader("Predict Rent Trend")
        forecast = predict_rent_trend(data)
        st.write("Forecasted Rent Trends:")
        st.write(forecast)
        st.write("The forecasted rent trends provide an insight into the expected changes in rent prices over the next few months. This information can be valuable for tenants and landlords alike, helping them make informed decisions about renting or leasing properties.")

        
    # Analyze Tenant Preferences
    elif option == 'Analyze Tenant Preferences':
      st.subheader("Analyze Tenant Preferences")
      st.write("In this section, we provide insights into tenant preferences using clustering analysis. We've clustered properties based on their furnishing status and area type, grouping them into user-friendly categories. This helps you understand popular preferences among tenants, such as furnished vs. unfurnished properties or preferences for specific area types. You can explore these clusters to gain valuable insights into tenant preferences and make informed decisions about your rental properties.")

    # Analyze tenant preferences using KMeans clustering
      cluster_labels = analyze_tenant_preferences(data)

    # Define user-friendly cluster labels
      cluster_label_mapping = {
          0: "Family-oriented",
          1: "Budget-conscious",
          2: "Luxury seekers"
      }

    # Map cluster labels to user-friendly labels
      data['Cluster Label'] = cluster_labels
      data['Cluster Label'] = data['Cluster Label'].map(cluster_label_mapping)

      st.write("Cluster Labels:")
      st.write(data['Cluster Label'].value_counts())

      st.write("Sample of Data with Cluster Labels:")
      st.write(data[['Area Locality', 'Furnishing Status', 'Area Type', 'Size', 'BHK', 'Bathroom', 'Tenant Preferred', 'Cluster Label']].head(50))

      #st.sidebar.markdown("This tab analyzes tenant preferences using KMeans clustering, providing user-friendly cluster labels and a sample of data with cluster labels.")


    # Detect Rent Outliers
    elif option == 'Detect Rent Outliers':
      st.subheader("Detect Rent Outliers")
      outliers = detect_rent_outliers(data)

      st.write("Outliers are rental properties with prices that significantly deviate from the average. They can provide valuable insights into unusual rental price patterns.")
      st.write("Outlier Labels (-1 for outliers, 1 for inliers):")

    # Add a description explaining what 1 and -1 mean
      st.write("In this table, a label of -1 indicates an outlier, while a label of 1 indicates a normal, non-outlying data point.")
    # Create a DataFrame with index, outlier label, and property details
      outliers_df = pd.DataFrame({
          'Index': data.index,
          'Outlier': outliers
      })

    # Merge the outliers_df with the original data to get property details
      outliers_df = outliers_df.merge(data, left_on='Index', right_index=True)

    # Display the table with index, outlier label, and property information
      st.write(outliers_df.reset_index(drop=True))

# Description above the graph
      st.write("The histogram below shows the distribution of rental prices. Outliers, which are rental prices significantly different from the average, are highlighted in red. These outliers can provide insights into unusual rental price patterns.")
      st.write("Outliers are indicated by the red points on the histogram. They represent rental prices that deviate significantly from the average prices in the dataset.")
      st.write("The histogram helps you visualize the distribution of rental prices and identify any unusual patterns or extreme values.")

    # Visualize the distribution of rental prices with outliers highlighted
      fig, ax = plt.subplots(figsize=(10, 6))
      sns.histplot(data['Rent'], bins=30, kde=True, ax=ax)
      outliers_idx = np.where(outliers == -1)[0]
      ax.scatter(outliers_idx, np.zeros_like(outliers_idx), color='red', label='Outliers')
      ax.legend()
      ax.set_title('Distribution of Rental Prices with Outliers')
      st.pyplot(fig)


    # Segment Rental Market
    elif option == 'Segment Rental Market':
        st.subheader("Segment Rental Market")
        segmented_data = segment_rental_market(data)
        st.write("In this section, we segment the rental market based on property size and number of bedrooms (BHK) using KMeans clustering. This helps us categorize properties into distinct groups, providing insights into the rental market's diversity.")
        st.write("Here are the segmented clusters along with the corresponding properties:")
        st.write("Segmented Data with Cluster Labels:")
        st.write(segmented_data)
        

        
    # Recommend Properties
    elif option == 'Recommend Properties':
      st.write("Provide personalized recommendations for rental properties based on user preferences.")
      st.subheader("Recommend Properties")
     
      user_preferences = [st.number_input("Enter property size:", step=1, value=1),
                          st.number_input("Enter number of bedrooms (BHK):", step=1, value=1),
                          st.number_input("Enter rent budget:", step=1, value=1)]
      recommended_properties = recommend_properties(data, user_preferences)
      st.write("Using your preferences for property size, number of bedrooms (BHK), and rent budget, we recommend properties that closely match your criteria. These recommendations are based on the nearest neighbors to your preferences in our dataset.")
      st.write("Here are the recommended properties:")
      st.write(recommended_properties)
    
