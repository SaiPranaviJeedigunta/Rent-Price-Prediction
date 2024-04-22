# **Rent Insights Hub**

## Overview
The Rent Insights Hub is a Streamlit web application designed to provide various insights and tools related to rental properties. This application allows users to explore and analyze rental data, visualize trends, predict rental prices, estimate property values.

### Features

**Data Overview:**  Summary statistics and visualization of the dataset. Interactive 3D scatter plot and animated plot. 

**Data Visualization:** Visualization of rental data through scatter plots, bar charts, and line plots. 

**Filtering:** Filter the dataset based on various criteria such as bathrooms, BHK, furnishing status, area type, and city. 

**Analysis:** Property type analysis, location-based analysis, price distribution, time series analysis, and comparative analysis. 

**Price Prediction:** Predict the rental price of a property based on specific criteria such as size, BHK, and bathrooms. 

**Rent Affordability Calculator :** Calculate the affordability of rent based on monthly income and filter available rental properties. 

**Demand Prediction :** Predict the demand for rental properties in different area types. 

**Property Valuation Tool :** Estimate the value of a property based on its features. 

**Predict Rent Trend :** Forecast the future trend of rental prices based on historical data. 

**Analyze Tenant Preferences :** Analyze tenant preferences using clustering analysis. 

**Detect Rent Outliers :** Identify rental properties with prices that significantly deviate from the average. 

**Segment Rental Market :** Segment the rental market based on property size and number of bedrooms using KMeans clustering. 

**Recommend Properties :** Provide personalized recommendations for rental properties based on user preferences. 



# Installation
To run the Cell Image Analyzer locally, you will need to have Python 3.6 or higher installed. Then, you can install the required packages by running:

```bash
pip install -r requirements.txt
```

This will install all the necessary dependencies, including Streamlit, OpenCV, and scikit-image.

# Usage
To start the app, simply run the following command:

```bash
streamlit run app.py
```

# Dataset
The app was developed as a machine learning exercise from the public dataset [House Rent Prediction Dataset](https://www.kaggle.com/datasets/iamsouravbanerjee/house-rent-prediction-dataset). This dataset contains information about rental properties including features such as size, number of bathrooms, BHK, furnishing status, tenant preferred, area type, city, rent, and posted date.

# Libraries Used

Streamlit: A web application framework used for building interactive web applications with Python. \
Pandas: A powerful data manipulation and analysis library. \
Matplotlib: A plotting library for creating static, interactive, and animated visualizations in Python. \
Seaborn: A statistical data visualization library based on Matplotlib, providing a high-level interface for drawing attractive and informative statistical graphics. \
Plotly: An interactive graphing library that allows for the creation of interactive plots and dashboards. \
Scikit-learn: A machine learning library providing simple and efficient tools for data mining and data analysis. \
Requests: A library for making HTTP requests in Python, used for fetching data from URLs. \
Statsmodels: A library for estimating and interpreting statistical models in Python. \
Certifi: For SSL certificate verification. 

