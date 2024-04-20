# **Rent Insights Hub**

## Overview
The Rent Insights Hub app is a machine learning-powered tool that allows users to explore insights and visualizations related to rent prices, and predict rent prices based on selected features.
### Features
 **Data Overview:** Provides summary statistics and allows users to view the first 5 rows of the dataset. \
**Data Visualization:** Offers various visualizations such as scatter plots, bar charts, and line plots to visualize different aspects of the dataset. \
**Filtering:** Allows users to filter the dataset based on criteria such as the number of bathrooms, BHK, furnishing status, tenant preferred, area type, and city. \
**Analysis:** Provides analysis of the dataset including property type analysis, location-based analysis, price distribution, time series analysis, and comparative analysis. \
**Price Prediction:** Allows users to predict rent prices based on selected features using a Linear Regression model. 

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
Streamlit: For building the web application interface. \
Pandas: For data manipulation and analysis.\
Matplotlib and Seaborn: For data visualization.\
Plotly Express: For interactive visualizations.\
Scikit-learn: For implementing the Linear Regression model for price prediction.\
Requests: For making HTTP requests to fetch the dataset.\
Certifi: For SSL certificate verification.\
io: For handling file-like objects.
