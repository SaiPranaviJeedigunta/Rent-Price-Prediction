import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
data = pd.DataFrame({
    "Posted On": ["2022-05-18", "2022-05-13", "2022-05-16", "2022-07-04", "2022-05-09"],
    "BHK": [2, 2, 2, 2, 2],
    "Rent": [10000, 20000, 17000, 10000, 7500],
    "Size": [1100, 800, 1000, 800, 850],
    "Floor": ["Ground out of 2", "1 out of 3", "1 out of 3", "1 out of 2", "1 out of 2"],
    "Area Type": ["Super Area", "Super Area", "Super Area", "Super Area", "Carpet Area"],
    "Area Locality": ["Bandel", "Phool Bagan, Kankurgachi", "Salt Lake City Sector 2", "Dumdum Park", "South Dum Dum"],
    "City": ["Kolkata", "Kolkata", "Kolkata", "Kolkata", "Kolkata"],
    "Furnishing Status": ["Unfurnished", "Semi-Furnished", "Semi-Furnished", "Unfurnished", "Unfurnished"],
    "Tenant Preferred": ["Bachelors/Family", "Bachelors/Family", "Bachelors/Family", "Bachelors/Family", "Bachelors"],
    "Bathroom": [2, 1, 1, 1, 1],
    "Point of Contact": ["Contact Owner", "Contact Owner", "Contact Owner", "Contact Owner", "Contact Owner"]
})

# Sidebar title
st.sidebar.title("House Rent Dataset Explorer")

# Sidebar options
option = st.sidebar.selectbox(
    'Select an option:',
    ('Data Overview', 'Data Visualization', 'Filtering', 'Statistics', 'Map Visualization',
     'Price Distribution', 'Property Type Analysis', 'Location-based Analysis', 'Time Series Analysis',
     'Price Prediction', 'Comparative Analysis', 'Heatmap Visualization', 'Data Export')
)

# Main content title
st.title("House Rent Dataset Explorer")

# Display selected option
if option == 'Data Overview':
    st.write("Number of Rows:", data.shape[0])
    st.write("Number of Columns:", data.shape[1])
    st.write("Data Types:", data.dtypes)
    st.write("First 5 Rows:", data.head())

elif option == 'Data Visualization':
    st.subheader("Data Visualization")
    # Scatter plot of Rent vs Size
    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=data, x='Size', y='Rent')
    st.pyplot()

elif option == 'Filtering':
    st.subheader("Filtering")
    # Slider for filtering by number of bathrooms
    bathroom_filter = st.slider('Select number of bathrooms:', min_value=1, max_value=5, value=1)
    filtered_data = data[data['Bathroom'] == bathroom_filter]
    st.write(filtered_data)

elif option == 'Statistics':
    st.subheader("Statistics")
    st.write(data.describe())

elif option == 'Map Visualization':
    st.subheader("Map Visualization")
    # Display a map with property locations
    st.map(data[['Area Locality', 'City']])

elif option == 'Price Distribution':
    st.subheader("Price Distribution")
    plt.hist(data['Rent'])
    st.pyplot()

elif option == 'Property Type Analysis':
    st.subheader("Property Type Analysis")
    property_type_counts = data['Area Type'].value_counts()
    st.write(property_type_counts)

elif option == 'Location-based Analysis':
    st.subheader("Location-based Analysis")
    location_counts = data['Area Locality'].value_counts()
    st.write(location_counts)

elif option == 'Time Series Analysis':
    st.subheader("Time Series Analysis")
    # Line plot of Rent over time
    data['Posted On'] = pd.to_datetime(data['Posted On'])
    plt.figure(figsize=(12, 6))
    plt.plot(data['Posted On'], data['Rent'])
    plt.xlabel('Date')
    plt.ylabel('Rent')
    plt.title('Rent Over Time')
    st.pyplot()

elif option == 'Price Prediction':
    st.subheader("Price Prediction")
    # Example linear regression model for price prediction
    from sklearn.linear_model import LinearRegression
    model = LinearRegression()
    X = data[['BHK', 'Size', 'Bathroom']]
    y = data['Rent']
    model.fit(X, y)
    # Use the model to predict rent for new data

elif option == 'Comparative Analysis':
    st.subheader("Comparative Analysis")
    # Boxplot comparing rent across different furnishing statuses
    plt.figure(figsize=(8, 6))
    sns.boxplot(x='Furnishing Status', y='Rent', data=data)
    plt.xlabel('Furnishing Status')
    plt.ylabel('Rent')
    plt.title('Rent by Furnishing Status')
    st.pyplot()

elif option == 'Heatmap Visualization':
    st.subheader("Heatmap Visualization")
    # Heatmap of correlation matrix
    corr_matrix = data.corr()
    sns.heatmap(corr_matrix, annot=True)
    st.pyplot()

elif option == 'Data Export':
    st.subheader("Data Export")
    # Export the data to a CSV file
    st.write("Exporting data to CSV file...")
    data.to_csv('rent_data_export.csv', index=False)
    st.write("Data exported successfully.")
