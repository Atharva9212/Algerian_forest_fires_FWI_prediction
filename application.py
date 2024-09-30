import streamlit as st
import pickle
import numpy as np
from sklearn.preprocessing import StandardScaler

# Load ridge regressor and standard scalar pickle
ridge_model = pickle.load(open('models/ridge.pkl', 'rb'))
scaler = pickle.load(open('models/scalar.pkl', 'rb'))

# Streamlit interface
st.title('Wildfire Prediction App')

# Input fields for the model features
st.write("Please enter the data for prediction:")

temperature = st.number_input('Temperature (°C)', value=20.0)
rh = st.number_input('Relative Humidity (%)', value=50.0)
ws = st.number_input('Wind Speed (km/h)', value=10.0)
rain = st.number_input('Rain (mm)', value=0.0)
ffmc = st.number_input('FFMC Index', value=85.0)
dmc = st.number_input('DMC Index', value=15.0)
isi = st.number_input('ISI Index', value=5.0)
classes = st.selectbox('Classes', [0, 1])  # 0 or 1
region = st.selectbox('Region', ['Brjaia Regions', 'sidi - Bel Regions'])

# Map categorical 'Region' to numerical values if needed by the model
region_map = {'Brjaia Regions': 0, 'sidi - Bel Regions': 1}
region_value = region_map[region]

# Prediction button
if st.button('Submit'):
    # Prepare input data
    input_data = np.array([[temperature, rh, ws, rain, ffmc, dmc, isi, classes, region_value]])
    
    # Preprocess the input data using the loaded scaler
    scaled_data = scaler.transform(input_data)

    # Perform prediction
    prediction = ridge_model.predict(scaled_data)
    
    # Display the prediction result
    st.success(f'The predicted wildfire value is: {prediction[0]}')




