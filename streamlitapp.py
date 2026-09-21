
import streamlit as st
import pandas as pd
import joblib

# Load the trained model
model = joblib.load('logi.sav')

st.title('Delivery Delay Prediction')
st.write('Enter the details below to predict if there will be a delivery delay.')

# Create input fields for each feature based on X.columns
delivery_distance = st.number_input('Delivery Distance', min_value=0.0, max_value=100.0, value=25.0)
traffic_congestion = st.slider('Traffic Congestion (1-5)', 1, 5, 3)
weather_condition = st.slider('Weather Condition (1-5)', 1, 5, 3)
delivery_slot = st.slider('Delivery Slot (1-3)', 1, 3, 2)
driver_experience = st.number_input('Driver Experience (years)', min_value=0, max_value=30, value=5)
num_stops = st.number_input('Number of Stops', min_value=0, max_value=20, value=5)
vehicle_age = st.number_input('Vehicle Age (years)', min_value=0, max_value=20, value=5)
road_condition_score = st.slider('Road Condition Score (1-5)', 1, 5, 3)
package_weight = st.number_input('Package Weight (kg)', min_value=0.0, max_value=50.0, value=10.0)
fuel_efficiency = st.number_input('Fuel Efficiency (km/l)', min_value=0.0, max_value=30.0, value=15.0)
warehouse_processing_time = st.number_input('Warehouse Processing Time (minutes)', min_value=0, max_value=200, value=60)

# Create a DataFrame from the inputs
input_data = pd.DataFrame([
    [
        delivery_distance, traffic_congestion, weather_condition, delivery_slot,
        driver_experience, num_stops, vehicle_age, road_condition_score,
        package_weight, fuel_efficiency, warehouse_processing_time
    ]
], columns=[
    'Delivery_Distance', 'Traffic_Congestion', 'Weather_Condition', 'Delivery_Slot',
    'Driver_Experience', 'Num_Stops', 'Vehicle_Age', 'Road_Condition_Score',
    'Package_Weight', 'Fuel_Efficiency', 'Warehouse_Processing_Time'
])

if st.button('Predict Delivery Delay'):
    prediction = model.predict(input_data)
    prediction_proba = model.predict_proba(input_data)

    if prediction[0] == 1:
        st.error(f"Prediction: Delivery WILL be delayed (Probability: {prediction_proba[0][1]:.2f})")
    else:
        st.success(f"Prediction: Delivery will NOT be delayed (Probability: {prediction_proba[0][0]:.2f})")


