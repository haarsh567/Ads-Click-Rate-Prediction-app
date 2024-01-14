import streamlit as st
import numpy as np
import pandas as pd
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Load the RandomForest model with the specified protocol
model = pickle.load(open('model.pkl', 'rb'))


# Function to make predictions
def RDF_prediction(features):
    # Replace 'model' with your actual machine learning model
    prediction = model.predict(features)
    return prediction


# Streamlit app
def main():
    st.title("Ads Click Through Rate Prediction")

    # Collecting user input
    daily_time_spent = st.slider("Daily Time Spent on Site", min_value=0.0, max_value=24.0, value=12.0, step=0.1)
    age = st.slider("Age", min_value=1, max_value=100, value=30)
    area_income = st.number_input("Area Income", min_value=0.0, value=50000.0, step=100.0)
    daily_internet_usage = st.slider("Daily Internet Usage", min_value=0.0, max_value=24.0, value=6.0, step=0.1)
    gender = st.radio("Gender", options=["Male", "Female"])

    # Mapping gender to numerical values
    gender_mapping = {"Male": 1, "Female": 0}
    gender_numeric = gender_mapping.get(gender, 0)

    # Features for prediction
    features = np.array([[daily_time_spent, age, area_income, daily_internet_usage, gender_numeric]])

    if st.button("Predict"):
        # Making prediction
        prediction = RDF_prediction(features)

        # Displaying the prediction
        st.success(f"Prediction: User will {'click' if prediction == 1 else 'not click'} on the ad.")


# Run the Streamlit app
if __name__ == "__main__":
    main()
