# app.py
import streamlit as st
import joblib
import numpy as np

# Load the model
model = joblib.load("iris_model.pkl")

st.title("Iris Flower Prediction App")

# Input fields
sepal_length = st.slider("Sepal Length", 4.0, 8.0, 5.0)
sepal_width = st.slider("Sepal Width", 2.0, 4.5, 3.0)
petal_length = st.slider("Petal Length", 1.0, 7.0, 4.0)
petal_width = st.slider("Petal Width", 0.1, 2.5, 1.0)

# Predict button
if st.button("Predict"):
    input_data = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
    prediction = model.predict(input_data)[0]
    target_names = ["Setosa", "Versicolor", "Virginica"]
    st.success(f"Predicted Iris Species: {target_names[prediction]}")
