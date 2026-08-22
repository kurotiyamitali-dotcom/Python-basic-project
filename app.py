import streamlit as st
import joblib
import numpy as np

st.set_page_config(
    page_title="Breast Cancer Prediction",
    page_icon="🩺"
)

st.title("Breast Cancer Prediction")

try:
    model = joblib.load("optional_best_model.pkl")
    st.success("Model loaded successfully.")
except Exception as e:
    st.error("Model could not be loaded.")
    st.write(e)
    st.stop()

feature_names = [
    "Mean Radius",
    "Mean Texture",
    "Mean Perimeter",
    "Mean Area",
    "Mean Smoothness",
    "Mean Compactness",
    "Mean Concavity",
    "Mean Concave Points",
    "Mean Symmetry",
    "Mean Fractal Dimension",
    "Radius Error",
    "Texture Error",
    "Perimeter Error",
    "Area Error",
    "Smoothness Error",
    "Compactness Error",
    "Concavity Error",
    "Concave Points Error",
    "Symmetry Error",
    "Fractal Dimension Error",
    "Worst Radius",
    "Worst Texture",
    "Worst Perimeter",
    "Worst Area",
    "Worst Smoothness",
    "Worst Compactness",
    "Worst Concavity",
    "Worst Concave Points",
    "Worst Symmetry",
    "Worst Fractal Dimension"
]

default_values = [
    14.0, 20.0, 90.0, 600.0, 0.10,
    0.10, 0.08, 0.05, 0.18, 0.06,
    0.4, 1.0, 2.5, 40.0, 0.007,
    0.02, 0.03, 0.01, 0.02, 0.003,
    16.0, 25.0, 105.0, 800.0, 0.13,
    0.25, 0.25, 0.12, 0.29, 0.08
]

values = []

for i, feature in enumerate(feature_names):
    value = st.number_input(
        feature,
        value=default_values[i],
        format="%.6f"
    )
    values.append(value)

if st.button("Predict"):

    data = np.array(values).reshape(1, 30)

    prediction = model.predict(data)

    if prediction[0] == 0:
        st.error("Prediction: Malignant")
    else:
        st.success("Prediction: Benign")