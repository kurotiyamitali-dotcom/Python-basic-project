import streamlit as st
import joblib
import numpy as np

# Load models
linear_model = joblib.load("linear_regression_model.pkl")
linear_scaler = joblib.load("linear_scaler.pkl")

logistic_model = joblib.load("logistic_regression_model.pkl")
logistic_scaler = joblib.load("logistic_scaler.pkl")

knn_model = joblib.load("knn_model.pkl")
knn_scaler = joblib.load("knn_scaler.pkl")

naive_model = joblib.load("naive_bayes_model.pkl")
naive_scaler = joblib.load("naive_scaler.pkl")

st.title("Session 23 - Machine Learning Models")

model_name = st.selectbox(
    "Select Model",
    ["Linear Regression", "Logistic Regression", "KNN", "Naive Bayes"]
)

# ---------------- LINEAR REGRESSION ----------------

if model_name == "Linear Regression":

    st.subheader("Linear Regression")

    med_inc = st.number_input("MedInc", value=3.0)
    house_age = st.number_input("HouseAge", value=25.0)
    ave_rooms = st.number_input("AveRooms", value=5.0)
    ave_bedrms = st.number_input("AveBedrms", value=1.0)
    population = st.number_input("Population", value=1000.0)
    ave_occup = st.number_input("AveOccup", value=3.0)
    latitude = st.number_input("Latitude", value=35.0)
    longitude = st.number_input("Longitude", value=-120.0)

    if st.button("Predict"):

        data = np.array([[
            med_inc,
            house_age,
            ave_rooms,
            ave_bedrms,
            population,
            ave_occup,
            latitude,
            longitude
        ]])

        data = linear_scaler.transform(data)
        prediction = linear_model.predict(data)

        st.write("Predicted Price:", prediction[0])


# ---------------- LOGISTIC REGRESSION ----------------

elif model_name == "Logistic Regression":

    st.subheader("Logistic Regression")

    mean_radius = st.number_input("Mean Radius", value=14.0)
    mean_texture = st.number_input("Mean Texture", value=20.0)
    mean_perimeter = st.number_input("Mean Perimeter", value=90.0)
    mean_area = st.number_input("Mean Area", value=600.0)
    mean_smoothness = st.number_input("Mean Smoothness", value=0.1)
    mean_compactness = st.number_input("Mean Compactness", value=0.1)
    mean_concavity = st.number_input("Mean Concavity", value=0.1)
    mean_concave_points = st.number_input("Mean Concave Points", value=0.05)
    mean_symmetry = st.number_input("Mean Symmetry", value=0.18)
    mean_fractal_dimension = st.number_input("Mean Fractal Dimension", value=0.06)

    radius_error = st.number_input("Radius Error", value=0.4)
    texture_error = st.number_input("Texture Error", value=1.0)
    perimeter_error = st.number_input("Perimeter Error", value=2.5)
    area_error = st.number_input("Area Error", value=40.0)
    smoothness_error = st.number_input("Smoothness Error", value=0.007)
    compactness_error = st.number_input("Compactness Error", value=0.02)
    concavity_error = st.number_input("Concavity Error", value=0.02)
    concave_points_error = st.number_input("Concave Points Error", value=0.01)
    symmetry_error = st.number_input("Symmetry Error", value=0.02)
    fractal_dimension_error = st.number_input("Fractal Dimension Error", value=0.003)

    worst_radius = st.number_input("Worst Radius", value=16.0)
    worst_texture = st.number_input("Worst Texture", value=25.0)
    worst_perimeter = st.number_input("Worst Perimeter", value=105.0)
    worst_area = st.number_input("Worst Area", value=800.0)
    worst_smoothness = st.number_input("Worst Smoothness", value=0.13)
    worst_compactness = st.number_input("Worst Compactness", value=0.25)
    worst_concavity = st.number_input("Worst Concavity", value=0.3)
    worst_concave_points = st.number_input("Worst Concave Points", value=0.12)
    worst_symmetry = st.number_input("Worst Symmetry", value=0.29)
    worst_fractal_dimension = st.number_input("Worst Fractal Dimension", value=0.08)

    if st.button("Predict"):

        data = np.array([[
            mean_radius,
            mean_texture,
            mean_perimeter,
            mean_area,
            mean_smoothness,
            mean_compactness,
            mean_concavity,
            mean_concave_points,
            mean_symmetry,
            mean_fractal_dimension,
            radius_error,
            texture_error,
            perimeter_error,
            area_error,
            smoothness_error,
            compactness_error,
            concavity_error,
            concave_points_error,
            symmetry_error,
            fractal_dimension_error,
            worst_radius,
            worst_texture,
            worst_perimeter,
            worst_area,
            worst_smoothness,
            worst_compactness,
            worst_concavity,
            worst_concave_points,
            worst_symmetry,
            worst_fractal_dimension
        ]])

        data = logistic_scaler.transform(data)
        prediction = logistic_model.predict(data)

        st.write("Prediction:", prediction[0])


# ---------------- KNN ----------------

elif model_name == "KNN":

    st.subheader("KNN")

    mean_radius = st.number_input("KNN Mean Radius", value=14.0)
    mean_texture = st.number_input("KNN Mean Texture", value=20.0)
    mean_perimeter = st.number_input("KNN Mean Perimeter", value=90.0)
    mean_area = st.number_input("KNN Mean Area", value=600.0)
    mean_smoothness = st.number_input("KNN Mean Smoothness", value=0.1)
    mean_compactness = st.number_input("KNN Mean Compactness", value=0.1)
    mean_concavity = st.number_input("KNN Mean Concavity", value=0.1)
    mean_concave_points = st.number_input("KNN Mean Concave Points", value=0.05)
    mean_symmetry = st.number_input("KNN Mean Symmetry", value=0.18)
    mean_fractal_dimension = st.number_input(
        "KNN Mean Fractal Dimension", value=0.06
    )

    radius_error = st.number_input("KNN Radius Error", value=0.4)
    texture_error = st.number_input("KNN Texture Error", value=1.0)
    perimeter_error = st.number_input("KNN Perimeter Error", value=2.5)
    area_error = st.number_input("KNN Area Error", value=40.0)
    smoothness_error = st.number_input("KNN Smoothness Error", value=0.007)
    compactness_error = st.number_input("KNN Compactness Error", value=0.02)
    concavity_error = st.number_input("KNN Concavity Error", value=0.02)
    concave_points_error = st.number_input(
        "KNN Concave Points Error", value=0.01
    )
    symmetry_error = st.number_input("KNN Symmetry Error", value=0.02)
    fractal_dimension_error = st.number_input(
        "KNN Fractal Dimension Error", value=0.003
    )

    worst_radius = st.number_input("KNN Worst Radius", value=16.0)
    worst_texture = st.number_input("KNN Worst Texture", value=25.0)
    worst_perimeter = st.number_input("KNN Worst Perimeter", value=105.0)
    worst_area = st.number_input("KNN Worst Area", value=800.0)
    worst_smoothness = st.number_input("KNN Worst Smoothness", value=0.13)
    worst_compactness = st.number_input("KNN Worst Compactness", value=0.25)
    worst_concavity = st.number_input("KNN Worst Concavity", value=0.3)
    worst_concave_points = st.number_input(
        "KNN Worst Concave Points", value=0.12
    )
    worst_symmetry = st.number_input("KNN Worst Symmetry", value=0.29)
    worst_fractal_dimension = st.number_input(
        "KNN Worst Fractal Dimension", value=0.08
    )

    if st.button("Predict"):

        data = np.array([[
            mean_radius,
            mean_texture,
            mean_perimeter,
            mean_area,
            mean_smoothness,
            mean_compactness,
            mean_concavity,
            mean_concave_points,
            mean_symmetry,
            mean_fractal_dimension,
            radius_error,
            texture_error,
            perimeter_error,
            area_error,
            smoothness_error,
            compactness_error,
            concavity_error,
            concave_points_error,
            symmetry_error,
            fractal_dimension_error,
            worst_radius,
            worst_texture,
            worst_perimeter,
            worst_area,
            worst_smoothness,
            worst_compactness,
            worst_concavity,
            worst_concave_points,
            worst_symmetry,
            worst_fractal_dimension
        ]])

        data = knn_scaler.transform(data)
        prediction = knn_model.predict(data)

        st.write("Prediction:", prediction[0])


# ---------------- NAIVE BAYES ----------------

elif model_name == "Naive Bayes":

    st.subheader("Naive Bayes")

    mean_radius = st.number_input("NB Mean Radius", value=14.0)
    mean_texture = st.number_input("NB Mean Texture", value=20.0)
    mean_perimeter = st.number_input("NB Mean Perimeter", value=90.0)
    mean_area = st.number_input("NB Mean Area", value=600.0)
    mean_smoothness = st.number_input("NB Mean Smoothness", value=0.1)
    mean_compactness = st.number_input("NB Mean Compactness", value=0.1)
    mean_concavity = st.number_input("NB Mean Concavity", value=0.1)
    mean_concave_points = st.number_input("NB Mean Concave Points", value=0.05)
    mean_symmetry = st.number_input("NB Mean Symmetry", value=0.18)
    mean_fractal_dimension = st.number_input(
        "NB Mean Fractal Dimension", value=0.06
    )

    radius_error = st.number_input("NB Radius Error", value=0.4)
    texture_error = st.number_input("NB Texture Error", value=1.0)
    perimeter_error = st.number_input("NB Perimeter Error", value=2.5)
    area_error = st.number_input("NB Area Error", value=40.0)
    smoothness_error = st.number_input("NB Smoothness Error", value=0.007)
    compactness_error = st.number_input("NB Compactness Error", value=0.02)
    concavity_error = st.number_input("NB Concavity Error", value=0.02)
    concave_points_error = st.number_input(
        "NB Concave Points Error", value=0.01
    )
    symmetry_error = st.number_input("NB Symmetry Error", value=0.02)
    fractal_dimension_error = st.number_input(
        "NB Fractal Dimension Error", value=0.003
    )

    worst_radius = st.number_input("NB Worst Radius", value=16.0)
    worst_texture = st.number_input("NB Worst Texture", value=25.0)
    worst_perimeter = st.number_input("NB Worst Perimeter", value=105.0)
    worst_area = st.number_input("NB Worst Area", value=800.0)
    worst_smoothness = st.number_input("NB Worst Smoothness", value=0.13)
    worst_compactness = st.number_input("NB Worst Compactness", value=0.25)
    worst_concavity = st.number_input("NB Worst Concavity", value=0.3)
    worst_concave_points = st.number_input(
        "NB Worst Concave Points", value=0.12
    )
    worst_symmetry = st.number_input("NB Worst Symmetry", value=0.29)
    worst_fractal_dimension = st.number_input(
        "NB Worst Fractal Dimension", value=0.08
    )

    if st.button("Predict"):

        data = np.array([[
            mean_radius,
            mean_texture,
            mean_perimeter,
            mean_area,
            mean_smoothness,
            mean_compactness,
            mean_concavity,
            mean_concave_points,
            mean_symmetry,
            mean_fractal_dimension,
            radius_error,
            texture_error,
            perimeter_error,
            area_error,
            smoothness_error,
            compactness_error,
            concavity_error,
            concave_points_error,
            symmetry_error,
            fractal_dimension_error,
            worst_radius,
            worst_texture,
            worst_perimeter,
            worst_area,
            worst_smoothness,
            worst_compactness,
            worst_concavity,
            worst_concave_points,
            worst_symmetry,
            worst_fractal_dimension
        ]])

        data = naive_scaler.transform(data)
        prediction = naive_model.predict(data)

        st.write("Prediction:", prediction[0])