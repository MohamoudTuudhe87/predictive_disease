import streamlit as st
import joblib
import numpy as np
import pandas as pd
import os

# Set page title and layout
st.set_page_config(page_title="Chronic Disease Predictor", layout="centered")

# Sidebar - Disease selection
st.sidebar.title("Select Disease")
disease = st.sidebar.radio("Choose a disease", ["Liver", "Heart", "Diabetes"])

# Load models once (cached)
@st.cache_resource
def load_model(name):
    return joblib.load(f"models/{name}_model.pkl")

liver_model = load_model("liver")
heart_model = load_model("heart")
diabetes_model = load_model("diabetes")

# Header
st.title(f"{disease} Disease Prediction")
st.write("Enter patient details below:")

# Collect input based on selected disease
input_data = {}

if disease == "Liver":
    age = st.number_input("Age", min_value=0)
    gender = st.selectbox("Gender", ["Male", "Female"])
    tb = st.number_input("Total Bilirubin")
    db = st.number_input("Direct Bilirubin")
    alkphos = st.number_input("Alkphos Alkaline Phosphotase")
    sgpt = st.number_input("Sgpt Alamine Aminotransferase")
    sgot = st.number_input("Sgot Aspartate Aminotransferase")
    tp = st.number_input("Total Proteins")
    alb = st.number_input("ALB Albumin")
    ag_ratio = st.number_input("A/G Ratio Albumin and Globulin Ratio")

    input_data = {
        'Age of the patient': age,
        'Gender of the patient': 1 if gender == "Male" else 0,
        'Total Bilirubin': tb,
        'Direct Bilirubin': db,
        'Alkphos Alkaline Phosphotase': alkphos,
        'Sgpt Alamine Aminotransferase': sgpt,
        'Sgot Aspartate Aminotransferase': sgot,
        'Total Protiens': tp,
        'ALB Albumin': alb,
        'A/G Ratio Albumin and Globulin Ratio': ag_ratio
    }

elif disease == "Heart":
    age = st.number_input("Age", min_value=0)
    sex = st.selectbox("Sex", ["Male", "Female"])
    cp = st.selectbox("Chest Pain Type", [0, 1, 2, 3])
    trestbps = st.number_input("Resting Blood Pressure")
    chol = st.number_input("Serum Cholesterol")
    fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl", [0, 1])
    restecg = st.selectbox("Resting ECG", [0, 1, 2])
    thalach = st.number_input("Max Heart Rate Achieved")
    exang = st.selectbox("Exercise Induced Angina", [0, 1])
    oldpeak = st.number_input("ST Depression")
    slope = st.selectbox("Slope of Peak Exercise ST Segment", [0, 1, 2])
    ca = st.selectbox("Number of Major Vessels", [0, 1, 2, 3])
    thal = st.selectbox("Thalassemia", [0, 1, 2])

    input_data = {
        'age': age,
        'sex': 1 if gender == "Male" else 0,
        'cp': cp,
        'trestbps': trestbps,
        'chol': chol,
        'fbs': fbs,
        'restecg': restecg,
        'thalach': thalach,
        'exang': exang,
        'oldpeak': oldpeak,
        'slope': slope,
        'ca': ca,
        'thal': thal
    }

elif disease == "Diabetes":
    pregnancies = st.number_input("Pregnancies", min_value=0)
    glucose = st.number_input("Glucose Level")
    blood_pressure = st.number_input("Blood Pressure")
    skin_thickness = st.number_input("Skin Thickness")
    insulin = st.number_input("Insulin Level")
    bmi = st.number_input("BMI")
    dpf = st.number_input("Diabetes Pedigree Function")
    age = st.number_input("Age", min_value=0)

    input_data = {
        'Pregnancies': pregnancies,
        'Glucose': glucose,
        'BloodPressure': blood_pressure,
        'SkinThickness': skin_thickness,
        'Insulin': insulin,
        'BMI': bmi,
        'DiabetesPedigreeFunction': dpf,
        'Age': age
    }

# Predict Button
if st.button("Predict"):
    df = pd.DataFrame([input_data])

    if disease == "Liver":
        prediction = liver_model.predict(df)[0]
        result = "Likely Liver Disease" if prediction == 1 else "No Liver Disease"
    elif disease == "Heart":
        prediction = heart_model.predict(df)[0]
        result = "Likely Heart Disease" if prediction == 1 else "No Heart Disease"
    elif disease == "Diabetes":
        prediction = diabetes_model.predict(df)[0]
        result = "Likely Diabetic" if prediction == 1 else "Not Diabetic"

    st.success(f"Prediction: {result}")

    # Optional: Save input to CSV
    df['Prediction'] = result
    df.to_csv("user_inputs.csv", mode='a', header=not os.path.exists("user_inputs.csv"), index=False)
    st.info("Input and prediction saved.")

# Show model info
st.sidebar.markdown("### Model Info")
st.sidebar.write("Trained using Machine Learning Models")
st.sidebar.write("Models used: Random Forest / Logistic Regression / SVM")