import streamlit as st
import pandas as pd
import numpy as np
import joblib

# ===========================
# LOAD MODEL & SCALER
# ===========================
model = joblib.load("model_voting.pkl")
scaler = joblib.load("scaler.pkl")

st.title("❤️ Heart Disease Prediction App")
st.write("Aplikasi ini menggunakan Voting Classifier (RF + SVM) untuk memprediksi risiko penyakit jantung.")

# ===========================
# FORM INPUT DATA
# ===========================
st.header("Masukkan Data Pasien")

age = st.number_input("Age", 18, 100)
sex = st.selectbox("Sex (0 = Female, 1 = Male)", [0, 1])
cp = st.selectbox("Chest Pain Type (0–3)", [0, 1, 2, 3])
restbp = st.number_input("Resting Blood Pressure", 80, 200)
chol = st.number_input("Cholesterol", 100, 600)
fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl (1/0)", [0, 1])
restecg = st.selectbox("Resting ECG (0–2)", [0, 1, 2])
maxhr = st.number_input("Max Heart Rate", 60, 220)
exang = st.selectbox("Exercise-induced angina (1/0)", [0, 1])
oldpeak = st.number_input("Oldpeak", -2.0, 7.0, step=0.1)
slope = st.selectbox("Slope (0–2)", [0, 1, 2])
ca = st.selectbox("Major Vessels (0–3)", [0, 1, 2, 3])
thal = st.selectbox("Thal (1=Normal, 2=Fixed, 3=Reversible)", [1, 2, 3])

# buat dataframe
data = pd.DataFrame({
    "Age": [age],
    "Sex": [sex],
    "ChestPainType": [cp],
    "RestingBP": [restbp],
    "Cholesterol": [chol],
    "FastingBS": [fbs],
    "RestingECG": [restecg],
    "MaxHR": [maxhr],
    "ExerciseAngina": [exang],
    "Oldpeak": [oldpeak],
    "ST_Slope": [slope],
    "CA": [ca],
    "Thal": [thal]
})

# Scaler
data_scaled = scaler.transform(data)

# Predict
if st.button("Prediksi"):
    pred = model.predict(data_scaled)[0]
    if pred == 1:
        st.error("❌ Terkena Penyakit Jantung")
    else:
        st.success("✅ Tidak Terkena Penyakit Jantung")
