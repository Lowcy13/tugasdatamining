import streamlit as st
import numpy as np
import joblib
import os

# ===============================
# LOAD MODEL & SCALER
# ===============================
scaler = joblib.load("scaler.pkl")
model_voting = joblib.load("model_voting.pkl")

st.title("Heart Disease Prediction App")
st.write("Aplikasi ini menggunakan Voting Classifier (Random Forest + SVM)")

# Debug: list files
st.write("📁 Files in directory:")
st.write(os.listdir("."))

# ===============================
# INPUT FORM
# ===============================
age = st.number_input("Age", min_value=1, max_value=120, value=40)
sex = st.selectbox("Sex (0 = Female, 1 = Male)", [0, 1])
cp = st.selectbox("Chest Pain Type (0–3)", [0, 1, 2, 3])
trestbps = st.number_input("Resting BP", min_value=50, max_value=200, value=120)
chol = st.number_input("Cholesterol", min_value=50, max_value=600, value=200)
fbs = st.selectbox("Fasting Blood Sugar (1=Yes,0=No)", [0, 1])
restecg = st.selectbox("Resting ECG (0–2)", [0, 1, 2])
thalach = st.number_input("Max HR", min_value=60, max_value=250, value=150)
exang = st.selectbox("Exercise Angina (1=Yes,0=No)", [0, 1])
oldpeak = st.number_input("Oldpeak", min_value=0.0, max_value=10.0, value=1.0)
slope = st.selectbox("ST Slope (0–2)", [0, 1, 2])

# ===============================
# PREDICT
# ===============================
if st.button("Prediksi"):
    input_data = np.array([[age, sex, cp, trestbps, chol, fbs,
                            restecg, thalach, exang, oldpeak, slope]])

    scaled = scaler.transform(input_data)

    proba = model_voting.predict_proba(scaled)[0][1]

    prediction = 1 if proba >= 0.55 else 0

    st.write(f"🔎 Probabilitas Penyakit Jantung: **{proba:.2f}**")

    if prediction == 1:
        st.error("⚠ Risiko Tinggi Penyakit Jantung")
    else:
        st.success("✅ Risiko Rendah Penyakit Jantung")
