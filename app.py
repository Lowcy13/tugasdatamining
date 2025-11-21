import streamlit as st
import numpy as np
import joblib
import os

st.title("Heart Disease Prediction App")
st.write("Aplikasi ini menggunakan Voting Classifier (Random Forest + SVM)")

# Debug – tampilkan file di folder Streamlit
st.write("📁 Files in directory:", os.listdir("."))

# Load model & scaler dengan error handling
try:
    model = joblib.load("model.pkl")
except:
    st.error("❌ model.pkl tidak ditemukan di repo! Upload file tersebut ke GitHub.")
    st.stop()

try:
    scaler = joblib.load("scaler.pkl")
except:
    st.error("❌ scaler.pkl tidak ditemukan di repo! Upload file tersebut ke GitHub.")
    st.stop()

st.subheader("Masukkan Data Pasien")

# Input form
age = st.number_input("Age", 1, 120)
sex = st.selectbox("Sex (0 = Female, 1 = Male)", [0, 1])
chestpain = st.selectbox("Chest Pain Type (0–3)", [0, 1, 2, 3])
bp = st.number_input("Resting BP", 50, 200)
cholesterol = st.number_input("Cholesterol", 50, 500)
fasting_bs = st.selectbox("Fasting Blood Sugar (1=Yes,0=No)", [0, 1])
restecg = st.selectbox("Resting ECG (0–2)", [0, 1, 2])
max_hr = st.number_input("Max HR", 50, 220)
exercise_angina = st.selectbox("Exercise Angina (1=Yes,0=No)", [0, 1])
oldpeak = st.number_input("Oldpeak", 0.0, 10.0)
st_slope = st.selectbox("ST Slope (0–2)", [0, 1, 2])

# Gabungkan input
input_data = np.array([
    age, sex, chestpain, bp, cholesterol,
    fasting_bs, restecg, max_hr, exercise_angina,
    oldpeak, st_slope
]).reshape(1, -1)

# Prediksi
if st.button("Prediksi"):
    # scaling
    try:
        input_scaled = scaler.transform(input_data)
    except:
        st.error("❌ Terjadi error saat scaling. Pastikan scaler.pkl sesuai model.")
        st.stop()

    pred = model.predict(input_scaled)

    if pred == 1:
        st.error("⚠ Risiko Tinggi Penyakit Jantung")
    else:
        st.success("✔ Tidak Berisiko Penyakit Jantung")
