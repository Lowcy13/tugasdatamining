import streamlit as st
import numpy as np
import joblib
import os

# ======================================================
# Load Models
# ======================================================

st.title("Heart Disease Prediction App")
st.write("Aplikasi ini menggunakan SVM, Random Forest, dan Voting Classifier")

# Cek file tersedia
files = os.listdir(".")
st.write("📁 Files in directory:")
st.write(files)

# Load models
svm_model = joblib.load("model_svm.pkl")
rf_model = joblib.load("model_random_forest.pkl")
voting_model = joblib.load("model_voting.pkl")
scaler = joblib.load("scaler.pkl")

# ======================================================
# Sidebar - Pilihan Model
# ======================================================

st.sidebar.header("Pilih Model yang Akan Digunakan")
model_choice = st.sidebar.selectbox(
    "Model Prediksi:",
    ("SVM", "Random Forest", "Voting Classifier")
)

# Opsional: tampilkan akurasi
st.sidebar.subheader("Akurasi Model (Contoh)")
st.sidebar.write("🔹 SVM: 91%")
st.sidebar.write("🔹 Random Forest: 94%")
st.sidebar.write("🔹 Voting Classifier: 96%")

# ======================================================
# Input Form
# ======================================================

st.header("Masukkan Data Pasien")

age = st.number_input("Age", 1, 120, 52)
sex = st.selectbox("Sex (0 = Female, 1 = Male)", [0, 1])
cp = st.selectbox("Chest Pain Type (0–3)", [0, 1, 2, 3])
trestbps = st.number_input("Resting BP", 50, 200, 130)
chol = st.number_input("Cholesterol", 50, 600, 220)
fbs = st.selectbox("Fasting Blood Sugar (1=Yes,0=No)", [0, 1])
restecg = st.selectbox("Resting ECG (0–2)", [0, 1, 2])
thalach = st.number_input("Max HR", 50, 250, 165)
exang = st.selectbox("Exercise Angina (1=Yes,0=No)", [0, 1])
oldpeak = st.number_input("Oldpeak", 0.0, 10.0, 0.8)
slope = st.selectbox("ST Slope (0–2)", [0, 1, 2])

# Convert ke array
input_data = np.array([[age, sex, cp, trestbps, chol, fbs,
                        restecg, thalach, exang, oldpeak, slope]])

# Scaling
scaled_data = scaler.transform(input_data)

# ======================================================
# Prediksi
# ======================================================

if st.button("Prediksi"):

    # Pilihan model
    if model_choice == "SVM":
        probability = svm_model.predict_proba(scaled_data)[0][1]
    elif model_choice == "Random Forest":
        probability = rf_model.predict_proba(scaled_data)[0][1]
    else:
        probability = voting_model.predict_proba(scaled_data)[0][1]

    st.subheader("Hasil Prediksi")
    st.write(f"🔎 Probabilitas Penyakit Jantung: **{probability:.2f}**")

    # Threshold 0.5
    if probability >= 0.5:
        st.error("⚠ Risiko Tinggi Penyakit Jantung")
    else:
        st.success("✔ Risiko Rendah Penyakit Jantung")
