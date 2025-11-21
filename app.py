import streamlit as st
import pickle
import numpy as np
import joblib

model_svm = joblib.load("model_svm.pkl")
model_rf = joblib.load("model_random_forest.pkl")
model_vote = joblib.load("model_voting.pkl")
scaler = joblib.load("scaler.pkl")

# =========================
# JUDUL
# =========================
st.title("❤️ Heart Disease Prediction App")
st.write("Menggunakan SVM, Random Forest, dan Voting Classifier")

# =========================
# PILIH MODEL (DITENGAH)
# =========================
st.subheader("Pilih Model Prediksi")

col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    model_name = st.selectbox(
        "Pilih Model",
        ["SVM", "Random Forest", "Voting Classifier"]
    )

# =========================
# INPUT DATA
# =========================
st.subheader("Masukkan Data Pasien")

age = st.number_input("Age", 20, 100, 52)
sex = st.selectbox("Sex (0=Female, 1=Male)", [0, 1])
cp = st.selectbox("Chest Pain Type (0–3)", [0, 1, 2, 3])
trestbps = st.number_input("Resting BP", 80, 200, 130)
chol = st.number_input("Cholesterol", 100, 400, 220)
fbs = st.selectbox("Fasting Blood Sugar (1=Yes,0=No)", [0, 1])
restecg = st.selectbox("Resting ECG (0–2)", [0, 1, 2])
thalach = st.number_input("Max HR", 60, 220, 165)
exang = st.selectbox("Exercise Angina (1=Yes,0=No)", [0, 1])
oldpeak = st.number_input("Oldpeak", 0.0, 10.0, 0.8)
slope = st.selectbox("ST Slope (0–2)", [0, 1, 2])

# =========================
# PREDIKSI
# =========================
if st.button("Prediksi"):
    # Susun input
    input_data = np.array([[age, sex, cp, trestbps, chol, fbs,
                            restecg, thalach, exang, oldpeak, slope]])

    # Scale data
    input_scaled = scaler.transform(input_data)

    # Pilih model
    if model_name == "SVM":
        prob = model_svm.predict_proba(input_scaled)[0][1]
    elif model_name == "Random Forest":
        prob = model_rf.predict_proba(input_scaled)[0][1]
    else:
        prob = model_vote.predict_proba(input_scaled)[0][1]

    # =========================
    # KATEGORI RISIKO
    # =========================
    if prob <= 0.35:
        kategori = "Rendah"
        warna = "green"
    elif prob <= 0.65:
        kategori = "Sedang"
        warna = "orange"
    else:
        kategori = "Tinggi"
        warna = "red"

    # =========================
    # TAMPILKAN HASIL
    # =========================
    st.subheader("🔎 Hasil Prediksi")

    st.write(f"**Probabilitas Penyakit Jantung: {prob:.2f}**")

    st.markdown(
        f"""
        <div style="padding:15px; border-radius:10px; background-color:{warna}; color:white;">
            <h3 style="margin:0;">Level Risiko: {kategori}</h3>
        </div>
        """,
        unsafe_allow_html=True
    )

    # Penjelasan level risiko
    st.write("### ℹ Informasi Level Risiko")
    st.info("""
**Rendah (0.00 – 0.35)**  
Pasien cenderung tidak memiliki penyakit jantung. Tetap jaga pola hidup sehat.

**Sedang (0.35 – 0.65)**  
Ada indikasi awal penyakit jantung. Disarankan pemeriksaan lanjutan.

**Tinggi (0.65 – 1.00)**  
Pasien sangat berisiko terkena penyakit jantung. Perlu pemeriksaan medis segera.
""")

