import streamlit as st

st.set_page_config(page_title="Deteksi Alzheimer", layout="centered")

import numpy as np
import cv2
import pickle

# =========================
# Load model (cache biar cepat)
# =========================
@st.cache_resource
def load_models():
    with open("models/cnn.pkl", "rb") as f:
        cnn = pickle.load(f)

    with open("models/elm.pkl", "rb") as f:
        elm = pickle.load(f)

    return cnn, elm

cnn_model, elm_model = load_models()

# =========================
# Import fungsi pipeline
# =========================
from src.preprocessing.image import preprocess
from src.cnn.forward import forward_cnn
from src.elm.elm import elm_predict

# =========================
# Label Mapping
# =========================
label_map = {
    0: "Non Demented",
    1: "Very Mild Demented",
    2: "Mild Demented",
    3: "Moderate Demented"
}

# =========================
# UI
# =========================


st.title("🧠 Deteksi Alzheimer (CNN + ELM)")
st.write("Upload citra MRI untuk klasifikasi tingkat Alzheimer")

uploaded_file = st.file_uploader("Upload gambar", type=["jpg", "png", "jpeg"])

# =========================
# Proses
# =========================
if uploaded_file is not None:

    # baca gambar
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, cv2.IMREAD_GRAYSCALE)

    st.image(image, caption="Gambar Input", use_column_width=True)

    with st.spinner("Memproses..."):

        # preprocessing
        img = preprocess(image)

        # CNN
        fitur = forward_cnn(img, cnn_model)

        # ELM
        pred = elm_predict(fitur.reshape(1, -1), elm_model)

        # ambil kelas
        kelas = np.argmax(pred)
        confidence = np.max(pred)

    # =========================
    # Output
    # =========================
    st.subheader("Hasil Prediksi")

    st.success(f"Kelas: {label_map[kelas]}")
    st.info(f"Confidence: {confidence:.4f}")

    # tampilkan semua probabilitas
    st.subheader("Detail Skor")
    for i, score in enumerate(pred[0]):
        st.write(f"{label_map[i]}: {score:.4f}")