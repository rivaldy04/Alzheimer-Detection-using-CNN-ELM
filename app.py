import streamlit as st
import pickle
import numpy as np
import cv2
from tensorflow.keras.models import load_model

from src.elm.elm import ELM, elm_predict
from src.preprocessing.image import preprocess
from src.cnn.forward import extract_features

st.set_page_config(
    page_title="Deteksi Alzheimer",
    layout="centered"
)

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
# Load Models
# =========================
@st.cache_resource
def load_models():
    cnn = load_model("Models/cnn_feature_extractor.keras")

    with open("Models/elm_model.pkl", "rb") as f:
        elm = pickle.load(f)

    return cnn, elm


cnn_model, elm_model = load_models()

# =========================
# UI
# =========================
st.title("🧠 Deteksi Alzheimer (CNN + ELM)")
st.write("Upload citra MRI untuk klasifikasi tingkat Alzheimer.")

uploaded_file = st.file_uploader(
    "Upload gambar MRI",
    type=["jpg", "jpeg", "png"]
)

# =========================
# Prediction
# =========================
if uploaded_file is not None:

    file_bytes = np.asarray(
        bytearray(uploaded_file.read()),
        dtype=np.uint8
    )

    image = cv2.imdecode(
        file_bytes,
        cv2.IMREAD_GRAYSCALE
    )

    st.image(
        image,
        caption="Gambar Input",
        channels="GRAY",
        use_container_width=True
    )

    with st.spinner("Memproses gambar MRI..."):

        # Preprocessing
        img = preprocess(image)

        # Feature Extraction
        features = extract_features(
            cnn_model,
            img
        )

        # ELM Prediction
        pred_class, pred_prob = elm_predict(
            features,
            elm_model
        )

        # Ambil probabilitas tertinggi
        confidence = np.max(pred_prob)

    # =========================
    # Hasil
    # =========================
    st.subheader("Hasil Prediksi")

    predicted_class = int(np.array(pred_class).flatten()[0])

    st.success(
        f"Diagnosis: {label_map[predicted_class]}"
    )

    st.info(
        f"Tingkat Keyakinan: {confidence:.2%}"
    )

    # =========================
    # Probabilitas
    # =========================
    st.subheader("Probabilitas Tiap Kelas")

    for i, prob in enumerate(pred_prob.flatten()):
        st.write(f"{label_map[i]}: {prob:.2%}")
        st.progress(float(prob))
