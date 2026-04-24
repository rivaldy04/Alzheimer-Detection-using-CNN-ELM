import cv2
import numpy as np

# =========================
# Preprocessing gambar
# =========================
def preprocess(image, size=(224, 224)):
    # resize
    img = cv2.resize(image, size)

    # normalisasi
    img = img / 255.0

    # ubah ke (C, H, W)
    img = np.expand_dims(img, axis=0)

    return img

# =========================
# Load dari path
# =========================
def load_image(path):
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    return img