import cv2
import numpy as np

# =========================
# Preprocessing gambar
# =========================
def preprocess(image):
    image = cv2.resize(image, (224, 224))
    image = image.astype("float32") / 255.0
    image = np.expand_dims(image, axis=-1)   # channel
    image = np.expand_dims(image, axis=0)    # batch
    return image