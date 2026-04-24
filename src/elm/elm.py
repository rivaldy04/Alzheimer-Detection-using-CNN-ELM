import numpy as np

# =========================
# Aktivasi (default sigmoid)
# =========================
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# =========================
# Forward ELM
# =========================
def elm_predict(x, model):
    W = model["W"]
    b = model["b"]
    beta = model["beta"]
    activation = model["config"]["activation"]

    # Hidden layer
    H = np.dot(x, W) + b

    # Aktivasi
    if activation == "sigmoid":
        H = sigmoid(H)
    elif activation == "relu":
        H = np.maximum(0, H)
    elif activation == "tanh":
        H = np.tanh(H)

    # Output
    y = np.dot(H, beta)

    return y