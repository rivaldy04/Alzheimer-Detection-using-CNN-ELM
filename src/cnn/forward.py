import numpy as np

# =========================
# Convolution
# =========================
def relu(x):
    return np.maximum(0, x)

def conv2d_single(input_img, kernels, bias, stride=1, padding=0, activation="relu"):
    """
    input_img : (H, W, C)
    kernels   : (F, kH, kW, C)
    bias      : (F,)
    """
    H, W, C = input_img.shape
    F, kH, kW, _ = kernels.shape

    # Padding
    if padding > 0:
        input_img = np.pad(
            input_img,
            ((padding, padding),
             (padding, padding),
             (0, 0)),
            mode='constant'
        )

    H_pad, W_pad, _ = input_img.shape

    # Ukuran output
    out_H = (H_pad - kH) // stride + 1
    out_W = (W_pad - kW) // stride + 1

    # Output: (H_out, W_out, F)
    output = np.zeros((out_H, out_W, F))

    for f in range(F):
        for i in range(out_H):
            for j in range(out_W):
                h_start = i * stride
                w_start = j * stride

                region = input_img[
                    h_start:h_start + kH,
                    w_start:w_start + kW,
                    :
                ]

                output[i, j, f] = (
                    np.sum(region * kernels[f]) + bias[f]
                )

    if activation == "relu":
        output = relu(output)

    return output


def conv2d_batch(X, kernels, bias,
                 stride=1,
                 padding=0,
                 activation="relu"):
    """
    X        : (N, H, W, C)
    kernels  : (F, kH, kW, C)
    bias     : (F,)
    """
    outputs = []

    for n in range(X.shape[0]):
        out = conv2d_single(
            X[n],
            kernels,
            bias,
            stride=stride,
            padding=padding,
            activation=activation
        )
        outputs.append(out)

    return np.array(outputs)

# =========================
# Max Pooling
# =========================
def maxpool2d_single(input_img, pool_size=(2, 2), stride=2):
    """
    Max Pooling untuk satu gambar.

    Parameters
    ----------
    input_img : ndarray
        Shape: (H, W, C)
    pool_size : tuple
        Ukuran pooling window.
    stride : int
        Langkah pergeseran window.

    Returns
    -------
    output : ndarray
        Shape: (out_H, out_W, C)
    """
    H, W, C = input_img.shape
    pH, pW = pool_size

    out_H = (H - pH) // stride + 1
    out_W = (W - pW) // stride + 1

    output = np.zeros((out_H, out_W, C))

    for c in range(C):
        for i in range(out_H):
            for j in range(out_W):
                h_start = i * stride
                w_start = j * stride

                region = input_img[
                    h_start:h_start + pH,
                    w_start:w_start + pW,
                    c
                ]

                output[i, j, c] = np.max(region)

    return output


def maxpool2d_batch(X, pool_size=(2, 2), stride=2):
    """
    Max Pooling untuk batch gambar.

    Parameters
    ----------
    X : ndarray
        Shape: (N, H, W, C)

    Returns
    -------
    output : ndarray
        Shape: (N, out_H, out_W, C)
    """
    outputs = []

    for n in range(X.shape[0]):
        pooled = maxpool2d_single(
            X[n],
            pool_size=pool_size,
            stride=stride
        )
        outputs.append(pooled)

    return np.array(outputs)

# =========================
# Global Average Pooling
# =========================
def global_avg_pool_single(input_img):
    """
    Global Average Pooling untuk satu gambar.

    Parameters
    ----------
    input_img : ndarray
        Shape: (H, W, C)

    Returns
    -------
    output : ndarray
        Shape: (C,)
    """
    return np.mean(input_img, axis=(0, 1))


def global_avg_pool_batch(X):
    """
    Global Average Pooling untuk batch gambar.

    Parameters
    ----------
    X : ndarray
        Shape: (N, H, W, C)

    Returns
    -------
    output : ndarray
        Shape: (N, C)
    """
    outputs = []

    for n in range(X.shape[0]):
        pooled = global_avg_pool_single(X[n])
        outputs.append(pooled)

    return np.array(outputs)

# =========================
# Forward CNN
# =========================
def forward_cnn(x, model):
    kernels = model["conv"]["kernel"]
    bias = model["conv"]["bias"]
    config = model["conv"]["config"]

    # Convolution + ReLU
    x = conv2d_single(
        x,
        kernels,
        bias,
        stride=config["stride"],
        padding=config["padding"],
        activation="relu"
    )

    # Max Pooling
    pool = model["pooling"]

    pool_size = pool["kernel_size"]
    if isinstance(pool_size, int):
        pool_size = (pool_size, pool_size)

    x = maxpool2d_single(
        x,
        pool_size=pool_size,
        stride=pool["stride"]
    )

    # Global Average Pooling
    x = global_avg_pool_single(x)

    return x

def extract_features(model, image):
    features = model.predict(image, verbose=0)
    return features