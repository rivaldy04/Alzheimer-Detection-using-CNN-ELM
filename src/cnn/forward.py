import numpy as np

# =========================
# Convolution
# =========================
def conv2d(input, kernel, bias, stride=1, padding=0):
    C_out, C_in, kH, kW = kernel.shape
    C_in, H, W = input.shape

    # padding
    if padding > 0:
        input = np.pad(input, ((0,0),(padding,padding),(padding,padding)), mode='constant')

    H, W = input.shape[1], input.shape[2]

    out_H = (H - kH)//stride + 1
    out_W = (W - kW)//stride + 1

    output = np.zeros((C_out, out_H, out_W))

    for f in range(C_out):
        for i in range(out_H):
            for j in range(out_W):
                region = input[:, i*stride:i*stride+kH, j*stride:j*stride+kW]
                output[f, i, j] = np.sum(region * kernel[f]) + bias[f]

    return output

# =========================
# ReLU
# =========================
def relu(x):
    return np.maximum(0, x)

# =========================
# Max Pooling
# =========================
def max_pooling(input, size=2, stride=2):
    C, H, W = input.shape

    out_H = (H - size)//stride + 1
    out_W = (W - size)//stride + 1

    output = np.zeros((C, out_H, out_W))

    for c in range(C):
        for i in range(out_H):
            for j in range(out_W):
                region = input[c, i*stride:i*stride+size, j*stride:j*stride+size]
                output[c, i, j] = np.max(region)

    return output

# =========================
# Global Average Pooling
# =========================
def global_avg_pooling(input):
    return np.mean(input, axis=(1, 2))  # (C,)

# =========================
# Forward CNN
# =========================
def forward_cnn(x, model):
    kernel = model["conv"]["kernel"]
    bias = model["conv"]["bias"]
    config = model["conv"]["config"]

    # Conv
    x = conv2d(
        x,
        kernel,
        bias,
        stride=config["stride"],
        padding=config["padding"]
    )

    # ReLU
    x = relu(x)

    # Max Pooling
    pool = model["pooling"]
    x = max_pooling(
        x,
        size=pool["kernel_size"],
        stride=pool["stride"]
    )

    # GAP
    x = global_avg_pooling(x)

    return x