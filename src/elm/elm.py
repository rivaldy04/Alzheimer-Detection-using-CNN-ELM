import numpy as np

def softmax(x):
    """
    Softmax untuk satu sampel atau batch.
    """
    x = np.atleast_2d(x)
    exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
    return exp_x / np.sum(exp_x, axis=1, keepdims=True)


def elm_predict(x, model):
    y_pred, y_prob = model.predict(x)
    return y_pred, y_prob

class ELM:
    def __init__(self, input_dim, hidden_dim, output_dim,
                 activation="tanh", reg=1e-3):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.reg = reg
        self.activation_name = activation

    def _activate(self, x):
        if self.activation_name == "sigmoid":
            return 1.0 / (1.0 + np.exp(-x))
        elif self.activation_name == "relu":
            return np.maximum(0, x)
        elif self.activation_name == "tanh":
            return np.tanh(x)
        else:
            raise ValueError("Unsupported activation function")

    def _softmax(self, x):
        exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=1, keepdims=True)

    def predict(self, X):
        H = self._activate(np.dot(X, self.W) + self.b)
        y_raw = np.dot(H, self.beta)
        y_prob = self._softmax(y_raw)
        return np.argmax(y_prob, axis=1), y_prob