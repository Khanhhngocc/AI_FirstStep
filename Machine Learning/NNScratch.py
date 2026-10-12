import numpy as np


# =============================
# Base Layer interface
# =============================
class Layer:
    def __init__(self):
        self.output = None
        self.input = None

    def forward(self, input):
        raise NotImplementedError

    def backward(self, grad_output, lr):
        raise NotImplementedError


# =============================
# Dense (Fully Connected) Layer
# =============================
class Dense:
    def __init__(self, input_size, output_size):
        # He/Xavier-like scaling for stability (simple version)
        std = np.sqrt(2.0 / (input_size + output_size))

        self.weights = np.random.randn(output_size, input_size) * std
        self.bias = np.zeros((output_size, 1))

    def forward(self, input):
        self.input = input
        y = np.dot(self.weights, input) + self.bias # (out, 1)
        self.output = y
        return y

    def backward(self, grad_output, lr):
        weights_gradient = (grad_output @ self.input.T)
        input_gradient = self.weights.T @ grad_output
        bias_gradient = grad_output

        self.weights -= lr * weights_gradient
        self.bias -= lr * bias_gradient

        return input_gradient


# =============================
# Generic Activation wrapper
# =============================
class Activation(Layer):
    def __init__(self, activation, activation_prime):
        self.activation = activation
        self.activation_prime = activation_prime
        self.input = None

    def forward(self, input):
        self.input = input
        self.output = self.activation(input)
        return self.output

    def backward(self, grad_output, lr):
        return grad_output * self.activation_prime(self.input)

# ----- Concrete activations
class Tanh(Activation):
    def __init__(self):
        def tanh(x):
            return np.tanh(x)

        def tanh_prime(x):
            return 1 - np.tanh(x) ** 2

        super().__init__(tanh, tanh_prime)

class Sigmoid(Activation):
    def __init__(self):
        def sigmoid(x):
            return 1 / (1 + np.exp(-x))

        def sigmoid_prime(x):
            s = sigmoid(x)
            return s * (1 - s)

        super().__init__(sigmoid, sigmoid_prime)

class ReLU(Activation):
    def __init__(self):
        super().__init__(
            lambda x: np.maximum(0.0, x),
            lambda x: (x > 0.0).astype(x.dtype)
        )


# =============================
# Softmax (vector) layer
# =============================
class Softmax(Layer):
    def forward(self, input):
        tmp = np.exp(input)
        self.output = tmp / np.sum(tmp)
        return self.output

    def backward(self, output_grad, lr):
        n = np.size(self.output)
        return np.dot((np.identity(n) - self.output.T) * self.output, output_grad)

# =============================
# Loss functions
# =============================
def mse(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)

def mse_prime(y_true, y_pred):
    return 2 * (y_true - y_pred) / np.size(y_true)

def binary_cross_entropy(y_true, y_pred):
    return np.mean(-y_true * np.log(y_pred) - (1 - y_true) * np.log(1 - y_pred))

def binary_cross_entropy_prime(y_true, y_pred):
    return ((1 - y_true) / (1 - y_pred) - (y_true / y_pred)) / np.size(y_true)



# =============================
# Inference & Training loops
# =============================
def predict(network, input):
    output = input
    for layer in network:
        output = layer.forward(output)

    return output

def train(network, loss, loss_prime, x_train, y_train, epochs=1000, lr=0.01, verbose=True):
    m = len(x_train)

    for e in range(epochs):
        error = 0

        for x, y in zip(x_train, y_train):
            #forward
            output = predict(network, x)

            #error
            error += loss(y, output)

            #backward
            grad = loss_prime(y, output)
            for layer in reversed(network):
                grad = layer.backward(grad, lr)

        error /= m
        if verbose:
            print(f"{e+1}/{epochs}, loss={error:.6f}")


# =============================
# Demo tasks (run directly)
# =============================
if __name__ == "__main__":
    demo = 'sine'

    if demo == 'xor':
        X = [
            np.array([[0.0], [0.0]]),
            np.array([[0.0], [1.0]]),
            np.array([[1.0], [0.0]]),
            np.array([[1.0], [1.0]])
        ]
        Y = [
            np.array([[0.0]]),
            np.array([[1.0]]),
            np.array([[1.0]]),
            np.array([[0.0]])
        ]

        net = [
            Dense(2, 4),
            Tanh(),
            Dense(4, 1),
            Sigmoid(),
        ]

        train(net, binary_cross_entropy, binary_cross_entropy_prime, X, Y, epochs=4000, lr=0.1, verbose=True)
        for x in X:
            p = predict(net, x)
            print(f"x={x.ravel()} pred={p.item():.6f}")


    elif demo == 'sine':
        xs = np.linspace(-np.pi, np.pi, 200)
        X = [np.array([[x]]) for x in xs]
        Y = [np.array([[np.sin(x)]]) for x in xs]

        net = [
            Dense(1, 32), ReLU(),
            Dense(32, 32), ReLU(),
            Dense(32, 1),  # linear output for regression
        ]
        train(net, mse, mse_prime, X, Y, epochs=2000, lr=0.01, verbose=True)

        # Print a few predictions
        for x in [-3.14, -1.57, 0.0, 1.57, 3.14]:
            p = predict(net, np.array([[x]]))
        print(f"x={x:.2f}, sin(x)={np.sin(x):.3f}, pred={float(p):.3f}")
