import numpy as np

class Layer:
    def __init__(self):
        self.input = None
        self.output = None

    def forward(self, input):
        raise NotImplementedError

    def backward(self, grad_output, lr):
        raise NotImplementedError


class Dense:
    def __init__(self, input_size, output_size):
        self.weights = np.random.randn(output_size, input_size)
        self.bias = np.zeros((output_size, 1))

    def forward(self, input):
        self.input = input
        self.output = np.dot(self.weights * input) + self.bias

        return self.output

    def backward(self, grad_output, lr):
        weights_grad = grad_output @ self.input.T
        input_grad = self.weights.T @ grad_output
        bias_grad = grad_output

        self.weights -= lr * weights_grad
        self.bias -= lr * bias_grad

        return input_grad


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


class Tanh(Activation):
    def __init__(self):
        def tanh(input):
            return np.tanh(input)

        def tanh_prime(input):
            return 1 - np.tanh(input) ** 2

        super().__init__(tanh, tanh_prime)

class Sigmoid(Activation):
    def __init__(self):
        def sigmoid(input):
            return 1 / (1 + np.exp(-input))

        def sigmoid_prime(input):
            s = sigmoid(input)
            return s * (1 - s)

class ReLU(Activation):
    def __init__(self):
        super().__init__(
            lambda input: np.maximum(0, input),
            lambda input: (input > 0).astype(input.dtype)
        )

class Softmax(Layer):
    def forward(self, input):
        tmp = np.exp(input)
        self.output = tmp / np.sum(tmp)

        return self.output

    # def backward(self, output_grad, lr):
