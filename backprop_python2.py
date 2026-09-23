import numpy as np


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))  # Pushed to the right!


def der(x):
    return x * (1.0 - x)  # Pushed to the right!


class MLP:
    def __init__(self, inputs):
        self.inputs = inputs
        self.l = len(self.inputs)
        self.li = len(self.inputs[0])
        self.wi = np.random.random((self.li, self.l))
        self.wh = np.random.random((self.l, 1))

    def pred(self, inp):
        s1 = sigmoid(np.dot(inp, self.wi))
        s2 = sigmoid(np.dot(s1, self.wh))
        return s2

    def train(self, inputs, outputs, iterations, learning_rate):
        for i in range(iterations):
            l0 = inputs
            l1 = sigmoid(np.dot(l0, self.wi))
            l2 = sigmoid(np.dot(l1, self.wh))

            l2_err = outputs - l2
            l2_delta = np.multiply(l2_err, der(l2))

            l1_err = np.dot(l2_delta, self.wh.T)
            l1_delta = np.multiply(l1_err, der(l1))

            updating_wh = learning_rate * np.dot(l1.T, l2_delta)
            updating_wi = learning_rate * np.dot(l0.T, l1_delta)

            self.wh += updating_wh
            self.wi += updating_wi


# The stuff below is NOT pushed to the right because it is the "Start" of the program
inputs = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
outputs = np.array([[0], [1], [1], [0]])

n = MLP(inputs)
print("before training\n\n", n.pred(inputs))
n.train(inputs, outputs, 10000, 1)
print("\nafter training\n\n", n.pred(inputs))
# Look at the weights (the 'trust knobs')
print("\nWeights for the first layer connections:")
print(n.wi[0])
# Ask the brain: "What is the answer for [1, 0]?"
new_puzzle = np.array([[1, 0]])
answer = n.pred(new_puzzle)

print(f"\nMy Prediction for [1, 0] is: {answer}")
