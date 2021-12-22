import copy
import numpy as np

# https://joparga3.github.io/standford_logistic_regression/
# https://see.stanford.edu/materials/aimlcs229/cs229-notes1.pdf

class LogisticRegression():
    def __init__(self):
        pass


    def fit(self, x, y, epochs):
        x = self._transform_x(x)
        y = self._transform_y(y)

        losses = []
        weights = np.zeros(x.shape[1])

        for i in range(epochs):
            x_dot_weights = np.matmul(x, weights.transpose())
            pred = self._sigmoid(x_dot_weights)
            error = self.compute_gradients(x, y, pred)
            weights = weights + 0.1 * error

            loss = self.compute_loss(y, pred)
            losses.append(loss)

        self.weights = weights


    def compute_gradients(self, x, y_true, y_pred):
        gradients = np.matmul(x.transpose(), y_true - y_pred)
        return np.array([np.mean(grad) for grad in gradients])


    def compute_loss(self, y_true, y_pred):
        y_zero_loss = y_true * np.log(y_pred)
        y_one_loss = (1-y_true) * np.log(1-y_pred)
        return -np.mean(y_zero_loss + y_one_loss)


    def predict(self, x):
        x_dot_weights = np.matmul(x, self.weights.transpose())
        probabilities = self._sigmoid(x_dot_weights)
        return [1 if p > 0.5 else 0 for p in probabilities]


    def _sigmoid(self, x):
        return 1 / (1 + np.exp(-x))


    def _transform_x(self, x):
        x = copy.deepcopy(x)
        return x.values


    def _transform_y(self, y):
        y = copy.deepcopy(y)
        return y.values.reshape(y.shape[0], 1)
