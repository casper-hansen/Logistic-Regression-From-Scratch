import copy
import numpy as np
from sklearn.metrics import accuracy_score

# https://joparga3.github.io/standford_logistic_regression/
# https://see.stanford.edu/materials/aimlcs229/cs229-notes1.pdf

class LogisticRegression():
    def __init__(self):
        pass


    def fit(self, x, y, epochs):
        x = self._transform_x(x)
        y = self._transform_y(y)

        self.losses = []
        self.train_accuracies = []
        self.weights = np.zeros(x.shape[1])

        for i in range(epochs):
            x_dot_weights = np.matmul(x, self.weights.transpose())
            pred = self._sigmoid(x_dot_weights)
            error = self.compute_gradients(x, y, pred)
            self.weights = self.weights + 0.1 * error

            pred_class = [1 if p > 0.5 else 0 for p in pred]
            self.train_accuracies.append(accuracy_score(y, pred_class))
            self.losses.append(self.compute_loss(y, pred))


    def compute_gradients(self, x, y_true, y_pred):
        gradients = np.matmul(x.transpose(), y_true - y_pred)
        return np.array([np.mean(grad) for grad in gradients])


    def compute_loss(self, y_true, y_pred):
        y_zero_loss = y_true * np.log(y_pred + 1e-9)
        y_one_loss = (1-y_true) * np.log(1 - y_pred + 1e-9)
        return -np.mean(y_zero_loss + y_one_loss)


    def predict(self, x):
        x_dot_weights = np.matmul(x, self.weights.transpose())
        probabilities = self._sigmoid(x_dot_weights)
        return [1 if p > 0.5 else 0 for p in probabilities]


    def _sigmoid(self, x):
        return np.array([self._sigmoid_function(value) for value in x])


    def _sigmoid_function(self, x):
        # Thanks to http://timvieira.github.io/blog/post/2014/02/11/exp-normalize-trick/
        if x >= 0:
            z = np.exp(-x)
            return 1 / (1 + z)
        else:
            z = np.exp(x)
            return z / (1 + z)


    def _transform_x(self, x):
        x = copy.deepcopy(x)
        return x.values


    def _transform_y(self, y):
        y = copy.deepcopy(y)
        return y.values.reshape(y.shape[0], 1)
