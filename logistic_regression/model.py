# https://web.stanford.edu/~hastie/ElemStatLearn//printings/ESLII_print12_toc.pdf
# The Elements of Statistical Learning
# Section 4.4.1: Fitting Logistic Regression Models

import numpy as np

class LogisticRegression():
    def __init__(self):
        pass

    def fit(self, x, y, n_iterations):
        '''
            Iteratively reweighted least squares (IRLS).
            Implemented from the equations shown in ESLII.

            x: input data
            y: target data
            z: adjusted response
            b: weights
            w: diagonal matrix of weights
        '''
        self.b = np.zeros(x.shape[1])

        for i in range(n_iterations):
            z, w = self.compute_adjusted_response(x, y, b_old=self.b)
            self.b = self.b_new(x, y, w, z)

    def compute_adjusted_response(self, x, y, b_old):
        # Calculate left side of equation 4.27: X*β_old
        x_dot_weights = np.dot(x, b_old)
        
        # Calculate right side of equation 4.27:  W^(-1)*(y − p)
        # Where W is (N, N) diagonal matrix of p(x|β_old)*(1-p(x|β_old))
        preds = self._sigmoid(x_dot_weights)
        diff = y - preds
        w = preds*(1-preds)
        w = np.diag(w)
        w_inverse = np.linalg.inv(w)

        # Full equation
        z = x_dot_weights + np.dot(w_inverse, diff)

        return z, w

    def b_new(self, x, y, w, z):
        # reused variables
        xT = x.transpose()
        x_dot_w = np.dot(xT, w)

        # Left side of equation 4.26: (X.T*W*X)**-1
        parenthesis = np.dot(x_dot_w, x)
        left_side = np.linalg.inv(parenthesis)
        
        # Right side of equation 4.26: X.T*W*z
        right_side = np.dot(x_dot_w, z)

        # Full equation
        return np.dot(left_side, right_side)

    def predict(self, x):
        pred = np.dot(x, self.b)
        return self._sigmoid(pred)

    def accuracy_score(self, y_true, y_pred):
        pass

    # link function
    def _sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def _create_diagonal(self, w):
        diagonal_size = (w.shape[1], w.shape[1])
        w_diag = np.zeros(diagonal_size)
        np.fill_diagonal(w_diag, np.diag(w))

        return w_diag