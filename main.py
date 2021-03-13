from logistic_regression import LogisticRegression
from data import x_train, x_test, y_train, y_test

lr = LogisticRegression()
lr.fit(x_train, y_train, n_iterations=100)