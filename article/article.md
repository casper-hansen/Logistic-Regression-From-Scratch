# Implementing Logistic Regression From Scratch In Python

Binary Logistic Regression is often mentioned in connection to classification tasks. The model is simple and one of the easy starters to learn about generating probablities, classifying samples, and understanding gradient descent. We will walk through some mathematical equations and pair them practical examples in Python afterwards, so that you can see exactly how to train your own custom binary logistic regression model.

## Binary Logistic Regression Explained

To understand and implement the algorithm, we need to understand 6 equations which are explained below. We will cautiously walk through them to give you the most intuition possible for how the algorithm works.

To perform a prediction, we use neural-network-like notation; we have weights (w), inputs (x) and bias (b). We can iterate over an error and multiple them together and add the bias at the end like below:

$$
z = \left( \sum_{i=1}^{n} w_i x_i \right) + b
$$

However, it is incredibly common to use vector notation. This just means that w becomes a list of values (in Python terms). Vector notation enables us to leverage faster computation time which can be highly beneficial if you want to do rapid prototyping with a larger dataset. We write a list of values with bolder characters, for example the vector **w**, and we can rewrite our equation above the following equation:

$$
z = \mathbf{w} \cdot \mathbf{x} + b
$$

Furthermore, to get our prediction, we need to use an activation function - in the case of binary logistic regression, it is called the sigmoid and is usually denoted by the Greek letter sigma. Another common notation is ŷ (y hat). Below, we see the stable version of the sigmoid equation [2]:

$$
\hat{y} = \sigma(z) = 
\begin{dcases}
    \frac{1}{(1 + exp(z))}, & \text{if } z\geq 0\\
    \frac{exp(z)}{(1 + exp(z))}, & \text{otherwise}
\end{dcases}
$$

We can visualize the sigmoid function by the following graph:

![](images/sigmoid.png)
<center>Sigmoid graph, showing how our input (x-axis) turns into an output between 0 and 1 (y-axis).</center><br>

Once we have the prediction, we can apply the basic gradient descent algorithm to optimize our model parameters, which are the weights and bias in this case. We will not be using stochastic or mini-batch gradient descent in this article, but you can use reference [1] to see how these algorithms are implemented in a slightly different way.

$$
\theta_{t+1} = \theta_{t} - \eta\, \nabla L(f(x;\theta), y)
$$

The scary part for new people is the unknown parameters and greek letters, so here is a list with explanations:

- θ (theta) is the parameter, for example our weights or bias
- θ<sub>t</sub> just means the current value of the parameter
- θ<sub>t+1</sub> just means the next value of the parameter
- η (eta) is the learning rate, which is usually set to a value between 0.1 and 0.0001.
- ∇L refers to the gradients (∇, nabla) of the (L)oss function, which we will introduce below. It takes in our inputs (x), our parameter values (θ), and the labels (y).

The loss function (also known as cost function) is a function used to measure how much our prediction differs from the labels. Binary cross entropy is the function used in this article for the binary logistic regression algorithm, which yields the error value:

$$
L_{\text{CE}}(\hat{y}, y) = -\frac{1}{m} \sum_{i=1}^{m} \mathbf{y} \, log(\mathbf{\hat{y}}) + (1-\mathbf{y}) \, log(1-\mathbf{\hat{y}})
$$

Looking at the plus sign in the equation; if y = 0, then the left side is equal to 0, and if y = 1, then the right side will be equal to 0. Effectively, this is how we measure how much our prediction ŷ differs from our label y, which can only be 0 or 1 in a binary classification algorithm.

Now, in order to calculate the gradients to optimize our weights using gradient descent, we need to calculate the derivative of our loss function - in other words, we need to calculate the partial derivative of binary cross entropy. We can quite compactly describe the derivative of the loss function as seen below; for a derivation, see Section 5.10 in [1].

$$
\frac{\partial L_{\text{CE}}(\hat{y}, y)}{\partial \mathbf{w}} = \frac{1}{m} (\mathbf{\hat{y}} - \mathbf{y}) \mathbf{x}_i^{T}
$$

$$
\frac{\partial L_{\text{CE}}(\hat{y}, y)}{\partial b} = \frac{1}{m} (\mathbf{\hat{y}} - \mathbf{y})
$$

When we have the final values from our derivative calculation, we can use it in the gradient descent equation and update the weights and bias.

Now we will see all of this in action in Python code with the help of NumPy for numerical computations.

## Python Example

Load dataset

```python
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_breast_cancer

def sklearn_to_df(data_loader):
    X_data = data_loader.data
    X_columns = data_loader.feature_names
    x = pd.DataFrame(X_data, columns=X_columns)

    y_data = data_loader.target
    y = pd.Series(y_data, name='target')

    return x, y

x, y = sklearn_to_df(load_breast_cancer())

x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=42)
```

Create model and fit

```python
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from logistic_regression import LogisticRegression as CustomLogisticRegression
from data import x_train, x_test, y_train, y_test

lr = CustomLogisticRegression()
lr.fit(x_train, y_train, epochs=150)
```

The fit method

```python
def fit(self, x, y, epochs):
    x = self._transform_x(x)
    y = self._transform_y(y)

    self.weights = np.zeros(x.shape[1])
    self.bias = 0

    for i in range(epochs):
        x_dot_weights = np.matmul(self.weights, x.transpose()) + self.bias
        pred = self._sigmoid(x_dot_weights)
        loss = self.compute_loss(y, pred)
        error_w, error_b = self.compute_gradients(x, y, pred)
        self.update_model_parameters(error_w, error_b)

        pred_to_class = [1 if p > 0.5 else 0 for p in pred]
        self.train_accuracies.append(accuracy_score(y, pred_to_class))
        self.losses.append(loss)
```

Sigmoid function

```python
def _sigmoid(self, x):
    return np.array([self._sigmoid_function(value) for value in x])

def _sigmoid_function(self, x):
    if x >= 0:
        z = np.exp(-x)
        return 1 / (1 + z)
    else:
        z = np.exp(x)
        return z / (1 + z)
```

Loss function

```python
def compute_loss(self, y_true, y_pred):
    # binary cross entropy
    y_zero_loss = y_true * np.log(y_pred + 1e-9)
    y_one_loss = (1-y_true) * np.log(1 - y_pred + 1e-9)
    return -np.mean(y_zero_loss + y_one_loss)
```

Loss function derivative to get gradients

```python
def compute_gradients(self, x, y_true, y_pred):
    # derivative of binary cross entropy
    difference =  y_pred - y_true
    gradient_b = np.mean(difference)
    gradients_w = np.matmul(x.transpose(), difference)
    gradients_w = np.array([np.mean(grad) for grad in gradients_w])

    return gradients_w, gradient_b
```

Updating model parameters

```python
def update_model_parameters(self, error_w, error_b):
    self.weights = self.weights - 0.1 * error_w
    self.bias = self.bias - 0.1 * error_b
```

After fitting is done, we can use the predict function and generate an accuracy score

```python
pred = lr.predict(x_test)
accuracy = accuracy_score(y_test, pred)
print(accuracy)
```

After getting our accuracy score, we compare with the logistic regression model from the scikit-learn library.

```python
model = LogisticRegression(solver='newton-cg', max_iter=150)
model.fit(x_train, y_train)
pred2 = model.predict(x_test)
accuracy2 = accuracy_score(y_test, pred2)
print(accuracy2)
```

We find that the accuracy is almost equal.

## Conclusion

## Sources

1. https://web.stanford.edu/~jurafsky/slp3/5.pdf

2. http://timvieira.github.io/blog/post/2014/02/11/exp-normalize-trick/