# Implementing Logistic Regression From Scratch In Python

Binary Logistic Regression is often mentioned in connection to classification tasks. The model is simple and one of the easy starters to learn about generating probabilities, classifying samples, and understanding gradient descent. We will walk through some mathematical equations and pair them with practical examples in Python afterward so that you can see exactly how to train your own custom binary logistic regression model.

## Binary Logistic Regression Explained

To understand and implement the algorithm, we need to understand 6 equations which are explained below. We will cautiously walk through them to give you the most intuition possible for how the algorithm works.

To perform a prediction, we use neural-network-like notation; we have weights (w), inputs (x) and bias (b). We can iterate over an error and multiple them together and add the bias at the end like below:

$$
z = \left( \sum_{i=1}^{n} w_i x_i \right) + b
$$

However, it is incredibly common to use vector notation. This just means that w becomes a list of values (in Python terms). Vector notation enables us to leverage faster computation time which can be highly beneficial if you want to do rapid prototyping with a larger dataset. We write a list of values with bolder characters, for example, the vector **w**, and we can rewrite our equation above the following equation:

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
Sigmoid graph, showing how our input (x-axis) turns into an output between 0 and 1 (y-axis). From Wikipedia: [Sigmoid function](https://en.wikipedia.org/wiki/Sigmoid_function).<br>

Once we have the prediction, we can apply the basic gradient descent algorithm to optimize our model parameters, which are the weights and bias in this case. We will not be using stochastic or mini-batch gradient descent in this article, but you can use reference [1] to see how these algorithms are implemented in a slightly different way.

$$
\theta_{t+1} = \theta_{t} - \eta\, \nabla L(f(x;\theta), y)
$$

The scary part for new people is the unknown parameters and greek letters, so here is a list with explanations:

- θ (theta) is the parameter, for example, our weights or bias
- θ<sub>t</sub> just means the current value of the parameter
- θ<sub>t+1</sub> just means the next value of the parameter
- η (eta) is the learning rate, which is usually set to a value between 0.1 and 0.0001.
- ∇L refers to the gradients (∇, nabla) of the (L)oss function, which we will introduce below. It takes in our inputs (x), our parameter values (θ), and the labels (y).

The loss function (also known as a cost function) is a function used to measure how much our prediction differs from the labels. Binary cross entropy is the function used in this article for the binary logistic regression algorithm, which yields the error value:

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

Seeing all these equations can be scary, especially if you do not have formal education. The best way to learn is by applying the equations - which is why we will turn our attention to a practical example in Python with the help of NumPy for numerical computations.

## Python Example

The very first thing in a Python example is to choose our dataset. We need it to be a binary classification dataset, so we choose one from the great library scikit-learn called the "Breast Cancer Wisconsin" dataset. We get several features that can be used to determine if a person has breast cancer, you can read more about it by following reference [3].

To load the dataset, we have created a small piece of code to get the data into training and testing sets in Pandas dataframes.

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

Afterward, we import all the data, libraries, and models that we need for us to do the training part. As is seen below, we will come back to the scikit-learn logistic regression model and compare it to our custom implementation at the end. We start by creating our custom model and using the fit methods with our training data for 150 iterations (epochs).

```python
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from logistic_regression import LogisticRegression as CustomLogisticRegression
from data import x_train, x_test, y_train, y_test

lr = CustomLogisticRegression()
lr.fit(x_train, y_train, epochs=150)
```

Now, we will dive deep into the fit method which handles the whole training cycle. There are many things to unpack, but you should notice that we use the math from the prior explanation on binary logistic regression.

Firstly, we take in the input and do a small transformation which will not be covered in this article. You can see the full details in the GitHub repository for this article in reference [4].

Secondly, we will explain all the functions seen in the fit method below after the walkthrough of the fit method. We start by doing the weight and sigmoid calculation. As we saw in the explanation, we have to multiply the inputs with the weights and add the bias. Then we input these weights into the sigmoid function and get predictions. 

Then comes the part where we have to think about gradient descent. We can compute the loss by the implemented `compute_loss` function and the derivative by `compute_gradients` function. The loss is not used in the model (only the derivative of the loss is used), but we can monitor the loss to determine when our model cannot learn more - which is how the 150 epochs were chosen for the model. At last, we update the parameters of the model, and then we start the next iteration and continue iterating until we reach 150 iterations.

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

To implement the stable version of the sigmoid function, we just had to implement the equation and run through every value. This is not done as a vector calculation like we saw when we multiplied the weights by the inputs - this is due to the if-else statement that makes sure no errors happen when the output of the sigmoid is negative or positive infinity. See more details in reference [2].

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

This is the loss function implemented as a vectorized solution exactly like we saw it in our explanation section. We are essentially finding all the errors by comparing our ground truth `y_true` to our predictions `y_pred` (also known as y hat from our explanation section).

```python
def compute_loss(self, y_true, y_pred):
    # binary cross entropy
    y_zero_loss = y_true * np.log(y_pred + 1e-9)
    y_one_loss = (1-y_true) * np.log(1 - y_pred + 1e-9)
    return -np.mean(y_zero_loss + y_one_loss)
```

Now we get to calculate the gradients, which are what we use to update the model parameters. The equations look complicated if you are new to math, but the implementation below is not too complicated. We start by finding the difference (how much our model predicted wrong) and use it to calculate the gradients for the bias by finding the average error. 

Afterwards, as we saw in the explanation section, we simply have to multiply the difference by the inputs (x). Afterwards, we need to find the average of each gradient, which is quite simple in Python. Now we can return the changes and update our model.

```python
def compute_gradients(self, x, y_true, y_pred):
    # derivative of binary cross entropy
    difference =  y_pred - y_true
    gradient_b = np.mean(difference)
    gradients_w = np.matmul(x.transpose(), difference)
    gradients_w = np.array([np.mean(grad) for grad in gradients_w])

    return gradients_w, gradient_b
```

Perhaps the least complicated part of the gradient descent algorithm is the update. We have already calculated the errors (gradients), now we just have to update the weights for the next iteration, so the model can learn from the next errors.

```python
def update_model_parameters(self, error_w, error_b):
    self.weights = self.weights - 0.1 * error_w
    self.bias = self.bias - 0.1 * error_b
```

After fitting over 150 epochs, we can use the predict function and generate an accuracy score from our custom logistic regression model. 

```python
pred = lr.predict(x_test)
accuracy = accuracy_score(y_test, pred)
print(accuracy)
```

We find that get an accuracy score of 92.98% with our custom model. Below, we see the code for how to do the prediction, which is a repetition of the code in the fit function. The extra addition is that we threshold and classify anything with a probability lower than 0.5 as class 0 (not breast cancer) and anything higher than 0.5 as class 1 (breast cancer).

```python
def predict(self, x):
    x_dot_weights = np.matmul(x, self.weights.transpose()) + self.bias
    probabilities = self._sigmoid(x_dot_weights)
    return [1 if p > 0.5 else 0 for p in probabilities]
```

After getting our accuracy score, we compare with the logistic regression model from the scikit-learn library by using the methods implemented in their library - we named the methods the same in our custom implementation for easy reference.

```python
model = LogisticRegression(solver='newton-cg', max_iter=150)
model.fit(x_train, y_train)
pred2 = model.predict(x_test)
accuracy2 = accuracy_score(y_test, pred2)
print(accuracy2)
```

We find that the accuracy is almost equal with scikit-learn being slightly better at an accuracy of 95.61%, slightly beating our custom logistic regression model by 2.5%.

## Conclusion

In this article, you have learned how to implement your custom binary logistic regression model in Python while understanding the underlying math. We have seen just how similar the logistic regression model can be to a simple neural network.

## Sources

1. Jurafsky, Logistic Regression: https://web.stanford.edu/~jurafsky/slp3/5.pdf

2. Vieira, Exp-normalize trick: http://timvieira.github.io/blog/post/2014/02/11/exp-normalize-trick/

3. UCI, Breast Cancer Wisconsin Dataset https://archive.ics.uci.edu/ml/datasets/breast+cancer+wisconsin+(diagnostic)

4. Hansen, Logistic Regression From Scratch https://github.com/casperbh96/Logistic-Regression-From-Scratch
