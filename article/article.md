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
Sigmoid graph, showing how our input (x-axis) turns into an output between 0 and 1 (y-axis).

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



## Conclusion

## Sources

1. https://web.stanford.edu/~jurafsky/slp3/5.pdf

2. http://timvieira.github.io/blog/post/2014/02/11/exp-normalize-trick/