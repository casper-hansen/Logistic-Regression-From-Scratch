# Implementing Logistic Regression From Scratch In Python

Binary Logistic Regression is often mentioned in connection to classification tasks. The model is simple and one of the easy starters to learn about generating probablities, classifying samples, and understanding gradient descent. We will walk through some mathematical equations and pair them practical examples in Python afterwards, so that you can see exactly how to train your own custom binary logistic regression model.

## Binary Logistic Regression Explained

Sigmoid prediction

$$
z = \left( \sum_{i=1}^{n} w_i x_i \right) + b
$$

$$
z = \mathbf{w} \cdot \mathbf{x} + b
$$

$$
\sigma(z) = 
\begin{dcases}
    \frac{1}{(1 + exp(z))}, & \text{if } z\geq 0\\
    \frac{exp(z)}{(1 + exp(z))}, & \text{otherwise}
\end{dcases}
$$

Gradient descent

$$
\theta_{t+1} = \theta_{t} - \eta\, \nabla L(f(x;\theta), y)
$$

Binary cross entropy

z is also known as our prediction ŷ.

$$
L_{\text{CE}}(\hat{y}, y) = -[y \, log(\hat{y}) + (1-y) \, log(1-\hat{y})]
$$

Binary cross entropy derivative (gradients)

The compute the gradients for the weights, we are just taking the derivative of our loss function. We can quite compactly describe the loss function as below; for a derivation, see Section 5.10 in [1].

$$
loss' = \frac{1}{m} (\mathbf{\hat{y}} - \mathbf{y}) \mathbf{x}_i^{T}
$$

## Python Example

## Conclusion

## Sources

1. https://web.stanford.edu/~jurafsky/slp3/5.pdf

2. http://timvieira.github.io/blog/post/2014/02/11/exp-normalize-trick/