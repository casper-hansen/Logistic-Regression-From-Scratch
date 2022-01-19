# Implementing Logistic Regression From Scratch In Python

Binary Logistic Regression is often mentioned in connection to classification tasks. The model is simple and one of the easy starters to learn about generating probablities, classifying samples, and understanding gradient descent. We will walk through some mathematical equations and pair them practical examples in Python afterwards, so that you can see exactly how to train your own custom binary logistic regression model.

## Binary Logistic Regression Explained

To understand and implement the algorithm, we need to understand 6 equations which are explained below. We will cautiously walk through them to give you the most intuition possible for how the algorithm works.

Sigmoid prediction

$$
z = \left( \sum_{i=1}^{n} w_i x_i \right) + b
$$

$$
z = \mathbf{w} \cdot \mathbf{x} + b
$$

$$
\hat{y} = \sigma(z) = 
\begin{dcases}
    \frac{1}{(1 + exp(z))}, & \text{if } z\geq 0\\
    \frac{exp(z)}{(1 + exp(z))}, & \text{otherwise}
\end{dcases}
$$

Once we have the prediction, we can apply the basic gradient descent algorithm. We will not be using stochastic or mini-batch gradient descent in this article, but you can use reference [1] to see how these algorithms are implemented in a slightly different way.

$$
\theta_{t+1} = \theta_{t} - \eta\, \nabla L(f(x;\theta), y)
$$

Binary cross entropy

The following is the way to measure how much the logistic regression prediction is different from the our label (the ground truth).

$$
L_{\text{CE}}(\hat{y}, y) = -\frac{1}{m} \sum_{i=1}^{m} [y \, log(\hat{y}) + (1-y) \, log(1-\hat{y})]
$$

Why can we implement cross entropy like this? Look at the plus sign in the equation - if y = 0, then the left side is equal to 0, and if y = 1, then the right side will be a small number. Effectively, this is how we measure how much our prediction ŷ differs from our label y, which can only be 0 or 1 binary classification algorithm.

Binary cross entropy derivative (gradients)

The compute the gradients for the weights, we are just taking the derivative of our loss function. We can quite compactly describe the loss function as below; for a derivation, see Section 5.10 in [1].

$$
\frac{\partial L_{\text{CE}}(\hat{y}, y)}{\partial \mathbf{w}} = \frac{1}{m} (\mathbf{\hat{y}} - \mathbf{y}) \mathbf{x}_i^{T}
$$

## Python Example

## Conclusion

## Sources

1. https://web.stanford.edu/~jurafsky/slp3/5.pdf

2. http://timvieira.github.io/blog/post/2014/02/11/exp-normalize-trick/