# Why not use OLS for Logistic Regression?

## My Question

> I am trying to understand the mathematical limitations of using Ordinary Least Squares (OLS) for Logistic Regression, specifically why we cannot find a closed-form solution and must use Gradient Descent with Log-Loss instead. In short, why can't I just minimize the following equation with respect to $m$ and $b$:
> $$
>f(m,b)=SSE=\sum_{i=1}^{n}(y_i-\frac{1}{1+e^{-mx_i-b}})
>$$
>When explaining, please don't be shallow; go deeper in your explanations if you find it necessary.

## Answer (by Gemini 3.1 Pro)

Using Sum of Squared Errors (SSE) with the sigmoid function introduces two major mathematical problems: non-convexity and transcendental equations that prevent a closed-form solution.

### 1. No Closed-Form Solution (Transcendental Equations)
To find a closed-form solution for OLS, you must calculate the partial derivatives of your objective function with respect to parameters $m$ and $b$, set them to zero, and solve for the parameters algebraically:

$$ \frac{\partial SSE}{\partial m} = 0, \quad \frac{\partial SSE}{\partial b} = 0 $$

The derivative of the SSE with the sigmoid function $\sigma(z) = \frac{1}{1+e^{-z}}$ (where $z_i = mx_i + b$) involves the derivative of the sigmoid itself, $\sigma'(z) = \sigma(z)(1 - \sigma(z))$. 

For parameter $m$, the gradient is:
$$ \frac{\partial SSE}{\partial m} = -2 \sum_{i=1}^{n} \left( y_i - \sigma(z_i) \right) \sigma(z_i) (1 - \sigma(z_i)) x_i = 0 $$

Because the parameters $m$ and $b$ are locked inside the denominator's exponential function $e^{-(mx_i+b)}$, they form a system of non-linear, transcendental equations. It is mathematically impossible to isolate $m$ and $b$ algebraically to create a closed-form formula like $(X^TX)^{-1}X^Ty$ used in Linear Regression.

### 2. The Non-Convexity Problem
Because you cannot solve it analytically, you must use numerical optimization like Gradient Descent. However, substituting a non-linear activation function (sigmoid) into the squared error loss function creates a **non-convex** surface.

The term $\sigma(z_i)(1 - \sigma(z_i))$ introduces "wiggles" into the cost function surface. A non-convex function has multiple local minima. If you run Gradient Descent on this SSE formulation, the algorithm is highly likely to converge to a local minimum rather than the absolute best fit (the global minimum), depending entirely on the random weight initialization.

### 3. Why Log-Loss (Binary Cross-Entropy) Fixes This
Logistic regression models the probability of a binary outcome, which follows a Bernoulli distribution. If you apply Maximum Likelihood Estimation (MLE) to find the parameters that maximize the likelihood of observing your training data given the model, you must maximize:

$$ L(m,b) = \prod_{i=1}^{n} \sigma(z_i)^{y_i} (1 - \sigma(z_i))^{1-y_i} $$

Taking the negative natural logarithm of this likelihood yields the Log-Loss (Binary Cross-Entropy) function:

$$ J(m,b) = - \sum_{i=1}^{n} \left[ y_i \log(\sigma(z_i)) + (1-y_i) \log(1-\sigma(z_i)) \right] $$

The logarithm perfectly cancels out the exponential nature of the sigmoid function. The resulting cost function $J(m,b)$ is strictly **convex**. It forms a single, smooth bowl shape with exactly one global minimum, guaranteeing that Gradient Descent will always converge to the optimal parameters.