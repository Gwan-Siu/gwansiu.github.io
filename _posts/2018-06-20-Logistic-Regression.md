---
layout:     post
title:      "Logistics Regression"
date:       2018-06-17 12:00:00
author:     "GwanSiu"
catalog: true
tags:
    - Statistics and Bayesian Analysis
---

## 1. Logistic Regression

**Data:** Inputs are continuous vectors of length $K$. Outputs are discrete.

$$
\begin{equation}
\mathbb{D}={x^{(i)},y^{(i)}}_{i=1}^{N}, \text{ where } \mathbf{x}\in\mathbb{R}
\end{equation}
$$

**Model:** Logistic function applied to dot product of parameters with input vector.

$$
\begin{equation}
p_{\theta}(y=1\vert x) =\frac{1}{1+exp(-\theta^{T}\mathbf{x})}
\end{equation}
$$

**Learning:** Finds the parameters that minimized some objective function.

$$
\begin{equation}
\theta^{\ast}=\arg\min_{\theta}J(\theta)
\end{equation}
$$

Usually, we minimize the negative log conditional likelihood:

$$
\begin{equation}
J(\theta) = -\log\prod_{i=1}^{N}p_{\theta}(y^{(i)}\vert \mathbf{x}^{(i)})
\end{equation}
$$

*Why can't maximize likelihood(as in naive bayes)*


**Prediction:** Output is the most probable class.

$$
\begin{equation}
\tilde{y} &= \arg\max p_{\theta}(y\vert x),y\in \{0,1\}
\end{equation}
$$

## 2. Learning Methods

**learning:** Three approaches to solve $\theta^{\ast}=\arg\min_{\theta}J(\theta)$

- **Approach 1:** Gradient Descent(Take larger-more certain-steps apposite the gradient)
- **Approach 2:** Stochastic Gradient Descent(SGD)(Take many small steps apposite the gradient)
- **Approach 3:** Newton's Method(use second derivatives to better follow curvature)

## 3. Newton's Method and Logistic Regression

The motivation behind **Newton's method** is to use a **quadreatic approximation** of our function to make a good guess where we should step next.

The **Taylor series expansion** for an infinitely differentiable function $f(x),x\in\mathbb{R},$ about a pint $\nu\in\mathbb{R}$ is:

$$
\begin{equation}
f(x) = f(\nu)+\frac{(x-\nu)f^{'}(x)}{1!}+\frac{(x-v)^{2}f^{''}(x)}{2!}+\frac{(x-v)^{3}f^{'''}(x)}{3!}+...
\end{equation}
$$


The **2nd-order Taylor series approximation** cuts off the expansion after the quadratic term:

$$
\begin{equation}

f(x)\approx f(v)+\frac{(x-v)f^{'}(x)}{1!}+\frac{(x-v)^{2}f^{''}(x)}{2!}
\end{equation}
$$

The vector version of Taylor series expansion for an infinitely differentiable function $f(x),\mathbf{x}\in \mathbb{R}^{K}$, about a point $\mathbf{v}\in\mathbb{R}^{K}$ is:

$$
\begin{equation}
f(x) = f(\nu)+\frac{(x-\nu)^{T}\nabla f(x)}{1!}+\frac{(x-v)^{T}\nabla^{2} f(x)(x-v)}{2!}+...
\end{equation}
$$

The **2nd-order Taylor series approximation** cuts off the expansion after the quadratic term:

$$
\begin{equation}
f(x) \approx f(\nu)+\frac{(x-\nu)^{T}\nabla f(x)}{1!}+\frac{(x-v)^{T}\nabla^{2} f(x)(x-v)}{2!}
\end{equation}
$$

Taking the derivative of $\tilde{f}(v)$ and setting to 0 gives us the closed form minimizer of this(convex) quadratic function:

$$
\arg\min_{x}\tilde{f}(x)=x-(\nabla^{2}f(x))^{-1}\nabla f(x)
$$

The added term $\nabla x_{nt}=-(\nabla^{2}f(x))^{-1}\nabla f(x)$ is called Newton's step.

**The algorithm of Newton's Method(Newton-Raphson method)**

Goal: $x^{\ast}=\arg\min_{x}f(x)$

1. Approximate the function with the 2nd order Taylor series:

$$
f(x) \approx f(\nu)+\frac{(x-\nu)^{T}\nabla f(x)}{1!}+\frac{(x-v)^{T}\nabla^{2} f(x)(x-v)}{2!}
$$

2. Compute its minimizer

$$
\arg\min_{x}\tilde{f}(x)=x-(\nabla^{2}f(x))^{-1}\nabla f(x)
$$

3. Step to that minimizer:

$$x = -(\nabla^{2}f(x))^{-1}\nabla f(x)$$

4. Repeat

## 3. Iterative Weighted Least Square(Newton's Method for Logistic Regression)

As we know, linear regression has closed-form solution, but there are no longer closed-form solution for logistic regression due to nonlinearity of the logistic sigmoid function. In fact, the lost function of logistic regression is convex, and it is guaranteed to find a global optimal solution. Furthermore, the error funciton can be solved by an efficient iterative optimization algorithm based on Newton-Rashon iterative optimization scheme.

<img src="hhttps://raw.githubusercontent.com/Gwan-Siu/BlogCode/master/other/196E58C3-B424-4977-806C-DFE0E6E8905E.png" width = "600" height = "400"/>

For **Logistic Regression:**

$$
\begin{equation}
-H^{-1}g = -(X^{T}SX)^{-1}(X^{T}(\mu-y))
\end{equation}
$$

take $-H^{-1}g$ back to $\theta\leftarrow \theta - H^{-1}g$, we obtain:

$$
\begin{align}
\theta&\leftarrow\theta-H^{-1}g \\
&=\theta - (X^{T}SX)^{-1}(X^{T}(\mu-y)) \\
&=(X^{T}SX)^{-1}((X^{T}SX)\theta-(X^{T}(\mu-y))) \\
&= (X^{T}SX)^{-1}X^{T}(SX\theta-(\mu-y)) \\
&= (X^{T}SX)^{-1}X^{T}S(X\theta-S^{-1}(\mu-y) \\ 
&= (X^{T}SX)^{-1}X^{T}Sz
\end{align}
$$

where $z=(X\theta-S^{-1}(\mu-y)$

Compare with the closed-form in linear regression: $\theta^{\ast}=(X^{T}X)^{-1}X^{T}y$, the step $\theta\leftarrow \theta-H^{-1}g$ can be reformlized as $\theta=(X^{T}SX)^{-1}X^{T}Sz$, where $z=(X\theta-S^{-1}(\mu-y)$. It's the same form as the closed-form of linear regression.

Therefore, we can see the above update yields the minimizer for the **weighted least squares problem:**

$$
\begin{align}
\theta^{\ast}\leftarrow \arg\min_{\theta}(z-X\theta)^{T}S(z-X\theta) \\
\arg\min_{\theta} \sum_{i=1}^{N}S_{ii}(z_{i}-\theta^{T}x^{(i)})^{2}
\end{align}
$$

where $S_{ii}$ is the weight of the $i\text{th}$ "training example" consisting of the pair $(x^{(i)},z_{i})$.

The explaination in [wiki](https://en.wikipedia.org/wiki/Iteratively_reweighted_least_squares):

The method of iteratively reweighted least squares (IRLS) is used to solve certain optimization problems with objective functions of the form:

$$
\begin{equation}
\arg\min_{\beta}sum_{i=1}^{n}(y_{i}-f_{i}(\beta))^{p}
\end{equation}
$$

by an iterative method in which each step involves solving a weighted least squares problem of the form:

$$
\begin{equation}
\beta^{t+1} = \arg\min_{\beta}\omega_{i}(\beta^{(t)})(y_{i}-f_{i}(\beta))^{2}
\end{equation}
$$

IRLS is used to find the maximum likelihood estimates of a generalized linear model, and in robust regression to find an M-estimator, as a way of mitigating the influence of outliers in an otherwise normally-distributed data set. For example, by minimizing the least absolute error rather than the least square error.