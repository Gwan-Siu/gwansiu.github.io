---
layout:     post
title:      "Linear Regression"
date:       2018-06-17 12:00:00
author:     "GwanSiu"
catalog: true
tags:
    - Statistics and Bayesian Analysis
---

## 1. Linear Regression Algorithm

### 1.1 Linear Regression and Basis Function

Originally, Linear Regression is a linear combination of input variables, the goal of linear regression is find a polynomial funciton to approximate the target function. The simplest form of linear regression is:

$$
\begin{align}
y(\omega, x) &= \omega_{0} + \omega_{1}x_{1}+...+\omega_{n}x_{n} \\
&= \mathbf{W}^{T}\mathbf{X}
\end{align}
$$

where $\omega$ is parameters, and $x$ is input variable, $x\in \mathbb{R}^{M}$, $n$ is the input number.

In general, linear regression can be considered as linear combinations of a fixed set of nonlinear function of input variables, which is known as basis function. The general form is:

$$
\begin{align}
y(\omega, x) &= \omega_{0} + \omega_{1}\phi(x)_{1}+...+\omega_{n}\phi(x){n} \\
&= \mathbf{W}^{T}\mathbf{\phi(X)}
\end{align}
$$

**Basis Functions**

The plain form of Linear Regression is polynomial basis functions, one of big limitation is that they are global functions of the input variable, so that changes in one region of input space affect all other regions. **This can be resolved by dividing the input sapce up into regions and fitting a different polynomial in each region, leading to spline functions.**

In fact, there are many other possible choices for basis functions, for example, **gaussian basis function:**

$$
\begin{equation}
\phi_{j}(x) = exp\left\{-\frac{(x-\mu_{j})^{2}}{2s^{2}}\right\}
\end{equation}
$$

where the $\mu_{j}$ govern the locations of the basis functions in input space, and the parameter $s$ governs their spatial scale.

Another possible choice is the sigmoidal basis function of the form:

$$
\begin{equation}
\phi_{j}(x) = \sigma\left(\frac{(x-\mu_{j})}{s}\right)
\end{equation}
$$

where $\sigma(\alpha)$ is the logistic sigmoid function defined by:

$$
\begin{equation}
\sigma(\alpha)=\frac{1}{1+exp(-\alpha)}
\end{equation}
$$

“tanh” function can be represented by sigmoid function $\sigma(\alpha): \text{tanh}(\alpha)=2\sigmoid(2\alpha)-1$, and hence a general linear combination of logistic sigmoid function is equivalent to a general linear combination of "tanh" functions.

Another possibility is **Fourier basis**, which is an expansion in sinusoidal functions.  Each basis function represents a specific frequence and the whole spaitial space. In signal processing, we can consider local information in both spatial space and frequency space by using bais function of **wavelets**. Wavelets are most applicable when the input values live on a regualr lattice, such as the seccesive time points in a temporal sequence, or the pixels in an image.  

### 1.2 Geometric Interpretation

Linear regression is to minimize the squared distance between predicted line and target line.  **The geometric interpretation is to project data into low-dimensional hyperplane, which minimize the distance the between predicted line and target line.** We can see this interpretation from the soltion of closed-form, the original form of linear regression is as example:

$$
\begin{align}
\arg\min_{\omega}\Arrowvert \mathbf{Y} - \omega^{T} \mathbf{X}\Arrowvert^{2} \\
&\frac{\partial}{\partial \omega}&\Arrowvert \mathbf{Y} - \omega^{T} \mathbf{X}\Arrowvert^{2} = 0 \\
\Rightarrow &\omega^{\ast} = \left(X^{T}X)\right)^{-1}X^{T}Y
\end{align}
$$

The predicted line is:

$$
\begin{equation}
\tilde{y} = X\omega{\ast} = X \left(X^{T}X)\right)^{-1}X^{T}Y
\end{equation}
$$

Hence, we see that $\tilde{y}$ is the orthogonal projection of target $y$ into the space spanned by the columns of $X$.

### 1.3 Probabilistic Interpretation

From probabilistic view, we assume the target variable and input data follow the linear relation:

$$
\begin{equation}
y_{i} = \omega^{T}x_{i} + \epsilon
\end{equation}
$$

where $y_{i}$ is target variable, $x_{i}$ is an input datum and $\epsilon$ is noise term, which usually is view as gaussian ditribution. Therefore, we can rewrite the above form:

$$
\begin{equation}
p(y_{i}\vert x_{i},\omega) = \frac{1}{\sqrt{2\pi}\sigma}exp\left(-\frac{(y_{i}-\omega x_{i})^{2}}{2\sigma^{2}}\right)
\end{equation}
$$

By indepedence assumption:

$$
\begin{equation}
L(\omega)=\prod_{i=1}^{n}p(y_{i}\vert x_{i},\omega)\left(\frac{1}{\sqrt{2\pi}\sigma}\right)^{2}exp\left(-\frac{\sum_{i=1}^{n}(y_{i}-\omega^{T}x_{i})^{2}}{2\sigma^{2}}\right)
\end{equation}
$$

Hence, the log-likelihood is:

$$
\begin{equation}
l(\omega) = n\log\frac{1}{\sqrt{2\pi}\sigma}-\frac{1}{\sigma^{2}}\frac{1}{2}\sum_{i=1}^{n}(y_{i}-\omega^{T}x_{i})^{2}
\end{equation}
$$

to simplify the log-likelihood:

$$
\begin{equation}
J(\omega) = \frac{1}{2}\sum_{i=1}^{n}(x_{i}^{T}\theta-y_{i})^{2}
\end{equation}
$$

Thus the form is equivalent to the form of MLE.
### 1.4 Analysis of Linear Regression

## 1.2 Learning of Linear Regression(Optimization)

### 1.2.1 Gradient Descent
### 1.2.2 Stochastic Gradient Descent
### 1.2.3 Closed Form

## 1.3 Regularization

### 1.3.1 L2 Regularization
### 1.3.2 L1 Regularization

## 1.4 Advanced Topic on Linear Regression

### 1.4.1 Locally-Weighted Linear Regression
### 1.4.2 Robust Regression
