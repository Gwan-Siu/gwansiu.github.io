---
layout:     post
title:      "Linear Regression"
date:       2018-06-17 12:00:00
author:     "GwanSiu"
catalog: true
tags:
    - Machine Learning
---

## 1. Linear Regression Algorithm

### 1.1 Linear Regression

Originally, `linear regression` is a linear combination of input variables, the goal of linear regression is find a polynomial funciton to approximate the target function. Given dataset $D=\{(x_{1},y_{1}),...,(x_{N}, y_{N})\}$, where $x_{i}\in \mathbb{R}^{K}, y\in\mtahbb{R}$ for $i=1,...,N$, $N$ is the number of data. The simplest form of linear regression is:

$$
\begin{align}
\hat{y_{i}} &= \omega_{0} + \omega_{1}x_{i}^{1}+...+\omega_{K}x_{i}^{K} \\
&= \mathbf{W}^{T}\mathbf{x_{i}}
\end{align}
$$

where $\mathbf{W}=\{\omega_{0},...,\omega_{K}\}\in\mathbb{R}^{K}$ is parameters.

Our goal is to find minimize the objective function, in this case, we minimize the least square error:

$$
\begin{equation}
\omega^{\ast}=\arg\min_{\omega}J(\omega)=\arg\min_{\omega}\frac{1}{2}\sum_{i=1}^{N}(\mathbf{W}^{T}x_{i}-y_{i})^{2}
\end{equation}
$$

There are two reasons to adopt least square error as objective function that 1. Reduce distance between true measurements and predicted hyperplane. 2. Has a nice probabilistic interpretation.


### 1.2 Learning as Optimization

The general optimization form is:

$$
\begin{equation}
\omega^{\ast}=\arg\min_{\omega}J(\omega)
\end{eqaution}
$$

Iterative method and direct method( or close form) can be used to solve to linear regression problem Iterative method includes gradient descent(GD) and stochastic gradient descent(SGD).

#### 1.2.1 Gradient method

Updated rule:

$$
\begin{equation}
\mathbf{W}_{t+1} = \mathbf{W}_{t} + \lambda\nabla J(\amthbf{W}) 
\end{eqaution}
$$

where $\lambda$ is step size, a hyperparameter.  $\displaystyle{J(\mathbf{W})=\sum_{i}^{N}J_{i}(\mathbf{W})}$, and $J_{i}(\mathbf{W})=\frac{1}{2}(\mathbf{W}x_{i}-y_{i})^{2}$.

The derivative of $J(\mathbf{W})$ is:

$$
\begin{align}
\displaystyle{
\frac{\mathrm{d}}{\mathrm{d}\omega_{k}}J_{i}(\mathbf{W}) &= \frac{\mathrm{d}}{\mathrm{d}\omega_{k}} \frac{1}{2}(\mathbf{W}x_{i}-y_{i})^{2} \\
&= \frac{1}{2}\frac{\mathrm{d}}{\mathrm{d}\omega_{k}}(\mathbf{W}x_{i}-y_{i})^{2} \\
&= (\mathbf{W}x_{i}-y_{i}) \frac{\mathrm{d}}{\mathrm{d}\omega_{k}} (\mathbf{W}x_{i}-y_{i}) \\
&= (\mathbf{W}x_{i}-y_{i}) \frac{\mathrm{d}}{\mathrm{d}\omega_{k}} (sum_{k=1}^{K}\omega_{k}x_{i}^{k}-y_{i}) \\
&= \mathbf{W}x_{i}-y_{i})x_{i}^{k}}
\end{align}
$$

thus we have:

$$
\begin{align}
\displaystyle{
\frac{\mathrm{d}}{\mathrm{d}\omega_{k}}J(\mathbf{W}) &= \frac{\mathrm{d}}{\mathrm{d}\omega_{k}}\sum_{i=1}^{N}J_{i}(\mathbf{W}) \\
&= sum_{i=1}^{N} \mathbf{W}x_{i}-y_{i})x_{i}^{k}}
}
\end{align}
$$

From obove analysis, gradient decent algorithm is requires to compute the grdient in the all dataset, it is time consuming. Instead of computing the gradient over the dataset, we randomly select a subset of dataset, and comput the gradient of the subset. This is the approximated gradient.

#### 1.2.2 Normal Equations

the objective function is:

$$
\displaystyle{
\begin{align}
J(\mathbf{W}) &= \frac{1}{2}\sum_{i=1}^{N}(x_{i}^{T}\mathbf{W}-y_{i})^{2} \\
&= \frac{1}{2}(X\mathbf{W}-y)^{T}(X\mathbf{W}-y) \\
&= \frac{1}{2}(\mathbf{W}^{T}X^{T}X\mathbf{W}-\mathbf{W}^{T}X^{T}y-y^{T}X\mathbf{W}+y^{T}y)
}
\end{align}
$$

To minimize $J(\mathbf{W})$, take the derivative and set to zero:

$$
\begin{align}
\nbala_{\mathbf{}/W}J &=\frac{1}{2}\nbala_{\mathbf{W}}\text{tr}(\mathbf{W}^{T}X^{T}X\mathbf{W}-\mathbf{W}^{T}X^{T}y-y^{T}X\mathbf{W}+y^{T}y) \\
&= \frac{1}{2}(X^{T}X\mathbf{W}+X^{T}X\mathbf{W}-2X^{T}y) \\
&=X^{T}X\mathbf{W}-X^{T}y = 0
\end{align}
$$

The normal equation:

$$
\begin{equation}
X^{T}X\mathbf{W} = X^{T}y
\end{equation}
$$

thus, we have:

$$
\begin{equation}
\mathbf{W} = (X^{T}X)^{-1}X^{T}y
\end{equation}
$$

In most situation, the number of data points $N$ is larger than the dimensionality of $K$, and thus $X$ is full column rank. Thus, $X^{T}X$ is invertible, which is necessary. If the assumption that $X^{T}X$ is a invertible matrix implies that it is positive definite, thus the critical point we have found is a minimum.

If the $X$ is less than full column rank, regularization term should be added.

#### 1.2.3 Direct and Iterative Methods

- **Direct methods:** we can achieve the solution in a single step by solving the normal equation.
  - Using Gaussian elimination or QR decomposition, we converge in a finite number of steps.
  - It can be infeasible when data are streaming in real time, or of very large amount.

- **Iterative Methods:** Stochastic or steepest gradient
    - Converging in a limiting sense
    - But more attractive in large practical problems
    - Caution is needed for deciding the learning rate $\alpha$.  

In general, linear regression can be considered as linear combinations of a fixed set of nonlinear function of input variables, which is known as basis function. The general form is:

$$
\begin{align}
\hat{y} &= \omega_{0} + \omega_{1}\phi(x)_{1}+...+\omega_{n}\phi(x){n} \\
&= \mathbf{W}^{T}\mathbf{\phi(X)}
\end{align}
$$

### 1.3 Non-linear Basis Function

Where $\phi(\cdot)$ denotes a transform function(or basic function). **This can be resolved by dividing the input sapce up into regions and fitting a different polynomial in each region, leading to spline functions.** The is reason is that the plain form of Linear Regression is polynomial basis functions, one of big limitation is that they are global functions of the input variable, so that changes in one region of input space affect all other regions. **In other words, the plain form of linear regression is very sensitive to noise.**
 

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

“tanh” function can be represented by sigmoid function $\sigma(\alpha): \text{tanh}(\alpha)=2\sigma(2\alpha)-1$, and hence a general linear combination of logistic sigmoid function is equivalent to a general linear combination of "tanh" functions.

Another possibility is **Fourier basis**, which is an expansion in sinusoidal functions.  Each basis function represents a specific frequence and the whole spaitial space. In signal processing, we can consider local information in both spatial space and frequency space by using bais function of **wavelets**. Wavelets are most applicable when the input values live on a regualr lattice, such as the seccesive time points in a temporal sequence, or the pixels in an image.  

## 2 Geometric Interpretation

Linear regression is to minimize the squared distance between predicted line and target line.  **The geometric interpretation is to project data into low-dimensional hyperplane, which minimize the distance the between predicted line and target line.** We can see this interpretation from the soltion of closed-form, the original form of linear regression is as example:

$$
\begin{align}
&\arg\min_{\omega}\Arrowvert \mathbf{Y} - \omega^{T} \mathbf{X}\Arrowvert^{2} \\
&\frac{\partial}{\partial\omega}\Arrowvert \mathbf{Y} - \omega^{T} \mathbf{X}\Arrowvert^{2} = 0 \\
\Rightarrow &\omega^{\ast} = \left(X^{T}X)\right)^{-1}X^{T}Y
\end{align}
$$

The predicted line is:

$$
\begin{equation}
\tilde{y} = X\omega{\ast} = X \left(X^{T}X\right)^{-1}X^{T}Y
\end{equation}
$$

Hence, we see that $\tilde{y}$ is the orthogonal projection of target $y$ into the space spanned by the columns of $X$.


<img src="https://raw.githubusercontent.com/Gwan-Siu/BlogCode/master/A59EBCAB-2B12-4371-B902-D118F9C2223D.png" width = "600" height = "400"/>

## 3 Probabilistic Interpretation

From probabilistic view, we assume the target variable and input data follow the linear relation:

$$
\begin{equation}
y_{i} = \mathbf{W}^{T}x_{i} + \epsilon
\end{equation}
$$

where $y_{i}$ is target variable, $x_{i}$ is an input datum and $\epsilon$ is noise term, which usually is independent on data and view as gaussian ditribution. Therefore, we can rewrite the above form:

$$
\begin{equation}
p(y_{i}\vert x_{i},\mathbf{W}) = \frac{1}{\sqrt{2\pi}\sigma}exp\left(-\frac{(y_{i}-\mathbf{W} x_{i})^{2}}{2\sigma^{2}}\right)
\end{equation}
$$

By indepedence assumption:

$$
\begin{equation}
L(\mathbf{W})=\prod_{i=1}^{n}p(y_{i}\vert x_{i},\mathbf{W})\left(\frac{1}{\sqrt{2\pi}\sigma}\right)^{2}exp\left(-\frac{\sum_{i=1}^{n}(y_{i}-\mathbf{W}^{T}x_{i})^{2}}{2\sigma^{2}}\right)
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
J(\omega) = \frac{1}{2}\sum_{i=1}^{n}(x_{i}^{T}\mathbf{W}-y_{i})^{2}
\end{equation}
$$

Thus the form is equivalent to the form of MLE.


## 4 Regularization

Consider the case that we have $n$ **sparse** data, each of which has $p$ attibutes, and $p>>n$. We assume $X_{i}=[x_{i1},...,x_{ip}]$

If we implement linear regression simply, the linear regression algorithm will be completely failed. Because $(X^{T}X)^{-1}$ is not invertible. In detail, $(X^{T}X)^{-1} \in \mathbb{R}^{p\times p}$, but the rank of this matrix is lower than $n$. 

$$
\text{rank}((X^{T}X)^{-1})\leq n<<p
$$

This case is called linear regression in high dimension. 

We can give another intuitive story. In high dimensional space, not every attibute contributes to the target result. Maybe some of them are caused by noise, and hence we should do feature selection in linear regression.

**Ridge regression** and **Lasso** are most commom variant regression model in high-dimensional space.

### 1.3.1 L2 Regularization

Linear regression with L2 norm penalty is called **Ridge Regression**. The form of ridge regression is:

$$
\begin{align}
J_{RR}(\omega) &= J(\omega)+\lambda\Arrowvert W\Arrowvert^{2}_{2} \\
&= \frac{1}{2}\sum_{i=1}^{N}(\omega^{T}x_{i}-y_{i})^{2}+\lambda\sum_{k=1}^{K}\omega^{2}_{k}
\end{align}
$$

**Bayesian interpretation:** MAP estimation with a **Gaussian prior** on the parameters:

$$
\begin{align}
\omega^{MAP} &= \arg\max_{\omega}\sum_{i=1}^{N}\log p_{\omega}(y_{i}\vert x_{i})+\log p(\omega) \\
&= \arg\max_{\omega}J_{RR}(\omega)
\end{align}
$$

where $p(\omega)\sim \mathbb{N}(0,\frac{1}{\lambda})$.

### 1.3.2 L1 Regularization

Linear Regression with L1 regularizer is called **LASSO**. The form of LASSO is:

$$
\begin{align}
J_{LASSO}(\omega) &= J(\omega) +\lambda\Arrowvert K\Arrowvert_{1} \\
&= \frac{1}{2}\sum_{i=1}^{N}(\omega^{T}x_{i}-y_{i})^{2}+\lambda\sum_{k=1}^{K}\vert\omega_{k}\vert
\end{align}
$$

**Bayesian interpretation:** MAP estimation with a **Laplace prior** on the parameters.

$$
\begin{align}
\omega_{MAP} &= \arg\max_{\omega}\sum_{i=1}^{N}\log p_{\theta}(y_{i}\vert x_{i}) + \log p(\omega) \\
&= \arg\max_{\omega} J_{LASSO}(\omega)
\end{align}
$$

**Optimization for LASSO**

Actually, we cannot directly apply SGD to **LASSO** learning problem because the L1 term is subdifferentiable. Consider the term of absolute value function:

$$
\begin{equation}
r(\theta) = \lambda\sum_{k=1}^{K}\vert \theta_{k}\vert
\end{equation}
$$

<img src="https://raw.githubusercontent.com/Gwan-Siu/BlogCode/master/other/4E612965-AF67-4BC0-8E5F-6E16CF48BE4E.png" width = "600" height = "400"/>

Many optimization algorithm exist to handle this issue:

1. Coordinate Descent.
2. Othant-Wise Limited memory Quasi-Newton(OWLQN)(Andrew & Gao, 2007) and provably convergent variants.
3. Block coordinate descent(Tseng & Yun, 2009)
4. Sparse Reconstruction by separable approximation(SpaRSA)
5. Fast Iterative Shrinkage Thresholding Algorithm.

### 1.3.3 Ridge Regression vs LASSO

In this part, I directly cite the content of the course 10701, CMU, 2016. The instructor is Prof.Eric Xing.

<img src="https://raw.githubusercontent.com/Gwan-Siu/BlogCode/master/other/A57E4FDF-5332-40B2-BCBC-580F9F37D355.png" width = "600" height = "400"/>

## 1.4 Advanced Topic on Linear Regression

### 1.4.1 Locally-Weighted Linear Regression

The plain form of linear regression do not consider spatial relation between training data and query data. In order to coorperate with spatial information, we reformulate the form of linear regression:

$$
\begin{equation}
J(\theta) = \frac{1}{2}\sum_{i=1}^{n}\omega_{i}(x_{i}^{T}\theta-y_{i})^{2}
\end{equation}
$$

here, $\omega$ is the parameterm, and $\omega$ is the weight, $\omega=exp\left(-\frac{(x_{i}-x)^{2}}{2\tau}\right)$, where $x$ is the query point for which we'd like to know its corresponding $y$.

**Eseentially, we put higher weights on(errors on) training examples that are close to thequery point(than those that are further away from the query). In addition, locally weighted linear regression is the second kind of non-parametric algorithm.** Locally-weighted linear regression is memory-based method, because the distance between query point and sample points needs to be computed.

<img src="https://i.stack.imgur.com/efEaJ.png" width = "600" height = "400"/>

### 1.4.2 Robust Regression

Locally weighted linear regression algorithm is to score the importance for each point. **Robust Rgression** is to score each point according to its "fittness".

The weight $\omega_{k}$ on robust regression is:

$$
\begin{equation}
\omega_{k} = \phi\left((y_{k}-y_{k}^{est})^{2}\right)
\end{equation}
$$

The weight $\omega_{k}$ for data point $k$ is large if the data point fits well and small if it fits badly.

Algorithm:
- For k=1:R
 - Let $(x_{k},y_{k})$ be the kth datapoint.
 - Let $y_{k}^{est} \text{ be predicted value of } y_{k}$.
 - Let $\omega_{k}$ be a weight for data point $k$ that is large if the data point fits well and small if it fits badly.
 - Repeat whole thing until converged.

**Robust Regression-Probabilistic Interpretation**

- What regular regression does:

Assume $y_{k}$ is the linear combination of input variables with noise:

$$
\begin{equation}
y_{k} = \theta^{T}\mathbf{x}_{k}+\mathbb{N}(0,\sigma^{2})
\end{equation}
$$

Learning: find the maximum likelihood estimator of $\theta$.

- What robust regression does:

Assume $y_{k}$ was generated using the followed schedule:

with probability $p$:

$$
\begin{equation}
y_{k} = \theta^{T}x_{k} + \mathbb{N}(0,\sigma^{2})
\end{equation}
$$

but otherwise

$$
\begin{equation}
y_{k}\sim \mathbb{N}(\mu, \sigma_{huge}^{2})
\end{equation}
$$

learning: computational task is to find the maximum likelihood estimator of $\theta, p,\mu \text{ and } \sigma_{huge}$.

**Iterative reweighting algorithm does this leanring task. EM** algorithm is one of famous instance of this algorithm.






