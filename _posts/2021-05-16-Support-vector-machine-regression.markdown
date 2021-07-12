---
layout:     post
title:      "Support Vector Machine for Regression"
date:       2021-05-15 12:00:00
author:     "GwanSiu"
catalog: true
tags:
    - Machine Learning
---

## 1. Introduction

We have discussed the support vector machine (SVM) and its application in classification problems in the previous posts. In this post, we will talk about the formulation of SVM in regression problems.

## 2. From Ridge Regression to SVM Regression 

We assume that there are $N$ observed data samples. Let $\mathbf{x}^{(i)} \in \mathbb{R}^{d}$ denote the feature vector of the $i$-th observed data, and $y^{(i)}\in \mathbb{R}$ is the corresponding target value. In ridge regression, the objective function is a regularized function, which is defined as follows:

$$
\begin{equation}
\frac{1}{2}\sum_{i=1}^{N}(f(\mathbf{x}^{(i)})-y^{(i)})^{2} + \frac{\lambda}{2}\Arrowvert \mathbf{w}\Arrowvert^{2}_{2},
\end{equation}
$$

where $f(\mathbf{x}) = \mathbf{w}^{T}\mathbf{x}+b$, where $\mathbf{w}\in\mathbb{R}^{d}$ and $b\in\mathbb{R}$ are the model paramters and bias term, respectively. 

Because the solutions of SVM are a sparse representation of the training samples, we replace the quadratic term with the $\epsilon$-insentive error function in Eq.(1). The $\epsilon$-insentive error function is defined as follows:

$$
\begin{equation}

h(x)=\begin{cases}
0, &\text{if} \vert x\vert < \epsilon, \\
\vert x\vert -\epsilon, &\text{otherwise},
\end{cases}
\end{equation}
$$

and is illustrated in Figure 1. It is shown that the $\epsilon$-insentive error function gives zero error if the obsolute difference between the prediction $f(\mathbf{x}^{(i)})$ and the target $y^{(i)}$ is less than $\epsilon$, where $\epsilon>0$. 

Therefore, we can re-write the regularized error function as follows:

$$
\begin{equation}
C\sum_{i=1}^{N}g(f(\mathbf{x}^{(i)})-y^{(i)})+\frac{1}{2}\Arrowvert \mathbf{w}\Arrowvert^{2}_{2},
\end{equation}
$$

where $C$ is a regularization parameter, which balances the regularized term and the $\epsilon$-insentive error function.

Next, we introduce two slack variables $\xi^{(i)}$ and $\hat{\xi}^{(i)}$ for each data point $\mathbf{x}^{(i)}$, where $\xi^{(i)}>0$ corresponds to a point for which $y^{(i)} > f(\mathbf{x}^{(i)}) + \epsilon$, and $\hat{\xi}^{(i)}>0$ corresponds to a point for which $y^{(i)}<f(\mathbf{x}^{(i)})-\epsilon$. By introducing the slack variables, we allow data points to lie outside the tube when the slack variables are nonzero, and hence we can derive the object function for SVM regression.

## 3. The Formulation of SVM Regression

The objective function of SVM for regression can be formulated as follows:

$$
\begin{align}
C\sum_{i=1}^{N}&(\xi^{(i)} + \hat{\xi}^{(i)}) + \frac{1}{2}\Arrowvert \mathbf{w}\Arrowvert_{2}^{2}, \\
s.t. & y^{(i)} \leq f(\mathbf{x}^{(i)}) + \epsilon + \xi^{(i)}, \\
& y^{(i)} \geq f(\mathbf{x}^{(i)}) - \epsilon - \hat{\xi}^{(i)}, \\
& \xi^{(i)}\geq 0, \hat{\xi}^{(i)}\geq 0.
\end{align}
$$

By introducing Lagrange multiplers $\alpha^{(i)}\geq 0, \mu^{(i)}\geq 0$ and $\hat{\mu}^{(i)}\geq 0$, the Lagrange function is

$$
\begin{equation}
\begin{split}
L= &C\sum_{i=1}^{N}(\xi^{(i)} + \hat{\xi}^{(i)}) + \frac{1}{2}\Arrowvert \mathbf{w} - \sum_{i}^{N}(\mu^{(i)}\xi^{(i)} + \hat{\mu}^{(i)}\hat{\xi}^{(i)}) \\
& -\sum_{i=1}^{N}\alpha_{i}(\epsilon + \xi^{(i)} + f(\mathbf{x}^{(i)})-y^{(i)}) -\sum_{i=1}^{N}\hat{\alpha}^{(i)}(\epsilon + \hat{\xi}^{(i)} - f(\mathbf{x}^{(i)}) + y^{(i)}).
\end{split}
\end{equation}
$$

We take the derivative of the Lagrange function with respect to $\mathbf{w}, b, \xi$ and $\hat{\xi}^{(i)}$ and set it to zero, giving

$$
\begin{align}
\frac{\partial L}{\partial \mathbf{w}} = 0 &\Rightarrow \mathbf{w} = \sum_{i=1}^{N}(\alpha^{(i)}-\hat{\alpha}^{(i)})\Phi(\mathbf{x}^{(i)})， \\

\frac{\partial L}{\partial b}=0 &\Rightarrow \sum_{i=1}^{N}(\alpha^{(i)} - \hat{\alpha}^{(i)}) = 0， \\

\frac{\partial L}{\partial \xi^{(i)}} = 0 &\Rightarrow \alpha^{(i)} + \mu^{(i)} = C， \\

\frac{\partial L}{\partial \xi^{(i)}} = 0 &\Rightarrow \hat{\alpha}^{(i)} + \hat{\mu}^{(i)} = C,

\end{align}
$$

where $\phi(\cdot)$ denote the basis function. We use the obtained results to eliminate the variables in the Lagrange function. Instead, we maximize the dual problem defined as follows:

$$
\begin{align}
\hat{L}(\mathbf{\alpha}, \hat{\alpha}) = &-\frac{1}{2}\sum_{i=1}^{N}\sum_{j=1}^{N}(\alpha^{(i)} - \alpha^{(j)})K(\mathbf{x}^{(i)}, \mathbf{x}^{(j)}) -\epsilon \sum_{i=1}^{N}(\alpha^{(i)} + \hat{\alpha}^{(i)}) + \sum_{i=1}^{N}(\alpha^{(i)} - \hat{\alpha}^{(i)})y^{(i)}, \\
s.t. &0\leq \alpha^{(i)}\leq C, i=1,\cdots, N, \\
& 0\leq \alpha^{(i)} \leq C, i=1, \cdots, N.
\end{align}
$$

We can still obtain the box constraints, which have been discussed in the post of SVM for classification.

We investigate more insights about this formulation through KKT condition, which states that the product of the dual variables and the constraints must vanish at the optimal solution point. The KKT conditions are given by

$$
\begin{align}
\alpha^{(i)} (\epsilon + \xi^{(i)} + f(\mathbf{x}^{(i)}) - y^{(i)}) &=0, \\
\hat{\alpha}^{(i)} (\epsilon + \xi^{(i)} - f(\mathbf{x}^{(i)}) + y^{(i)}) &= 0, \\
(C- \alpha^{(i)})\xi^{(i)} &= 0, \\
(C - \hat{\alpha}^{(i)}) &= 0.
\end{align}
$$

As observed, 

- A coefficient $\alpha^{(i)}$ can only be nonzero if $\epsilon + \xi^{(i)}+f(\mathbf{x}^{(i)}) - y^{(i)}=0$, which implies that data point either lies on the upper boundary of the $\epsilon$-tube ($\xi^{(i)}=0$) or lies above the upper boundary $(\xi^{(i)} > 0)$. Similarly, a nonzero value for $\hat{\alpha}^{(i)}$ implies $\epsilon + \hat{\xi}^{(i)} - f(\mathbf{x}^{(i)}) + y^{(i)} = 0$, and such points must lie either on or below the lower boundary of the $\epsilon$-tube. Futhermore, the two constraints $\epsilon + \xi^{(i)} + f(\mathbf{x}^{(i)}) - y^{(i)} = 0$ and $\epsilon + \hat{\xi}^{(i)} - f(\mathbf{x}^{(i)}) + y^{(i)} = 0$ are incompatible. We can find that $\xi^{(i)}$ and $\hat{\xi}^{(i)}$ are nonnegative while $\epsilon$ is strictly positive, and for every data point $\mathbf{x}^{(i)}$, either $\alpha^{(i)}$ and $\hat{\alpha}^{(i)}$ must be zero.

In the infernce stage, we can predict a new data point $\mathbf{x}^{test}$ through

$$
\begin{equation}
f(\mathbf{x}^{test}) = \sum_{i=1}^{N}(\alpha^{(i)} - \hat{\alpha}^{(i)})K(\mathbf{x}^{test},\mathbf{x}^{(i)}) + b,
\end{equation}
$$

where 

$$
\begin{align}
b &= y^{(i)} - \epsilon - \mathbf{w}^{T}\phi(\mathbf{x}^{(i)}) \\ 
&=y^{(i)} - \epsilon - \sum_{j=1}^{N}(\alpha^{(j)} - \hat{\alpha}^{(j)})K(\mathbf{x}^{(i)}, \mathbf{x}^{(j)}).
\end{align}
$$

The support vectors are those data points that either $\alpha^{(i)}\neq 0$ or $\hat{\alpha}^{(i)}\neq =0$.

## Reference

All materials of this post come from the reference listed as follows:

[1] Bishop, Christopher M. Pattern recognition and machine learning. springer, 2006.












