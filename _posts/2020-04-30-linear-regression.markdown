---
layout:     post
title:      "Linear Regression(1)"
date:       2020-04-29 12:00:00
author:     "GwanSiu"
catalog: true
tags:
    - Machine Learning
---

> Linear regression is an old topic in the machine learning community, and this topic has been studied by researchers for the past decades. In this post, I will highlight some kerypoints on regression models. Specifically, I will begin with the univariate regression model, and consider it as the basic block to build the multiple regression model.

# The univariate regression
Suppose that we have a dataset $\mathcal{D}={\mathbf{x},mathbf{y}}$ with $n$ samples, where observation $\mathbf{y}$ is a n-dimension vector, i.e. $\mathbf{y}=(y_{1}, y_{2}, \cdots, y_{n})\in\mathbb{R}^{n}$, and measurement $\mathbf{x}$ is also a n-dimension vector, i.e. $\mathbf{x}=(x_{1}, x_{2}, \cdots, x_{n)\in\mathbb{R}^{n}$. We additionally assume that obseravation and measurement can be modeled as 

$$
\begin{equation}
\mathbf{y} =\beta^{\ast}\mathbf{x} +\epsilon
\end{equation},
$$\tabularnewline

where $\beta^{\ast}\in \mathbb{R}$ is the ground-truth coefficient, which is unknown, and $\mathbf{\epsilon}=(\epsilon_{1}, \epsilon_{2},\codts, \epsilon_{n})\in\mathbb{R}^{n}$ is the noise term, and $\mathbb{E}[\epsilon_{i}]=0, \text{Var}(\epsilon_{i})=\sigma^{2}, \forall i$, $\text{Cov}(x_{i}, x_{j})=0,\text{for } i\neq j$. T


Our goal is to estimate the coefficient in the underlying model, and we commonly use the least mean square estimator(LMSE). We formulate it as follows:

$$
\begin{equation}
\hat{\beta} =\underset{\beta}{\mathrm{argmin}}\,\sum_{i=1}^{n}(y_{i}-\beta x_{i})^{2}=\underset{\beta}{\mathrm{argmin}}\,\Arrowvert \mathbf{y}-\beta\mathbf{x}\Arrowvert^{2}.
\end{equation}
$$

To obtain the optimal $\hat{\beta}$, we take the derivative of Eq.(2), and set the first-order derivative to zero. The univariate linear regression coefficient of $\mathbf{y}$ on $\mathbf{x}$ is 

$$
\begin{equation}
	\hat{\beta} = \frac{\sum_{i=1}^{n}x_{i}y_{i}}{\sum_{i=1}^{n}x_{i}^{2}} =\frac{\mathbf{x}^{T}\mathbf{y}}{\Arrowvert \mathbf{x}\Arrowvert^{2}}.
\end{equation}
$$

Next, we consider the incepter in the underlying linear model. Eq.(1) is reformulated as follows

$$
\begin{equation}
y = \beta_{0}^{\ast}+\beta_{1}^{\ast}x+\epsilon.
\end{equation}
$$

Correspondingly, we alternative optimize the following problem, i.e.

$$
\begin{equation}
(\hat{\beta}_{0}, \hat{\beta}_{1})=\underset{\beta_{0}, \beta_{1}}{\mathrm{argmin}}\, \sum_{i=1}^{n}(y_{i}-\beta_{0}-\beta_{1} x_{i})^{2}=\underset{\beta}{\mathrm{argmin}}\,\Arrowvert \mathbf{y}-\beta_{0}\mathbf{1}-\beta_{1}\mathbf{x}\Arrowvert^{2}.
\end{euqation}
$$

by solving the above problem, and we can obtain

$$
\begin{equation}
\begin{split}
\hat{\beta}_{0} &= \bar{y}-\hat{\beta}_{1}\bar{x}  \\
\hat{\beta}_{1} &= \frac{(\mathbf{x}-\bar{x}\mathbf{1})^{T}(\mathbf{y}-\bar{y}\mathbf{1})}{\Arrowvert \mathbf{x}-\bar{x}\mathbf{1}\Arrowvert^{2}_{2}}
\end{split}
\end{equation}
$$

note that

$$
\begin{equation}
\hat{\beta}_{1}=\frac{cov(\mathbf{x},\mathbf{y})}{var{\mathbf{x}}}=cov(\mathbf{x},\mathbf{y})\sqrt{\frac{var{y}}{var(x)}}
\end{equation}
$$






