---
layout:     post
title:      "Relevance Vector Machine"
date:       2021-05-15 12:00:00
author:     "GwanSiu"
catalog: true
tags:
    - Machine Learning
---

## 1. Introduction

Support vector machine (SVM) has been widely used in classification and regression problems. The main advantage of SVM is that the decision boundary is sparsely represented by the training set, so we can do the inference more fast. However, SVM suffers from some limitations listed as follows:
- The outputs of SVM do not have a probabilistic interpretation. 
- The SVM is originally formulated for binary classifiction problem, and the extension to $K>2$ classes is problematic.
- The hyperparameter $C$ is chosen through the cross validation.
- The predictions are represented as linear combination of kernel fucntions that are centred on the training data points and that requires to be positive definite.

In this post, we will introduce *relevance vector machine (RVM)* for regression and classification. The RVM is a Bayesian sparse kernel machine and shares many similar characteristics with SVM while avoiding the major limiations of SVM.

The content of this post is organized as follows: we will introduce the RVM for regression tasks in Section 2, and then describe how the RVM is applied to the classifiction tasks in Section 3. Before we start the topic, we should review Baye's rule, given by

$$
\begin{equation}
p(\mathbbf{w}\vert D) = \frac{p(D\vert \mathbf{w})p(\mathbf{w})}{p(D)}
\end{equation}
$$

## 2. RVM for Regression

Given a dataset $D=\{(\mathbf{x}^{(i)}, y^{(i)})\}_{i=1}^{N}$ with $N$ samples, where $\mathbf{x}^{(i)}\in\mathbb{R}^{1\times D}, y^{(i)}\mathbb{R}$. $X=[\mathbf{x}^{(i)}^{T}, \cdots,\mathbf{x}^{(N)}^{T}]$, $Y=[y^{(i)}, \cdots, y^{(N)}]$. According to the Bayes'theorem, given by

$$
\begin{equation}
p(\mathbbf{w}\vert D) = \frac{p(D\vert \mathbf{w})p(\mathbf{w})}{p(D)}.
\end{equation}
$$

In the learning stage, we would like to estimate the model parameters $\mathbf{w}$ based on the observations. In the inference stage, given the testing data $\mathbf{x}^{\ast}$, we can compute the marginal distribution of the predicted target value $y^{\ast}$ as follows:

$$
\begin{equation}
p(y^{\ast}\vert \mathbf{x}^{\ast}) = \displaystyle{\int} p(y^{\ast} \vert \mathbf{x}^{\ast}, \mathbf{w}) p(\mathbbf{w}\vert D) \mathrm{d}\mathbf{w}.
\end{equation}
$$

Next, we will explicitly describe the likelihood funciton, the prior, and the calculation of the posterior distribution, respectively.

The likelihood function is given by

$$
\begin{align}
p(Y\vert X, \mathbf{w}, \beta) &= \prod_{i=1}^{N} p(y^{(i)}\vert \mathbf{x}^{(i)}, \mathbf{w}, \beta), \\
&=\mathcal{N}(y^{(i)}\vert f(\mathbf{w}, \mathbf{x}^{(i)}), \beta^{-1}).
\end{align}
$$

We adopt a zero-mean Guassian as the prior. Specifically, we introduce a seperate hyperparameter $\alpha_{i}$ for each of the weight parameter $w_{i}$ instead of a single shared hyperparameter. The prior is defined by

$$
\begin{equation}
p(\mathbf{w}\vert \mathbf{\alpha}) = \prod_{i=1}^{D} \mathcal{N}(w_{i}\vert 0, \alpha_{i}^{(-1)}),
\end{equation}
$$

where $\alpha_{i}$ represents the precision of the corresponding parameter $w_{i}$ and $\mathbf{\alpha}=[\alpha_{1}, \cdots, \alpha_{D}]$.

The posterior distribution for the weights is Gaussian, given by

$$
\begin{align}
p(\mathbf{w}\vert X, Y, \beta,\mathbf{\alpha}) &= \frac{p(Y\vert X, \mathbf{w}, \beta)p(\mathbf{w}\vert \mathbf{\alpha})}{p(Y\vert X, \mathbf{\alpha}, \beta)}, \\
&\propto \mathcal{N}(\mathbf{w}\\mathbf{m}, \Sigma).
\end{align}
$$

where $p(Y\vert X, \mathbf{\alpha}, \beta)$ is called evidence. The mean and variance are given by

$$
\begin{align}
\mathbf{m} &= \beta \Sigma \Phi^{T} Y,\\
\Sigma &= (A + \beta\Phi^{T}\Phi)^{-1}.  
\end{align}
$$

We can use type-2 maximumum likelihood to determine the values of $\mathbf{\alpha}$ and $\beta$, which is also called evidence approximation. Specifically, we maximize the marginal likelihood function obteined by integrating out the weight parameters, which is given by

$$
\begin{equation}
p(Y\vert X, \mathbf{\alpha}, \beta) = \displaystyle{\int} p(Y\vert X, \mahtbf{w}, \beta)p(\mathbf{w}\vert \mathbf{\alpha})\mathrm{d}\mathbf{w}.
\end{equation}
$$

The convolution of two Gaussian is still a Gaussian. The log marginal likelihood is given by 

$$
\begin{align}
\ln p(Y\vert X, \mathbf{\alpha}, \beta) &= \ln \mathcal{N}(Y\vert 0, C)
&= -\frac{1}{2}\left(N\ln(2\pi) + \ln\vert C\vert + Y^{T}C^{-1}Y \right),
\end{align} 
$$

where $C$ is a $N\tiems N$ matrix given by

$$
\begin{equation}
C= \beta^{-1} I + \Phi A^{-1} \Phi^{T}.
\end{equation}
$$

Our goal is to maximize Eq. with respect to the hyperparameter $\mathbf{\alpha}$ and $\beta$. There are two approaches to do so. First, we take the derivatives of the marginal likelihood and set it to zero, leading to following re-estimation equations, given by

$$
\begin{align}
\alpha_{i}^{new} &= \frac\gamma_{i}}{m_{i}^{2}} \\ 
(\beta^{new})^{-1} &= \frac{\left \Arrowvert Y - \Phi\mathbf{m}\right\Arrowvert^{2}}{N - \Sigma_{i}\alpha_{i}},
\end{align}
$$

where $\mathbf{m}_{i}$ is the $i$-th components of the posterior mean. The quantity $\gamma_{i}$ measures how well the correspongding parameter $w_{i}$ is determined by the data and is defined by

$$
\begin{equation}
\gamma_{i} = 1- \alpha_{i}\Sigma_{ii},
\end{equation}
$$

in which $\Sigma_{ii}$ is the $i$-th diagonal component of the posterior covariance $\Sigma$.

## 3. RVM for Classifiction

In this section, we will generalize the framework of the RVM for classification tasks. For simplicity, we consider the binary classification problem, in which the target variable $y$ is either 0 or 1. We adopt sigmoid function used in logistic regression for classification, which is given by

$$
\begin{equation}
h(x) = \frac{1}{1 + \exp(-x)}.
\end{equation}
$$

The different of the RVM is the prior term, which is the ADR prior. This means that a separate precision hyperparameter is associated with each weight parameter. 

For a fixed value of $\mathbf{\alpha}$, we can obtain the parameters $\mathbf{w}$ by maximizing the posterior distribution, given by

$$
\begin{align}

\ln p(\mathbf{w}\vert X, Y, \mathbf{\alpha}) = \ln \left( p(Y\vert X, \mathbf{w})p(\mathbf{w}\vert \mathbf{\alpha}) \right) - \ln p(Y\vert \mathbf{\alpha}), \\
&= \sum_{i=1}^{N}\left( y^{(i)}\ln \hat{y}^{(i)} + (1 - y^{(i)})\ln (1-\hat{y}^{(i)}), \right) - \frac{1}{2}\mathbf{w}^{T}A\mathbf{w} + C
\end{align}
$$

where $A=\text{diag}(\alpha_{i})$, and $C$ is a constant value. We can use iterative reweighted least squares (IRLS) algorithm to maximize the above equation, leading to

$$
\begin{align}
    \mathbf{w}^{\ast} &= A^{-1}\Phi^{T}(Y-\hat{Y}), \\
    \Sigma &= (\Phi^{T}B\Phi + A)^{-1}.
\end{align}
$$

Because the likelihood distribution and the prior distribution are not conjugate, we cannot obtain an analytical formulation by integrating over  parameters $\mathbf{w}$. Therefore, we use the Laplacian approximation to evaluate the marginal likelihood function, given by

$$
\begin{equation}
p(Y\vert X,\mathbf{\alpha}) &=\displaystyle{\int} p(Y\vert X, \mathbf{w})p(\mathbf{w}\vert \mathbf{\alpha})\mathrm{d}\mathbf{w}, \\
&\approx p(Y\vert \mathbf{w}^{\ast})p(\mathbf{w}^{\ast}\vert \mathbf{\alpha})(2\pi)^{M/2}\vert \Sigma\vert^{1/2}.
\end{equation}
$$

We take the derivative of the marginal likelihood with respect to $\alpha_{i}$, and set it equal to zero, leading to

$$
\begin{equation}
-\frac{1}{2}(w_{i}^{\ast})^{2} + \frac{1}{2\alpha_{i}} - \frac{1}{2}\Sigma_{ii} = 0 .
\end{equation}
$$

Let $\gamma_{i}= 1- \alpha_{i}\Sigma_{ii}$, and we can obtain

$$
\begin{equation}
\alpha_{i}^{new} = \frac{\gamma_{i}}{(w_{i}^{\ast})^{2}}.
\end{equation}
$$

This is the re-estimation formula obtained for the regression RVM.

One of the potential advantage of the relevance vector machine is that it can make probabilistic predictions. We can easily generalize the RVM to a multi-class classification problem, in which $C\geq 2$ classes. We adopt $K$ linaer model of the form

$$
\begin{equation}
\alpha_{k} = \mathbf{w}_{k}^{T}\mathbf{x},
\end{equation}
$$

which can be combined using a softmax function to give outputs

$$
\begin{equation}
\hat{y}_{k} = \frac{\exp(\alpha_{k})}{\sum_{i}\exp(\alpha_{i})}.
\end{equation}
$$

The log likelihood function is then given by

$$
\begin{equation}
\ln p(\hat{Y}\vert \mathbf{w}^{(1)}, \cdots, \mathbf{w}^{(K)}) = \prod_{n=1}^{N}\prod_{k=1}^{K}\hat{y}_{nk}^{y_{nk}},
\end{equation}
$$

where the target value $y_{nk}$ is a one-hot vector with the dimension of $K$ for each data point $\mathbf{x}_{n}$, and $T$ is a matrix with element $y_{nk}$.







