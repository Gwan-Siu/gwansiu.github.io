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
&= \mathbf{\omega}^{T}\mathbf{x}
\end{align}
$$

where $\omega$ is parameters, and $x$ is input variable, $x\in \mathbb{R}^{M}$, $n$ is the input number.

In general, linear regression can be considered as linear combinations of a fixed set of nonlinear function of input variables, which is known as basis function. The general form is:

$$
\begin{align}
y(\omega, x) &= \omega_{0} + \omega_{1}\phi{x}_{1}+...+\omega_{n}\phi{x}_{n} \\
&= \mathbf{\omega}^{T}\mathbf{\phi{x}}
\begin{align}
$$

### 1.2 Geometric Interpretation
### 1.3 Probabilistic Interpretation
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
