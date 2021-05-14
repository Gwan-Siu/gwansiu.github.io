---
layout:     post
title:      "Laplacian Approximation"
date:       2021-05-15 12:00:00
author:     "GwanSiu"
catalog: true
tags:
    - Machine Learning
---

## 1. Laplacian Approximation

In this post, we will talk about Laplacian approximation, which has been widely used in machine learning research. **The core idea of Laplacian approximation is to use a well-defined unimodal function to approximate the integral value of a sophisticated function with a Gussian density function**. 

Suppose we consider a function $g(x)\in \mathcal{L}^{2}$ and this function achieves its maximum value at the point $x_{0}$. We can to compute
$$
\begin{equation}
\displaystyle{\int_{a}^{b}}g(x)\mathrm{d}x. \label{eq.1}
\end{equation}
$$
Let $h(x)=\log g(x)$. We can re-write the Eq.($\ref{eq.1}$)