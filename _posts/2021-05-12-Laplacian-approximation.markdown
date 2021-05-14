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
Let $h(x)=\log g(x)$. We can re-write the Eq.($\ref{eq.1}$) as follows:

$$
\begin{equation}
\displaystyle{\int_{a}^{b}\exp(h(x))}\mathrm{d}x.
\end{equation}
$$
We will take a Taylor series approximation of $h(x)$ at the point $x_{0}$, and we have
$$
\begin{equation}
\displaystyle{\int_{a}^{b}}\exp(h(x))\mathrm{d}x \approx \displaystyle{\int_{a}^{b}\exp(h(x_{0}))+h^{\prime}(x_{0})(x-x_{0})+\frac{1}{2}h^{\prime\prime}(x_{0})(x-x_{0})}\mathrm{d}x.
\end{equation}
$$
Because the function $g(x)$ achieves its maximum at the point $x_{0}$ and $\log(\cdot)$ is a monotonic function, we have $h^{\prime}(x_{0})=0$. Therefore, Eq.(2) can be simplified as follows:
$$
\begin{equation}
\displaystyle{\int_{a}^{b}}\exp(h(x))\mathrm{d}x \approx \displaystyle{\int_{a}^{b}}\exp(h(x_{0}) + \frac{1}{2}h^{\prime\prime}(x_{0})(x-x_{0})^{2})\mathrm{d}x
\end{equation}
$$
It is found that the term $h(x_{0})$ does not depend on $x$, so that it can be pulled outside the integral. In addition, we can rearrange the term $\frac{1}{2}h^{\prime\prime}(x_{0})(x-x_{0})^{2})$, and have

$$
\begin{equation}
\displaystyle{\int_{a}^{b}}\exp(h(x))\mathrm{d}x \approx \exp(h(x_{0})) \displaystyle{\int_{a}^{b}-\frac{1}{2}\frac{(x-x_{0})^{2}}{-h^{\prime\prime}(x_{0})^{-1}}}.
\end{equation}
$$

It is observed that the term $displaystyle{\int_{a}^{b}-\frac{1}{2}\frac{(x-x_{0})^{2}}{-h^{\prime\prime}(x_{0})^{-1}}}$ is proportional to a Gaussian distribution with the mean $x_{0}$ and the variance $-h^{\prime\prime}(x_{0})^{-1}$.

Let $\Phi(x\vert \mu, \sigma^{2})$ be a cumulative function for a Gaussian distribution with the mean $\mu$ and the variance $\sigma^{2}$. We can re-write Eq.(4) as follows:

$$
\begin{align}
\displaystyle{\int_{a}^{b}\exp(h(x))}\mathrm{d}x &\approx \exp(h(x_{0})) \displaystyle{\int_{a}^{b}-\frac{1}{2}\frac{(x-x_{0})^{2}}{-h^{\prime\prime}(x_{0})^{-1}}} \\
&=\exp(h(x_{0}))\sqrt{\frac{2\pi}{-h^{\prime\prime}(x_{0})}}\left[\Phi(b\vert x_{0}, -h^{\prime\prime}(x_{0})^{-1}) - \Phi(a\vert x_{0}, -h^{\prime\prime}(x_{0})^{-1})\right] 
\end{align}
$$

Given that $\exp(h(x_{0}))=g(x_{0})$. If $b=\infty, a=-\infty$, the term inside the bracket is equal to 1. By these condistions, the Laplacian approximation is equal to the value $g(x_{0})$ mutiplied by a constant term $\sqrt{\frac{2\pi}{-h^{\prime\prime}(x_{0})}}$, which depends on the curvature of $h(x)$. It means that the function $h(x)$ should be twice differentiable.


