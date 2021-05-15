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

We will talk about the Laplacian approximation in this post, which has been widely used in machine learning research. **The core idea of Laplacian approximation is to use a well-defined unimodal function to approximate the integral value of a sophisticated function with a Gaussian density function**.

Suppose we consider a function $g(x)\in \mathcal{L}^{2}$ and this function achieves its maximum value at the point $x_{0}$. The integral value of this function in $[a, b]$ is computed as follows:

$$
\begin{equation}
\displaystyle{\int_{a}^{b}}g(x)\mathrm{d}x.
\end{equation}
$$

Let $h(x)=\log g(x)$. We can re-write the Eq.(1) as follows:

$$
\begin{equation}
\displaystyle{\int_{a}^{b}\exp(h(x))}\mathrm{d}x.
\end{equation}
$$

We will take the Taylor series approximation of $h(x)$ at the point $x_{0}$, and have

$$
\begin{equation}
\displaystyle{\int_{a}^{b}}\exp(h(x))\mathrm{d}x \approx \displaystyle{\int_{a}^{b}\exp(h(x_{0}))+h^{\prime}(x_{0})(x-x_{0})+\frac{1}{2}h^{\prime\prime}(x_{0})(x-x_{0})}\mathrm{d}x.
\end{equation}
$$

Because the function $g(x)$ achieves its maximum at the point $x_{0}$ and $\log(\cdot)$ is a monotonic function, we have $h^{\prime}(x_{0})=0$. Therefore, Eq.(3) can be simplified as follows:

$$
\begin{equation}
\displaystyle{\int_{a}^{b}}\exp(h(x))\mathrm{d}x \approx \displaystyle{\int_{a}^{b}}\exp(h(x_{0}) + \frac{1}{2}h^{\prime\prime}(x_{0})(x-x_{0})^{2})\mathrm{d}x
\end{equation}
$$

It is found that the term $h(x_{0})$ does not depend on $x$, so that it can be pulled outside the integral. In addition, we can rearrange the term $\frac{1}{2}h^{\prime\prime}(x_{0})(x-x_{0})^{2}$, and have

$$
\begin{equation}
\displaystyle{\int_{a}^{b}}\exp(h(x))\mathrm{d}x \approx \exp(h(x_{0})) \displaystyle{\int_{a}^{b}}\exp\left(-\frac{1}{2}\frac{(x-x_{0})^{2}}{-h^{\prime\prime}(x_{0})^{-1}}}\right)\mathrm{d}x.
\end{equation}
$$

It is observed that the term $\displaystyle{\int_{a}^{b}}\exp\left(-\frac{1}{2}\frac{(x-x_{0})^{2}}{-h^{\prime\prime}(x_{0})^{-1}}\right)\mathrm{d}x$ is proportional to a Gaussian distribution with the mean $x_{0}$ and the variance $-h^{\prime\prime}(x_{0})^{-1}$.

Let $\Phi(x\vert \mu, \sigma^{2})$ be a cumulative function for a Gaussian distribution $\mathcal{N}(x\vert \mu, \sigma^{2})$ with the mean $\mu$ and the variance $\sigma^{2}$. We can re-write Eq.(5) as follows:

$$
\begin{align}
\displaystyle{\int_{a}^{b}\exp(h(x))}\mathrm{d}x &\approx \exp(h(x_{0})) \displaystyle{\int_{a}^{b}}\exp\left(-\frac{1}{2}\frac{(x-x_{0})^{2}}{-h^{\prime\prime}(x_{0})^{-1}}}\right)\mathrm{d}x \\
&=\exp(h(x_{0}))\sqrt{\frac{2\pi}{-h^{\prime\prime}(x_{0})}}\left[\Phi(b\vert x_{0}, -h^{\prime\prime}(x_{0})^{-1}) - \Phi(a\vert x_{0}, -h^{\prime\prime}(x_{0})^{-1})\right] 
\end{align}
$$

Given that $\exp(h(x_{0}))=g(x_{0})$. If $b=\infty, a=-\infty$, the term inside the bracket is equal to 1. By these condistions, the Laplacian approximation is equal to the value $g(x_{0})$ mutiplied by a constant term $\sqrt{\frac{2\pi}{-h^{\prime\prime}(x_{0})}}$, which depends on the curvature of $h(x)$. It means that the function $h(x)$ should be twice differentiable.

## 2. Application in Machine Learning.

In machine learning, we usually need to compute the mean of a posterior distribution from the observed data. Let $x$ denote the observed data. We use the notations $p(x\vert \theta)$ and $p(\theta)$ to represent the likelihood distribution and the prior distribution, respectively. The mean of the posterior distribution $p(\theta\vert x)$ is computed as follow:

$$
\begin{align}
\mathbb{E}[\theta] &= \displaystyle{\int}\theta p(\theta\vert x)\mathrm{d}\theta \\
&= \frac{\displaystyle{\int}\theta p(x\vert \theta)p(\theta)\mathrm{d}x}{\displaystyle{\int} p(x\vert \theta)p(\theta)\mathrm{d}\theta} \\
&= \frac{\displaystyle{\int}\theta \exp\left(\log(p(x\vert \theta)p(\theta))\right)\mathrm{d}\theta}{\displaystyle{\int} \exp\left(\log(p(x\vert \theta)p(\theta))\right)\mathrm{d}\theta}
\end{align}
$$

Let $h(\theta)=\log(p(x\vert \theta)p(\theta))$, and the function $\log(\cdot)$ is monotonic function. Therefore, the function $h(x)$ is proportional to the posterior density function. It means that the function $h(x)$ achieves its maximum value at the posterior mode. We assume $\hat{\theta}$ is the posterior mode. According to the description of Laplacian approximation, we have

$$
\begin{align}
\displaystyle{\int}\theta p(\theta\vert y)\mathrm{d}\theta &= \frac{\displaystyle{\int} \theta\exp(h(\hat{\theta})+\frac{1}{2}h^{\prime\prime}(\hat{\theta})(\theta-\hat{\theta})^{2})\mathrm{d}\theta}{\displaystyle{\int} \exp(h(\hat{\theta})+\frac{1}{2}h^{\prime\prime}(\hat{\theta})(\theta-\hat{\theta})^{2})\mathrm{d}\theta}, \\

&=\frac{\displaystyle{\int} \theta\exp(\frac{1}{2}h^{\prime\prime}(\hat{\theta})(\theta-\hat{\theta})^{2})\mathrm{d}\theta}{\displaystyle{\int} \exp(\frac{1}{2}h^{\prime\prime}(\hat{\theta})(\theta-\hat{\theta})^{2})\mathrm{d}\theta}, \\

&= \frac{\displaystyle{\int} \theta \sqrt{ \frac{2\pi}{-h^{\prime\prime}(\hat{\theta})}} \mathcal{N}(\theta)\vert \hat{\theta}, -h^{\prime\prime}(\hat{\theta})^{-1} \mathrm{d}\theta }{ \displaystyle{\int} \sqrt{ \frac{2\pi}{-h^{\prime\prime}(\hat{\theta})}} \mathcal{N}(\theta)\vert \hat{\theta}, -h^{\prime\prime}(\hat{\theta})^{-1} \mathrm{d}\theta }, \\

&= \hat{\theta}.
\end{align}
$$

It is found that the Laplacian approximation to the posterior mean is the posterior mode. **Laplacian approximation works well when the posterior distribution is unimodal and relatively symmetric around the model.** Intuitively, if the mass of the posterior distribution mainly concentrates around $\hat{\theta}$, the approximation will be better.






