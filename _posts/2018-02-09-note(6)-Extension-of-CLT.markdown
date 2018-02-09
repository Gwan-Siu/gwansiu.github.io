---
layout: post
title: Note(6)--Extension of CLT
date: 2018-02-09
author: Gwan Siu
catalog: True
tags:
    - Statisctics and Bayesian Analysis
---

> In this blog, we just talk about some extension of CLT.
> 
> 1. Lyapunov CLT: just requires only independent but not identically idstributed.
> 2. Multivariate CLT: Original CLT is applied to one dimension, what happens to high dimension space.
> 3. CLT with estimated variance: when we use CLT, we should know about the variance of random variables. Could we replace estimated variance with true variance when we use CLT?
> 4. How about the Convergenvce rate in CLT? Berry Esseen
> 5. The delta method, we use a continuous function to construct another limiting distribution.

#### 1. Berry-Esseen Theorem.

The central limit theorem is from the asymptotic view. It just states that the average of i.i.d random variables converges in distribution to standard normal distribution as $n\rightarrow \infty$, but it does not answer the question that how close to this standard normal distribution. The `berry-esseen` theorem answer this question.

**Berry-Esseen:** Suppose that $X_{1},...,X_{n}\sim P$. Let $\mu=\mathbb{E}[X_{1}],\sigma^{2}=\mathbb{E}[\arrowvert X_{1}-\mu\arrowvert^{2}]$, and $\mu_{3}=\mathbb{E}[\arrowvert X_{1}-\mu\arrowvert^{3}]$. Let

$$
\begin{equation}
F_{n}(x)=\mathbb{P}(\frac{\sqrt{n}(\widetilde{\mu}-\mu)}{\sigma}\leq x),
\end{equation}
$$

denote the CDF of the normalized sample average. If $\mu_{3}<\infty$ then,

$$
\begin{equation}
\sup_{x}\arrowvert F_{n}(x)-\Phi(x)\arrowvert\leq \frac{9\mu_{3}}{\sigma^{3}\sqrt{n}}
\end{equation}
$$

This bound is roughly saying that if $\mu/sigma^{3}$ is small then the convergence to normality in distribution happens quite fast.

#### 2.Multivariate CLT-CLT in high dimensional space

**Multivariate CLT:** If $X_{1},...,X_{n}$ are i.i.d with mean $\mu\in \mathbb{R}^{d}$, and covariance matrix $\Sigma\in \mathbb{R}^{d\times d}$(with finite entries) then,

$$
\begin{equation}
\sqrt{n}(\widetilde{\mu}-\mu)\overset{d}{\rightarrow} N(0,\Sigma)
\end{equation}
$$

**Notes:**

1. You might wonder what convergence in distribution means for random vectors. A random vector still has a CDF, typically we define this case:

$$
\begin{equation}
F_{X}(x_{1},...,x_{d})=\mathbb{P}(X_{1}\leq x_{1},...,X_{d}\leq x_{d})
\end{equation}
$$

so we can still define convergence in distribution via pointwise convergence of the CDF. In order to define points of continuity it turns out that the correct definition is that a point of continuity of the CDF if the boundary of the rectangle whose upper right corner is (x_{1},...,x_{d}) has probability 0.

2. Although $d$ can be larger than 1, it is taken to be fixed as $n\rightarrow \infty$. Central limit theorems, when $d$ is allowed to grow, i.e. high-dimensional CLTs are rare and are an active topic of research.

3. The proof of this result follows directly from the proof of the univariate CLT and a powerful result in asymptotic known as the [Cramer-World device](https://en.wikipedia.org/wiki/Cram%C3%A9r%E2%80%93Wold_theorem). The [Cramer-World device](https://en.wikipedia.org/wiki/Cram%C3%A9r%E2%80%93Wold_theorem) roughly asserts that if $a^{T}X_{n}\overset{d}{\rightarrow}a^{T}X$ for all vectors $a\in \mathbb{R}^{d}$ then $X_{n}^{d}\overset{d}{\rightarrow}X$.

#### 3. CLT with estimated variance

In the typical case of the CLT, we need to know the variance $\sigma$. In practics, we can use estimated variance to replace with variance. The estimated variance is defined as:

$$
\begin{equation}
\widetilde{\sigma}_{n}^{2} = \frac{1}{n-1}\sum_{i=1}^{n}(X_{i}-\widetilde{\mu})^{2}
\end{equation}
$$

It turns out that we can replace the standard deviation in the CLT by $\widetilde{\sigma}$ and still have the same convergence in distribution, i.e.

$$
\begin{equation}
\frac{\sqrt{n}(\widetilde{\mu}-\mu)}{\widetilde{\sigma}_{n}}\overset{d}{\rightarrow}N(0,1)
\end{equation}
$$

This prood can be derived from a sequence of applications of Slutsky's theorem and the continuous mapping theorem.

**Proof:**First observe that if we can show that $\frac{\sigma}{\widetilde{\sigma}_{n}}\overset{d}{\rightarrow}1$, then an application of Slutsky's theorem and the CLT gives us the desired result.

Since square-root is a continuous map, by the continuous mapping theorem, it suffices to show that $\frac{\sigma^{2}}{\widetilde{\sigma}^{2}_{n}}\overset{d}{\rightarrow}1$. We will instead show the stronger statement that,

$$
\begin{equation}
\widetilde{\sigma}^{2}_{n}\overset{p}{\rightarrow}\sigma^{2}
\end{equation}
$$

which implies that the desired statement via the continuous mapping theorem. Note that:

$$
\begin{align}
\widetilde{\sigma}^{2}_{n} &= \frac{1}{n-1}\sum_{i=1}^{n}(X_{i}-\widetilde{\mu})^{2}\\
&\overset{p}{\rightarrow}\frac{1}{n}\sum_{i=1}^{n}(X_{i}-\widetilde{\mu})^{2}
\end{align}
$$

using the fact that $\frac{n-1}{n}\rightarrow 1$. Now,

$$
\begin{equation}
\frac{1}{n}\sum_{i=1}^{n}(X_{i}-\widetilde{\mu})^{2}=\frac{1}{n}\sum_{i=1}^{n}X_{i}^{2}-(\frac{1}{n}\sum_{i=1}^{n}X_{i}^{2})^{2}\overset{p}{\rightarrow}\mathbb{E}[X^{2}]-(\mathbb{E}[X])^{2}
\end{equation}
$$

using the WLLN. This concludes the proof.

#### 4. The Delta Method

Delta Method states that suppose we have a sequence of random variables $X_{n}$ that converges in distribution to a Gaussian distribution then can we characterize the limiting distribution of $g(X_{n})$ where $g$ is a smooth function. (Continuous mapping theorem).

**Delta Method:** Suppose that,

$$
\begin{equation}
\frac{\sqrt{n}(X_{n}-\mu)}{\sigma}\overset{d}{\rightarrow}N(0,1)
\end{equation}
$$

and that $g$ is a continuously differentiable function such that $g^{`}(\mu)\neq 0$. Then,

$$
\begin{equation}
\frac{\sqrt{n}(g(X_{n})-g(\mu))}{\sigma}\overset{d}{\rightarrow}N(0,[g^{`}(\mu)]^{2})
\end{equation}
$$

**Proof:** The basic idea is simply to use Taylor's approximation. We know that

$$
\begin{eqaution}
g(X_{n})\approx g(\mu) + g^{`}(\mu)(X_{n}-\mu)
\end{equation}
$$

so that,

$$
\begin{equation}
\frac{\sqrt{n}(g(X_{n})-g(\mu))}{\mu}\approx g^{`}(\mu)\frac{\sqrt{n}(X_{n}-\mu)}{\sigma}\overset{d}{\rightarrow}N(0,[g^{`}(\mu)]^{2})
\end{equation}
$$

Alternatively, here is a more formal proof. By a rigorous application of Taylor's theorem we obtain

$$
\begin{equation}
\frac{\sqrt{n}(g(X_{n})-g(\mu)}{\sigma}\approx g^{`}(\mu)\frac{\sqrt{n}(X_{n}-\mu)}{\sigma}\overset{d}{\rightarrow}N(0,[g^{`}(\mu)]^{2})
\end{equation}
$$

where $\widetilde{\mu}$ is on the line joining $\mu$ to $\widetilde{\mu}$. We know by the WLLN that $\widetilde{\mu}\overset{p}{\rightarrow}\mu$ and so $\widetilde{\mu}\overset{p}{\rightarrow}\mu$ Since $g$ is continuously differentiable， we can use the continuous mapping theorem to conclude that,

$$
\begin{equation}
g^{`}(\widetilde{\mu})\overset{p}{\rightarrow}g^{`}(\mu)
\end{equation}
$$

Now, we apply Slutsky's theorem to obtain that,

$$
\begin{equation}
g^{`}(\widetilde{\mu})\frac{\sqrt{n}(X_{n}-\mu)}{\sigma}\overset{d}{\rightarrow} g^{`}(\mu)N(0,1)\overset{d}{=}N(0,[g^{`}(\mu)]^{2})
\end{equation}
$$

#### 5. Multivariate Delta Method: How about delta method in high dimensional space?

Suppose random vector $X_{1},...,X_{n}\in \mathbb{R}^{d}$, and $g:\mathbb{R}^{d}\mapsto \mathbb{R}$ is a continuously differentiable function, then

$$
\begin{equation}
\sqrt{n}(g(\widetilde{\mu}_{n})-g(\mu))\overset{d}{\rightarrrow}N(0,\bigtriangleup_{\mu}(g)^{T}\Sigma\bigtriangleup_{\mu}(g))
\end{equation}
$$

where

$$
\begin{equation}
\bigtriangleup_{g}(\mu)=\pmatrix{\frac{\partial g(x)}{\partial x_{1}}\cr ...\cr \frac{\partial g(x)}{\partial x_{d}}}_{x=\mu}
\end{equation}
$$

is the gradient of $g$ evaluated at $\mu$.




