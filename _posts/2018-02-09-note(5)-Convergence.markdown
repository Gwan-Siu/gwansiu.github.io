---
layout: post
title: Note(5)-Convergence
date: 2018-02-09
author: Gwan Siu
catalog: True
tags:
	- Statisctics and Bayesian Analysis
---

>  In this blog, we will talk about Slutsky's theorem and center limit theorem.

#### 1. Continuous Mapping Theorem

If a sequence $X_{1},...,X_{n}$ converges in probability to $X$ then for any continuous function $f$, $f(X_{1}),...,f(X_{n})$ converges in probability to $f(X)$. The same is true for convergence in distribution.

**This theorem is very useful. For example, if we have a consistent estimator for some parameters, then we can construct another consistent estimators for some function of the parameters by this theorem.**

#### 2. Slutsky's Theorem

From continuous mapping theorem, we know that if $X_{n}$ converges in probability to $X$ and $Y_{n}$ converges in probability to $Y$ then $X_{n} + Y_{n}$ converges in probability to $X+Y$. The same is true of product, i.e. $X_{n}Y_{n}$ converges in probability to $XY$.

However, this is not true for convergence in distribution, i.e. if $X_{n}$ converges in distribution to $X$ and $Y_{n}$ converges in distribution to $Y$ then $X_{n} + Y_{n}$ does not necessarily converge in distribution to $X+Y$.

The one exception to this is known as `Slutsky's theoem`. It says that if $Y_{n}$ converges in distribution to a constant $c$, and $X$ converges in distribution to $X$: then $X_{n}+Y_{n}$ converges in distribution to $X+c$ and $X_{n}Y_{n}$ converges in distribution to $cX$.

#### 3. The Central Limit Theorem

**The central limit theorem is one of the most famous examples of convergence in distribution.**

Let $X_{1},...,X_{n}$ be a sequence of independent random variables with mean $\mu$ and variance $\sigma^{2}$. Assume that the mgf $\mathbb{E}[\text{exp}(tX_{i})]$ is finite for $t$ in a neighborhood around zero. Let

$$
\begin{equation}
S_{n}=\frac{\sqrt{n}(\widetilde{\mu}-\mu)}{\sigma}
\end{equation}
$$ 

then $S_{n}$ converges in distribution to $Z\sim N(0,1)$.

**Notice:**

1. The central limit theorem is incredibly genreal. It does not matter what the distribution of $X_{i}$ is, the average $S_{n}$ converegs in distribution to a Gaussian(under fairly mild assumptions).

2. The most general version of the CLT does not require any assumption about mgf. It just requires that the mean and variance are finite. 

#### 4. Lyapunov CLT: Only independence but not identically distributed

The typical case of CLT requires `i.i.d` condition, but we need conditions to ensure that one or a small number of random variables do not dominate the sum. If `i.i.d` condition is eliminated, then we should need another condition to ensure CLT is satisfied, that condition is Lyapunov condition.

Define the variance of the average:

$$
\begin{equation}
s_{n}^{2}=\sum_{i=1}^{n}\sigma_{i}^{2}
\end{equation}
$$

**Lyapunov CLT:** Suppose $X_{1},...,X_{n}$ are independent but not necessarily identically distributed. Let $\mu_{i}=\mathbb{E}[X_{i}]$ and let $\sigma_{i}=\text{Var}(X_{i})$. Then if we satify the Lyapunov condition:

$$
\begin{equation}
\lim_{n\rightarrow\infty}\frac{1}{s^{3}_{n}}\sum_{i=1}^{n}\mathbb{E}[\arrowvert X_{i}-\mu\arrowvert^{3}]=0
\end{equation}
$$

then

$$
\begin{equation}
\frac{1}{s_{n}}\sum_{i=1}^{n}\arrowvert X_{i}-\mu_{i}\arrowvert \overset{d}{\rightarrow}N(0,1)
\end{equation}
$$

Consider the case that Lyapunov's condition is violated. In particular, consider the extreme case, when all the random variables are deterministic, except X_{1} which has mean $\mu_{1}$ and variance $\sigma_{1}^{2}>0$.Then $s_{n}^{3}=\sigma_{1}^{3}$ and the third absolute moment $\mathbb{E}[\arrowvert X_{1}-\mu\arrowvert^{3}]>0$ so that the Lyapunov condition. Roughly, what can happen in the non-idnetically distributed case is that only one random variable can dominate the sum in which case you are not really averaging many things so you do not have CLT.

Consider another typical cse of Lyapunov CLT. Supppose we have absolute moments are bounded by some constant $C>0$ say that and the variance of any paricular random variable is not too small. In this case,

$$
\begin{equation}
s_{n}^{2}=\sum_{i=1}^{n}\sigma^{2}_{i}\geq n\sigma_{\text{min}}^{2}
\end{equation}
$$

and 

$$
\begin{equation}
\sum_{i=1}^{n}\mathbb{E}[\arrowvert X_{i}-\mu\arrowvert^{3}]\leq Cn
\end{equation}
$$

In this case, we will have that the Lyapunov ratio $\leq \frac{C}{\sqrt{n}\sigma^{3}_{\text{min}}}$.