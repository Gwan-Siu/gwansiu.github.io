---
layout: post
title: Note(5): Convergence
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

Let $X_{1},...,X_{n}$ be a sequence of independent random variables with mean $\mu$ and variance $\sigma^{2}$. Assume that the mgf $\mathbbP{E}[\text{exp}(tX_{i})]$ is finite for $t$ in a neighborhood around zero. Let

$$
\begin{equation}
S_{n}=\frac{\sqrt{n}(\widetilde{\mu}-\mu)}{\sigma}
\end{equation}
$$ 

then $S_{n}$ converges in distribution to $Z\sim N(0,1)$.

**Notice:**

1. The central limit theorem is incredibly genreal. It does not matter what the distribution of $X_{i}$ is, the average $S_{n}$ converegs in distribution to a Gaussian(under fairly mild assumptions).

2. The most general version of the CLT does not require any assumption about mgf. It just requires that the mean and variance are finite. 
