---
layout: post
title: Note(1)--Basic Review of Basic Probability 
date: 2018-01-30
author: Gwan Siu
catalog: True
tags:
    - Statisctics and Bayesian Analysis
---

>The reference materials are based on cmu 10-705,[2016](http://www.stat.cmu.edu/~larry/=stat705/) and [2017](http://www.stat.cmu.edu/~siva/705/main.html).

>In note(1), we just review some necessary basic probability knowledge.

#### 1. Axioms of Probability

- $\mathbb{P}(A)\geq 0$
- $\mathbb{P}(\Omega)=0$
- If $A$ and $B$ are disjoint, then $\mathbb{P}(A\cup B)=\mathbb(P)(A)+\mathbb{P}(B)$

#### 2. Random Variable.

##### 2.1 The Definition of Random Variable
Let $\Omega$ be a smaple space(a set of posible events) with a probability distribution(also called a probability measure $\mathbb{P}$). A `random variable` is a mapping function: $X:\Omega \rightarrow \mathbb{R}$. We wirte:

$$
\begin{algn}
\mathbb{P}(X\in A)=\mathbb{P}({\omega \in \Omega: X(\omega)\in A})
\edn{align}
$$

and we can write $X\sim P$ that means $X$ has a distribution $P$.

##### 2.5 Cumulative distribution
The cumulative distribution function($cdf$) of $X$ is

$$
\begin{align}
    F_{X}(x)=F(x)=\mathbb{P}(X\leq x)
\end{align}
$$
The property of cdf:
- 1. $F$ is **right-continuous function**. At each point $x$, we have $F(x)=\lim_{n\rightarrow}\infty F(y_{n})=F(x)$ for any sequence $y_{n}\rightarrow x$ with $y_{n} >x$.
- 2. $F$ is **non-decreasing**. If $x<y$ then $F(x)\leq F(y)$.
- 3. $F$ is **normalized**. $\lim_{n\rightarrow -\infty}F(x)=0$ and $\lim_{x\rightarrow \infty}F(x)=1$.

Conversely, any $F$ satisfying these three properties is a cdf for some random variable.

If $X$ is discrete, its `probability mass function`($pmf$) is:

$$
\begin{align}
p_{X}(x)=p(x)=\mathbb{P}(X=x)
\end{align}
$$

If $X$ is continuous, then its `probability density function`($pdf$) satisfies:

$$
\begin{align}
\mathbb{P}(X\in A)=\int_{A}p_{X}(x)dx=\int_{A}p(x)dx
\end{align}
$$
and the $p_{X}(x)=p(x)=F^{'}(x)$. The following are all equivalent:
$$
X\sim P,X\sim F, X\sim p
$$

$$
\fbox{Suppose that $X\sim P$}
$$



