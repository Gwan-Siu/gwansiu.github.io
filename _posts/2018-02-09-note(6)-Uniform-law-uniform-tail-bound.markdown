---
layout: post
title: Note(6)--Uniform Laws or Uniform Tail Bounds
date: 2018-02-10
author: Gwan Siu
catalog: True
tags:
    - Statistics and Bayesian Analysis
---

> Previously, we discussed LLNs and tail bounds that apply to a collection of random variables taken together. In this blog, we mainly focus on uniform laws or uniform tail bounds.

#### 1. Uniform convergence of the CDF

*How can one estimate the CDF of an univariate random variable given a random sample?*

Suppose we have $X_{1},...,X_{n}\sim F_{X}$, so a little bit of thought might suggest a natural estimator is the `empirical CDF`, i.e.

$$
\begin{equation}
\widetilde{F}_{n}(x)=\frac{1}{n}\sum_{i=1}^{n}\mathbb{I}(X_{i}\leq x)
\end{equation}
$$

where $\mathbb{I}$ is an indicator function. Unlike in a classical statistics problem we are not estimating a simple parameter, rather we are estimating an entire function.

Suppose we fixed a value $x$ and we decide to try to estimate $F_{X}(x)$. We could use the empirical CDF at $x$, but this time it is a rather simple prpbelm. Thus, we have:

$$
\begin{equation}
\mathbb{E}[\widetilde{F}_{n}(x)]=\frac{1}{n}\sum_{i=1}^{n}\mathbb{E}[\mathbb{I}(X_{i}\leq x)]=\mathbb{P}(X\leq x)=F_{X}(x)
\end{equation}
$$

The indicator are bounded random random variables so we could just use Hoffding's bound to conclude that

$$
\begin{equation}
\mathbb{P}(\arrowvert \widetilde{F}_{n}(x)-F(x)\arrowvert\geq \epsilon)\leq 2\text{exp}(-2n\epsilon^{2})
\end{equation}
$$

This just show that for a single point $x$, we can use simple tail bounds to say that the empirical CDF is close to the true CDF. A more difficult question is to ask whether the empirical CDF and true CDF are close everywhere. In other words, we would like to know the behaviour of the worst case:

$$
\begin{equation}
\bigtriangleup=\sup_{x\in \mathbb{R}}\arrowvert \widetilde{F}_{n}(x)-F_{X}(x)\arrowvert
\end{equation}
$$

Reasoning about the $\bigtriangleup$ requires us to reason about the CDF everywhere, hence the name *uniform bounds* or *uniform LLNs*.

**Notes:**

1. The Glivenko-Cantelli theorem is like a WLLN but it is a uniform WLLN that ensures essentially that the WLLN is true at every point $x\in \mathbb{R}$.

2. There is a corresponidng strong GC theorem that guarantee convergence almost surely.

**Actually, we can estimate CDF of a random variable with no assumptions. This is constrast to estimating the density of a random variable which typically requires strong smoothness assumptions.**

#### 2. Equivalent forms, generalizations and empirical process theory

The empirical probability of a set $A$ is often denoted as:

$$
\begin{equation}
\mathbb{P}_{n}(A)=\frac{1}{n}\sum_{i=1}^{n}\mathbb{I}(X_{i}\in A)
\end{equation}
$$

The quantity $\bigtriangleup$ above can be equivalently written as,

$$
\begin{equation}
\bigtriangleup = \sup_{A\in\mathcal{A}}\arrowvert \mathbb{P}_{n}(A)-\mathbb{A}\arrowvert
\end{equation}
$$

where $\mathcal{A}$ is a collection of sets,

$$
\begin{equation}
\mathcal{A}={A(x):A(x)=(-\infty,x])}
\end{equation}
$$

since in this case, $\mathbb{P}(A(x))=F_{X}(x)$. 

To generalize the CDF question to more generally about other interesting collections of sets $\mathcal{A}$, i.e. we are interested in collections of sets $\mathcal{A}$, for which we have uniform convergence, i.e:

$$
\begin{equation}
\bigtriangleup(\mathcal{A})=\sup_{A\in\mathcal{A}}\arrowvert \mathbb{P}_{n}(A)-\mathbb{P}(A)\arrowvert
\end{equation}
$$

converges in probability to 0. This is well known as *Vapnik-Cervonenkis* theory.

Even more generally, one can replace the indicators with general functions, i.e. let $\mathcal{F}$ be a class of integrable, real-valued functions, and suppose we have an i.i.d sample $X_{1},...,X_{n}\sim P$, then we could be intested in,

$$
\begin{equation}
\bigtriangleup(\mathcal{F})=\sup_{f\in\mathcal{F}}\arrowvert \frac{1}{n}\sum_{i=1}^{n}f(X_{i})-\mathbb{E}[f]\arrowvert
\end{equation}
$$

This quantity is known as an *empirical process* and empirical process theory is the area of statistics that asks questions about the convergence in probability, almost surely or in distribution for the quantity $\bigtriangleup(\mathcal{F})$ for interesting classes of functions of $\mathcal{F}$.

We refer to classes for which $\bigtriangleup(F)\overset{p}{\rightarrow}0,$ as *Glivenko-Cantelli* classes. The class of functions:

$$
\begin{equation}
\mathcal{F}={\mathbb{I}(-\infty,x]\arrowvert x\in \mathbb{R}}
\end{equation}
$$

which defines the uniform convergence of the CDF is an example of a *Glivenko-Cantelli* class.

#### 3. Failure of an uniform law

In generally, very complex classes of functions or sets will fail to be Glivenko-Cantelli. Thus, we need some methods to measure the complexity of functions or sets. A failure case is shown here.

Suppose we draw $X_{1},...,X_{n}\sim P$ where $P$ is some continuous distribution over $[0,1]$. Suppose further that $\mathcal{A}$ is all subsets of $[0,1]$ with finitely many elements.

Then observe that since the distribution is continuous we have that, $\mathbb{P}(A)=0$ for each $A\in \mathcal{A}$, however for the finite set${X_{1},...,X_{n}}$ we have that $\mathbb{P}_{n}(A)=1$, i.e.

$$
\begin{equation}
\bigtriangleup(\mathcal{A})=\sup_{A\in\mathcal{A}}\arrowvert \mathbb{P}_{n}(A)-\mathbb{P}(A)\arrowvert =1
\end{equation}
$$

no matter what how large $n$ is. So the collection of sets $\mathcal{A}$ is not Glivenko-Cantelli. Roughly, the collection of sets is "too large".

#### 4. Estimation of Statistical Functionals

Often we want to estimate some quantity which can be written as a simple functional of the CDF, and a natural estimate just replaces the true CDF with the empirical CDF(such estimators are known as plug-in esimators). *Functional is a function of a function*. Here are some classical examples:

1. **Expectation Functionals:** For a given function $g$, we can view the usual empirical estimator of its Expectation as a plug-in estimate where we replace the population CDF by the empirical CDF,

$$
\widetilde{\mathbb{E}}[g(X)]=\frac{1}{n}\sum_{i=1}^{n}g(X_{i})=\int_{x}g(x)f\widetilde{F}_{n}(x)
$$

2. **Quantile Functionals:** For an $\alpha\in[0,1]$, the $\alpha-th$ quantile of a distribution is given as:

$$
\begin{equation}
Q_{\alpha}(F)=\inf{\{t\in\mathbb{R}\arrowvert F(t)\geq \alpha}\}
\end{equation}
$$

Taking $\alpha=0.5$ gives the median. A natural plug-in estimator of $Q_{\alpha}(F)$ is to simply take $Q_{\alpha}(\widetilde{F}_{n})$.

3. **Goodness-of-fit Functionals:** 

In data analysis, we want to test the hypothesis that we have are $i.i.d$ from some known distribution $F_{0}$. The rough idea is we form a statistic to test this hypothesis which(hopefully) takes large values when the distribution is not $F_{0}$ and takes small values otherwise. Typical tests of this form include the Kolmogorov-Smirnov test, where we compute the plug-in quantity:

$$
\begin{equation}
\widetilde{T}_{KS}=\sup_{x\in\mathbb{R}}\arrowvert \widetilde{F}_{n}(x)-F_{0}\arrowvert
\end{equation}
$$

which is natural because if the true distribution is $F_{0}$ we know by the Glivenko-Cantelli theorem that $T_{KS}$ is small. Similarly, one can use the Cramer-von Mises test which uses the plug-in statistic,

$$
\begin{equation}
\widetilde{T}_{CVM}=\int_{x}(\widetilde{F}_(x)-F_{0}(x))^{2}dF_{0}(x)
\end{equation}
$$

There are many other Statistical functionals for which the usual estimators can be thought of as plug-in estimators. For example: variance, correlation, and higher moments can all be expressed in this function.

In each of the above cases we are interested in estimating some functional $\gamma(F)$ and we use the plug-in estimator $\gamma(\widetilde{F}_{n})$. Analogous to the continuous mapping theorem, there is a Glivenko-Cantelli theorem that provides a WLLN for these estimators. We need to first defines a notion of continuity. Suppose $\gamma$ satisfies the property that for every $\epsilon >0$, there is a $\delta >0$ such that if

$$
\begin{equation}
\sup_{x}\arrowvert \widetilde{F}_{n}-F(x)\arrowvert \leq \delta
\end{equation}
$$

then

$$
\begin{equation}
\arrowvert \gamma(F)-\gamma(\widetilde{F}_{n})\arrowvert \leq \epsilon
\end{equation}
$$

for such functionals $\gamma$, it is a simple consequence of the Glivenko-Cantelli theorem that $\gamma(\widetilde{F}_{n})$ converges in probability to $\gamma(F)$.