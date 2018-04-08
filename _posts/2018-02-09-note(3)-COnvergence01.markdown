---
layout: post
title: Note(3)--Convergence 01 
date: 2018-04-08
author: Gwan Siu
catalog: True
tags:
    - Statistics and Bayesian Analysis
---

> In this blog, the convergence of random variable is discussed. At a high level, the last several lectures focus on non-asymptotic properties of averge of i.i.d random variabls. For symptotic property, we just want to ask the question: what happens to the average of $n$ i.i.d random variables as $n\rightarrow \infty$.

#### 1. Almost Sure Convergence

The **definition** of `almost sure convergence` is:

$$
\begin{equation}
\mathbb{P}(\lim_{n\rightarrow \infty} X_{n}=X)=1
\end{equation}
$$

This formulation can be explained that the sample space inside the probability statement here grows with $n$ and it requires some machinery to be precise here.

Alternatively, we can define `almost sure convergence` through sequence of random variables. We suppose that $X_{n}$ convergesd almost surely to $X$ if we let $\Omega$ be a set of probability mass 1, i.e. $\mathbb{P}(\OMega)=1$, and for every $\omega \subseteq \Omega$, and for every $\epsilon >0$, we have that there exist a natural number $N$, for all $n\geq N$, we have:

$$
\begin{equation}
\arrowvert X_{n}(\omega)-X(\omega)\arrowvert \leq \epsilon
\end{equation}
$$

Roughly, the way to think about this type of convergence is to imagine that there is some set of exceptional events on which the random variables can disagree, but the exceptional events has the probability 0 as $n \rightarrow \infty$. Barring, these exceptional events the sequence converges just like sequences of real numbers do. The exceptional events is where the "almost" is almost sure arises.

#### 2. Convergence in Probability

**Definition:** A sequence of random variable $X_{1},...,X_{n}$ converges in probability in probability to a random varibale $X$ if for every $\epsilon >0$, we have that:

$$
\begin{equation}
\lim_{n\rightarrow \infty} \mathbb{P}(\arrowvert X_{n}-X\arrowvert \geq \epsilon)
\end{equation}
$$

The intuition of convergence in probability is that we can consider $X$ is deterministic, i.e. $X=c$ with prbability 1. COnvergence in probability is saying that as $n$ gets large the distribution of $X_{n}$ gets more peaked around the value $c$.

`COnvergence in probabililty can be viewed as a statement about the convergences of probability, while almost sure convergence is a convergence of the values of a sequence of random variables.`

**Convergence in probability does not imply almost sure convergence:**  Suppose we have a sample space $S=[0,1]$, with the uniform distribution, we draw  $s\sim U[0,1]$ and define $X(s)=s$.

We define the sequence as:

$$
\begin{align}
X_{1}(s) = s +\mathbb{I}_{[0,1]}(s), \text{ } X_{2}(s) = s +\mathbb{I}_{[0,1/2]}(s), \text{ } X_{3}= s + \mathbb{I}_{[1/2,1]}(s) \\
X_{4}(s) = s +\mathbb{I}_{[0,1/3]}(s), \text{ } X_{5}(s) = s +\mathbb{I}_{[1/3,2/3]}(s), \text{ } X_{6}= s + \mathbb{I}_{[2/3,1]}(s) 
\end{align}
$$
This sequence converges in probability but not almost surely. Roughly, the "1+s" spike becomes less frequent down the sequence(allow convergence in probability) but the limit is not well defined. For any $s$, $X_{n}(s)$ alternates between $1$ and $1+s$.


#### 3. Weak Law of Large Numbers

**Definition:** Suppose that $Y_{1},...,Y_{n}$ are i.i.d with $\amthbb{E}[Y_{i}]=\mu$ and $\text{Var}(Y_{i})=\sigma^{2}<\infty$. Define, for $i\in {1,...,n}$,

$$
\begin{equation}
X_{i}=\frac{1}{i}\sum_{j=1}^{i}Y_{j}
\end{equationi}
$$

The WLLN says that the sequence $X_{1},X_{2},...$ converges in probability to $\mu$.

**Proof:** The proof is simply an application of Chebyshev's inequality. We note that by Chebyshev's inequality:

$$
\begin{equation}
\mathbb{P}(\arrowvert X_{n}-\mathbb{E}[X]\arrowvert \geq \epsilon)\leq \frac{\sigma^{2}}{n\epsilon^{2}}
\end{euqation}
$$

This in turn implies that

$$
\begin{equation}
\lim_{n\rightarrow \infty}\mathbb{E}(\arrowvert X_{n}-\mathbb{E}[X]\arrowvert \geq \epsilon)=0
\end{equation}
$$

as desired.

**Notes:**

1. Strictly speaking that WLLN is true even without the assumption of finite variance, as long as the first absolute moment is finite. This proof is a bit more difficult.

2. There is a statement that says that under similar assumptions the average converges almost surely to the expectation. This is known as the strong law of large numbers. (a little bit hard to prove)

**Consistency and convergence in probability:** we usually construct an estimator $\widetilde{\theta}_{n}$ for some quantity $\theta^{*}$. The estimator is consitent if the sequence of RVs $\wildetilde{\theta}_{n}$ converges in probability to $\theta^{*}$.**

Actually, WLLN/Chebyshev can already be used to prove some rudimentary consistency guarantees. For instance, if we consider the sample variance:

$$
\begin{equation}
\widetilde{S}_{n} = \frac{1}{n-1}\sum_{i=1}^{n}(X_{i}-\widetilde{\mu}_{n})^{2}
\end{equation}
$$

then by Chebyshev inequality, we can obtain:

$$
\begin{equation}
\mathbb{P}(\arrowvert \widetilde{S}_{n}-\sigma^{2}\arrowvert \geq \epsilon)\leq \frac{\text{Var}(S_{n})}{\epsilon^{2}}
\end{equation}
$$

so a sufficient condition for consistency is that $\text{Var}(S_{n})\rightarrow 0$ as $n\rightarrow \infty$.

#### 5. COnvergence in quadratic mean

**Definition:** A sequence converges to $X$ in quadratic mean is that:

$$
\begin{equation}
\mathbb{E}(X_{n}-X)^{2}\rightarrow 0
\end{equation}
$$
 
as $n\rightarrow \infty$.

#### 6. Convergence in distribution

**Definition:** A sequence converges to $X$ in distribution if:

$$
\begin{equation}
\lim_{n\rightarrow \infty} F_{X_{n}}(t) = F_{X}(t)
\end{equation}
$$

for all points $t$ where the CDF $F_{X}$ is continuous. COnvergence in distribution is the weakest form of convergence.

**Note:** a.s $\rightarrow$ prob $rightarrow$ distribution; q.m $\rightarrow$ prob.