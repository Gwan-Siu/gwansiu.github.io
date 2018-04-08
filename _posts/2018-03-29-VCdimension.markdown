---
layout: post
title: Note(7)--Uniform Laws, Empirical Process Theory and VC dimension
date: 2018-04-08
author: Gwan Siu
catalog: True
tags:
    - Statistics and Bayesian Analysis
---

## 1. Uniform Convergence and Empirical Risk Minimization

Empirical risk minimization is core topic in machine learning. Let's take a binary classification as example.

Given a training set $\{(X_{1},y_{1}),...,(X_{n},y_{n})\}$ that we assume are drwawn i.i.d from the distribution $P$. Each $X_{i}\in\mathcal{R}^{d}, y_{i}\in {-1, +1}$. Our goal is to find a classifier $f:\mathcal{R}^{d}\rightarrow{-1, +1}$ to minmize the error on future unseen sample: $\min\,\mathcal{P}(f(X)\neq y)$.

For a given classfier $f$ we can estimate its mis-classification error(training error in machine learning) as:

$$
\begin{equation}
\tilde{R}_{n}(f)=\frac{1}{n}\sum_{i=1}^{n}\mathcal{I}(f(X_{i}\neq y_{i})
\end{equation}
$$

If $f$ is some fixed classifier we know by Hoeffding's bound that:

$$
\begin{equation}
\mathcal{P}{\vert \tilde{R}_{n}(f)-\mathcal{P}(f(X)\neq y)\vert \leq t}\geq 2\,\text{exp}(-2nt^{2})
\end{equation}
$$

Intuitively, we should a select a better classfier from soem set of classifier $\mathcal{F}$, then a natural way is to find the best classifier on trainging set:

$$
\begin{equation}
\tilde{f}=\arg\min_{f\in \mathcal{F}}\tilde{R}_{n}(f)
\end{equation}
$$

This procedure is known as *empirical risk minimization*. This terminology is to answer a question: **How do we argue that in some cases this procedure will indeed select a good classifier?** This question is highly related to uniform convergence.

Let $f^{\ast}$ be the classifier in $F$. We would like to bound the excess risk of the classifier we choose, i.e.:

$$
\begin{equation}
\Delta =\mathcal{P}(\tilde{f}(X)\neq y)-\mathcal{P}(f^{\ast}(X)\neq y)
\end{equation}
$$ 

The formulation means that we want the error casued by chosen $\tilde{f}$ is close to the the error caused by the true $f$.

The typical way to deal with this formulation is to consider the decomposition:

$$
\begin{equation}
\Delta=\underbrace{\mathca{P}(\tilde{f}(X)\neq y)-\tilde{R}_{n}(\tilde{f})}_{T_{1}}+\underbrace{\tilde{R}_{n}(\tilde{f})-\tilde{R}_{n}(f^{\ast})}_{T_{2}}+\underbrace{\tilde{R}_{n}(f^{\ast})-\mathcal{P}(f^{\ast}\neq y)}_{T_{3}}
\end{equation}
$$

The first term dependent on the data because of $\tilde{f}$ and cannot be suitable for Hoeffding's inequality, so its empirical risk is not the sum of independent variables, the second term is $\leq 0$ and the third term is small just by the Hoeffding inequality. 

Instead we have to rely on the uniform convergence bound, suppose we can show that with probability at least $1-\delta$,

$$
\begin{equation}
\sum_{f\in\mathcal{F}}\[\tilde{R}_{n}(f)-\tilde{R}_{n}(f^{\ast})\]\leq \Theta
\end{equation}
$$

then we can conclude that excess risk with probability at least $1-\delta$ satisfies:

$$
\begin{equation}
\Delta=\mathcal{P}(\tilde{f}(X)\neq y)-\mathcal{P}(f^{\ast}(X)\neq y)\leq 2\Theta
\end{eqution}
$$

Therefore, we have to show the uniform convergence of the empirical risk to the true error over the collection of classifiers we are interested in.

Let's generalize the formulation, given a collection of sets $A$, we would like to understand the uniform convergence of empirical frequencies to probabilities, i.e. we want to bound:

$$
\begin{align}
\Delta(\mathcal{A}) &= \sup_{A\in\mathcal{A}}\vert \mathcal{P}_{n}(A)-\mathcal{A}\vert \\
&= \sup_{A\in\mathcal{A}}\vert \frac{1}{n}\sum_{i=1}^{n}\mathcal{I}(X_{i}\in A)-\mathcal{P}(A)\vert
\end{align}
$$

where $x_{1},...,X_{n}$ are an i.i.d sample from some distribution $P$.

## 2. Finite collections and Hopythesis Set Complexity

How do we estimate $\Delta(\mathcal{A})$. We turn our view back to the collections of set $\mathcal{A}$ has finite cardinality $\vert \mathcal{A}\vert$. In this case, for a fixed $A$ we have by the Hoeffding's inequality:

$$
\begin{equation}
\mathcal{P}(\vert \mathcal{A}-\mathcal{A}\vert \geq t)\leq 2\,\text{exp}{-2nt^{2}}
\end{equation}
$$

we want a strong gaurantee that this convergence happens uniformly for all sets in $\mathcal{A}$, so we can use the union bound, i.e.:

$$
\begin{align}
\mathcal{P}(\Delta(A)\geq t) &= \mathcal{P}(\cup_{A\in\mathcal{A}}(\vert \mathcal{P}_{n}(A)-\mathcal{P}(A)\vert\geq t)) \\
&\leq \sum_{A\in \mathcal{A}} \mathcal{P}(\vert \mathcal{P}_{n}(A)-\mathcal{P}(A)\vert \geq t) \\
&\leq 2\vert \mathcal{A}\vert \text{exp}(-2nt^{2})
\end{align}
$$

So if we want that with probability $1-\delta$ the deviation be smaller than $t$ we need to choose:

$$
\begin{equation}
t\geq \sqrt{\frac{\In(2\vert \mathcal{A}\vert/\delta)}{2n}}
\end{equation}
$$

In orther words we have that with probability at least $1-\delta$,

$$
\begin{equation}
\Delta(\mathcal{A})\leq \sqrt{\frac{\In(2\vert \mathcal{A}\vert/\delta)}{2n}}
\end{equation}
$$

To obtain unifrom convergence over $\mathcal{A}$ we pay a price which is logarithmic in the size of the collection.

## 3. VC dimension

VC dimension measure the complexity of $\vert \mathcal{A}\vert$, which is highly related to uniform convergence.

### 3.1 Shattering
Let's ${z_{1},...,z_{n}}$ be a finite set of $n$ points. We let $N_{A}(z_{1},...,z_{n})$ be the number of distinct sets in the collection of sets

$$
\begin{equation}
{{z_{1},...,z_{n}}\cap A: A\in \mathcal{A}}
\end{equation}
$$

$N_{A}(z_{1},...,z_{n})$ is counting the *number of subsets of* ${z_{1},...,z_{n}}$ that the collection of sets $\mathcal{A}$ picks out. Note that, $N_{A}(z_{1},...,z_{n})\leq 2^{2}$(each one have chance to be selected.)

We now define the $n-$th shatter coefficient of $\mathcal{A}$ as:

$$
\begin{equation}
s(A,n)=\max_{z_{1},...,z_{n}}N_{A}(z_{1},...,z_{n})
\end{equation}
$$

The shatter coefficient is the maximal number of different subsets of $n$ points that can be picked out by the collection $\mathcal{A}$

### 3.2 VC Theory and Glivenko-Cantelli Therory

**VC Theorem:** For any distribution $\mathcal{P}, and class of sets $\mathcal{A}$ we have that:

$$
\begin{equation}
\mathcal{P}(\Delta(A)\geq t)\leq 8\,s(\mathcal{A}, n)\text{exp}(-nt^{2}/32)
\end{equation}
$$

Notes: There are two noteworthy aspects of this Theorem:

1. *Distribution free*: it can apply to any distritution on the samples.
2. VC theorem essentially reduce the question of uniform convergence to a combinatorial question about the collection of sets, i.e. we now need only to understand the shatter coefficients which are completely independent from the probability/statistics.

**Glivenko-Cantelli:** We related Glivenko-Cantelli theorem to VC theorem, the empirical CDF converges in probability to the true CDF.  We have:

$$
\begin{equation}
\mathcal{P}(\sup_{x}\vert \tilde{F}_{n}(x)-F_{x}(x)\vert\geq t) ]leq 8\,(n+1)\text{exp}(-nt^{2}/32)
\end{equation}
$$

Now verifying convergence in probability is straightforwward, for any $t> 0, \lim_{n\rightarrow \infty}8(n+1)\text{exp}(-nt^{2}/32)=0$.

**VC dimension:** VC dimension define the complexity of a set system $\mathcal{A}$, the VC dimension $d$ is the largest integer $d$ for which $s(\mathcal{A}, d)=2^{d}$. Once we have for any $n>d$, we have that $s(\mathcal{A}, n)<2^{n}$. There is a phase transition of shattering coefficients: once it is no longer exponential the shattering coefficients become polynomial in $n$. That n is the VC dimension of a set system $\mathcal{A}$.

### 3.3 Sauer's Lemma

If $\mathcal{A}$ has finite VC dimension $d$, then for $n>d$ we have that,

$$
\begin{equation}
s(\mathcal{A}, n)\leq (n+1)^{d}
\end{equation}
$$

We can use Sauer's lemma to conclude that for a system $\mathcal{A}$ of VC dimension $d$.

$$
\begin{equaiton}
\mathcal{P}(\Delta(\mathcal{A})\geq t)\leq 8(n+1)^{d}\text{exp}(-nt^{2}/32)
\end{equation}
$$

Doing the usual thing we see that with probability $1-\delta$,

$$
\begin{equation}
\Delta{\mathcal{A}}\leq \sqrt{\frac{32}{n}\[d\log(n+1)+\log(8/ \delta)\]}
\end{equation}
$$

There are some important notes:

1. If $d<\infty$ then $\Delta(\mathcal{A})\xrightarrow{p}0$, and so we have a uniform LLN for the collection of sets $\mathcal{A}$.
2. There are covnerses to the VC theorem that say roughly that if the VC dimension is infinite then there exists a distribution over the samples for which we do not have a uniform LLN.
3. Roughly, one should thinnk of the VC result as saying for a class with VC dimension d,

$$
\begin{equaiton}
\Delta(\mathcal{A})\approx \sqrt{\frac{d\log n}{n}}
\end{equation}
$$

### 3.4 VC Theory, Empirical Risk Minimization and Uniform Convergence

If $\mathcal{A}$ has VC dimension $d$ then we can use the VC theorem in a straightforwward way to conclude that with probability $1-\delta$:

$$
\begin{equation}
\sup_{f\in \mathcal{F}}\vert \tilde{R}_{n}(f)-\mathcal{P}(f(X)\neq y)\vert =\Delta(\mathcal{A})\leq \sqrt{\frac{32}{n}[d\log(n+1)+\log(8/\delta)]}
\end{equation}
$$

This way leads to a uniform convergence gurantee for empirical risk minimization over linear classifiers since they induce relatively simple sets whose VC dimension is well-understood.







