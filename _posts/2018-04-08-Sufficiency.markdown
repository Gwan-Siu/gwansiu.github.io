---
layout: post
title: Note(8)-- Sufficiency Statistics
date: 2018-04-08
author: Gwan Siu
catalog: True
tags:
    - Statistics and Bayesian Analysis
---

### 1.Sufficient Statistic and Factorization Theorem

#### 1.2 The Definition of Sufficient Statistic

**Sufficiency Statistics:** A statistics $T(X_{1},...,X_{n})$ is said to be sufficient for the parameter $\theta$ if the conditional distribution $p(X_{1},...,X_{n}\vert T(X_{1},...,X_{n})=t;\theta)$ does not depend on $\theta$ for any value of $t$.

Rough interpretation, once we know the value of the sufficient statistic, the joint distribution no longer has any more information about the parameter $\theta$. From the view of data reduction, once we know the value of the sufficient statistic, we can throwing away all the data.

**Actually, we can understand sufficient statistic from two views: (1). data reduction viewpoint where we could like to discard non-informative pieces of the dataset. (2). the risk reduction viewpoint where we want to construct esitmators that only depend on meaningful variation viewpoint where we want to construct estimators that only depend on meaningful variation in the data.**

#### 1.3 The Factorization Theorem

Besides definition of sufficient statistic, we can use Neyman-Fisher Factorization criterion to justify whether a statistic is sufficient.

**Theorem:** $T(X_{1},...,X_{n})$ is sufficient for $\theta$ if and only if the joint pdf/pmf of $(X_{1},...,X_{n})$ can be factored as 

$$
\begin{equation}
p(x_{1},...,x_{n};\theta) = h(x_{1},...,x_{n})\times g(T(x_{1},...,x_{n});\theta)
\end{eqaution}
$$

**Proof:** Factorization $\Rightarrow$ sufficiency:

$$
\begin{align}
p(x_{1},..,x_{n}\vert T=t;\theta) &= \frac{p(x_{1},...,x_{n}, T=t;\theta)}{p(T=t;\theta)} \\
&= \frac{\mathcal{I}(T(x_{1},...,x_{n})=t)h(x_{1},...,x_{n})\times g(t;\theta)}{\sum_{x_{1},...,x_{n};T(x_{1},...,x_{n})=t}h(x_{1},...,x_{n})\times g(t;\theta)} \\
&=\frac{\mathcal{I}(T(x_{1},...,x_{n})=t)h(x_{1},...,x_{n})}{\sum_{x_{1},...,x_{n};T(x_{1},...,x_{n})=t}h(x_{1},...,x_{n})}
\end{align}
$$

which does not depend on $\theta$.

Sufficiency $\Rightarrow$ Factorization: we simply define $g(t;\theta)=p(T=t;\theta)$, and $h(x_{1},...,x_{n})=p(x_{1},...,x_{n}\vert T=t;\theta)$, where by sufficiency we note that the latter function does not depend on $\theta$. Now, it is straightforward to verify that factorization theorem holds.

### 2. Sufficient Statistic-The Partition Viewpoint

It is better to describe sufficiency in terms of partitions of the sample space.

**Example 1:** Let $X_{1},X_{2},X_{3}\sim \text{Bernoulli}(\theta)$, Let $T=\sum X_{i}$.

<img src="https://raw.githubusercontent.com/Gwan-Siu/BlogCode/master/other/99457940-310D-40EE-B02B-0978CF1140E6.png" width="600" height="400"/>


1. A partition $B_{1},...,B_{k}$ is sufficient if $f(x\vert X\in B)$ does not depend on $\theta$.
2. A statistic $T$ includes a parition. For each t, ${x:T(x)=t}$ is one element of the partition. $T$ is sufficient if and only if the partition os sufficient.
3. Two statistic can generate the same partition: example: $\sum_{i} X_{i}$ and $3\sum_{i}X_{i}$.
4. If we split any element $B_{i}$ of a sufficient partition into samller pieces, we get another sufficient partition.

**Example 2:** Let $X_{1},X_{2},X_{3}\sim \text{Bernoulli}(\theta)$. Then $T=X_{1}$ is not sufficient. Look at its partition:

<img src="https://raw.githubusercontent.com/Gwan-Siu/BlogCode/master/other/CE915738-6F1F-4207-BE4A-6DFF2DCB2D2C.png" width="600" height="400"/>

### 3. Sufficient Statistic-the Risk Reduction Viewpoint

**Suppose** we observe $X_{1},...,X_{n}\sim p(X;\theta)$ and we would like to estimate $\theta$, i.e. we want to construct some function of the daa that is close in some sense to $\theta$. We construct an estimator $\tilde{\theta}=(X_{1},...,X_{n})$. In order to evaluate our estimator we might consider how far our estimate is from $\theta$ on average, i.e. we can define:

$$
\begin{equation}
R(\tilde{\theta}, \theta)=\mathcal{E}[(\tilde{theta}-\theta)^{2}]
\end{equation}
$$

then we decompose it into bias and variance, i.e.:

$$
\begin{equation}
\mathcal{E}[(\tilde{\theta}-\theta)^{2}]=(\mathcal{E}[\tilde{\theta}]-\theta)^{2}+\mathcal{E}[\tilde{\theta}-\mathcal{E}[\tilde{\theta}]]^{2}
\end{equation}
$$

where the first term is referred to as the bias and second is the variance. Hence, we can see that estimator is not depend only on sufficient statistics can be improved. This is known as the Rao-blackwell theorem.

Let $\tilde{\theta}$ be an estimator. Let $T$ be any sufficient statistic and define $\tilde{\theta}=\mathcal{E}[\tilde{\theta}\vert T]$

**Rao-Blackwell Theorem:**

$$
\begin{equation}
R(\tilde{\theta},\theta)\leq R(\tilde{\theta}, \theta)
\end{equation}
$$

**Proof:**

$$
\begin{align}
   R(\tilde{\theta}, \theta) &= \mathcal{E}[(\mathcal{E}[\tilde{\theta}\vert T]-\theta)^{2}] \\
   &= \mathcal{E}[(\mathcal{E}[(\tilde{\theta}-\theta)\vert T])^{2}] \\
   &\leq \mathcal{E}[(\mathcal{E}[(\tilde{\theta}-\theta)^{2}\vert T])] \\
   &= R(\tilde{\theta},\theta)
\end{align}
$$






