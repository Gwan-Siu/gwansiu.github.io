---
layout:     post
title:      "Basic of Statistical Learning"
date:       2020-05-06 12:00:00
author:     "GwanSiu"
catalog: true
tags:
    - Machine Learning
---

## 1. The Bayes Decision Rule for Minimum Error

- The a-postieriori probability of a sample

$$
\begin{equation}
P(Y=i\vert X) = \frac{p(X\vert Y=i)P(Y=i)}{P(X)}=\frac{\pi_{i}p_{i}(X\vert Y=i)}{\sum_{i=1}\pi_{i}p_{i}(X\vert Y=i)} = q_{i}(X)
\end{equation}
$$

- Bayes Test:

$$
\begin{equation}
\begin{split}
\frac{q_{1}(x)}{q_{2}(x)} &\lesseqgtr 1 \\

\Reftarrow\frac{\pi_{1}p_{1}(X\vert Y=1)}{\pi_{2}p_{2}(X\vert Y=2)}&\lesseqgtr 1 \\
\Rightarrow \frac{p_{1}}{p_{2}}&\lesseqgtr \frac{\pi_{2}}{\pi_{1}}
\end{split}
\end{equation}
$$

- Likelihood ratio:

$$
\begin{equation}
\ell(X) = \frac{p_{1}}{p_{2}}
\end{equation}
$$

- Discriminant function:

$$
\begin{equation}
h(x)=\log \ell(x)=\log(p_{1})-\log(p_{2})\lesseqgtr \log(\pi_{1})-\log(\pi_{2})
\end{equation}
$$


## 2. Bayer Error

In machine learning, we should compute the probability of error so as to decide whether a calssifier is good or not.

**Probability eorr:** the probability that a sample is assigned to the wrong class. Given a datum $X$, the risk is defined as follows:

$$
\begin{equation}
r(X)=min(q_{1}(X), q_{2}(X))
\end{equation}
$$


The bayes error (the expected risk):

$$
\begin{equation}
\begin{split}
\epsilon &=  \mathbb{E}(r(X))=\int r(x)p(x)\mathrm{d}x\\
&=\int \min(\pi p_{1}(x), \pi_{2}p_{2})\mathrm{d}x \\
&=\pi_{1}\int_{L1}p_{1}(x)\mathrm{d}x +\pi_{2}\int_{L2}p_{2}(x)\mathrm{d}x \\
&=\pi_{1}\epsilon_{1} + \pi_{2}\epsilon_{2}
\end{split}
\end{equation}
$$








