---
layout:     post
title:      "Information Theory and Machine Learning"
date:       2021-05-15 12:00:00
author:     "GwanSiu"
catalog: true
tags:
    - Machine Learning
---

In this article, most masterials are referred to [1]

## 1. Entropy
The entropy of a probability distribution is a measurement of uncertainty. Besides that, we also can use entropy to define the **information content** of a data source. You can find an example in [1]. Suppose we have a dataset $\mathcal{D}=(X_{1},\cdots, X_{n})$ generated from a distribution $p$. If the dataset $\mathcal{D}$ has high information content, it means that the distribution $p$ has high entropy and the value of each observation $X_{i}$ is hard to be predicted. Conversely, if the distribution $p$ has $0$ entropy, then every $X_{n}$ is the same, so the dataset $\mathcal{D}$ does not contain much information content.

## 1.1 Entropy for discrete random variable

The entropy of a discrete random variable $X$ drawn from the distribution $p$ over $K$ states is defined by

$$
\begin{equation}
\mathbb{H}(X) = -\sum_{k=1}^{K}p(X=k)\log_{2}p(X=k)=-\mathbb{E}_{X}[\log p(X)].
\end{equation}
$$

If we use log base $2$, an unit of entropy is called bits. If we use log base $e$, an unit of entropy is called nats. **It is worthing noticing that the uniform distribution in all discrete distributions has maximum entropy**. In contrast, the distribution with minimum entropy is any delta-function which puts all mass in one state, so there is no uncertainty in this distribution.

In general, 

$$
\begin{equation}
\mathcal{H}(X, Y) \leq \mathcal{H}(X) + \mathcal{H}(Y),
\end{equation}
$$

with equality iff $X$ and $Y$ are independent. This inequality can be viewed as the upper bound of the entropy of the joint distribution, and is reasonable. Because the random variables are correlated, it can reduce the "the degree of the freedom" of the system, and hence the overall entropy is reduced.  

The lower bound on $\mathcal{H}(X, Y)$ is 

$$
\begin{equation}
\mathcal{H}(X, Y) \geq \max (\mathcal{H}(X), \mathcal{H}(Y)) \geq 0.
\end{equation}
$$

This is veru intuitive. If $Y$ is a determinstic function of $X$, then $\mathcal{H}(X, Y)=\mathcal{H}(X)$. This means that you cannot reduce the entropy by just adding more unknows into problems. Actually, you need to observe more data.


## 1.2 Conditional entropy

The entropy of $Y$ given $X$ represents the uncertainty of $Y$ after we have observed $X$, which is averaged over possible values of $X$.

$$
\begin{align}
\mathbb{H} &= \mathbb{E}_{p(X)}[\mathbb{H}(p(Y\vert X))] \\
&=\sum_{x}p(x)\mathbb{H}(p(Y\vert X)) \\
&= -\sum_{x}p(x)\sum_{y} p(y\vert x)\log p(y\vert x) \\
&=-\sum_{x,y}p(x,y)\log p(y\vert x)  \\
&=-\sum_{x,y} p(x, y)\log \frac{p(x,y)}{p(x)} \\
&=-\sum_{x,y} p(x, y)\log p(x,y) - \sum_{x}p(x)\log \frac{1}{p(x)} \\
&= \mathcal{H}(X,Y) - \mathcal{H}(X)
\end{align}
$$

If $Y$ is a determinstic function of $X$, then knowing all information of $X$ is equivalent to knowing $Y$, so $\mathcal{H}(Y\vert X)=0$. If $X$ and $Y$ are independent, observing $X$ has not benefit to knowing information about $Y$, and $\mathcal{H}(Y\vert X)=\mathcal{H}(Y)$. Since $\mathcal{H}(X, Y)\leq \mathcal{H}(Y) + \mathcal{H}(X)$, we have

$$
\begin{equation}
\mathcal{H}(Y\vert X)\leq \mathcal{H}(Y),
\end{equation}
$$

with equality iff $X$ and $Y$ are independent. It is worth noticing that the words "on average" in the definition are necessary because for any paricular observation, $\mathcal{H}(Y\vert x)> \mathcal{Y}$ is adimissable.

One more thing, by definition, we can re-write the Eq.() as follows:

$$
\begin{equation}
\mathcal{H}(X_{1}, X_{2}) = \mathcal{H}(X_{1}) + \mathcal{H}(X_{2}\vert X_{1}).
\end{equation}
$$
 
This can be generalized and we can obtain the **chain rule of entropy**:

$$
\begin{equation}
\mathcal{H}(X_{1},X_{2},\cdots, X_{n}) = \sum_{i=1}^{n}\mathcal{H}(X_{i}\vert X_{1},\cdots, X_{i-1})
\end{equation}
$$

## 1.3 Perplexity

The perplexity is a measure of predictability. The perplexity of a discrete probability distribution $p$ is defined as

$$
\begin{equation}
\textb{perplexity}(p)=2^{\mathcal{H}(p)}.
\end{equation}
$$

Suppose we have an empirical distribution based on dataset $D$, which is defined by:

$$
\begin{equation}
p_{D}(x\vert D) =\frac{1}{N} \sum_{n=1}^{N}\delta_{x_{n}}(x).
\end{equation}
$$

We can measure how well $p$ predicts $D$ by computing

$$
\begin{equation}
\text{perplexity}(p_{D}, p)=2^{\mathcal{H}(p_{D}, p)}
\end{equation}
$$

## 1.4 Differential entropy for continuous random variables

If $X$ is a continuous random variable with pdf $p(x)$, the differential entropy is defined as follows:

$$
\begin{equation}
h(x)=-\displaystyle{\int_{\mathcal{X}}} \mathrm{d}x p(x)\log p(x).
\end{equation}
$$

Unlike the entropy of a discrete probability distribution, the differential entropy can be negative. For example, suppose $X\sim U(0, a)$, then

$$
\begin{equation}

h(X)=-\displaystyle{\int}^{a}_{0}\mathrm{d}x \frac{1}{a}\log\frac{1}{a} = \log a.
\end{equation}
$$
If $a=\frac{1}{8}$, then $H(X)=-3$.

In general, to compute the differential entropy for a continuous random variables. We usually discrete or quantize the varibales. A simple approach is to divide the distribution into several bins based on its empirical quantiles, but the number of bins used is very critical. A heuristical method is given by

$$
\begin{equation}
B=N^{1/3}\frac{\max(\mathcal{D})-\min(\mathcal{D})}{3.5\sigma(\mathcal{D})}
\end{equation}
$$

where $\sigma(mathcal{D})$ is the empirical standard deviation of the data, and $N=\vert \mathcal{D}\vert$ is the number of datapoints in the empirical distribution. **However, the quantization technique is not work for high-dimensional random variables due to the curse of dimensionality.**

## 2 Relative Entropy(KL divergence)

### 2.1 Definition

Kullback-Leiber (KL) divergence, also known as the information gain or relative entropy, is a distance metric to measure how "close" or "similar" of given two distributions $p$ and $q$. 

The KL divergence is defined as 

$$
\begin{equation}
D_{KL}(p \Arrowvert q) = \sum_{k=1}^{K}p_{k}\log\frac{p_{k}}{q_{k}}.
\end{equation}
$$

We can easily extend the definition to continuous distributions, which is defined as

$$
\begin{equation}
D_{KL}(p\Arrowvert q)=\displaystyle{\int}\mathrm{d}x p(x)\log\frac{p(x)}{q(x)}.
\end{equation}
$$

Next, we can rewrite the above equation of the discrite distribution as follows:

$$
\begin{align}
D_{KL}(p \Arrowvert q) &= \sum_{k=1}^{K}p_{k}\log p_{k} - \sum_{k=1}^{K}\log q_{k}, \\
&= -\mathbb{H}(p) + \mathbb{H}(q).
\end{align}
$$

We can find that the first term is the negative entropy, and the second term is the cross entropy. Therefore, we can interpret KL divergence as the "extra number of bits" you need to pay when compressing data samples from the ground-truth distribution $p$ using the incorrect distribution $q$ as the basis of your coding scheme. 

## 2.2. KL Divergence and MLE

**In fact, minimizing KL divergence is equivalent to MLE.**

Suppose we want find a distribution $q$ that is as close as possible to the distribution $p$, as measured by KL divergence,

$$
\begin{equation}
q^{\ast} =\arg\min_{q} D_{KL}(p \Arrowvert q) =\arg\min_{q}\displaystyle{\int} p(x)\log p(x)\mathrm{d}x - \displaystyle{\int} p(x)\log q(x)\mathrm{d}x.
\end{equation}
$$

Suppose $p$ is an empirical distribution, which puts a probability atom on the observed training data and zero mass everywhere else, which is:

$$
\begin{equation}
p_{D}(x) =\frac{1}{N}\sum_{n=1}^{N} \delta(x - x_{n}).
\end{equation}
$$

We minimize the KL divergence between $p_{D}$ and $q$, given by

$$
\begin{align}
D_{KL}(p_{D}\Arrowvert q) &= -\displaystyl{\int} p_{D}(x)\log q(x)\mathrm{d}x + \displaystyle{\int}p(x)\log p(x)\mathrm{d}x, \\
&= -\displaystyle{\int}\left(\frac{1}{N}\sum_{n}\delta(x - x_{n})\right)\log q(x)\mathrm{d}x + C, \\
&= -\frac{1}{N}\sum_{n}\log q(x_{n}) + C,
\end{align}
$$

where $C=\int p(x)\log p(x)\mathrm{d}x$ is constant and independent of $q$. This is called the cross entropy objective, and is equal to the average negative log likelihood of $q$ on the training set. Therefore, we can find that minimizing KL divergence to the empirical distribution is equivalent to maximizing likelihood.

This shows that likelihood-based training puts too much weight on the training set. In the following description, I will refer the masterials. 

In most applications, the empirical distribution is not really a good representation of the true distribution, since it just puts "spikes" on a finite set of points, and zero density everywhere else. Even if the dataset is large, the universe from which the data is sampled is usually even larger. We can smootht the empirical distribution using **kernel density estimation**, but that would require a similar kernel on the space of images. An alternative method is **data augmentation**, which perturbs the observed data samples in way that we believe reflects plausible "natural variation".

## 2.3 Forward v.s. reverse KL Divergence

Suppose we want to approximate a distribution $p$ using a simple distribution $q$. To achieve this goal, we can minimize $D_{KL}(p\Arrowvert q)$ or $D_{KL}(q\Arrowvert p)$. These two formulations lead to different behaviours.

The forward KL divergence is defined by 

$$
\begin{equation}
D_{KL}(p\Arrowvert q) = \displaystyle{\int} p(x)\log \frac{p(x)}{q(x)}\mathrm{d}x.
\end{equation}
$$

Minimizing this equation with respect to (wrt) $q$ is an **M-projection or moment projection**.

Suppose an input $x$ for which $p(x)>0$ but $q(x)=0$, the term $\log\frac{p(x)}{q(x)}$ will be infinite. Minimizing $D_{KL}(p\Arrowvert q)$ will force $q$ to include all the areas of space for which $p$ has non-zero probability. In other words, the distribution $q$ will be **zero-avoiding** or **mode covering**, and typically leads to over-estimation of the support of $p$.


The reversed KL divergence, also called the exclusive KL divergence, defined by

$$
\begin{equation}
D_{KL}(q\Arrowvert p) = \displaystyle{\int} q(x)\log \frac{q(x)}{p(x)}\mathrm{d}x.
\end{equation}
$$

Minimizing this equation wrt $q$ is known as an **I-projection or information projection**.

In this case, suppose an input $x$ for which $p(x)=0$ but $q(x)>0$, the term $\log\frac{q(x)}{p(x)}$ will be infinite. Minimizing the reverse KL divergence will force $q$ to avoid all the areas of space for which $p$ has zero probability. This is called **zero-forcing** or **mode-seeking**. In this case, the distribution $q$ under-estimate the support of $p$.



















## Reference

1. Murphy, Kevin P. Probabilistic Machine Learning: An introduction. MIT Press, 2022.