---
layout: post
title: Note(2)--Inequality 
date: 2018-01-30
author: Gwan Siu
catalog: True
tags:
    - Statistics and Bayesian Analysis
---

> In the early stage of statistics community, researchers paid their attention to investigate *Averages of independent random variables concentrate around their expectation.*  Researchers try to answer this question from the asymptotic(i.e. the number of random variables we average $\rightarrow \infty$), and the non-symptotic viewpoint(i.e. the number of random variables is fixed finite number). The asymptotic viewpoint is typically characterized by the the Laws of Large Numbers and Center Limit Theorems while the non-asymptotic viewpoint is focus on concentration  inequality.

> In this artical, we just give you a brief introduction to some inequalities in statistics.

#### 2.1 Markov Inequality

Firstly, the fundamental inequality is `Markov inequality`, which claim that for a positive random variable $X\geq 0$:

$$
\begin{align}
\mathbb{P}(X\geq t) \leq \frac{\mathbb{E}[X]}{t}
\end{align}
$$

Intuitively, if the mean of a random variable is very small then the random variable is unlikely to be too large too often, i.e. the probability that it is large is small. 

**Proof:** Fix an arbitrary  $t>0$. Define the indicator function:

$$
\begin{equation}
\mathbb{I}=\begin{cases}
1, \text{ if } X\geq t \\
0, \text{ if } X<t
\end{cases}
\end{equation}
$$

We have that

$$
\begin{equation}
t\mathbb{I}(t)\leq X
\end{equation}
$$

so that

$$
\begin{equation}
\mathbb{E}[X]\geq \mathbb{E}[t\mathbb{I}(t)] = t\mathbb{E}[\mathbb{I}(t)]=t\mathbb{P}(X\geq t)
\end{equation}
$$

#### 2.2 Chebyshev Inequality

If we just know about the mean of a random variable, we can derive markov inequality. However, markov inequality is relative relax. If we know more about information, such as variance, we can derive more tight bound. That is the `Chebyshev inequality`.

`Chebyshev inequality` states that for a random variance $X$, with $\text{Var}(X)=\sigma^{2}$:

$$
\mathbb{P}(\vert X-\mathbb{E}[X]\vert \geq k\sigma)\leq \frac{1}{k^{2}}, \forall k\geq 0.
$$ 

**Proof:** We can derive Chebyshev inequality from markov inequality. For a random variable $X$, with a variance $\text{Var}(X)$, then we have:

$$
\begin{align}
\mathbb{P}(\vert X-\mathbb{E}[X]\vert\geq k\sigma ) &= \mathbb{P}(\vert X-\mathbb{E}[X]\vert^{2}\geq k^{2}\sigma^{2}) \\
&\leq \frac{\mathbb{E}[\vert X-\mathbb{X}\vert^{2}]}{k^{2}\sigma^{2}} \\
&=\frac{1}{k^{2}}
\end{align}
$$

**Example :** We average i.i.d random variables $X_{1},...,X_{n}$ with the mean $\mu$ and variance $\sigma$, the result is till a random variable with the mean $\mu$ and variance $\sigma /n$, then we have the empirical mean:

$$
\begin{equation}
\tilde{\mu}_{n}=\frac{1}{n}\sum_{i=1}^{n}X_{i}
\end{equation}
$$

We apply Chebyshev ineqality to investigate how $\tilde{\mu}_{n}$ convergent to $\mu$:

$$
\begin{equation}
\mathbb{P}(\vert \tilde{\mu}_{n}-\mu \vert \geq \frac{k\sigma}{\sqrt{n}})\leq \frac{1}{k^{2}}
\end{equation}
$$

#### 2.3 Chernoff Method

From Chebyshev inequality, we can observe that if the random variable $X$ has a finite $k-$th central moment then we have that:

$$
\begin{equation}
\mathbb{P}(\vert X-\mathbb{E}\vert \geq t)\leq \frac{\mathbb{E}[\vert X-\mathbb{E}\vert^{k}]}{t^{k}}
\end{equation}
$$

For many random variables, the moment generating function will exist in a neighborhood around 0, i.e. the mgf is finite for all $\vert t\vert \leq b$ where $b>0$ is some constant. In these case, we can use the mgf to produce a tail bound.

$$
\begin{equation}
\mathbb{E}((X-\mu)>\epsilon)=\mathbb{P}(\text{exp}(t(X-\mu))>\text{exp}(t\epsilon))\leq \frac{\mathbb{E}[\text{exp}(t(X-\mu))]}{\text{exp}(t\epsilon)}
\end{equation}
$$

In euqation, $t$ is a parameter and we can choose specific $t$ to obtain tight bound, i.e. we can write this bound as:

$$
\begin{equation}
\mathbb{E}((X-\mu)>\epsilon)\leq \inf_{0\leq t \leq b} \text{exp}(-t(\mu+\epsilon))\mathbb{E}[\text{exp}(tX)]
\end{equation}
$$

where $\mathbb{E}[\text{exp}(tX)]$ is moment generated function, and this bound is called `Chernoff's bound`. 

##### 2.3.1 Gaussian Tail Bounds via Chernoff

Suppose that $X\sim N(\mu, \sigma^{2})$, then the mgf of $X$ is:

$$
\begin{equation}
M_{X}(t)=\mathbb{E}[\text{exp}(tX)]=\text{exp}(t\mu+t^{2}\sigma^{2}/2)
\end{equation}
$$

Talor seriers:

$$
\begin{equation}
e^{tX} = 1 + tX + \frac{t^{2}X^{2}}{2!}+\frac{t^{3}X^{3}}{3!}+...+\frac{t^{n}X^{n}}{n!}+...+...
\end{equation}
$$

Hence,

$$
\begin{equation}
M_{X}(t)= \mathbb{E}[\text{exp}(tX)] = 1 + t\mathbb{E}[X] + \frac{t^{2}\mathbb{E}[X^{2}]}{2!}+\frac{t^{3}\mathbb{E}[X^{3}]}{3!}+...+\frac{t^{n}\mathbb{E}[X^{n}]}{n!}+...+...
\end{equation}
$$

The mgf is defined for all $t$. To apply the Chernoff bound we then need to compute:

$$
\begin{equation}
\inf_{t\geq 0} \text{exp}(-t(\mu+\epsilon))\text{exp}(t\mu+t^{2}\sigma^{2}/2)=\inf_{t\geq 0}\text{exp}(-t\epsilon + t^{2}\sigma^{2}/2)
\end{equation}
$$

which is minimized when $t=\mu/\sigma^{2}$ which in turn yields the tail bound,

$$
\begin{equation}
\mathbb{P}(X-\mu>\epsilon)\leq \text{exp}(\epsilon/(2\sigma^{2}))
\end{equation}
$$

This is often referred to as *one-side* or *upper tail bound*. We can use the fact that Normal distribution is symmetric and thus we can obtain *two-sided Gaussian tail bound*:

$$
\begin{equation}
\mathbb{P}(\arrowvert X-\mu \arrowvert >\epsilon) \leq 2\text{exp}(-\epsilon^{2}/(2\sigma^{2}))
\end{equation}
$$

*To be notice that Guassian tail bound is more sharper than Chebyshev bound*. Suppose we consider the average of i.i.d Gaussian random variables, i.e. we have $X_{1},..., X_{n}\sim N(\mu, \sigma^{2})$ and we construct the estimate:

$$
\begin{equation}
\tilde{\mu}=\frac{1}{n}\sum_{i=1}^{n} X_{i}
\end{equation}
$$

Using the fact that the average of Gaussian random variables is still Guassian that $\tilde{\mu}\sim N(\mu, \sigma^{2}/n)$. In this case, we can obtain that:

$$
\begin{equation}
\mathbb{P}(\vert \tilde{\mu}-\mu \vert \geq k\sigma/\sqrt{n})\leq 2\text{exp}(-k^{2})
\end{equation}
$$

Comparing with Chebyshev inequatlity, we should know two things:

1. Both iunequality say roughly that the deviation of the average from the expected value goes down as $1/\sqrt{n}$.
2. However, the gaussian tail bound says that if the random variables are Gaussian then the chance that deviation is much bigger than $\sigma/\sqrt{n}$ goes down *exponentially fast*.  Take a concrete example:

More generally, Chebyshev tells us that with probability at least $1-\delta$:

$$
\begin{equation}
\vert \tilde{\mu}-\mu\vert \leq \frac{\sigma}{\sqrt{n\delta}}
\end{equation}
$$

while the exponential tail bound tells us that:

$$
\vert \tilde{\mu}-\mu \vert \leq \sigma\sqrt{\frac{\text{ln}(2/\sigma)}{n}}
$$

The first term goes up polynomially as $\delta \rightarrow 0$, while the second term refined bound goes up only logarithmically.

#### 2.3.2 sub-Gaussian

Fomally, a random variable $X$ with mean $\mu$ is `sub-Gaussian` if there exists a positive number $\sigma$ such that

$$
\begin{equation}
\mathbb{E}[\text{exp}(t(X-\mu))]\leq \text{exp}(\sigma^{2}t^{2}/2)
\end{equation}
$$

for all $t\in \mathbb{R}$. Gaussian random variables with variance $\sigma^{2}$ satisfy the above condition with equality, so a $\sigma$-sub-Gaussian random variable basically just has an mgf that is dominated by a Gaussian with variance $\sigma$. Roughly, `sub-Gaussian` random variables whose tails decay faster thatn Gaussian.

It is straightforward to go through the above Chernoff bound to conclude that for a sub-Gaussian random variable we have the same two-side exponential tail bound:

$$
\begin{equation}
\mathbb{P}(\vert X-\mu \vert >\epsilon) \leq 2\text{exp}(-\epsilon^{2}/(2\sigma^{2}))
\end{equation}
$$

Suppose we have n i.i.d random variable $\sigma$ sub-Gaussian random variables $X_{1},...,X_{n}$, then by independent we obtain that:

$$
\begin{align}
\mathbb{E}[\text{exp}(t(\tilde{\mu}-\mu))] &= \mathbb{E}[\text{exp}(t/n\sum_{i=1}^{n}(X_{i}-\mu))] \\
&= \prod_{i=1}^{n}\mathbb{E}[\text{exp}(t(X_{i}-\mu)/n)] \\
&\leq \text{exp}(t^{2}\sigma^{2}/(2n))
\end{align}
$$

Alternatively, the average of $n$ independent $\sigma-$sub Gaussian random variables is $\sigma/\sqrt{n}-$sub Gaussian. THis yields the tail bound for the average of sub Gaussian random variables:

$$
\begin{equation}
\mathbb{P}(\vert \tilde{\mu}-\mu \arrowvert \geq k\sigma/\sqrt{n})\leq 2\text{exp}(-k^{2})
\end{equation}
$$

##### 2.3.3 Hoeffding's Inequality

**Lemma:** Suppose that $a\leq X \leq b$. Then

$$
\begin{equation}
\mathbb{E}(\text{exp}(tX))\leq \text{exp}(t\mu)\text{exp}(t^{2}(b-a)^{2}/8)
\end{equation}
$$

where $\mu=\mathbb{E}[X]$.

**Proof(Use property of convex function):** We will assume that $\mu=0$. Since $a\leq X\leq b$, we can write $X$ as a convex combination of $a$ and $b$, namely, $X=\alpha b+(1+\alpha)a$ where $\alpha=(X-a)/(b-a)$. BY the convexity of the function $y\rightarrow e^{ty}$, we have:

$$
\begin{equation}
\text{exp}(tX)\leq \alpha \text{exp}(tb)+(1-\alpha)\text{exp}(ta)=\frac{X-a}{b-a}\text{exp}(tb)+\frac{b-X}{b-a}\text{exp}(ta)
\end{equation}
$$

Take the expectation of both sides and use the fact that $\mathbb{E}[X]=0$ to obtain:

$$
\begin{equation}
\mathbb{E}[\text{exp}(tX)]\leq -\frac{a}{b-a}+\frac{b}{b-a}\text{exp}(ta)=\text{exp}(g(\mu)) 
\end{equation}
$$

where $\mu=t(b-a), g(\mu)=-\gamma\mu+\text{log}(1-\gamma+\gamma\text{exp}(\mu))$ and $\gamma=-a/(b-a)$. Note that $g(0)=g^{\cdot}(0)=0$. Also, $g^{\cdot}\leq 1/4$ for all $\mu>0$. By Taylor's theorem, there is a $\xi\in (0,\mu)$ such that 

$$
\begin{equation}
g(\mu)=g(0)+\mu g^{`}(0) +\frac{\mu^{2}}{2}g^{``}(\xi)=\frac{\mu^{2}}{8}g^{``}(\xi)\leq \frac{\mu^{2}}{8}=\frac{t^{2}(b-a)^{2}}{8}
\end{equation}
$$

Hence, $\mathbb{E}[\text{exp}(tX)]\leq \text{exp}(g(\mu)) \leq \text{exp}(t^{2}(b-a)^{2}/8)$.

Next we apply *Chernoff's method*.

**Lemma:** Let $X$ be a random variable. Then

$$
\begin{equation}
\mathbb{P}(X>\epsilon) \leq \inf_{t\geq 0}\text(exp)(-t\epsilon)\mathbb{E}[\text{exp}(tX)]
\end{equation}
$$

**Proof:** For any $t>0$:

$$
\begin{equation}
\mathbb{P}(X>\epsilon) = \mathbb{P}(\text{exp}(X)>\epsilon)=\mathbb{P}(\text{exp}(tX)>\t\epsilon)\leq\text{exp}(-t\epsilon)\mathbb{E}[\text{exp}(tX)]
\end{equation}
$$

**Theorem(Hoeffding's inequality):** Let $Y_{1},...,Y_{n}$ be iid observation such that $\mathbb{E}(Y_{i})=\mu$ and $a\leq Y_{i}\leq b$. Then for any $\epsilon>0$,

$$
\begin{equation}
\mathbb{P}(\vert \tilde{Y}_{n}-\mu\vert \geq \epsilon)\leq 2\text{exp}(-2n\epsilon^{2}/(b-a))
\end{equation}
$$

##### 2.3.4 Generalization to sub-Gaussian distribution.

Suppose $X_{1},...,X_{n}$ with variance $\sigma_{1},...,\sigma_{n}$ are sub-Gaussian. Then we using iid principle you can verify that their average $\tilde{\mu}$ is $\sigma-$sub gaussian, where:

$$
\begin{equation}
\sigma=\frac{1}{n}\sqrt{\sum_{i=1}^{n}\sigma^{2}_{i}}
\end{equation}
$$

Then we have exponential tail inequality,

$$
\begin{equation}
\mathbb{P}(\arrowvert \frac{1}{n}\sum_{i=1}^{n}(X_{i}-\mathbb{E}[X_{i}])\arrowvert\geq)\leq \text{exp}(-t^{2}/(2\sigma)^{2})
\end{equation}
$$

**Notice:** the random variables just is required to be independent but no longer need to be identically distribution.(i.e. they can have different means adn sub-Gaussian parameters.)