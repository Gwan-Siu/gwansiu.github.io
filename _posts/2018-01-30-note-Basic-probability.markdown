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
\begin{align}
\mathbb{P}(X\in A)=\mathbb{P}({\omega \in \Omega: X(\omega)\in A})
\end{align}
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

**Definition:**Suppose that $X\sim P$ and $Y\sim Q$. We say that $X$ and $Y$ have the same distribution if $\mathbb{P}(X\in A)=Q(Y\in A)$ for all $A$. In that case we say that $X$ and $Y$ are *equal in distribution* and we write $X、overset{d}{=}Y$}

**Lemma 1.1:** $X\overset{d}{=}Y$ *if and only if* $F_{X}(t)=F_{Y}(t)$ for all $t$.

#### 3. Expectation, Variance and Generated Function

##### 3.1 Expetation and Its Properties.
The `mean` or `expected value`  of $g(X)$ is

$$
\begin{align}
\mathbb{E}(g(X)) = \int g(x)dF(x)=\int g(x)dP(x) = 
\begin{cases}
\int_{-\infty}^{\infty}g(x)p(x)dx\text{, if }X\text{ is continuous.}\\
\sum_{j}g(x_{j})p(x_{j})\text{ if }X\text{ is discrete.}
\end{cases}
\end{align}
$$

Properties of expected value:
- 1. **Linearity of Expectation:** $\mathbb{E}(\sum_{j=1}^{k}c_{j}g_{j}(X))=\sum_{j=1}^{k}c_{j}\mathbb{E}(g_{j}(X))$.
- 2. If $X_{1},...,X_{n}$ are independent then $\prod$
- 
$$
\begin{align}
\mathbb{E}(\prod_{i=1}^{n}X_{i})=\prod_{i=1}^{n}\mathbb{E}(X_{i})
\emnd{align}
$$

- 3. $\mu$ is often used to denote $\mathbb{E}(X)$.

`Roughly, Expectation is a linear operator. More insight, Expectation is weight average from the mathematics perspective.`

##### 3.2 Variance and Its Properties

The defintion of `Variance` is: $\text{Var}(X)=\mathbb{E}\[(X-\mu)^{2}\]$ or $\text{Var}(X)=\mathbb{E}(X^{2})-(E\[X\])^{2}$. It's sum of distances between each point and the mean. Physically, it describes the degree of difussion of points.

If $X_{1},...,X_{n}$ are independent then

$$
\begin{align}
\text{Var}(\sum_{i=1}^{n}a_{i}X_{i})=\sum_{i}a^{2}_{i}\text{Var}(X_{i})
\end{align}
$$

The covariance is

$$
\begin{align}
\text{Cov}(X,Y)=\mathbb{E}[(X-\mu_{x})(Y-\mu_{y})]=\mathbb{E}(XY)-\mu_{X}\mu_{Y}
\end{align}
$$
and the correlation is $\rho_{X,Y} =\text{Cov}(X,Y)/\sigma_{x}\sigma_{y}$. Recall that $-1\leq\rho \geq 1$.

Proof: $\Cov(X,Y)=\math{E}(XY)-\mu_{X}\mu_{Y}$.

$$
\begin{align}
    Cov(X,Y) &= \mathbb{E}\[(X-\mu_{X})(Y-\mu_{Y})\] \\
    &= \mathbb{E}\[XY-X\mu_{Y}-Y\mu_{X}-\mu_{X}\mu_{Y}\] \\
    &=\mathbb{E}\[XY\]-\mu_{X}\mu_{Y}
\end{align}
$$

Proof: $-1\leq \rho_{X,Y} \leq 1$.(Cauchy-Schwarz Inequality)

$$
|\rho_{X,Y}| &= \arrowvert \frac{\text{Cov}(X,Y)}{\sigma_{X}\sigma_{Y}}\arrowvert \\
&=\arrowvert \frac{E\[(X-\mu_{X})(Y-\mu_{Y})\]}{\sigma_{X}\sigma_{Y}} \arrowvert\\
&\leq \arrowvert \frac{E\[(X-\mu_{X})\]}{\sgima_{X}} \arrowvert \arrowvert \frac{E\[(Y-\mu_{Y})\]}{\sigma_{Y}}\arrowvert
&\leq \arrowvert \frac{E\[(X-\mu_{X})^{2}\]}{\sgima_{X}^{2}} \arrowvert \arrowvert \frac{E\[(Y-\mu_{Y})^{2}\]}{\sigma_{Y}^{2}}\arrowvert \text{,Convex}\\
&= 1
$$

##### 3.3 Conditional Expectation and Variance

The **conditional expectation** of $Y$ given $X$ is the random variable $\mathbb{E}(Y|X)$ whose value, when $X=x$ is

$$
\begin{align}
\mathbb{E}(Y|X=x)=\int yp(y|x)dy
\end{align}
$$
where $p(y|x)=p(x,y)/p(x)$.

The *Law of Total Expectation or Law of Iterated Expectation:*

$$
\begin{align}
\mathbb{E}(Y)\mathbb{E}\[\mathbb{E}\[Y|X\]\]=\int \mathbb{E}(Y|X=x)p_{X}(x)dx
\end{align}
$$

The *Law of Total Variance* is

$$
\begin{align}
\text{Var}(Y)=\text{Var}\[E\[Y|X\]\]+\mathbb{E}\[\text{Var}(Y|X)\]
\end{align}
$$

##### 3.4 Moment Generated Function

The *mement generated function*(mgf) is

$$
\begin{align}
    M_{X}(t)=\mathbb{E}\[E^{tX}\]
\end{align}
$$

If $M_{X}(t)=M_{Y}(t)$ for all $t$ in an interval around 0 then $X\overset{d}{=}Y$.

The moment generated function can be used to "generate" all the moments of a distribution, i.e. we can take derivatives of the mgf with respect to $t$ and evaluated at $t=0$, i.i. we have that 

$$
\begin{align}
    M_{X}^{n}(t)|_{t=0}=\mathbb{E}(X^{n})
\end{align}
$$

##### 4. Independence

**(Definition):** $X$ and $Y$ are *independent* if and only if

$$
\begin{align}
\mathbb{E}(X\in A, Y\in B) = \mathbb{P}(X\in A)\mathbb{Y\in B}
\end{align}
$$

for all $A$ and $B$.

**Theorem 1.2** Let $(X,Y)$ be a bivariate random vector with $p_{X,Y}(x,y)$. $X$ and $Y$ are *independent iff* $p_{X,Y}=(x,y)=p_{X}(x)p_{Y}(y)$.

$X_{1},...,X_{n}$ are independent if and only if

$$
\begin{align}
\mathbb{P}(X_{1}\in A_{1},..., X_{n}\in A_{n}) =\prod_{i=1}^{n}\mathbb{P}(X_{i}\in A_{i})
\end{align}
$$
Thus, $p_{x_{1},...,x_{n}}=\prod_{i=1}^{n}p_{X_{i}}(x_{i})$.

If $X_{1},...,X_{n}$ are independent and identically distributed we say they are `iid` and we write

$$
X_{1},...,X_{n}\sim P \text{ or }X_{1},...,X_{n}\sim F \text{ or } X_{1},...,X_{n} \sim p
$$

`Independence and condition: A and B are independent events then `$P(A|B)=P(A)$`Also, for any pair of events A and B`

$$
\begin{align}
P(AB)=P(A|B)P(B)=P(B|A)P(A)
\end{align}icassp 
$$   

Independece means that knowing `B` does not change the probability of `A`.

##### 5.Transformations

Let $Y=g(X)$ where: $g:\mathbb{R}\rightarrow \mathbb{R}$. Then

$$
\begin{align}
F_{Y}(y) = \mathbb{P}(Y\leq y)=\mathbb{P}(g(X)\leq y)=\int_{A(y)}p_{X}(x)dx
\end{align}
$$
where $A_{y}={x:g(x)\leq y}$.

The density is $p_{Y}(y)=F^{`}_{Y}(y)$. If $g$ is monotonic, then

$$
\begin{align}
p_{Y}(y)=p_{X}(h(y))\arrowvert \frac{dh(y)}{dy}\arrowvert
\end{align}
$$
where $h=g^{-1}$.

Let $Z=g(X,Y)$. For example, $Z=X+Y$ or $Z=X/Y$. Then we find the pdf of $Z$ as follows:

- 1. For each $z$, find that set $A_{z}={(x,y):g(x,y)\leq z}$.
- Find the CDF:
$$
\begin{align}
F_{Z}(z)P(Z\leq z)=P(g(X,Y)\leq z)=P({(x,y):g(x,y)\leq z}) = \int \int_{A_{z}}p_{X,Y}(x,y)dxdy
\end{align}
$$
- 3.The pdf is $p_{Z}(z)=F^{`}_{Z}(z)$.

#### 6. Important Distributions

##### 6.1 Bernoulli Distribution
$X\sim \text{Bernoulli}(\theta)$ if $\mathbb{P}(X=1)=\theta$ and $\mathbb{P}(X=0)=1-\theta$ and hence

$$
p(x)=\theta^{x}(1-\theta)^{1-x}, \text{ }x=0,1
$$

`Mean:` $\mu_{theta}=\mathbb{E}\[\theta\]=theta$.\\
`Variance:` $\text{Var}(theta) = \mathbb{E}\[(\theta-\mu_{theta})^{2}\]=\theta(1-\theta)$

##### 6.2 Binomial Distribution

$X\sim \text{Binomial}(\theta)$ if

$$
\begin{align}
p(x)=\mathbb{P}(X=x)=\binom{n}{x} \theta^{x}(1-\theta)^{n-x}, \text{ }x\in {0,...,n}
\end{align}
$$

`Mean:` $\mu_{\theta}=n\theta$.\\
`Variance:` $\text{Var}(\theta)=n\theta(1-\theta)$. (Indicated function is used to prove the mean and variance.)

##### 6.3 Multinomial Distribution

The miltivariate version of a Binomial distribution is called a Multinomial distribution. Consider drawing a ball from an urn with has balls with $k$ different colors labeled "Color 1, color 2,..., color k." Let $p=(p_{1},p_{2},...,p_{k})$ where $\sum_{j=1}^{n}p_{j}=1$ and $p_{j}$ is the probability of frawing color $j$. Draw $n$ balls from the urn(independently and with replacement) and let $X=(X_{1},...,X_{k})$ be the count of the number of balls of each color drawn. We say that $X$ has a Multinomial(n,p) distribution. Then,

$$
\begin{align}
p(x)=\binom{n}{x_{1},...,x_{n}}p_{1}^{x_{1}}...p_{k}^{x_{k}}
\end{align}
$$ 

`Mean:` $\mathbb{E}\[X_{i}\]=np_{i}$\\
`Variance:` $\text{Var}(X_{i})=np_{i}(1-p_{i})$

##### 6.4 Chi-squared Distribution

$X\sim \chi^{2}_{p}$ if $X=\sum_{j=1}^{n}Z_{j}^{2}$ where $Z_{1},...,Z_{n}\sim N(0,1)$. The pdf of $\chi$ is:

$$
\begin{align}
f(n,x)=\frac{x^{k/2-1}e^{-x/2}}{2^{k/2}\Gamma(k/2)}
\end{align}
$$

The `mean`: $\mu=n$, and the `variance:` $\text{Var}(\chi)=2n$. n is the degree of freedom. 

The cdf of $\chi$:

$$
\begin{align}
F(n,x)=\frac{1}{\Gamma(k/2)}\gamma(\frac{k}{2},\frac{x}{2})
\end{align}
$$

**Non-centeral chi-squared(More on this below).** $X\sim \chi_{1}^{2}(\mu^{2})$ if $X=Z^{2}$ where $Z\sim N(\mu,1)$.

##### 6.5 Gaussian Distribution(Normal Distribution)

$X\sim N(\mu,\sigma^{2})$ if

$$
\begin{align}
p(x)=\frac{1}{\sigma\sqrt{2\pi}}e^{-\frac{(x-\mu)^{2}}{2\sigma^{2}}}
\end{align}
$$

If $X\in \mathbb{R}^{d}$ then $X\sim N(\mu,\Sigma)$ if

$$
\begin{align}
p(x) = \frac{1}{(2\pi)^{d/2}|\Sigma|}\text{exp}(-\frac{1}{2}(x-\mu)^{T}\Sigma^{-1}(x-\mu))
\end{align}
$$
where $\mathbb{E}\[Y\]=\mu$ and $\text{cov}\[Y\]=\Sigma$. The moment generating function is

$$
M(t)=\text{exp}(\mu^{T}t+\frac{t^{T}\Sigma t}{2})
$$

**Theorem** (a). If $Y\sim N(\mu,\Sigma)$, then $\mathbb{E}\[Y\]=\mu,\text{cov}(Y)=\Sigma$.\\
(b). If $Y\sim N(\mu,\Sigma)$ and $c$ is a scalar, then $cY\sim N(c\mu,c^{2}\Sigma)$.

**Theorem** Suppose that $Y\sim N(\mu,\Sigma)$. Let \\
$$
\begin{align}
Y=\begin{pmatrix}
Y_{1} \\
Y_{2} \\
\end{pmatrix},
\mu = \begin{pmatrix}
\mu_{1} \\
\mu_{2} \\
\end{pmatrix},
\Sigma = \begin{pmatrix}
\Sigma_{11} & \Sigma_{12} \\
\Sigma_{21} & \Sigma_{22} \\
\end{pmatrix}
\end{align}
$$

where $Y_{1}$ and $\mu_{1}$ are $p\times 1$, and $\Sigma_{11}$ is $p\times p$. \\
(a). $Y_{1}\sim N_{p}(\mu_{1},\Sigma_{11}),Y_{2}\sim N_{n-p}(\mu_{2},\Sigma_{22})$\\
(b). $Y_{1}$ and $Y_{2}$ are independent if and only if $\Sigma_{12}=0$. \\
(c). If $\Sigma_{22}> 0$, then the condition distribution of $Y_{1}$ given $Y_{2}$ is

$$
Y_{1}|Y_{2}\sim N_{p}(\mu_{1}+\Sigma_{12}\Sigma_{22}^{-1}(Y_{2}-\mu_{2}), \Sigma_{11}-\Sigma_{12}\Sigma_{22}^{-1}\Sigma_{21})
$$

**Lemma:** Let $Y\sim N(\mu,\sigma^{2}I)$, where $Y^{T}=(Y_{1},...,Y_{n}),\mu^{T}=(\mu_{1},...,\mu_{n})$ and $\sigma^{2}>0$ is a scalar. Then the $Y_{i}$ are independent, $Y_{i}\sim N_{1}(\mu,\sigma^{2})$ and

$$
\frac{\Vert Y\Vert^{2}}{\sigma^{2}}=\frac{Y^{T}Y}{\sigma^{2}}\sim \chi^{2}_{n}(\frac{\mu^{T}\mu}{\sigma^{2}})
$$

**Theorem** Let $Y\sim N(\mu,\Sigma)$. Then:\\
(a). $Y^{T}\Sigma^{-1}Y\sim \chi_{n}^{2}(\mu^{T}\Sigma^{-1}\mu)$.\\
(b). $(Y-\mu)^{T}\Sigma^{-1}(Y-\mu)\sim \mu$.
(c). $(Y-\mu)^{T}\Sigma^{-1}(Y-\mu)\sim \chi_{n}^{2}(0)$.

#### 7. Sample Mean and Variance
Let $X_{1},...,X_{n}\sim P$. The sample mean is 
$$
\hat{\mu_{n}}=\frac{1}{n}\sum_{i}X_{i}
$$
and the sample variance is 
$$
\hat{\sigma_{n}^{2}}=\frac{1}{n-1}\sum_{i}(X_{i}-\hat{\mu}_{n})^{2}.
$$
The *sampling distribution* of $\hat{\mu}_{n}$ is
$$
G_{n}(t)=\mathbb{P}(\hat{\mu}_{n}\leq t)
$$
**Pratics Problem.** Let $X_{1},..,X_{n}$ be `iid` with $\mu=\mathbb{E}(X_{i})=\mu$ and $\sigma^{2}=\text{Var}(X_{i})=\sigma^{2}$. Then
$$
\mathbb{E}(\hat{\mu})=\mu,\text{Var}(\hat{\mu})=\frac{\sigma^{2}}{n},\mathbb{E}(\hat{\sigma}^{2}_{n})=\sigma^{2}
$$

**Theorem** If $X_{1},...,X_{n}\sim N(\mu,\sigma^{2})$ then \\
(a). $\hat{\mu}_{n}\sim N(\mu,\frac{\sigma^{2}}{n})$.\\
(b). $\frac{(n-1)\hat{\sigma}^{2}_{n}}{\sigma^{2}}\sim \chi^{2}_{n-1}$
(c). $\hat{\mu}_{n}$ and $\hat{\sigma}_{n}^{2}$ are independent.

**Proof:** $\mathbb{E}\[\hat{\mu}\]=\mu$\\

$$
\begin{align}
\mathbb{E}\[\hat{\mu}_{n}\] &=\frac{1}{n}\sum_{i}\mathbb{E}\[X_{i}\] \\
&=\frac{1}{n}n\mu \\
&=\mu
\end{align}
$$

**Proof:** $\mathbb{E}\[\hat{\sigma}^{2}\]=\sigma^{2}$.\\

$$
\begin{align}
    \mathbb{E}\[\hat{\sigma}^{2}\] &=\frac{1}{n}\sum_{i=1}^{n}\mathbb{E}\[(X_{i}-\hat{\mu})^{2}\] \\
    &= \frac{1}{n}\sum_{i=1}^{n}\mathbb{E}\[(X_{i}-\mu+\mu-\hat{\mu})^{2}\] \\
    &= \frac{1}{n}\sum_{i=1}^{n}\mathbb{E}\[(X_{i}-\mu)^{2}+2(X_{i}-\mu)(\mu-\hat{\mu})+(\mu-\hat{\mu})^{2}\] \\
    &= \frac{1}{n}\mathbb{E}\[\sum_{i=1}^{n}(X_{i}-\mu)^{2}\]+2n(\hat{\mu}-\mu)^{2}+n(\mu-\hat{\mu})^{2} \\
    &=\frac{1}{n}\[\text{Var}(X)-n\text{\hat{\mu}}\] \\
    & = \sigma^{2}-\frac{\sigma^{2}}{n} \\
    & = \frac{n-1}{n}\sigma^{2}
\end{align}
$$
