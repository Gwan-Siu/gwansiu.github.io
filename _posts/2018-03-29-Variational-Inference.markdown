---
layout: post
title: Variational Inference
date: 2018-03-29
author: Gwan Siu
catalog: True
tags:
    - Statistics and Bayesian Analysis
---

## 1. What's variational inference?

Variational inference is an approximate technique, which is widely used in inference problem in modern statistic. As we all know, inferene problem is especially important in Bayesian statistics, which frames all inference about unknown quntities as a calculation about the posterior. In facr, posterior is usually intractable in modern Bayesian statistics. In order to sovle this kind of problem, Monte Carlo Markov Chain(MCMC) and variational inference are two main methods. MCMC is a technique of sampling, for example, Metropli-Hast algorithm and gibbs sampling, and variational inference(VI) turns inference into optimization problems. 

The core idea of VI is to posit a family of distribution and then to find the member of that family which is close to the target, where closeness is measured using the Kullback-Leibler divergence.

### 1.1 Core idea of variational inference

Suppose observed data $X$ and its latent variables $Z$, the prior of latent variables is $p(z)$ and the data likelihood is $p(x\vert z)$, in many situation, we want to inference latent variables $Z$ through posterior $p(z\vert x)$. In fact, the posterior is intractble due to the normalization term, where we have to integration over all latent variable $Z$. 

In this case, MCMC is usually adopted as an approximate computation for posterior inference. However in many situations, like when large data sets are involved or if the model is too complex, more faster approximate techniques is necessary, and VI is another strong alternative.

As in EM, we start by writing the full-data likelihood:

$$
\begin{align*}
p(x,z) = p(z)p(x\vert z)\\
p(z\vert x) = \frac{p(x,z)}{p(x)}
\end{align*}
$$

The difference with EM is that we view $z$ as parameters, not just specific parameters such as cluster membership or missing data.

Thus, we posit a family of approximate distribution $D$ over latent variables. We then try to find the member of family that minimizes the Kullback-Leibler divergence to the true posterior. This turns inference problem to optimization algorithm. 

$$
q^{*}(z) = \arg \min_{q(z)\in D} D_{KL}(q(z)\vert\vert p(z\vert x))
$$

The figure below shows the core idea of VI: to approximate posterior $p(z\vert x)$ with $q(z)$. We optimize $q(z)$ for minimal value of KL divergence.

<img src="https://raw.githubusercontent.com/Gwan-Siu/BlogCode/master/EM_and_VI/image/VI.png" width = "600" height = "400" alt="图片名称" align=center />

**Comparision of MCMC and VI**

| MCMC | VI  |
| :----------: | :---: |
| More computationally intensive | Less intensive |
| Gaurantess producing asymptotically exact samples from the target distribution | No such gaurantees |
| Slower | Faster, expecially for large data sets and complex distributions |
| Best for precise inference | Useful to explore many scenarios quickly or large data sets |

### 1.2 How does variational inference work?

The goal of the VI is:
    
$$
\begin{equation}
q^{*}(z) = \arg \min_{q(z)\in D}D_{KL}(q(z)\vert \vert p(z\vert x))
\end{equation}
$$

In fact, to minimize the $D_{KL}(q(z)\vert \vert p(z\vert x))$ is equivalent to maximize ELBO.

$$
\begin{aligned}
D_{KL} &= \mathbb{E}_{q(z)}[\log q(z)] -\mathbb{E}_{q(z)}[\log p(z\vert x)] \\
&= \mathbb{E}_{q(z)}[\log q(z)]-\mathbb{E}_{q(z)}[\log p(x,z)] + \mathbb{E}_{q(z)}[\log p(x)] \\
\Rightarrow \log p(x) &= \mathbb{E}_{q(z)}[\log p(x,z)] -\mathbb{E}_{q(z)}[\log q(z)] + D_{KL}(q(z)\vert \vert p(z\vert x))  \\
\log p(x) &= \mathrm{ELBO}(q)+ D_{KL}(q(z)\vert \vert p(z\vert x))
\end{aligned}
$$

where $\mathrm{ELBO}(q)= \mathbb{E}_{q(z)}[\log p(x,z)] -\mathbb{E}_{q(z)}[\log q(z)]$, which is called evidence lower bound, and $\mathbb{E}_{q(z)}[\log p(x)] =\log p(x)$  because $p(x)$ is independent with $q(z)$. 

The figure is like the one in EM, but the difference with EM is the likelihood $\ln p(X\vert\theta)$ is fixed. The goal of VI is to minimize the KL divergence of $q$ and $p$, thus it's equivalent to maximize the ELBO $\mathcal{L}(q,\theta)$.

<img src="https://am207.github.io/2017/wiki/images/klsplitup.png" width = "600" height = "400" alt="图片名称" align=center />


Therefore, we can turn our goal to an optimization problem: maximize $\mathrm{ELBO}$ is equivalent to minimize the DL divergence. Our objective function is $\mathrm{ELBO}$:

$$
\begin{align}
\mathrm{ELBO} &= \mathbb{E}_{q(z)}[\log p(x,z)] - \mathbb{E}_{q(z)}[\log q(z)] \\
&= \mathbb{E}_{q(z)}[\log p(x\vert z)]+\mathbb{E}_{q(z)}[\log p(z)]-\mathbb{E}_{q(z)}[\log q(z)] \\
&= \mathbb{E}_{q(z)}[\log p(x\vert z)]-D_{KL}(q(z)\vert p(z))
\end{align}
$$

1. The first term is the expectation of the data likelihood and thus $\mathrm{ELBO}$ encourage distributions put their mass on configurations of latent variables that explain observed data. 
2. The second term is the negative KL divergence between the variational distribution and the prior, so the $\mathrm{ELBO}$ force $q(z)$ to close to the prior $p(z)$.

**Hence, maximize** $\mathrm{ELBO}$ ** means to balance the likelihood and prior.**

## 2. Mean-field Theory

In the section 1, we know the core idea of VI and know the goal of VI is to maximize the $\mathrm{ELBO}$. The sucessive question is how to choose the proposed family of distribution. Intuitively, the complexity of family of distributions we choose directly determine the complexity of the optimization problem. 

**The more flexibility in the family of distributions, the closer the approximation and the harder the optimization.**

The priciple that we choose the family of distribution is **mean-field theory**.  **What's the mean-field theory?** the latent variables are mutually independent and each governed by a distinct factor in the variational distribution. A generic member of the mean-field variational family is given by the below equation-

$$
\begin{equation}
q(z) = \prod_{j=1}^{m}q(z_{j})
\end{equation}
$$

the latent variable in mean-field theory is mutually independent, so it cannot capture the correlation in the original space. Once the latent variable of the posterior is dependent, the mean-field approximate will be affected. The example is below:

<img src="http://7xpqrs.com1.z0.glb.clouddn.com/Fu9ZVDbU07MHvRwdhShbD7NisdZ4" width = "600" height = "400" alt="图片名称" align=center />

**Notice that we are not making any comment about the conditional independence or lack thereof of posterior parameters. We are merely saying we will find new functions of these parameters such that we can multiply these functions to construct an approximate posterior.**

### 3. CAVI Algorithm

Now, we have goal(maximize $\mathrm{ELBO}$) and choose mean-field family of distribution $q(z)=\prod_{i=1}^{M} q(z_{i})$. **The next question is how to maximize $\mathrm{ELBO}$?** At this time, coordinate ascent variational inference(CAVI) algorithm is introduced.

The figure below is the process of CAVI algorithm:

<img src="https://raw.githubusercontent.com/Gwan-Siu/BlogCode/master/EM_and_VI/image/CAVI.png" width = "600" height = "400" alt="图片名称" align=center />

First of all, we compute the optimal distribution $q_{i}$:
    
$$
\begin{align}
q &= \mathbb{E}[p(\cdot)] \\
&= exp(\log \mathbb{E}[p(\cdot)]) \\
&\approx exp(\log \mathbb{E}[p(\cdot)]-Var(p(\cdot))/(2\ast \mathbb{E}[p(\cdot)]^{2}) \\
&= exp(\log \mathbb{E}[p(\cdot)])\ast exp(h(p(\cdot))) \\
&< exp(\log \mathbb{E}[p(\cdot)])
\end{align}
$$

Apply second order Taylor expansion, we have $q=exp(\log \mathbb{E}[p(\cdot)])\ast exp(h(p(\cdot)))$, and $q^{\ast}_{j}(z_{j})=\mathbb{E}[p(z_{j}\vert z_{-j}, x)]$, hence $q_{j}$ is propotional to the log conditional expectation:

$$
\begin{equation}
q_{j}^{\ast}(z_{j}) \propto exp(\mathbb{E}_{-j}[\log p(z_{j}\vert z_{-j},x)])
\end{equation}
$$

Due to that the latent variable are multually independent, the RHS can be rewritten as:

$$
\begin{align}
\mathbb{E}_{-j}[\log p(z_{j}\vert z_{-j}, x)] &= \mathbb{E}_{-j}[\log \frac{p(z_{j}, z_{-j}\vert x)}{p(z_{-j})}] \\
&= \mathbb{E}_{-j}[\log p(z_{j}, z_{-j}\vert x)]-E_{-j}[\log p(z_{-j})] \\
&= \mathbb{E}_{-j}[\log p(z_{j}, z_{-j}, x)] - \mathbb{E}[\log p(z_{-j})]-E_{-j}[\log p(x)] \\
&= \mathbb{E}_{-j}[\log p(z_{j}, z_{-j}, z)-const]
\end{align}
$$

Hence, we can conclude that once we know the log expectation of joint distribution, we obatain the optimal variational distribution $q_{j}^{\ast}$:

$$
\begin{align}
q_{j}^{\ast} &\propto exp(\mathbb{E}[\log p(z_{j}, z_{-j}, x)]-const) \\
q_{j}^{\ast}(z_{j}) &\propto exp(\mathbb{E}_{-j}[\log p(z_{j}, z_{-j}, x)])
\end{align}
$$

最后，在平均场变分分布族的假设下，ELBO可以被分解为对每一个隐变量zj的函数。根据隐变量分解后的ELBO中，利用q分布的平均场性质，第一项将联合概率的期望迭代求出，第二项分解了变分分布的期望。当我们最大化qj时，也就最大化了分解后的ELBO。

$$
\begin{align}
\mathrm{ELBO}(q) &= \mathbb{E}_{q(z)}[\log p(z,x)] - \mathbb{E}_{q(z)}[\log q(z)] \\
\mathrm{ELBO}(q_{j}) &= \mathbb{E}_{j}[\mathbb{E}_{-j}[p(z_{j},, z_{-j}, x)]]- \mathbb{E}_{j}[\log q_{j}(z_{j})] + const
\end{align}
$$

## 4. Variational inference and GMM

In this section, CAVI algorithm is used for Mixture of Gaussians model(GMM). It will be helpful to understand how CAVI works.

###  4.1 Joint distribution computation

Given observed data $X=(x_{1},...,x_{n})$ from $K$ independent gaussian distribution with mean $\mu_{k}$. One-hot vector $c_{i}\in \mathbb{R}^{k}$ indicate the distribution to which each data belong. The hyperparameter $\sigma^{2}$ is fixed. latent variables are $\mu, c$. The prior is:

$$
\begin{align}
\mu_{k} \sim \mathcal{N}(0, \sigma^{2}) \\
c_{i} \sim Categorical(\frac{1}{K},...,\frac{1}{K}) \\
x_{i} \sim \mathcal{N}(c_{i}^{T}, \mu, 1)
\end{align}
$$

According to bay theroem, we can compute the joint distribution:

$$
\begin{align}
p(\mu, c,x) &= p(\mu)p(c,x\vert \mu) \\
&= p(\mu)p(c)p(x\vert c, \mu) \\
&=p(\mu)\prod_{i=1}^{n}p(c_{i})p(x_{i}\vert c_{i}, \mu)
\end{align}
$$

once we have joint distribution, we can compute marginal distribution. However, the formulation has no analytical solution, and the computational complexity is $\mathcal{O}(K^{n})$.

$$
\begin{align}
p(x) &= \int \sum_{c}p(\mu, c, x)\mathrm{d}\mu \\
&= \int p(\mu)\prod_{i=1}^{n}\sum_{c_{i}}p(c_{i})p(x_{i}\vert c_{i}, \mu)\mathrm{d}\mu
\end{align}
$$

### 4.2 GMM and CAVI

Now, we should compute variational ditribution $q(z)$, where $m=(m_{1},...,m_{k}), s^{2}=(s_{1}^{2},...,s_{K}^{2}), \phi=(\phi_{1},...,\phi_{n})$ are variational parameters, hence the formulation of variational distribution is:

$$
\begin{align}
q(z) &= q(\mu,c) \\
&= \prod_{k=1}^{K}q(\mu_{k};m_{k},s_{k}^{2})\cdot \prod_{i=1}^{n}q(c_{i};\phi_{i})
\end{align}
$$

- 1. we can obtain the formulation $\mathrm{ELBO}$, which is a function of $m,s^{2},\phi$.

$$
\begin{align}
\mathrm{ELBO}(q) &= \mathrm{ELBO}(m, s^{2}, \phi) \\
&= \mathbb{E}_{q(z)}[\log p(x\vert z)] + \mathbb{E}_{q(z)}[\log p(z)]-\mathbb{E}_{q(z)}[\log q(z)] \\
&= \sum_{i=1}^{n} \mathbb{E}[\log p(x_{i}\vert c_{i}, \mu;\phi_{i}, m, s^{2})] +(\sum_{k=1}^{K}\mathbb{E}[\log p(\mu_{k};m_{k},s^{2}_{k})]+\sum_{i=1}^{n}\mathbb{E}[\log p(c_{i};\phi_{i})]) \\
&- (\sum_{k=1}^{K}\mathbb{E}[\log q(\mu_{k};m_{k}, s_{k}^{2})]+\sum_{i=1}^{n}\mathbb{E}[\log q(c_{i};\phi_{i}])
\end{align}
$$

- 2. from section 3, we obtain how CAVI algorithm update latent variables. Now, we applyied it into GMM to compute **cluster indicator** $c$ and update $c$, noted $\mu$ is fixed:

    $$
    \begin{align}
        q^{\ast}_{j} &\propto exp(\mathbb{E}_{\mu}[\log p(c_{i}, \mu, x_{i})]) \\
        &\propto exp(\mathbb{E}[\log p(x_{i}\vert c_{i}, \mu)\cdot \log p(c_{i},\mu)]) \\
        &\propto exp(\mathbb{E}_{\mu}[\log p(x_{i}\vert c_{i}, \mu)]+\mathbb{E}_{\mu}[\log p(c_{i}, \mu)]) \\
        &\propto exp(\mathbb{E}_{\mu}[\log p(c_{i}\vert c_{i},\mu)]+\log p(c_{i}))
    \end{align}
    $$

the second term $\log p(c_{i})$ is log prior and it is a constant. Hence, we pay our attention to the first term: the distribution of $c_{i}$ gaussian distribution. In detail, we simplify it due to $c_{i}=(c_{i1},...,c_{ik})$ is one-hot vector, and we have:

$$
\begin{align}
\mathbb{E}[\log p(x_{i}\vert c_{i},\mu)] &= \sum_{k}c_{ik}\mathbb{E}_{\mu_{k}}[\log p(x_{i}\vert \mu_{k})] \\
&= \sum_{k}c_{ik}\mathbb{E}_{\mu_{k}}[-\frac{(x-\mu_{k})^{2}}{2}]+const \\
&= \sum_{k}c_{ik}(\mathbb{E}_{\mu_{k}}[\mu_{k}]x_{i}+\mathbb{E}_{\mu_{k}}[\mu_{k}^{2}]/2) + const
\end{align}
$$

from the formulation above, $\mathbb{E}[\mu_{k}]$ and $\mathbb{E}[\mu_{k}^{2}]$ can be computed. For each data point $i$, parameter $\phi_{ik}$ in the $k$th component of the latent variable $c$. The updated formulation is:

$$
\begin{equation}
\phi_{ik}\propto exp(\mathbb{E}[\mu_{k}]x_{i}+\mathbb{E}[\mu_{k}^{2}]/2)
\end{equation}
$$

smiliarly, we can compute latent variable $\mu$ of GMM. Firstly, we should calculate the optimal variational distribution $q(\mu_{k})$, and the update the parameter $m_{k}, s_{k}^{2}$ of $\mu_{k}$:

$$
\begin{align}
m_{k} &= \frac{\sum_{i}\phi_{ik}\cdot x_{i}}{1/\sigma^{2}+\sum_{i}\phi_{ik}} \\
s_{k}^{2} &= \frac{1}{1/\sigma^{2}+\sum_{i}\phi_{ik}}
\end{align}
$$

The algorithm of GMM and CAVI is below:

<img src="https://raw.githubusercontent.com/Gwan-Siu/BlogCode/master/EM_and_VI/image/GMMCAVI.png" width = "600" height = "400" alt="图片名称" align=center />

Reference

1. [AM207-Lecture-Variational-Inference](https://am207.github.io/2017/lectures/lecture24.html)