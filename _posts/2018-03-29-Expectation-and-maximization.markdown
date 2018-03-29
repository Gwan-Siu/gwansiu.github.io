---
layout: post
title: Expectation and Maximization
date: 2018-03-29
author: Gwan Siu
catalog: True
tags:
    - Statistics and Bayesian Analysis
---


## 1. Introduction

In this article, exctation maximization(EM) algorithm will be introduced. EM algorithm is introduced as early as 1950 by Ceppellini etal. in the context of gene frequency estimation, the expectation maximization algorithm was nanalyzed more generally by Hartley and by Baum et al. In this article, I will begin my story with an example of flipping coins and maximum likelihood estimator(MLE). This example is from the paper [what is the expectation maximization algorithm?](https://www.nature.com/articles/nbt1406)

### 1.1 Example of flipping coins

Suppose we have two coins A and B with unknown biases, $\theta_{A}$ and $\theta_{B}$ respectively. We do the experiemt that we fip the coins repeatedly. During the experiment, we keep track the record of the $x={x_{1},x_{2},...,x_{n}}$ and $z={z_{1},z_{2},...,z_{n}}$, where $x$ indicates the number of heads we observe when we fip the coin and $z$ imply the identity of coin we use to fip. Given $x$ and $z$, we want to estimated biases $\theta$ of these two coins. Obviously, we can use MLE to estimate the biase $\theta$ based on the bays theorem:

$$\begin{equation*}
p(\theta \vert x, z)=\frac{p(\theta)p(x,z\vert \theta)}{p(x,z)}
\end{equation*}$$

where we assume the normalization term and prior term are constant, thus to maximize the posterior is equivalent to maximize the likelihood $p(x,z\vert \theta)$. That's a **commom parameter estimation problem.** This kind of estimated problem is called parameter estimated with complete data.

Now, we consider a more challenging variant problem that we still we want to estimate baises of two coins A and B, but in this time, we only has the record $x$ and do not know about the $z$, where $z$ is called hidden parameter or latent factor. Parameter estimated problem with complete data is converted into parameter estimated problem with imcomplete data, so the MLE is not effective for the problem of imcomplete data. This is a big limitation of MLE and thus we need another method to sovle for imcomplete data problem.

However, if we have some way to know the value of $z$, we can use MLE to inference $\theta$ which is the same as parameter estimation with complete data. Thus, we can reduce parameter estimation for this problem with incomplete data to maximum likelihood estimation with complete data.

One iterative scheme for obtaining completions caould works as follows: starting from some initial parameters, $\tilde{\theta}^{(t)}=(\tilde{\theta}^{(t)}_{A}$, $\tilde{\theta}^{(t)}_{B})$, determine for each of the five sets whether coin $A$ or coin $B$ was more liekly to have generated the observed flips(using the current parameter estimates). Then, assume these completions (that is , guessed coin assignments) to be correct, and apply the regular maximum likelihood estimation procedure to get $\tilde{\theta}^{(t+1)}$. Finally, repeat these two steps until convergence. 

The expectation maximization algorithm is a refinement on this basic idea that the expectation maximization algorithm computes probabilities for each possible completion of the missing data, using the current parameters $\widetilde{\theta}^{(t)}$. These probabilities are used to create a weighted training set consisting of all possible completions of the data. Then, a modified version of maximum likelihood estimation that deals with weighted training examples provides new parameter estimates, $\widetilde{\theta}^{(t+1)}$. 

The figure below show the procedure of maximum likelihood estimator and expectation maximization algorithm. The figure is from the paper:[ what is the expectation maximization algorithm?](https://www.nature.com/articles/nbt1406)

<img src="https://raw.githubusercontent.com/Gwan-Siu/BlogCode/master/EM_and_VI/image/MlE_EM.png" width = "600" height = "400"/>

**Summary**

**Intuitively, the expectation maximization algorithm alternates between the steps of guessing a probability distribution over completions of missing data given the current model( known as the E-step). and then re-estimating the model parameters using these completions(known as the M-step).**

## 2 Mathematical foundations(Convergence at intuitive level)

In the complete data case, the objective function $\log(P(x,z\vert \theta))$ has a single global optimum, which can often be found in closed form. In contrast, in the incomplete data case the function $\log(P(x\vert \theta))$ has multiple local maximum and no closed form solution.

To deal with this, the expectation maximization algorithm reduces the difficult task of optimization $\log(P(x\vert \theta))$ into a sequence of simpler optimization subproblems, whose objective functions have unique global maxima that can often be computed in closed form. These subproblems are chosen in a way that gurantees their corresponding solutions $\widetilde{\theta}^{(1)},\widetilde{\theta}^{(2)},...$ and will converge to a local optimmum of $\log(P(x\vert \theta))$.

More specifically, the expectation maximization algorithm alternates between two phases. During the E-step, expectation maximization chooses a function $g_{t}$, that lower bounds $\log(P(x\vert \theta))$ everywhere, and for which $g_{t}(\widetilde{\theta}^{(t)}=\log(P(x\vert \widetilde{\theta}^{(t)}))$. During the M-step, the expectation maximization algorithm moves to a new parameter set $\widetilde{\theta}^{(t+1)}$ that maximizes $g_{t}$. As the value of the lower-bound $g_{t}$ matches the objective function at $\widetilde{\theta}^{(t)}$, it follows that $\log(P(x\vert \widetilde{\theta}^{(t)}))=g_{t}(\widetilde{\theta}^{(t)})\leq g_{t}(\widetilde{\theta}^{(t+1)})=\log(P(x\vert \widetilde{\theta}^{(t+1)}))$. So the objective function monotonically increases during each iteration of expectation maximization.

## 3 Jenson's inequality

Jenson's inequality states that for convex function $f$, if $\lambda \in [0,1]$, then

$$
\begin{equation*}
\lambda f(x_{1})+(1-\lambda)f(x_{2}) \geq f(\lambda x_{1}+(1-\lambda)x_{2})
\end{equation*}
$$

Gerenrally, the formula above can be rewritten as:

$$
\begin{equation*}
\mathbb{E}[f(x)]\geq f(\mathbb{E}[x])
\end{equation*}
$$

The physical meaning is shown as below:

<img src="https://raw.githubusercontent.com/Gwan-Siu/BlogCode/master/EM_and_VI/image/jenson's_inequality.png" width = "600" height = "400"/>

## 4 EM Algorithm and Jensen's inequality

We re-write the complete log likelihood function by multiplication it by $\frac{q(z)}{q(z)}$, where $q(z)$ represents an arbitrary distribution for the random variable $Z$.

$$
\begin{align*}
\log p(x\vert \theta) &= \log \sum_{Z}p(z\vert \theta)p(x\vert z,\theta)\frac{q(z)}{q(z)} \\
&= \log \mathbb{E}_{q}\bigg[\frac{p(z\vert \theta)p(x\vert z, \theta)}{q(z)}\bigg] \\
&\geq \mathbb{E}_{q}\bigg[\log \frac{p(z\vert \theta)p(x\vert z, \theta)}{q(z)}\bigg] \\ 
\end{align*}
$$

**Note that this function is concave function, so the symbol of inequality changes the direction**, 

let' s $\mathcal{L}(\theta \vert q)= \mathbb{E}_{q}\bigg[\log \frac{p(z\vert \theta)p(x\vert z, \theta)}{q(z)}\bigg]$, then

$$
\begin{equation}
\mathcal{L}(\theta \vert q) = \mathbb{E}_{q}[\log p(z\vert \theta)]+\mathbb{E}_{q}[\log p(x\vert z,\theta)] -\mathbb{E}_{q}[\log q(z)] 
\end{equation}
$$

This is the objective function of EM algorithm. Note that the third term in the equation is known as entropy.

## 5 The EM algorithm

The EM algorithm proceeds by coordinate ascent. At each iteration $t$, we have update two parameters: $q^{(t)}$ and $\theta^{(t)}$.

At the E-step: we update the posterior value $q$ of the random variable given the observations while holding $\theta^{(t)}$ fixed.

$$
\begin{align*}
q^{(t+1)} &= 
\arg\max_{q} \mathcal{L}(q,\theta^{(t)}) \\ 
&= p(z\vert x, \theta^{(t)})
\end{align*}
$$

At the M-step, we update the model parameters to maximize the expected complete log likelihood function.

$$
\begin{equation}
\theta^{(t+1)} = \arg\max_{\theta} \mathcal{L}(q^{(t+1)},\theta)
\end{equation}
$$

In detail, let's see how it works:

$$
\begin{align*}
\mathcal{L}(p(z\vert x, \theta),\theta) &= \sum_{Z}p(z\vert x, \theta) \log \frac{p(x,z\vert \theta)}{p(z\vert x, \theta)} \\
&= \sum_{Z}p(z\vert x, \theta)\log \frac{p(x,z\vert \theta)p(x\vert \theta)}{p(x,z\vert \theta)} \\
&= \sum_{Z}p(z\vert x, \theta)\log p(x\vert \theta) \\
&= \log p(x\vert \theta)\sum_{Z}p(x\vert x, \theta) \\
&= \log p(x\vert \theta)
\end{align*}
$$

Therefore, as we maximize the objective function, we are also maximizing the log likelihood function.

<img src="https://raw.githubusercontent.com/Gwan-Siu/BlogCode/master/EM_and_VI/image/EMAlgorithm_19_0.png" width = "600" height = "400"/>

## 6. Code for EM

Please visit [my github](http://localhost:8888/notebooks/Documents/BlogCode/EM_and_VI/EM_Algrithm.ipynb#EM-Algorithm-and-Jensen's-inequality) 


## Reference

1. [AM207-lecture-EM](https://am207.github.io/2017/lectures/lecture23.html)
2. [stat663](https://people.duke.edu/~ccc14/sta-663/EMAlgorithm.html)


