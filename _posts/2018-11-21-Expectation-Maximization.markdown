---
layout:     post
title:      "Expectation Maximization"
date:       2018-11-21 12:00:00
author:     "GwanSiu"
catalog: true
tags:
    - Machine Learning
---

## 1. Introduction

In this article, I will introduce Expectation Maximization(EM) algorithm. EM algorithm is introduced as early as 1950 by Ceppellini etal, and then it widely applied for parameters estiamtion of **incomplete data(known X, unkown Y)**. Our story begin with maximimum lieklihood of **complete data(known X, known Y)**.

Conventional problem of machine learning is parameters estimation problem. We take an exmaple of fipping coins. Suppose we have two coins A and B with unknown biases, $\theta_{A}$ and $\theta_{B}$ respectively. We set an experiment that we fip the coin repeatly. During the experiment, we fip the coin followed the i.i.d principle and keep track the record of the $X=\{x_{1},...,x_{N}\}$ and $Y=\{y_{1},...,y_{N}\}$, where $N$ is the number of samples, $x_{i}$ denotes the number of head we observed and y is an indicator of identity of coins. we assume these data comes from an underlying distribution $P(X,Y\vert \Theta)$, at this time, we assume bernoulli distribution. Thanks to the bayes'rule: $\displaystyle{p(\Theta\vert X, Y)\frac{p(X,Y\vert\Theta)p(\Theta)}{p(X,Y)}}$,  Our goal is estimate the parameters $\Theta$. We can adopt maximize log-likelihood estiamtion(MLE): $\log p(\Theta\vert X, Y)=\log p(X,Y\vert\Theta)$

$$
\begin{align}
\log(p(X,Y\vert \Theta))&=\log(p(X\vert Y, \Theta)p(Y\vert \Theta))\\
&=\log(\prod_{i=1}^{N}p(x_{i}\vert y_{i}, \Theta)p(y_{i}\vert\Theta)) \\
&=\sum_{i=1}^{N}\log(p(x_{i}\vert y_{i}, \Theta)p(y_{i}\vert\Theta)) \\
&=\sum_{i=1}^{N}\log(\alpha_{y_{i}}p(x_{i}\vert y_{i}, \Theta))
\end{align}
$$

where $p(y_{i}\vert\Theta)=\alpha_{y_{i}}$.

In practice, we only abtain observed data $X$ and label $Y$ is unkonwn. How to solve this kind of problem. Could we directly apply MLE to this kind of problem? 

## 2. EM Algorithm

### 2.1 Intuition of EM Algorithm

In this time, we consider a more challenging variant problem that we still we want to estimate baises of two coins A and B, we only has the record $Y$ and do not know about the $Y$, where $Y$ is called hidden parameter or latent factor. Parameter estimated problem with **complete data** is converted into parameter estimated problem with **imcomplete data**, the likelihood of data is:

$$
\begin{equation}
p(x_{i}\vert \Theta) = \sum_{i=1}^{K}\alpha_{i}p(x_{i}\vert \Theta_{\alpha_{i}})
\end{equation}
$$

at this time, $\alpha_{i}$ and $\Theta_{\alpha_{i}}$ are unknown. We can see that the MLE is not effective for the problem of imcomplete data due to the sum inside of log. This is a big limitation of MLE and thus we need another method to sovle for imcomplete data problem. 

However, if we have some way to know the value of $z$, we can use MLE to inference $\theta$ which is the same as parameter estimation with complete data. Thus, we can reduce parameter estimation for this problem with incomplete data to maximum likelihood estimation with complete data. This is the basic idea of EM algorithm. **Note: EM algorithm is proprosed to solve problems with discrete latent factors.**

One iterative scheme for obtaining completions caould works as follows: starting from some initial parameters, $\theta^{(t)}$, determine for each of the five sets whether coin $A$ or coin $B$ was more liekly to have generated the observed flips(using the current parameter estimates). Then, assume these completions (that is , guessed coin assignments) to be correct, and apply the regular maximum likelihood estimation procedure to get $\tilde{\theta}^{(t+1)}$. Finally, repeat these two steps until convergence. 

The expectation maximization algorithm is a refinement on this basic idea that the expectation maximization algorithm computes probabilities for each possible completion of the missing data, using the current parameters $\widetilde{\theta}^{(t)}$. These probabilities are used to create a weighted training set consisting of all possible completions of the data. Then, a modified version of maximum likelihood estimation that deals with weighted training examples provides new parameter estimates, $\widetilde{\theta}^{(t+1)}$. 

The figure below show the procedure of maximum likelihood estimator and expectation maximization algorithm. The figure is from the paper:[ what is the expectation maximization algorithm?](https://www.nature.com/articles/nbt1406)

<img src="https://raw.githubusercontent.com/Gwan-Siu/BlogCode/master/EM_and_VI/image/MlE_EM.png" width = "600" height = "400"/>


**Intuitively, the expectation maximization algorithm alternates between the steps of guessing a probability distribution over completions of missing data given the current model( known as the E-step). and then re-estimating the model parameters using these completions(known as the M-step).**

### 2.2 Expectation Step

Let's go further step into mathematics of EM algorithm. Our goal is to draw parameters estimation of incomplete data under the framework of MLE of complete data. That is:

$$
\begin{align}
\log(p(X\vert \Theta)) &= \log \sum_{Y}(p(X,Y\vert \Theta)) \\
&=\log \sum_{Y}p(Y\vert\Theta)p(X\vert Y, \Theta) \\
&=\log \sum_{Y}\alpha_{Y}p(X\vert Y, \Theta)
\end{align}
$$ 

we can see that $\log(p(X\vert \Theta))$ is a probability distribution due to $\alpha_{Y}$. Thus, it is hard to directly maximize a probability distribution. An alternative way is to maximize the expectation of $\log(p(X\vert \Theta))$, denote as $Q(\Theta,\Theta^{(t)})$. That is the E-step of EM algorithm:

$$
\begin{align}
Q(\Theta,\Theta^{(t)}) &= \sum_{Y\in\gamma}\log(p(\Theta\vert X,Y))p(Y\vert X,\Theta) \\
&= \sum_{Y\in\gamma}\log(\prod_{i=1}^{N}p(x_{i}\vert y_{i},\theta_{y_{i}})p(y_{i} \vert \theta_{y_{i}}))\prod_{i=1}^{N}p(y_{i}\vert x_{i},\Theta^{(t)}) \\
&= \sum_{Y\in\gamma}\sum_{i=1}^{N}\log(p(x_{i}\vert y_{i},\theta_{y_{i}})p(y_{i}\vert \theta_{y_{i}}))\prod_{i=1}^{N}p(y_{i}\vert x_{i},\Theta^{(t)}) \\
&= \sum_{y_{1}=1}^{K}...\sum_{y_{N}=1}^{K}\sum_{i=1}^{N}\log(p(x_{i}\vert y_{i}\theta)p(y_{i}\vert \theta_{y_{i}}))\prod_{j=1}^{N}p(y_{j}\vert x_{j},\Theta^{(t)}) \\
&= \sum_{y_{1}=1}^{K}...\sum_{y_{N}=1}^{K}\sum_{i=1}^{N}\log(\alpha_{y_{i}}p(x_{i}\vert y_{i}, \theta_{y_{i}}))p(y_{1},...,y_{N}\vert x_{i},\Theta^{(t)}) \\
&=\sum_{i=1}^{N}\sum_{l=1}^{K}p(l\vert x_{i}, \Theta^{(t)})\log(\alpha_{l}) + \sum_{i=1}^{N}\sum_{l=1}^{K}p(l\vert x_{i}, \Theta^{(t)})\log(p(x_{i}\vert y_{i}, \theta_{y_{i}})) \\
\end{align}
$$

### 2.3 Maximization Step

M-step is to re-estimate the hidden parameter and model parameters using complete data likelihood.

Firstly, to obtain the $\alpha_{l}$, we should take the derivative respect to $\alpha_{l}$, with the constraint is $\displaystyle{\sum_{l=1}^{K}\alpha_{l}=1}$. Thus, lagrange function is:

$$
\begin{equation}
\mathcal{L}(\alpha_{l},\lambda)= \sum_{i=1}^{N}\sum_{l=1}^{K}p(l\vert x_{i}, \Theta^{(t)})\log(\alpha_{l}) + \lambda(\sum_{l=1}^{K}\alpha_{l}-1)
\end{equation}
$$

thus, the $\frac{\partial L}{\partial \alpha_{l}}= \frac{1}{\alpha_{l}}\displaystyle{\sum_{i=1}^{N}} p(l\vert x_{i}, \Theta^{(t)})=0$, and we obtain $\alpha_{l}=\displaystyle{\frac{\sum_{i=1}^{N} p(l\vert x_{i}, \Theta^{(t)})}{-\lambda}}$. Due to $\displaystyle{\sum_{l=1}^{K}\alpha_{l}=1}$, we can easily get $\lambda=-N$.


### 2.4 Gaussian Mixture Model

Given $N$ samples $X$ and we assume all data samples come from $K$ underlyding gaussian distribution. Thus, $\Theta=\{\mu_{1},...,mu_{K}, \Sigma_{1},...,\Sigma_{K}\}$. The multivarite gaussian model is:

$$
\begin{equation}
\mathcal{N}(\mu,\Sigma)=\frac{1}{(2\pi)^{d/2}\vert\Sigma\vert^{1/2}}\text{exp}(-\frac{1}{2}(x-\mu)^{T}\Sigma^{-1}(x-\mu))
\end{equation}
$$

Follow the above analysis:

$$
\begin{equation}
Q(\Theta,\Theta^{(t)}) = \sum_{i=1}^{N}\sum_{l=1}^{K}p(l\vert x_{i}, \Theta^{(t)})\log(\alpha_{l}) + \sum_{i=1}^{N}\sum_{l=1}^{K}p(l\vert x_{i}, \Theta^{(t)})\log(p(x_{i}\vert y_{i}, \theta_{y_{i}}))
\end{equation}
$$

$$
\begin{equation}
\alpha^{(t+1)}_{l}=\frac{\sum_{i=1}^{N} p(l^{(t)}\vert x_{i}, \Theta^{(t)}_{l^{(t)}})}{N}
\end{equation}
$$

where $\displaystyle{p(l^{(t)}\vert x_{i}, \Theta^{(t)}_{l^{(t)}})=\frac{p(l^{(t)}\vert \Theta^{(t)}_{l^{(t)}})p(x_{i}\vert l^{(t)}, \Theta^{(t)}_{l^{(t)}})}{\sum_{l}^{K}}p(l^{(t)}\vert \Theta^{(t)}_{l^{(t)}})p(x_{i}\vert l^{(t)}, \Theta^{(t)}_{l})}=\frac{\pi^{(t)}_{k}\mathcal{N}(x\vert \mu_{k}^{(t)}, \Sigma_{k}^{(t)})}{\sum_{k=1}^{K}\pi^{(t)}_{k}\mathcal{N}(x\vert \mu_{k}^{(t)}, \Sigma_{k}^{(t)})}$

to abtain the $\mu$, we only focus on the second part:

$$
\begin{align}
L &=\sum_{i=1}^{N}\sum_{l=1}^{K}p(l\vert x_{i}, \Theta^{(t)})\log(p(x_{i}\vert y_{i}, \theta_{y_{i}})) \\
&= \sum_{i=1}^{N}\sum_{l=1}^{K}p(l\vert x_{i}, \Theta^{(t)})[\frac{1}{2}\log((2\pi)^{-d}\vert\Sigma\vert^{-1})+(x_{i}-\mu_{l}\Sigma^{-1}_{l}(x_{i}-\mu_{l}))]
\end{align}
$$

and set the detivative with respect with $\mu$ to 0, and we have $\sum_{i}^{N}p(l\vert x_{i}, \Theta^{(t)})(x_{i}-\mu_{l})=0$. Thus, $\mu_{l}^{new}=\frac{\sum_{i}^{N}p(l\vert x_{i}, \Theta^{(t)})x_{i}}{\sum_{i}^{N}p(l\vert x_{i}, \Theta^{(t)})}$.

To obtain $\Sigma_{l}^{new}$:

$$
\begin{align}
L &=\sum_{i=1}^{N}\sum_{l=1}^{K}p(l\vert x_{i}, \Theta^{(t)})\log(p(x_{i}\vert y_{i}, \theta_{y_{i}})) \\
&= \sum_{i=1}^{N}\sum_{l=1}^{K}p(l\vert x_{i}, \Theta^{(t)})[\frac{1}{2}\log((2\pi)^{d}\vert\Sigma\vert^{-1})+(x_{i}-\mu_{l}\Sigma^{-1}_{l}(x_{i}-\mu_{l}))] \\
&=\frac{1}{2}\sum_{i=1}^{N}\sum_{l=1}^{K}p(l\vert x_{i}, \Theta^{(t)})\log((2\pi)^{d}) +\frac{1}{2}\sum_{i=1}^{N}\sum_{l=1}^{K}p(l\vert x_{i}, \Theta^{(t)})\log\vert\Sigma\vert^{-1} - \frac{1}{2}\sum_{i=1}^{N}\sum_{l=1}^{K}p(l\vert x_{i}, \Theta^{(t)})(x_{i}-\mu_{l}\Sigma^{-1}_{l}(x_{i}-\mu_{l})
\end{align}
$$

Take the derivative respect to $\Sigma^{-1}$, and we have:

$$
\begin{align}
&=\frac{1}{2}\sum_{i=1}^{N}p(l\vert x_{i}, Theta^{(t)})(2\Sigma_{l}-\text{diag}(\Sigma_{l})-\frac{1}{2}\sum_{i=1}^{N}p(l\vert x_{i}, \Theta^{(t)})(2N_{l,i}-\text{diag}(N_{l,i}))) \\
&=\frac{1}{2}\sum_{i=1}^{N}p(l\vert x_{i}, Theta^{(t)})(2M_{l,i}-\text{diag}(M_{l,i})) \\
&= 2S-\text{diag}(S)
\end{align}
$$

where $M_{l,i}=\Sigma_{l}-M_{l,i}$ and where $S=\frac{1}{2}\sum_{i=1}^{N}p(l\vert x_{i},\Theta^{(t)})M_{l,i}$. Setting $2S-\text{diag}(S)=0$, which implies that $S=0$. This gives:

$$
\sum_{i=1}^{N}p(l\vert x_{i}, \Theta^{(t)})(\Sigma_{l}-N_{l,i})=0
$$

and we obtain: $\Sigma_{l}^{new}=\frac{\sum_{i=1}^{N}p(l\vert x_{i}, \Theta^{(t)})(x_{i}-\mu_{l})^{T}(x_{i}-\mu_{l})}{\sum_{i=1}^{N}p(l\vert x_{i}, \Theta^{(t)})}$.

Finally, we have:

$$
\begin{align}
\alpha_{l}^{(t+1)} &= \frac{\sum_{i}^{N}p(l\vert x_{i}, \Theta^{(t)})}{N} \\
\mu_{l}^{(t+1)} &= \frac{\sum_{i}^{N}p(l\vert x_{i}, \Theta^{(t)})x_{i}}{\sum_{i}^{N}p(l\vert x_{i}, \Theta^{(t)})} \\
\Sigma_{l}^{(t+1)} &= \frac{\sum_{i=1}^{N}p(l\vert x_{i}, \Theta^{(t)})(x_{i}-\mu_{l})^{T}(x_{i}-\mu_{l})}{\sum_{i=1}^{N}p(l\vert x_{i}, \Theta^{(t)})}
\end{align}
$$



### 2.4 Another View of EM Algorithm


Given observed variable $x$, unobserved variable(latent variable) $z$ and model paramters $\theta$, what we want is maximize likelihood w.r.t $\theta$:

$$
\begin{equation}
\mathcal{L}(\theta)=\log p(x\vert \theta)=\log\int p(x,z\vert \theta)\mathrm{d}z
\end{equation}
$$

where we rewrite the marginal for observed variable in terms of an integral over the joint distribution for unobserved variable. Our goal is to find a distribution $q(z)$ such that

$$
\begin{equation}
\mathcal{L}(\theta)=\log\int q(z)\frac{p(x,z\vert \theta)}{q(z)}\mathrm{d}z \geq \int q(z)\log \frac{p(x,z\vert\theta)}{q(z)} =\mathcal{F}(q,\theta)
\end{equation}
$$

where $\mathcal{F}(q,\theta)$ is functional, and is a lower bound on the log likelihood. In detail, the lower bound on the log likelihood:

$$
\begin{equation}
\mathcal{F}(q,\theta) = \int q(z)\log \frac{p(x,z\vert\theta)}{q(z)} = \int q(z)\log p(x, z\vert\theta)\mathrm{d}z + \mathcal{H}(q)
\end{equation} 
$$

where $\mathcal{H}(q)=-\int q(z)\log q(z)\mathrm{d}z$ is entropy of $q$. Thus, we can iteratively update:

**E-step:** maximize $\mathcal{F}(q,\theta)$ w.r.t distribution over hidden variables given the parameters:

$$
\begin{equation}
q^{(t+1)} =\arg\max_{q(z)} \mathcal{F}(q(z), \theta^{(t)})
\end{equation}
$$

**M-step:** maximize $\mathcal{F}(q,\theta)$ w.r.t the parameters given the hidden distribution:


<img src="https://raw.githubusercontent.com/Gwan-Siu/BlogCode/master/other/52AAF55B-C0F9-437E-B0C7-C56606E383D7.png" width = "600" height = "400"/>

$$
\begin{equation}
\theta^{(t+1)} = \arg\max_{\theta}\mathcal{F}(q,\theta) =\arg\max_{\theta}\int q^{(t+1)}(z)\log p(x,z\vert \theta)\mathrm{d}z
\end{equation}
$$

which is equivalent to optimizing the expected complete-data likelihood $p(x,z\vert \theta)$, since the `entropy of q(z)` does not depend on $\theta$.

Although we focus on the lower bound on the log likelihood, the likelihood is non-decreasing in every iteration:

$$
\begin{equation}
\mathcal{L}(\theta^{(t)})\underset{\text{E-step}}{\leq} \mathcal{F}(q^{(t+1), \theta^{(t)}})\underset{\text{M-step}}{\leq}\mathcal{F}(q^{(t+1), \theta^{(t+1)}})\leq \mathcal{L}(\theta^{(t+1)})
\end{equation}
$$

E-step usually construct the upper bound of loglikelihood, and M-step updates model parameters $\theta$. Usually, EM algorithm converges to a local minimum. Thus, **please try more times when you implement EM algorithm.**


<img src="https://raw.githubusercontent.com/Gwan-Siu/BlogCode/master/EM_and_VI/image/EMAlgorithm_19_0.png" width = "600" height = "400"/>


The difference between the log likelihood and the bound:

$$
\begin{align}
\mathcal{L}(\theta)-\mathcal{F}(q(z), \theta) &= \log p(x\vert \theta) - \int q(z)\log \frac{p(x, z\vert \theta)}{q(z)}\mathrm{d}z \\
&= \log p(x\vert \theta) - \int q(z)\log \frac{p(z\vert x, \theta)p(x\vert \theta)}{q(z)}\mathrm{d}z \\
&= -\int q(z)\log \frac{p(z\vert x, \theta)}{q(x)} \\
&= \text{KL}(q(z), p(z\vert x, \theta))
\end{align}
$$

Thus, the likelihood of variable can be decomposed into evidence bound and KL divergence between $q(z)$ and $p(z\vert x,\theta)$.

$$
\begin{equation}
\mathcal{L}(\theta) = F(q,\theta) + \text{KL}(q(z), p(z\vert x, \theta))
\end{equation}
$$

GMM is a problem of discrete latent variable. EM algoeirhm is a generalized maximimum likelihood estimation in latent variable models.







