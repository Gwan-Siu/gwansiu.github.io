---
layout:     post
title:      "Maximum Likelihood and KL Divergence"
date:       2018-10-27 12:00:00
author:     "GwanSiu"
catalog: true
tags:
    - Machine Learning
---

## 1. Maximum Likelihood

In this session, I will prove that maximum likelihood is equivalent to minimize KL divergence between true distribution $p_{data}$ and model distribution $p_{\theta}$. 

$$
\begin{align}
D_{KL}(p_{data}\Arrowvert p_{\theta}) &= \mathbb{E}_{x\sim p_{data}}[\log \frac{p_{data}}{p_{\theta}}] \\
&=\mathbb{E}_{x\sim p_{data}}[\log p_{data}]-\mathbb{E}_{x\sim p_{data}}[\log p_{\theta}] 
\end{align}
$$

suppose we have $N$ samples from $p_{data}$, i.e. $x\sim p_{data}$, then based on the law of large number, we have

$$
\begin{equation}
-\mathbb{E}_{x\sim p_{data}}[\log p_{\theta}] = -\lim_{N\rightasrrow\infty}\frac{1}{N}\log p_{\theta}
\end{equation}
$$

thus the second term of $D_{KL}(p_{data}\Arrowvert p_{\theta})$ can be formulated

$$
\begin{align}
-\mathbb{E}_{x\sim p_{data}}[\log p_{\theta}] &= -\frac{1}{N}\log p_{\theta} \\
&=\text{cNLL}
\end{align}
$$

where NLL is negative log-lolikelihood, and c is constant.

## 2. The Reverse of KL Divergence

$$
\begin{align}
D_{KL}(p_{\theta}\Arrowvert p_{data}) &= \mathbb{E}_{x\sim p_{\theta}}[\log \frac{p_{\theta}}{p_{data}}] \\
&=\mathbb{E}_{x\sim p_{\theta}}[\log p_{\theta}] - \mathbb{E}_{x\sim p_{\theta}}[\log p_{data}]
\end{align}
$$


## 3. Discussion on $D_{KL}(p_{\theta}\Arrowvert p_{data})$ and $D_{KL}(p_{data}\Arrowvert p_{\theta})$

$D_{KL}(p_{\theta}\Arrowvert p_{data})$ is defined and finite only if the support $p_{\theta}$ is contained in the support of $p_{data}$. The same to $D_{KL}(p_{data}\Arrowvert p_{\theta})$.  The difference between $D_{KL}(p_{\theta}\Arrowvert p_{data})$ and $D_{KL}(p_{data}\Arrowvert p_{\theta})$: 

1. Minimize $D_{KL}(p_{\theta}\Arrowvert p_{data})$ is to force the support of model distribution contains all example. it penalizes the model that assign a low probability mass to data sample, and likely finds a model $p_{\theta}$ that cover all modes of $p_{data}$, at the cost of placing probability mass where $p_{data}$ has none. 

2. Minimize $D_{KL}(p_{\theta}\Arrowvert p_{data})$ ensure that the support of model distribution contains the support of empirical data distribution, which is a set of $p_{data}$. It penalizes model for generating implausible data, in other words, minimize $D_{KL}(p_{\theta}\Arrowvert p_{data})$ is mode searching, the optimal $p_{\tehta}$ typically concentrate around the largest mode of $p_{\data}$, at the cost of ignoring the smaller mode of $p_{data}$.


We assume we try to model a muitlmodal $P$ with simpler, unitmodel  model $Q$, we show that figure A is the mutimodel $P$, and $B$ is the result of minimizing $D_{KL}(p_{\theta}\Arrowvert p_{data})$ and the output of minimizing $D_{KL}(p_{data}\Arrowvert p_{\theta})$.

<img src="https://raw.githubusercontent.com/Gwan-Siu/BlogCode/master/other/Selection_007.png" width = "600" height = "400"/> 


## Reference

[1] Huszár, Ferenc. "How (not) to train your generative model: Scheduled sampling, likelihood, adversary?." arXiv preprint arXiv:1511.05101 (2015).

[2] Ke Li Jitendra Malik. "Implicit Maximum Likelihood Estimation" arXiv preprint arXiv:1809.09087v2 (2018).
