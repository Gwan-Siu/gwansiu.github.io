---
layout:     post
title:      "Kalman Filter"
date:       2018-12-07 12:00:00
author:     "GwanSiu"
catalog: true
tags:
    - Machine Learning
---

## 1. Introduction

Kalman filter and its variants, i.e., extended Kalman fiter, are one of classical algorithm and have been regarded as the optimal solution in tracking and data prediction task. Kalman filter is derived from minimum square error estimator. In this article, I will give a brief introduction to kalman filer based on the reference[1,2].

## 2. From HMM to Kalman Filter

Hidden markov model and kalman filter are viewed as dynamic system or state-space model. The structure of state-space model is shown as below:

<img src="https://raw.githubusercontent.com/Gwan-Siu/BlogCode/master/other/E81671EF-4D4C-416B-B9D0-883BFC2D6C77.png" width = "600" height = "400"/>

The assumption of state-space model is that, given latent variables, observed variables are conditional independent. In state-space model, we focus on `transition probability` and `emission(or measurement) probability`.

$$
\begin{align}
p(x_{t}\vert x_{t-1})&: \text{ transition probability}
p(y_{t}\vert x_{t})&: \text{ emission probability}
\end{align}
$$

there are 3 kinds of dynamic model(DM) we can summary here

||$p_{x_{t}\vert x_{t-1}}$|$p(y_{t}\vert x_{t})$|$p(x_{1})$|
| ------ | ------ | ------ |------ |
|Discrete state DM(HMM)|$A_{x_{t-1},x_{t}}$|Any|$\pi$|
|Linear Gaussian DM(Kalman Filter)|$\mathcal{N}(Ax_{t-1}+B,Q)$|$\mathcal{N}(Hx_{t}+C,R)$|$\mathcal{N}(\mu_{0},\sigma_{0})$|
|non-linear,non-gaussian,DM(particle filter)|$f(x_{t-1})$|$g(y_{t-1})$|$f_{0}(x_{1})$|

in state-space model, we are interested in 4 things:

1. **evaluation:** $p(y_{1},...,y_{t})$
2. **parameter learning:** $\displaystyle{\arg\max_{\theta}\log p(y_{1},...,y_{t}\vert \theta)}$
3. **state decoding:** $p(x_{1},...,x_{t}\vert y_{1},...,y_{t})$
4. **filtering:**$p(x_{t}\vert y_{1},...,y_{t})$



## Reference
[1]. Faragher R. Understanding the basis of the Kalman filter via a simple and intuitive derivation[J]. IEEE Signal processing magazine, 2012, 29(5): 128-132.
[2]. Lecture-Kalman filter, Richard Xu.