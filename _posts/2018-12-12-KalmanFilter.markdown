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
p(x_{t}\vert x_{t-1})&: \text{ transition probability} \\
p(y_{t}\vert x_{t})&: \text{ emission probability}
\end{align}
$$

in state-space model, we are interested in 4 things:

1. **evaluation:** $p(y_{1},...,y_{t})$
2. **parameter learning:** $\displaystyle{\arg\max_{\theta}\log p(y_{1},...,y_{t}\vert \theta)}$
3. **state decoding:** $p(x_{1},...,x_{t}\vert y_{1},...,y_{t})$
4. **filtering:**$p(x_{t}\vert y_{1},...,y_{t})$

there are 3 kinds of dynamic model(DM) we can summary here

||$p({x_{t}\vert x_{t-1}})$|$p(y_{t}\vert x_{t})$|$p(x_{1})$|
| ------ | ------ | ------ |------ |
|Discrete state DM(HMM)|$A_{x_{t-1},x_{t}}$|Any|$\pi$|
|Linear Gaussian DM(Kalman Filter)|$\mathcal{N}(Ax_{t-1}+B,Q)$|$\mathcal{N}(Hx_{t}+C,R)$|$\mathcal{N}(\mu_{0},\sigma_{0})$|
|non-linear,non-gaussian,DM(particle filter)|$f(x_{t-1})$|$g(y_{t-1})$|$f_{0}(x_{1})$|

Different from hidden markov model, in which the latent variables are discrete, in kalman fiter, latent variables are continuous, and is linear gaussian.

In this article, for kalman fiter, we focus on filtering,$p(x_{t}\vert y_{1},...,y_{t})$. **The purpose of filtering is to extract information from a signal, and ignore everything else.**

## 3. Kalman Filter(linear gaussian dynamic system)

the transition probability is

$$
\begin{align}
p(x_{t}\vert p_{t-1})&\sim \mathcal{N}(Ax_{t-1}+B,Q_{t})\\
x_{t}=Ax_{t-1}+B+\omega,&\quad \omega\sim\mathcal{N}(0,Q_{t})
\end{align}
$$ 

the measurement probability is

$$
\begin{align}
p(y_{t}\vert x_{t})&\sim\mathcal{N}(Hx_{t},R_{t}) \\
y_{t}=Hx_{t}+\nu,&\quad \nu\sim\mathcal{N}(0, R_{t})
\end{align}
$$

Our goal is to do the filtering $p(x_{t}\vert y_{1},...,y_{t})$. In kalman filter, it contains two stages: **prediction** and **update**.


$$
\begin{align}
\text{Prediction}&: \quad p(x_{t}\vert y_{1},...,y_{t-1})\\
\text{Update}&:\quad p(x_{t}\vert y_{1},...,y_{t})=\displaystyle{\frac{p(y_{t}\vert x_{t})p(x_{t}\vert y_{1},...,y_{t-1})}{\int_{x_{t}}p(y_{t}\vert x_{t})p(x_{t}\vert y_{1},...,y_{t-1})\mathrm{d}x_{t}}}\propto p(y_{t}\vert x_{t})p(x_{t}\vert y_{1},...,y_{t-1})
\end{align}
$$

actuall, we can expand the term $p(x_{t}\vert y_{1},...,y_{t-1})$

$$
\begin{align}
p(x_{t}\vert y_{1},...,y_{t-1}) &= \displaystyle{\int_{x_{t-1}}p(x_{t},x_{t-1}\vert y_{1},...,y_{t-1})\mathrm{d}x_{t-1}} \\
&=\displaystyle{\int_{x_{t-1}}p(x_{t}\vert x_{t-1},y_{1},...,y_{t-1})p(x_{t-1}\vert y_{1},...,y_{t-1})\mathrm{d}x_{t-1}} \\
&=\displaystyle{\int_{x_{t-1}}p(x_{t}\vert x_{t-1})p(x_{t-1}\vert y_{1},...,y_{t-1})\mathrm{d}x_{t-1}}
\end{align}
$$

where the term $p(x_{t}\vert x_{t-1})p(x_{t-1}\vert y_{1},...,y_{t-1})$ is the prediction term at time $t-1$. Thus, we can see that we can adopt filtering in recursive procedure.

from above analysis, $p(x_{t}\vert y_{1},...,y_{t})$ must be gaussian distribution because $p(y_{t}\vert x_{t})$ and $p(x_{t}\vert y_{1},...,y_{t-1})$ are gaussian distribution. Thus, we need to know to $\hat{\mu_{t}}$ and $\hat{\Sigma_{t}}$. We the below idea to derive them

1. We obtain the mean and variance of $p(y_{t}\vert x_{t})$ and $p(x_{t}\vert y_{1},...,y_{t-1})$.
2. We can find the conditional distribution through their joint distribution.

**Rewrite the formulation**

At this time, zero-mean variable $\nabla x_{t-1}=x_{t-1}-\mathbb{E}[x_{t-1}],\nabla x_{t-1}\sim\mathcal{N}(0,\hat{\Sigma}^{-1})\Rightarrow x_{t-1}=\nabla x_{t-1}+\mathbb{E}[x_{t-1}]$.

$$
\begin{align}
x_{t}&=Ax_{t-1}+B+\omega,\quad \omega\sim\mathcal{N}(0,Q_{t}) \\
\Rightarrow x_{t}&=A(\nabla x_{t-1}+\mathbb{E}[x_{t-1}]) +\omega_{t} \\
&=A\mathbb{E}[x_{\tau-1}] + A\nabla x_{t-1} +\omega_{t} \\
&=A\mathbb{E}[x_{t-1}] + \nabla x_{t}
\end{align}
$$

$$
\begin{align}
y_{t}&=Hx_{t}+\nu_{t},\quad \nu_{t}\sim\mathcal{N}(0,R_{t})\\
\Rightarrow y_{t}&=Hx_{t}+\nu_{t} \\
&=H(A\mathbb{E}[x_{t-1}] + A\nabla x_{t-1} +\omega_{t}) +\nu_{t}\\
&=HA\mathbb{E}[x_{t-1}] + HA\nabla x_{t-1} +H\omega_{t} +\nu_{t}\\
&=HA\mathbb{E}[x_{t-1}] + \nabla y_{t}
\end{align}
$$

therefore, we can see

$$
\begin{align}
p(x_{t}\vert y_{1},...,y_{t-1})&=\mathcal{N}(A\mathbb{E}[x_{t-1}],\mathbb{E}[(\nabla x)(\nabla x)^{T}])\\
p(y_{t}\vert y_{1},...,y_{t})&=\mathcal{N}(HA\mathbb{E}(x_{t-1}), \mathbb{E}[(\nabla y_{t})(\nabla y_{t})^{T}])
\end{align}
$$

to derive the covariance, we assume $\mathcal{COV}(x_{t-1},\omega_{t})=0,\mathcal{COV}(x_{t-1},\nu_{t})=0,\mathcal{COV}(\omega_{t},\nu_{t})=0$.

$$
\begin{align}
\mathbb{E}[(\nabla x)(\nabla x)^{T}]&=\mathbb{E}[(A\nabla x_{t-1} +\omega_{t})(A\nabla x_{t-1} +\omega_{t})^{T}]\\
&=A\hat{\Sigma}_{t-1}A^{T} + Q_{t} \\
&=\bar{\Sigma}_{t} \\
\mathbb{E}[(\nabla y)(\nabla x)^{T}]&=\mathbb{E}[(HA\nabla x_{t-1} +H\omega_{t} +\nu_{t})(A\nabla x_{t-1} +\omega_{t})^{T}]\\
&=H(A\hat{\Sigma}_{t-1}A^{T} + Q_{t}) \\
&=H\bar{\Sigma}_{t}\\
\mathbb{E}[(\nabla x)(\nabla y)^{T}]&=\bar{\Sigma}_{t}H^{T} \\
\mathbb{E}[(\nabla y)(\nabla y)^{T}]&=\mathbb{E}[(HA\nabla x_{t-1} +H\omega_{t} +\nu_{t})(HA\nabla x_{t-1} +H\omega_{t} +\nu_{t})^{T}] \\
&=H(A\nabla x_{t-1} +\omega_{t})H^{T}+R_{T} \\
&=H\bar{\Sigma}_{t}H^{T}+R_{T}
\end{align}
$$

thus we can obtain that 

$$
\begin{equation}
p(x_{t},y_{t}\vert y_{1},...,y_{t-1})\sim\mathcal{N}(\hat{\mu}_{t},\hat{\Sigma}_{t})
\end{equation}
$$

where 

$$
\begin{equation}
\hat{\mu}=\left[\begin{matrix}
A\mathbb{E}[x_{t-1}]\\
HA\mathbb{E}[x_{t-1}]
\end{matrix}
\right]
\end{equation}
$$

$$
\begin{equation}
\hat{\Sigma}_{t}=\left[\begin{matrix}
\bar{\Sigma}_{t} & H\bar{\Sigma}_{t}\\
\bar{\Sigma}_{t}H^{T} & H\bar{\Sigma}_{t}H^{T}+R_{T}
\end{matrix}
\right]
\end{equation}
$$

we can obtain $p(x_{t},y_{t}\vert y_{1},...,y_{t-1})\rightarrow p(x_{t}\vert y_{1},...,y_{t})$ by

$$
\begin{equation}
p(u\vert v)\sim\mathcal{N}(\mu_{u}+\Sigma_{uv}\Sigma_{v}^{-1}(v-\mu_{v}),\Sigma_{u}-\Sigma_{uv}\Sigma^{-1}_{v}\Sigma_{uv})
\end{equation}
$$

where $p(u)\sim\mathcal{N}(\mu_{u},\Sigma_{u})$ and $p(v)\sim\mathcal{N}(\mu_{v},\Sigma_{v})$.

from the analysis of conditonal gaussian distribution, we have:

$$
\begin{align}
\mathbb{E}[p(x_{t}\vert y_{1},...,y_{t})] &=\hat{\mu}\\
&=\mu_{u}+\Sigma_{uv}\Sigma_{vv}^{-1}(v-\mu_{v}) \\
&=\mathbb{E}[x_{t}] + \mathbb{E}[\nabla x_{t}(\nabla y_{t})^{T}]\mathbb{E}[\nabla y_{t}(\nabla y_{t})^{T}]^{-1}(y_{t}-\mathbb{E}[y_{t}]) \\
&=A\hat{\mu}_{t-1}+\bar{\Sigma}_{t}^{T}H(H\bar{\Sigma}_{t}H^{T}+R_{t})^{-1}(y_{t}-HA\hat{\mu}_{t-1})} \\
\mathbb{COV}[p(x_{t}\vert y_{1},...,y_{t})]&=\hat{\Sigma_{t}} \\
&=\mathbb{E}[\nabla x_{t}(\nabla x_{t})^{T}]-\mathbb{E}[\nabla x_{t}(\nabla y_{t})^{T}]\mathbb{E}[\nabla y_{t}(\nabla y_{t})^{T}]^{-1}\mathbb{E}[\nabla y_{t}(\nabla x_{t})^{T}] \\
&=\bar{\Sigma}_{t}-\bar{\Sigma}_{t}H^{T}(H(\bar{\Sigma}_{t})H^{T}+R_{t})^{-1}H\bar{\Sigma}_{t} \\
&=(I-KH)\bar{\Sigma}_{t}
\end{align}
$$

## 4. Extended Kalman Filter(non-linear gaussian system)

Kalman filter: linear gaussian dynamic system

$$
\begin{align}
x_{t}&=Ax_{t-1} + B +\omega\quad\omega\sim\mathcal{N}(0,Q_{t})\\
y_{t}&=Hx_{t}+\nu\quad\nu\sim\mathcal{N}(0,R_{t})
\end{align}
$$

Extended kalman filter: non-linear gaussian dynamic system

$$
\begin{align}
x_{t}&=F(x_{t-1})+\omega_{t}\quad\omega_{t}\sim\mathcal{N}(0,Q_{t})\\
y_{t}&=H(x_{t-1})+\nu_{t}\quad\nu_{t}\sim\mathcal{N}(0,R_{T})
\end{align}
$$

we expand $F(x_{t-1})$ around a particular point $x_{t-1}^{p}$:

$$
\begin{equation}
x_{t}=F(x_{t-1}^{p}) + F^{\prime}(x_{t-1}^{p})(x_{t-1}-x_{t-1}^{p}) + \Omega(x_{t-1}^{p}) +\omega_{t}
\end{equation}
$$

where $\Omega(x_{t-1}^{p})$ is higher order term, let $J_{p}=F^{\prime}(x_{t-1}^{p})$:

$$
\begin{align}
x_{t}&=J_{p}x_{t-1}+(F(x_{t-1}^{p})-J_{p}x_{t-1}^{p})+ \Omega(x_{t-1}^{p}) +\omega_{t}\\
&\approx Ax_{t-1} + B + \omega_{t}
\end{align}
$$

### 4.1 Transition probability-extended kalman filter

kalman filter: $x_{t}=Ax_{t-1}+\omega_{t}\quad\omega_{t}\sim\mathcal{N}(B,Q_{t})$

**Mean:** $\bar{\mu}_{t}=\mathbb{E}[x_{t}\vert y_{1},...,y_{t-1}]=A\hat{\mu_{t-1}}+B$

**Covariance**$\bar{\Sigma}_{t}=\mathbb{E}[(\nabla x_{t})(\nabla x_{t})^{T}]=A\hat{\Sigma}_{t-1}A^{T}+Q_{t}$


Extended kalman filter: $x_{t}\approx Ax_{t-1} + \omega_{t}\quad \omega_{t}\sim\mathcal{N}((F(x_{t-1}^{p})-J_{p}x_{t-1}^{p}), Q_{t})$.

**Mean:** $\bar{\mu}_{t}=\mathbb{E}[x_{t}\vert y_{1},...,y_{t-1}]=J_{p}x_{t-1}+(F(x_{t-1}^{p})-J_{p}x_{t-1}^{p})$

**Covariance:** $\bar{\Sigma}_{t}=\mathbb{E}[(\nabla x_{t})(\nabla x_{t})^{T}]=J_{p}\hat{\Sigma}_{t-1}J_{p}^{T}+Q_{t}$

because $x_{t-1}^{p}$ is arbitray point in time $t-1$, we can set $x_{t-1}^{p}=\hat{\mu}_{t-1}$ and simplified the formulation

**Mean:** $\bar{\mu}_{t}=\mathbb{E}[x_{t}\vert y_{1},...,y_{t-1}]=J_{p}x_{t-1}+(F(\hat{\mu}_{t-1})-J_{p}\hat{\mu}_{t-1})$

**Covariance:** $\bar{\Sigma}_{t}=\mathbb{E}[(\nabla x_{t})(\nabla x_{t})^{T}]=F^{\prime}(\hat{\mu}_{t-1})\hat{\Sigma}_{t-1}F^{\prime}(\hat{\mu}_{t-1})^{T}+Q_{t}$

### 4.2 Measurement probability-extended kalman filter

**Measurement Equation:** $y_{t}=H(x_{t})+\nu_{t}\quad \nu_{t}\sim\mathcal{N}(0,R_{t})$

$$
\begin{equation}
y_{t}=H(x_{t}^{p}) + H^{\prime}(x_{t}^{p})(x_{t}-x_{t}^{p})+\Omega + \nu_{t}
\end{equation}
$$

let $J_{p}=H^{\prime}(x_{t}^{p})$:

$$
\begin{align}
y_{t} &= H(x_{t}^{p}) + J_{p}(x_{t}-x_{t}^{p})+\Omega+\nu_{t} \\
&=H(\bar{x}_{t}) + J_{p}(x_{t}-\bar{x}_{t}) + \nu_{t},\quad x_{t}^{p}=\bar{x}_{t} 
\end{align}
$$

thus, we have

$$
\begin{equation}
\underbrace{y_{t}-H(\bar{x}_{t})+J_{p}\bar{x}_{t}}_{\mathbb{Y}_{t}_}\approx \underbrace{J_{p}x_{t}}_{H}+\nu_{t}
\end{equation}
$$






## Reference
[1]. Faragher R. Understanding the basis of the Kalman filter via a simple and intuitive derivation[J]. IEEE Signal processing magazine, 2012, 29(5): 128-132. 
[2]. Lecture-Kalman filter, Richard Xu.
[3].