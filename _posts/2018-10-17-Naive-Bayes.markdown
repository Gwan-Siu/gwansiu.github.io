---
layout:     post
title:      "Naive Bayes"
date:       2018-06-17 12:00:00
author:     "GwanSiu"
catalog: true
tags:
    - Machine Learning
---

## 1. Introduction

Naive Bayes is a classical supervised learning method, which is simple but effective. Usually, given a dataset $X=\{x_{1},...,x_{N}\}$ and its label $Y=\{y_{1},...,y_{K}\}$, where $N$ denotes the number of data and $K$ is the number of classes, $x_{i}\in\mathbb{R}^{N},y\in\mathbb{R}$. we build a parametric model $p(y\vert x,\theta)$ to simulate the generation of data, and our goal is to estimate paramters $\theta$ by bayes theorem: $\displaystyle{p(y\vert x)=\frac{p(x\vert y)p(y)}{p(x)}}$.

In fact, maximum likelyhood estimation(MLE) and maximum a-posterior estimation are commomly adopted for paramters estimation. Additionally, variational method is more advanced method for parameter estimation, which is beyond our scope.

$$
\begin{equation}
\mathbf{\theta}^{\text{MLE}} =\arg\max_{\mathbf{\theta}}\prod_{i=1}^{N}p(x_{i}\vert \mathbf{\theta})
\end{equation}
$$

$$
\begin{equation}
\mathbf{\theta}^{\text{MAP}}=\arg\max_{\mathbf{\theta}}\prod_{i=1}^{N}p(x_{i}\vert \mathbf{\theta})p(\theta)
\end{equation}
$$

Usually, the value of $\mathbf{\theta}^{\text{MLE}}$ and $\mathbf{\theta}^{\text{MAP}}$ are not the same.

Our story begins with bayes theorem:

$$
\begin{equation}
p(Y=y_{k}\vert \mathbf{x}_{i})=\frac{p(\mathbf{x}_{i}\vert Y=y_{k})p(Y=y_{k})}{p(\mathbf{x}_{i})}=\frac{p(x_{i}^{1},x_{i}^{2},...,x_{i}^{n}\vert Y=y_{k})}{p(\mathbf{x}_{i})}
\end{equation}
$$

In naive bayes, we assume the attribute of $x_{i}$ is **conditional independent.** Thus, the formulation above can be written as:

$$
\begin{equation}
\displaystyle{p(Y=y_{k}\vert \mathbf{x}_{i})=\frac{\prod_{i}^{N}P(x_{i}\arrowvert Y=y_{k})P(Y=y_{k})}{\sum_{j}^{K}P(Y=y_{j})\prod_{i}^{N}P(x_{i}\arrowvert Y=y_{j})}}
\end{equation}
$$

In the next Seesion, variant models of naive bayes are introduced.

## 2. Model of Naive Bayes

#### 2.1 Bernoulli Naive Bayes

**Suppose:** Binary vectors of length $K$: $x\in \{0,1\}^{K}$.

**Generative story:**

1. $Y\sim \text{Bernoulli}(\Phi)$, binary classification.
 
$$\begin{equation}
p(y)=\left\begin{cases} \Phi\quad y=1\\
1-\Phi \quad y=0
\end{cases}
\end{equation}
$$


2. $X_{K}\sim \text{Bernoulli}(\theta_{k,Y}),\forall k\in\{1,...,K\}$.

**Model:**

$$
\begin{align}
p_{\phi,\theta}(x,y) &= p_{\phi,\theta}(x_{1},...,x_{K},y)\\
&=p_{\phi}(y)\prod_{k=1}^{K}p_{\theta_{k}}(x_{k}\vert y) \\
&=(\Phi)^{y}(1-\Phi)^{(1-y)}\prod_{k=1}^{K}(\theta_{k,y})^{x_{k}}(1-\theta_{k,y})^{(1-x_{k})}
\end{align}
$$

**Classification:** Find the class that maximizes the posterior:

$$
\begin{equation}
\tilde{y}=\arg\max_{y}p(y\vert x)
\end{equation}
$$

**Training:** Find the **class-conditional** MLE parameters

$P(Y)$ is independent with the class, and is used in all the data. For each $P(X_{k}\arrowvert Y)$, we condition on the data with the corresponding class.

$$
\begin{align}
\Phi &= \frac{\sum_{i=1}^{N}\mathbb{I}(y^{(i)}=1)}{N} \\
\theta_{k,0}&=\frac{\sum_{i}^{N}\mathbb{I}(y^{(i)}=0\wedge x_{k}^{(i)}=1)}{\sum_{i=1}^{N}\mathbb{I}(y^{(i)}=0)} \\
\theta_{k,1} &= \frac{\sum_{i=1}^{N}\mathbb{I}(y^{(i)}=1\wedge x^{(i)}_{k}=1)}{\sum_{i=1}^{N}\mathbb{I}(y^{(i)}=1)}\\
&\forall k\in\{1,...,K\}
\end{align}
$$

#### 2.2 Model 2: Multinomial Naive Bayes

**Suppose:** Option 1: integer vector(word Ids),$x=\{x_{1},...,x_{M}\},\text{where } x_{i}\in\{1,...,K\}$ a word id, for $i=1,...,M$.

**Generative story:**

For $i\in \{1,...,N\}$: $y^{(i)}\sim \text{Bernoulli}(\Phi)$.

For $j\in \{1,...,M_{i}\}$: $x_{j}^{(i)}\sim \text{Multinomial}(\theta_{y^{(i)}},1)$

**Model**:

$$
\begin{align}
p_{\phi,\theta} &= p_{\phi}(y)\prod_{k=1}^{K}p_{\theta_{k}}(x_{k}\vert y) \\
&= (\phi)^{y}(1-\phi)^{(1-y)}\prod_{j=1}^{M_{i}}\theta_{y,x_{j}}
\end{align}
$$

#### 2.3 Model 3: Gaussian Naive Bayes

**Support:** $X\in \mathbb{R}^{K}$, $X$ is a continuous variable, all above are discrete variables. 

**Model:** Product of **prior** and the event model:

$$
\begin{align}
p(x,y) &= p(x_{1},...,x_{k},y)\\
&= p(y)\prod_{k=1}^{K}p(x_{k}\vert y)
\end{align}
$$

Gaussian Naive Bayes assumes that $p(x_{k}\vert y)$ is given by a Normal distribution.

- When $X$ is multivariate-Gaussian vector:
 - The joint probability of a data and it label is:
 
 $$
 \begin{align}
 p(x_{n},y_{n}^{k}=1\arrowvert \tilde{\mu},\Sigma) &= p(y_{n}^{k}=1)\times p(x_{n}\arrowvert y_{n}^{k}=1, \tilde{\mu},\Sigma) \\
 &= \pi_{k}\frac{1}{(2\pi\arrowvert \Sigma\arrowvert)^{1/2}}exp(-\frac{1}{2}(x_{n}-\tilde{\mu}_{k})^{T}\Sigma^{-1}(x_{n}-\tilde{\mu}_{k}))
 \end{align}
 $$

- The **naive Bayes** simplification:

$$
\begin{align}
p(x_{n},y_{n}^{k}=1\arrowvert \mu,\sigma) &= p(y_{n}^{k}=1)\times \prod_{j}p(x_{n}^{j}\arrowvert y_{n}^{k}=1, \mu_{k}^{j}, \sigma_{k}^{j}) \\
&= \pi_{k}\prod_{j}\frac{1}{\sqrt{2\pi}\sigma_{k}^{j}}exp(-\frac{1}{2}(\frac{x_{n}^{j}-\mu_{k}^{j}}{\sigma_{k}^{j}})^{2})
\end{align}
$$

- More generally:
 - where $p(\bullet \arrowvert \bullet)$ is an arbitrary conditional(discrete or continuous) 1-D density:
 
 $$
 \begin{equation}
 p(x_{n},y_{n}\arrowvert \eta, \pi) = p(y_{n}\arrowvert \pi)\times \prod_{j=1}^{m}p(x_{n}^{j}\arrowvert y_{n}, \eta)
 \end{equation}
 $$

#### 1.1.4 Model 4: Multiclass Naive Bayes

**Model:** The only change is that we permit $y$ to range over $C$ classes.

$$
\begin{align}
p(x,y) &= p(x_{1},...,x_{k}, y) \\
&= p(y)\prod_{k=1}^{K}p(x_{k}\arrowvert y)
\end{align}
$$

Now, $y\sim \text{Multinomial}(\Phi,1)$ and we have a seperate conditional distribution $p(x_{k}\arrowvert y)$ for each of the $C$ classes.

## 3  Regularization

In this case, we take bernoulli naive bayes model as an example for spam and non-spam document classification. In real scenario, we usually oberve some words that is never seen before. In this case, what happen to MLE estimation of $p(x_{k}\vert y)$

$$
\begin{equation}
\displaystyle{\theta_{k,0} = \frac{\sum_{i=1}^{N}\mathbb{I}(y^{(i)}=0\land x_{k}^{(i)}=1)}{\sum_{i=1}^{N}\mathbb{I}(y^{(i)}=0)}}
\end{equation}
$$

the result is that $\theta_{k,0}^{\text{MLE}}=0$.

In the test time, the posterior probability would be:

$$
\begin{equation}
p(y\vert \mathbf{x})=\frac{p(\mathbf{x}\vert y)p(y)}{p(\mathbf{x})}=0
\end{equation}
$$

### 3.1 Added- 1 Smooth

In order to avoid this case, the simplest method is to add single pseudo-abservation to the data. This converts the true observations $D$ into a new dataset $D^{'}$ from we derive the models

$$
\begin{align}
& D=\{(x_{i},y_{i})\}_{i=1}^{N} \\
& D^{'}=D\cup\{(\mathbf{0}, 0),(\mathbf{0},1),(\mathbf{1},0),(\mathbf{1},0)\}
\end{align}
$$

where $\mathbf{0}$ is the vector of all zeros and $\mathbf{1}$ is the vector of all ones. This has the effect of pretending that we observed each feature $x_{k}$ with each class $y$.

The parameter estimation will become:

$$
\begin{align}
\Phi &= \frac{\sum_{i=1}^{N}\mathbb{I}(y^{(i)}=1)+2}{N+4} \\
\theta_{k,0}&=\frac{1+\sum_{i}^{N}\mathbb{I}(y^{(i)}=0\wedge x_{k}^{(i)}=1)}{2+\sum_{i=1}^{N}\mathbb{I}(y^{(i)}=0)} \\
\theta_{k,1} &= \frac{1+\sum_{i=1}^{N}\mathbb{I}(y^{(i)}=1\wedge x^{(i)}_{k}=1)}{2+\sum_{i=1}^{N}\mathbb{I}(y^{(i)}=1)}\\
&\forall k\in\{1,...,K\}
\end{align}
$$

### 3.2 Added- $\lambda$ Smooth

Suppose we have a dataset obtained by repeatedly rolling a K-side(weight) die. Give data $D=\{(x_{i})\}_{i=1}^{N}$ where $x_{i}\in\{1,...,K\}$, we have the following MLE:

$$
\begin{equation}
\displaystyle{\phi_{k}=\frac{\sum_{i=1}^{N}\mathbb{I}(x_{i}=k)}{N}}
\end{equation}
$$

with add-$\lambda$ smoothing, we add pseudo-observations as before to obtain a smoothed estimate:

$$
\begin{equation}
\phi_{k}=\frac{\lambda+\sum_{i=1}^{N}\mathbb{I}(x_{i}=k)}{k\lambda + N}
\end{equation}
$$





