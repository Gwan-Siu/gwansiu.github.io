---
layout:     post
title:      "Optimal Transport Problem and Wasserstein Distance"
date:       2018-12-02 12:00:00
author:     "GwanSiu"
catalog: true
tags:
    - Analysis
---

## 1. Introduction to Optimal Transport Problem

Optimal transport problem is a classical problem in mathematics area. Recently, many researchers in machine learning community pay more attention to optimal transport, because Wasserstein distance provide a good tool to measure the similarity of two distribution. Optimal transport problem has two versions: Monge's formulation and Kantorovich formulation.

I utilize the math symbol in [1]. Consider two signals $I_{0}$ and $I_{1}$ defined over their support set $Omega_{0}$ and $Omega_{1}$, where $Omega_{0}, Omega_{1}\in\mathbb{R}$. $I_{0}(x)$ and $I_{1}(y)$ are denoted as signal intensities, where $I_{0}(x)\geq 0, I_{1}(y)\geq 0$ for $x\in \Omega_{0},y\in\Omega_{1}$. In addition, the total amount of signal for both signals should be equal to the same constant, i.e. $\displaystyle{\int_{\Omega_{0}}I_{0}(x)\mathrm{d}x}$. In other words, $I_{0}$ and $I_{1}$ are assumeed to be probability density functions(PDFs).

### 1.1 Monge Formulation

Monge's optimal transport problem is to find a function $f:\Omega_{0}\rightarrow \Omega_{1}$ that pushes $I_{0}$ and $I_{1}$ and minimizes the objective function:

$$
\begin{equation}
M(I_{0}, I_{1})=\displaystyle{\inf_{f\in MP}\int_{\Omega_{0}}c(x,f(x))I_{0}(x)\mathrm{d}x}
\end{equation}
$$

where $c:\Omega_{0}\times\Omega_{1}\rightarrow\mathbb{R}^{+}$ is the cost of moving pixel intensity $I_{0}$ from $x$ to $f(x)$, i.e. Monge consider Euclidean distance as the cost function in his original formulation, $c(x,f(x))=\vert x-f(x)\vert$, and MP stands for a **measure preserving map(or transport maps)** that moves all the signal intensity from $I_{0}$ to $I_{1}$. That is, $\forall B\in\Omega_{1}$, the MP requirement is that:

$$
\begin{equation}
\displaystyle{\int_{\{x:f(x)\in B\}}I_{0}(x)\mathrm{d}x=\int_{B}I_{1}(y)dy}
\end{equation}
$$

if $f$ is one to one, this means $\forall A\in \Omega_{0}$

$$
\begin{equation}
\displaystyle{\int_{A}I_{0}(x)\mathrm{d}x=\int_{f(A)}I_{1}(y)\mathrm{d}y}
\end{equation}
$$

Rigirous speaking, Monge formulation of the problem seeks to rearrange signal $I_{0}$ into signal $I_{1}$ while minimizing a specific cost function. If $f$ is smooth and one-to-one, the equation(2) can be rewritten in differential form as

$$
\begin{equation}
\text{det}(Df(x))I_{1}(f(x))=I_{0}(x)
\end{equation}
$$

almost everywhere, where $Df$ is Jacobian of $f$. Note that both the objective function and the constraint in (1) are nonlinear with respect to $f(x)$. 

### 1.2 Kantorovich Formulation

Kantorovich formulated the transport problem by optimizing over the joint distribution of $I_{0}$ and $I_{1}$, which is denoted as $gamma$. The physical meaning is how much mass is being moved to different coordinates, i.e., let $A\subset \Omega_{0}$ and $B\subset\Omega_{1}$. To make a distinction between a probability distribution and density function, we define a probability distribution of $I_{0}$ is $I_{0}(A)=\int_{A}I_{0}(x)\mathrm{d}x$. The quatity $\gamma(A\times B)$ tells us how much mass in set $A$ is being moving to set $B$. Thus, the MP constraint can be expressed as $\gamma(\Omega_{0}\times B)=I_{1}(B)$ and $\gamma(A\times \Omega_{1})=I_{0}(A)$. 

The Kantorovich formulation can be written as

$$
\begin{equation}
K(I_{0}, I_{1})=\displaystyle{\min_{\gamma\in MP}\int_{\Omega_{0}\times\Omega_{1}}c(x,y)\mathrm{d}\gamma(x,y)}
\end{equation}
$$

Kantorovich formualtion has a discrete setting, i.e., for PDFs of the form $\displaystyle{I_{0}=\sum_{i=2}^{M}p_{i}\delta(x-x_{i}), \text{and }I_{1}=\sum_{j=1}^{N}q_{j}\delta(y-y_{j})}$. Kantorovich formulation allows mass splitting. Thus, Kantorovich formulation can be rewritten as:

$$
\begin{align}
K(I_{0}, I_{1})&=\dispalystyle{\min_{\gamma}\sum_{i}\sum_{j}c(x_{i},y_{j})}\gamma_{ij}\\
s.t. \displaytyle{\sum_{j}}\gamma_{ij}&=p_{i},\displaystyle{sum_{i}}\gamma_{ij}=q_{j} \\
\gamma_{ij}&\geq 0, i=1,...,M,j=1,...,N
\end{align}
$$

where $\gamma_{ij}$ defiines how much of the mass particle $m_{i}$ at $x_{i}$ needs to be moved to $y_{j}$. The optimization obove has a linear objective function and linear constraints. Therefore, it is a linear programming problem. This problem is convex, but not strictly, and the constraints provides a polyhedral set of $M\times N$ matrices.

### 1.3 Kantorovich-Rubinstein Duality

## 2. Wasserstein Distance

### 2.1  Wasserstein Metric

$\Omega$ denotes a bounded subset of $\mathbb{R}^{d}$, and $p(\Omega)$ is the set of probability densities supported on $\Omega$. The p-Wasserstein metric, $W_{p}$, for $p\geq 1$ on $p(\Omega)$ is then defined as using the optimal transportation problem with the cost function $c(x,y)=\vert x-y\vert$. For $I_{0}$ and $I_{1}$ in $p(\Omega)$,

$$
\begin{equation}
W_{p}(I_{0}, I_{1})=\displaystyle{(\inf_{\gamma\in MP}\int_{\Omega\times\Omega}\vert x-y\vert^{p}\mathrm{d}\gamma(x,y))^{\frac{1}{p}}}
\end{equation}
$$

For any $p\geq 1$, $W_{p}$ is a metric on $p(\Omega)$. The metric space $(p(\Omega), W_{p})$ is denoted as the p-Wasserstein space. The convergence with respect to $W_{p}$ is equivalent to the weak convergence of measure, i.e., $W_{p}(I_{n}, I)\rightarrow 0$ as $n\rightarrow\infty$ if and ony if for every bounded and continuous function $f:\Omega\rightarrow\mathbb{R}$

$$
\begin{equation}
\displaystyle{\int_{\Omega}f(x)I_{n}(x)\mathrm{d}x\rightarrow \int_{\Omega}f(x)I(x)\mathrm{d}x}
\end{equation}
$$

For $p=1$, the p-Wasserstein metric is known as the Monge-Rubinstein metric or the Earth mover's distance. For p-Wasserstein metric in one dimension, the optimal map has a closed form solution. We define $F_{i}$ be the cumulative distribution function of $I_{i}$ for $i=0,1$, i.e.,

$$
\begin{equation}
\displaystyle{F_{i}=\int_{\inf(\Omega)}^{x}I_{i}(x)\mathrm{d}x}\quad \text{for }i=0,1
\end{equation}
$$

cumulative distribuiton $F_{i}$ is nondecreasing from $0$ to $1$. We also define the pseudoinverse of $F_{0}$ as follows: for $z\in(0,1), F^{-1}(z)$ is the smallest $x$ for which $F_{0}(x)\geq z$, i.e.,

$$
\begin{equation}
F_{0}^{-1}(z)=\inf\{x\in\Omega:F_{0}(x)\geq z\}
\end{equation}
$$

the pseudoinverse provides a closed-form solution for the p-Wasserstein distance:

$$
\begin{equation}
\displaystyle{W_{p}(I_{0}, I_{1})=(\int_{0}^{1}\vert F_{0}^{-1}(z)-F_{1}^{-1}(z)\vert^{p}\mathrm{d}z)^{\frac{1}{p}}}
\end{equation}
$$

We assume $I_{0}$ is the empirical distribution $P$ of a dataset $X_{1}, X_{2},...,X_{n}$ and $I_{1}$ is also the empirical distribution $Q$ of a dataset $Y_{1},Y_{2},...,Y_{n}$ of the same size, then the distance takes a very siple function of order statistics:

$$
\begin{equation}
\displaystyle{W_{p}(P,Q)=\lgroup\sum_{i=2}^{n}\Vert X_{i}-Y_{i}\Vert^{p}\rgroup^{\frac{1}{p}}}
\end{equation}
$$


### 2.2 Sliced-Wasserstein Metric

The idea behind the sliced-Wasserstein metric is to first obtain a set of 1-D respresentations for a higher-dimensional probability distribution through projection, and then calculate the distance between two distributions as a functional on the Wasserstein distance of their 1-D respresentations. In this sense, the distance is obtained by solving several 1-D optimal transport problems, which have closed-form solutions.

The projection of high-dimensional PDFs can be Radon transform, which is well-known in image processing area.

The d-dimensional Radon transform $\mathcal{R}$ maps a function $I\in L_{1}(\mathbb{R}^{d})$ where $L_{1}(\mathbb{R}^{d}):=\{I:\mathbb{R}^{d}\rightarrow \mathbb{R}\vert \int_{\mathbb{R}^{d}}\vert I(x)\vert \mathrm{d}x\leq \infty\}$ into the set of its integrals over the hyperplanes of $\mathbb{R}^{n}$. It is defined as:

$$
\begin{equation}
\displaystyle{\mathcal{R}I(t,\theta):=\int_{\mathbb{R}}I(t\theta+s\theta^{\bot})\mathrm{d}s,\quad\forall t\in\mathbb{R},\forall\theta\in\mathbb{S}^{d-1}}
\end{equation}
$$

Thus, the sliced-Wasserstein metric for PDFs $I_{0}$ and $I_{1}$ on $\mathbb{R}^{d}$ is defined as

$$
\begin{equation}
\displaystyle{SW_{p}(I_{0}, I_{1})=\lgroup\int_{\mathbb{R}^{d-1}}W_{p}^{p}(\mathcal{R}I_{0}(\bullet,\theta),\mathcal{R}I_{1}(\bullet,\theta))\mathrm{d}\theta\rgroup^{\frac{1}{p}}}
\end{equation}
$$

where $p\geq 1$, and $W_{p}$ is the p-Wasserstein metric, which, for 1-D PDFs,$\mathcal{R}I_{0}(\bullet,\theta),\mathcal{R}I_{1}(\bullet,\theta)$ has a cloased-form solution. 

## 3. Compared with Other Metrics.


**Reference**

[1]. Kolouri S, Park S R, Thorpe M, et al. Optimal mass transport: Signal processing and machine-learning applications[J]. IEEE Signal Processing Magazine, 2017, 34(4): 43-59.