---
layout:     post
title:      "Dimension Reduction"
date:       2018-10-27 12:00:00
author:     "GwanSiu"
catalog: true
tags:
    - Machine Learning
---

## 1. Introduction

In this article, I will talk about dimension reduction algorithm, which is a hot topic in the community of both machine learning and representation learning. The structure of this article is arranged as followed: In session 2, I will begin the story with principle component analysis(PCA) algorithm, which is one of very famous dimension reduction algprithm. After that, I will talk about the relation between PCA and SVD decomposition. In session 3, I will talk about PCA under probabilistic framework, which is called probabilistic PCA(PPCA). In session 4, factor analysis as well as kernel PCA are briefly introduced, both of which are variants of PPCA.  In session 5, independent component analysis(ICA) is dicussed, which is different from all algorithm discussed in above sessions, because it is based on non-gaussian prior.

## 2. Principle Component Anlysis(PCA)

### 2.1 Standard PCA

Given data $X=(x_{1},x_{2},...,x_{n}), \forall i\leq n, x_{i}\in \mathbb{R}^{D}$, our goal is to project the data onto a $M$ dimensional space, where $M\leq D$, while maximizing the variance of the projected data. At this time, we assume M is given for simplication.(in practise, we can choose M according to our applications). We define the projection matrix $W\in\mathbb{R}^{D\times M}$, so the projected data $x_{n}$ is $\hat{x}_{n}=Wx_{n}$. To maximize the covariance of projected data is:

$$
\begin{equation}
\arg\max_{W}\Vert(\hat{x}-\mu)\Vert^{2}
\end{equation}
$$

where $\mu=W\bar{x}=W(\frac{1}{N}\displaystyle{\sum_{1}^{N}x_{n}})$. Thus, we have:

$$
\begin{align}
\Vert(\hat{x}-\mu)\Vert^{2}&=\frac{1}{N}\displaystyle{\sum_{1}^{N}}(Wx_{n}-W\bar{x})^{2} \\
&=W^{T}SW
\end{align}
$$

where $S$ is covariance matrix defined by:

$$
\begin{equation}
S=\frac{1}{N}\displaystyle{\sum_{1}^{N}}(x_{n}-\bar{x})(x_{n}-\bar{x})^{T}
\end{equation}
$$

We now maximize our object function:$W^{T}SW$, and we set a constraint on $W:W^{T}W=1$. We abtain lagrange function: $L(W)=W^{T}SW+\lambda(W^{T}W-1)$, and set the derivative w.r.t $w_{i}$ to 0, then we get the result:

$$
\begin{equation}
Sw_{i}=\lambda_{i}w_{i}
\end{equation}
$$

and so the variance will be maximized when we set $w_{i}$ equal to the eigenvector of covariance matrix $S$ with corresponding eigenvalue $\lambda_{i}$.

### 2.2 PCA and SVD Desomposition

SVD is a generalized version of eigen decomposition. In contrast to eigen decomposition which requires decomposed matrix is a square matrix, SVD decomposition can be applied for any matrix.

$$
\begin{equation}
X = USV^{T}
\end{equation}
$$

where $X\in \mathbb{R}^{N\times D}$, $U$ is matrix with dimension $\mathbb{R}^{N\times N}$ whose column vectors are othorgonal: $U^{T}U=I$, V is a $D\times D$ matrix whose rows and columns are orthogonal, $S$ is sigular value matrix with dimension $\mathbb{N\times D}$, containing at most $r=\min(N,D)$ sigular values $\sigma_{i}\geq 0$ on the main diagonal, and resting are 0. The columns of $U$ is left singular vector, and the columns of $V$ are right singular vectors.

Due to at most $r=\min(N,D)$ non-zero singular values, we can truncated matrix $S$ so that the computational complexity can be correspondingly reduced.

$$
\begin{equation}
X = \hat{U}{S}{V}^{T}
\end{equation}
$$

where $\hat{U}\in\mathbb{R}^{N\times r}$, $S\in\mathbb{R}^{r\times r}$, $V\in\mathbb{R}^{r\times D}$, and $X\in\mathbb{R}^{N\times D}$. The time complexity of truncated SVD is $\mathcal{O}(ND\min(N,D))$. Truncated SVD also can be considered as $r$ rank approximation.

For connection between eigenvector and singular vectors is the following. For an arbitrary matrix $X$, if $X=USV^{T}$, we have:

$$
\begin{equation}
X^{T}X = VS^{T}D^{T}DSV^{T}=VS^{T}SV^{T}=VDV^{T}
\end{equation}
$$

where $D=S^{2}$ is a diagonal matrix containing the squares singular values. Hence:

$$
\begin{equation}
(X^{T}X)V=VD
\end{equation}
$$

so the eigenvetors of $X^{T}X$ are the right sigular vectors of $X$, i.e. $V$, and the eigenvalues of $X^{T}X$ are equal to the squared singular value of $X$, i.e. $D$. Similarly, we have:

$$
\begin{align}
XX^{T}&=USV^{T}VS^{T}U^{T}=UDU^{T} \\
(XX^{T})U&=UD \\
\end{align}
$$

so the eigenvetors of $XX^{T}$ are the left sigular vectors of $X$, i.e. $U$, and the eigenvalues of $XX^{T}$ are equal to the squared singular value of $X$, i.e. $D$. Thus, we can summarize all this as follows:

$$
\begin{equation}
U=\text{evec}(XX^{T})\quad V=\text{evec}(X^{T}X)\quad S=\text{eval}(X^{T}X)=\text{eval}(XX^{T})
\end{equation}
$$

Back to the PCA. Let $X=USV^{T}$ be truncated SVD of $X$. We know that projected $\hat{W}=V$, and that $Z=X\hat{W}$, so:

$$
\begin{equation}
Z=USV^{T}V=US
\end{equation}
$$

the reconstruction is given by $X=ZW^{T}$, so we find:

$$
\begin{equation}
X=USV^{T}
\end{equation}
$$

this is exactly the same as truncated SVD.


## 3. Probabilistic PCA(PPCA)

### 3.1 PCA under probabilistic framework

In this part, we view PCA from generative model viewpoint in which a sampled value of the observed variable is obtained by firstly choosing a value for latent variable in latent space and then sampling the observed variable conditioned on this latent value.**Differnet from conventional PCA, PPCA is based on the a mapping from latent space to data space**. Intuitively, we can consider $p(x)$ as being defined by taking an isotropic Gaussian "spray can" and moving it across the principle subspace spraying Gaussian ink with density determined by $\sigma^{2}$ and weighted by the prior distribution.

Specifically, we assume observed data $x$ is a linear transform of latent variable $z$ plus additive gaussian 'noise', so that:

$$
\begin{equation}
x=Wz+\mu+\epsilon
\end{equation}
$$

where $z, x\in\mathbb{R}^{D\times1}$, $\epsilon is 'isotropic gaussian' with variance \sigma, \epsilon\in\mathbb{R}^{D}$. The likelihood of $p(x)$ is given by:

$$
\begin{equation}
p(x)\int p(x\vert z)p(z)\mathrm{d}z
\end{equation}
$$

under the framework of linear gaussian model, we can find $p(x)$ is still gaussian distribution, and is given by:

$$
\begin{equation}
p(x)=\mathcal{N}(x\vert \mu, C)
\end{equation}
$$

where $C$  is $D\times D$ covariance matrix, and is defined by:

$$
\begin{equation}
C=WW^{t}+\sigma^{2}I
\end{equation}
$$

we can derive this result in a straight forward:

$$
\begin{align}
\mathcal{E}[x] &= \mathcal{E}[Wz+\mu+\epsilon]= \mu \\
\text{cov}[x] &= \mathcal{E}[(Wz+\epsilon)(Wz+\epsilon)^{T}] \\
&= \mathcal{E}[Wzz^{T}W] + \mathcal{E}[\epsilon\epsilon^{T}] \\
&= WW^{T} +\sigma^{2}I
\end{align}
$$

where $z$ is assumed orthogonal and $\epsilon$ are independent random variables and hence are uncorrelated.

**Note:** the distribution $p(x)$ is invariant to rotation of latent space coordinates. Specifically, we consider a matrix $\hat{W}=WR$ where $R$ is rotation matrix, e.g. $RR^{T}=I$. Then:

$$
\begin{equation}
\hat{W}\hat{W}^{T} = WRR^{T}W^{T}=WW^{T}
\end{equation}
$$

and hence is dependent of $R$. Thus there is a whole family of matrices $\hat{W}$ all of which can give the same result.

Besides the distribution $p(x)$, we will also require the posterior distribution $p(z\vert x)$, and is given by:

$$
\begin{equation}
p(z\vert x)=\mathcal{N}(z\vert M^{-1}W^{T}(x-\mu), \sigma^{2}M^{-1})
\end{equation}
$$

the mean of posterior depends on $x$, whereas the posterior covariance is independent of $x$.

### 3.2 MLE for PPCA

the likelihood $p(x)$ is governed by the parameters $\mu, W$ and $\sigma^{2}$. Our goal is to find these parameters by MLE. Thus, the date log likelihood is given by:

$$
\begin{align}
\ln p(X\vert \mu, W,\sigma)&=\displaystyle{\sum_{i=1}^{N}}\ln p(x_{i}\vert \mu, W, \sigma) \\
&= -\frac{N}{2}\ln\vert C\vert -\frac{1}{2}\displaystyle{\sum_{i=1}^{N}}x_{i}^{T}C^{-1}x_{i} \\
&=-\frac{N}{2}\ln \vert C\vert +\text{tr}(C^{-1}\Sigma)
\end{align}
$$

where $C=WW^{t}+\sigma^{2}I$ and $S=\frac{1}{N}\displaystyle{\sum_{i=1}^{N}x_{i}x_{i}^{T}}=\frac{1}{N}X^{T}X$. The maxima of the log-likelihood are given by:

$$
\begin{equation}
\hat{W} = V(\Lambda-\sigma^{2}I)^{\frac{1}{2}}R
\end{equation}
$$

where $R$ is an arbitrary $L\times L$ orthogonal matrix, $V$ is the $D\times L$ matrix whose colummns are the first $L$ eigenvectors of $S$, and $\Lambda$ is the corresponding diagonal matrix of eigenvalues. Withoud  loss the generality, we can set $R=I$. Furthermore, the MLE of the noise variance is given by:

$$
\begin{equation}
\hat{\sigma}^{2}=\frac{1}{D-L} \displaystyle{\sum_{j=L+1}^{D}\lambda_{j}}
\end{equation}
$$

which is the average variance associated with the discarded dimensions.

thus, as $\sigma^{2}\rightarrow 0$, we have $\hat{W}\rightarrow V$, as in principle PCA. What about \hat{Z}? The posterior can be easily calculated:

$$
\begin{equation}
p(z_{i}\vert x_{i}, \hat{\theta}) = \mathcal{N}(z_{i}\vert \hat{F}^{-1}\hat{W}^{T}x_{i}, \sigma^{2}\hat{F}^{-1})
\end{equation}
$$

where $\hat{F}=\hat{W}^{T}W+\sigma^{2}I$, in contrast to $C=WW^{t}+\sigma^{2}I$. Hence, as $\sigma^{2}\rightarrow 0$, we find $\hat{W}\rightarrow V,\hat{F}\rightarrow I$ and $\hat{z}_{i}\rightarrow V^{T}x_{i}$. Thus the posterior mean is obtained by an orthogonal projection of the data onto the column space of $V$, as in classical PCA.

### 3.3 EM for PPCA

PPCA is a generative model having latent variables. We can apply EM algorithm for PPCA.

Let $\hat{Z}$ is $L\times N$ matrix storing the posterior means along its columns. Similarly, let $\hat{X}=X^{T}$ store the original data along its columns. When $\sigma^{2}=0$, we have:

The E-step:

$$
\begin{equation}
\hat{Z} = (W^{T}W)^{-1}W^{T}\hat{X}
\end{equation}
$$

The M-step:

$$
\begin{equation}
W=\hat{X}\hat{Z}^{T}(\hat{Z}\hat{Z}^{T})^{-1}
\end{equation}
$$

## 4. Generalized PPCA-Factor analysis and Kernel PCA

## 5. Independent Component Analysis(ICA)

















