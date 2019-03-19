---
layout:     post
title:      "Dimension Reduction:PCA, FA, and ICA"
date:       2018-10-27 12:00:00
author:     "GwanSiu"
catalog: true
tags:
    - Machine Learning
---

## 1. Introduction
 
Dimension reduction algorithms are one of hot topic in machine learning and representation learning, and some of them, such as principle component analysis (PCA), factor analysis (FA) and independent component analysis (ICA), will be discussed in this article. The structure of this article is arranged as followed: In session 2, principle component analysis(PCA) algorithm will be discussed, which is one of very famous dimension reduction algprithm. In session 3, PCA will be analysed under the probabilistic framework, which is called probabilistic PCA (PPCA). In session 4, kernel PCA are briefly introduced, one of which is the variant of PPCA. In session 5, factor analysis is introduced. In session 6, independent component analysis(ICA) is dicussed, which is different from all algorithm discussed in above sessions, because it is based on non-gaussian prior.

## 2. Principle Component Anlysis(PCA)

### 2.1 Standard PCA

Given data $X=(x_{1},x_{2},...,x_{n}), \forall i\leq n, x_{i}\in \mathbb{R}^{D}$, our goal is to project the data onto a $M$ dimensional space, where $M\leq D$, while maximizing the variance of the projected data. At this time, we assume M is given for simplication.(in practise, we can choose M according to our applications). We define the projection matrix $W\in\mathbb{R}^{D\times M}$, so the projected data $x_{n}$ is $\hat{x}_{n} = Wx_{n}$. To maximize the covariance of projected data is:

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

SVD is a generalized version of eigen decomposition. Instead of eigen decomposition, which requires decomposed matrix is a square matrix, SVD decomposition can be applied for any matrix. In addition, the computational complexity of eigen decomposition is $O(n^{3})$.

$$
\begin{equation}
X = USV^{T}
\end{equation}
$$

where $X\in \mathbb{R}^{N\times D}$, $U$ is matrix with dimension $\mathbb{R}^{N\times N}$ whose column vectors are othorgonal: $U^{T}U=I$, V is a $D\times D$ matrix whose rows and columns are orthogonal, $S$ is sigular value matrix with dimension $\mathbb{N\times D}$, containing at most $r=\min(N,D)$ sigular values $\sigma_{i}\geq 0$ on the main diagonal, and resting are 0. The columns of $U$ is left singular vector, and the columns of $V$ are right singular vectors.

Due to at most $r=\min(N,D)$ non-zero singular values, we can truncated matrix $S$ so that the computational complexity can be correspondingly reduced.

$$
\begin{equation}
X = \hat{U}\hat{S}\hat{V}^{T}
\end{equation}
$$

where $\hat{U}\in\mathbb{R}^{N\times r}$, $\hat{S}\in\mathbb{R}^{r\times r}$, $\hat{V}\in\mathbb{R}^{r\times D}$, and $X\in\mathbb{R}^{N\times D}$. The time complexity of truncated SVD is $\mathcal{O}(ND\min(N,D))$. Truncated SVD also can be considered as $r$ rank approximation.

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

**so the eigenvetors of** $XX^{T}$ (Covariance matrix S) **are the left sigular vectors of** $X$, i.e. $U$, and the eigenvalues of $XX^{T}$ are equal to the squared singular value of $X$, i.e. $D$. Thus, we can summarize all this as follows:

$$
\begin{equation}
U=\text{evec}(XX^{T})\quad V=\text{evec}(X^{T}X)\quad S=\text{eval}(X^{T}X)=\text{eval}(XX^{T})
\end{equation}
$$

Back to the PCA. Let $X=\hat{U}\hat{S}\hat{V}^{T}$ be truncated SVD of $X$. We know that projected $\hat{W}=V$, and that $Z=X\hat{W}$, so:

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


<img src="https://raw.githubusercontent.com/Gwan-Siu/BlogCode/master/other/PPCA.png" width = "600" height = "400"/> 


Specifically, we assume observed data $x$ is a linear transform of latent variable $z$ plus additive gaussian 'noise', so that:

$$
\begin{equation}
x=Wz+\mu+\epsilon
\end{equation}
$$

where $z, x\in\mathbb{R}^{D\times1}$, $\epsilon \text{is 'isotropic gaussian' with variance} \sigma, \epsilon\in\mathbb{R}^{D}$. The likelihood of $p(x)$ is given by:

$$
\begin{equation}
p(x)=\int p(x\vert z)p(z)\mathrm{d}z
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

under the framework of linear gaussian system, the posterior distribution $p(z\vert x)$ also is gaussian distribution, and is given by:

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
&= \displaystyle{-\frac{ND}{2}\ln (2\pi) -\frac{N}{2}\ln\vert C\vert -\frac{1}{2}\sum_{i=1}^{N}(x_{i}-\mu)^{T}C^{-1}(x_{n}-\mu)} \\
&= -\frac{N}{2}\ln\vert C\vert -\frac{1}{2}\displaystyle{\sum_{i=1}^{N}}x_{i}^{T}C^{-1}x_{i} \\
&=-\frac{N}{2}\ln \vert C\vert +\text{tr}(C^{-1}\Sigma)
\end{align}
$$

we set the derivative of the log likelihood with respect to $\mu$ equal to zero gives the expected result $\mu=\bar{x}$ where $\bar{x}$ is the mean of data, and back-substituting the formulation

$$
\begin{equation}
\ln p(X\vert W,\mu,\sigma^{2}) = -\frac{N}{2}(D\ln (2\pi)+\ln\vert C\vert +\text{Tr}(C^{-1}S))
\end{equation}
$$

where $C=WW^{t}+\sigma^{2}I$ and $S=\frac{1}{N}\displaystyle{\sum_{i=1}^{N}x_{i}x_{i}^{T}}=\frac{1}{N}X^{T}X$. 

Maximize with respect to $W$ and $\sigma$ ande we have the exact closed-form solution

$$
\begin{equation}
W_{ML} = U_{M}(L_{M}-\sigma^{2}I)^{\frac{1}{2}}R
\end{equation}
$$

where $U_{M}$ is the $D\times M$ matrix whose colummns are the first $M$ eigenvectors of data covariance matrix $S$, and $\Lambda$ is the corresponding diagonal matrix $M\times M$ of eigenvalues, and $R$ is an arbitrary $M\times M$ orthogonal matrix, $R$ is an arbitrary $L\times L$ orthogonal matrix. Withoud  loss the generality, we can set $R=I$, we can see that the columns of $W$ are the principle component eigenvectors scaled by the variance parameters $\lambda_{i}-\sigma^{2}$. Furthermore, the MLE of the noise variance is given by:

$$
\begin{equation}
\sigma^{2}_{ML}=\frac{1}{D-L} \displaystyle{\sum_{j=L+1}^{D}\lambda_{j}}
\end{equation}
$$

which is the average variance associated with the discarded dimensions.

when $M=D$, there is no reduction of dimensionality, then $U_{M}=U,L_{M}=L$, and we make use of the orthogonality properties $UU^{T}=I$ and $RR^{T}=I$, we can see that the covariance $C$ of the marginal distribution for $x$ becomes

$$
\begin{equation}
C=U(L-\sigma^{2}I)^{1/2}RR^{T}(L-\sigma^{2}I)^{1/2}U^{T}+\sigma^{2}I=ULU^{T}=S
\end{equation}
$$

thus, we can obtain the standard maximum likelihood solution for an unconstrained Gaussian distribution in which the covariance matrix is given by the sample covariance.

Given the point $x$ in data space, we can find the corresponding point in latent space by

$$
\begin{equation}
p(z\vert x)=\mathcal{N}(z\vert M^{-1}W^{T}(x-\mu), \sigma^{2}M^{-1})
\end{equation}
$$

the mean is given by:

$$
\begin{equation}
\mathbb{E}[z\vert x] = M^{-1}W_{ML}^{T}(x-\bar{x})
\end{equation}
$$

where $M=WW^{T} + \sigma^{2}I$. Back project to data space by $W\mathbb{E}[z\vert x] +\mu$. When $\lim \sigma\rightarrow 0$, then the posterior mean reduces to

$$
\begin{equation}
(W_{ML}^{T}W_{ML})^{-1}W_{ML}^{T}(x-\bar{x})
\end{equation}
$$



### 3.3 EM for PPCA

PPCA is a generative model having latent variables. We can apply EM algorithm for PPCA. Firstly, we take the expectation of the complete-data log likelihood and take its expectation with respect to the posterior distribution of the latent distribition evaluated using "old" parameters values. Secondly, maximization of this expected complete-data log likelihood and we update the "new" parameters.

the complete-log data likelihood:

$$
\begin{equation}
\ln p(X,Z\vert \mu, W, \sigma^{2}) = \sum_{i=1}^{N}(\ln p(x_{i}\vert z_{i}) + \ln p(z_{n}))
\end{equation}
$$

taking the expectation with respect to the posterior distribution over the latent variables, we obtain

$$
\begin{align}
\mathbb{E}[\ln p(X,Z\vert \mu, W, \sigma^{2})] &= -\sum_{i=1}^{N}(\frac{D}{2}\ln(2\pi\sigma^{2})+\frac{1}{2}\text{Tr}(\mathbb{E}[z_{n}z_{n}^{T}])+\frac{1}{2\sigma^{2}}\Vert x_{i}-\mu\Vert^{2} \\
&-\frac{1}{\sigma^{2}}\mathbb{E}[z_{n}]^{T}W^{T}(x_{i}-\mu)+\frac{1}{2\sigma^{2}}\text{Tr}(\mathbb{E}[z_{n}z_{n}^{T}]W^{T}W))
\end{align}
$$


The E-step, we use old parameter to evaluate:

$$
\begin{align}
\mathbb{E}[z_{n}] &= M^{-1}W^{T}(x_{i}-\mu) \\
\mathbb{E}[z_{n}z_{n}^{T}] &= \sigma^{2}M^{-1} + \mathbb{E}[z_{n}]\mathbb{E}[z_{n}]^{T}
\end{align}
$$

The M-step, we maximize with respect to $W$ and $\sigma$, keeping the posterior statistics fixed

$$
\begin{align}
W_{new} &= [\sum_{i=1}^{N}(x_{i}-\bar{x})\mathbb{E}[z_{n}]^{T}][\sum_{i=1}^{N}\mathbb{E}[z_{n}z_{n}]]^{-1} \\
\sigma_{new}^{2} &= \frac{1}{ND}\sum_{i=1}^{N}(\Vert x_{i}-\bar{x}\Vert^{2} -2\mathbb{E}[z_{n}]^{T}W_{new}^{T}(x_{n}-\bar{x})+\text{Tr}(\mathbb{E}[z_{n}z_{n}^{T}]W_{new}^{T}W_{new}^{T}))
\end{align}
$$

EM algorithm can be more computational efficient than conventional PCA in high-dimensional space. Conventional PCA takes $O(D^{3})$ computaion due to the eigendecomposition of the covariance matrix. In EM algorithm, we only take $O(MD^{2})$ because we are usually interested in first largest M dimension. 

### 3.4 Analysis of PPCA

One problem of mixture model with discrete latent variables is that they only use a single latent varible or K latent variables to generate observations, this assumption of discrete latent varibales set the limitation in its representational power. The latent space of PPCA assumed is a continuous space rather than a discrete sapce, because latent variables of PPCA follow Gaussian distribution.

## 4. Kernal PCA


## 5. Factor Analysis (FA)

PPCA is a special case of factor analysis, the covariance matrix of factor analysis is a semi-positive and sysmmetry matrix instead of a indentity matrix, which is used in PPCA. Usually, we assume that the obtained data is sufficient and we can easily apply multiple-Gaussian structure in the data. In other words, we consider our training data $m$ is much larger than the dimension $n$. If the number of training data $m$ is much less than the dimension $n$.  In such a problem, it might be
difficult to model the data even with a single Gaussian, much less a mixture of Gaussian. Specifically, since the m data points span only a low-dimensional subspace of $\mathbb{R}^{n}$, if we model the data as Gaussian, and estimate the mean and covariance using the usual maximum likelihood estimators,

$$
\begin{align}
\mu &= \frac{1}{m}\sum_{i=1}^{m}x^{(i)} \
\Sigma &= \frac{1}{m}\sum_{i=1}^{m}(x^{(i)}-\mu)(x^{(i)}-\mu)^{T}
\end{align}
$$

we would find that the matric $\Sigma$ is singular. This means that $\Sigma^{-1}$ does not exist, and $1/\vert \Sigma\vert^{1/2}=1/0$. But both of these terms are needed in computing the usual density of a multivariate Gaussian distribution. Another way of stating this difficulty is that maximum likelihood estimates of the parameters result in a Gaussian that places all of its probability in the affine space spanned by the data (M-projection), and this corresponds to a singular covariance matrix. The number of unkown parameters of a covariance matrix $\Sigma$ (off-diagonal) is $\frac{N(n-1)}{2}$, if we were fitting a full, uncontraint covariance matrix $\Sigma$ to data, it was necessary that $m\geq n+1$ in order for the maximum likelihood estimate of $\Sigma$ not be singular. 

Therefore, we need to set some contraints for covariance matrix $\Sigma$ in order to estimate it under likelihood framework.

- 1. The covariance matrix $\Sigma$ is diagonal

$$
\begin{equation}
\Sigma_{jj} = \frac{1}{m}\sum_{i=1}^{m}(x_{j}^{i}-\mu_{j})^{2}
\end{equation}
$$

Thus, $\Sigma_{jj}$ is just the empirical estimate of the covariance of the $j-th$ coordinate of the data. Recall that the contours of a Gaussian density are ellipses. A diagonal $\Sigma$ corresponds to a Gaussian where the major axes of these ellipses are axisaligned.

- 2. we may place a further restriction on the covariance matrix that not only must it be diagonal, but its diagonal entries must all be equal. In this setting, we have $\Sigma=\sigma I$ where $\sigma^{2}$ is the parameter under our control. The maximum likelihood estimate of $\sigma^{2}$ can be found to be:

$$
\begin{equation}
\sigma^{2} = \frac{1}{mn}\sum_{j=1}^{n}\sum_{i=1}^{m}(x_{j}^{(i)}-\mu_{j})^{2}
\end{equation}
$$

This model corresponds to using Gaussians whose densities have contours that are circles. With these two contraints, FA model is PPCA.

## 6. Independent Component Analysis (ICA)


















