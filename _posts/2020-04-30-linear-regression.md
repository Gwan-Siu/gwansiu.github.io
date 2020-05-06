---
layout:     post
title:      "Linear Regression(1)"
date:       2020-04-29 12:00:00
author:     "GwanSiu"
catalog: true
tags:
    - Machine Learning
---

> Linear regression is an old topic in the machine learning community, and this topic has been studied by researchers for the past decades. In this post, I will highlight some kerypoints on regression models. Specifically, I will begin with the univariate regression model, and consider it as the basic block to build the multiple regression model.


# 1. The univariate regression

Suppose that we have a dataset $\mathcal{D}=\{x_i, y_i\}_{i=1}^n$ with $n$ samples, where observation $\mathbf{y}$ is a $n$ dimensional vector, 
i.e. $\mathbf{y}=(y_1, y_2, \cdots, y_n)\in \mathbb{R}^n$, and the measurement $\mathbf{x}$ is also a n-dimension vector, i.e. $\mathbf{x}=(x_1, x_2, \cdots, x_n)\in \mathbb{R}^n$. We additionally assume that obseravation and measurement can be modeled as 

$$
\begin{equation}
\mathbf{y} =\beta^{\ast}\mathbf{x} +\epsilon
\end{equation},
$$

where $\beta^{\ast}\in \mathbb{R}$ is the ground-truth coefficient, which is unknown, and $\mathbf{\epsilon}=(\epsilon_{1}, \epsilon_{2},\cdots, \epsilon_{n})\in\mathbb{R}^{n}$ is the noise term, and $\mathbb{E}[\epsilon_{i}]=0, \text{Var}(\epsilon_{i})=\sigma^{2}, \forall i$, $\text{Cov}(x_{i}, x_{j})=0,\text{for } i\neq j$. 


Our goal is to estimate the coefficient in the underlying model, and we commonly use the least mean square estimator(LMSE). We formulate it as follows:

$$
\begin{equation}
\hat{\beta} =\underset{\beta}{\mathrm{argmin}}\,\sum_{i=1}^{n}\frac{1}{n}(y_{i}-\beta x_{i})^{2}=\underset{\beta}{\mathrm{argmin}}\,\Arrowvert \mathbf{y}-\beta\mathbf{x}\Arrowvert^{2}.
\end{equation}
$$

To obtain the optimal $\hat{\beta}$, we take the derivative of Eq.(2), and set the first-order derivative to zero. The univariate linear regression coefficient of $\mathbf{y}$ on $\mathbf{x}$ is 

$$
\begin{equation}
	\hat{\beta} = \frac{\sum_{i=1}^{n}x_{i}y_{i}}{\sum_{i=1}^{n}x_{i}^{2}} =\frac{\mathbf{x}^{T}\mathbf{y}}{\Arrowvert \mathbf{x}\Arrowvert^{2}}.
\end{equation}
$$

Next, we consider the incepter in the underlying linear model. Eq.(1) is reformulated as follows

$$
\begin{equation}
y = \beta_{0}^{\ast}+\beta_{1}^{\ast}x+\epsilon.
\end{equation}
$$

Correspondingly, we alternative optimize the following problem, i.e.

$$
\begin{equation}
(\hat{\beta}_{0}, \hat{\beta}_{1})=\underset{\beta_{0}, \beta_{1}}{\mathrm{argmin}}\, \sum_{i=1}^{n}(y_{i}-\beta_{0}-\beta_{1} x_{i})^{2}=\underset{\beta}{\mathrm{argmin}}\, \Arrowvert \mathbf{y}-\beta_{0}\mathbf{1}-\beta_{1}\mathbf{x}\Arrowvert^{2}.
\end{equation}
$$

by solving the above problem, and we can obtain

$$
\begin{equation}
\begin{split}
\hat{\beta}_{0} &= \bar{y}-\hat{\beta}_{1}\bar{x}  \\
\hat{\beta}_{1} &= \frac{(\mathbf{x}-\bar{x}\mathbf{1})^{T}(\mathbf{y}-\bar{y}\mathbf{1})}{\Arrowvert \mathbf{x}-\bar{x}\mathbf{1}\Arrowvert^{2}_{2}}
\end{split}
\end{equation}
$$

note that

$$
\begin{equation}
\hat{\beta}_{1}=\frac{cov(\mathbf{x},\mathbf{y})}{var(\mathbf{x})}=cov(\mathbf{x},\mathbf{y})\sqrt{\frac{var{y}}{var(x)}}
\end{equation}
$$

>Discussion: there is a question about when we consider the incepter term in the linear model. Without the incepter term, we assum that the underlying linear model pass through the origin. In most applications, we do not know any prior knowlage on the underlying linear model. It is applicable to consider the incepter term when we build the linear model.

# 2. The Multivariate Regression

In the univariate regression, the dimension $d$ of data we cconsider is just 1. In this session, we consider high dimensional data, i.e. $d>1$, and use the univariate regression model as the basic block to further develop the multivariate regression model.

Assume that there is a dataset $D=\{(\mathbf{x}_i,y_i)\}_{i=1}^n$ with $n$ data points, where $y_i\in\mathbb{R},\mathbf{x}_i\in\mathbb{R}^{p\times 1}$. $p$ denotes the dimension of each sample $\mathbf{x}_i, \mathbf{X}=[\mathbf{1}, \mathbf{x}_1,\mathbf{x}_2,\cdots,\mathbf{x}_n]\in\mathbb{R}^{n\times(p+1)}$ 

(For an intercept term, we can just append a column $\mathbf{1}\in\mathbb{R}^n$ of all 1s to the matrix X) and $\mathbf{Y}=[y_1,y_2,\cdots,y_n]\in\mathbb{R}^{n\times 1}$. 

Additionally, the columns of $\mathbf{X}$ are linear independent, and rank($\mathbf{X}$)=p (assume $n>> p$).

The underlying linear model is

$$
\begin{equation}
\mathbf{Y} = \mathbf{X}\mathbf{\beta}^{\ast} +\mathbf{\epsilon}
\end{equation}
$$

where $\mathbf{\beta}^{\ast}=(\beta_{0},\beta_{1},\cdots, \beta_{p})\in\mathbb{R}^{(p+1)\times 1}$ is the ground-truth coefficient. The error term $\mathbf{\epsilon}=(\epsilon_{1},\cdots, \epsilon_{n})\in\mathbb{R}^{n}$ are as before (i.e. satisfying $\mathbb{E}\[\mathbf{\epsilon}\]=0$ and $Cov(\mathbf{\epsilon})=\sigma^{2}I$).

As before, we aims to estimate $\mathbf{\beta}^{\ast}$ by applying LMSE, and solve the the following optimization problem,

$$
\begin{equation}
\underset{\mathbf{\beta}}{\mathrm{argmin}}\,\frac{1}{n}\Arrowvert \mathbf{Y}-\mathbf{X}\mathbf{\beta}\Arrowvert^{2} 
\end{equation}
$$

there are two ways to estimate $\mathbf{\beta}$, i.e. iterative undated schedule based on gradeint information, and the analitical form.

## 2.2 Learning or estimation method

**Learning as optimization**
let $L(\mathbf{\beta}) = \frac{1}{n}\Arrowvert \mathbf{Y}-\mathbf{X}\mathbf{\beta}\Arrowvert^{2}$

1. update rule:

$$
\begin{equation}
\mathbf{\beta}_{t+1} = \mathbf{\beta}_{t}+\alpha \frac{\partial L(\mathbf{\beta})}{\partial \mathbf{\beta}}
\end{equation}
$$

2. The gradient is

$$
\begin{equation}
\frac{\partial L(\mathbf{\beta})}{\partial \mathbf{\beta}} = \frac{1}{n}\mathbf{X}^{T}(\mathbf{X}\mathbf{\beta}-\mathbf{Y})
\end{equation}
$$

** Analitical form (normal equaiton)**

Take the derivative and set the first-order derivative to 0, and we obtain

$$
\begin{equation}
\hat{\mathbf{\beta}}=(\mathbf{X}^{T}\mathbf{X})^{-1}\mathbf{X}^{T}\mathbf{Y}
\end{equation}
$$

the predicted value are 

$$
\begin{equation}
\mathbf{Y} = \mathbf{X}\hat{\mathbf{\beta}}=\mathbf{X}(\mathbf{X}^{T}\mathbf{X})^{-1}\mathbf{X}^{T}\mathbf{Y}
\end{equation}
$$

- 1. In the analitical form, to solve the LMSE problem requires the inverse of $\mathbf{X}^{T}\mathbf{X}$. To be noted that we assume that *the columns of $\mathbf{X}$ are linear independent*, and thus the matrix $\mathbf{X}^{T}\mathbf{X}$ is a full-rank matrix. The inverse exist. Moreover, the second-order derivative $\frac{\partial^{2}L(\mathbf{\beta})}{\partial \mathbf{\beta}\partial \mathbf{\beta}^{T}}=\mathbf{X}^{T}\mathbf{X}$ is posive-definite, and the solution $\hat{\mathbf{\beta}}$ is unique.

- 2. Both Iterative updated schedule and analitcal can obtain the solution. How to choose the learning schedule in real applications? Well, in the analitical form, it requires the inverse of matrix. The complexity of obtaining the inverse matrix is $O(p^{3})$ in general. When $p$ is very large, e.g. high-dimensional data, analytical form solution is much more expensive. In fact, it can be improved by some techniques, e.g. QR decompositon, etc., but this content beyond the scope of this post, and is not discussed.  

# 3. More perspectives on linear regression

**Geometry interpretion:** let $\mathbf{H}=\mathbf{X}(\mathbf{X}\mathbf{X}^{T})^{-1}\mathbf{X}$, the linear regression fit $\hat{y}\in\mathbb{R}^{n}$ is exactly projection of $\mathbf{y}\in\mathbb{R}^{n}$ onto the linear subspace $span{\mathbf{x}_{1},\cdots,\mathbf{x}_{n}}=col(\mathbf{X})\subset \mathbb{R}^{n}$

> Let $L\subset \mathbb{R}^{n}$ be a linear subspace, i.e. $L=span\{\nu_{1},\cdots, \nu_{k}\}$ for some $\nu_{1},\cdots,\nu_{k}\in\mathbb{R}^{n}$. If $V\in\mathbb{R}^{n\times k}$ contains $\nu_{1},\cdots,\nu_{k}$ on its columns, then

$$
\begin{equation}
span\{\nu_{1},\cdots,\nu_{k}\}=\{\alpha_{1}\nu_{1}+\cdots+\alpha_{k}\nu_{k}:\alpha_{1},\cdots,\alpha_{k}\in\mathbb{R}\}=col(V)
\end{equation}
$$

>The function $F:\mathbb{R}^{n}\rightarrow\mathbb{R}^{n}$ that projects points onto $L$ is called the projection map onto $L$. This map is a linear operator, i.e. $F(x)=P_{L}x$, where $P_{L}\in\mathbb{R}^{n\times n}$ is the projection matrix onto $L$.

>The matrix $P_{L}$ is symmetric: $P_{L}^{T}=P_{L}$, and idempotent: $P_{L}^{2}=P_{L}$. Furthermore, we have

- $P_{L}x = x$ for all $x\in L$, and
- $P_{L}x = 0$ for all $x\perp L$.

For any subspace $L\subseteq \mathbb{R}^{n}$, its orthogonal complement is $L^{\perp}=\{\mathbf{x}\in\mathbb{R}^{n}:\mathbf{x}\perp L\}=\{\mathbf{x}\in\mathbb{R}^{n}:\mathbf{x}\perp\mathbf{\nu} for any \mathbf{\nu}\in L\}$.

Fact: $P_{L}+P_{L^{\perp}}=I$, so that $P_{L^{T}}=I-P_{L}$.

<img src="https://raw.githubusercontent.com/Gwan-Siu/BlogCode/master/A59EBCAB-2B12-4371-B902-D118F9C2223D.png" width = "600" height = "400"/>

for linear regression of $\mathbf{y}$ on $\mathbf{X}$, the residual vector is 

$$
\begin{equation}
\mathbf{r} = \mathbf{y}-\hat{\mathbf{y}} = (I-P_{col(\mathbf{X})})\mathbf{y} = P_{col(\mathbf{X})^{\perp}}\mathbf{y}
\end{equation}
$$

so the residual $\mathbf{r}$ is orthongonal to any $\mathbf{\nu}\in col(\mathbf{X})$. In particular, the residual $\mathbf{r}$ is orthogonal to each of $\mathbf{x_{1}},\cdots,\mathbf{x_{n}}$.

Moreover, the projection map $P_{L}$ onto any linear subspace $L\subseteq\mathbb{R}^{n}$ is always non-expensive, that is, for any points $\mathbf{x,z}\in\mathbb{R}^{n}$, we have $\Arrowvert P_{L}x-P_{L}z\Arrowvert_{2}\leq \Arrowvert x-z\Arrowvert_{2}$.

Therefore, if $\mathbf{y}_1,\mathbf{y}_2 \in \mathbb{R}^n$, and $\hat{\mathbf{y}}_{1},\hat{\mathbf{y}}_{2} \in \mathbb{R}^{n}$ are regression fit, then

$$
\begin{equation}
\Arrowvert \hat{\mathbf{y}}_{1} - \hat{\mathbf{y}}_{2}\Arrowvert_{2}= \Arrowvert P_{L}\mathbf{y}_{1}- P_{L}\mathbf{y}_{2}\Arrowvert_{2}\leq \Arrowvert \mathbf{y}_{1} - \mathbf{y}_{2}\Arrowvert_{2}
\end{equation}
$$

Q: If the columns of $\mathbf{X}$ are not linear independent, $\mathbf{X}$ is not full rank. This implies that $\mathbf{X}^{T}\mathbf{X}$ is singular and the least squares coefficients $\hat{\mathbf{\beta}}$ are not uniquely defined. 

Q: Is $\hat{\mathbf{y}}$ still the projection of $\mathbf{y}$ onto the column space of $\mathbf{X}$? Yes, but just more than one way to express that projection in terms of the column vectors of $\mathbf{X}$.

**Mark:** The non-full-rank case accurs in most applications, because the input vector might be encoded by a redudant fashion. In signal and image analysis, rank deficiencies usually accurs in the case that the dimension of data $p$ excess the number of data $n$.

**How to solve this issue?**

- the features are typically reduced by filtering.
- filtering is controlled by regularization.

# 3. Connection between the univariate regression and multivariate regression

## 3.1 Univariate regression review

We hold the orthogonality assumption in the above assumption. In the rest part, we go further to explain why we need this. 

In the univariate linear regression, let's consider the simplest the coefficients of $\mathbf{y} \in \mathbb{R}^n$ on a single predictor $\mathbf{x} \in \mathbb{R}^n$ as 

$$
\begin{equation}
\hat{\beta}=\frac{<\mathbf{x}, \mathbf{y}>}{\Arrowvert \mathbf{x}\Arrowvert^{2}}
\end{equation}
$$

Given $p$ predictor variable $\mathbf{x}_{1},\cdots,\mathbf{x}_{p}\in\mathbb{R}^{n}$, the univariate linear regression coefficients of $\mathbf{y}$ on $\mathbf{x}_{i}$ is 

$$
\begin{equation}
\hat{\beta}_{j}=\frac{<\mathbf{x}_{j},\mathbf{y}>}{\Arrowvert\mathbf{x}_{j}\Arrowvert^{2}}
\end{equation}
$$

Note: if $\mathbf{x}_{1},\cdots,\mathbf{x}_{p}$ are orthogonal, we can estimate $\beta_{j}$ in the multivariate regression in terms of $\mathbf{y}$ on $\mathbf{x}_{j}$ only.

Next, we consider the intercept term, the coefficient of $\mathbb{x}$ can obtained by implementing two steps:

- 1. Regress $\mathbf{x}$ on $\mathbf{1}$, yielding the coefficient

$$
\begin{equation}
\frac{<\mathbf{1}, \mathbf{x}>}{\Arrowvert \mathbf{x}\Arrowvert^{2}}=\frac{<\mathbf{1}, \mathbf{x}>}{n}=\hat{\mathbf{x}}
\end{equation}
$$

and the residual $\mathbf{z}=\mathbf{x}-\mathbf{\hat{x}}\mathbf{1}\in\mathbb{R}^{n}$

- 2. regress $\mathbf{y}$ on $\mathbb{z}$, yielding the coefficient

$$
\begin{equation}
\hat{\beta}_{1}=\frac{<\mathbf{z},\mathbf{y}>}{\Arrowvert \mathbf{z}\Arrowvert^{2}}=\frac{<\mathbf{x}-\hat{x}\mathbf{1}, \mathbf{y}>}{\Arrowvert \mathbf{x}-\hat{x}\mathbf{1}\Arrowvert^{2}}
\end{equation}
$$

## 3.2 Multivariate regression by orthogonalization

We can extend this idea to multivariate linear regression of $\mathbf{y}\in\mathbb{R}^{n}$ on $\mathbf{X}=\[\mathbf{x}_{1},\cdots,\mathbf{x}_{p}\]\in\mathbb{R}^{N\times p}$. Consider the p-step procedure:

- 1. Let $\mathbf{z}_{1}=\mathbf{x}_{1}$.
- 2. (Normalization): For $j=2,\cdots,p$, regress $\mathbf{x}_{j}$ onto $\mathbf{z}_{1},\cdots,\mathbf{z}_{j-1}$ to get coefficients $\hat{\gamma}_{jk}=\frac{<\mathbf{z}_{k}, \mathbf{x}_{j}>}{\Arrowvert \mathbf{x}_{j}\Arrowvert^{2}_{2}}$ for $k=1,\cdots,j-1$, and residual vector

$$
\begin{equation}
\mathbf{z}_{j} = \mathbf{x}_{j}-\sum_{k=1}^{j-1}\hat{\gamma}_{jk}\mathbf{z}_{k}
\end{equation}
$$

- 3. Regress $\mathbf{y}$ on $\mathbf{z}_{p}$ to get the coefficient $\hat{\beta}_{p}$

***Claim:* the output $\hat{\beta}$ of this algorithm is exactly the coefficient in the multivariate linear regression of $\mathbf{y}$ on $\mathbf{x}_{1},\cdots,\mathbf{x}_{p}$.

## 3.3. Correlated data and variance inflation

Suppose $\mathbf{x}_{1},\mathbf{x}_{2},\cdots,\mathbf{x}_{n}$ are correlated, this make the predicted coefficient $\hat{\mathbf{\beta}}_{j}=\frac{<\mathbf{z}_{j}, \mathbf{y}>}{\Arrowvert \mathbf{z}_{j}\Arrowvert_{2}^{2}}$ unstable, as the denominator is very small. 

We can explicit compute the variance of the $j$-th multiple regression:

$$
\begin{equation}
Var(\hat{\beta}_{j})=\frac{Var(<\mathbf{z}_{j},\mathbf{y}>)}{\Arrowvert\mathbf{z}_{j}\Arrowvert_{2}^{4}}=\frac{\Arrowvert \mathbf{z}_{j}\Arrowvert_{2}\sigma^{2}}{\Arrowvert\mathbf{z}_{j}\Arrowvert_{2}^{4}}=\frac{\sigma^{2}}{\Arrowvert\mathbf{z}_{j}\Arrowvert_{2}^{2}}
\end{equation}
$$

we can see that the correlated variables inflates the variance of multiple regression coefficients. We can explain it based on the Z-score for the $j$-th multiple regression:

$$
\begin{equation}
Z_{j}=\frac{\hat{\beta}_{j}}{\sqrt{Var(\hat{\beta}_{j})}}=\frac{\hat{\beta}_{j}}{\sigma}\cdot \Arrowvert \mathbf{z}_{j}\Arrowvert_{2}
\end{equation}
$$

so if $\mathbf{x}_{j}$ is highly correlated, its regression coefficients will likely be not significant,(because Z-score is small).

## 3.4 Shortcomings of regression

- 1. Predicitve ability: the linear regression model do not predict well, especially when $p$ is large.
- 2. Interpretative ability: in some case, we need to select a smaller subset that have strongest effects on the output.

# 4. More perspectives on linear regression

## 4.1. The Gauss-Markov theorem

The Gauss-markov theorem states that ordinary least square estimation has the smallest mean squared error of all linear estimations with no bias. (To be noted: smallest MSE within the class of linear unbiased setimator.)

From the persepctive of biased-variance decomposition, consider the mean squared error of an estimator $\hat{\mathbf{\beta}}$ in estimating $\beta$:

$$
\begin{equation}
\begin{split}
MSE(\hat{\beta}) &= \mathbb{E}[(\hat{\mathbf{\beta}}-\beta)^{2}] \\
&= Var(\hat{\mathbf{\beta}})+ (\mathbb{E}[\hat{\mathbf{\beta}}]-\mathbf{\beta})^{2} \\
&= Var(\hat{\mathbf{\beta}}+0\,(unbiased)
\end{split}
\end{equation}
$$

**Note:** However, there may exist a biased estamation which can reduce the variance by a great margin with a little payoff of increasing MSE, because such estimators are biased. Specifically, these estimators shrink or set some coefficients in some dimensions to zero. 


To investigate the sampling propoerties of $\hat{\beta}$, we assume that the observations $y_{i}$ are uncorrelated and have constant variance $\sigma^{2}$, and the $\mathbf{x}_{i}$ are fixed.

The variance-covariance matrix of the least squrares parameter estimation is given by 

$$
\begin{equation}
    Var(\mathbf{\beta})=(\mathbf{X}^{T}\mathbf{X})^{-1}\sigma^{2}
\end{equation}
$$
 
where the variance $\sigma^{2}$ is estimated by

$$
\begin{equation}
\hat{\sigma^{2}} =\frac{1}{N-p-1}\sum_{i=1}^{N}(y_{i}-\hat{y}_{i})^{2}
\end{equation}
$$

This is the unbiased estimation for variance $\sigma^{2}$, i.e. $\mathbb{E}\[\sigma^{2}\]=\sigma^{2}$.

Proof:

$$
\begin{equation}
\begin{split}
Var(\hat{\mathbf{\beta}}) &= Var((\mathbf{X}^{T}\mathbf{X})^{-1}\mathbf{X}^{T}\mathbf{y}) \\
&= Var((\mathbf{X}^{T}\mathbf{X})^{-1}\mathbf{X}^{T}(\mathbf{X}\mathbf{\beta}+\mathbf{\epsilon})) \\
&=Var((\mathbf{X}^{T}\mathbf{X})^{-1}\mathbf{X}^{T}\mathbf{X}\mathbf{\beta}+\mathbf{X}^{T}\mathbf{X})^{-1}\mathbf{X}^{T}\mathbf{\epsilon}) \\
&=\mathbf{\beta} + Var(\mathbf{X}^{T}\mathbf{X})^{-1}\mathbf{X}^{T}\mathbf{\epsilon})\quad (\text{Since } \mathbf{\beta}\text{ is treated as a constant in the frequentist approach.})
\end{split}
\end{equation}
$$

Now, we have $Var(\mathbf{X}^{T}\mathbf{X})^{-1}\mathbf{X}^{T}\mathbf{\epsilon}) = \mathbb{E}^{2}\[(\mathbf{X}^{T}\mathbf{X})^{-1}\mathbf{X}^{T}\mathbf{\epsilon}\]-(\mathbb{E}\[(\mathbf{X}^{T}\mathbf{X})^{-1}\mathbf{X}^{T}\mathbf{\epsilon}\])^{2}$

Since $\mathbb{E}\[(\mathbf{X}^{T}\mathbf{X})^{-1}\mathbf{X}^{T}\mathbf{\epsilon}\]=0$, then

$$
\begin{equation}
\begin{split}
Var(\mathbf{X}^{T}\mathbf{X})^{-1}\mathbf{X}^{T}\mathbf{\epsilon}) &= \mathbb{E}[[(\mathbf{X}^{T}\mathbf{X})^{-1}\mathbf{X}^{T}\mathbf{\epsilon}][(\mathbf{X}^{T}\mathbf{X})^{-1}\mathbf{X}^{T}\mathbf{\epsilon}]^{T}] \\
&= \mathbb{E}[(\mathbf{X}^\{T}\mathbf{X})^{-1}\mathbf{X}^{T}\mathbf{\epsilon}\mathbf{\epsilon}^{T}\mathbf{X}(\mathbf{X}^{T}\mathbf{X})^{-1}]\\
&=(\mathbf{X}^{T}\mathbf{X})^{-1}\mathbf{X}^{T}\mathbb{E}[\mathbf{\epsilon}\mathbf{\epsilon}^{T}]\mathbf{X}(\mathbf{X}^{T}\mathbf{X})^{-1} \\
&= \sigma^{2} \mathbf{X}^\{T}\mathbf{X})^{-1}\mathbf{X}^{T}\mathbf{X}(\mathbf{X}^{T}\mathbf{X})^{-1} \\
&= \sigma^{2} \mathbf{X}^\{T}\mathbf{X})^{-1}
\end{split}
\end{equation}
$$


## 4.2 Prediction error and mean squre error

Suppose that we use $\hat{f}$ to predict $f$,  we would like to predict $y_{0}$ via $\hat{f}(x_{0})$.

Question: how does the prediction error(PE) relate to the mean square error (MSE)?

$$
\begin{equation}
\begin{split}
PE(\hat{f}(x_{0})) &= \mathbb{E}[(y_{0}-\hat{f}(x_{0}))^{2}] \\
&= \mathbb{E}[(y_{0}-f(x_{0})+f(x_{0})-\hat{f}(x_{0}))^{2}] \\
&= \mathbb{E}[(y_{0}-f(x_{0}))^{2}]+\mathbb{E}[(f(x_{0})-\hat{f}(x_{0}))^{2}] +2\mathbb{E}[(y_{0}-f(x_{0}))(f(x_{0})-\hat{f}(x_{0}))] \\
&=\sigma^{2}+MSE(\hat{f}(x_{0}))
\end{split}
\end{equation}
$$

since $\mathbb{E}[(y_{0}-f(x_{0}))]=0$, and $MSE(\hat{f}(x_{0}))=[Bias(\hat{f}(x_{0}))]^{2}+Var(\hat{f}(x_{0}))$, we look at the PE and MSE across all the input $\mathbf{x}_{1},\cdots,\mathbf{x}_{n}$., and we have

$$
\begin{equation}
PE(\hat{f})=\frac{1}{n}\sum_{i=1}^{n}PE(\hat{f}(x_{i})),\quad MSE(\hat{f})=\frac{1}{n}\sum_{i=1}^{n}MSE(\hat{f}(x_{i}))
\end{equation}
$$


PE on all input data is 

$$
\begin{equation}
\begin{split}
PE(\hat{f})&=\sigma^{2} + MSE(\hat{f}) \\
&= \sigma^{2}+\frac{1}{n}\sum_{i=1}^{n}[Bias(\hat{f}(x_{i}))]^{2}+\frac{1}{n}\sum_{i=1}^{n}Var(\hat{f}(x_{i}))
\end{split}
\end{equation}
$$

if $\hat{ƒ}(x_{i})=x_{i}^{T}\beta$, then

$$
\begin{equation}
\begin{split}
PE(\hat{f})&=\sigma^{2}+\frac{1}{n}\sum_{i=1}^{n}[Bias(x_{i}^{T}\beta)]^{2}+\frac{1}{n}\sum_{i=1}^{n}Var(x_{i}^{T}\beta) \\
&= \sigma^{2} + 0 + \frac{p\sigma^{2}}{n}
\end{split}
\end{equation}
$$

how to derive the last term?

$$
\begin{equation}
\begin{split}
frac{1}{n}\sum_{i=1}^{n}Var(x_{i}^{T}\beta) &= \frac{1}{n}trace(Var(\mathbf{X}\hat{\beta})) \\
&=\frac{1}{n}trace(Var(\mathbf{X}(\mathbf{X}^{T}\mathbf{X})^{-1}\mathbf{X}^{T}\mathbf{y}))\quad (H=\mathbf{X}^{T}\mathbf{X})^{-1}\mathbf{X}^{T}) \\
&=\frac{1}{n}trace(\mathbf{H}\sigma^{2}\mathbf{I}\mathbf{H}) \\
&=\frac{\sigma^{2}}{n}trace(\mathbf{H}) \\
&=\frac{\sigma^{2}}{n}trace(\mathbf{X}(\mathbf{X}^{T}\mathbf{X})^{-1}\mathbf{X}^{T})\\
&=\frac{\sigma^{2}}{n}p
\end{split}
\end{equation}
$$

For linear regression, its prediction error is just $\sigma^{2}+\frac{p\sigma^{2}}{n}$. The second term is the additive variance, i.e. $\frac{1}{n}\sum_{i=1}^{n}Var(x_{i}^{T}\hat{\beta})$. This means that each additional predictor variable will add the same amount of variance $\sigma^{2}/n$, regardless of whether its true coefficient is large or small.

Therefore, we can shrink some coefficients to zero by introducing bias a little so as to reduce a larger variance.



















































