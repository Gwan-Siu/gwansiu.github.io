---
layout:     post
title:      "Naive Bayes and Logistics Regression"
date:       2018-06-17 12:00:00
author:     "GwanSiu"
catalog: true
tags:
    - Statistics and Bayesian Analysis
---

## 1. Naive Bayes Algorithm

Naive bayes algorithm is supervised learning algorithm, it has two formulations for different inputs, such as **discrete-valued inputs** and **continuous-valued inputs.** MAP is usually used in parameters estimation of naive bayes. 

In statistics machine learning, our goal is to inference class for new sample, and the posterior distribution is proportional to the likelihood multiplied by prior. $P(Y\arrowvert X_{new}, X_{train})=\frac{P(X_{new}\arrowvert X_{train} Y)P(Y)}{P(X)}$. In Naives Bayes, the assumption is conditional independence: 

$$
\begin{equation}
(\forall i,j,k)P(X=x_{i}\arrowvert Y=y_{j}, Z=z_{k})=P(X=x_{i}\arrowvert Z=z_{k})
\end{equation}
$$

Now, let we see more detail about naives bayes. 

### 1.1 Naive Bayes for Discrete-Valued Inputs
In general, we assume that Y is any discrete-valued variable, and attributes $X_{1},...,X_{n}$ are any discrete or realvalued attributes. Our goal is to train a classifier that will output the probability
distribution over possible values of Y, for each new instance X that we ask it to classify. 

The expression for the probability that Y will take on its kth possible
value, according to Bayes rule, is：

$$
\begin{equation*}
P(Y=y_{k}\arrowvert X_{1},...,X_{n})=\frac{P(X_{1},...,X_{n}\arrowvert Y=y_{k})}{\sum_{j}P(Y=y_{j})P(X_{1},...,X_{n}\arrowvert Y=y_{j})}
\end{equation*}
$$

where the sum is taken over all possible values y j of Y. Now, we hold the assumption of conditional independence. The formulation can be rewrite as:

$$
\begin{equation*}
P(Y=y_{k}\arrowvert X_{1},...,X_{n})=\frac{\prod_{i}^{n}P(X_{i}\arrowvert Y=y_{k})}{\sum_{j}P(Y=y_{j})\prod_{i}^{n}P(X_{i}\arrowvert Y=y_{j})}
\end{equation*}
$$



### 1.2 Naive Bayes for Continuous-Valued Inputs
### 1.3 Regularization
#### 1.3.1 Added- 1 Smooth
#### 1.3.2 Added- $\lambda$ Smooth
### 1.4 Appendix of Naive Bayes
#### 1.4.1 Bernoulli Naive Bayes
#### 1.4.2 Multinomial Naive Bayes
#### 1.4.3 Gaussian Naive Bayes

## 2. Logistic Regression and Naive Gaussian Bayes
### 2.1 Discriminative Model and Generative Model
### 2.2 Logistic Regression and Gaussian Naive Bayes
### 2.3 Learning Process
#### 2.3.1 Two Classes
#### 2.3.2 Multi Classes

## 3.Summation of Relation Between Naives Classifier and Logistic Regression
## 4. Summary


