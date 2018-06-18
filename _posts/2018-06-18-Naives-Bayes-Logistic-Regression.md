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

Now, let we see more detail about naives bayes classifier. 

In general, we assume that Y is any discrete-valued variable, and attributes $X_{1},...,X_{n}$ are any discrete or realvalued attributes. Our goal is to train a classifier that will output the probability
distribution over possible values of Y, for each new instance X that we ask it to classify. 

The expression for the probability that Y will take on its kth possible
value, according to Bayes rule, is：

$$
\begin{equation}
P(Y=y_{k}\arrowvert X_{1},...,X_{n})=\frac{P(X_{1},...,X_{n}\arrowvert Y=y_{k})}{\sum_{j}P(Y=y_{j})P(X_{1},...,X_{n}\arrowvert Y=y_{j})}
\end{equation}
$$

where the sum is taken over all possible values y j of Y. Now, we hold the assumption of conditional independence. The formulation can be rewrite as:

$$
\begin{equation}
P(Y=y_{k}\arrowvert X_{1},...,X_{n})=\frac{\prod_{i}^{n}P(X_{i}\arrowvert Y=y_{k})P(Y=y_{k})}{\sum_{j}P(Y=y_{j})\prod_{i}^{n}P(X_{i}\arrowvert Y=y_{j})}
\end{equation}
$$

Equation(3) is the fundamental formulation of naive bayes algorithm for naives bayes algorithm.  Given a
new instance $X_{new} = X_{1},...,X_{n}$, this equation shows how to calculate the probability that $Y$ will take on any given value, given the observed attribute values of $X$ new and given the distributions $P(Y)$ and $P(X_{i}\arrowvert Y)$ estimated from the training data. If we are interested only in the most probable value of Y, then we have the Naive Bayes classification rule(MAP):

$$\begin{equation}
Y\leftarrow \arg\max_{y_{k}} \frac{\prod_{i}^{n}P(X_{i}\arrowvert Y=y_{k})}{\sum_{j}P(Y=y_{j})\prod_{i}^{n}P(X_{i}\arrowvert Y=y_{j})}
\end{equation}
$$

usually, we consider denominator as constant, and equation above can be simplified as:

$$\begin{equation}
Y\leftarrow \arg\max_{y_{k}} P(Y=y_{k})\prod_{i}^{n}P(X_{i}\arrowvert Y=y_{k})
\end{equation}
$$

### 1.1 Naive Bayes for Discrete-Valued Inputs

Assume the n input attributes $X_{i}$ each take on $J$ possible discrete values, and $Y$ is a discrete variable taking on $K$ possible discrete values, then our learning task is to estimage two sets of parameters. The first is:

$$
\begin{equation}
\theta_{ijk} \equiv P(X_{i}=x_{ij}\arrowvert Y=y_{k})
\end{equation}
$$

for each input attribute $X_{i}$, each of its possible values $x_{ij}$, and each of the possible values $y_{k}$ of $Y$. Therefore, there will be nJK such parameters, and note only $n(J-1)K$ of these are independent, given that they must satisfy $\sum_{j}\theta_{ijk}=1$ for each pair of $i,k$ values.

The second parameter set is prior probability over $Y$:

$$
\begin{equation}
\pi_{k} \equiv P(Y=y_{k})
\end{equation}
$$
Note there are $K$ parameters, $(K-1)$ of which are independent.

**Estimation of** $\theta$

Givening a set of training examples D, the MLE for $\theta_{ijk}$ are given by:

$$
\begin{equation}
\tilde{\theta}_{ijk} = \tilde{P}(X_{i}=x_{ij}\arrowvert Y=y_{k}) = \frac{\text{#}D{X_{i}=x_{ij}\wedge Y=y_{k}}}{\text{#}D{Y=y_{k}}}
\end{equation}
$$

where the $\text{#}D{x}$ operator returns the number of elements in the set $D$ that satisfy property $x$.

One danger of this maximum likelihood estimate is that it can sometimes result
in θ estimates of zero, if the data does not happen to contain any training
examples satisfying the condition in the numerator. To avoid this, it is common to use a “smoothed” estimate which effectively adds in a number of additional “hallucinated” examples, and which assumes these hallucinated examples are spread evenly over the possible values of $X_{i}$. This smoothed estimate is given by:

$$
\begin{equation}
\tilde{\theta}_{ijk} = \tilde{P}(X_{i}=x_{ij}\arrowvert Y=y_{k}) = \frac{\text{#}D{X_{i}=x_{ij}\wedge Y=y_{k}}+l}{\text{#}D{Y=y_{k}}+lJ}
\end{equation}
$$

where J is the number of distinct values $X_{i}$ can take on, and $l$ determines the strength of this smoothing(i.e., the number of hallucinated examples is $lJ$). This distribution over the $\theta_{ijk}$ parameters, with equal-valued parameters. If $l$ is set to 1, this approach is called Laplace smoothing.

**Estimation of** $\pi$

Maximum likelihood for $\pi_{k}$ are:

$$
\begin{equation}
\tilde{\pi}=\tilde{P}(Y=y_{k})=\frac{\text{#}D{Y=y_{k}}}{\arrowvert D\arrowvert}
\end{equation}
$$

where $\arrowvert D\arrowvert$ denotes the number of elements in the training set $D$.

Alternatively, we can obtain a smoothed estimate, or equivalently a MAP estimate based on a Dirichlet prior over the $\pi_{k}$ parameters assuming equal priors on each $\pi_{k}$, by using the following expression:

$$
\begin{equation}
\tilde{\pi}=\tilde{P}(Y=y_{k})=\frac{\text{#}D{Y=y_{k}}+l}{\arrowvert D\arrowvert +lK}
\end{equation}
$$

where $K$ is the number of distinct values $Y$ can take on, and $l$ again determines the strength of the prior assumptions relative to the observed data $D$.

### 1.2 Naive Bayes for Continuous-Valued Inputs

In the case of continuous inputs $X$, we use Gaussian distribution to represent $P(X_{i}\arrowvert Y)$. Specifically, for each possible discrete value $y_{k}$ of $Y$, the distribution of each continuous $X_{i}$ is Gassuain, and is defined by a mean and standard deviation specific to $X_{i}$ and $y_{k}$. In order to train such a Naive Bayes classifier we must therefore estimate the mean and standard deviation of each of these Gaussians:

$$
\begin{align}
\mu_{ik} &=\mathbb{E}[X_{i}\arrowvert Y=y_{k}] \\
\sigma_{ik}^{2} &= \mathbb{E}[(X_{i}-\mu_{ik})^{2}\arrowvert Y=y_{k}]
\end{align}
$$

for each attibute $X_{i}$ and each possible value $y_{k}$ of $Y$. These are totally $2nK$ of these parameters, and all of them must be estimated independently. 

Also, we must estimate the prior distribution over Y:

$$
\begin{equation}
\pi_{k} = P(Y=y_{k})
\end{equation}
$$

The above formulation assume that e data X is generated by a mixture of class-conditional (i.e., dependent on the value of the class variable Y) Gaussians. 

Furthermore, the Naive Bayes assumption introduces the additional constraint that the attribute values $X_{i}$ are independent of one another within each of these mixture components. In particular problem settings where we have additional information, we might introduce additional assumptions to further restrict the number of parameters or the complexity of estimating them. 

For example, if we have reason to believe that noise in the
observed $X_{i}$ comes from a common source, then we might further assume that all of the $σ_{ik}$ are identical, regardless of the attribute $i$ or class $k$.

**Maximum Likelihood Estimation of** $\mu_{ik}$:

$$
\begin{equation}
\tilde{\mu}_{ik} = \frac{1}{\sum_{j}\delta(Y^{j}=y_{k})}\sum_{j}X_{i}^{j}\delta(Y^{j}=y_{k})
\end{equation}
$$

where the superscript $j$ refers to the jth training example, and where $\delta(Y=y_{k})$ is 1 if $Y = y_{k}$ and 0 otherwise. Note the role of δ here is to select only those training examples for which $Y = y_{k}$.

**Maximum Likelihood Estimation of** $\delta^{2}_{ik}$

$$
\begin{equation}
\tilde{\sigma}_{ik}^{2} = \frac{1}{\sum_{j}\delta(Y^{j}=y_{k)}}\sum_{j}(X_{i}^{j}-\tilde{\mu}_{ik})^{2}\delta(Y^{j}=y_{k})
\end{equation}
$$

This maximum likelihood estimator is biased, so the minimum variance unbiased
estimator (MVUE) is sometimes used instead. It is

$$
\begin{equation}
\tilde{\sigma}_{ik}^{2} = \frac{1}{\sum_{j}\delta(Y^{j}=y_{k)}-1}\sum_{j}(X_{i}^{j}-\tilde{\mu}_{ik})^{2}\delta(Y^{j}=y_{k})
\end{equation}
$$

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
#### 2.2.1 Logistic Regression
Logistic regression is discriminative algorithm, which learns functions of the form $f:X\rightarrow Y$, or $P(Y\arrowvert X)$ in the case where $Y$ is discrete-valued, and $X=(X_{1},...,X_{n})$ is any vector containing dicrete or continuous variables.

Logistic Regression assumes a parametric form for the distribution $P(Y\arrowvert X)$, then directly estimates its parameters from the training data. The parametric model assumed by Logistic Regression in the case where $Y$ is boolean is:

$$
\begin{align}
P(Y=1\arrowvert X) &=\frac{1}{1+exp(\omega_{0}+\sum_{i=1}^{n}\omega_{i}X_{i})} \\
P(Y=0\arrowvert X) &=\frac{exp(\omega_{0}+\sum_{i=1}^{n}\omega_{i}X_{i})}{1+exp(\omega_{0}+\sum_{i=1}^{n}\omega_{i}X_{i})} 
\end{align}
$$

One highly convenient property of this form for $P(Y\arrowvert X)$ is that it leads to a simple linear expression for classification. To classify any given $X$ we generally want to assign the value yk that maximizes $P(Y = y_{k}\arrowvert X)$. Put another way, we assign the label $Y = 0$ if the following condition holds:

$$
\begin{equation}
1<\frac{P(Y=0\arrowvert X)}{P(Y=1\arrowvert X)}
\end{equation}
$$

substituting from equation(17) and equation(18), we can get:

$$
\begin{equation}
1<exp(\omega_{0}+\sum_{i=1}^{n}\omega_{i}X_{i})
\end{equation}
$$

and taking the natural log of both sides we have a linear classification rule that assigns label $Y = 0$ if $X$ satisfies:

$$
\begin{equation}
1 < \omega_{0}+\sum_{i=1}^{n}\omega_{i}X_{i}
\end{equation}
$$

and assigns Y = 1 otherwise.

Interestingly, the parametric form of $P(Y\arrowvert X)$ used by Logistic Regression is precisely the form implied by the assumptions of a Gaussian Naive Bayes classifier. Therefore, we can view Logistic Regression as a closely related alternative to GNB, though the two can produce different results in many cases.

#### 2.2.2 Gaussian Naive Bayes
Here we can derive the form of $P(Y\arrowvert X)$ entailed by the assumptions of a Gaussian Naive Bayes (GNB) classifier, showing that it is precisely the form used by Logistic Regression.  In particular, consider a GNB based on the following modeling assumptions:

1. $Y$ is boolean, governed by a Bernoulli distribution, with parameter $\pi=P(Y = 1)$.
2. $X = (X_{1},...,X_{n})$, where each $X_{i}$ is a continuous random variable.
3. For each $X_{i}$, $P(X_{i}\arrowvert Y = y_{k})$ is a Gaussian distribution of the form $N(\mu_{ik},\sigma_{i})$.
4. For all $i$ and $j\neq i$, $X_{i}$ and $X_{j}$ are conditionally independent given $Y$.

**Note here we are assuming the standard deviations σi vary from attribute to attribute, but do not depend on** $Y$.

In general, Bayes rules allow us to write:

$$
\begin{equation}
P(Y=1\arrowvert X)=\frac{P(Y=1)P(X\arrowvert Y=1)}{P(Y=1)P(X\arrowvert Y=1)+P(Y=0)P(X\arrowvert Y=0)}
\end{equation}
$$

Dividing both the numerator and denominator by the numerator yields:

$$
\begin{equation}
P(Y=1\arrowvert X) = \frac{1}{1+\frac{P(Y=0)P(X\arrowvert Y=0)}{P(Y=1)P(X\arrowvert Y=1}}
\end{equation}
$$

or equivalently,

$$
\begin{equation}
P(Y=1\arrowvert X) = \frac{1}{1+exp(\ln\frac{P(Y=0)P(X\arrowvert Y=0)}{P(Y=1)P(X\arrowvert Y=1}})
\end{equation}
$$

Due to our conditional independence assumption we can write this:

$$
\begin{align}
P(Y=1\arrowvert X) &= \frac{1}{1+exp(\ln\frac{P(Y=0)}{P(Y=1)}+\sum_{i}\ln\frac{P(X_{i}\arrowvert Y=0)}{P(X_{i}\arrowvert Y=1)})} \\
&= \frac{1}{1+exp(\ln\frac{1-\pi}{\pi}+\sum_{i}\ln\frac{P(X_{i}\arrowvert Y=0)}{P(X_{i}\arrowvert Y=1)})}
\end{align}
$$

Note the final step express $P(Y=0)$ and $P(Y=1)$ in terms of the binomial parameter $\pi$.

Now consider just the summation in the denominator. Given our assumption that $P(X_{i}\arrowvert Y=y_{k})$ is Gaussian, we can expand this term as follows:

$$
\begin{align}
\sum_{i}\ln\frac{P(X_{i}\arrowvert Y=0)}{P(X_{i}\arrowvert Y=1)} &= \sum_{i}\ln\frac{\frac{1}{\sqrt{2\pi\sigma_{i}^{2}}}exp(\frac{-(X_{i}-\mu_{i0})^{2}}{2\sigma_{i}^{2}})}{\frac{1}{\sqrt{2\pi\sigma_{i}^{2}}}exp(\frac{-(X_{i}-\mu_{i1})^{2}}{2\sigma_{i}^{2}})} \\
&=\sum_{i}\ln exp(\frac{(X_{i}-\mu_{i1})^{2}-(X_{i}-\mu_{i0})^{2}}{2\sigma_{i}^{2}}) \\
&=\sum_{i}(\frac{(X_{i}-\mu_{i1})^{2}-(X_{i}-\mu_{i0})^{2}}{2\sigma_{i}^{2}}) \\
&=\sum_{i}(\frac{(X_{i}^{2}-2X_{i}\mu_{i1}+\mu_{i1}^{2})-(X_{i}-2X_{i}\mu_{i0}+\mu_{i0}^{2}}{2\sigma_{i}^{2}}) \\
&=\sum_{i}(\frac{2X_{i}(\mu_{i0}-\mu_{i1})+\mu_{i1}^{2}-\mu_{i0}^{2}}{2\sigma_{i}^{2}}) \\
&=\sum_{i}(\frac{\mu_{i0}-\mu_{i1}}{\sigma_{i}^{2}}X_{i}+\frac{\mu_{i1}^{2}-\mu_{i0}^{2}}{2\sigma_{i}^{2}})
\end{align}
$$

we can write the form, which is the same as the form of Logistic Regression:

$$
\begin{equation}
P(Y=1\arrowvert X) = \frac{1}{1+exp(\ln\frac{1-\pi}{\pi}+\sum_{i}(\frac{\mu_{i0}-\mu_{i1}}{\sigma_{i}^{2}}X_{i}+\frac{\mu_{i1}^{2}-\mu_{i0}^{2}}{2\sigma_{i}^{2}}))}
\end{equation}
$$

where $\omega_{0}=\ln\frac{1-\pi}{\pi}+\sum_{i}\frac{\mu_{i1}^{2}-\mu_{i0}^{2}}{2\sigma_{i}^{2}}$ and $\omega_{i}=\frac{\mu_{i0}-\mu_{i1}}{\sigma_{i}^{2}}$ for i=1,...,n.

Hnece, we have:

$$
\begin{equation}
P(Y=0\arrowvert X) = \frac{exp(\ln\frac{1-\pi}{\pi}+\sum_{i}(\frac{\mu_{i0}-\mu_{i1}}{\sigma_{i}^{2}}X_{i}+\frac{\mu_{i1}^{2}-\mu_{i0}^{2}}{2\sigma_{i}^{2}}))}{1+exp(\ln\frac{1-\pi}{\pi}+\sum_{i}(\frac{\mu_{i0}-\mu_{i1}}{\sigma_{i}^{2}}X_{i}+\frac{\mu_{i1}^{2}-\mu_{i0}^{2}}{2\sigma_{i}^{2}}))}
\end{equation}
$$

### 2.3 Learning Process
#### 2.3.1 Two Classes
#### 2.3.2 Multi Classes

## 3.Summation of Relation Between Naives Classifier and Logistic Regression
## 4. Summary


