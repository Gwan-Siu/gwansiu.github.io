---
layout:     post
title:      "Support Vector Machine"
date:       2018-12-27 12:00:00
author:     "GwanSiu"
catalog: true
tags:
    - Machine Learning
---

## 1. Support Vector Machine

Let's consider two classes classification problem, and there are two point sets, each of which is $N$ point. One set of points lie in a hyperplane $H_{1}:\boldsymbol{x}_{i}\boldsymbol{w}+b=1$. Another set of points lie in a hyperplane $H_{2}:\boldsymbol{x}_{i}\boldsymbol{w}+b=-1$, where $\boldsymbol{w}$ is normal vector and the perpendicular distance from the origin are $\frac{\vert 1-b\vert}{\Arrowvert \boldsymbol{w}\Arrowvert}$ and $\frac{\vert -1-b\vert}{\boldsymbol{\Arrowvert w\Arrowvert}}$. $d_{+}(d_{-})$ denotes the shorest distance from the separating hyperplane to the closest positive(negative) example. Thus, $d_{+}=d_{-}=\frac{1}{\Arrowvert w\Arrowvert}$ and the margin between these 2 hyperplanes is $\frac{2}{\Arrowvert w\Arrowvert}$. Support vector machine(SVM) simply seeks for the largest margin, for simplication, we firstly consider linear separable case, and the formulation of linear separable SVM is

$$
\begin{align}
&\max_{\boldsymbol{w},b} \frac{2}{\Arrowvert w\Arrowvert} \\
&\textbf{s.t. } y_{i}(\boldsymbol{w}\boldsymbol{x}+b) \geq 1
\end{align}
$$

we can rewrite the above formulation

$$
\begin{align}
&\max_{\boldsymbol{w},b} \frac{1}{2}\Arrowvert w\Arrowvert \\
&\textbf{s.t. } y_{i}(\boldsymbol{w}\boldsymbol{x}+b) \geq 1
\end{align}
$$

in the case of constrint optimization problem, we solve it by its dual form, and the Lagrangian function is

$$
\begin{align}
\mathcal{L}(\boldsymbol{w}, b, \alpha) &= \frac{1}{2}\Arrowvert \boldsymbol{w}\Arrowvert^{2} + \displaystyle{\sum_{i=1}^{N}}\alpha_{i}(1-y_{i}(\boldsymbol{w}\boldsymbol{x}+b)) \\
&s.t. \alpha_{i}\geq 0, \text{for} i=1,...,n
\end{align}
$$

the primal problem is 

$$
\begin{equation}
\min_{\boldsymbol{w}}\max_{\alpha\geq 0} \mathccal{L}(\boldsymbol{w}, \alpha)
\end{equation}
$$

and the dual problem is

$$
\begin{equation}
\max_{\alpha\geq 0}\min_{\boldsymbol{w}} \mathccal{L}(\boldsymbol{w}, \alpha)
\end{equation}
$$

the weakly duality theorem is

$$
\begin{equation}
d^{\ast}= \max_{\alpha\geq 0}\min_{\boldsymbol{w}} \mathccal{L}(\boldsymbol{w}, \alpha) \leq \min_{\boldsymbol{w}}\max_{\alpha\geq 0} \mathccal{L}(\boldsymbol{w}, \alpha)=p^{\ast}
\end{equation}
$$

because the objective function is convex, we have strong duality theorem that iff there exist a saddle point of $\mathccal{L}(\boldsymbol{w}, \alpha)$, then

$$
\begin{equation}
d^{\ast} =d^{\ast}
\end{equation}
$$

if there exist a saddle point of $\mathccal{L}(\boldsymbol{w}, \alpha)$, then the saddle point satisfies the "Karush-Kuhn-Tucker"(KKT) conditions:

$$
\begin{align}
\frac{\partial\mathcal{L}}{\partial \boldsymbol{w}}&=0, \text{ for }i=1,...,N \\
\frac{\partial\mathcal{L}}{\beta_{i}}&=0, \text{ for }i=1,...,N \\
\alpha_{i}g_{i}(\boldsymbol{w})&=0, \text{ for }i=1,...,N \text{ Complementary slackness}
g_{i}(\bioldsymbold{w}) &\leq 0, \text{ for }i=1,...,N \text{ Primal feasibility}
\alpha_{i} &\geq 0, \text{ for } i=1,...,N \text{ Dual feasibility}
\end{align}
$$

Back to the dual form of SVM, we have

$$
\begin{align}
\frac{\partial\mathcal{L}}{\partial \boldsymbol{w}}&=\boldsymbol{w}-\displaystyle{\sum_{i=1}^{N}\alpha_{i}y_{i}x_{i}} \\
\Rightarrow& \boldsymbol{w}=\displaystyle{\sum_{i=1}^{N}\alpha_{i}y_{i}x_{i}} \\
\frac{\partial\mathcal{L}}{\partial b}=\displaystyle{\sum_{i=1}^{N}\alpha_{i}y_{i}}=0
\end{align}
$$

subtitute the result back to the dual form of SVM, we have

$$
\begin{align}
\max_{\alpha}f(\alpha) &= \displaystyle{\sum_{i=1}^{N}\alpha_{i}-\frac{1}{2}\sum_{i=1}^{N}\sum_{j=1}^{N}\alpha_{i}\alpha_{j}y_{i}y_{j}x_{i}x_{j}} \\
s.t. &\alpha_{i} \geq 0, \text{ for } i=1,...,N \\
&\displaystyle{\sum_{i=1}^{N}\alpha_{i}y_{u}=0}
\end{align}
$$

Once we have the Lagrance multipliers ${\alpha_{i}}$, we can decide the parameter vector $\boldsymbol{w}$ as a linear combination of small data points.

$$
\begin{equation}
\boldsymbol{w}=\sum_{i\in SV}\alpha_{i}y_{i}x_{i}
\end{equation}
$$

This sparse "representation" can be viewed as data compression as in the construction of KNN classifier. In addition, to compute the weight ${\alpha_{i}}$ and to use support vector machine, we need to specify the inner products between the examples $x_{i}x_{i}^{T}$.

We make decisions by comparing each new example $z$ with only the support vectors:

$$
\begin{equation}
y^{\ast}=\text{sign}(\displaystyle{\sum_{i\in SV}\alpha_{i}y_{i}(x_{i}^{T}z)+b})
\end{equation}
$$

For non-linear separable problems, we allow "error" $\xi_{i}$ in classification, it is based on the output of the discriminant function $w^{T}x+b$. Now, we have a slightly different optimization problem

$$
\begin{align}
&\max_{\boldsymbol{w},b} \frac{1}{2}\Arrowvert w\Arrowvert +C\sum_{i=1}^{N}\xi_{i} \\
&\textbf{s.t. } y_{i}(\boldsymbol{w}\boldsymbol{x}+b) \geq 1 \\
\xi_{i}\geq 0, \text{ for }i=1,...,N
\end{align}
$$

the dual form is 

$$
\begin{align}
\max_{\alpha}f(\alpha) &= \displaystyle{\sum_{i=1}^{N}\alpha_{i}-\frac{1}{2}\sum_{i=1}^{N}\sum_{j=1}^{N}\alpha_{i}\alpha_{j}y_{i}y_{j}x_{i}x_{j}} \\
s.t. & 0\leq \alpha_{i} \leq C, \text{ for } i=1,...,N \\
&\displaystyle{\sum_{i=1}^{N}\alpha_{i}y_{u}=0}
\end{align}
$$

The `C` parameter trades off correct classification of training examples against maximization of the decision function’s margin. For larger values of `C`, a smaller margin will be accepted if the decision function is better at classifying all training points correctly. A lower C will encourage a larger margin, therefore a simpler decision function, at the cost of training accuracy. In other words`C` behaves as a regularization parameter in the SVM.



## 2. Kernel Method

for non-linear decision boundary, we adopt kernel trick to replace the inner product $x_{i}x_{i}^{T}$, the formulation can be rewritten

$$
\begin{align}
\max_{\alpha}f(\alpha) &= \displaystyle{\sum_{i=1}^{N}\alpha_{i}-\frac{1}{2}\sum_{i=1}^{N}\sum_{j=1}^{N}\alpha_{i}\alpha_{j}y_{i}y_{j}k(x_{i},x_{j})} \\
s.t. & 0\leq \alpha_{i} \leq C, \text{ for } i=1,...,N \\
&\displaystyle{\sum_{i=1}^{N}\alpha_{i}y_{u}=0}
\end{align}
$$

**Theorem(Mercer Kernel):** Let $K:\mathbb{R}^{n}\times \mathbb{R}^{n}\rightarrow \mathbb{R}$ be given. Then for $K$ to be a valid(mercer) kernel, it is necessary and sufficient that for any ${x_{1},...,x_{m}},m<\infty$, the corresponding kernel matrix is symmetric positive semi-definite.


- Linear kernel

$$
\begin{equation}
K(x,x^{prime})= x^{T}x^{\prime}
\end{equation}
$$

- Polynomial Kernel

$$
\begin{equation}
K(x, x^{\prime}) = (1+x^{T}x^{\prime})^{d}
\end{equation}
$$

- Radial Basis Kernel

$$
\begin{equation}
K(x,x^{\prime}) = \text{exp}(-\frac{1}{2}\Arrowvert x-x^{\prime}\Arrowvert^{2})
\end{equation}
$$ 

## 3. Maximum Entropy Discrimination

Maximum entropy discrimination is built on the topic of model selection followed the maximum entropy principle. The idea is followed: the difference between generative model and dicriminant model depending on the estimation criterion used for adjusting the model parameters and structure. Generative approaches rely on a full joint probability distribuition over examples and class label, while discriminative methods only model the conditional relation of a label given the example is relevant(or model a decision boudary directly). We interprete the posterior probability of a label given example as a parametric decision boundary, that is the bayes predictor

$$
\begin{equation}
h_{1}(x;p(w))=\arg\max_{y\in \gamma(x)}\int p(w)f(x,y;w)\mathrm{d}w
\end{equation}
$$

specifically, we adopt the maximum entropy principle that we want to select the best model that the model with largest entropy given our observed data.

**Maximum Entropy Dicriminantion**

Given data set $\mathcal{D}={(x_{i}, y_{i})}_{i=1}^{N}$, find

$$
\begin{align}
Q_{ME} &= \arg\max H(Q) \\
s.t. & y_{i}(f(x_{i}))_{ME}\geq \xi_{i}, \forall i\\
&\xi_{i}\geq,\forall i
\end{align}
$$

however, the overall ME formulation hold a strict assumption that the training examples can be separated with the specified margin. This assumption may be violated in practice. In additon, the prior may prefer to some parameter values over others, this requires us to incorporate a prior distribution $Q_{0}(f)$. we instead to minimize relative entropy discrimination

$$
\begin{align}
Q_{MRE} &=\arg\min KL(Q\Arrowvert Q_{0}) + U(\xi) \\
s.t. & y_{i}(f(x_{i}))_{ME}\geq \xi_{i}, \forall i\\
&\xi_{i}\geq,\forall i
\end{align}
$$

$Q_{MRE}$ is a copnvex problem, and thus there exists an unique solution. Also, we have a theorem


the posterior distribution:

$$
\begin{equation}
Q_{MRE} = \frac{1}{Z(\alpha)} Q_{0}(w)\displaystyle{exp{\sum_{i=1}^{N}\alpha_{i}y_{i}f(x_{i};w)}}
\end{equation}
$$

Dual Optimization problem:

$$
\begin{align}
\max_{\alpha}& -\log Z(\alpha)-U^{\ast}(\alpha) \\
s.t.& \alpha_{i}(y)\geq 0, \forall i
\end{algin}
$$

$U^{\ast}(\alpha)$ is the conjugate of the $U(\cdot)$, i.e. $U^{\ast}(\alpha)=\sup_{\xi}(\sum_{i,y}\alpha_{i}(y)\xi_{i}-U(\xi))$


where $\alpha_{i}\geq 0$ are Lagrange multipliers. The idea is very similar to support vector in SVM.

We can find $Q_{M}$: start with $\alpha_{i}=0$ and follow gradient of unsatisfied constraints, iterative ascent on $J(\alpha)$ until convergence.

**Example on SVM**

For $f(x)=w^{T}x+b, Q_{0}(w)=\mathcal{N}(0, I), Q_{0}(b)=\text{non-informative prior}$, the Lagrange multipliers $\alpha$ are obtained by maximizing $J(\alpha)$ subject to $0\leq \alpha_{t}\leq C$ and $\sum_{t}\alpha_{t}y_{t}=0$, where

$$
\begin{equation}
J(\alpha) = \sum_{t} [\alpha_{t} + \log (1-\frac{a_{t}}{C})] -\frac{1}{2}\sum_{s,t}\alpha_{s}\alpha_{t}y_{s}y_{t}x_{s}^{T}x_{t} 
\end{equation}
$$ 

- Separable $\Rightarrow$  SVM recovered exactly
- Inseparable $\Rightarrow$ SVM recovered with different misclassification penalty


## 4. Max-margin Learning


## Reference

[1] lecture note on support vector machine, Eric xing, cmu-10701, fall 2016.

[2] lecture note on advanced topics in Max-Margin Learning, Eric xing, cmu-10701, fall 2016.

[3] [https://scikit-learn.org/stable/auto_examples/svm/plot_rbf_parameters.html](https://scikit-learn.org/stable/auto_examples/svm/plot_rbf_parameters.html)

[4] Burges, Christopher JC. "A tutorial on support vector machines for pattern recognition." Data mining and knowledge discovery 2.2 (1998): 121-167.
