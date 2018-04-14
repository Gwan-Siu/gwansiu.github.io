---
layout:     post
title:      "The completement of real number"
date:       2018-04-14 12:00:00
author:     "GwanSiu"
catalog: true
tags:
    - Analysis
---

>本节主要讲述实数的完备性

## 1. 实数完备性主要体现在两个等价的叙述：

### 1.1 完备性(1)

假设实数列${a_{n}}$具有如下性质：
1. ${a_{n}}$ 单调递增，则:$a_{n}\leq a_{n+1}\forall n.$
2. ${a_{n}}$上方有界，即:$\exists k$ 使得$a_{n}\leq k\forall n$.

则$\displaystyle\lim_{n\rightarrow\infty}\in \mathbb{R}$, or $\exists l\in \mathbb{R}$,使得$\lim_{n\rightarrow}l$.

### 1.2 完备性(2).
设$S\subseteq \mathbb{R}$, 且存在$k\in\mathbb{R}$使得$x\leq k\forall x\in S$, 则$\exists x_{o}\in R$使:

1. $x\leq x_{o}, \forall x\in S$($x_{o}$是$S$的上界)
2. 若$x\leq y\forall x\in S$, 则$y\geq x_{o}$.(任意上界都不会比$x_{o}更小$)。

## 2. 区间套定理
设$I_{n}=[a_{n}, b_{n}]$, $I_{1}\supset I_{2}\supset I_{3}\supset ...$且$\vert I_{n}\vert =b_{n}-a_{n}\rightarrow 0$当$n\rightarrow 0$, 则存在唯一一个点$x_{0}\in R$使得$\cap_{n=1}^{\infty}I_{n}={x_{0}}$.

\textbf{证明:}因$I_{1}\supset I_{2}\supset I_{3}\supset ...$，则${a_{n}}$为递增数列，${b_{n}}$为递减数列，且$a_{n}\leq b_{1}$,$b_{n}\geq a_{1}\forall n$，依照实数的完备性，$\exists a\in\mathbb{R},\lim_{n\rightarrow\infty}a_{n}=a,\exists b\in\mathbb{R}, \lim_{n\rightarrow\infty}b_{n}=b$.

$$
\begin{equation*}
b-a=\lim_{n\rightarrow\infty}b_{n}-\lim_{n\rightarrow\infty}a_{n}=b_{n}-a_{n}=0
\end{equation*}
$$

令$x_{0}=b-a$,则$\cap_{n=1}^{\infty}I_{n}={x_{o}}$.

## 3. Bolzano-Weierstrass 定理: 任意有界数列必定包含收敛子序列
设$x_{n}\in \mathbb{R}$, $\vert x_{n}\vert M\forall n$, 则存在$n_{k}$使得$\lim_{k\rightarrow \infty}x_{n_{k}}$存在。

1. 若$S$为有穷集合(finite set),则数列中必有一数在数列中重复出现无穷次，此重复出现子数列即为收敛子列。
2. 若S是无穷集合，因${x_{n}}$有界，令$S\subset[a_{1},b_{1}]$,$a_{1},b_{1}\in R$, 并令$I_{1}=[a_{1},b_{1}]$. 任取$x_{n_{1}}\in S\cap I$, 将$I_{1}$等分成二等分，因$S$是无穷集合，此二等分闭区间中必有一个含$S$的无穷多元素的集合，令其为$I_{2}=[a_{2},b_{2}]$,则$\vert I_{2}\vert =\vert b_{2}-a_{2}\vert =\frac{b_{1}-a_{1}}{2}$, 任取$x_{n_{2}}\in S\cap I_{2}$, $n_{2}>n_{1}$.

重复上述过程，得到区间$I_{3}=[a_{3},b_{3}]$, $\vert I_{3}\vert=\vert b_{3}-a_{3}\vert=\frac{b_{1}-a_{1}}{2^{2}}$, $I_{3}$含有S中无穷多的元素，任取$x_{3}\in S\cap I, n_{2}>n_{1}$.

依此类推至无穷，得到区间套$I_{k}=[a_{k},b_{k}]$, $\vert I_{k}\vert=b_{k}-a_{k}=\frac{b_{1}-a_{1}}{2^{k-1}}\rightarrow 0$ 当$k\rightarrow \infty$. I_{n}中含有S中无穷多的元素，任取$x_{n_{k}}\in S\cap I_{k}, n_{k}>n_{k-1}$.

依照区间套定理, $\exists x_{o}\in R$ 使得 $\cap_{k=1}^{\infty}I_{k}={x_{o}}$, 又$x_{n_{k}},x_{o}\in I_{k}$. 所以，$\vert x_{n_{k}}-x_{o}\vert < \vert b_{k}-a_{k}\vert <\frac{b_{1}-a_{1}}{2^{k-1}}\rightarrow 0$当$k\rightarrow\infty$.

因此$\lim_{k\rightarrow\infty}x_{n_{k}}=x_{o}$.
