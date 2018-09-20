---
layout:     post
title:      "数学分析笔记系列(1)-实数的完备性"
date:       2018-09-20 12:00:00
author:     "GwanSiu"
catalog: true
tags:
    - Analysis
---

>九月，便开始了PhD的征途，在我面前的将是知识的星辰大海。我从事的是low-level image processing以及machine learning领域，强有力的数学工具将有助于问题的思考以及让我对繁杂数学符号少一些恐惧。九月中旬，我开始旁听香港城市大学Ciarlet教授的泛函分析课程，弱鸡的实数分析让我很难follow他的进度。因此，我才打算在科研空闲之余重新捡起数学分析相关知识。
>本次笔记系列主要参考书目:
>[1]. “Analysis I&II”, 3rd edition, Terence Tao
>[2]. "Understanding Analysis", 2nd edition, Stephen Abbott


大部分数学分析的课堂都是从一个经典的问题开始:为什么我们需要具有完备性的实数系？且我们指出一个事实: `There is no rational number whose square is 2`. 而在Tao的书中，他使用Peano Axioms构造自然数系，使得自然数加法封闭和乘法封闭。通过引入减法，将自然数系拓展到整数系，这意味着整数系减法是封闭。同样，通过引入除法，将整数系拓展到有理数系。因此，有理数是加减乘除法都封闭的数系。不过，我们依旧不能找到$r\in\mathbb{Q}$, such that $r^{2}=2$. 这意味着有理数系是有“洞”的，我们需要构造一个更“强”的数系去填补有理数系的“洞”。

## 1. 实数完备性主要体现在两个等价的叙述：

### 1.1 完备性(1)

假设实数列$(a_{n})_{n=1}^{\infty}$具有如下性质：
1. $(a_{n})_{n=1}^{\infty}$ 单调递增，则:$a_{n}\leq a_{n+1}\forall n.$
2. $(a_{n})_{n=1}^{\infty}$上方有界，即:$\exists k$ 使得$a_{n}\leq k\forall n$.

则$\displaystyle\lim_{n\rightarrow\infty} a_{n}\in \mathbb{R}$, or $\exists l\in \mathbb{R}$,使得$\displaystyle\lim_{n\rightarrow\infty}a_{n}=l$.

### 1.2 完备性(2).
设$S\subseteq \mathbb{R}$, 且存在$k\in\mathbb{R}$使得$x\leq k\forall x\in S$, 则$\exists x_{o}\in R$使:

1. $x\leq x_{o}, \forall x\in S$($x_{o}$是$S$的上界)
2. 若$x\leq y\forall x\in S$, 则$y\geq x_{o}$.(任意上界都不会比$x_{o}更小$)。

## 2. 区间套定理
设$I_{n}=[a_{n}, b_{n}]$, $I_{1}\supset I_{2}\supset I_{3}\supset ...$且$\vert I_{n}\vert =b_{n}-a_{n}\rightarrow 0$当$n\rightarrow 0$, 则存在唯一一个点$x_{0}\in R$使得$\cap_{n=1}^{\infty}I_{n}={x_{0}}$.

**证明:** 因$I_{1}\supset I_{2}\supset I_{3}\supset ...$，则${a_{n}}$为递增数列，${b_{n}}$为递减数列，且$a_{n}\leq b_{1}$,$b_{n}\geq a_{1}\forall n$，依照实数的完备性，$\exists a\in\mathbb{R},\lim_{n\rightarrow\infty}a_{n}=a,\exists b\in\mathbb{R}, \lim_{n\rightarrow\infty}b_{n}=b$.

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
