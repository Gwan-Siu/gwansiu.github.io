---
layout: post
title: Note of PRML(2)
subtitle: The information theory and machine learning
date: 2017-09-03
author: GwanSiu
catalog: True
tags:
    - Machine Learning
---

## 1. 信息量和熵
### 1.1 信息量
概率$p(x)$是用来衡量事件(x)的不确定性。通常，事件的发生都携带着信息，而信息内容的大小则取决于事件发生的概率。出于对信息内容的量化，数学家们便需要定义一种测度，该测度不仅需要满足数学上对测度的要求，还要求在事件的不确定性上是单调的，且必须能够反映事件的信息量。通常上，对于事件(x)，该事件的信息量$h(x)$等于事件发生概率$p(x)$的倒数再取对数：$h(x)=-\text{log}(p(x))$。

该公式表明小概率事件的信息量比大概率的信息量大，确定性事件不包含任何信息。  

### 1.2 熵(信息量编码角度)

从信息论和编码的角度，熵(Entropy)的本质是事件信息量$-\text{log}(p(x))$的期望, 熵表达着若从发射端将一个随机事件$X$传输到接收端所需要的最短平均编码长度(最短平均压缩长度):

$H(x)=-\sum_{x}p(x)\text{log}_{2}p(x) \tag{1}$

根据noiseless, coding theorem, 上式表明:若随机事件$p(x)$的依旧用真是分布$p(x)$编码$-\text{log}_{2}(p(x))$, 便可以达到noiseless coding的状态。

### 1.3 熵(系统的混乱程度)

从系统学的角度上出发，熵(Entropy)描述的是系统的混乱程度，即熵是描述系统的“有序”与“无序”状态。熵越大，系统的混乱程度越大(无序)，熵越小，系统则越趋于稳定(有序)。

举个例子:
假设将N个小球放入一些瓶子中，每个瓶子里的小球表示为$n_{i}$，那么一共可以分W组:

 $$ W=\frac{N!}{\prod_{i} n_{i}!}$ \tag{2}$

在两边同时取对数的平方:

$$ H=\frac{1}{N}\text{In}W=\frac{1}{N}\text{In}N!-\frac{1}{N}\sum_{i}\text{In}n_{i}! \tag{3}$$

使用String's approximation:

$$\text{In}N!\simeq N\text{In}N-N$$

其中，$\sum_{i}n_{i}=N$, 式子(3)可得:

$$ H=-\lim\limits_{n\to\infty} \sum_{i}(\frac{n_{i}}{N})\text{In}(\frac{n_{i}}{N})=-\sum_{i}p_{i}\text{In}p_{i} \tag{4}$$

当 limit $N\to\infty$, $\frac{n_{i}}{N}$收敛到一个定值$p_{i}$:$p_{i}=\lim\limits_{n\to\infty} \frac{n_{i}}{N}$, $p_{i}$是小球被分配到第i瓶子的概率。从物理学的角度上说，瓶内小球的排列分布方式是属于围观层面的，而小球总体的分布方式$\frac{n_{i}}{N}$是宏观层面的。

将每个瓶子都看成是一种离散的随机状态$X$，瓶子i用$x_{i}$表示,小球的概率则表示为$p(X=x_{i})=p_{i}$, 则整个离散随机系统的信息熵则为:

$H[p]=-\sum_{i}p(x_{i})\text{log}_{2}p(x_{i}) \tag{4}$

由下图可知，概率分布$p(x)$较集中(有序)的系统的熵较小，而概率分布$p(x)$较分散的(无序)的系统的熵较大。

<img src="http://static.zybuluo.com/GwanSiu/ovg1hri0wvh8033pd0o23bxf/image.png" width="400" height="400"/>

### 1.4 交叉熵

先考虑一联合概率分布$p(x,y)$, 假设x的信息$p(x)$已知，那需要多少额外的信息量$p(y\arrowvert x)$来表示y的值? 

$$H[y\arrowvert x]=\int\int p(y,x)\text{In}p(y\arrowvert x)dydx \tag{5}$$

其中，$p(y,x)=p(x)p(y\arrowvert x)$, 带入上式可得:

$$H[x,y]=H[y\arrowvert x]+H[x]$ \tag{6}$

$H[x,y]$为$p(x,y)$的信息熵，$H(x)$是$p(x)$是信息熵，而$H[y\arrowvert x]$便是用分布$p(x)$去表示分布$p(y)$所需要的额外信息，这可以看成是分布$p(x)$和分布$p(y)$的距离误差。

交叉熵便是真实分布$p(x)$产生的信息，使用非真实分布$q(y)$进行编码，所需要的最小平均长度: $H(p(x),q(x))$=-\sum_{i}p(x_{i})\text{log}_{2}q(x_{i})$$。

在机器学习中，交叉熵(cross entropy)可以作为机器学习的损失函数，$p(x)$是真实的概率分布，而训练出来的分布为$q(x)$，交叉熵则用来刻画$p(x)$与$q(x)$的相似度，使用梯度下降原则，找到相似度最大的$q(x)$。

## 2.相对熵(KL散度)

KL散度(Kullback-Leibler divergence)描述了两个分布$p(x)$与$q(x)$的距离。假设真实的概率分布为$p(x)$，使用非真实分布$q(x)$对该信息进行编码，那么所需要的额外信息(距离)为$d=H(p,q)-H(p)$。因此，KL散度所需额外信息的测度:

$$begin{aligned*}
\text{KL}(p\Vert q)&=H(p,q)-H(p) \\
&= -\int p(x)、text{In}(q(x))dx-(-\int p(x)\text{In}p(x)dx) \\
&= -\int p(x)\text{In}\lgroup \frac{q(x)}{p{x}} \\ 
\end{aligned*}
$$
