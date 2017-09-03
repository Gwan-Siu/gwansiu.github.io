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

 $$ W=\frac{N!}{\prod_{i} n_{i}!} \tag{2}$$

在两边同时取对数的平方:

$$ H=\frac{1}{N}\text{In}W=\frac{1}{N}\text{In}N!-\frac{1}{N}\sum_{i}\text{In}n_{i}! \tag{3}$$

使用String's approximation:

$$\text{In}N!\simeq N\text{In}N-N$$

其中，$\sum_{i}n_{i}=N$, 式子(3)可得:

$$ H=-\lim\limits_{n\to\infty} \sum_{i}(\frac{n_{i}}{N})\text{In}(\frac{n_{i}}{N})=-\sum_{i}p_{i}\text{In}p_{i} \tag{4}$$

当 limit $N\to\infty$, $\frac{n_{i}}{N}$收敛到一个定值$p_{i}$:$p_{i}=\lim\limits_{n\to\infty} \frac{n_{i}}{N}$, $p_{i}$是小球被分配到第i瓶子的概率。从物理学的角度上说，瓶内小球的排列分布方式是属于围观层面的，而小球总体的分布方式$\frac{n_{i}}{N}$是宏观层面的。

将每个瓶子都看成是一种离散的随机状态$X$，瓶子i用$x_{i}$表示,小球的概率则表示为$p(X=x_{i})=p_{i}$, 则整个离散随机系统的信息熵则为:

$$H[p]=-\sum_{i}p(x_{i})\text{log}_{2}p(x_{i}) \tag{4}$$

由下图可知，概率分布$p(x)$较集中(有序)的系统的熵较小，而概率分布$p(x)$较分散的(无序)的系统的熵较大。

<img src="http://static.zybuluo.com/GwanSiu/ovg1hri0wvh8033pd0o23bxf/image.png" width="400" height="400"/>

### 1.4 交叉熵与条件熵

先考虑一联合概率分布$p(x,y)$, 假设x的信息$p(x)$已知，那需要多少额外的平均信息量$H(y\arrowvert x)$(条件熵)来表示y的值? 

$$H[y\arrowvert x]=\int\int p(y,x)\text{In}p(y\arrowvert x)dydx \tag{5}$$

其中，$p(y,x)=p(x)p(y\arrowvert x)$, 带入上式可得:

$$H[x,y]=H[y\arrowvert x]+H[x] \tag{6}$$

$H[x,y]$为$p(x,y)$的信息熵，$H(x)$是$p(x)$是信息熵，而$H[y\arrowvert x]$便是用分布$p(x)$去表示分布$p(y)$所需要的额外信息，这可以看成是分布$p(x)$和分布$p(y)$的距离误差。

交叉熵便是真实分布$p(x)$产生的信息，使用非真实分布$q(y)$进行编码，所需要的最小平均长度: $H(p(x),q(x))=-\sum_{i}p(x_{i})\text{log}_{2}q(x_{i})$。

在机器学习中，交叉熵(cross entropy)可以作为机器学习的损失函数，$p(x)$是真实的概率分布，而训练出来的分布为$q(x)$，交叉熵则用来刻画$p(x)$与$q(x)$的相似度，使用梯度下降原则，找到相似度最大的$q(x)$。

## 2.相对熵(KL散度)

### 2.1 KL散度
KL散度(Kullback-Leibler divergence)描述了两个分布$p(x)$与$q(x)$的距离。假设真实的概率分布为$p(x)$，使用非真实分布$q(x)$对该信息进行编码，那么所需要的额外信息(距离)为$d=H(p,q)-H(p)$。因此，KL散度所需额外信息的测度:

$$
\begin{aligned}
\text{KL}(p\Vert q)&=H(p,q)-H(p) \\
&= -\int p(x)\text{In}(q(x))dx-(-\int p(x)\text{In}p(x)dx) \\
&= -\int p(x)\text{In}(\frac{q(x)}{p{x}})\\ 
\end{aligned}
$$

其中，$\text{KL}(p\Vert q)\geq 0$,只有当$p(x)=q(x)$时，$\text{KL}(p\Vert q)= 0$(使用Jensen不等式可证)。

Jensen不等式:
$$f(E[x])\leq E[f(x)] \tag{8}$$
将Jensen不等式用在KL散度上:

$$\text{KL}(p\Vert q)=-\int p(x)\text{In}\lgroup \frac{q(x)}{p(x)} \rgroup \geq -\text{In}\int q(x)dx=0 \tag{9}$$

**注意:KL散度具有不对称性，即:$\text{KL}(p\Vert q) \neq \text{KL}(q\Vert p)$。这表明，使用$p(x)$对$q(x)$进行所需要的信息量与使用$q(x)$对$p(x)$进行所需要的信息量不同。**

### 2.2 KL散度与机器学习
假设数据$X=(x_{1},...,x_{n})$由未知的真实分布$p(x)$产生，我们使用数据$X$，通过贝叶斯法则建立模型:$p(X\arrowvert \Theta)$，其中$\Theta$是模型的参数。我们可以最小化真实分布$p(x)$与模型$p(X\arrowvert \Theta)$的KL散度，找到与真实分布$p(x)$最近的$p(X\arrowvert \theta^{*})$。但由于$p(x)$未知，不能直接使用。于是，便通过数据$X$估计$p(x)$的均值，因此便有:

$$\text{KL}(p\Vert q) \simeq \frac{1}{N} \sum_{n=1}^{N} (-\text{In}q(x_{n}\arrowvert \theta)+\text{In}p(x_{n}))$ \tag{10}$$

因此最小化KL散度等效于最大化似然函数。

## 2.3 KL散度与互信息(multual information)
如果给定一组不相互独立的随机变量(x,y),问随机变量的mutual information是多少？假设，已知联合概率分布$p(x,y)$, 若使用x,y独立的情况去编码非独立情况，KL散度便可以测得$p(x,y)$与$p(x)p(y)$的距离(multual information).

$$\begin{aligned}
I(x,y)&=\text{KL}(p(x,y)\Vert p(x)p(y)) \\
&=-\int \int p(x,y)\text{In}(\frac{p(x)p(y)}{p(x,y)})dxdy 
\end{aligned}
$$

由KL散度可知，$I(x,y)\geq 0$,将KL进一步分解成两个信息的和，multual information则可以表示成:

$$ I(x,y)=H(x)-H(x\arrowvert y)=H(y)-H(y\arrowvert x) \tag{11} $$

从贝叶斯理论分析，$p(x)$为先验经验，而$p(x\arrowvert y)$为观察数据y后的后验经验。Multual information则可以表示成观察信息y后，x不确定性的冗余。(原文: The multual information therefore represents the reduction in uncertainty in X as a consequence of the new observation y.)

## 3.JS散度

由于KL散度具有不对称性，而JS散度便是进一步完善了这个问题，JS散度将测度映射固定在区间[0,1]。通理可知，若分布$p(x)=q(x)$, JS=0, 若$p(x)$与$q(x)$相距无穷远，则JS$\rightarrow \infty$。

**JS散度**

$$JS(p(x)\Vert q(x))=\frac{1}{2}\text{KL}(p(x)\Vert \frac{p(x)+q(x)}{2})+ \frac{1}{2}\text{KL}(q(x)\Vert \frac{p(x)+q(x)}{2}) \tag{12} $$

但是JS散度和KL散度依旧有一个严重的问题，当$p(x)$与$q(x)$相距很远，KL散度很大，没有实际意义，而JS散度接近1，此时JS散度的梯度会发生弥散现象。