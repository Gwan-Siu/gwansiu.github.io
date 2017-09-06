---
layout: post
title: Note of PRML(3)
subtitle: Linear Regression Model
date: 2017-09-06
author: GwanSiu
catalog: True
tags:
    - Machine Learning
---

## 1. Linear Regression Model

假设目标数数据$t$是由一连续函数$f(x)$产生，根据数学的逼近原理，连续函数$f(x)$可以分解成函数空间中基函数的线性组合。因此线性回归模型本质上可以看成将数据X使用不同空间的基函数展开，是这些空间基函数的线性组合去逼近$f(x)$:

$$y(x,w)=w_{0}+w_{1}\Phi_{1}(x_{1})+...+w_{D}\Phi_{D}(x_{D}) \tag{1}$$

其中, $X=(x_{1},...,x_{D})$为输入数据，$w_{0},w_{1},...,w_{D}$为模型的参数，$\Phi_{i}$为数据预处理时空间变换的基函数。

当$\Phi_{i}(x_{i})=x_{i}$时，线性回归变成最简单的形式，输入数据$X$不作任何预处理:

$$y(x,w)=w_{0}+w_{1}x_{1}+...+w_{D}x_{D} \tag{2}$$

式(2)表明，$y(x,w)$是输入数据$x$的线性组合，因此$y$只能存在于$x$的线性空间。若$f(x)$与$x$是非线性关系中，则需要借助预处理函数$\Phi_{i},i=0,...,N-1$，可以看成将$X$在基函数为$\Phi_{i},i=0,...,N-1$的空间中展开$。不同基函数的选择对数据$X$的空间会造成不同的影响，下面便对不同
基函数进行讨论:

- 1. 若是基函数为多项式函数$\Phi_{j}(x)=x^{j}$: 首先，多项式函数的线性组合可以逼近任意连续函数，因此选用多项式函数作为基函数是合理的。多项式作为基函数的缺点在于多项式函数一种全局变换，改变输入空间的某一个区域会影响输入空间的其他区域。解决办法是使用样条函数将输入空间分割成不同的区域，在每个区域可以应用不同多项式函数---这便是分段多项式。
- 2. 高斯核函数$\Phi_{j}(x)=exp{-\frac{(x-\mu_{j})^{2}}{2s^{2}}}$:可行性论证，高斯核函数的线性组合可以逼近任意连续函数，因此高斯核函数作为基函数是合理的。在高斯核函数中，$\mu_{j}$可以控制基函数在输入空间的位置，而参数$s$则控制基函数在输入空间的尺度。(翻译过来很别扭: where the $\mu_{j}$ govern the locations of the basis functions in input space, and the parameter $s$ governs their scale.)
- 3. Sigmoid 函数$\Phi_{j}(x)=\sigma (\frac{x-\mu_{j}}{s})$与tanh 函数$\text{tanh}(\alpha)=2\sigma(2\alpha)-1$，这两个函数也是可以任意逼近连续函数。通常，该函数在深度学习作为激活函数，增加非线性。
- 4.Fourier basis和wavelet: 任意连续函数可以分解成Fourier series，作为基函数可行。Fourier basis表示是带宽有限时域无限的信息，或者，时域有限频域无限的信息。而wavelet则表示局部的频域和时域的信息。

## 2.Least Squares

由第一部分可知，连续函数$f(x)$可以展开成一系列基函数的线性组合。实际情况，该基函数是未知的，由逼近原理可知，连续函数$f(x)$可以由其他基函数进行逼近。

假设目标变量$t$由函数$y(x,w)$加上高斯噪声形成:

$$t=y(x,w)+\eplson \tag{3}$$

其中,$\eplson$为均值为零，precision为$\beta$的高斯噪声，则式(3)可以写成:

$$p(t\arrowvert x,w,\beta)=N(t\arrowvert y(x,w),\beta^{-1}) \tag{4}$$

在噪声为高斯函数的假设下，该函数的均值为:

$$ E[t\arrowvert x]=\int tp(t\arrowvert x)dt = y(x,w) \tag{5}$$

由上式可以看出，高斯噪声的假设意味着当$x$给定时，$t$的条件分布为单一的高斯分布，这在实际应用不符合。将该例子的推广便是混合高斯分布。

### 2.1 Least Squares Algorithm

假设输入数据$X=(x_{1},...,x_{N})$,与之对应的目标函数$t=(t_{1},...,t_{N})$,数据点之间是独立抽取，互不影响的(IID)，因此根据贝叶斯定理可得，likelehood函数为:

$$p(t\arrowvert X,w,p)=\prod_{n=1}^{N} N(t_{n}\arrowvert w^{T}\Phi(x_{n}),\beta^{-1}) \tag{6}$$

两边取对数可得:

$$\begin{aligned}
\text{In}p(t\arrowvert w,\beta^{-1}) &=\sum_{n=1}^{N} \text{In} N(t_{n}\arrowvert w^{T}\Phi(x_{n}),\beta^{-1}) \\
&=\frac{N}{2}\text{In}\beta-\frac{N}{2}\text{In}(2\pi)-\beta E_{D}(w) 
\end{aligned}$$

其中，$E_{D}=\frac{1}{2}\sum_{n=1}^{N}(t_{n}-w^{T}\Phi(x_{n}))^{2}$

最大化似然函数，则对两边对求导:

$$\nabla = \beta \sum_{n=1}^{N} (t_{n}-w^{T}\Phi(x_{n}))\Phi(x_{n})^{T} \tag{8}$$

设式(8)为零：

$$\sum_{n=1}^{N}t_{n}\Phi(x_{n})^{T}-w^{T}(\sum_{n=1}^{N} \Phi(x_{n})\Phi(x_{n})^{T}) \tag{9} $$

解得:

$$ w_{ML} = (\Phi^{T}\Phi)^{-1}\Phi^{T}t \tag{10} $$

其中, $\Phi^{dagger}=(\Phi^{T}\Phi)^{-1}\Phi^{T}$,为矩阵$\Phi$的伪逆。$w_{ML}$为该回归方程的解析解。

具体求解：

$$
E_{D}=\frac{1}{2}\sum_{n=1}^{N}(t_{n}-w_{0}-\sum_{j=1}^{N}w_{j}\Phi_{j})^{2} \tag{11} \tag{11}$$

求导并令导数为零可得:
$$ w_{0} = \bar{t}-\sum_{j=1}^{M-1}w_{j}\bar{\Phi_{j}} \tag{12}$$

其中:
$$ \bar{t}=\frac{1}{N}\sum_{n=1}^{N}t_{n} \tag{13} \\
\bar{\Phi_{j}}=\frac{1}{N}\sum_{n=1}^{N}\Phi_{j}(x_{n}) \tag{14} $$

可以看出，偏差(bias)$w_{0}$是对目标值均值和基函数加权平均的差值的补偿。

而方差$\sigma^{2}=\frac{1}{\beta}$为:

$$ \frac{1}{\beta_{ML}}=\frac{1}{N}\sum_{n=1}^{N}(t_{n}-w_{ML}^{T}\Phi(x_{n}))^{2} \tag{15} $$

因此, 方差可以认为是目标函数到回归函数残差。

### 2.2 Geometry of least squares

$$\Vert t-w^{T}\Phi \Vert^{2} \tag{16}$$

由以上分析可知，一个连续函数可以分解成所在函数空间基的线性组合。而最小二乘法所做的事情则是当数据$X$在基函数为$\Phi$空间展开，寻找目标变量在该子空间的投影。

<img src="http://static.zybuluo.com/GwanSiu/9r998c70k2twssvrs3jy76yc/image.png" width="400" height="400"/>

### 2.3 Sequential Learning

式(10)表明,最小二乘法有可解析解，但是在求解析解的过程设计到矩阵的逆运算，当数据量$X$成千上万时，求逆的过程则会变得非常缓慢，从而不能实时求解。因此当数据量$X$成千上万时，可以使用stochastic gradient descent：

$$ w^{\tau +1}=w^{\tau}-\eta\nabla E_{n} \tag{17}$$

其中，$\eta$为学习率，此方法需要对$w_{0}$赋一个初始值，如果目标函数是凸函数的话，初始值对最优解没有影响，如果目标函数是非凸函数的话，初始值则对最后的优化结果影响很大。将式(8)代入式(17)中:

$$w^{\tau +1}=w^{\tau}+\eta(t_{n}-w^{(\tau)T}\Phi_{n}) \tag{18}$$

### 2.4 Least Squares and Regularization

为了防止模型过拟合，则需要将模型的VC维度降低,因此需要对模型的参数数量限制:

$$E_{D}(w)+\lambda E_{W}(w) \tag{19}\\
E_{D}(w)=\frac{1}{2}\sum_{n=1}^{N}(t_{n}-w^{T}\Phi(x_{n}))^{2}  \tag{20} \\
E_{W}(w)=\frac{1}{2}\Vert w \Vert^{p} \tag{21} $$

**定性分析:$\lambda$为超参数,该参数控制约束项的强弱.$\lambda$越大，则表明该约束项越强，$\lambda$越小，则表明该约束项越弱。**

目标函数具体表现为:

$$\frac{1}{2}\sum_{n=1}^{N}(t_{n}-w^{T}\Phi(x_{n}))^{2}+\frac{1}{2\lambda}\Vert w \Vert^{p} \tag{22} $$

从上式可知，p值不同，对函数参数的约束效果不同：

- 1.当p=1时, 式(22)变成1范数约束，1范数可以将参数稀疏化。
- 2.当p=2时, 则变成普通的2范数约束，变成岭回归问题(Ridge)，参数的数值会因$\lambda$变大而变得很小。该方法专门用于共线数据分析，是一种有偏回归方法。
$$\frac{1}{2}\sum_{n=1}^{N}(t_{n}-w^{T}\Phi(x_{n}))^{2}+\frac{1}{2\lambda}\Vert w \Vert^{2}  \\
w = (\lambda I+\Phi^{T}\Phi)^{-1}\Phi^{T}t$$

- 3.当$0\leq p \geq$时，目标函数则退化成非凸函数。

**为什么1范数会讲参数变得稀疏？**

<img src="http://static.zybuluo.com/GwanSiu/yvaanmz1egkbm7th6era2c8v/image.png" width="400" height="400"/>

由上图可知，函数求解的过程是一个"撞边"的过程，刚刚好1范数的解，会造成解在某个维度上为零。

### 2.5 多输出问题

以上回归问题是单输出问题，即:$y\in R，w\in R^{M\times 1}$,其中$M$子空间的维度(基函数的个数)。若回归问题变成多输出问题，即:$y\in R^{K \times 1}$, 回归方程则变为:

$$ y(x,w)=W^{T}\Phi(x) \tag{23}$$

其中，$W \in R^{M \times K}，\Phi(x)\in R^{M\times}, y\in R^{K \times 1}$，和单输出问题相比，$w$由$M\times 1$的向量变成一个$M\times K$的矩阵。

条件概率分布则变为:

$$p(**T**\arrowvert x,W,\beta)=N(t\arrowvert W^{T}\Phi(x),\beta^{-1}I) \tag{24}$$

log Likelehood函数则为:

$$\begin{aligned}
\text{In}p(T\arrowvert w,\beta^{-1}) &=\sum_{n=1}^{N} \text{In} N(t_{n}\arrowvert W^{T}\Phi(x_{n}),\beta^{-1}T) \\
&=\frac{NK}{2}\text{In}(\frac{\beta}{2\pi})-\frac{\beta}{2}\sum_{n=1}^{N}\Vert t_{n}-W^{T}\Phi(x_{n}) \Vert^{2} 
\end{aligned}$$

其中，$**T**=(t_{1},...,t_{N})$,每个小t都是一个Kx1的向量，而$W$则是$M\times K$
的矩阵。

求解可得:

$$W_{ML}=(\Phi^{T}\Phi)^{-1}\Phi^{T}T \tag{26}$$

具体可得:

$$w_{k}=(\Phi^{T}\Phi)^{-1}\Phi^{T}t_{k}=\Phi^{dagger}t_{k} \tag{27}$$

因此，从式(27)可知，多输出的回归问题可以分解成多个单输出的回归问题求解，但仅仅可以通过求解$\Phi^{\dagger}$
便可以求得问题的解。

**注意:**该问题的假设是噪声是零均值，方差为$\beta^{-1}$的高斯噪声,可以将其推广成任意的协方差矩阵。同样可以分解成$K$个子回归问题。$W$仅仅决定该高斯分布的均值，求解均值的过程与求解方差的过程是相互独立的。



