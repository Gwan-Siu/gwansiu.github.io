---
layout:     post
title:      "Summer School Note"
subtitle:   "Super Resolution(1)"
date:       2017-07-11 12:00:00
author:     "GwanSiu"
catalog: true
tags:
    - Super Resolution
---

> Time: Summmer School in Polyu  
> Lecturer: Prof.Wan-Chi Siu, Prof.Lei Zhang

## 1. 图像插值与超分辨率的区别
图像插值技术(Image Interpolation)和超分辨率技术(Super Resolution,简称:SR)都是把图像从低分辨率图像(low resolution image)重构成高分辨率图像(hign resolution image)并同时需要最小化图像伪像(image artifacts)的过程. **插值和超分辨率都是图像逆问题的求解，是经典的不适定性问题(ill-posed problem, 即问题的解是不唯一的)，图像插值可以看做成超分辨的一种特例，但超分辨率所使用的方法并不能完全用在图像插值上，因为二者在本质上解决的问题是不一样的(下面会详细解释)。**

### 1.1 什么是图像插值(Image Interpolation)?
图像插值(Image Interpoaltion)的过程主要是将LR图像(low resolution)通过上采样恢复成HR图像(High resolution)。图像插值假设原本的HR图像是经过下采样得到LR图像，LR图像的混叠(aliasing)是由下采样导致的。图像插值的数学模型:

$$Y=DX+n \label{1}$$

其中，$Y$是观察到的LR图像，$X$是HR图像，$D$是下采样操作，$n$是噪声。  
下采样会导致混叠(aliasing)现象的发生，什么混叠？为什么会发生混叠？让我们转到频域去看看发生了什么事？
![image.png-181.9kB][1]

深蓝代表HR图像的频谱，通过下采样(down sampling)，采样速率降低，当采样速率低于奈奎斯特率的时候，就会发生频谱混叠现象。因此，图像插值(image interpolation)将LR图像恢复成HR图像时，本质问题就是通过上采样(up sampling)将频谱混叠的信号分开还原成未混叠的状态。**普遍使用的是领域插值法，根据已知领域信息去插值未知位置信息，例如：Bicubic, New edge-directed Interpolation 和 Soft Decision Adaptive Interpolation.图像插值主要分成Polynomial-based的方法和Edge-directed的方法。**
![image.png-77.3kB][3]


### 1.2 什么是超分辨率？(What's the super resolution?)
超分辨率(super resolution)是将图像从一张或者多张低分辨率图像通过上采样，去模糊和去噪技术恢复成高分辨率图像。  
超分辨率的数学模型:

$$Y=DHX+n$$

其中，$D$是下采样操作，$H$是模糊操作(模糊操作是一个高斯滤波的过程，相当于先过滤图像中的高频成分),$n$是噪声。  
与图像插值(image interpolation)比较，超分辨率假设观察到的图像$Y$先经过了一个高斯滤波的过程，这意味着重构过程是有一个增加高频图像高频信息的过程，即重构的$X$包含$Y$中没有的高频信息。超分辨率的频谱变化过程如下:
![image.png-154.6kB][2]

红色代表HR图像的频谱，蓝色代表LR图像的频谱，通过超分辨率，增加了高频信息。**注意:这里假设LR图像中不存在混叠(aliasing)的情况，实际上，混叠(aliasing)是存在的，当下采样导致图像采样速率低于奈奎斯特率的时候，LR图像的频谱存在混叠，这时要先解决混叠和增加高频信息两个问题。**这就解释了虽然图像插值是超分辨率的一个特例，但是超分辨率的方法不一定能够适用于图像插值。
超分辨率(SR)主要分成单张LR图像的超分辨率重构问题和多张LR的超分辨率重构问题，在方法上又可以分成reconstruction-based的方法和learning-based的方法，目前深度学习用到的都是单张LR图像的超分辨率重构，具体分类如下：
![image.png-70.5kB][4]












[1]: http://static.zybuluo.com/GwanSiu/42xcl9limpvtkcd5dm5xyjft/image.png
[2]: http://static.zybuluo.com/GwanSiu/v6p8ecllqghyirll48idq39n/image.png
[3]: http://static.zybuluo.com/GwanSiu/1l062x3sxgg8szaazk8cfcxd/image.png
[4]: http://static.zybuluo.com/GwanSiu/hn7qebuqgz88kvtaybhzl78f/image.png
