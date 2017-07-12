---
layout:     post
title:      "Summer School Note"
subtitle:   "Super Resolution"
date:       2017-07-11 12:00:00
author:     "GwanSiu"
catalog: true
tags:
    - Super Resolution
---

> Time: Summmer School in Polyu  
> Lecturer: Prof.Wan-Chi Siu, Prof.Lei Zhang

## 1. 图像插值与超分辨率的区别
图像插值技术(Image Interpolation)和超分辨率技术(super resolution)都是把图像从低分辨率图像(low resolution image)恢复高分辨率图像(hign resolution image)并同时需要最小化图像伪像(image artifacts). **但插值和超分辨率都是图像逆问题的求解，是经典的不适定性问题(ill-posed problem, 即问题的解是不唯一的)，但二者在所要解决的问题上在本质是不同的，超分辨率所使用的方法并不能用在图像插值上。**

### 1.1 什么是图像插值(Image Interpolation)?
图像插值(Image Interpoaltion)的过程主要是将LR图像(low resolution)通过上采样恢复成HR图像(High resolution)。图像插值假设原本的HR图像是经过下采样得到LR图像，LR图像的混叠(aliasing)是由下采样导致的。图像插值的数学模型:
$$Y=DX \label{1}$$
其中，$Y$

### 1.2 什么是超分辨率？(What's the super resolution?)
超分辨率(super resolution)是将图像从一张或者多张低分辨率图像通过上采样，去模糊和去噪技术恢复成高分辨率图像。  
超分辨率的数学模型：
$$