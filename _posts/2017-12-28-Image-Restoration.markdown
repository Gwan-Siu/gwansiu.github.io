---
layout:     post
title:      "Image Processing-Image Restoration"
date:       2017-12-27 12:00:00
author:     "GwanSiu"
catalog: true
tags:
    - Image Processing
---
>In this article, image restoration and its techniques are introduced: (1). the basic concept of image restoration and its difference from image enhancement. (2).The inverse filter (3).The wiener filter (4). MAP formulation

## 1. Image Degradations
Image restoration is to reconstruct the latent high quality image from its degraded observation. The acquired image is degraded observation of the unknown latent image, while degradation comes from various factors such as optical blur, motion blur, spatial quantization, noise corruption.

In contrast to image enhancement, **in image restoration the degradation is modelled**. This enables the effects of the degradation to be(largely) removed.

<img src="http://static.zybuluo.com/GwanSiu/iey2t7oret9dooos0s1hc6yb/image.png" width = "600" height = "400" alt="Degradation"/>

$$
\begin{equation}
\text{spatial domain: }g(x,y) = h(x,y)\ast f(x,y)+n(x,y)\\
\text{Frequency domain: }G(\mu,\upsilon) = H(\mu,\upsilon)F(\mu,\upsilon)+N(\mu,\upsilon)
\end{equation}
$$

- $f(x,y)$: original image.
- $g(x,y)$: observed image corrupted by degradation.
- $h(x,y)$: degradation filter, **point spread function or impulse response of the imaging system. Model degradation as a convolution with linear, shift invariant filter.**
- $\hat{f}(x,y)$: estimated image, compute from $g(x,y)$.
- $n(x,y)$: addictive noise.

**The most challenging issue of image restoration is loss of information and noise.** For image blurring, blurring acts as a low pass filter and attenuates higher spatial frequencies.

<img src="http://static.zybuluo.com/GwanSiu/joq9c3mgszy13lxkq0hvnjx8/image.png" width = "600" height = "400" alt="Blurring"/>

## 2. The Inverse Filtter
From the perspective of frequency domain, image degradation can be modeled as: $G(\mu,\upsilon) = H(\mu,\upsilon)F(\mu,\upsilon)+N(\mu,\upsilon)$. To simplified the formulation, the noise term is ignored, and then the estimated function is obtained from inverse process:

$$
\hat{F}(x,y) = \frac{G(\mu,\upsilon)}{H(\mu,\upsilon)}
$$

The work flow of inverse filter:

<img src="http://static.zybuluo.com/GwanSiu/oteijqtausi9ksa8s0g2vq9d/image.png" width = "600" height = "400" alt="Inverse Filter"/>

**The problem of noise amplification:**

$$
\begin{equation}
G(\mu,\upsilon) = H(\mu,\upsilon)F(\mu,\upsilon)+N(\mu,\upsilon)\\
F(\mu,\upsilon)=\frac{G(\mu,\upsilon)}{H(\mu,\upsilon)} = F(\mu,\upsilon)+\frac{N(\mu,\upsilon)}{H(\mu,\upsilon)}
\end{equation}
$$

<img src="http://static.zybuluo.com/GwanSiu/r3yushkqqirgpyne27xa39ij/image.png" width = "600" height = "400" alt="Problem of inverse filter"/>

## 3. The Wiener Filter

<img src="http://static.zybuluo.com/GwanSiu/w07o7w4muyibmbi83qos4td6/image.png" width = "600" height = "400" alt="Wiener filter"/>

A Wiener filter minimizes the least square error $\int_{-\infty}^{+\infty}\int_{-\infty}^{+\infty}(f(x,y)-\hat{f}(x,y))^{2}dxdy$, restoration with a wiener filter $W(\mu,\upsilon)$, detail proven is in [appedix](#appendix):

$$
\begin{equation}
G(\mu,\upsilon) = H(\mu,\upsilon)F(\mu,\upsilon)+N(\mu,\upsilon) \\
\hat{F}(\mu,\upsilon)=W(\mu,\upsilon)G(\mu,\upsilon)
\label{eq:equation1}
\end{equaition}
$$

where $W(\mu,\upsilon)=\frac{H^{\*}(\mu,\upsilon)}{|H(\mu,\upsilon)|^{2}+K(\mu,\upsilon)}$.
- $K(\mu,\upsilon)=\frac{S_{\eta}(\mu,\upsilon)}{S_{f}(\mu,\upsilon)}$, signal-noise-ratio.
- $S_{f}(\mu,\upsilon)=|F(\mu,\upsilon)|^{2}$ power spectral density of $f(x,y)$.
- $S_{\eta}(\mu,\upsilon)=|N(\mu,\upsilon)|^{2}$ power spectral density of $\eta(x,y)$.

**Analysis of frequency behaviour:**
From the equation \eqref{eq:equation1}, we can conclude that:
- If $K=0$, then $W(\mu,\upsilon)=\frac{1}{H(\mu,\upsilon)}$, i.e. an inverse filter.
- If $K>>|H(\mu,\upsilon)|$ for large $\mu,\upsilon$, then high frequency are attenuated.
- $|F(\mu,\upsilon)|$ and $N(\mu,\upsilon)$ are often known approximately and $K$ is set to a constant scalar which is determined empirically.

<img src="http://static.zybuluo.com/GwanSiu/31lemtyempkfnkznpy9pmnf2/image.png" width = "600" height = "400" alt="The frequency behavior of Wiener filter"/>

## 4. MAP Formulation
Given an observed image $\hat{g}$ with $n$ pixels and unkonwn original image $f$, the degradation process can be formulated in the form of vector:

$$
\begin{equation}
\hat{g} = Af + n
\end{equation}
$$

where $\hat{g}$ and $f$ are image vector with n dimension, and $A$ is an $n\times n$ matrix. 

To estimate $f(x,y)$ by optimizing a cost function:

$$
\begin{equation}
\hat{f}=\text{arg }\min_{f}\Vert g-Af \Vert^{2} +\lambda p(f)
\end{equation}
$$

where $\Vert g- Af \Vert^{2}$ is likelihood function and $\lambda p(f)$ is a prior regularization term.

Continue....(To figure out how super-resolution work.)

## Appendix
<span id="appendix">Wiener filter</span> essentially aims to minimize the least square error, and the proven is shown in detail:

$$
\begin{equation}
\varepsilon = \int_{-\infty}^{+\infty}\int_{-\infty}^{+\infty}(f(x,y)-\hat{f}(x,y))^{2} dxdy
\end{equation}
$$

According to Paeawval's theorem:

$$
\begin{align}
\varepsilon &= \int_{-\infty}^{+\infty}\int_{-\infty}^{+\infty}(f(x,y)-\hat{f}(x,y))^{2} dxdy \\
\varepsilon &= \int_{-\infty}^{+\infty}\int_{-\infty}^{+\infty}(F(\mu,\upsilon)-\hat{F}(\mu,\upsilon))^{2} d\mu d\upsilon \\
\text{where} \\
\hat{F}&=WG=WHF+WN \\
F-\hat{F} = (1-WH)F-WH \\
\text{Therefore}
\varepsilon &= \int_{-\infty}^{+\infty}\int_{-\infty}^{+\infty}|(1-WH)F-WN|^{2} d\mu d\upsilon \\
&= \int_{-\infty}^{+\infty}\int_{-\infty}^{+\infty}|(1-WH)F|^{2}-|WN|^{2} d\mu d\upsilon \\
\text{since } f(x,y) \text{ and } \eta \text{ are uncorrelated. Note:integrand is sum of two squares} 
\end{align}
$$

take a derivative to minimize the least square error:

$$
\frac{\partial}{\partial z}2(-(1-W^{\*}H^{\*})H|F|^{2}+W^{\*}|N|^{2})=0
$$

$$
\begin{align}
W^{\*} &=\frac{H|F|^{2}}{|H|^{2}|F|^{2}+|N|^{2}} \\
W^{\*} &=\frac{H|F|^{2}}{|H|^{2}+\frac{|N|^{2}}{|F|^{2}}}
\end{align}
$$


