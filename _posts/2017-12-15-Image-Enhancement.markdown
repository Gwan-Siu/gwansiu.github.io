---
layout:     post
title:      "Image Processing-image enhancement"
date:       2017-12-15 12:00:00
author:     "GwanSiu"
catalog: true
tags:
    - Image Processing
---
>In this chapter, image enhancement techology is introduced. It mainly contains: (1) Quantization and smapling; (2) Point operation: histogram equalization; (3) Spatial filter: box average, gaussian kernel, local mean filter, bilateral filter, and non-local filter.

## 1. Image Quantization
Image is an high dimensional array. Conventionally, sensors in the camera convert continuous signals to discrete signals and stroe them in 2 or 3 dimensional array. This process usually contains **spatial sampling** and **quantization**, as shown in the figure.
 
<img src="http://static.zybuluo.com/GwanSiu/8fsscwc7t44zw0isny8beezv/image.png" width = "600" height = "400" alt="Fig 1: Left:Sampling, Right: Quantization"/>

Resolution represents total numbers of sensor in the camera. In other words, it measures how many sensors are used to represent an object. Fixed quantization level, the high resolution is, the clearer the image is.
 
<img src="http://static.zybuluo.com/GwanSiu/rgxvz5p6tyrgv9nx1emhfqr2/QuantizationAndSampling.png" width = "600" height = "400" alt="Quantization"/>

<img src="http://static.zybuluo.com/GwanSiu/mjje01g4ygd11ea12txyaqym/Sampling.png" width = "600" height = "400" alt="Sampling"/>

## 2. Image Enhancement
**Image enhancement is to improve the quality of an image or accentuate particular features.** Techniques are more general purpose, for example, how to imporve contrast? How to sharpen edges? How to reduce noise? Image enhancement is different from image restoration. **In term of image restoration, a strong model of degradation process is not assumed.**

Gerenrally, point operations and spatial filter are two main techniques of image enhancement. **Point operations is independent of values of its neighborhoods, while spatial filter relies on its neighborhoods.** In this blog, histogram equalization and temporal averaging, which are point operations, will be introduced. In term of spatial filter, box averaging, gaussian kernel, local mean filter, bilateral filter, and non-local mean filter will be discussed. The advantages and disadvantages of these methods will be discussed.

### 2.1 Point operations---histogram equalization
**What's the point operations?** Point operation is a global mapping, which map original intensity to a new internsity: $i_{n} =f(i)$, where $i_{n}$ is new intensity and $i$ is original intensity.  

**What can point operator do?** (1)It can change contrast and brightness by transforming pixel intensities. (2)Assign the same new intensity value to all pixels given original value. For example, affine mapping: $i_{n} = \alpha i + \beta$.

**Histogram equalization:** apply a monotonic function **f(i)** to the intensities so that the identities is less peaked(Flattened), the tranformed cdf is linear, the idea of histogram equalization is shown below.

<img src="http://static.zybuluo.com/GwanSiu/ola2640fbn0xp286x2gqz8v6/image.png" width = "600" height = "400" alt="Idea of histogram equalization"/>

There are 2 steps to implement histogram equalization:
1. Compute the cumulative probability distribution distribution $C(i)$ from the intensity histogram. **To be noted, the range of cdf is from 0 to 1. Normalization is necessary during implementing.**
2. Map pixel intensity as $i_{n} = f(i)$, where $f(i)=\frac{i_{max}}{N}C_{i}$, and $N$ is #pixels.

This is an example of histogram equalization:

<img src="http://static.zybuluo.com/GwanSiu/f8lfjjxe19k18hffu118r0ea/HistogramEqualization.png" width = "600" height = "400" alt="Histogram equalization"/>

### 2.2 Point operations----temporal average
**Average N noise samples with zero mean and variance $\sigma^{2}$ result has zero mean and variance $\sigma^{2}/N$. Why divided by N？Just review the concept of the sum of variable.**

<img src="http://static.zybuluo.com/GwanSiu/80fhttmsh3k40jey3wwd7ox4/temporal%20filter.png" width = "600" height = "400" alt="Histogram equalization"/>

### 2.3 Spatial filter----Box average(local mean average)
**Image are not smooth because adjacent pixels are different.** Smoothing is equivalent to make adjacent pixels look more similar with each other. Conventionally, average of its neighborhoods is adopted. 

**Why average is effective?**  what we want is to make adjacent pixels look more similar with each other. In other words, the variations of adjacent pixels should be minimized. The objective funcion:

$$
\begin{equation}
\min_{x_{i}\in N_{x}} \Vert x-x_{i} \Vert^{2}
\end{equation}
$$






