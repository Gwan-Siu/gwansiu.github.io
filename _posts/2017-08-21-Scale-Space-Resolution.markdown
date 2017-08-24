---
layout:     post
title:      "Scale Space v.s. Image Resolution"
date:       2017-08-21 12:00:00
author:     "GwanSiu"
catalog: true
tags:
    - Image Processing
---

>初学者经常会把图像的scale和Resolution混淆，这篇博文目的是详细解说图像的Scale和Resolution。

## 1. 什么是尺度空间？
图像的尺度是指图像内容的粗细程度。尺度的概念是用来模拟观察者距离物体的远近的程度，具体来说，观察者距离物体远，看到物体可能只有大概的轮廓；观察者距离物体近，更可能看到物体的细节，比如纹理，表面的粗糙等等。**从频域的角度来说，图像的粗细程度代表的频域信息的低频成分和高频成分。**粗质图像代表信息大部分都集中在低频段，仅有少量的高频信息。细致图像代表信息成分丰富，高低频段的信息都有。  
**尺度空间又分为线性尺度空间和非线性尺度空间。**在本篇博文中，博主仅仅讨论线性尺度空间的构造，非线性尺度空间不在本文的讨论之中。在数学上，空间(space)指的是具有约束条件的集合(set)。而图像的尺度空间是指同一张图像不同尺度的集合。在该集合中，粗尺度图像不会有细尺度图像中不存在的信息，换言之，任何存在于粗尺度图像下的内容都能在细尺度图像下找到，细尺度图像通过filter形成粗尺度图像过程，不会引入新的杂质信息。粗尺度图像形成的过程是高频信息被过滤的过程，smooth filter理所当然成为首选，而加入不引入信息杂质的线性滤波器结构约束，通过证明，**高斯核便是实现尺度变换的唯一线性核。**

由此可见，图像的尺度空间是一幅图像经过几个不同高斯核后形成的模糊图片集合，用来模拟人眼看到物体的远近程度，模糊程度。**注意:图像尺度的改变不等于图像分辨率在改变，下图便是很好的例子，图像的分辨率是一样的，但是尺度却不一样。**

```python
from skimage import data, filters,io
import matplotlib.pyplot as plt
%matplotlib inline

image = io.imread('/Users/xiaojun/Desktop/Programme/DataSet/mxnet_mtcnn_face_detection-master/anner.jpeg')
img1 = filters.gaussian(image, sigma=1.0)
img2 = filters.gaussian(image, sigma=2.0)
img3 = filters.gaussian(image, sigma=3.0)

plt.figure('gaussian',figsize=(8,8))
plt.subplot(221)
plt.imshow(image)
plt.axis('off')
plt.title('original image')
plt.subplot(222)
plt.imshow(img1)
plt.axis('off')
plt.title('gaussian kernel with sigmma=1.0')
plt.subplot(223)
plt.imshow(img2)
plt.axis('off')
plt.title('gaussian kernel with sigmma=2.0')
plt.subplot(224)
plt.imshow(img3)
plt.title('gaussian kernel with sigmma=3.0')
plt.axis('off')
```
![image.png-223.1kB][1]

## 2. 什么是图像的分辨率？
图像的分辨率(Image Resolution)本质上是图像的在水平和垂直方向的量化程度，直观上理解是指图像能展现的细节程度。量化的过程是模拟信号转变成数字信号的过程，这一过程是**不可逆**的信息损失过程。因此，量化级别的高低决定了该数字信号能否更好的表示原本的模拟信号。图像是二维数组，水平像素和垂直像素的数量便是图像量化的级别，多像素图像更能展示图像的细节。如下图:
[---][2]


[1]: http://static.zybuluo.com/GwanSiu/sha5x9d15ozenfab6zxczy2w/image.png
[2]: http://4k.com/wp-content/uploads/2014/06/4k-resolution-on-eyes.jpg