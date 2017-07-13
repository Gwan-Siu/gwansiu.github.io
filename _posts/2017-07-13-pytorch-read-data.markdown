---
layout:     post
title:      "Pytorch Learning"
subtitle:   "How to read your own dataset"
date:       2017-07-13 12:00:00
author:     "GwanSiu"
catalog: true
tags:
    - pytorch
---

## 1.数据的存放
首先，将自己的数据集按照如下形式存放：
![---][1]

## 2. 数据的预处理
数据的预处理主要是三步：<p>torch.transoforms.Compose</p>,<p>torchvision.datasets</p>和<p>torch.utils.data.DataLoader</p>,分别是数据预处理的方法，加载数据集和形成数据生成器。

**<p>torch.transoforms.Compose：</p>** 可以把图像预处理的方法都集中起来，按照编写的顺序方式，按顺序对图像进行预处理。**注意：图像预处理的操作只对于PIL格式图像，在处理完之后需要转化成Tensor:<p>transforms.Tosensor</p>**
'''python
import torch
import torchvision
import torchvision.transforms as transform

transform = {'train': transform.Compose([transform.Scale(224,224),
transform.ToTensor(),transform.Normalize(mean=(0.5,0.5,0.5),std=(0.5,0.5,0.5))]),
'test':transform.Compose([[transform.Scale(224,224),
transform.ToTensor(),transform.Normalize(mean=(0.5,0.5,0.5),std=(0.5,0.5,0.5))])}
'''

















[1]: http://static.zybuluo.com/GwanSiu/0hfowvav2axd1ggtjs0hdqpu/image.png

