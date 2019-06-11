---
layout:     post
title:      "Faster R-CNN"
date:       2017-07-22 12:00:00
author:     "GwanSiu"
catalog: true
tags:
    - Segmentation and Dectection
---

## 1. Introduction

在我的文章[Fast RCNN](http://gwansiu.com/2017/07/22/Fast-RCNN/)提到: Fast RCNN将特征提取，分类器和bounding-box regressors融合在一起，使RCNN的多阶段训练变成2阶段训练。Region proposals占用了大量的时间复杂度，原因是Fast RCNN使用selective search方式通过CPU进行计算region proposals。在Faster RCNN中则提出使用**RPN**结构将Region proposals融合进网络中，使用GPU进行计算，从而进一步降低时间复杂度。**RPN的使用条件是，经验发现在conv features map中包含着物体的位置信息，因此可以在conv features map中进行region proposals的提取。**

## 2. What's the RPN?
RPN(Region Proposals Network)是一种特殊的[全卷积网络结构](http://gwansiu.com/2017/07/04/note-cs231n/),它类似于sliding windows, 在conv features map上的每个位置滑动提取region proposals，并将其映射成一个低维向量,如:ZFnet中的256个维度
，或者VGGnet中的516个维度。为了保证提取的特征具有scale-invariant的特性，作者提出了anchor的机制，anchor其实是RPN的windows的大小，在conv features maps的每一个位置使用不同尺度和长宽比的windows提取特征，在文章中n=3，意味着是3个尺度，3个同的长宽比，每一个conv features maps的位置产生9个anchors, 使得最后一层fc能在9个不同的anchors中都能学到物体以及物体的位置信息。所以这就是为什么这样做能使得faster RCNN具有scale-invariant的特性。那么问题来了，**是不是anchors的种类越多越能保证scale-invariant property呢？** 博主认为，并不是越多越好，要控制在一定的数量，理由是：1.anchors多了就会造成faster RCNN的时间复杂度提高，anchors多的最极端情况就是overfeats中sliding windows。 2.使用多尺度的anchors未必全部对scale-invariant property都有贡献是相等的，我认为存在一种边际效益递减的现象([经济学术语](https://zh.wikipedia.org/wiki/%E5%A0%B1%E9%85%AC%E9%81%9E%E6%B8%9B))。RPN与anchors的示意图如下:  
![image.png-631.7kB][2]


## 3. The diagram of Faster RCNN

![--][3]  
在[cs231n note](http://cs231n.stanford.edu/slides/2017/cs231n_2017_lecture11.pdf)中对这张图作了描述:将RPN插入到Fast RCNNz中从features map中提取region proposals,联合训练4个loss函数:
1. RPN classify object/not object.
2. RPN regress box coordinate.
3. Final classification score(object class).
4. Final box coordinate.

Faster RCNN的流程如上图:思想很简单，就是提出RPN将Region Proposals融合进Fast RCNN中。换做博主，直接将RPN网络插入到卷积层与ROI层之间一起训练就好了。**但是这种想法是不对的，一定程度上，我并没有考虑两种网络结构的兼容性。**原因一:作者一直强调:**shared convolutional feature.** RPN只是提取region proposals, 前提是不管如何提取region proposals，convolutional feature都能提取出有效特征，所有conv features的有效性要share在所有region proposals, 一起训练会造成RPN网络会以不同方式改变conv的参数，需要谨慎考虑的是DL网络结构找到的都是local minimum。原因二:Fast RCNN的训练时是取决于固定size的region proposals,这个条件并不能保证党RPN与Fast RCNN同时训练时，网络会收敛。  

为了保证“shared convolutional feature”,作者采用了四步训练:  
1. 使用Imagenet的预训练模型，端到端微调训练RPN结构。
2. 使用第一步训练好的RPN结构提取region proposals训练Fast RCNN. Fast RCNN使用image net上调的参数训练。
3. 初始化RPN结构，冻结Fast RCNN的conv layer，只训练RPN网络结构。(这时已经有shared conv feature property)
4. 继续冻结Fast RCNN, 仅仅微调Fast RCNN的fc layer，从而使两个网络彻底融合在一起。

最后来一张RCNN, Fast RCNN与Faster RCNN的测试时间对比:

![image.png-198.3kB][4]

## 5.结束语
RCNN, Fast RCNN和Faster RCNN系列就算完成了，这也算是我了解目标检测入门的三篇经典文章吧，如有错误，请在评论处批评指正。网上也有许多写这三篇论文的相关文章，大家也可以相互参考。我做研究的习惯是，先抓住主要矛盾，了解基本问题，之后再仔细推敲和思考。希望能大家有帮助。

如果喜欢我的文章，请顺手点个赞。

[2]: http://static.zybuluo.com/GwanSiu/4vg0mk6eeyc9qwue9sngyujn/image.png
[3]: http://shuokay.com/content/images/faster-rcnn/faster-rcnn-global.png
[4]: http://static.zybuluo.com/GwanSiu/oy8s26k8vg8o09h1tz57opzq/image.png
