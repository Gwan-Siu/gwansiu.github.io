---
layout: post
title: RF,GBDT and XGboost
date: 2017-10-29
author: GwanSiu
catalog: True
tags:
    - Machine Learning
---

> 通过该篇博文，你将会了解两方面的内容: (1)bootstrap, bagging 和boost的基本概念极其所代表的基本模型。(2) Random Forest, GBDT, XGboost和LightGBM模型。

## 1. The basic concept of bootstrap, bagging and boost.

**Bootstrap:** bootstrap是一种有放回的抽样方法，在机器学习中又称**自助法**。Bootstrap在进行小样本估计时，效果更好。通过对子样本统计量的方差估计，进而构造置信区间。其思想和步骤如下：

1. 使用有放回的采样技术，从原有的样本集合中抽取一定数量的样本作为子样本集合。
2. 计算子样本集合的统计量T。
3. 重复(1)(2)步骤N次，得到N个不同的统计量T。
4. 计算N个统计量T的方差，得到统计量的方差，进而构造想相对应的置信区间。

 <img src="http://upload-images.jianshu.io/upload_images/2671133-d7b7e56ca01b269b?imageMogr2/auto-orient/strip%7CimageView2/2" width = "300" height = "200" align=center />