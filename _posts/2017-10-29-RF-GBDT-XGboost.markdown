---
layout: post
title: RF,GBDT and XGboost
date: 2017-10-29
author: GwanSiu
catalog: True
tags:
    - Machine Learning
---

> 通过该篇博文，你将会了解两方面的内容: (1)Emsembling learning; (2)bootstrap, bagging 和boost的基本概念极其所代表的基本模型;(3) Random Forest, GBDT, XGboost和LightGBM模型。

## 1. What‘s the emsembling learning?
集成学习(emsembling learning)是通过采样的方法训练多个弱分类器，最后将多个弱分类器组合起来形成强分类器的算法框架(通俗地说:三个臭皮匠赛过诸葛亮)。**为什么可以弱分类器可以形成强分类器(why?)** Emsembling Learning训练的弱分类器是要具有差异性的(差异性可能由不同算法，不同参数所导致)，从而导致弱分类器形成的决策边界不同。最后将所有弱分类结合后能得到更加合理的决策边界，从而减少整体的错误，实现更好的分类效果。**Emsembling Learning主要分成Baggging和boost两类。**

<img src="http://www.datakit.cn/images/machinelearning/EnsembleLearning_Combining_classifiers.jpg" width = "300" height = "200" align=center />

## 2. The basic concept of bootstrap, bagging and boost.

**Bootstrap:** bootstrap是一种有放回的抽样方法，在机器学习中又称**自助法**。Bootstrap在进行小样本估计时，效果更好。通过对子样本统计量的方差估计，进而构造置信区间。其思想和步骤如下：

1. 使用有放回的采样技术，从原有的样本集合中抽取一定数量的样本作为子样本集合。
2. 计算子样本集合的统计量T。
3. 重复(1)(2)步骤N次，得到N个不同的统计量T。
4. 计算N个统计量T的方差，得到统计量的方差，进而构造想相对应的置信区间。

 <img src="http://static.zybuluo.com/GwanSiu/vye8tevgtehauji37gt3nw8r/image.png" width = "300" height = "200" align=center />

 **Bagging(Boostrap aggregation):** 从训练集中使用Bootstrap的方法采样(有放回采样)样本子集合，分别在这些子集合上训练分类器，得到一组函数序列$f_{1},...,f_{n}$, 最后将所有训练好的弱分类器进行组合形成强的分类器：对于分类问题则进行投票，对于回归问题则可以进行简单的平均的方法。Bagging的代表算法是Random Forest。Bagging的主要思想如下图：

 <img src="http://static.zybuluo.com/GwanSiu/8naiwi9u9a4quonkmlsqdxp0/image.png" width = "300" height = "200" align=center/>

 **Boosting:**Boost同样是构造出一个函数序列(弱分类器)$f_{1},...,f_{n}$,与bagging不同的是，在boost算法框架下，后一个分类器依靠前一个分类器来生成：$f_{n}=F(f_{n-1})$。Boost算法更关注前一次错分的样本，boost算法的代表作是Adaboost，GDBT,xgboost算法。

 ## 3.Summary about bagging and boosting.
 
**Bagging和boosting的区别主要在四个方面:**

 1. **采样方式：**Bagging采用的是均匀采样的方式，而boosting则是根据错误率来进行采样，因此，boosting算法的精度要优于Bagging.
 2. **训练集的选择:** Bagging是随机选择训练集，在迭代中，训练集之间是相互独立的，而Boosting的训练集的选择则与前一次学习的结果有关。
 3. **弱分类器:** Bagging的训练出的弱分类没有权重，而在Boosting中，弱分类器是有权重的。
 4. **计算方式:**Bagging的各个弱分类器可以并行生成，而boosting的弱分类器只能按顺序生成。因此，bagging可以通过并行训练节省大量的时间开销。

**从机器学习的Bias和Variance理论分析Bagging与Boosting:**
在周志华老师的西瓜书中: Boosting主要是关注降低偏差(bias)，因此boosting能基于泛化能力较强的弱分类器上构建很强的集成模型。而Bagging主要关注降低方差(Variance)，因此它能在不剪枝的决策树上，神经网络上的效果更明显。

Bagging算法可以并行训练很多的具有差异性且相互独立的弱分类器，这样的做法可以有效地降低最终模型的方差(variance)，因为采用了很多弱分类器后，最后得到的强分类器$f$会有效地逼近真实的函数$h$。因此，对于Bagging算法的每个弱分类器而言，关键是要降低模型的偏差(bias)，所以要采用深度很深甚至不剪枝的决策树。

对于boosting算法，每一步迭代是在上一次迭代的基础对数据进一步拟合，所以可以保证偏差(bias)降低。因此，对于Boosting算法来说，弱分类器需要关注方差(variance),即关注更简单的分类器，因此，通常会采用深度很浅的决策树。   


参考文献
[1] http://blog.sina.com.cn/s/blog_4a0824490102vb2c.html(http://blog.sina.com.cn/s/blog_4a0824490102vb2c.html)

