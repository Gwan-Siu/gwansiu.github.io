---
layout:     post
title:      "Random Forest"
date:       2018-12-16 12:00:00
author:     "GwanSiu"
catalog: true
tags:
    - Machine Learning
---

## 1. Random Forest

Decision tree is a classical algorithm in machine learning area, but decision tree suffer from the drawback of low biase and high variance(overfitting). Random forest is a boosting algorithm that boots a collection of decision trees, because boost algorithm can effectively reduce the variance. Random forest is very effective and powerful algorithm in machine learning area, and payoff of random forest is explanation of model.

For random forest, each tree

1. N samples with replacement and each sample is i.i.d.(Bootstrap strategy)
2. No prune for each tree
3. Randomly select m feature.

The error rate of random forest depends on two things:

1. The correlation between any 2 trees, i.e. increasing the correlation between 2 trees means increasing the error rate of random forest.
2. The strength of each individual tree in the forest. In othr words, a tree with a low error rate is a strong classifier. Increasing the strength of the individual trees means to decreases the error rate of forest.

The only hyper-perameter is $m$, the number of randomly selected feature. Reducing $m$ will reduce both the correlation and the strength, and increasing $m$ will increase both. The value of $m$ can be quickly found by using the oob rate.
