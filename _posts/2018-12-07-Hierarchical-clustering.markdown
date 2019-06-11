---
layout:     post
title:      "Hierarchical Clustering"
date:       2018-12-07 12:00:00
author:     "GwanSiu"
catalog: true
tags:
    - Machine Learning
---

## 1. Introducation to Hierarchical Clustering

Clustering algorithm is a process to group similar object together. From the view of ininput type, clustering algorithm can be divied into two categories: **similarity-based clustering** and **feature-based clustering**. In similarity-based clustering, the input is $N\times N$ dissimilarity matrix or distance matrix $D$. In feature-based clustering, the input is $N\times D$ feature matrix. From the view of output, partitional clustering and hierarchical clustering are two kinds of clustering algorithm. In this article, I will talk about hierarchical clustering, figure 1 is shown the structure of hierarchical clustering. Hierarchies are commonly used to organize information. There are two kinds structures of hierarchical clustering: **bottom-up agglomerative clustering**, **top-bottom divise**.

## 2. Bottom-up Agglomerative Clustering

Agglomerative clustering starts with $N$$ groups, where $N$ is the number of sample. In other words, it means that the initial group contains one sample. Then at each step, it merges the two most similar group until there is a single group containing all samples.

**Complexity analysis:** In each step, merge two similar group will take $O(N^{2})$, and there are $O(N)$ steps in algorithm, and thus the total cost is $O(N^{3})$. By applying priority queue, it can be reduced to $O(N^{2}\log N)$. For a large $N$, we can apply K-means first, and the complexity becomes $O(KND)$.

In agglomerative clustering, there are 3 commonly used ways to define "similarity" of group: single link, complete link, and average link.

### 2.1 Single Link

single link clustering is called nearest neighbor clustering, the distance between 2 groups $G$ and $H$ is defined as the distance between the two closest members of each group:

$$
\begin{equation}
d_{SL}(G,H) = \min_{i\in G, i^{'}\in H}d_{i,.i^{'}}
\end{equation}
$$

single link algorithm actually takes $O(N^{2})$ time.

### 2.2 Complete Link

complete link is called furthest neighbor clustering. The distance between 2 groups is defined as the distance between 2 most distant pairs:

$$
\begin{equation}
d_{CL}(G,H)=\max_{i\in G,i^{'}\in H}d_{i,i^{'}}
\end{equation}
$$

complete link can preserve compactness of group because all the data in one group are similar, while single link violate the compactness property.

### 2.3 Average Link

average link measure the average distance between 2 groups:

$$
\begin{equation}
d_{avg}(G,H) =\frac{1}{n_{G}n_{H}}\sum_{i\in G}\sum_{i^{'}\in H}d_{i,i^{'}}
\end{equation}
$$

where $n_{G}$ and $n_{H}$ are the number of elements in groups $G$ and $H$.

## 3. Top-Up Agglomerative Clustering

Top-up clustering is opposite to bottom-up clustering. It starts with a single group, and then recursively divides each clsuter into 2 subclusters, in top-down fashion. Since there are $2^{N-1}$ ways to split a group of $N$ items into 2 groups, it is hard to compute th optimal split, and thus various heuristics are adopted. More detail you can see in [1]

## 4 Summary

- In the first iteration, all hierarchical clustering algorithm need to copute similarity of all paris of $n$ individual instances which is $O(n^{2})$.
- In each of the subsequent $n-2$ merging iterations, comput the distance between the most recently created cluster and all other existing clusters.
- In order to maintain an overall $O(n^{2})$ performance, computing similarity to each other cluster must be done in constant time.
- Else $O(n^{2}\log n)$ or $O(n^{3})$ if done naively. 


## Reference

[1] Robert C. Machine learning, a probabilistic perspective[J]. 2014.
[2] [Lecture:Clustering and Distance Metrics, 10701-Introduction to machine learning, Eric Xing](http://www.cs.cmu.edu/~mgormley/courses/10701-f16/slides/lecture15-clustering.pdf)