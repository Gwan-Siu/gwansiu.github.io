---
layout:     post
title:      "Analysis--Set Theory"
date:       2017-12-27 12:00:00
author:     "GwanSiu"
catalog: true
tags:
    - Analysis
---

>In this article, ZFC axiom will be discussed.

## 1. ZFC Axiom

**Axiom 1(Sets are objects):** If A is a set, then A is also an object. In particular, given two sets A and B, it is meaningful to ask whether A is also an element of B.

**Axiom 2(Equality of sets):** Two sets A and B are equal, A=B, iff every element of A is an element of B and vice versa. To put it another way, A=B if and only if every element $x$ of A belongs also to B, and every element $y$ of B belongs also to A.

**Axiom 3(Empty set):** There exists a set $\varnothing$, known as the empty set, which contains no elements, i.e., fpr every object $x$ we have $x\in \varnothing$

**Axiom 4(Singleton sets and pair sets):** If $a$ is an object, then there exists a set {$a$} whose only element is $a$, i.e., for every object $y$, we have $y\in {a}$ if and only if $y=a$; we refre to {$a$} as the singleton set whose element is $a$. Furthermore, if $a$ and $b$ are objects, then there exists a set {$a,b$} whose only elements are $a$ and $b$; i.e., for every object $y$, we have $y\in {a,b}$ if and only if $y=a$, or $y=b$; we refere to this set as the pair set formed by $a$ and $b$.

**Axiom 5(Pairwise union):** Given any two sets $A,B$, there exists a set $A\cup B$, called the union of A and B, which consists of all the elements which belong to A or B or both. In other words, for any object $x$,
$$
x\in A\cup B \rightleftarrow(x\in A\text{ or }x\in B)
$$

**Axiom 6(Axiom of specification):**  Let A be a set, and for each $x\in A$, let $P(x)$ be a property pertaining to $x$ (i.e., $P(x)$ is either a true statement or a false statement). Then there exists a set, called {$x\in A: P(x)\text{ is true}$}(or simply {$x\in A:P(x)$} for short), whose elements are precisely the elements $x$ in A for which $P(x)$ is true. In other words, for any object $y$,
$$
y\in {x\in A:P(x)\text{ is true}} \rightleftarrow (y\in A \text{ and }P(y)\text{ is true}).
$$

**Axiom 7(Replacement).** Let A be a set. For any object $x\in A$, and any object $y$, suppose we have a statement $P(x,y)$ pertaining to $x$ and $y$, such that for each $x\in A$ there is at most one $y$ for which $P(x,y)$ is true. Then there exists a set {$y:P(x,y) \text{ is true for some }x\in A$}，such that for any object $z$,
$$
z\in{y:P(x,y)\text{ is true for some }x\in A}\rightleftarrow P(x,y) \text{ is true for some }x\in A.
$$

**Axiom 8(Infinity).** There exists a set $\mathbb{N}$, whose elements are called natural numbers, as well as an object 0 in $\mathbb{N}$, and an object $n++$ assigned to every natural number $n\in \mathbb{N}$, such that the Peano axioms hold.

**Axiom 9(Universal specification).** Suppose for every object x we have property $P(x)$ pertaining to $x$ (so that for every $x$, p(X) is either a true statement or a false statement). Then there exists a set {$x:P(x) \text{ is true}$} such that for every object $y$,
$$
y\in {x:P{x} \text{ is true}} \rightleftarrow P(y)\text{ is true}.
$$

**Axiom 10(Regularity):** If A is a non-empty set, then there is at least one element $x$ of $A$ which is either not a set, or is disjoint from A.

**Axiom 11（Power set axiom):** Let $X$ and $Y$ be sets. Then there exists a set, denoted $Y^{X}$, which consists of all the functions from $X$ to $Y$, thus,
$$
f\in Y^{X} \rightleftarrow (f \text{ is a function with domain X anf range Y.})
$$

**Axiom 12 (Union):** Let A be a set, all of whose elements are themselves sets. Then there exists a set $\bigcup A$ whose elements are precisely those objects which are elements of the elements of A, thus for all objects as 
$$
x\in \bigcup A \leftrighrarrow (x\in S \text{ for some }S\in A)
$$