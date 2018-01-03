---
layout:     post
title:      "Analysis--Natural Numbers"
date:       2017-12-27 12:00:00
author:     "GwanSiu"
catalog: true
tags:
    - Analysis
---

> At the begining of this year, I plan to learn mathematics analysis and it will be helpful for my future research. Thus, I decide to open a new session in my blog and write some article about that. This's my first article about analysis and the reference book is Analysis I & II written by Terence Tao. In this article, I talk about Peano axiom, addiction and multiplication.

## 1. What's Natural Numbers?

I belive most of us have learn natural number from primary school. At that time, we just know how to calculate natural number or apply some operation, such as addiction, substraction and etc. However, we didn't know what natural number is and why addiction should be like that. During the period of secondary school, we may receive an **informal definition** about natural numbers.

**Definition 1.1(informal):** A natural number is any element of the set

$$
\mathbb{N}:={0,1,2,3...,},
$$

which is the set of all the numbers created by starting with 0 and then counting forward **indefinitely.** We call $\mathbb{N}$ the set of natural numbers. This definition **indeed** solve the problem of what natural numbers are. **However**, it still make us get confused in some sense. For instance, it don't give us a definition how to keep counting indefinitely without cycling back to 0. Also, how to perform operator such as addiction, multiplication, and exponentiation on natural numbers? 

Actually, we can define complicated operations in terms of simple operations. For instance, exponentiation is nothing but repeated multiplication; multiplication is nothing but repeated addiction. **What's the addiction?** Addiction is nothing but counting forward, or increment. 

Thus, to define natural numbers, we will use **two fundamental concept:** the zero number 0, and the increment operation. Let's start Peano axiom!

## 2. Peano's axiom

**Axiom 2.1.** *0 is a natural number.*
**Axiom 2.2.** *If n is a natual number, then n++ is also a natural number.* 

**Axiom 2.3.** *0 is not the successor of any natural number, i.e., we have* $n++ \neq 0$ *for every natural number n.*

**Axiom 2.4.** *Different natual numbers must have different successors, i.e., if n,m are natual numbers and* $n \neq m$ *, then* $n++ \neq m++$. *Equivalently, if* $n++ = m++$, then we must have $n = m$.

**Axiom 2.5.(Principle of mathematical induction)** *Let* $P(n)$ be any property pertaining to a natual number on $n$. Suppose that $P(0)$ is true, and suppose that whenever $P(n)$ is true, $P(n++)$ is also true. Then $P(n)$ is true for every natural number n.