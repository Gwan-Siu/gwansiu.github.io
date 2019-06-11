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

> In fact, English and Chinese are used in this session of analysis due to my limited English ability and I need to make some concept more understandable in Chinese.

## 1. What's Natural Numbers?

I belive most of us have learn natural number from primary school. At that time, we just know how to calculate natural number or apply some operation, such as addiction, substraction and etc. However, we didn't know what natural number is and why addiction should be like that. During the period of secondary school, we may receive an **informal definition** about natural numbers.

**Definition 1.1(informal):** A natural number is any element of the set

$$
\mathbb{N}:=\{0,1,2,3...,\},
$$

which is the set of all the numbers created by starting with 0 and then counting forward **indefinitely.** We call $\mathbb{N}$ the set of natural numbers. This definition **indeed** solve the problem of what natural numbers are. **However**, it still make us get confused in some sense. For instance, it don't give us a definition how to keep counting indefinitely without cycling back to 0. Also, how to perform operator such as addiction, multiplication, and exponentiation on natural numbers? 

Actually, we can define complicated operations in terms of simple operations. For instance, exponentiation is nothing but repeated multiplication; multiplication is nothing but repeated addiction. **What's the addiction?** Addiction is nothing but counting forward, or increment. 

Thus, to define natural numbers, we will use **two fundamental concept:** the zero number 0, and the increment operation. Let's start Peano axiom!

## 2. Peano's axiom

**Axiom 2.1.** *0 is a natural number.*
**Axiom 2.2.** *If n is a natual number, then n++ is also a natural number.* 

> **Axiom 2.1 and Aximo 2.2** 规定了自然数的起始点是0，并规定了自然数增量依旧是自然数。因此，我们可以定义1，2，3都是自然数，但是这并不*足够*描述我们所理解的自然数。如果我们考虑这样一个数字系统仅有{0，1，2，3}(现在请允许使用集合符号，方便解释)，3的增量等于0，相当于这个数字系统形成一个环域，这样的系统也依旧满足**Axiom 2.1 and Aximo 2.2**,但并不是我们通俗理解的自然数系统，因此，我们需要更多公理去避免环域的情况发生。

**Axiom 2.3.** *0 is not the successor of any natural number, i.e., we have* $n++ \neq 0$ *for every natural number n.*

>**Axiom 2.3** 仅仅规定了0不能是任何自然数的后继，因此，从**Axiom 2.3**,我们可以回答4不等于0等一系列的问题。注意我在这里的用词**仅仅**，这里只是保证自然数在** ++** 后不回到0，假设4++=4这种自循环情况，依旧满足 **Axiom 2.1~2.3**的情况，因此我们还需要公理2.4。

**Axiom 2.4.** *Different natual numbers must have different successors, i.e., if n,m are natual numbers and* $n \neq m$ *, then* $n++ \neq m++$. *Equivalently, if* $n++ = m++$, then we must have $n = m$.

>**Axiom 2.4**规定了不同自然数的后继必定是不同的，这保证了自然数在++过程中不会有4++=4，以及6=2的情况。(非正式)证明，假设6=2，那么5++=1++，则5=1,往后一直推导，则有4=0，这违反了 **Axiom 2.3**，因此，题设不成立。6不等于2。**Axiom 2.1~2.4**可以足够保证我们可以将不同的自然数分开(即1是1，不会等于2)，但还有若该自然数系统存在一个奇奇怪怪的数字，如pi,0.5,0.67,1.5等。举个(informal)例子半个++增量，$\mathbb{N}={0,0.5,1,1.5,2,2.5,3,3.5,...}$, 这个例子可以看到，增量是原来的半个。我们现有的 **Axiom2.1~2.4** 仅仅只是说自然数可以由0和增量++得到。但并未就*得到*这个概念予以明确的定义，因此，我们不能保证我们所定义的自然数系统中不会有一些奇奇怪怪的数字或者符号。

**Axiom 2.5.(Principle of mathematical induction):** *Let* $P(n)$ *be any property pertaining to a natual number on* $n$. *Suppose that* $P(0)$ *is true, and suppose that whenever* $P(n)$ *is true*, $P(n++)$ *is also true. Then* $P(n)$ *is true for every natural number n.*

>**Axiom 2.5**就是我们通俗意义上说的*数学归纳法*，它保证了所定义的自然数系统是不会存在一些奇奇怪怪的数字或者符号。相对于**Axiom**,其实**Axiom 2.5**更多是一种**Axiom scheme**(公式模板)。数学归纳法通常分为两步: (1).base case成立; (2).假设case n是成立的，证明case n++是成立的。
>>注意: 这里仅仅只是规定了什么是自然数，自然数有哪些property，但并没有涉及到自然数的运算。

## 3. Addiction

有了自然数，我们便可以在自然数定义的基础上定义自然数的运算。在自然数的定义了规定了*increament*原始的运算法则，因此，我们便需要在这基础上定义更复杂的运算。在更复杂的运算中，我们便需要从复杂运算运算中简单运算开始，加法(Addiction)。

**Definition 3.1(Addition of natural numbers).** Let m be a natural nummber. To add zero to m, we define 0+m := m. Now suppose inductively that we have defined how to add $n$ to $m$. Then we can add $n++$ to $m$ by defining $(n++)+m := (n+m)++$.

>**特别注意:**该定义只是跟我们阐述了两件事：1.$0+m=m$, 2.$(n++)+m=(n+m)++$。因此，我们并不能从该定义得出$0+m=m+0$(加法交换律)。可见，该定义本质上是不对称的(asymmetric), $3+5$表示5增加3次，而$5+3$表示3增加5次。而这两者的结果是否相等，不能直接得出。因此，我们需要一些lemmas来辅助我们证明交换律，结合律以及分配律。

**Lemma 3.2.** *For any natural number n, n+0=n.*(数学归纳法)
**Lemma 3.3.** *For any natural numbers n and m, n+(m++)=(n+m)++.*（数学归纳法）

>从**Lemma 3.2-3.3**中可以推出交换律，结合律以及分配率。

**Proposition 3.4(Addiction is commutative).** *For any natural numbers n and m, n+m=m+n.*

**Proposition 3.5(Addiction is associative).** *For any natural numbers a,b and c, we have (a+b)+c=a+(b+c).*

**Proposition 3.6(Cancellation is law).** *Let a,b,c be natural numbers such that a+b=a+c. Then we have b=c.*

有了加法的运算之后，我们便可以对自然数比较大小,也就是定义一个集合的order。

**Definition 3.7(Ordering of the natural numbers).** Let $n$ and $m$ be natural numbers. We say that $n$ is *greater than or equal to* m, and write $n\geq m$ or $m \leq n$, iff we have $n=m+\alpha$ for some natural number $\alpha$. We say that $n$ is *strictly greater than m*, and write $n>m$ or $m<n$, iff $n\geq m$ and $n\neq m$.

**Proposition 3.8(Basic properties of order for natural numbers).** *Let a,b,c be natural numbers. Then*

- *(Order is reflexive)* $\alpha \geq \alpha$.
- *(Order is transitive)* If $\alpha \geq b$ and $b \geq c$, then $\alpha \geq c$.
- (Order is anti-symmetric) If $\alpha \geq b$ and $b \geq \alpha$, then a=b.
- (Addiction preserves order) $\alpha \geq b$ if and only if $\alpha +c \geq b++c$.（加法是具有保序性的）
- $a<b$ if and only if $a++\leq b$.
- $\alpha < b$ if and only if $b=a+d$ for some positive number $d$.

## 4. Multipication

定义加法后，为了简化加法的运算，顺理成章地可以定义更为复杂的运算:乘法(Multiplication)。通俗理解上，乘法不过是几次重复的加法。

**Definition 4.1(Multiplication of natual number).** Let $m$ be a natural number. To multiply zero to m, we define $0\times m$ := 0. Now suppose inductively that we have definded how to multiply $n$ to $m$. Then we can multiply $n++$ to $m$ by defining $(n++)\times m := (n\times m)+m$.

>同样，乘法的定义也是不对称，且并没有很直接可以得出交换律的。但我们可以使用证明*Lemma 3.2*和*Lemma 3.3*的手法证明交换律以及结合律。

**Lemma 4.2(Multiplication is commutative)** Let n,m be natural numbers. Then $n\times m = m\times n$.

**Proposition 4.3(Distributive law)** For any natual numbers a,b,c, we have a(b+c)=ab+bc  and (b+c)a=ab+bc.

**Porposition 4.4(Multiplication is associative)**. For any natural numbers a,b,c, we have $(a\times b)\times c=a\times (b\times c).$

**Proposition 4.5(Multiplication preserves order).** If a,b are natural numbers such that $a<b$ and c is positive, then $ac<bc$.

**Corollary 4.6(Cancellation law).** Let a,b,c be natural numbers such that ac=bc and c is non-zero, Then $a=b$.

有了自然数的乘法后，我们在这插播一个著名的算法-欧几里得算法。

**Definition 4.7(Euclidean algorithm).** Let n be a natural number, and let q be a positive number. Then there exist natual numbers m,r sucha that $0\leq r\leq q$ and $n=mq+r$.

## 5. Exponentiation

同样的道理，我们在乘法的基础上定义幂运算。通俗来讲，当一个数重复相乘的时候，我们为了简化运算，从而定义出幂运算(exponentiation).

**Definition 5.1(Exponentiation for natural natural numbers).** Let $m$ be a natural number. To raise $m$ to the power 0, we define $m^{0}=1$; in particular we define $0^{0}=1$. Now suppose that $m^{n}$ has been defined for some natural number $n$, then we define $m^{n++}:=m^{n}\timnes m$.

## 6. Positive Numbers

其实，有了自然数及其加法的定义，我们可以对自然数进一步划分，将自然数划分成positive numbers and non-positive numbers.

**Definition 6.1(Positive natural numbers)** A natural number $n$ is said to be pisitive iff it is not equal to 0.

由此我们可以推论出，两个正数相加还是正数，两个非零正数相乘还是正数。

**Proposition 6.2** If $a$ is positive and $b$ is a natural number, then $a+b$ is positive.
>从propostion 6.2 可以推出一些有趣的lemma和corollary.

**Corollary 6.3**. If $a$ and $b$ are postiive numbers such that $a+b=0$, then $a=0$ and $b=0$.

**Lemma 6.4**. Let $a$ be a positive number. Then there exists exactly one natural number $b$ such that $b++=a$.(Hint:存在性与唯一性都要进行证明，如何证明存在？如何证明唯一性？)

**Proposition 6.5(Positive natural nummbers have no zero divisors).** Let n,m be natural numbers. Then $n\times m=0$ if and only if at least one of $n,m$ is equal to zero. in particular, if $n$ and $m$ are both positive, then $nm$ is also positive.

## 7. isomorphic, axiomatic and constructive, Recursive definitions.
### 7.1 isomorphic

从很简单的一个问题开始: 在我们已经定义的自然数系统$\mathbb{N}$之外，是否存在另外一个自然数系统，与我们定义的自然数不一定？ 比如 HinduArabic number system {0,1,2,...}以及罗马数字 {O,I,II,III,IV,V,...}, 问这两个系统是否不同？答案是：这两个系统等价，存在一个one-to-one mapping使得$0\leftrightarrow O$, $1\leftrightarrow I$,...,而这种等价，我们称为同构(isomorphic).

### 7.2 Axiomatic and Constructive
公理化与构造法表现在自然数上，是在探索自然数的产生。自然数是被构造出来？还是被定义出来？

### 7.3 Recursive definition

**Proposition(Recursive definition).** Suppose for each natural number $n$, we have some function $f_{n} : \mathbb{N}\rightarrow \mathbb{N}$ from the natural numbers to the natural numbers. Let $c$ be a natural number. Then we can assign a unique natural number $a_{n}$ to each nutural num ber n, such that $a_{0}=c$ and $a_{n++}=f_{n}(a_{n})$ for each natural number n. 