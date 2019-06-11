---
layout:     post
title:      "Analysis: Interger and Rational Number"
date:       2018-04-08 12:00:00
author:     "GwanSiu"
catalog: true
password: 123456
tags:
    - Analysis
---

## 1. Integers

### 1.1 The definition of intergers

**(Definition-integer):** An `integer` is an expression of the form $a-b$, where $a$ and $b$ are natural numbers. Two integers are considered to be equal, $a-b=c-d$, if and only if $a+d=c+b$. We ler $\mathbb{Z}$ denote the set of all integers.

`(This definition define what's the integers and the equality of integers. The natural number is the foundation of intergers, hence we use natural numbers to define intergers and the equality of integers(Firstly, we should check if it is well-defined.))`

`Integer number system is the extension of integers system, so the operation in intergers systems should be well-defined in the natural numbers and the properties, such as: reflexivity, sysmmetry, transitivity and substitution, should be verifed.`

Compared with natural number, negtive number is included in the integers system, of course it should be well-defined.

**(Definition: negation of integers):** If $(a-b)$ is an integer, we define the negation $-(a-b)$ to be the integer $(b-a)$. In particular, if $n=n-0$ is a positive natural number, we can define its negation $-n=0-n$.

Now, in the integer number system, we have positive number, zero and negative number system. But this justification is not be verified. We need to prove it.

**(Trichotomy of integers):** Let $x$ be an integer. Then exactly one of the three statements is true: (a). $x$ is equal to zero; (b). $x$ is equal to a positive number $n$; or (c). $x$ is the negation $-n$ of a positive natual number $n$.

**(Laws of algebra for integers).** Let $x,y,z$ be integers. Then we have:

$$
\begin{aligned}
x +y &= y+x \\
(x+y) + z &= x+(y+z) \\
x+0=0+x&=x \\
x+(-x)=(-x)+x &= 0 \\
xy&=yx  \\
(xy)z &= x(yz) \\
x1 = 1x &= x \\
x(y+z) &= xy + xz \\
(y+z)x &= yx + zx
\end{aligned}
$$

From the law of algebra for intergers, we should prove one proposition and corollary:

**Proposition(Integers have no zero divisors):** Let $a$ and $b$ be integers such that $ab=0$. Then either $a=0$ or $b=0$(or both).

**Corollary:** If $a,b,c$ are integers such that $ac=bc$ and $c$ is non-zero, then $a=b$.

### 1.2 the order of intergers

**Definition(Order of the integers):** let $n$ and $m$ be intergers.  We say that $n$ is greater than or equal to $m$, and write  $n\geq m$ or $m\leq n$, iff we have $n=m+a$ for some natual number $a$. We say that $n$ is strictly greater than $m$, and write $n>m$ or $m<n$, iff $n\geq m$ and $n\neq m$.

**(properties of order).** Let $a,b,c$ be integers.

1. $a>b$ if and only if $a-b$ is positive natural numerb.
2. (Addiction preserve order): if $a>b$, then $a+c>b+c$.
3. (Positive multiplication preserve order): If $a>b$ and $c$ is positive, then $ac>bc$.
4. (Negation reserves order) If $a>b$, then $-a<-b$.
5. (Order is transitive) If $a>b$ and $b>c$, then $a>c$.

## 2. Rational Number

The integer system is constructed with operations of addition, substruction, multiplication, and order and verified all the expected algebraic and order-theoretic properties. Hence, we use similar way to define `rational number`.

### 2.1 The definition of rational number

**(Definition of rational number):** A `rational number` is an expression of the form $a//b$, where $a$ and $b$ are integers and $b$ is non-zero; $a//0$ is note considered to be a rational number.  Two rational numbers are considered to be equal, $a//b=c//d$, if and only if $ad=cb$. The set of all rational numbers is denoted $Q$.

`Similarly, the definition well define what is the rational number and the equality of two rational number. In addiction, rational defined by integers numbers.`

The same as rational number, firstly we should veried the operation of addition, substruction, multiplication and then add division into rational number system. Secondly, the order of rational number system is re-defined, which can be compatible with integers system.

**The property of rational numerb** $\mathbb{Q}$ can make up the definition of `field`, **a field is any set where addition and multiplication are well-defined operations that are commutative, associative, and obey the familiar distributive property** $a(b+c)=ab+ac$.

**(Laws of algebra for integers).** Let $x,y,z$ be rationals. Then we have:

$$
\begin{aligned}
x +y &= y+x \\
(x+y) + z &= x+(y+z) \\
x+0=0+x&=x \\
x+(-x)=(-x)+x &= 0 \\
xy&=yx  \\
(xy)z &= x(yz) \\
x1 = 1x &= x \\
x(y+z) &= xy + xz \\
(y+z)x &= yx + zx
\end{aligned}
$$

if $x$ is non-zero, we also have:

$$
\begin{equation}
xx^{-1}=x^{-1}x=1
\end{equation}
$$


**Definition:** A rational number $x$ is said to be positive iff we have $x=a/b$ for some positive intergers $a$ and $b$. It is said to be negative iff we have $x=-y$ for some positive rational $y$ (i.e., x=(-a)/b) for some positive integers $a$ and $b$)

**(Trichotomy of integers):** Let $x$ be an rational numebr. Then exactly one of the three statements is true: (a). $x$ is equal to zero; (b). $x$ is equal to a positive rational number; or (c). $x$ is the negative rational number.

### 3. Absolute value and exponentiation

**Definition(absolute value):** If $x$ is a rational number, the *absolute value* $\vert x\vert$ of $x$ is defined as follows. If $x$ is positive, then $\vert x\vert:=x$. If $x$ is negative, then $\vert x\vert:=-x.$ If $x$ is zero, then $\vert x\vert:=0$.

**Definition(Distance):** Let $x$ and $y$ be rational numbers. The quantity $\vert x-y\vert$ is call the `distance` between $x$ and $y$, and is sometimes denoted $d(x,y)$, thus $d(x,y):=\vert x-y\vert$. For instance, $d(3,5)=2$.

**(Properties of absolute value and distance):** Let $x,y,z$ be rational numnbers.

1. (Non-degeneracy of absolute value) We have $\vert x\vert \geq 0$, also, $\vert x\vert=0$ if and only if $x$ is 0.
2. (Triangle inequality for absolute value) We have $\vert x+y\vert \leq \vert x\vert +\vert y\vert$.
3. We have the inequalities $-y\leq x\leq y$ if and only if $y\geq \vert x\vert$. in particular, we have $-\vert x\vert \leq x \leq \vert x\vert$.
4. (Multiplicativity of absolute value) We have $\vert xy\vert=\vert x\vert \vert y\vert$. In particular, $\vert -x\vert =\vert x\vert$.
5. (Non-degeneracy of distance) We have $d(x,y)\geq 0$. Also, $d(x,y)=0$ if and only if $x=y$.
6. (Symmetry of distance) $d(x,y)=d(y,x)$.
7. (Triangle inequality for distance) $d(x,y)\leq d(x,y)+d(y,z)$.

**Definition**($\epsilon$-closeness). Let $\epsilon>0$ be a rational number, and let $x,y$ be rational numbers. We say that y is $\epsilon$-close to $x$ iff we have $d(x,y)\leq \epsilon$.

**Property:** Let $x,y,z,w$ be rational numbers.

1. If $x=y$, then $x$ is $\epsilon$-close to $y$ for every $\epsilon>0$. Conversely, if $x$ is $\epsilon$-close to $y$ for every $\epsilon >0$, then we have $x=y$.
2. Let $\epsilon >0$. If $x$ is $\epsilon$-close to $y$, then $y$ is $\epsilon$-close to $x$.
3. Let $\epsilon,\delta>0$. If $x$ is $\epsilon$-close to $y$, snf $y$ is $\delta$-close to $z$, then $x$ and $z$ are $(\epsilon+\delta)$-close.
4. Let $\epsilon,\delta>0$. If $x$ and $y$ are $\epsilon$-close, and $z$ and $w$ are $\delta$-close, then $x+z$ and $y+w$ are $(\epsilon+\delta)$-close, and $x-z$ and $y-w$ are also $(\epsilon+\delta)$-close.
5. Let $\epsilon >0$. then $x$ is $\epsilon$-close to $y$, they are also $\epsilon^{`}$-close for every $\epsilon^{'}>\epsilon$.
6. Let $\epsilon > 0$. If $y$ and $z$ are both $\epsilon$-close to $x$, and $w$ is between $y$ and $z$ (i.e., $y\leq w\leq z\leq z$ or $z\leq w\leq y$), then $w$ is also $\epsilon$-close to $x$.
7. Let $\epsilon>0$, if $x$ and $y$ are $\epsilon$-close, and $z$ is non-zero, then $xz$ and $yz$ are $\epsilon\vert z\vert$-close.
8. Let $\epsilon,\delta>0$. If $x$ and $y$ are $\epsilon$-close, and $z$ and $w$ are $\delta$-close, then $xz$ and $yw$ are $(\epsilon\vert z\vert +\delta\vert x\vert +\epsilon\delta)$-close.

## 3. Exponentiation

**Defintion:**(Exponentiation to a natural number, I). Let $x$ be a rational number. To raise $x$ to be the power 0， we define $x^{0}:=1$
; in particular, we define $0^{0}=1$. Now suppose inductively that $x^{n}$ has been defined for some natural number $n$, then we define $x^{n+1}:=x^{n}\times x$.

(Properties of exponentiation, I). Let $x,y$ be rational numbers, and let $n,m$ be natural numbers.

1. We have $x^{n}x^{m}=x^{n+m}$, $(x^{n})^{m}=x^{nm}$, and $(xy)^{n}=x^{n}y^{n}$.
2. Suppose $n>0$, then we have $x^{n}=0$ if and only if $n=0$.
3. If $x\geq y \geq 0$, then $x^{n}\geq y^{n}\geq 0$, if $x>y\geq 0$ and $n>0$, then $x^{n}>y^{n}\geq 0$.
4. We have $\vert x^{n}\vert =\vert x\vert^{n}$.

**Defintion:**(Exponentiation to a natural number, II) Let $x$ be a non-zero rational number. Then for any negative integer $-n$, we define $x^{-n}:= 1/x^{n}$.

(Properties of exponentiation, I). Let $x,y$ be rational numbers, and let $n,m$ be natural numbers.

1. We have $x^{n}x^{m}=x^{n+m}$, $(x^{n})^{m}=x^{nm}$, and $(xy)^{n}=x^{n}y^{n}$.
2. If $x\geq y>0$, then $x^{n}\geq y^{n}>0$ if $n$ is positive, and $0<x^{n}<y^{n}$ if $n$ is negative.
3. If $x,y>0$, $n\neq 0$, and $x^{n}=y^{n}$, then $x=y$.
4. We have $\vert x^{n}\vert =\vert x\vert^{n}$.


