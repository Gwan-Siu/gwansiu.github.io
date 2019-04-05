---
layout:     post
title:      "Sampling Method"
date:       2019-04-05 12:00:00
author:     "GwanSiu"
catalog: true
tags:
    - Machine Learning
---

## 1. Introduction

In this article, I will discuss some sampling methods: **Inverse Samling**, **Reject-Accept Sampling**, **Important Sampling**.

## 2. Inverse Sampling

The intiution of inverse sampling is that the inverse CDF of a distribution is assumed to be easily tractable, we can draw a sample from uniform distribution (the range of CDF is from 0 to 1.), and then transform uniform samples into samples from a different distribution based the inverse distribution. **The procedure of inverse sampling is illustrated as:**

<img src="http://www.howardrudd.net/wp-content/uploads/2015/02/InverseTransform23.png" width = "600" height = "400"/>

The algorithm of inverse sampling is:

1. get a uniform sample $\mu$ from U (0,1)
2. obtain the sample $x$ through $x=F^{-1}(\mu)$ where F is the CDF distribution we desire.
3. repeat.

**Why does inverse sampling work?**

To be note that:

$$
\begin{equation}
            F^{-1}(\mu)= \text{ smallest }x \text{ such that } F(x)\geq \mu
\end{equation}
$$

What's the ditribution does the random variable $y=F^{-1}(\mu)$ follow?

The CDF of y is $p(y\leq x)$. Since the CDF is monotonic, we can obtain without loss of generity:

$$
\begin{equation}
    p(y\leq x)=p(F(y)\leq F(x))=p(\mu\leq F(x))=F(x)
 \end{equation}
$$

Thus we get the CDF and hence the pdf that we want to sample from

**Limitation of inverse sampling**

Not all the pdf has analytical CDF, for example, gaussian ditribution. In addition, for some complex distributions, the inverse CDF may be complicated and it is hard to do inverse sampling.

## 2. Rejection Sampling

### 2.1 Basic Rejection Sampling

The basic idea is come up with von Neumann. If you have a function you are trying to sample from, whose functional form is well known, basically accept the sample by generating a uniform random number at any $x$ and accepting it if the value is below the value of the function at that $x$. The procedure of basic sampling is illustrated as:

<img src="https://am207.github.io/2017/wiki/images/Rejection.png" width = "600" height = "400"/>


The procedure of basic rejection sampling:

1. Draw $x$ uniformly from $[x_{min}, x_{max}]$.
2. Draw $y$ uniformly from $[0, y_{max}]$.
3. If $y&lt;f(x)$, accept. Else, reject.
4. Repeat.

The intuitive explaination: This works as more samples will be accepted in the regions of $x$-space where the function $f$ is higher: indeed they will be accepted in the ratio of the height of the function at any given $x$ to $y_{max}$. **From the perspective of probability interpretation**,the accept-to-total ratio reflects the probability mass in each x silver.

```python

## target function

f = lambda x: np.exp(-x)
#f = lambda x x**2

## domain limits
xmin = 0 # lower bound of feasiable domain
xmax = 10 #upper bound of feasiable domain

## range limit for y
ymax=1
#ymax = 100

N = 10000 #the total of samples
accept =0 #count the total number of acceptance
samples = np.zeros(N)
count = 0

while (accept <N):
    
    x = np.random.uniform(xmin,xmax)
    y = np.random.uniform(0, ymax)
    
    if y<f(x):
        samples[accept]=x
        accept += 1
    count +=1

print('Count: ', count, ', Accepted: ', accept)

hinfo = np.histogram(samples, 30)
plt.hist(samples, bins=30, label='Smaples')
print(hinfo[0][0])
xvals = np.linspace(xmin, xmax, 10000)
plt.plot(xvals, hinfo[0][0]*f(xvals), 'r', label='f(x)')
plt.grid('on')
plt.legend()

```

### 2.2 Modified Reject Sampling

**low acceptasnce rate is the dterminstic limition of basic rejection sampling.** For basic rejection sampling, we need know the supremum of function we want to sample from. In practice, it is hard to evaluate the function on interest domain and find a tight bound for this function. Furthermore, even if you find a tight bound for this function, you may abserve that the accept rate is very low, epescially in low density regions.

In order to overcome low acceptance rate while preserving the simplicity of rejection sampling, proposal density $g(x)$ is introduced. This proposal density $g(x)$ is commom density function $f^{'}(x)$ with a scaling factor $M$. (Why? Because it is impossible to find a density function always above another density function. The integration of density function is always equal to 1. $\int p(x) dx=1$) The reason we introduce proposal density function is to increase acceptance rate, but this is not the only way to do so.

The proposal density must has the following charicteristics:

1. $g(x)$ is easy to sample and calculate pdf
2. the range of M is $[0, \infty]$ so that $Mg(x)&gt;f(x)$ in your entire interest domain.
3. ideally g(x) will be somewhat close to $f(x)$ so that you will sample more in high density regions and much less in low density region.


Obviously, the optimal value of $M$ is the supremum over your domain of interest of $\frac{f}{g}$. At that position x, the acceptance rate is 1. Ideally, the value of $M$ should be as close as to 1, since the acceptance rate is $\frac{1}{M}$. In other words, proposl density function $g(x)$ should be as indentical as $f(x)$.

The proportion of samples from $g(x)$ that are accept at each position $x$ and then average over $x$:

$$
\begin{equation}
\int dxg(x)\text{prop}(x)=\int dxg(x)\frac{f(x)}{Mg(x)}=\frac{1}{M}\int dxf(x)=\frac{1}{M}
\end{equation}
$$

The procedure of modified rejection sampling is:

1. Draw $x$ from your proposal distribution $g(x)$.
2. Draw $y$ uniformly from $[0, 1]$.
3. If $y&lt;\frac{f(x)}{Mg(x)}$, accept the sample, otherwhile, reject the sample.
4. Repeat the procedure

<img src="https://am207.github.io/2017/wiki/images/rejsteroid.png" width = "600" height = "400"/>

```python

p = lambda x: np.exp(-x)  # our distribution
g = lambda x: 1/(x+1)  # our proposal pdf (we're thus choosing M to be 1)
invCDFg = lambda x: np.log(x +1) # generates our proposal using inverse sampling

# domain limits
xmin = 0 # the lower limit of our domain
xmax = 10 # the upper limit of our domain

# range limits for inverse sampling
umin = invCDFg(xmin)
umax = invCDFg(xmax)

N = 10000 # the total of samples we wish to generate
accepted = 0 # the number of accepted samples
samples = np.zeros(N)
count = 0 # the total count of proposals

# generation loop
while (accepted < N):
    
    # Sample from g using inverse sampling
    u = np.random.uniform(umin, umax)
    xproposal = np.exp(u) - 1
    
    # pick a uniform number on [0, 1)
    y = np.random.uniform(0,1)
    
    # Do the accept/reject comparison
    if y < p(xproposal)/g(xproposal):
        samples[accepted] = xproposal
        accepted += 1
    
    count +=1
    
print("Count", count, "Accepted", accepted)

# get the histogram info
hinfo = np.histogram(samples,50)

# plot the histogram
plt.hist(samples,bins=50, label=u'Samples');

# plot our (normalized) function
xvals=np.linspace(xmin, xmax, 1000)
plt.plot(xvals, hinfo[0][0]*p(xvals), 'r', label=u'p(x)')
plt.plot(xvals, hinfo[0][0]*g(xvals), 'k', label=u'g(x)')



# turn on the legend
plt.legend()
```

## 3. Important Sampling

**Different from reject sampling, import sampling does not has any rejection action. To replace with rejection action, important sampling adopt the approach of weighted sample.** In detail, we want to draw sample from $h(x)$, where a function whose integral or expectation we desire, is large. In the case of expecation, it would indeed be even better to draw more samples where $h(x)f(x)$ is large, where $f(x)$ is the pdf we are calculating the integral with respect to. I will show how importan sampling work below:

**Why improtant?** Often, in the computation of an expectation or other integral, the integrand has a very small value on a dominant fraction of the whole integration volume. If the points are chosen evenly in the integration volume, the small minority of the points close to the ‘peak’ give the dominant contribution to the integral.

**(Example: Expectation)**

$$
\begin{equation}
\mathbb{E}_{f}[h]=\int_{V}f(x)h(x)dx
\end{equation}
$$

Choose a distribution $g(x)$, which is close to the function $f(x)$, but which is simple enough so that it is possible to generate random $x$-values from this distribution. The integral can now be re-written as:

$$
\begin{equation}
\mathbb{E}_{f}[h]=\int h(x)g(x)\frac{f(x)}{g(x)} dx
\end{equation}
$$

Therefore if we choose random number $x_{i}$ from distribution $g(x)$, we obtain:

$$
\begin{equation}
\mathbb{E}_{f}[h(x)]=\lim_{N\rightarrow \infty}\frac{1}{N}\sum_{x_{i}\sim g(\cdot)}h(x_{i})\frac{f(x_{i})}{g(x_{i})}
\end{equation}
$$

Let $w(x_{i})=\frac{f(x_{i})}{g(x_{i})}$, the formulation can be rewritten:

$$
\begin{equation}
\mathbb{E}_{f}[h(x)]=\lim_{N\rightarrow \infty}\frac{1}{N}\sum_{x_{i}\sim g(\cdot)}h(x_{i})\omega(x_{i})
\end{equation}
$$

Now the variance(error) of monte carol is that:

$$
\begin{equation}
\widetilde{V}=\frac{V_{f}[h(x)]}{N}
\end{equation}
$$

where $N$ is the sample size.

With the important sampling this formula has now changed to

$$
\begin{equation}
\widetilde{V}=\frac{V_{g}[\omega(x)h(x)]}{N}
\end{equation}
$$

Our goal is to minimize the $V_{g}[\omega(x)h(x)]$.

As a somewhat absurd notion, this variance should be set to zero, if

$$
\begin{equation}
\omega(x)h(x)=C\Rightarrow f(x)h(x)=Cg(x)
\end{equation}
$$

which leads to (since $g(x)$ is density thus we need normalization):

$$
\begin{equation}
g(x) = \frac{f(x)h(x)}{\int f(x)h(x)dx}=\frac{f(x)h(x)}{\mathbb{E}_{f}[h(x)]}
\end{equation}
$$

Actually, the expection is what we expect to estimate. Let's ignore the denominator, this formula tell us that to achieve low variance, we must have $g(x)$ large where the product $f(x)h(x)$ is large. Didirectly, maximizing the latter in some fashion was our original intuition.

Or from another perspective, $\frac{f(x)}{g(x)}$ ought to be large where $h(x)$ is large. This means that, as we say earlier, choose more samples near the peak.

In detail, We have a $f$ that we might or might not know. We have a pdf $g$ which we choose to be higher than $f$ at the points where hh has peaks. Now what we are left to do is to sample from $g$, and this will give us an oversampling at the place hh has peaks, and thus we must correct this there by multiplying by weights $w=\frac{f}{g}&lt;1$ in thse places.

Be careful to choose $g(x)$ appropriately, it should have thicker tails than $f$, or the ratio $\frac{f}{g} will be too big and count contribute too much in the tails. All of these considerations may be seen in the diagram below:

<img src="https://am207.github.io/2017/wiki/images/importance.png" width = "600" height = "400"/>

Another way of seeing this whole thing is that we will draw the sample from a proposal distribution and re-weight the integral appropriately so that the expectation with respect to the correct distribution is used. And since $\frac{f}{g}$ is flatter than $f$, the variance of $h\times\frac{f}{g}$ is smaller that the variance of $h\times f$ and therefore the error will be smaller for all N.


```python
from scipy import stats
from scipy.stats import norm

mu = 2;
sig = .7;

f = lambda x: np.sin(x)*x
infun = lambda x: np.sin(x)-x*np.cos(x)
p = lambda x: (1/np.sqrt(2*np.pi*sig**2))*np.exp(-(x-mu)**2/(2.0*sig**2))
normfun = lambda x: norm.cdf(x-mu, scale=sig)

# Range of integration 
xmax = np.pi
xmin = 0

# Number of draws
N = 1000

#Just Want to plot the function
x = np.linspace(xmin,xmax,1000)
plt.figure(figsize=(18,8))
plt.subplot(1,2,1)
plt.plot(x, f(x), 'b', label='Original $x\sin(x)$')
plt.plot(x, p(x), 'r', label='Important Sampling Function:')
plt.plot(x, np.ones(1000)/np.pi, 'k')
xis = mu + sig*np.random.randn(N,1)
plt.plot(xis, 1/(np.pi*p(xis)), '.', alpha=0.1)
plt.xlim([0, np.pi])
plt.ylim([0, 2])
plt.xlabel('x')
plt.legend()

## VANILLA MONTE CAROL
Ivmc = np.zeros(1000)
for k in np.arange(0, 1000):
    x = np.random.uniform(low=xmin, high=xmax, size=N)
    Ivmc[k] = (xmax-xmin)*np.mean(f(x))
    
print('Mean basic MC estimate:', np.mean(Ivmc))
print('Standard deviation of our estimates:', np.std(Ivmc))

## IMPORTANCE SAMPLING, choose gaussian so it is 
## similar to the original functions 

Iis = np.zeros(1000)
for k in np.arange(0, 1000):
    xis = mu + sig * np.random.randn(N,1)
    xis = xis[(xis<xmax) & (xis>xmin)]
    
    # normalization for gaussian from 0 to pi
    normal = normfun(np.pi)-normfun(0)
    Iis[k] = np.mean(f(xis)/p(xis))*normal

print('Mean important sampling MC estimate:', np.mean(Iis))
print('Standard deviation of our estimates:', np.std(Iis))

plt.subplot(1,2,2)
plt.hist(Iis, 30, histtype='step', label='Importance Sampling')
plt.hist(Ivmc, 30, color='r', histtype='step', label='Vanilla')
plt.grid('on')
plt.legend()
```