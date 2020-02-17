---
layout:     post
title:      "Deconvolution"
date:       2020-02-13 12:00:00
author:     "GwanSiu"
catalog: true
tags:
    - Image Processing
---

## 1. Reasons to Cause Blurry Images

- lens imperfections
- Camera shake
- Scene motioin
- Depth defous

In this post, only blurs caused by lens imperfections and camera shake are discussed.

### 1.1 Lens imperfections

1. What are lens imperfections?

- Ideal lens: A point maps to a point at a certain plane.
- Real lens: A point maps to a circle that has non-zero minimum radius among all plans. **The blur kernel is shift-invariant**.

<center class="half">
    <img src="https://raw.githubusercontent.com/Gwan-Siu/BlogCode/master/Image%20Processing/Deconvolution/ideal_lens.png" width="400"/> <img src="https://raw.githubusercontent.com/Gwan-Siu/BlogCode/master/Image%20Processing/Deconvolution/real_lens.png" width="400"/>
</center>

2. What causes lens imperfections?

- Aberrations, e.g. Chromatic aberration, spherical aberration, and oblique aberrations. Note: oblique aberrations is not shift-invariant.

<img src="https://raw.githubusercontent.com/Gwan-Siu/BlogCode/master/Image%20Processing/Deconvolution/aberration.png" width = "600" height = "400"/>

- Diffraction, e.g. small aperture, and large aperture.

<img src="https://raw.githubusercontent.com/Gwan-Siu/BlogCode/master/Image%20Processing/Deconvolution/aperture.png" width = "600" height = "400"/>

3. We condiser lens as a low-pass filter, and we call it *point spread function* (PSF). The fourier transform of PSF is optical transform function (OTF), which is equatl to aperture shape.

<img src="https://raw.githubusercontent.com/Gwan-Siu/BlogCode/master/Image%20Processing/Deconvolution/OTP_PSF.png" width = "600" height = "400"/>

## 2. Non-blind Deconvolution

The image sensing model is defined as 

$$
\begin{equation}
y = x\ast b + n
\end{equation}
$$

where $y$ denotes an observed blurry image, $x$ is the latent image and $b$ represents the blur kernel. $\ast$ denotes the convolution operator.

### 2.1 Inverse Filter

First of all, we consider the smplest case that noise is free in the image sensing model, i.e. $y = x \ast b$. If the kernel information $b$ and the blurry image $bm{y}$ are given, how can we recover the latent image $x$? (Fourier transform)

$$
\begin{equation}
Y = X\cdot B
\end{equation}
$$

where $Y=\mathcal{F}(y), X=\mathcal{x}, B=\mathcal{b}$, and $\mathcal{F}(\cdot)$ reprents the Fourier transform. Therefore, we can directly obtain the latent image $x$ by

$$
\begin{equation}
x = \mathcal{F}^{-1}\lbrace \frac{Y}{b}\rbrace
\end{equation}
$$

**Limitation: the inverse filter is very sensitive to the noise in the image. In other words, if the observed blurry image contains noise, the inverse filter leads to a poor result.**

### 2.2. Winer Filter

In this case, we consider the noise in the image sensing model, i.e. $y = x\ast b + n$. The Fourier transform of the image sensing model is $Y=X\cdot B$. 

Problem statements: Find function $H(\omega)$ that minimizes expected error in Fourier domain.

$$
\begin{equation}
\min_{H}\, \mathbb{E}[\Arrowvert X-HY\Arrowvert^{2}]
\end{equation}
$$

Expand the squares:

$$
\begin{equation}
\min_{H} \Arrowvert 1 - HC\Arrowvert \mathbb{E}[\Arrowvert X\Arrowvert^{2}] - 2(1 - HC)\mathbb{E}[XN] + \Arrowvert H\Arrowvert^{2}\mathbb{E}[\Arrowvert N\Arrowvert^{2}]
\end{equation}
$$

> Assumption: 1. the latent image and noise are independent. It means that $\mathbb{E}[XN]=\mathbb{X}\mathbb{N}$
> 2. the expectation of noise is zero, $\mathbb{E}[N] = 0$.

Simplify:

$$
\begin{equation}
\min_{H}\, \Arrowvert 1 - HC\Arrowvert^{2} \mathbb{E}[\Arrowvert X\Arrowvert^{2}] + \Arrowvert H\Arrowvert^{2}\mathbb{E}[\Arrowvert N\Arrowvert^{2}]
\end{equation}
$$

To solve this problem, take it detivative with respect to $H$ and set it to zero. We obtain:

$$
\begin{equation}
\begin{split}
H &= \frac{C\mathbb{E}[\Arrowvert X\Arrowvert^{2}]}{C^{2}\mathbb{E}[\Arrowvert X\Arrowvert^{2}] + \mathbb{E}[\Arrowvert N\Arrowvert^{2}]} \\
&= \frac{C}{C+\frac{\mathbb{E}[\Arrowvert N\Arrowvert^{2}]}{\mathbb{E}[\Arrowvert X\Arrowvert^{2}]}}
\end{split}
\end{equation}
$$

Apply inverse kernel and do not divide by zero

$$
\begin{equation}
\bar{x} = \mathcal{F}^{-1\lbrace\frac{\vert F(c)\vert^{2}}{\vert F(c)\vert^{2} + 1/SNR(\omega)}\cdot \frac{\mathcal{F}(Y)}{\mathcal{C}} \rbrace}
\end{equation}
$$

where $1/SNR(\omega)=\frac{\sigma_{s}(\omega)}{\sigma_{n}(\omega)}=\frac{\mathbb{E}[\Arrowvert N\Arrowvert^{2}]}{\mathbb{E}[\Arrowvert X\Arrowvert^{2}]}$.

- derived as solution to maximum-likelihood problem under **gaussian noise assumption**
- requires estimate of **signal-to-noise ratio at each frequency.**

**Mark:**

- 1.**Can we undo lens blur by deconvolving a PNG or JPEG image without any preprocessing?**
	+ All the blur processes we discuss are optical degradation
	+ Blur model is accurate only if our images are linear

- 2. Are PNG or JPEG images linear?
	+ No, because of gamma encoding.
	+ Before deblurring, linearize images first.

<img src="https://raw.githubusercontent.com/Gwan-Siu/BlogCode/master/Image%20Processing/Deconvolution/linearity.png" width = "600" height = "400"/>

- 3. How do we linearize PNG or JPEG images?

	+ inverse gamma correction

### 2.3 Generalization Formulation

Such a deconvolution problem that they are ill-conditioned. The optical low-pass filter removes high-frequency information in the original image. Given a PSF or OTF of an optical system, it can be predicted which frequencies are lost. Such an ill-posed problem usually has infinitely solutions, and thus prior information is required, e.g. smoothness, sparseness, sparse gradients, non-local priors, and many others. Without any prior information that is additionally imposed on the estimated solution, it is generally impossible to recover "good" images, which sometimes contains noise and ringing artifacts.


A general formulation for the image reconstruction is 

$$
\begin{equation}
\min_{x}\,\frac{1}{2}\Arrowvert Cx-b\Arrowvert^{2} + \Gamma(x)
\end{equation}
$$

where $x\in\mathbb{R}^{N}$ is a vector of unknown image, $b\in\mathbb{R}^{M}$ is the vectorized measurements, e.g. observed blurry image, and $B\in \mathbb{R}^{N\times M}$ is convolution matrix. Note: the convolution matrix $B$ is a circulant Toeplitz matrix, it means that its eigenvaluess are the Fourier transform of $c$. $\Gamma(x)$ denotes the prior term.

Take an example of the gradient regularization. Generally, for the anisotropic case, the regularizer is modeled as $\Gamma(x)=\lambda\Arrowvert Dx\Arrowvert_{p}$, with $D=[D_{x}^{T}, D_{y}^{T}]^{T}$. $D\in\mathbb{R}^{2M\times N}$ represents the finite differences approximation of the horizontal and vertical image gradients:

$$
\begin{equation}
\begin{split}
D_{x}x =\text{vec}(d_{x}\ast x),&\quad d_{x} = 
left(\begin{array}{ccc}
0& 0& 0 \\
0& -1& 1 \\
0& 0& 0
\end{array}\right) 
\\

D_{y}x = \text{vec}(d_{y}\ast x),&\quad d_{y} = \left\begin{array}{ccc}
0 & 0& 0 \\
0 & -1& 0 \\
0 & 1& 0

\end{array}\right)
\end{split}
\end{equation}
$$

where the operator $vec(\cdot)$ vectorize a 2D image and $d_{x}$ and $d_{y}$ are the convolution kernels representing forward finite differences.

- $p=2$, it is $L_{2}$ gradient regularization (Tikhonov regularization).
- $p=1$, it is total variation norm.
- $p=0.8$, it means that gradients of an image are sparsely distributed.

### 2.4 Example

The formulation to the deconvolution problem with TV norm gradient regularization:

$$
\begin{equation}
\min_{x}\, \Arrowvert c\ast x-b\Arrowvert^{2} + \lambda \Arrowvert \nabla x\Arrowvert_{1}
\end{equation}
$$

we can rewrite the problem formulation as

$$
\begin{equation}
\begin{split}
\min_{x}&\, \underbrace{\Arrowvert c\ast x-b\Arrowvert^{2}}_{f(x)} +\underbrace{\lambda\Arrowvert z\Arrowvert_{1}}_{g(z)} \\
s.t.& Dx-z=0
\end{split}
\end{equation}
$$

the augmented Lagragian function is:

$$
\begin{equation}
L_{\rho}(x,z,y)=f(x)+g(z)+y^{T}(Dx-z)+\frac{\rho}{2}\Arrowvert Dx-z\Arrowvert^{2}_{2}
\end{equation}
$$

the iterative updates rules are:

$$
\begin{equation}
\begin{split}
x&\leftarrow prox_{f,\rho}(x) =\arg\min_{x} L_{\rho}(x,z,y)=\arg\min_{x} f(x) +\frac{\rho}{2}\Arrowvert Dx-v\Arrowvert^{2}_{2}, v=z-mu \\
z&\leftarrow prox_{g,\rho}(z) =\arg\min_{z} L_{\rho}(x,z,y)=\arg\min_{z} g(z) + \frac{\rho}{2}\Arrowvert v-z\Arrowvert_{2}^{2},v=Dx+\mu \\
u&\leftarrow \mu+Dx-z
\end{split}
\end{equation}
$$

where $\mu=(1/ \rho)y$.

#### Efficient implementation of $x$-update

For $x$-update, this is a quadratic program:

$$
\begin{equation}
prox_{f, \rho}=\arg\min_{x}\frac{1}{2}\Arrowvert Cx-b\Arrowvert_{2}^{2} + \frac{\rho}{2}\Arrowvert Dx-v\Arrowvert_{2}^{2}, v = z -\mu
\end{equation}
$$

we expand it, take it derivative with respect to $x$, and set it as zero. We can get the solution:

$$
\begin{equation}
\bar{x} = (C^{T}C+\rho D^{T}D)^{-1}(C^{T}b+\rho D^{T}v)
\end{equation}
$$ 
5)

or using a large-scale, iterative method such as gradient descent, conjugate gradient, or the simultaneous algebraic
reconstruction method (SART).

To invert the above equation analytically using inverse filtering, the convolution operation in the spatial domain is converted to the multiplication operation in the frequency domain. We can see that $Cx$ and $Dx$ can be expressed as the convolutions, i.e. $c\ast x$ and $d_{x/y}\ast x$. Therefore, we can quickly solve the problem in the frequency domain:

$$
\begin{equation}
\begin{split}
(C^{T}C + \rho D^{T}D)&\Leftrightarrow \mathcal{F}^{-1}(\mathcal(F)(c)\otimes \mathcal{F}(c)+ \mathcal{F}(d_{x})\otimes \mathcal{F}(d_{x}) + \mathcal{F}(d_{y})\otimes \mathcal{F}(d_{y})) \\
(C^{T}b+\rho D^{T}v)&\Leftrightarrow \mathcal{F}^{-1}(\mathcal{F}\otimes \mathcal{F} +\rho(\mathcal{F}(d_{x})\otimes\mathcal{F}(v_{1})+\mathcal{F}(d_{y})\otimes \mathcal{F}(v_{2})))
\end{split}
\end{equation}
$$


which gives rise to the inverse filtering proximal operator, which applies only operator and directly obtain the solution:

$$
\begin{equation}
prox_{f,\rho} =\mathcal{F}^{-}\lbrace \frac{\mathcal{F}^{-1}(\mathcal{F}\otimes \mathcal{F} +\rho(\mathcal{F}(d_{x})\otimes\mathcal{F}(v_{1})+\mathcal{F}(d_{y})\otimes \mathcal{F}(v_{2})))}{\mathcal{F}^{-1}(\mathcal(F)(c)\otimes \mathcal{F}(c)+ \mathcal{F}(d_{x})\otimes \mathcal{F}(d_{x}) + \mathcal{F}(d_{y})\otimes \mathcal{F}(d_{y}))} \rbrace
\end{equation}
$$

#### Efficient Implementation of $z$-update

For the $z$-update, this is a LASSO problem, and we can directly get an analytical solution. The the proximal operator is

$$
\begin{equation}
prox_{g,\rho}(v) = \mathcal{S}_{\lambda/\rho}=\arg\min_{z} \lambda\Arrowvert z\Arrowvert_{1} + \frac{\rho}{2}\Arrowvert v - z\Arrowvert_{2}^{2}
\end{equation}
$$
with $v=Dx+\mu$ and $\mathcal{S}_{\kappa}$ being the element-wise soft thresholding operator


$$
\begin{equation}
\mathcal{S}_{\kappa}=\begin{cases}
v-\kappa, v>\kappa \\
0, \vert v\vert \leq \kappa \\
v+\kappa, v<-\kappa
\end{cases}
\end{equation}
$$

that can be implemented very efficiently.

#### The Isotropic case

In isotropic case, the sum of $\ell_{2}$-norms is used to approximate to the horizontal and vertical image gradients as a regularizer. In this case, $z\in\mathbb{R}^{2\times M}$, so that $z=[D_{x}x\, D_{y}y]^{T}$, we can use the $\ell_{2,1}$-norm to write the isotropic version of the regularizer as 

$$
\begin{equation}
\Gamma(x) = \lambda\Arrowvert z\Arrowvert_{2,1}=\lambda\sum_{i=1}^{M}\left\Arrowvert \begin{array}
(D_{x}x)_{i} \\
(D_{y}y)_{i}
\end{array}
\right\Arrowvert_{2}
\end{equation}
$$

this is known as the group lasso problem.

The deconvolution problem with an isotropic TV prior is formulated in ADMM notation as

$$
\begin{equation}
\begin{split}
\min_{x}&\, \underbrace{\frac{1}{2}\Arrowvert Cx-b\Arrowvert_{2}^{2}}_{f(x)} + \underbrace{\lambda\sum_{i=1}^{M}\Arrowvert \begin{array}
z_{i} \\
z_{i+M}
\end{array}
Arrowvert_{2}}_{g(z)} \\
s.t.&\, Dx-z=0
\end{split}
\end{equation}
$$

where $z_{i}$ is the $i$-th element of $z$. For $1\leq i\leq M$, it is meant to represent the finite differences approximattion in horizontal direction, $(D_{x}x)_{i}$, and for $M+1\leq i\leq 2M$, the finite differences approximation in vertical direction, $(D_{y}x)_{i}$. Notice that if the $\ell_{2}$-norm in $g(z)$ with the $\ell_{1}$-norm, then we get $\sum_{i=1}^{M}\Arrowvert(z_{i}, z_{i+M})\Arrowvert_{1}$ which reduces to $\Arrowvert z\Arrowvert_{1}$ and we recover the anisotropic case.

The way to update $x$ and $\mu$ are the same as above, and the only change is the $z$-update, which is 

$$
\begin{equation}
z\leftarrow prox_{g,\rho}(v)=\arg\min_{z}\lambda\sum_{i=1}^{M}\Arrowvert\left\begin{array}{ccc}
z_{i} \\
z_{i+M}
\end{array}\right\Arrowvert
+\frac{\rho}{2}\left\Arrowvert v-z\right\Arrowvert_{2}^{2}\, v=Dx+\mu
\end{equation}
$$

The corresponding proximal operator of $g(z)$, the group lasso, is block soft thresholding. The z-update rule then becomes

$$
\begin{equation}
\left(\begin{array}
z_{i} \\
z_{i+M}
\end{array}\right)
\leftarrow \mathcal{S}_{\lambda/\rho}\left(\begin{array}
v_{i} \\
v_{i+M}
\end{array} \right)\, i\leq i\leq M
\end{equation}
$$
$\mathcal{S}_{\kappa}$ being the vector soft thresholding operator.


## 3. Blind Deconvolution

If we only know about the measurement, i.e. observed blurry images $y$, can we recover the unknown image $x$ and kernel $c$?

> Ref： Removing camera shake from a single photograph.

Due to the ill-posed nature of the deconvolution problem, prior information plays a significant role when only observed blurry image is given. 

- The image "looks like" a natural image
	+ Gradient in natural images follow a characteristic "heavy-tail" distribution.

<img src="https://raw.githubusercontent.com/Gwan-Siu/BlogCode/master/Image%20Processing/Deconvolution/heavy_tail_distribution.png" width = "200" height = "200"/>

- The kernel "look like" a motion PSF.
	+ Shake kernels are very spars, have continous contours, and are always positive.

<img src="https://raw.githubusercontent.com/Gwan-Siu/BlogCode/master/Image%20Processing/Deconvolution/kernel_prior.png" width = "200" height = "200"/>

### 3.1 Regularized Blind Deconvolution

<img src="https://raw.githubusercontent.com/Gwan-Siu/BlogCode/master/Image%20Processing/Deconvolution/deconvolution_reg_form.png" width = "600" height = "400"/>



## Reference

1. [Lecture 12, "Deconvolution", CMU 12-862, Fall 2018.](http://graphics.cs.cmu.edu/courses/15-463/lectures/lecture12.pdf)
2. [Lecture 6, "Image Deconvolution", Stanford EE367/CS4481: Computational imaging and display, Winter 2020](https://stanford.edu/class/ee367/reading/lecture6_notes.pdf)
3. Fergus, R., Singh, B., Hertzmann, A., Roweis, S.T. and Freeman, W.T., 2006. Removing camera shake from a single photograph. In ACM SIGGRAPH 2006 Papers (pp. 787-794).





