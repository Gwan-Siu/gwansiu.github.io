---
layout:     post
title:      "深度卷积神经网络的发展:从AlexNet到DenseNet"
date:       2017-08-06 12:00:00
author:     "GwanSiu"
catalog: true
tags:
    - Deep Learning
---

## 1. Introduction  
众所周知，深度卷积神经网络(简称:CNN)应用广泛且效果比一般传统方法要好，在图像任务级别包括图像分类和图像检测，在像素任务级别上包括图像分割，图像去噪和图像超分辨率等。在我刚刚入门的时候，常常会会对各种CNN的变种结构，不知道他们之间的关系是如何，只知道这个网络是最新最好的，然后拿来使用测试。但作为一个学习者，本着探究的精神，我认为”拿来主义“并不可取，知其然而不知其所以然的后果是当自己真正设计网络结构时，通常自己心里常常会冒出几个问题:1.要设计多少层的神经网络？2.神经网络应该要多宽？3.kernel的大小多少合适？4.kernel的步长应该是多少？

在这篇博文中，博主将分析几个重要的CNN结构，包括12年的AlexNet，VGG，googleNet家族(inception v1-v4, xception), Resnet和identity mapping以及17年CVPR的最佳论文DenseNet。通过对这些网络结构的了解和分析，会穿插介绍CNN网络训练的难点和要解决的基本问题，回答CNN网络的四大玄学问题:depth, width, kernel size, kernel stride。下面引用中科院计算所刘昕老师的CNN演化结构关系的[PPT](http://mclab.eic.hust.edu.cn/valse2016/dl/%E5%88%98%E6%98%95.pdf)
![---][1]

## 2. AlexNet
### 2.1 AlexNet的基本结构
![image.png-547.4kB][6]

**深度(depth):**是指除去第一层输入层的层数
**宽度(width):**是指神经元个数=滤波器的个数=feature map的层数

AlexNets的深度为8:5层卷积层，3层全连接层(fc层，维度分别为:2048，2048，1000)，采用Relu激活函数，其中卷积层的结构为:*conv-relu-max pooling*。  kernel的大小为:11x11,5x5,3x3,3x3,3x3,网络的宽度为:48，128，192，192，128.  
AlexNet用两路GPU进行训练:Conv1,Conv2,Conv4,Conv5在同一个GPU进行训练，Conv3,Fc6,Fc7,Fc8之间需要两路GPU相互信息进行训练。
过拟合:dropout + 数据增广

### 2.2 为什么神经网络使用的条件？
神经网络是90年代就被来的算法，其中LeNet在1989年被提出来解决手写字体识别问题。但这个90年代的作品曾由于自身的缺点(训练困难)以及时代的特点(不具有大量可训练数据)一度曾走出机器学习界的主流，直到12年Alex Krizhevsky在ImageNet比赛上用深度神经网络获胜，便掀起了深度学习的热潮。深度神经网络之所以到12年才火起来，外部原因是异构计算的兴起(cpu与gpu之间相互辅助)，数据容易获取以及数据量的爆炸式增长，内部原因是神经网络的深度问题逐渐被解决(深度神经网络的稳定性训练问题)。

**条件一:神经网络的训练需要大量的数据**:Andrew Ng在16年的NIPS的报告上曾指出(如下图:)，当数据量很小的时候，深度神经网络与其他机器学习算法表现无异，不能体现神经网络具有强大拟合能力。这一点可以从机器学习基本理论--VC理论中解释:

$$E_{out}(g)\le E_{in}(g) + \sqrt{\frac{8}{N}\text{In}(\frac{4(2N)^{d_{vc}}}{\delta})}$$

其中，$E_{out}$是测试集的error,$E_{in}$是训练集的error, 而$\Omega(N,H,\delta)$是error gap,

$$\Omega(N,H,\delta)=\sqrt{\frac{8}{N}\text{In}(\frac{4(2N)^{d_{vc}}}{\delta})}$$

深度神经网络具有较高的VC维度，从上述公式可知:当$d_{vc}$增加时，N需要指数型增长才能保证模型不会过拟合，若没有大量的数据支持，神经网络无法发挥优势。浅层神经网络VC维度小，拟合能力低，与其他浅层网络相比，并不具有很大的优势。

**条件二:神经网络需要深:** Alex在12年比赛的时候用的是深度神经网络而非浅层网络结构，为什么神经网络要深会达到如此好的效果呢？神经网络的数学原理是weierstrass approximate theory,在deep learning book中指出，一层隐藏层便可逼近任意函数。既然一层隐藏层能够逼近任意函数，那为什么还需要深度神经网络呢？

1. **逼近能力的大小:**这个可以通过Fourier series和wavelet series来类比，Fouerier analysis理论中，任意函数都可以用三角级数来逼近，但是面对斜率非常大，即非常陡峭的函数时，需要大量的三角函数来进行拟合，而wavelet中哈尔小波，便可以很轻易对这种陡峭函数进行拟合，这便是拟合能力的体现。通俗细节见[2](https://www.zhihu.com/question/22864189)，本篇博客不展开叙述。而回到神经网络中，深度神经网络中，每一层卷积层通常会有非线性变换，这种非线性变换便起到了增加网络非线性拟合能力的作用。这里提一点，在早期，3层神经网络算是深度神经网络。
2. **减少计算量:**假设输入是256x256x3的图片，输出前后的图像大小不变，现在有两种情况:第一种是只有一层隐藏层，滤波器的大小是256x256,神经元个数是10。第二种情况是有两层隐藏层，滤波器大小是3x3，具有weight shared property，每层神经元个数是10。第一种情况计算量是256x256x3x10=1966080，而第二种情况的计算量为256x3x3x3x10=69120。相比之下，使用weight shared property的深度卷积网络足足少了28倍的计算量。
![---][3]

### 2.3 梯度消失和梯度爆炸(难点)
在这个问题上，我引用中科院计算所刘昕老师[ppt上的内容](http://mclab.eic.hust.edu.cn/valse2016/dl/%E5%88%98%E6%98%95.pdf)
#### 2.3.1 什么是梯度消失？
![---][4]

#### 2.3.2 梯度消失的解决方式？
1. **激活函数:**Relu,leaky Relu, PReLu,Selu函数的使用，克服sigmoid函数和tanh函数中saturation区域内梯度消失的缺点，其中SeLu函数具有自归一化的特性,具体可参考论文[3]。
2. **Batch Normalization:** 在训练的时候，数据逐层保持统计特性的不变。
3. **辅助损失函数:** Google Net在训练深度网络时使用两个损失函数，使得浅层神经元依旧可以接受梯度信息从而避免梯度消失。
4. **layer-wise Pre-train:**2006年hinton提出的训练方法，先用无监督数据分层训练，再用有监督数据fine-tune。VGG是用类似的方法训练的，但这种方法已经不提倡使用了。
5. **LSTM:**通过构建门机制(记忆和遗忘模块)避免RNN的梯度消失问题，从而可以对较长的时间序列进行处理。

#### 2.3.3 什么是梯度爆炸？
神经网络是一个级联系统，从系统学的角度出发，微小的扰动通过多级级联系统会被放大，从而导致输出值爆炸增长。在电路系统中通常采用反馈的方式控制，而在神经网络中则使用Xavier方差不变准则保持网络节点尺度不变。
![image.png-346.7kB][5]

#### 2.3.4 梯度爆炸的解决方式？
![image.png-200.2kB][6]


### 2.4 AlexNet小结
AlexNet开启了深度学习领域的热潮，并在文章[2]中指出，网络的深度和宽度是网络容量的关键。显然，AlexNet大量参数对函数的拟合能力是有冗余的，此后，人们便围绕着网络稳定训练问题(梯度消失和梯度爆炸)对深度神经网络的深度，宽度，kernel的大小，kernel的步长展开了各种CNN的变式。

## 3 ZFNet与VGGNet
### 3.1 ZFNet的基本结构[1,4]  
![image.png-587.9kB][7]
13年，ZFNet[1]通过提出deconvlotion的方式可视化AlexNet的filer response,以及对提取的feature进行分析，认为AlexNet成功的原因有三点: 1.大量可以进行训练的数据。2.GPU的高效运算资源。3.采用dropout防止模型过拟合。ZF-Net研究得出，采用较小的卷积核，如:7x7,filter response会更具有discriminative,且具有较少的“dead” features。因此, ZFNet对于AlexNet的改进有两点:  
1. Conv1:采用7x7的卷积核，步长为2(AlexNet采用11x11的卷积核，步长为4)。
2. 将网络整体变宽:Conv3,4,5的feature map个数变成:512,1024,512,而不是384，384，256.
![image.png-1548.5kB][8]

### 3.2 VGGNet的基本结构[2,4]
![image.png-302.1kB][9]
由于ZFNet的成功，这使得VGGNet在kernel size和depth上走上极致:1.采用更小的卷积核(3x3,1x1) 2.堆叠卷积层把网络造深。

**为什么使用小卷积核？**  
**减少计算量，增强网络的非线性近似能力。**解释: VGGNet使用kernel factorization技术将1个7x7的卷积核等效成3个3x3的卷积核，**从感知域(receptive field)的角度理解**，3个3x3卷积核的receptive field等效于1个7x7卷积核的receptive field。**从计算量的角度上发出**，在receptive field等效的情况下，小卷积核的计算量远远小于大的卷积核。举个例子，假设一个三层的3x3xC的卷积核，C为通道数，参数个数为$3(3^{2}C^{2})=27C^{2}$,而一个7x7xC的卷积核，参数个数为$7^{2}C^{2}=49C^{2}$，大卷积核的计算量近似于小卷积核的两倍。**从增强非线性能力上理解**，三个卷积层叠加意味着使用了三个非线性激活函数,这比一个大卷积核仅使用一层的非线性激活函数的非线性能力要强。永远要记得一句话:**no free lunch**，使用了小的卷积核堆叠使网络加深，这会使得在BP的过程梯度消失问题突出，训练困难。从VGG的论文中可见，VGG使用的是layer-wise train的方式，意思是先训练浅层，再慢慢逐渐加深，一段一段训练。

**VGG网络的参数分析**  
![image.png-660.9kB][10]

从VGG网络的参数分析:  
1. 大部分的显存消耗在conv1, conv2层；
2. 大部分的参数都集中在fc6，fc7,fc8层.  

大量参数使得使网络的容量巨大，但这并不代表每一个参数都对最后的结果有贡献。一方面，在后来的研究中，神经网络存在着很大的参数冗余性问题，可以一步压缩参数个数而不降低效果。另一方面，大量参数使得网络容易过拟合，conv层通常是作为特征提取层，而fc层是非线性映射层。从直觉上理解，conv层参数多样化可以使得提取的特征多样化，fc层具有降维和空间变换的能力，但大量参数容易造成过拟合会显得得不偿失。在Network in Networks的论文中便使用了global average pooling替代fc层，由于大量参数的减少，使得网络效果进一步提升。另外，在这篇论文还值得一提的是，1x1卷积核的提出可以实现：1.降维变换在卷积过程中跨通道压缩信息，从而减少计算量。2. 1x1卷积的堆叠，可以在不影响receptive field的情况下进一步提高网络的非线性特性(因为激活函数往往接在卷积层后面使用)。3.分离通道信息，在这xception Net中使用。

## 3.3 VGGNet小结
VGGNet使用kernel factorization的方法将大的卷积核分解成小卷积核，并进一步加深网络，使网络达到了更好的效果。其中，kernel factorization的方法为后面GoogleNet的诞生提供了很好的降低参数的思路，但仅仅加深网络而并没有解决梯度消失的问题，使得VGGNet训练起来非常困难。后面，GoogleNet,ResNet便为我们提供了很好的解决思路:如何让深度神经网络稳定性训练?

## 4. GoogleNet
### 4.1 Inception v1
#### 4.1.1 Inception Module
在论文[1]中指出，最直接提高神经网络性能的方法是增加网络的深度和宽度，但随着网络深度和宽度的增加，显存等硬件开销是非常巨大的，那么是否存在一种方法既能增加网络深度和宽度又不至于让参数增加太多。答案是: Inception Module. 使用Inception Module的模块的GoogleNet不仅比Alex深，而且参数比AlexNet足足减少了12倍。下图便是Inception Module模块:
![image.png-185.2kB][11]
GoogleNet作者的初始想法是用多个不同类型的卷积核代替一个3x3的小卷积核(如左图)，这样做的好处是可以使提取出来的特征具有多样化，并且特征之间的co-relationship不会很大，最后用concatenate的方法把feature map连接起来使网络做得很宽，然后堆叠Inception Module将网络变深。但仅仅简单这么做会使一层的计算量爆炸式增长，如下图：
![image.png-307.2kB][12]
解决办法就是在插入1x1卷积核作为bottleneck layer进行降维，以上式3x3 conv为例:$28x28x192x1x1x256$, 计算量主要来自于输入前的feature map的维度256，和1x1卷积核的输出维度:192。那可否先使用1x1卷积核将输入图片的feature map维度先降低，进行信息压缩，在使用3x3卷积核进行特征提取运算？答案是:可行的。于是便有了Inception with dimensioin reduction的版本，也就是Inception v1的版本。下图便是Inception v1的计算量:
![image.png-338.4kB][13]

#### 4.1.2 GoogLeNet Architecture
![image.png-491.5kB][14]

**Stem部分:**论文指出Inception module要在在网络中间使用的效果比较好，因此网络前半部分依旧使用传统的卷积层代替。
**辅助函数(Axuiliary Function):**从信息流动的角度看梯度消失的因为是梯度信息在BP过程中能量衰减，无法到达浅层区域，因此在中间开个口子，加个辅助损失函数直接为浅层网络提供梯度信息。
**Classifier部分:**从VGGNet以及NIN的论文中可知，fc层具有大量层数，因此用average pooling替代fc,减少参数数量防止过拟合。在softmax前的fc之间加入dropout，p=0.7,进一步防止过拟合。

### 4.2 Inception v2
Inception v2的论文,亮点不在于网络结构的变化中，提出"Batch Normalization"思想:消除因conv而产生internal covariant shift, 保持数据在训练过程中的统计特性的一致性。Inception v2的结构是在卷积层与激活函数之间插入BN层:conv-bn-relu.具体详情可见我写的[关于BN的博文](http://gwansiu.com/2017/06/06/BN/),以及[相关论文](https://arxiv.org/pdf/1502.03167.pdf). BN的提出进一步促进了神经网络稳定性的发展，BN可以缓解梯度消失和防止过拟合，减少dropout的使用。我之前浏览一些论坛的时候，就有人开玩笑说: Initialization is very important for neural network. If you find a good Initialization, please use BN.

### 4.3 Inception v3
[Inception v3](https://arxiv.org/abs/1512.00567)使用factorization的方法进一步对inception v2的参数数量进行优化和降低，并在文章中对于神经网络的设计提出了4个原则以及总结了之前的经验。该篇文章建议可以好好读一读，是inception系列写得非常好第一篇文章。

**通用的网络设计原则([本节结束出附上原文](#jump))**
1. **在网络的浅层部分要尽量避免representation bottlenecks，即: information representation(在神经网络中指的是feature map和fc层的vector)不能在某一层内进行极端压缩，否则会造成信息丢失严重的现象**例如: 400000维度的向量在fc层的映射下变成400维度。文章指出Feed-forward networks 从输入到输出是一个有向的非循环图，从输入到输出任意一个节点切开，都应该存在大量信息流通过。最好的降维办法是将信息缓慢降维，最终达到所需要的维度。从理论上讲，信息的内容并不能仅仅用信息的维度进行衡量，因为这并没有考虑到信息与信息之间的相关性结构。信息的维度仅仅是信息内容的粗略估计。
2. 高维信息更容易进行局部处理，增加网络的非线性激活函数可以使网络产生更多的disentangled features,从而使网络训练得更快。
3. 假设前提:如果输出被用来进行信息融合，具有强关联性的相邻单元在降维过程中会更少造成信息损失。这意味着空间信息融合可以在低维流形上进行而没有较多的信息损失。举个例子:在进行3x3卷积之前，可以对信号进行降维处理而不会产生过多的副作用，即:3x3卷积前先用1x1卷积进行降维，inception v1使用的方法。这在一定程度上也表明，给定一个具有较强压缩性的信号(其实就是冗余度很高，信号内的信息具有高dependence),降维有利于加速训练
4. 平衡网络的深度和宽度,最优的效果是平衡好每一层的feature map的个数和网络整体的深度。虽然深度和宽度都对网络的效果有贡献，但深度和宽度的提高会增加网络的整体计算量。

**Factorization**
**1. Factorization convolutions into smaller convolution.** 用小的卷积核替代大的卷积核，并使用1x1卷积核预处理进行空间降维。
**2. Spatial Factorization into Asymmetric Convolution.**使用receptive field等效的原理，进一步将nxn的卷积核裂化成1xn和nx1的卷积核串联形式(Figure 6)或者并联形式(Figure 7)，进一步减少计算量。如下图：
![image.png-51.7kB][15]
![image.png-94.7kB][16]

**Utility of Auxiliary Classifier** 为浅层网络提供梯度信息，避免梯度消失。
**Efficient Grid Size Reduction**  
这是为了避免representation bottleneck而设计的，pooling层通常是使用在激活函数之后，pooling是对信息进行降维处理。若pooling放在inception module之前便违反的了representation bottleneck的原则，但如果在按照inception-pooling的顺序则会浪费多余的计算量。**为什么？**因为pooling最终是丢弃一些信息，既然某些信息固定是要被丢弃的，那为什么还要花费计算资源去计算呢？因此，想法是把inception module和pooling layer结合起来，只要把inception module的输出的stride=2便同样可以达到降维的效果，而且还降低计算量。假设一个dxdxk的输入，使用2k个filters，输出便是$\frac{d}{2}\times\frac{d}{2}\times 2k$，inception-pooling结构的计算量是$2d^{2}k^{2}$,而将inception和pooling结合起来的计算量是$2(\frac{d}{2})^{2}k^{2}$.

**Model Regularization via Label Smoothing**

#### 4.3.1 the structure of inception v3
![image.png-115.8kB][17]

*<span id = "jump">Gengeral design principle(原文)</span>*  
1. Avoid representational bottlenecks, especially early in the network. Feed-forward networks can be repre- sented by an acyclic graph from the input layer(s) to the classifier or regressor. This defines a clear direction for the information flow. For any cut separating the in- puts from the outputs, one can access the amount of information passing though the cut. One should avoid bottlenecks with extreme compression. In general the representation size should gently decrease from the in- puts to the outputs before reaching the final represen- tation used for the task at hand. Theoretically, infor- mation content can not be assessed merely by the di- mensionality of the representation as it discards impor- tant factors like correlation structure; the dimensionality merely provides a rough estimate of information content.
2. Higher dimensional representations are easier to pro- cess locally within a network. Increasing the activa- tions per tile in a convolutional network allows for more disentangled features. The resulting networks will train faster.
3. Spatial aggregation can be done over lower dimen- sional embeddings without much or any loss in rep- resentational power. For example, before performing a more spread out (e.g. 3 × 3) convolution, one can re- duce the dimension of the input representation before the spatial aggregation without expecting serious ad- verse effects. We hypothesize that the reason for that is the strong correlation between adjacent unit results in much less loss of information during dimension re- duction, if the outputs are used in a spatial aggrega- tion context. Given that these signals should be easily compressible, the dimension reduction even promotes faster learning.
4. Balance the width and depth of the network. Optimal performance of the network can be reached by balanc- ing the number of filters per stage and the depth of the network. Increasing both the width and the depth of the network can contribute to higher quality net- works. However, the optimal improvement for a con- stant amount of computation can be reached if both are increased in parallel. The computational budget should therefore be distributed in a balanced way between the depth and width of the network.

### 4.4 Inception v4
Inception v4是将Inception module和residual module结合起来。原因很直觉，双通路结构的residual module使梯度在BP过程中更顺畅的到达网络节点。文章通过对比了两种inception-res modules，发现加入residual module后的inception收敛得更快，但是最终的error rate并没有显著提高。可见，residual module可以加速训练。结构如下图:
![image.png-132.6kB][18]
![image.png-165kB][19]
![image.png-170.1kB][20]

### 4.5 Xception
文章指出，单个卷积核是同时映射cross-channel correlation信息和spatial correlation信息。因此，便很自然想到，基于cross-channel correlation information和spatial correlation information可以完全分离的假设，设想可否将cross-channel correlation信息和spatial correlation分开，这便Xception所做的事情。Xception使用1x1卷积核现将cross-channel correlation informatio分开，然后再使用3x3的卷积核对每一部分的channel information进行spatial information的提取，结构如下图。文章进行了两组实验，一组是将Xception与inception v3进行对比，一组是将Xception 与Xception with residual module进行对比，发现，Xception with residual module的效果更好。
![image.png-38.8kB][21]

### 4.6 Summary of Inception Module
为了同时增加网络深度和宽度而又同时需要控制参数，inception v1被提出来;之后通过conv layer与conv layer之间的特性研究，提出BN的方法，形成Inception v2;在Inception v3中，把factorization的方法用到极致，连pooling layer都不曾放过，进一步减少参数总量降低网络的冗余度，从而提高网络性能。搞完网络冗余度之后，便联想到将Inception v3和residual module结合起来看看会怎么样，便有了inception v4。最后，将1x1卷积极致发挥，分离cross-channel correlation和spatial correlation,搞出Xception。一切看似非常自然，真想为Google的工程师们鼠标双击6666。

## 5. ResNet
何凯明大神在ResNet的文章指出神经网络的深度训练存在两个问题:  
1. 梯度消失/梯度爆炸
2. 当网络不断加深的时候，会存在network degradation的现象。**Network degradation是指网络仅仅靠堆叠conv layer加深网络会使得深度网络的效果不如浅层的网络，而且这样并不是因为过拟合所造成的，如下图。**Degradation问题是一个优化问题，深度模型不能轻易被优化。
作者并且认为:
> 1. The deeper model should be able to perform at
least as well as the shallower model.
> 2. A solution by construction is copying the learned
layers from the shallower model and setting
additional layers to identity mapping.

![image.png-64.9kB][22]

### 5.1 Residual Learning
Residual Learning是ResNet的重要思想,现在来解释一下为什么要进行Residual Learning? 我们知道无论是传统的机器学习还是深度学习都希望学到一个非线性映射$H(x)$，当数据经过$x$经过非线性映射$H$时，同类数据相互靠拢，不同类数据离的越远越好(minimize intra-classes variance adn maximize inter-classes variance). 在深度学习中，便假设多层非线性层堆叠便可无限逼近需要学习的复杂函数$H(x)$,这也等效成可以无限逼近残差函数:$F(x)=H(x)-x$。如果$H(x)$过于复杂，那么会让神经网络的学习到一步映射$H$很困难，那么就采取两步的学习方式:先学习残差函数$F(x)$,再通过简单的线性映射学到$H(x)=F(x)+x$。回想一下，degradation现象是说深层模型的优化很困难，意思是指复杂函数$H$的优化非常困难，现在通过两步学习，学习残差函数便会缓解深度模型degradation的问题，而且identity mapping使得浅层网络更容易接受梯度信息，进一步减少梯度消失的影响。
The formulation of identity mapping by shortcuts:

$$y_{l} = h(x_{l}) + F(x_{l},w_{l})\\
x_{l+1} = f(y_{l})$$

其中，$x_{l}$和$x_{l+1}$是$l$层的输入与输出，$F$是残差函数(residual function), $f$是两路信号合并后的非线性函数，在ResNet论文中$f$是ReLU函数。

![image.png-171.6kB][23]

### 5.2 The structure of ResNet

![image.png-364.9kB][24]

### 5.3 The study of Identity mapping
何凯明在论文[Identity mapping in deep residual network](https://arxiv.org/abs/1603.05027)对Identity mapping进行了深入的探讨。

The formulation of identity mapping by shortcuts:

$$y_{l} = h(x_{l}) + F(x_{l},w_{l})\\
x_{l+1} = f(y_{l})$$

其中，$x_{l}$和$x_{l+1}$是$l$层的输入与输出，$F$是残差函数(residual function), $f$是两路信号合并后的非线性函数，在ResNet论文中$f$是ReLU函数。

对shortcut connection $H$进行研究发现，clean road是最好的，也就是什么也不加，笔直大路让信息流过去。

![image.png-209.1kB][25]

对于残差函数逼近方式$F(x)$和激活函数$f$的研究，发现f是identity mapping:$x_{l}=y_{l}=x_{l}+F(x_{l},w_{l})$效果更好，F则为非对称结构(asymmetric)的pre-activation的方式:BN->ReLu->conv.

![image.png-143.5kB][26]

### 5.4 ResNet小结
ResNet针对梯度消失问题和Network degradation问题提出identity mapping的双通路方法。Shortcut connection的使用使得ResNet不必像GoogLeNet那样使用辅助函数帮助浅层网络接收梯度信息，Residual lerning的思想解决了深度模型优化难的问题，最终使得网络可以进一步加深直到上千层。并且shortcut connection的方案进一步启发了后续DenseNet的诞生。经过这几个网络的分析，网络一直在追求深度发展，深度是最终决定网络效果的好坏，因为网络一深，意味着非线性层的增加，网络对复杂函数的拟合能力便可以寄一步增加。最后发一张cs231n课堂网络深度发展的趋势图：
![image.png-324.7kB][27]
![image.png-571.6kB][28]

## 6.DenseNet

ResNet,HighwayNet,FractalNets都使用shortcut connection让深层的信息直接回传到浅层而取得了成功，Densenet便把shortcut connections的模式发挥到极致，每一层都互相连接，使得每一层的输入都有前面所有层的输出信息。这样做使得DenseNet可以有效的减缓梯度消失和degradation的行为，让每一层网络的输入特征多样化使得计算更加有效，shortcut connection的使用起到了深度监督式学习的效果。下图为DenseNet的连接方式：
![image.png-173.6kB][29]

### 6.1 the structure of DenseNet
![image.png-114.4kB][30]
DenseNet主要由输入端的卷积层，Dense Block, transition layer, global average pooling之后的classifier所构成。
1. Concatenation in Dense Block: 每一层输出都会和自己的输出进行简单的合并，传到下一层输入中。这样使得下一层的输入特征多样化，有效的提高计算并且帮助网络整合浅层网络特征学到discriminative feature。同时，同一 Dense bolck里的神经元相互连接达到feature reused的效果,这也就是为什么DenseNet不需要很宽也能达到很好的效果。(注意:深度和宽度都是网络的关键因素，网络的宽度必须在网络达到一定效果之后才能发挥作用）。另外，之所以不选择使用ResNet中加法合并，是因为加法是一种简单的特征融合行为，会造成信息的丢失或者紊乱。
2. Compression in transition layer: DenseNet有一个特点是:参数使用量远远少于ResNet,除了feature reused减少了网络宽度之外，就是在transition layer使用了1x1卷积进行了信息压缩，这是在GoogLeNet inception v3中使用的手法，使得模型可以更加compact。另外，参数量减少带来另外一个好处就是减少模型的复杂度防止过拟合现象发生。
3. Deeply supervision:shortcut connections形成了多通路模型，使得信息流从输入到输出畅通无阻，梯度信息也可以直接从loss function直接反馈回网络的各个节点，有一种大脑直接控制身体部位的行为。

### 6.2 DenseNet小结
ResNet,HighwayNet,FractalNets，DenseNet揭示了多通路模型的成功，clean shortcut connections可以feature resued,提高模型的学习效率(compact,discriminative features)有效的减缓梯度消失和network degradation的现象，并且一定程度上达到deeply supervision learning的效果，transition layer则进一步降维压缩模型参数，减少计算量。

## 7 Reference
1. http://cs231n.stanford.edu/slides/2017/cs231n_2017_lecture9.pdf
2. A. Krizhevsky, I. Sutskever, and G. E. Hinton, "Imagenet classification with deep convolutional neural networks," in Advances in neural information processing systems, 2012, pp. 1097-1105.  
3. G. Klambauer, T. Unterthiner, A. Mayr, and S. Hochreiter, "Self-Normalizing Neural Networks," ArXiv e-prints, vol. 1706, Accessed on: June 1, 2017Available: http://adsabs.harvard.edu/abs/2017arXiv170602515K  
4. X. Glorot and Y. Bengio, "Understanding the difficulty of training deep feedforward neural networks," in Proceedings of the Thirteenth International Conference on Artificial Intelligence and Statistics, 2010, pp. 249-256.  
5. M. D. Zeiler and R. Fergus, "Visualizing and Understanding Convolutional Networks," ArXiv e-prints, vol. 1311, Accessed on: November 1, 2013Available: http://adsabs.harvard.edu/abs/2013arXiv1311.2901Z
6. K. Simonyan and A. Zisserman, "Very deep convolutional networks for large-scale image recognition," arXiv preprint arXiv:1409.1556, 2014.
7. M. Lin, Q. Chen, and S. Yan, "Network In Network," ArXiv e-prints, vol. 1312, Accessed on: December 1, 2013Available: http://adsabs.harvard.edu/abs/2013arXiv1312.4400L 
8. C. Szegedy et al., "Going Deeper with Convolutions," ArXiv e-prints, vol. 1409, Accessed on: September 1, 2014Available: http://adsabs.harvard.edu/abs/2014arXiv1409.4842S  
9. S. Ioffe and C. Szegedy, "Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift," ArXiv e-prints, vol. 1502, Accessed on: February 1, 2015Available: http://adsabs.harvard.edu/abs/2015arXiv150203167I  
10. C. Szegedy, V. Vanhoucke, S. Ioffe, J. Shlens, and Z. Wojna, "Rethinking the Inception Architecture for Computer Vision," ArXiv e-prints, vol. 1512, Accessed on: December 1, 2015Available: http://adsabs.harvard.edu/abs/2015arXiv151200567S  
11. C. Szegedy, S. Ioffe, V. Vanhoucke, and A. A. Alemi, "Inception-v4, Inception-ResNet and the Impact of Residual Connections on Learning," in AAAI, 2017, pp. 4278-4284.  
12. F. Chollet, "Xception: Deep Learning with Depthwise Separable Convolutions," arXiv preprint arXiv:1610.02357, 2016.  
13. K. He, X. Zhang, S. Ren, and J. Sun, "Deep residual learning for image recognition," in Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition, 2016, pp. 770-778.  
14. K. He, X. Zhang, S. Ren, and J. Sun, "Identity mappings in deep residual networks," in European Conference on Computer Vision, 2016, pp. 630-645: Springer.
15. G. Huang, Z. Liu, K. Q. Weinberger, and L. van der Maaten, "Densely connected convolutional networks," arXiv preprint arXiv:1608.06993, 2016.

如果你喜欢这篇文章，可以给我一些鼓励或者在这篇文章下面点个赞,谢谢。
<img src="http://static.zybuluo.com/GwanSiu/908ff37t1pz2eclzhy99g47s/image.png" width="400" height="400"/>



[1]: http://static.zybuluo.com/GwanSiu/060jkut7852av647kzb9uta9/image.png
[2]: http://static.zybuluo.com/GwanSiu/wm3hipwf8s1aec2qflu8xxka/image.png
[3]: http://2.bp.blogspot.com/-20IX-B8hzi4/WL0mEr5UNVI/AAAAAAAABXA/SlaQshGBr2QTdzFw-3tzewOWo2jh5G2cgCK4B/s1600/andrewNg_1.PNG
[4]: http://static.zybuluo.com/GwanSiu/k3dubq01ljxd3ruhi6nzo3qg/image.png
[5]: http://static.zybuluo.com/GwanSiu/o4isv928kkq2mtfg4l2moxjm/image.png
[5]: http://static.zybuluo.com/GwanSiu/tv8pbj9tcyrjf5mh4vuxx91e/image.png
[6]: http://static.zybuluo.com/GwanSiu/bxrj8i3jb7baaq4q8zwgwd0f/image.png
[7]: http://static.zybuluo.com/GwanSiu/5kwrc4kbhwatczxown9wvt6r/image.png
[8]: http://static.zybuluo.com/GwanSiu/dgvwl7maa3qmm0mjvzcdopic/image.png
[9]: http://static.zybuluo.com/GwanSiu/oucizb4g0bw05w0af6pd95tx/image.png
[10]: http://static.zybuluo.com/GwanSiu/trhs00bu0e9eij5juny9lft4/image.png
[11]: http://static.zybuluo.com/GwanSiu/wxfua3higfo1tuu9ohmzew70/image.png
[12]: http://static.zybuluo.com/GwanSiu/05kbg8ad47ofqlujfnrxpyiu/image.png
[13]: http://static.zybuluo.com/GwanSiu/kjjyu34g7c42ba3kwzuprbm5/image.png
[14]: http://static.zybuluo.com/GwanSiu/eq2u6qwc41u6vu3qvv1mj3qn/image.png
[15]: http://static.zybuluo.com/GwanSiu/5f8zwb8175bxs43z6xwl0pft/image.png
[16]: http://static.zybuluo.com/GwanSiu/vs4p6016dvaxvqmc339hg8ed/image.png
[17]: http://static.zybuluo.com/GwanSiu/6dbbu8u21ojzj6qwt4z2okfw/image.png
[18]: http://static.zybuluo.com/GwanSiu/rafvofancodomovdm64hv8pf/image.png
[19]: http://static.zybuluo.com/GwanSiu/ok21u55nayddjqoxn1bi6v5h/image.png
[20]: http://static.zybuluo.com/GwanSiu/831z5dlrg06bn0hl3uunbguh/image.png
[21]: http://static.zybuluo.com/GwanSiu/j3kk20r9t5chkvzarjtpc74i/image.png
[22]: http://static.zybuluo.com/GwanSiu/k0l2xf048g0k5n3zup8puyu2/image.png
[23]: http://static.zybuluo.com/GwanSiu/xwrcozpm79tf1p1jt1srt68r/image.png
[24]: http://static.zybuluo.com/GwanSiu/p46ld6ba5zt4c898coc07skz/image.png
[25]: http://static.zybuluo.com/GwanSiu/8v1596g41u7xed388p19tbhz/image.png
[26]: http://static.zybuluo.com/GwanSiu/931h7994wbx39t7vyia57j50/image.png
[27]: http://static.zybuluo.com/GwanSiu/vwbgv9az2jrvuun4q0ghjxvg/image.png
[28]: http://static.zybuluo.com/GwanSiu/7y3ml4um7molmnswrc7t0p4a/image.png
[29]: http://static.zybuluo.com/GwanSiu/w283nbyrs0bej3klvdthh094/image.png
[30]: http://static.zybuluo.com/GwanSiu/kebiuiv8uacw78cc8wa2bfxv/image.png