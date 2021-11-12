# AttMVS Attention-Aware

## Abstract

attention-enhanced matching confidence volume → robust photo-consistency

将提取出特征的pixel-wise matching confidence与局部场景的上下文信息相结合

attention-guided regularization将匹配置信度代价体变为概率体

## Introduction

【问题一】

对应像素使用的光度一致性度量是vector-valued，而卷积MVS大都是基于标量值度量的，例如ZNCC

用向量表示肯定有更丰富的信息，如何构建好的matching confidence volume MCV是很核心的问题

> MVSNet直接用纯光度一致性构建，但不同通道权重往往不一样
> 

✨我们综合光度一致性和局部场景的上下文信息构建MCV，importance of the matching information from different channels is adaptively adjusted

【问题二】

如何有效正则化MCV → later depth probability volume LPV

有了LPV之后就可以通过回归或者多分类得到深度图

✨我们通过attention-guided方法分层回归

通过一些方法改进了深度图GT的质量

## AttMVSNet

### Feature extractor

提取perceptual features，物理上反映的是多视点的photo-consistency

10层的encoder，输出维度[1/4H, 1/4W, 16]

BN和ReLU被换为 → Instance Normalization, LeakyReLU

### Attention-enhanced matching confidence

之前构建代价体都是pixel-wise的，没有考虑全局上下文信息

这篇文章photo-consistency + contextual cues → attention-enhanced matching confidence volume

1. 通过全局average pooling把每个特征体压缩到单通道$v_i$(individual channel，个人是这么理解的)
2. 然后计算$v_i$的方差，得到$w_v$
3. 通过squeeze-and-excitation计算attentional channel-weighed vector $w_v^*$，他是$w_v$通过两层线性变换和激活函数的输出（可以理解为这里通过了一个网络）
4. 最后把之前通过warp方法得到的特征体乘以$w_v^*$，就得到了attention-enhanced代价体

✨核心要做的事情是特征体16维每个维度的重要性不一样，通过attention的机制加权构建代价体

> 但具体构建的方法看不太懂，讲的就不太清晰
> 

### Attention-guided hierarchical regularization

首先把代价体经过多次CNN的下采样，得到多个层级的金字塔结构，从最小的开始通过一个网络模块后输出，不断向上传递

通过的这个网络结构分为三部分：pre-contextual understanding, ray attention module(RAM), post contextual understanding module

pre和post都是3D卷积，首先下采样但把通道数量升上去，最后再做逆操作

中间的RAM还是用的跟代价体构建一样的表示，其实就是通过图三中间那样一个不很复杂的网络

$$w_v^* = Sigmoid(f_2(ReLU(f_1(w_v, s_1)), s_2))$$

> attention核心做的事情就是充分利用上下两层的信息，该减的地方减，该补回来的地方补回来
> 

### Depth regression

 还是通过MVSNet的d*P(d)来回归深度图

loss = loss_depth + loss_grad

- loss_depth: 核心修改点是，求完est和gt差的绝对值之后不再除总个数，而是除以有真值点的总数，$d^*$是GT，$\hat{d}$是估计值，$N_d$是有真值像素的总数， $\delta = \frac{D_{max}-D_{min}}{Z-1}$感觉这个有点用
    
    $$\mathcal{L}_{\mathrm{depth}}\left(d^{*}, \hat{d}\right)=\frac{1}{\delta \mathcal{N}_{d}} \sum_{(i, j)}\left|d_{i, j}^{*}  -\hat{d}_{i, j}\right|$$
    
- loss_grad: 加强图像梯度维度的一致性
    
    $$\mathcal{L}_{\mathrm{grad}}\left(d^{*}, \hat{d}\right)=\sum_{(i, j)}\left(\frac{1}{\mathcal{N}_{x}}\left|\varphi_{x}\left(d_{i, j}^{*}\right) -\varphi_{x}\left(\hat{d}_{i, j}\right)\right|+\frac{1}{\mathcal{N}_{y}}\left|\varphi_{y}\left(d_{i, j}^{*}\right)- \varphi_{y}\left(\hat{d}_{i, j}\right)\right|\right)$$
    
    $\varphi$代表像素梯度，$N_x$代表有真值像素并且x方向相邻点也有真值的总数（能计算梯度的点总数）
    
    > 这部分感觉也比较巧妙的水了一个公式
    > 

### 点云重建

深度假设越多消耗资源越大

最后有一个迭代优化深度图的过程，大致是采用了ZNCC等一些指标进行约束，具体实现没有所以也摸不到头脑

## 实验

坦克数据集重叠度很大，fusion时采用了更严格的阈值，抑制outliers

在advanced中的效果没有intermediate好，猜想可能是这些场景的深度范围特别大，超出了作者计算资源的假设