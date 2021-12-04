# PatchmatchNet

## Abstract

learnable cascade formulation of Patchmatch

首次提出iterative multi-scale Patchmatch和adaptive propagation and evaluation scheme for each iteration.

在性能上的提升较多

---

## Introduction

学习方法效果确实很好，但是大都没考虑消耗，普遍做法是下采样图像，在低维算代价体和深度图

在低分辨率估计深度图非常影响准确性

对于实际几M的图像不能利用全部分辨率

【Patchmatch】

利用随机迭代方法估计nearest neighbor field

深度图固有的空间一致性被用来快速找到解，而不需遍历所有可能

内存消耗与深度假设独立

传统patchmatch的时间高效和深度学习的高性能融合

【contributions】

1. patchmatch融合进端到端的MVS框架中(coarse-to-fine的框架中)
2. 增加了传播和代价评估两步骤；在代价体聚集时估计场景的可见性；训练时的随机化处理遮挡提升鲁棒性
3. 在ETH3D上也测试了

---

## Related Work

Gipuma：multi-view extension of Patchmatch stereo

本文是把传统的PatchMatch融合进深度学习中

adaptively combine the information of multiple views based on visibility information

robust training strategy to include randomness → 在可视和泛化性上更好

---

## Methods

### Multi-scale Feature Extraction

首先通过特征金字塔提取不同分辨率的pixel-wise的特征

### Learning-based Patchmatch

> 这一部分整体都没怎么看懂
> 
1. **initialization**：随机假设和随机扰动

从dmin到dmax随机假设初始深度，后续迭代中给一些扰动(扰动是在逐渐减小的)，前面的假设会用来refine当前的假设

1. **propagation**：把假设传播给邻居

只有物理上同一表面的像素深度值上有相关性，因此不应该只从静态的邻居传播深度假设

Deformable Convolution Networks

2D CNN对特征图上的每个像素学习一组2D offsets(再加上一个固定的假设)，最后通过插值得到这一点的深度假设

1. **evaluation**：计算所有假设的匹配代价，选择最好的，循环(2)(3)直到收敛

【differential warping】

【matching cost computation】

进行聚集，使得每一个像素和每一个深度假设都只有一个cost

compute the cost per hypothesis via group-wise correlation and aggregate over the views with a pixel view weight(通过一个小网络进行加权的聚集)

整体构建的思路还是很巧妙的，大体思想是对特征体的每个channel加权计算per pixel and depth hypothesis的single cost

核心目的是让src和ref都可见的点权重变大，深度不连续的地方变小

【cost aggregation】

最核心的问题是不要跨边界聚集

还是通过上面类似propagation的方法聚集（公式略）

【depth regression】

还是通过softmax转换为概率体，可以这么吹：

used for sub-pixel depth regression and measure estimation confidence

### depth map refinement

首先将深度图上采样到原始图像尺寸，用图像进行深度剩余网络训练（记得把深度图归一化到0～1）

---

## Experiments

【训练集选择】

之前都是选择跟ref评分最高的两个src进行训练

这里随机从最高的4个中选择2个，提高泛化性，也相当于数据增强了，并且可以着重训练weak visibility的部分

---

## Conclusion

与视差范围无关，不依赖3D cost volume正则化

时间和空间消耗非常牛

未来打算放到移动设备上