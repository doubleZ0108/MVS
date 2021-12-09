# CasMVSNet

## Abstract

之前的问题：构建3D代价体，随着分辨率的增加cubic增长

memory and time efficient cost volume formulation complementary

首先构建特征金字塔，每一步通过上一步的结果缩小深度假设范围

同样也是coarse-to-fine: gradually higher cost volume resolution and adaptive adjustment of depth intervals

DTU准确度提升了35%，GPU和运行时间降了50%

可以集成到现有方法中

## Introduction

3D CNN可以捕捉更多的几何结构（光度一致性、遮挡、透视变化的畸变）

【cascade表示】

1. 特征金字塔提取多尺度特征
2. 早期cost volume构建在大尺度的语义特征上（稀疏采样）
3. 后期cost volume通过之前估计的深度图适应性的调整深度假设范围，构建精细的代价体

这种适应性深度假设和图像分辨率调整使得计算资源被用在more meaningful region，从而降低GPU和时间消耗

【两类问题】

- multi-view stereo 主线上的DTU那些
- stereo matching：end-point-error(EPE), GwcNet…

------

## Related Work

基于3D代价体的方法受限于下采样cost volume和最后通过插值生成高分辨率视差

cascade可以与之前的方法融合在一起，提高分辨率和性能

------

## Methodology

### 代价体表示

【构建3D代价体的三步】

1. 假设离散的深度假设平面
2. 将每个视点提取的2D特征投影到假设平面上，构建feature volume
3. 最终fuse together构建3D cost volume

pixel-wise构建cost是不稳定的，在遮挡、重复纹理、低纹理、反射等区域都不好 → 3D CNNs at multiple scales可以用来聚集上下文信息，使得正则化时更鲁棒

【MVS问题中的构建】

相机前平行平面当作深度假设平面，深度假设范围通过稀疏重建得到(colmap)

通过单应变换将2D feature map投影到ref视点的假设平面上，构建feature volume

最终通过方差将每个视点的特征体聚合成一个cost volume

【SM问题中的构建】

视差水平作为假设平面，范围要针对指定场景决定

由于左右视点都被矫正过，因此只是一个x轴的平移(相当于MVS中的投影变换，只是变得很简单)

之后通过类似方法进行聚合，不过方法有

- 直接聚集，不进行特征降维
- sum of absolute differences
- 计算左右相关性，product only a single-channel correlation map for each disparity level
- group-wise correlation

### 级联代价体

固定的代价体尺寸是 $W \times H \times D \times F$ 分辨率 深度假设 特征通道，都对acc有提升，但影响效率 16G的P100最大能跑$1600 \times 1184 \times 256 \times 32$

【深度假设范围】

公式计算起来很简单$R_{k+1} = R_{k} \cdot w_k, \ w_k < 1$ 下一步比上一步的深度假设缩小

【深度假设间隔】

两假设平面间的距离$I$也是最开始大一点粗糙一点，$I_{k+1} = I_k \cdot p_k, \ p_k < 1$，逐步假设变小变精确

【深度假设平面数】

上两步已经得到了$R_k, I_k$，则数量$D_k = \frac{R_k}{I_k}$

通过上述三步，显著的减少复杂度

【空间分辨率】

每一步空间分辨率翻倍 1/4 → 1/2 → 1

> 级联代价体每步学的都是depth residual，而不是整个深度图

### 特征金字塔

之前方法是首先通过MVSNet生成低分辨率的深度图，构建低分辨率的代价体，然后迭代优化

这样的构建方法只是用最终的feature map，只有上层语义信息，缺少底层表示

这里采用不同分辨率的特征图构建代价体，实验中通过1/16, 1/4, 1的特征图构建三个代价体

------

## 实验

- 640*512，三个视点
- 金字塔三层
- 深度假设：48，32，8
- 深度间隔：4倍，2倍，1倍
- 特征图：1/16，1/4，1

### 消融实验

【阶段数】

overall质量先显著增加，后平稳，最终选择的是3阶段模型

- 深度假设数：96，48，48
- 深度间隔：2，2，1
- cascade？：将原始固定的代价体替换为三阶段的代价体
- upsample？：对特征图插值，以增加代价体的空间分辨率
- feature pyramid？：通过特征图金字塔增加代价体分辨率

空间分辨率的提高对重建结果的影响大

级联代价体不同阶段分别学习效果更好(不进行参数共享)