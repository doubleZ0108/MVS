# CVP-MVSNet

 ## Abstract

build cost volume pyramid in a coarse-to-fine manner

而不是固定分辨率的代价体，使得网络更轻量化，可以迭代生成高质量的深度图

首先通过对最粗糙图像的前平行平面均匀采样构建最初的cost volume，然后进行pixel-wise depth residual进行refine

与PointMVSNet相比，代价体金字塔比直接在3D点上处理更好

## Introduction

> 传统方法和引入CNN部分介绍的很好！

MVSNet内存消耗cubic级别，R-MVSNet减少了内存但需要更多时间，Point-MVSNet运行时间和迭代次数成正比

CVP首先构建图像金字塔，对于coarest ref图像，采样整个场景的深度范围来构建紧凑的代价体，在金字塔下一级，对当前深度假设的邻域进行residual depth search，构建partial cost volume，最后用3D CNN正则化

也是coarse-to-fine manner，但与PointMVS不同的是四点：

1. 还是构建cost volume，而不是直接卷积3D点云
2. 基于深度采样和图像分辨率的关系构建cost volume pyramid
3. 多尺度3D CNN，覆盖较大的感受野
4. 可以输出小分辨率深度图和小分辨率图像

【**key sight】**

- cost volume pyramid
- coarse-to-fine manner → relation between depth residual search range and image resolution

## Algorithm

> 要做的核心事情 ref, src, depth map那些介绍的很好！！

### Feature Pyramid

因为最终的depth map是低分辨率的，所有没必要用高分辨率的图像，低分辨率的input也可以的

降低内存消耗，提升性能

1. 首先对于每张图片构建(L+1)个级别的图像，从大到每个级别除2的分辨率
   - Ii0就是原图， IiL是最小的图
2. 对金字塔的每一层作用一个同样的CNN
3. 最终得到每层的特征图维度是16，表示为$\left\{\mathbf{f}*{i}^{l}\right\}*{i=0}^{N}, \mathbf{f}_{i}^{l} \in \mathbb{R}^{H / 2^{l} \times W / 2^{l} \times F}$

![Untitled](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/eb87189f-b109-4eb7-97ad-2b6ba6f935f5/Untitled.png)

### Cost Volume Pyramid

之前的方法构建single fixed分辨率的cost volume，因此限制了高分辨率图像的使用

构建代价体金字塔，迭代的estimate and refine深度图

1. 首先用最低分辨率的图像和深度的均匀采样构建cost volume得到粗糙的深度估计
2. 再迭代的通过coarse estimation and depth residual hypotheses构建partial cost volume

【第一阶段】

最小分辨率的图像特征 + 深度假设间隔

通过homography将src投影到ref上（注意 内参矩阵要根据分辨率放缩）

还是通过插值，将src的feature map投影到ref上

最终通过方差将所有的cost聚合成一个，维度是$\mathbf{C}^L \in \mathbb{R}^{W / 2^{l} \times H / 2^{l} \times M \times F}$

M是深度假设间隔(影像深度估计准确度的重要参数)，F是特征维度

从中可以得到一个性质：the correct depth for each pixel has the smallest feature variance(photometric consistency)

【第二阶段】

> 我们的核心目标是获取原图对应的深度图，思想是迭代的从最低分辨率的深度图不断refine获取上层 具体而言，首先对$D^{l+1}$进行双三次插值得到$D^{l+1}*{\uparrow}$，再通过构建partial cost volume回归得到residual depth map $\Delta D^{l}$ 上一层的深度图 $D^{l} = D^{l+1}*{\uparrow} + \Delta D^l$

对于每个像素点的深度剩余表示为 $D^{l+1}_{\uparrow}(u, v) + m \Delta d_p$

- $D^{l+1}_{\uparrow}(u, v)$ 是这个点第一阶段得到的最粗糙深度经过插值后的结果
- $\Delta d_p = \frac{s_p}{M}$ 是要微调的深度剩余 $s_p$: depth search range at this point $M$: number of sampled depth residual
- $m$ 应该可以理解为一个深度剩余调整的范围

讲这个增加了 $\Delta$ 的点通过相机内外参投影到ref图上，还是通过方差哪种方式进行聚集

最终$\mathbf{C}^l \in \mathbb{R}^{W / 2^{l} \times H / 2^{l} \times M \times F}$

### Depth Map Inference

【深度假设范围】

$s_p$ 深度采样很重要，会影响到最终深度估计的精度

深度假设平面没必要太过密集，投影过来的3D点太近，无法为深度推断提供额外信息

实验中，计算图像中距离0.5像素的点的平均深度间隔

对于某个像素的深度剩余，首先把它投影到src上，在src的极线上找相邻的两个像素，沿着src相机通过这两个点到3D空间，就得到了深度剩余refine时候的范围

【深度图估计】

同样采用3D卷积将cost volume pyramid正则化成probability volume

但$P^L$是绝对深度聚集来的， $\{ P^l \}^{L-1}_{l=0}$是剩余深度聚集来的

之后首先对$P^L$ soft-argmax得到coarse depth map，上采样之后，迭代的加上 $\{ P^l \}^{L-1}_{l=0}$ soft-argmax获取深度剩余，不断refine深度图

### Loss

计算loss的时候也将ground truth depth变成了一个金字塔，每一层都做l1损失的比较

## Experiment

【配置】

首先使用MVSNet生成了160*128尺寸的深度图，同时把深度图GT也下采样到这个尺寸

用二者构建第二层的cost volume pyramid

过程中深度假设M不断减小取精

【结果】

深度图更加平滑，在边缘区域捕捉更多高频细节

【消融实验】

金字塔数取2时最好，更多主要问题是coarest太小，生成的initial深度图也太差，核心问题还是数据集本身处理的就不大160*128

剩余优化时取周围2个像素投影过来的范围做精细化调整最好

## Abstract

build cost volume pyramid in a coarse-to-fine manner

而不是固定分辨率的代价体，使得网络更轻量化，可以迭代生成高质量的深度图

首先通过对最粗糙图像的前平行平面均匀采样构建最初的cost volume，然后进行pixel-wise depth residual进行refine

与PointMVSNet相比，代价体金字塔比直接在3D点上处理更好

## Introduction

> 传统方法和引入CNN部分介绍的很好！

MVSNet内存消耗cubic级别，R-MVSNet减少了内存但需要更多时间，Point-MVSNet运行时间和迭代次数成正比

CVP首先构建图像金字塔，对于coarest ref图像，采样整个场景的深度范围来构建紧凑的代价体，在金字塔下一级，对当前深度假设的邻域进行residual depth search，构建partial cost volume，最后用3D CNN正则化

也是coarse-to-fine manner，但与PointMVS不同的是四点：

1. 还是构建cost volume，而不是直接卷积3D点云
2. 基于深度采样和图像分辨率的关系构建cost volume pyramid
3. 多尺度3D CNN，覆盖较大的感受野
4. 可以输出小分辨率深度图和小分辨率图像

【**key sight】**

- cost volume pyramid
- coarse-to-fine manner → relation between depth residual search range and image resolution

## Algorithm

> 要做的核心事情 ref, src, depth map那些介绍的很好！！

### Feature Pyramid

因为最终的depth map是低分辨率的，所有没必要用高分辨率的图像，低分辨率的input也可以的

降低内存消耗，提升性能

1. 首先对于每张图片构建(L+1)个级别的图像，从大到每个级别除2的分辨率
   - Ii0就是原图， IiL是最小的图
2. 对金字塔的每一层作用一个同样的CNN
3. 最终得到每层的特征图维度是16，表示为$\left\{\mathbf{f}*{i}^{l}\right\}*{i=0}^{N}, \mathbf{f}_{i}^{l} \in \mathbb{R}^{H / 2^{l} \times W / 2^{l} \times F}$

### Cost Volume Pyramid

之前的方法构建single fixed分辨率的cost volume，因此限制了高分辨率图像的使用

构建代价体金字塔，迭代的estimate and refine深度图

1. 首先用最低分辨率的图像和深度的均匀采样构建cost volume得到粗糙的深度估计
2. 再迭代的通过coarse estimation and depth residual hypotheses构建partial cost volume

【第一阶段】

最小分辨率的图像特征 + 深度假设间隔

通过homography将src投影到ref上（注意 内参矩阵要根据分辨率放缩）

还是通过插值，将src的feature map投影到ref上

最终通过方差将所有的cost聚合成一个，维度是$\mathbf{C}^L \in \mathbb{R}^{W / 2^{l} \times H / 2^{l} \times M \times F}$

M是深度假设间隔(影像深度估计准确度的重要参数)，F是特征维度

从中可以得到一个性质：the correct depth for each pixel has the smallest feature variance(photometric consistency)

【第二阶段】

> 我们的核心目标是获取原图对应的深度图，思想是迭代的从最低分辨率的深度图不断refine获取上层 具体而言，首先对$D^{l+1}$进行双三次插值得到$D^{l+1}*{\uparrow}$，再通过构建partial cost volume回归得到residual depth map $\Delta D^{l}$ 上一层的深度图 $D^{l} = D^{l+1}*{\uparrow} + \Delta D^l$

对于每个像素点的深度剩余表示为 $D^{l+1}_{\uparrow}(u, v) + m \Delta d_p$

- $D^{l+1}_{\uparrow}(u, v)$ 是这个点第一阶段得到的最粗糙深度经过插值后的结果
- $\Delta d_p = \frac{s_p}{M}$ 是要微调的深度剩余 $s_p$: depth search range at this point $M$: number of sampled depth residual
- $m$ 应该可以理解为一个深度剩余调整的范围

讲这个增加了 $\Delta$ 的点通过相机内外参投影到ref图上，还是通过方差哪种方式进行聚集

最终$\mathbf{C}^l \in \mathbb{R}^{W / 2^{l} \times H / 2^{l} \times M \times F}$

### Depth Map Inference

【深度假设范围】

$s_p$ 深度采样很重要，会影响到最终深度估计的精度

深度假设平面没必要太过密集，投影过来的3D点太近，无法为深度推断提供额外信息

实验中，计算图像中距离0.5像素的点的平均深度间隔

对于某个像素的深度剩余，首先把它投影到src上，在src的极线上找相邻的两个像素，沿着src相机通过这两个点到3D空间，就得到了深度剩余refine时候的范围

【深度图估计】

同样采用3D卷积将cost volume pyramid正则化成probability volume

但$P^L$是绝对深度聚集来的， $\{ P^l \}^{L-1}_{l=0}$是剩余深度聚集来的

之后首先对$P^L$ soft-argmax得到coarse depth map，上采样之后，迭代的加上 $\{ P^l \}^{L-1}_{l=0}$ soft-argmax获取深度剩余，不断refine深度图

### Loss

计算loss的时候也将ground truth depth变成了一个金字塔，每一层都做l1损失的比较

## Experiment

【配置】

首先使用MVSNet生成了160*128尺寸的深度图，同时把深度图GT也下采样到这个尺寸

用二者构建第二层的cost volume pyramid

过程中深度假设M不断减小取精

【结果】

深度图更加平滑，在边缘区域捕捉更多高频细节

【消融实验】

金字塔数取2时最好，更多主要问题是coarest太小，生成的initial深度图也太差，核心问题还是数据集本身处理的就不大160*128

剩余优化时取周围2个像素投影过来的范围做精细化调整最好