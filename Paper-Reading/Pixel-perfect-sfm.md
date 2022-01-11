# Pixel Perfect SFM

Pixel-Perfect Structure-from-Motion with Featuremetric Reﬁnement

ICCV 2021 oral

Colmap优化建图组件

## 核心思想

保证跨视角可复检的特征点相对位置一致

finding local features that are repeatable across multiple views，传统方法检测出特征点后位置就保持不变，有可能会一直传播误差到最终的3D点云，改为神经网络提取的深度特征

1. 特征匹配后通过Featuremetric(深度特征度量)对特征点位置进行优化
2. 增量重建过程中通过Featuremetric进行BA优化

## Introduction

特征提取：从一张图里提取特征很容易受到限制，采用CNN提取高维特征可以很好的考虑全局信息

multi-view geometric optimization with BA is commonly used to refine cameras and points using reprojection errors

adjust key points and bundles, before and after reconstruction, by direct image alignment in learned feature space

## Pipeline

输入输出还是之前的

- 输入：一组待重建图像
- 输出：场景图（3D点、相机位资、相机内参）

track：一个3D点在不同图像中的2D观测

![Untitled](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/d91e1c2d-db8f-48bf-bb56-78de3f236dcf/Untitled.png)

首先用CNN提取深度特征，根据稀疏特征匹配得到初始tracks，调整每一个track对应特征点的位置

根据调整之后的特征点进行SFM重建，重建时BA优化的残差由重投影午茶变为Featuremetric误差

## 实验

进行了非常大量的实验，在准确性、完整性、相机位资估计、性能、耗时都有很大程度的提升

![Untitled](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/09060f6a-d7ef-477c-b6ba-faaca7ab5f65/Untitled.png)

绿色点是本文提出的方法，在不同视角下保持一致；而原始特征点蓝色容易受到早生的影响

[ICCV 2021 | COLMAP 优化建图组件 Pixel-Perfect SFM](https://mp.weixin.qq.com/s/yNlgtZA2yFATbtVikV2uDw)