# PatchmatchNet

## Abstract

cascade formulation of Patchmatch

首次提出iterative multi-scale Patchmatch和adaptive propagation and evaluation scheme for each iteration.

主要是在性能上的提升较多，目标定位感觉是在边缘设备上能用

## Introduction

学习方法效果确实很好，但是大都没考虑消耗，普遍做法是下采样图像，在低维算代价体和深度图

在低分辨率估计深度图非常影响准确性

对于实际几M的图像不能利用全部分辨率

【Patchmatch】

利用随机迭代方法估计nearest neighbor field

深度图固有的空间一致性被用来快速找到解，而不需遍历所有可能

内存消耗与深度假设独立

1. patchmatch融合进端到端的MVS框架中(coarse-to-fine的框架中)
2. 增加了传播和代价评估两步骤；在代价体聚集时估计场景的可见性；训练时的随机化处理遮挡
3. 在ETH3D上也测试了

## Related Work

Gipuma：multi-view extension of Patchmatch stereo

本文是把传统的PatchMatch融合进深度学习中

adaptively combine the information of multiple views based on visibility information

robust training strategy to include randomness → 在可视和泛化性上更好