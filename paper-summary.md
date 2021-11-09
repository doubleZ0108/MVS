# MVS系列核心创新点 & 缺陷

## MVSNet

- 深度学习MVS整体pipeline
- 把MVS问题解耦成视点深度图的估计+fusion的过程
- cost volume(image features + camera parameters homography)
- variance-based方法融合每个视点的特征体为一个cost volume
- [ ]  所有视点对于cost volume构建的贡献相同
- [ ]  数据集由于没有深度图真值，自己构建无法做到100%精确
- [ ]  被遮挡的像素不该被用于训练
- [ ]  3D 卷积正则化内存消耗很大
- [ ]  fusibile和GT深度图都是自己生成的，后续基本没人动，但他们都不准

## R-MVSNet

- 通过GRU单元sequentially regularize 2D cost map along the depth direction
- 空间复杂度从O(N^3) → O(N^2)
- 深度范围可以无穷假设(这点没看出来)
- 4.2节有一个深度图优化做到sub-pixel级别精确度效果还可以
- [ ]  空间降低的同时，时间消耗提升

## PointMVSNet

- 直接在点云上迭代处理，不再主要依赖cost volume，keep the continuity of the surface structure
- predict the depth in a coarse-to-fine manner，每轮迭代深度假设间隔都可以缩小，实现更精准的预测
- 3D空间中聚集K近邻的信息
- 构建特征金字塔的时候取每个下采样之前的层一起构建特征体
- 可以在ROI局部稠密重建，动态交互灵活
- [ ]  时间消耗和迭代次数成正比
- [ ]  依赖其他方法先生成粗糙的深度图

## P-MVSNet

- highlight ref图像的贡献，构建pixel-wise match confidence volume
    - 特征提取的时候对ref图多一个步骤，分辨率也保留的更高
- 把pixel-wise → 聚合为patch-wise