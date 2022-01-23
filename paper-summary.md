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
- 3D正则化采用了各向同性和各向异性的混合模式
- 对特征的提取开始出现分级的思想
- [ ] @作者 更高精度的数据集，内存消耗和计算复杂度降低，融合语义信息
- [ ] @doubleZ 个人感觉这篇跟MVSNet大体框架极为相似，而且有意回避MVSNet的做法，并且不提供源码，把网络结构这些画出来占论文空间

## CVP-MVSNet

- 金字塔的思想：特征提取金字塔构建，cost volume pyramid
- coarse-to-fine的思想 ← relation between depth residual search range and image resolution
- [ ]  不是end-to-end的方法，需要先用其他方法生成深度图
- [ ]  消融实验做的不是很多
- [ ]  处理的图像分辨率太小160*128，而且金字塔也只设置了两层，核心还是图片尺寸太小
- [ ]  @作者 未来希望把工作集成进learning-based structure-from-motion框架

## AttMVSNet

- attention-enhanced matching confidence volume 综合光度一致性和局部场景上下文信息
- attention-guided regularization 层次化的聚合和正则化MCV方法
- loss不仅有深度图像素维度的一致性，还有深度图梯度维度的一致性
- 使用了ZNCC等进行了深度图GT的“滤波”，让真值更可靠
- [ ]  没有消融实验
- [ ]  深度假设自适应

## CasMVSNet

- 可以集成到现有的方法中
- 将单一代价体 → 分解为级联表示
- 通过上一步的深度图缩放下一步的深度假设
- 生成的是全分辨率的深度图
- 准确度提升的同时，GPU和运行时间也显著降低
- 不仅在MVS问题上进行构建和实验，还在Stereo Matching上产出结果，不仅内容丰富而且可信度更高

## PatchmatchNet

- 将传统的patchmatch思想融合进深度学习(这件事不是他做的，他只是用到了MVS上)
- 深度传播和代价体聚合的时候通过网络学习不跨边界的偏移而不是固定窗口
- 尽量使用边缘内的邻居信息而不是固定窗口的信息
- pixel-wise view weight用来检验ref和src的相关性效果不错
- [ ]  感觉只是把Patchmatch算法融合进了MVS框架里
- [ ]  主要是在性能方面尤其是内存消耗上的提升，其他方面提升不很大
- [ ]  文章前半部分的表述不太清晰，对于patchmatch的背景介绍太少，经常看不懂
- [ ]  整体架构还是级联金字塔，只是换了一种方式构建分级的架构