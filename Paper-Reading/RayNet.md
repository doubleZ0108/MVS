# RayNet

RayNet: Learning Volumetric 3D Reconstruction with Ray Potentials(射线势)

CVPR 2018

由非常著名的Max-Planck-Gesellschaft(德国马克思-普朗克研究所)联合苏黎世理工和微软共同提出

---

## 马尔可夫基础理论

**马尔可夫性质**：随机过程未来的状态仅依赖于当前状态，即给定现在状态时与过去状态是条件独立的

**势函数**：两变量间的相关关系

**团**: 图中节点的子集，任意两个节点间都有边

**MRF**：每个节点代表一个变量，节点间的边代表两变量间的依赖关系

多个变量间的联合概率分布可以基于团分解为多个势函数的乘积

---

## Abstract

通过不同视角的图像进行稠密重建

当前CNN方法取得了很好的效果，但是并没有考虑到成像物理学(透视几何、遮挡等)

融合ray-potential马尔可夫随机场的方法明确的建模了这些物理过程，但不能处理不同视角间巨大的表面外观差异

RayNet结合二者的优势，通过CNN学习view-invariant的特征表示，通过MRF明确编码透视投影和遮挡

同时在训练的时候进行经验风险最小的训练，而不是loss最小

---

## Introduction

**passive 3D reconstruction**：ill-posed 问题，核心在于occlusion and surface appearance variation，每次重建的顺序不同会非常影响最终的结果

> well-posed：1. 解是存在的 2. 解是唯一的 3. 解根据厨师情况连续变化，不会跳变，即解是稳定的

【解决遮挡】

**MRF with high-order ray potentials**: 明确建模了图像的物理成像过程 along each viewing ray，加强了沿射线第一个可见表面像素的一致性，这样对于遮挡问题可以做到全局一致性

但这种方法太过复杂，并且把问题局限在了像素级别的颜色比较

【解决表面变化】

手工标注的图像相似性度量没法解决这个问题，通过CNN这类学习的方法可以很好的对局部视角变化和光照变化鲁棒

但完全没考虑投影集合和遮挡问题，因此需要很大的模型和很多的labelled训练数据

✨【Ray的融合方案】

- **local information around every pixel**: CNN learns surface appearance variations
- **global information about the entire scene**: MRF explicitly encodes the physics of perspective projection and occlusion

同时使用最小化经验风险的度量，把MRF的输出随机反传回CNN，这里的MRF类似于regularizer 

---

## Model

CNN提取图像特征 → 聚集相邻视点得到深度表示 → 输入MRF进行遮挡约束

将MRF构建为可微函数，因此可以做到端到端反向传播

### CNN

图像通过2D CNN，得到32维特征

网络同样是权重共享并且CNN-BN-ReLU等

one ray per pixel，接下来的目的就是计算每一个ray方向的深度分布

大体是将每个体素向ref和src投影，再计算匹配点对的inner product得到平均的surface probability

### MRF

由于没考虑遮挡，刚刚得到的深度分布noisy

将体素是否有重建点定义为随机事件$o_i \in \{ 0,1 \}$

根据MRF的性质，可以把联合概率分解成空间的点和射线上的点势函数的乘积

$$p(\mathbf{o})=\frac{1}{Z} \prod_{i \in \mathcal{X}} \underbrace{\varphi_{i}\left(o_{i}\right)}_{\text {unary }} \prod_{r \in \mathcal{R}} \underbrace{\psi_{r}\left(\mathbf{o}_{r}\right)}_{\text {ray }}$$

- unary：通过伯努力分布刻画，表示空间中的某个格子是否被填充
- ray：加强沿射线的第一个占用体素的一致性

通过几轮马尔可夫过程的迭代，优化估计沿射线到达体素的距离

但计算环路高阶模型的精确解是NP-hard问题，因此通过循环和积信念传播计算近似解

前向factor-to-variable可以在限行时间求解，因此设定了固定的迭代次数，然后再通过variable-to-factor反向更新

### Loss

不仅仅用和GT的l1损失，还乘以概率分布构建经验风险最小的loss

### Training

MRF需要可微才能进行反向传播，但反向传播需要存下来所有迭代过程中的belief信息，内存消耗是不能容忍的

反向传播的时候采用mini-batch，随机从input中提取子集，而且从道理上讲每次只融合一小部份ray的信息更有利

同时还预训练了第一部分的CNN网络

---

## Experiment

【数据集】

- Aerial
- DTU

【评价指标】

- 准确度
- 完整度
- per pixel men depth error
- Chamfer distance

边缘处保留更多信息

因为目的是获得更完整重建，因此没有剔除不可靠的点来刷acc的分

---

## Conclusion

**CNN** learns view-invariant feature representations(difficult to model)

**MRF** with ray potentials explicitly models perspective projection and enforces occlusion constraints across viewpoints(不通过学习)

【future work】

通过octree数据结构表示进一步提升重建图像的分辨率

扩展预测每个体素的语义标签