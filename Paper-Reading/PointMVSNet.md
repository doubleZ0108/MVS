# Point-MVSNet

Point-Based Multi-View Stereo Network

ICCV 2019 oral → TPAMI

## 摘要

cost volume → directly process point clouds

predict the depth in a coarse-to-fine manner

首先生成粗糙的深度图，将其转化为点云，通过点云真值进行refine

leverage 3D geometry priors and 2D texture information jointly → feature-augmented point cloud

**key sight**: 节省内存，重建质量更高，同时还有潜在应用(forested depth inference)

## Introduction

【**framework】**

1. small 3D cost volume generate an initial coarse depth map, then converted to point cloud

2. iteratively regress point cloud from initial point cloud using 

   PointFlow

   - residual between current iteration and ground truth
   - 3D flow estimation based on geometry priors inferred from the predicted point cloud and the 2D image

【改进】

- MVSNet construct 3D volume with the fixed resolution: adaptively samples potential surface points in the 3D space ← keep the continuity of the surface structure
- only process valid information near the object surface instead the whole 3D space 计算量小
- 可以先看一下粗糙的场景重建，在感兴趣的地方稠密重建 更加灵活在动态交互

## 相关工作

- 传统MVS问题由来已久
- learning-based方法使用2D网络难以在MVS上进行扩展
- 3D cost volume正则化方法的提出: 3D geometry of the scene can be captured by the network explicitly, and the photometric matching can be performed in 3D space, alleviating the influence of image distortion caused by perspective transformation and potential occlusions
- voxel grid表示方法太低效 → point-based network，同时克服了fix cost volume representation，同时保留空间的连续性

本文: PointFlow module, which estimates the 3D flow based on joint 2D-3D features of point hypotheses

## 方法

1. coarse depth prediction: 现有的MVS方法就足够了
2. 2D-3D feature lifting: 结合2D图像信息+3D几何先验
3. iterative depth refinement: PointFlow

### Coarse depth prediction

use MVSNet to predict a relatively low-resolution cost volume

图像1/8下采样，深度假设层从256缩减到48 or 96，内存消耗为MVSNet的1/20

### 2D-3D feature lifting

【图像特征金字塔】

3-scale，每个下采样的前一层都被提取用来构建最终的特征金字塔 [Fi1, Fi2, Fi3]

【动态特征提取】

fetched multi-view image feature variance + normalized 3D coordinates in world space

- fetched image feature: 3D点的特征可以由多视点feature map的相机投影变换构建，由于刚才算的图像特征是不同尺寸的，因此相机内参也要放缩不同尺度，同样也使用了方差来聚集多个feature map

  (为什么我理解这一步就是把上一步得到的特征求方差构建C呢，也没看到投影变换啊 公式1)？

- normalized point coordinate: 公式(2) 把刚刚的图像特征和这个点的空间位置信息组合在一起

  (空间信息Xp是哪来的啊？)

这样就得到了feature augmented point，它作为下一步PointFlow的输入

这其中，每一次迭代预测深度剩余的时候，点的位置Xp都会更新，因此会取出不同的point feature → fetch features from different areas of images dynamically according to the updated point position

### PointFlow

因为已知相机参数，可以把深度图投影到点云上，for each point, estimate its displacement to the ground truth surface along the reference camera direction by observing its neighboring points from all views

【point hypotheses generation】

由于透视变换，2D特征图的上下文信息不能和3D欧拉空间形成对应关系

对于每个点都构建一系列假设点p‘，这些点是在相机方向周围的点，用他们替换当前点

gather neighborhood image feature information at different depth

通过K近邻得到一张有向图

【edge convolution】

use DGCNN to enrich feature aggregation between neighboring points

通过刚刚得到的feature augmented point对应的一组点，edge convolution是关于这些点的一个非线性函数，最后再通过一些聚集操作(max pooling, average pooling,…)

【flow prediction】

输入feature augmented point，输出是flow → 进而得到深度残差图

【迭代和上采样】

上一轮的深度图可以上采样再进行迭代

每轮迭代深度假设的间隔都可以缩小，更精准的预测

## 实验

尤其在边缘处，高频特征处效果更好

牛在test的时候可以多跑几个迭代让点云变得更加denser，并且深度假设越来越小，让结果更加精准

本文可以看做数据驱动的点云上采样(用reference view里的信息)

可以对ROI感兴趣区域单独稠密，原文里叫 forested depth inference

【消融】

- edge convolution: 局部邻居权重不同 >> 周围所有点对中间点贡献相同
- nearest neighbor: KNN >> 只用邻接像素好很多 (相邻的点深度可能突变或被遮挡，因此只用邻居信息会聚集无关特征)
- feature pyramid: leveraging context information at diffferent scales for feature fetching >> 只用最后一层输出的特征

[Chen_Point-Based_Multi-View_Stereo_Network_ICCV_2019_paper.pdf](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/57e7822d-092f-41ab-b01c-ace5f0468b30/Chen_Point-Based_Multi-View_Stereo_Network_ICCV_2019_paper.pdf)

Point-Based Multi-View Stereo Network

ICCV 2019 oral → TPAMI

## 摘要

cost volume → directly process point clouds

predict the depth in a coarse-to-fine manner

首先生成粗糙的深度图，将其转化为点云，通过点云真值进行refine

leverage 3D geometry priors and 2D texture information jointly → feature-augmented point cloud

**key sight**: 节省内存，重建质量更高，同时还有潜在应用(forested depth inference)

## Introduction

【**framework】**

1. small 3D cost volume generate an initial coarse depth map, then converted to point cloud

2. iteratively regress point cloud from initial point cloud using 

   PointFlow

   - residual between current iteration and ground truth
   - 3D flow estimation based on geometry priors inferred from the predicted point cloud and the 2D image

【改进】

- MVSNet construct 3D volume with the fixed resolution: adaptively samples potential surface points in the 3D space ← keep the continuity of the surface structure
- only process valid information near the object surface instead the whole 3D space 计算量小
- 可以先看一下粗糙的场景重建，在感兴趣的地方稠密重建 更加灵活在动态交互

## 相关工作

- 传统MVS问题由来已久
- learning-based方法使用2D网络难以在MVS上进行扩展
- 3D cost volume正则化方法的提出: 3D geometry of the scene can be captured by the network explicitly, and the photometric matching can be performed in 3D space, alleviating the influence of image distortion caused by perspective transformation and potential occlusions
- voxel grid表示方法太低效 → point-based network，同时克服了fix cost volume representation，同时保留空间的连续性

本文: PointFlow module, which estimates the 3D flow based on joint 2D-3D features of point hypotheses

## 方法

1. coarse depth prediction: 现有的MVS方法就足够了
2. 2D-3D feature lifting: 结合2D图像信息+3D几何先验
3. iterative depth refinement: PointFlow

### Coarse depth prediction

use MVSNet to predict a relatively low-resolution cost volume

图像1/8下采样，深度假设层从256缩减到48 or 96，内存消耗为MVSNet的1/20

### 2D-3D feature lifting

【图像特征金字塔】

3-scale，每个下采样的前一层都被提取用来构建最终的特征金字塔 [Fi1, Fi2, Fi3]

【动态特征提取】

fetched multi-view image feature variance + normalized 3D coordinates in world space

- fetched image feature: 3D点的特征可以由多视点feature map的相机投影变换构建，由于刚才算的图像特征是不同尺寸的，因此相机内参也要放缩不同尺度，同样也使用了方差来聚集多个feature map

  (为什么我理解这一步就是把上一步得到的特征求方差构建C呢，也没看到投影变换啊 公式1)？

- normalized point coordinate: 公式(2) 把刚刚的图像特征和这个点的空间位置信息组合在一起

  (空间信息Xp是哪来的啊？)

这样就得到了feature augmented point，它作为下一步PointFlow的输入

这其中，每一次迭代预测深度剩余的时候，点的位置Xp都会更新，因此会取出不同的point feature → fetch features from different areas of images dynamically according to the updated point position

### PointFlow

因为已知相机参数，可以把深度图投影到点云上，for each point, estimate its displacement to the ground truth surface along the reference camera direction by observing its neighboring points from all views

【point hypotheses generation】

由于透视变换，2D特征图的上下文信息不能和3D欧拉空间形成对应关系

对于每个点都构建一系列假设点p‘，这些点是在相机方向周围的点，用他们替换当前点

gather neighborhood image feature information at different depth

通过K近邻得到一张有向图

【edge convolution】

use DGCNN to enrich feature aggregation between neighboring points

通过刚刚得到的feature augmented point对应的一组点，edge convolution是关于这些点的一个非线性函数，最后再通过一些聚集操作(max pooling, average pooling,…)

【flow prediction】

输入feature augmented point，输出是flow → 进而得到深度残差图

【迭代和上采样】

上一轮的深度图可以上采样再进行迭代

每轮迭代深度假设的间隔都可以缩小，更精准的预测

## 实验

尤其在边缘处，高频特征处效果更好

牛在test的时候可以多跑几个迭代让点云变得更加denser，并且深度假设越来越小，让结果更加精准

本文可以看做数据驱动的点云上采样(用reference view里的信息)

可以对ROI感兴趣区域单独稠密，原文里叫 forested depth inference

【消融】

- edge convolution: 局部邻居权重不同 >> 周围所有点对中间点贡献相同
- nearest neighbor: KNN >> 只用邻接像素好很多 (相邻的点深度可能突变或被遮挡，因此只用邻居信息会聚集无关特征)
- feature pyramid: leveraging context information at diffferent scales for feature fetching >> 只用最后一层输出的特征