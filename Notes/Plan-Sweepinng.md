# Plane Sweeping | 平面扫描

- 输入：一系列经过校准的照片以及拍摄相机对应的投影矩阵
- 假设(定义)：所有物体只有漫反射，有一个虚拟相机cam x，定义一个近平面和一个远平面，在这之间物体被一系列密集的平行平面划分

<img src="https://upload-images.jianshu.io/upload_images/12014150-783f766dc9699bd9.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240" width="40%" /> <img src="https://upload-images.jianshu.io/upload_images/12014150-428eaa236dce7b98.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240" width="40%" />


## 核心思想

如果平行平面足够密集，物体表面的任意一点p一定位于某平面Di上，可以看到p的相机看到点p必定是同一颜色；假设与p在同一平面的另一点p’，不位于物体表面，则投影到每个相机上呈现的颜色不同， 于是Plane Sweeping算法假设：

> 对于平面上任意一点p，其如果投影到每个相机上的颜色均相同，那么可以说这个点很大概率是物体表面上的点
> 

## 核心步骤

对于平行平面Di上的每个点p，将其投影到所有相机上，之后根据投影的颜色进行匹配计算，得到点p对应的分数以及其对应的深度，p得分越高，代表其投影在各相机上的颜色越接近
p计算之后，将其投影到虚拟相机cam x上，从后向前扫描，如果某一个平面Dj上的点q投影到cam x后，发现得分高于之前该点的分数，则更新该点的分数和深度，直至平面扫描结束

## 数学建模

### Homography

相机C看$x_\pi$和C’看$x_\pi$存在单应关系

<img src="https://upload-images.jianshu.io/upload_images/12014150-f18528ef490079dc.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240" width="40%" />


> 省略公式和推导
> 

### Cost Function

有了点的对应关系，接下来就是找到cost function对其优化。由于噪声的影响，不能只利用颜色信息，需要结合窗口信息进行比较

<img src="https://upload-images.jianshu.io/upload_images/12014150-fad32970b77b139e.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240" width="40%" />


- **局限性**：基于窗口的匹配，窗口内像素与中心像素极可能不在一个平面(阶梯状、不连续性)，因此会干扰中心像素的匹配
- 改进之一
    - photo consistency：直接估计窗口内平面方程，有了平面方程直接带入该点坐标即是深度值

## Reference

- [Plane-sweeping - 代码天地](https://www.codetd.com/article/2992701)
- Multi-resolution real-time stereo on commodity graphics hardware
- Real-time Plane-sweeping Stereo with Multiple Sweeping Directions