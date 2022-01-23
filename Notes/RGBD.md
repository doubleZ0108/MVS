# RGB-D｜深度图像

深度图像 = 普通RGB三通道彩色图像 + Depth Map

> RGB图和深度图是配准的，像素之间一一对应
> 

## Depth Map｜深度图

包含与视点场景对象表面距离有关信息的图像通道，通道本身类似于灰度图像，每个像素值是传感器测出距离物体的实际距离

**分类**

![image.png](https://upload-images.jianshu.io/upload_images/12014150-49bc2802210e0459.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

- 与相机距离成比例：较近的表面较暗; 其他表面较轻
- 与标称平面的距离相关：靠近焦平面的表面较暗; 远离焦平面的表面更轻

---

## RGB-D相机

- 结构光法
    - Kinect v1
    - iPhone X
- 飞行时间法(TOF)
    - Kinect v2
    - Phab 2 Pro

### 结构光法

不依赖光照和纹理；夜间可用；主动投影已知图案；功耗低；紧纪律精度高

![image.png](https://upload-images.jianshu.io/upload_images/12014150-f934ec8a47921f6f.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

- **时分复用编码**：投影N个连续序列的不同编码光，接收端根据接收到的N个连续的序列图像识别每个编码点
    - 优：可得到较高分辨率深度图（有大量的3D投影点）；受物体本身颜色影响小（采用二进制编码）
    - 缺：只适用静态场景；计算量大（识别一个编码点需要计算连续N次投影）
    
- **空分复用编码**：根据邻域内的一个窗口中所有点的分布识别编码
    - 优：适用于运动物体
    - 缺：不连续的物体表面（遮挡）可能产生错误的窗口解码

    ![image.png](https://upload-images.jianshu.io/upload_images/12014150-ab722ce690a1d755.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)
    
    

### TOF飞行时间法

- 原理：连续发射不可见光脉冲到被测物体上，接受从物体反射回的光脉冲，探测光脉冲的飞行时间计算被测物体距离
- 分类（根据不同的调制方法）
    - 脉冲调制：通过电荷累计计算时间，对元器件要求高
    - 连续波调制(main)：利用相位偏移计算时间
    
    ![image.png](https://upload-images.jianshu.io/upload_images/12014150-4a5c6529cfa1411f.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)
    
- 优：可调节发射脉冲的频率改变测量距离；测量精度不会随着测量距离增大而降低；抗干扰能力强；适合距离比较远的(无人驾驶)
- 缺：功耗大；分辨率低深度图质量差

### RGB-D相机问题

- **深黑色物体影响**：深黑色物体可以吸收大量的红外光导致测量不准

![image.png](https://upload-images.jianshu.io/upload_images/12014150-b8a2fbe7147b6e95.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

- **表面光滑物体影响**：镜面反射时，相机投射的结构光只有当接收器在特定位置时才能接收到
    - 当物体表面超过一定光滑程度时，深度相机精度会急剧下降，甚至没有深度值
- **(半)透明物体影响**：深度值的歧义性
    - 半透明时同一次发射的结构光可能会接收到两次
    - 全透明时接收不到
    
    ![image.png](https://upload-images.jianshu.io/upload_images/12014150-0c3137f38f691371.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)
    
- **视差影响**：结构光深度相机发射端和接收端通常有一定的间距

### RGB-D特点

- 优：规避了纯cv的弱点
    - 缺乏纹理
    - 光照不足
    - 过度曝光
    - 软件计算复杂度高
    - 快速运动
    
    ![image.png](https://upload-images.jianshu.io/upload_images/12014150-6d36055328d24e3c.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)
    
- 缺
    - 受深色物体、(半)透明物体、镜面反射物体、视差影响
    - 深度图质量与硬件密切相关
    - 功耗大，成本高

## References

- 🌟RGB-D数据集：[RGB-D (Kinect) Object Dataset](https://rgbd-dataset.cs.washington.edu/demos.html)
- [RGB-D相机介绍_LazyMe-CSDN博客](https://blog.csdn.net/weixin_46581517/article/details/105232489)
- [RGB-D（深度图像） & 图像深度_JNing-CSDN博客](https://blog.csdn.net/JNingWei/article/details/73609127)