# 立体匹配｜Stereo Matching

立体匹配也称 ~视差估计~、 ~双目深度估计~

- **输入**：一对在同一时刻捕捉的，经过 ~极线校正~ 的左右图像 `Il`和 `Ir`
- **输出**：参考图像(一般选为左图)每个像素对应的视差值对应的视差图`d`
- 根据公式 `z = b*f / d`可获得深度图
    - `b`: 两相机光心距离
    - `f`: 相机光心到成像平面的焦距
    - `d`: 两相机的视差

![Untitled](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/9f1efcb4-af0c-45c3-9635-7ecab6694597/Untitled.png)

## 背景概念

### 对极几何

[对极几何｜Epipolar Geometry](https://www.notion.so/Epipolar-Geometry-5f85b69f8f2841de931da533d1d12898) 

### 视觉模型

- 汇聚式

![Untitled](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/ff36a8dd-7d7c-465a-8cc2-35b5c53d7daa/Untitled.png)

- 平行式

![Untitled](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/5bd072e2-e454-47cf-b15b-8d332814ed8e/Untitled.png)

> 平行式立体视觉模型中，两摄像机光轴平行，因此左右图像间对极线互相平行，且位于相同的图像平面上，因此视差矢量平行于图像的水平线，使得视差矢量退化为标量
> 

---

## 基本流程

匹配代价计算 -> 代价聚合 -> 视差计算 -> 视差优化

### 匹配代价计算

- **目的**：通过匹配代价函数计算~待匹配像素~ 和 ~候选像素~ 间的相关性，匹配代价越小说明相关性越大

> 在右图里寻找哪个像素对应左图中的这里，得到了视差，就可以根据焦距和光心距离计算到深度了
> 
- **细节**：会将视察搜索的范围限定在`Dmin~Dmax`间，因此对于参考图像的每一个像素，用一个`W*H*D`的三维矩阵(::DSI::-Disparity Space Image)存储视差范围内每个像素的匹配代价
- **算法**
    - 摄影测量：灰度绝对值差 AD、灰度绝对值之和 SAD、归一化相关系数 NCC
    - CV：互信息 MI、Census变换 CT、Rank变换 RT、BT等

![Untitled](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/037f5256-1cbd-4f99-9bae-3092a4a03269/Untitled.png)

### 代价聚合

- 上一步计算出匹配代价的**问题**：只考虑了局部信息，通过两个像素邻域内一定大小窗口内的像素信息计算代价值，但这很容易受到噪声影响，当影像处于弱纹理或重复纹理区域(有意义信息很少，噪声影响很大的区域)，代价值极有可能无法准确反映像素之间的相关性
- **根本目的**：考虑全局信息，对DSI进行优化，让优化过的代价值能准确的反映像素之间的相关性
- **大致步骤**：类似于 ~视差传播~，
    1. 信噪比高的区域初始匹配效果很好，原始代价就能很好的反映相关性，可以更准确的得到最优视差值
    2. 通过建立邻接像素之间的关系，每个像素在某个视差下的新代价值会根据相邻像素在统一视差或附近视差下的代价值重新计算(如相邻像素应该具有连续的视差值)
    3. 传播至信噪比低、匹配效果不太好的区域
    4. 最终得到新的矩阵`S`
- **常用方法**：扫描线法、动态规划法、SGM算法中的路径聚合法

![Untitled](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/6558a44b-b66d-413f-907e-7748ac82054b/Untitled.png)

> 注：这一步极为关键，直接决定了算法的准确性
> 

### 视差计算

使用赢家通吃算法(WTA, Winner-Takes-All)对代价矩阵`S`选择最小代价值对应的视差值作为最佳视差

> 再次注：因为这部没什么本质性的操作，因此要求聚合效果非常好
> 
> 
> ![Untitled](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/fd321a12-fe96-4050-8275-dd74abddc595/Untitled.png)
> 

### 视差优化

- **目的**：对上一步得到的视差图进行进一步优化，改善视差图的质量，主要包括 ~剔除错误视差~、 ~适当平滑~、**子像素精度优化**等
- **算法**
    - 剔除因为遮挡和噪声而导致的错误视差：左右一致性检查算法(Left-Right Check)
    - 剔除孤立异常点：剔除小连通区域算法
    - 对视差图进行平滑：中值滤波(Median Filter)、双边滤波等平滑算法(Bilateral Filter)
    - 其他提高视差图质量的方法：鲁棒平面拟合(Robust Plane Fitting)、亮度一致性约束(Intensity Consistent)、局部一致性约束(Locally Consistent)等
- 子像素精度优化：由于WTA算法得到的视差值是整像素精度，可以对其进一步子像素细化
    - 一元二次曲线拟合算法：通过最优视差下的代价以及左右两个视差下的代价值拟合一条一元二次曲线，取二次曲线的极小值点代表的视差值作为精细化后的子像素视差值

![Untitled](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/c53d04c9-d70a-454e-8979-ed95023fba80/Untitled.png)

---

## References

- [3D视觉之立体匹配（Stereo Matching） - 知乎](https://zhuanlan.zhihu.com/p/161276985)