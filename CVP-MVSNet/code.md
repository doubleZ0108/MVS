# CVP-MVSNet代码结构

## 数据集

【Train】

- Cameras：每个视点的相机信息和配对信息，跟MVSNet一致，删除了一些没用的东西，pair配对信息也完全一样
- Depth：深度图真值，跟MVSNet一致，不同点是没有mask图，训练的深度图也是128*160
- Rectified：训练用的图片，结构跟MVSNet是一样的，只不过分辨率从640*512 → 160*128小了很多（注意视点从1开始标号，光照从0开始标号）

> 图像缩小了4倍，但是相机参数没改？

【Test】

- Cameras：跟Train相比相机外参没变，内参变大了10倍左右，但整体是跟MVSNet完全一样的
- Rectified：测试用的图片，尺寸与MVSNet一样是全尺寸1600*1200

> 有趣的是MVSNet图片格式是jpg，CVP格式是png，jpg一张700K，png一张2.6M

## 参数

- 增加了`nscale`指定金字塔的层数
- `epoch`从16增大到了40
- `loss_function`有sl1和mse两个选项

## 具体模块

### Data模块

- `read_image()`：读test图片时会被裁剪为(1184, 1600)

- ```
  MVSDataset
  ```

  - `metas`数组

    - train：跟MVSNet大体一样，共27097条数据，内容的顺序稍微换了以下

    ```python
    # scan   ref src                                light
    ('scan2', 0, [10, 1, 9, 12, 11, 13, 2, 8, 14, 27], 0)
    ```

    - test：共1078条数据，内容也是一样的，区别在于固定了light为3

  - `getitem()`: 根据metas读取图片、相机参数、深度图信息

    - ref src图片

    - ref src相机内外参：这里跟MVSNet不太一样，MVSNet直接在这里就把内外参乘起来，最终返回的是`proj_matrices`，这里只是读取内外参直接返回，在`homo_warping()`开始时多了一步把他们乘起来

    - 深度假设最小值/最大值

    - ref深度图GT金字塔数组: 首先读取深度图GT(128*160)，下采样一次(64*80)，把下采样的图片放到128*160图片的左上角，再放到数组里

      > 最终数组里有两个元素，都是128*160的深度图数据，0号就是正常的GT深度图，1号左上角是下采样1倍的深度图

    - ref深度图mask: 这里的mask开始空的

### [train.py](http://train.py)

- 网络输入：ref图，一组src图，内外参，深度假设范围

- 网络输出：`depth_est_list`

- 按照金字塔结构计算

  loss求和后反传

  - 每一层的`depth_gt`需要进行下采样(Data模块中已经下采样过了，这里就是取左上角的区域)
  - `mask = depth_gt>425` 而不是预先像MVSNet一样处理好的

- loss：smooth_l1和mse两种可选项

- 优化器：Adam，跟MVSNet采用一样的schedule策略

### [eval.py](http://eval.py)

- 同样读入图片尺寸被裁剪为(1184, 1600)并转换到0～1的数值
- 测试过程只生成深度图，还要单独fusion生成点云

### [modules.py](http://modules.py)

- `conv()`: 2D卷积+LeakyReLU网络基模块，默认参数只改变第二维度尺寸，不改变分辨率

- `conditionIntrinsics()`: 根据特征金字塔的缩放构建相机内参金字塔(对于每个视角相机外参都是一样的)

  - 返回值：[B, scale, 3, 3]

- `calSweepingDepthHypo()`: 计算最粗糙的深度假设，深度假设层D=48，从425～1065共采样48层

  - 返回值：[B, D] 这里D=48

  > 其中这个函数前面的参数都没用

- `calDepthHypo()`: 金字塔refine阶段计算更精细的深度假设

  - 训练阶段：精细化的深度假设 = 刚刚得到的粗糙深度图值 ± 4 * intervel(代码里硬编码为6.8085)
  - 测试阶段：思路还是在粗糙的深度图值附近偏移一点 根本不是给人看的 写的太乱太复杂了...

- `proj_cost()`: 金字塔refine阶段计算cost volume剩余

  - 跟homo_warping步骤基本一模一样，区别在于聚集每个src最高分辨率特征体warp后的volume

### [net.py](http://net.py)

- ```
  FeaturePyramid()
  ```

  : 特征提取网络模块

  - 对每张ref和src进行特征提取
  - 对于一张图片的返回值维度是[ [B, 16, H, W], [B, 16, H/2, W/2]]

- `CostRegNet()`：cost正则化网络模块

- ```
  network → forward()
  ```

  返回值

  - **深度图预测值列表**（第0位是最大尺寸的深度图）
  - 光度一致性得到的 **置信度图**

- 两个loss函数二选一

  - `smooth_l1_loss()`
  - `mse_loss()`

1. 特征金字塔构建：对ref和src分别调用特征提取网络，每张图得到两层的特征金字塔
2. 相机内参金字塔构建：根据特征金字塔放缩尺寸放缩相机内参
3. 估计初始深度图
   1. 初始深度范围假设
   2. 构建ref volume
   3. homo构建warp volume
   4. 通过方差聚合得到cost volume
   5. 通过代价体正则化网络得到cost_reg
   6. 再通过softmax得到prob volume
   7. 概率体和深度假设求乘积回归得到原始深度图
4. 上采样深度图并通过特征金字塔refine
   1. 两倍上采样深度图
   2. 计算深度假设范围
   3. 计算cost volume
   4. 通过代价体正则化网络得到cost_reg
   5. 再通过softmax得到prob volume
   6. 概率体和深度假设回归得到更精细的深度图
   7. 把所有得到的深度图都放到数组里
5. 计算光度一致性：这里完全是copy MVSNet，估计作者自己也没看懂