# CasMVSNet代码结构

## 数据集

相较于MVSNet增加了`Depths_raw`文件夹

- `scans/`: 保存了原始分辨率的深度图GT和mask(1200, 1600) 代码里用的是这个
- `scanx_train/`: 低分辨率的深度图和mask(128, 160)

Cas和MVSNet与CVP很大的不同是，或者应该说CVP和其他两个很大的不同在于，CVP的数据集中train就是train，test就是test，而MVSNet这套数据集train里有完整的test结果(GT)，所以在train完一个Epoch之后会做一个完整的test，并通过2mm等指标观察当前模型的性能，而不需要完整DTU定量测试

## 参数设置

基本参数与MVSNet和CVP-MVSNet大同小异，只列重点：

- batch_size: 1batch大约对应5G显存，如果GPU数量>1，这里的batchsize指的是每个GPU的batch数量
- 深度假设：`numdepth=192`, `interval_scale=1.06`，这两个还是参考MVSNet的设置
- `eval_fraq`: 进行测试的频率，一般设置为3，即训练三个Epoch后进行一次完整的test
- `share_cr`：是否共享代价体回归
    - True：每层的cost regularization都是同一个CostRegNet
    - False：每层的代价体正则化都是ModuleList中的一项（虽然每项的配置是一模一样的）
- `ndepths`：每阶段深度假设层数，48,32,8
- `depth_inter_r`：每阶段深度假设比率，4,2,1
- `dlossw`：每阶段loss权重，0.5,1.0,2.0
- `cr_base_chs`：代价体回归base channel
- `using_apex` `sync_bn`：apex相关配置，主要为了使用sync_bn

## 金字塔结构

cost volume尺寸为[B, C, D, H, W]

- stage1
    - 分辨率：H/4, W/4 (128, 160)
    - 特征：C=32通道
    - 深度假设：D=48层，就是从425～935的48层
- stage2
    - H/2, W/2 (256, 320)
    - C=16
    - D=32，深度假设间隔interval = 2.5 * 1.06 * 2
- stage3
    - H, W (512, 640)
    - C=8
    - D=8，深度假设间隔interval = 2.5 * 1.06 * 1

## Data模块

### train

以下几个模块都与MVSNet和CVP-MVSNet大同小异，几乎没差别：

- `build_list()`
- `read_cam_file()`
- `read_img()`
- `read_depth()`

增加了几个方法：

- `prepare_img()`: 将原尺寸图片降分辨率并裁剪中间的一块，1600*1200 → 640*512，用的下采样方法是cv2的nearest最近邻差值方法
- `read_mask_hr()`: 读取mask 做下采样 再做金字塔，最终三阶段的mask尺寸为(160*128) / (320*256) / (640*512)，同时读取的时候过滤只保留了像素>10的部分
- `read_depth_hr()`: 读取depth真值，操作同上

`get_item()`

- `imgs`: 1张ref + nview-1张src
- `proj_matrices`: 三阶段的投影矩阵，每阶段的投影矩阵跟之前不一样的地方在于维度为(2,4,4)，不提前把内外参乘起来，本质上只是把内外参统一存到一个tensor里；数据集里的相机参数是/4之后的 也就是第一阶段的，后两个阶段内参依次*2
- `depth`: 三阶段深度图真值
- `depth_values`: min～max interval个深度假设层
- `mask`: 三阶段的mask

## Train

- 在初始化阶段有一些关于多gpu、模型并行、apex等操作
- 建立model
- 建立Dataset和DataLoader等也有对应的并行化版本

`train()`

- lr warmup策略
- train一个Epoch
- 保存断点
- test一个Epoch，通过test的error等指标可反映出这轮训练的效果，dtu test集也有GT，这样可以避免定量测试

`cas_mvsnet_loss()`

对三层金字塔做loss相加，默认三层权重分别为[0.5, 1.0, 2.0]

最终返回：

- `total_loss`：加权后用于网络梯度反传的loss累计
- `depth_loss`：不加权前的loss累计

### FeatureNet

有两种架构，且对不同金字塔数量区别对待，默认是3层：

输出：(H/4, W/4, 32) / (H/2, W/2, 16) / (H, W, 8)  H=612 W=640

首先都先通过MVSNet的8层CNN并进行4倍下采样

stage1：再经过一个1*1卷积 → (H/4, W/4, 32)

- `unet`：
    - stage2: （8层输出反卷积 + 拼接conv1 + 再卷积一次） → (H/2, W/2, 16)
    - stage3: 同上 → (H, W, 8)
- `fpn`：
    - stage2: （8层输出上采样两倍 + conv1卷积一次）整体再卷一次 → (H/2, W/2, 16)
    - stage3: （再上采样两倍 + conv0卷积一次）整体再卷一次 → (H, W, 8)

### CostRegNet

输入：stage1回归[B, 32, 48, H/4, W/4] ，stage2回归[B, 16, 32, H/2, W/2]，stage3回归[B, 8, 8, H, W]

标准的3D U-Net，只不过网络里凶残到降了8倍分辨率[1, 64, 6, 16, 20]

例如stage1: [32, 48, 128, 160] → [8, 48, 128, 160] → [16, 24, 64, 80] → [32, 12, 32, 40] → [64, 6, 16, 20]，再一路反卷积加回去，最终把C维回归掉，变为[1, 48, 128, 160]

### DepthNet

一个完整的MVSNet流程，特征提取 → 构建代价体 → 代价体正则化 → 概率体深度回归 → 得到深度图

- 输入参数：
    - 当前阶段的ref和特征体 + src特征体们
    - 当前阶段的投影矩阵
    - 当前阶段的深度假设
    - 深度假设层数 [48, 32, 8]
    - 是否共享代价体正则化（默认为False，即每层的正则化网络都是ModuleList中的一项，区别在于网络第一层in_channel分别对应FeatureNet网络的三层输出channel维度）
- 每一阶段的深度图和置信图都被保存下来返回了

## CasMVSNet

- 默认参数：
    - `ndepths`=[48, 32, 8]
    - `depth_interval_ratio`=[4, 2, 1]
    - 不共享代价体正则化网络
    - 特征提取用的是`fpn`(特征金字塔网络)
    - 不进行refine
    
    ---
    
    - `num_stage`：默认是3阶段
    - `stage_infos`: 每阶段的scale 4/2/1
1. **特征提取**：ref和srcs图片都经过特征提取金字塔，每个视点得到三阶段的特征体
2. for 每层金字塔
    1. 取出这层金字塔要用的 特征体、投影矩阵、scale
    2. **深度假设**：初始是 从425～935的192个数，同MVSNet的配置，192层深度假设间隔，interval=1.06覆盖DTU的最大最小范围
        1. 第一层：[B, 48, 512, 640]单独处理，本质上就是对425～935范围采48层深度间隔，只不过为了统一数据尺寸，在整张图片上进行了复制，就相当于每个像素都是425～935采48个深度
        2. 第二层：[B, 32, 512, 640] 在第一阶段深度图基础上缩小假设深度范围，层数也缩小到32层；具体实现是将第一阶段的深度图上采样到原图分辨率，每个像素分别向前后采16层作为depth_min和depth_max，在depth_min~depth_max范围内再均匀划分32层间隔；同时要注意每一阶段向前后采获取minmax时的interval是递减的，对应参数中的`depth_interval_ratio`
        3. 第三层：[B, 8, 512, 640] 同上，进一步缩小范围和假设层数
    3. 跑一个完整的DepthNet进行MVSNet流程
    4. 上采样到原图分辨率(512, 640)再走下一轮深度假设、正则化…
        
        > 这里挺奇怪的是深度图没有上采样2倍，而是直接上采样到原图尺寸
        >