# PointMVSNet实验配置

## 环境配置

总体环境可以复用MVSNet的conda环境，激活MVSNet的conda环境后做如下修改

1. 在`install_dependencies.sh`中选择当前没安装过的手动安装，不建议直接执行shell文件(比如cuda的版本可能出问题)
2. `pip install -r requirements.txt`
    > 注意官方shell里最后一行少了后缀名
3. 运行`compile.sh`进行编译
    > PointMVSNet跟其他MVS方法有很大区别在于它的代码有C的部分，而且需要链接和编译，整体架构也离主脉络较远，但也要钦佩作者不循规蹈矩的勇气

【编译的一些问题】
- 可能会报错`TH_INDEX_BASE`找不到，全剧搜索这个变量将其改为0即可
- `gather_knn_kernel.cu`可能会报错`AT_CHECK`找不到，将`CHECK`开头所有宏函数的定义和使用都注释掉能成功编译

## Train

1. 在`dtu_wde3.yaml`中修改数据集路径
    > 其他参数可以不用动
2. 推荐把`CHECKPOINT_PERIOD`改为1，默认4次才保存模型
3. 如果想指定输出的位置，可以在第一行(DATA外面)指定`OUTPUT_DIR`
4. 运行`train.sh`即可
5. 需要查看tensorboard，修改`tensorboard.sh`里的位置运行该文件即可

## Test / Eval

1. 跟MVSNet很不一样，首先要下载DTU Rectified部分的数据集，有121G(sad...)，然后新建`Eval`，把这个文件夹方金曲，最后再把`Eval`这个文件夹放到`train/dtu/`里，跟`Cameras`, `Depths`, `Rectified`并列，最终的文件结构如下：

```
.
├── Cameras
│   ├── train/
│   ├── 00000000_cam.txt
│   ├── pair.txt
│   └── ...
├── Depths
│   └── scan1_train/
│   └── ...
├── Eval
│   └── Rectified
│       ├── scan1/
│       │   ├── rect_001_0_r5000.png
│       │   └── ...
│       └── ...
└── Rectified
    └── scan1/
    └── ...
```

2. 修改`.yaml`里`TEST.WEIGHT`的位置，运行`test.sh`即可

3. 注意运行`test.py`时必须指定只使用单GPU`CUDA_VISIBLE_DEVICES=0`，否则会报错`TypeError: forward() missing 3 required positional arguments: 'data_batch', 'img_scales', and 'inter_scales'`

4. 最终的输出结果会在数据集`Eval/`里

## fusion

【报错：Error: no kernel image is available for execution on the device】

- 原因: 主要是不同显卡的CUDA架构问题，根据[Matching CUDA arch and CUDA gencode for various NVIDIA architectures](https://arnon.dk/matching-sm-architectures-arch-and-gencode-for-various-nvidia-cards/)找到自己的GPU，再到fusibile的`CMakeLists.txt`中修改`-gencode arch=compute_70,code=sm_70`为自己的型号即可
- [参考解决方案](https://github.com/YoYo000/MVSNet/issues/28)

