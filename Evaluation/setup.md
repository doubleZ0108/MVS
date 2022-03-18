# DTU数据集定量测试环境搭建及最佳实践

## 环境搭建

首先在DTU官网下载定量测试的数据集（这部分之前师兄下过就没重新下）

![image.png](https://upload-images.jianshu.io/upload_images/12014150-0e79a26c640044ea.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

最终的数据结构应该是

```python
.
├── ObsMask
└── Points
    └── stl
```

然后下载官方matlab测试代码，我用的是MVSNet_PyTorch里evaluation中的matlab代码

主要修改的是两个文件`BaseEvalMain_web.m`和`ComputeStat_web.m`

其中需要修改的也主要是几个路径：

- dataPath：官网下的两个数据集的位置
- plyPath：所有你自己生成ply的位置
- resultsPath：测试结果的位置

再就是for循环里的`DataInName`的名字构成跟自己一致即可

## 官方Pipeline

1. 首先运行`BaseEvalMain_web.m`文件生成每个场景的`Eval.mat`文件
2. 再运行`ComputeStat_web.m`生成`TotalStat_Eval.mat`

## 最佳实践Pipeline

希望整体通过一个shell文件进行测试，每次测试不同的结果只需修改shell即可

Q1：shell运行matlab代码

需要注意的是不加.m后缀

```bash
matlab -nodesktop -nosplash -r "BaseEvalMain_web"
```

Q2：如何传递参数

在文件名字符串前添加参数即可 `plyPath='$PLYPATH';`

Q3：如何指定测试的场景

传递一个set字符串数组参数，并在matlab代码里通过`str2num()`转换为`UsedSets`数组

我的整体pipeline为：

1. 例如CVP-MVSNet再fusion之后会生成fusibile_fused文件夹，里面有每个场景的一大堆信息和需要的模型，首先通过一个python脚本把这些final3d_model.ply模型复制到一个文件夹，并以cvpmvsnet12.ply类似的名字命名
2. 修改shell文件里的前几个参数， PLYPATH换成刚刚存放所有模型的位置，METHOD为cvpmvsnet
3. 指定要测试的场景
4. 直接运行shell脚本即可
5. 进入漫长的等待中...(可以同时运行多个shell测试多组模型结果)