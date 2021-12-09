# MVSNet实验配置

## 环境配置

1. 安装[Anaconda](https://zhuanlan.zhihu.com/p/414113053)
2. 创建conda环境 `conda create -n MVSNet python=3.6`，并激活`conda activate MVSNet`
3. 首先在conda中安装Pyorch
   - 首先通过`cat /usr/local/cuda/version.txt`查看CUDA版本 (比如我的是`CUDA Version 10.0.130`)
   - 然后在[pytorch官网](https://pytorch.org/get-started/previous-versions/)之前的版本中找

   <center><img src="https://upload-images.jianshu.io/upload_images/12014150-3e5a8e8fc6cbf657.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240" width="50%" /></center>

   - 安装这条命令即可
   - 安装完在命令行进入python，通过`import torch`和`torch.cuda.is_available()`测试是否安装成功，如果输出True则万事大吉
4. 安装OpenCV `conda install -c https://conda.anaconda.org/menpo opencv`
5. 安装tensorboard(由于版本和体系问题非常烦)，推荐按照我的这个方法来安，`conda install protobuf==3.19.1 tensorboardX==1.8 tensorboard==1.14.0 absl-py grpcio`
   > 首先制定protobuf的版本，然后把tensorboardX和tensorboard的版本都定死到这个版本，最后再补充安装一些可能会没有的库
   > 
   > 在我配tensorboard遇到的主要问题就是protobuf版本跟tensorboard不匹配，如果直接`conda install tensorboard`会直接安装最新的protobuf，后续怎么都降不下来

汇总的步骤大致如下，记得切换自己的版本

```
conda create -n MVSNet python=3.6

# PyTorch
# cuda 10.0
conda install pytorch==1.2.0 torchvision==0.4.0 cudatoolkit=10.0 -c pytorch

# opencv
conda install -c https://conda.anaconda.org/menpo opencv

conda install -c conda-forge plyfile
conda install scipy scikit-learn matplotlib Pillow tqdm 

# tensorboard(注意要换源)
conda install protobuf==3.19.1 tensorboardX==1.8 tensorboard==1.14.0 absl-py grpcio
```

最终通过一个简单的测试看看自己的环境是否可用

```
import torch
torch.cuda.is_available()

import cv2
import tensorboardX
```

## Train

## Test / Eval