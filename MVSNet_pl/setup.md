# MVSNet_pl实验配置

## 环境配置

因为该仓库使用的是pytorch_lightning框架，因此当然不可以复用之前的conda环境啦，作者readme里已经说的很详细了

直接安装`requirements.txt`会报错pytorch lightning版本和torchvision版本冲突，因此我最终安装的是`torch==1.4.0 torchvision==0.5.0 pytorch-lightning==0.6.0`

然后安装Inplace-ABN `pip install git+https://github.com/mapillary/inplace_abn.git@v1.0.11`

## Train

使用一个shell脚本运行总是会报错找不到inplace_abn库，因此只能直接用命令行跑python指令

不过一直使用的是单卡进行训练（16G V100可以最大跑batch=3），多卡跑好像会有问题，issue里有人也在说这事

注意：跑几百step可能会自己断掉然后报错，找了一系列博客说是torch相关的版本问题