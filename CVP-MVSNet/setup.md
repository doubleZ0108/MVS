# CVP-MVSNet实验配置

## 环境配置

需要下载作者提供的数据集，跟MVSNet的数据集不是很一样，但都不大2G+1G

整体环境推荐不要复用MVSNet，主要是PyTorch版本比较新，再依照requirements安装下没安过的库即可

> 如果opencv-python在conda中无法安装成功，就直接用pip安装，一样可以用的
> 
> 我最开始复用了MVSNet的cuda和torch环境，会在很多位置报错`RuntimeError: output with device cuda:0 and dtype Float doesn't match the desired device cuda:0 and dtype Double`，定位到报错位置将变量改为`xx.double()`进行格式转换会消除一些，但有些始终无法修改成功，所幸从头安装了

## Train

修改下`train.sh`里的配置，直接运行就可以，对GPU的消耗不是很大，迭代的速度也比较快

## Test / Eval

安排好数据集的位置就可以直接运行`sh eval.sh`进行测试即可，最终深度图结果会在`outputs_pretrained`里

一点需要注意的是，如果运行时报错则需要指定一个GPU: `CUDA_VISIBLE_DEVICES=0`

eval之后会在`outputs`中生成每个场景的`confidence`和`depth_est`

## fusion

记得要使用python2进行fusion，我最后是重新安装了一个python2的conda环境成功的

安装好环境，链接一下fusibile的位置到`CVP-MVSNet/fusibile/`目录下

```shell
ln -s /media/public/yan1/doublez/realdoubleZ/Developer/MVS/fusibile/build/fusibile fusion/fusibile
```

fusion之后会在`outputs`里的`fusibile_fused`生成每个场景的文件夹，文件夹里就是那些一大堆内容，最核心的是
- `cams/`
- `images/`
- `consistencyCheck/final3d_model.ply`