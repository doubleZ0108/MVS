# CVP-MVSNet代码结构

## 数据集

【Train】

- Cameras：每个视点的相机信息和配对信息，跟MVSNet一致，删除了一些没用的东西，pair配对信息也完全一样
- Depth：深度图真值，跟MVSNet一致，不同点是没有mask图，训练的深度图也是128*160
- Rectified：训练用的图片，结构跟MVSNet是一样的，只不过分辨率从640*512 → 160*128小了很多（注意视点从1开始标号，光照从0开始标号）

> 图像缩小了4倍，但是相机参数没改？
> 

【Test】

- Cameras：跟Train相比相机外参没变，内参变大了10倍左右，但整体是跟MVSNet完全一样的
- Rectified：测试用的图片，尺寸与MVSNet一样是全尺寸1600*1200

> 有趣的是MVSNet图片格式是jpg，CVP格式是png，jpg一张700K，png一张2.6M
> 

## 参数

- 增加了`nscale`指定金字塔的层数
- `epoch`从16增大到了28
- `loss_function`有sl1和mse两个选项

## 具体模块

### train.py

- 网络输入：ref图，一组src图，内外参，深度假设范围
- 网络输出：depth_est_list
- 按照金字塔结构计算**loss求和后反传**
    - 每一层的depth_gt需要进行下采样
    - mask = depth_gt > 425 而不是预先像MVSNet一样处理好的

### Data模块

- `read_image()`：读test图片时会被裁剪为(1184, 1600