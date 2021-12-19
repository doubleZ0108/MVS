# MVSNet_pl代码结构

## Datasets

### dtu.py

依然是老几样

- build_metas()
- build_proj_mats()：相机参数提前乘好了
- getitem()

新加的在

- 读如的图片先进行了正则化(图像增强的一种)
- ref的相机参数取过inv了

---

## utils

- optimizer: sgd | adam | radam | ranger
- lr scheduler: steplr | cosine | poly
- ckpt的读取

### optimizer

提供了4种优化器

- RAdam
- PlainRAdam
- AdamW
- Ranger: RAdam + Lookahead

### warmup

optimizer中learning rate逐步warm-up策略

在实际使用过程中lr是越来越小的

### visualization

对深度图和概率图转3维进行彩色可视化

> 注意是对一张图片用的，要从batch数据中取一张出来
> 

### 其他

- metric
    - abs_error(): est和gt深度图的金额uduiwucha
    - acc_threshold(): 几mm的准确值
- 参数
    - 

---

## train

---

## mvsnet