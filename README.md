# MVS
> Multi-View Stereo based on deep learning

## 📃论文列表

✨[MVS系列核心创新点 & 缺陷](https://github.com/doubleZ0108/MVS/blob/master/paper-summary.md)

|      | 简称/笔记    | 论文题目      | 出处(年份)     | 原文<br />代码| 推荐值                 |
| :--: | :-------------------: | ------------------ | -------------- | ---------- | :----: |
| 1    | [MVSNet](https://github.com/doubleZ0108/MVS/blob/master/Paper-Reading/MVSNet.md) | MVSNet: Depth Inference for Unstructured Multi-view Stereo   | ECCV 2018      | [paper](https://openaccess.thecvf.com/content_ECCV_2018/html/Yao_Yao_MVSNet_Depth_Inference_ECCV_2018_paper.html)<br /> [code](https://github.com/YoYo000/MVSNet) | ★★★★★ |
| 2    | [R-MVSNet](https://github.com/doubleZ0108/MVS/blob/master/Paper-Reading/R-MVSNet.md) | Recurrent MVSNet for High-resolution Multi-view Stereo Depth Inference | CVPR 2019      | [paper](https://openaccess.thecvf.com/content_CVPR_2019/html/Yao_Recurrent_MVSNet_for_High-Resolution_Multi-View_Stereo_Depth_Inference_CVPR_2019_paper.html)<br />[code](https://github.com/YoYo000/MVSNet) | ★★★ |
| 3    | [Point-MVSNet](https://github.com/doubleZ0108/MVS/blob/master/Paper-Reading/PointMVSNet.md) | Point-based multi-view stereo network                        | ICCV 2019 oral | [paper](https://openaccess.thecvf.com/content_ICCV_2019/html/Chen_Point-Based_Multi-View_Stereo_Network_ICCV_2019_paper.html)<br />[code](https://github.com/callmeray/PointMVSNet) | ★★★★ |
| 4    | [P-MVSNet](https://github.com/doubleZ0108/MVS/blob/master/Paper-Reading/P-MVSNet.md) | P-MVSNet: Learning Patch-wise Matching Confidence Aggregation for Multi-View Stereo | ICCV 2019      | [paper](https://openaccess.thecvf.com/content_ICCV_2019/html/Luo_P-MVSNet_Learning_Patch-Wise_Matching_Confidence_Aggregation_for_Multi-View_Stereo_ICCV_2019_paper.html) | ★ |
| 5    | [CVP-MVSNet](https://github.com/doubleZ0108/MVS/blob/master/Paper-Reading/CVP-MVSNet.md) | Cost Volume Pyramid Based Depth Inference for Multi-View Stereo | CVPR 2020 oral | [paper](https://openaccess.thecvf.com/content_CVPR_2020/html/Yang_Cost_Volume_Pyramid_Based_Depth_Inference_for_Multi-View_Stereo_CVPR_2020_paper.html)<br />[code](https://github.com/JiayuYANG/CVP-MVSNet) | ★★★★★ |
| 6 | [RayNet](https://github.com/doubleZ0108/MVS/blob/master/Paper-Reading/RayNet.md) | RayNet: Learning Volumetric 3D Reconstruction with Ray Potentials | CVPR 2018 oral | [paper](https://openaccess.thecvf.com/content_cvpr_2018/html/Paschalidou_RayNet_Learning_Volumetric_CVPR_2018_paper.html)<br />[code](https://github.com/paschalidoud/raynet) | ★★★☆ |
| 7 | [AttMVSNet](https://github.com/doubleZ0108/MVS/blob/master/Paper-Reading/AttMVSNet.md) | Attention-Aware Multi-View Stereo                            | CVPR 2020      | [paper](https://openaccess.thecvf.com/content_CVPR_2020/html/Luo_Attention-Aware_Multi-View_Stereo_CVPR_2020_paper.html) | ★☆ |
| 8 | [CasMVSNet](https://github.com/doubleZ0108/MVS/blob/master/Paper-Reading/CasMVSNet.md) | Cascade Cost Volume for High-Resolution Multi-View Stereo and Stereo Matching | CVPR 2020 oral | [paper](https://openaccess.thecvf.com/content_CVPR_2020/html/Gu_Cascade_Cost_Volume_for_High-Resolution_Multi-View_Stereo_and_Stereo_Matching_CVPR_2020_paper.html)<br />[code](https://github.com/alibaba/cascade-stereo) | ★★★★★ |
| 9 | [PatchmatchNet](https://github.com/doubleZ0108/MVS/blob/master/Paper-Reading/PatchmatchNet.md) | PatchmatchNet: Learned Multi-View Patchmatch Stereo          | CVPR 2021oral  | [paper](https://openaccess.thecvf.com/content/CVPR2021/html/Wang_PatchmatchNet_Learned_Multi-View_Patchmatch_Stereo_CVPR_2021_paper.html)<br />[code](https://github.com/FangjinhuaWang/PatchmatchNet) | ★★★☆ |

## 🔬项目/代码

- fusibile深度图融合点云 [![fusibile env setup](https://img.shields.io/badge/🧪_环境配置-fusibile-yellow)](https://github.com/doubleZ0108/MVS/blob/master/fusibile/setup.md)
- MVSNet [![MVSNet env setup](https://img.shields.io/badge/🧪_环境配置-MVSNet-yellow)](https://github.com/doubleZ0108/MVS/blob/master/MVSNet/setup.md) [![MVSNet code doc](https://img.shields.io/badge/🔨_代码笔记-MVSNet-9cf)](https://github.com/doubleZ0108/MVS/blob/master/MVSNet/code.md)
- MVSNet_pl [![MVSNet_pl env setup](https://img.shields.io/badge/🧪_环境配置-MVSNet__pl-yellow)](https://github.com/doubleZ0108/MVS/blob/master/MVSNet_pl/setup.md) [![MVSNet_pl code doc](https://img.shields.io/badge/🔨_代码笔记-MVSNet__pl-9cf)](https://github.com/doubleZ0108/MVS/blob/master/MVSNet_pl/code.md)
    > pytorch-lightning version of MVSNet
- PointMVSNet [![PointMVSNet env setup](https://img.shields.io/badge/🧪_环境配置-PointMVSNet-yellow)](https://github.com/doubleZ0108/MVS/blob/master/PointMVSNet/setup.md)
- CVP-MVSNet [![CVP-MVSNet env setup](https://img.shields.io/badge/🧪_环境配置-CVP__MVSNet-yellow)](https://github.com/doubleZ0108/MVS/blob/master/CVP-MVSNet/setup.md) [![CVP-MVSNet code doc](https://img.shields.io/badge/🔨_代码笔记-CVP__MVSNet-9cf)](https://github.com/doubleZ0108/MVS/blob/master/CVP-MVSNet/code.md)




