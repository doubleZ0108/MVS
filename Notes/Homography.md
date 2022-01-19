## 几何变换类型

- 保距变换 isometry
- 相似变换 similarity
- 仿射变换 affine
- 射影变换 projective -> homography

## What is Homography?

![image.png](https://upload-images.jianshu.io/upload_images/12014150-94d19bc74f1983f5.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

A **Homography** is a transformation ( a 3×3 matrix ) that maps the points in one image to the corresponding points in the other image.

$$
H=\left[\begin{array}{lll}h_{00} & h_{01} & h_{02} \\h_{10} & h_{11} & h_{12} \\h_{20} & h_{21} & h_{22}\end{array}\right]
$$

- homography只针对同一平面

## How to calculate a Homography ?

摄影变换的自由度为8，一对点能产生两个方程，共需要4对对应点，即可求取H矩阵；如超过4对，通过最小二乘法或RANSAC求取最优参数

### 理论推导

![image.png](https://upload-images.jianshu.io/upload_images/12014150-35803ff0b19696c5.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

```python
[U,S,V]=svd(A);
h=V(:,9);
H= reshape(h,3,3);

```

### 工程实践

If you have more than 4 corresponding points, it is even better. OpenCV will robustly estimate a homography that best fits all corresponding points.
Usually, these point correspondences are found automatically by matching features like SIFT or SURF between the images.

```python
'''
pts_src and pts_dst are numpy arrays of points
in source and destination images. We need at least
4 corresponding points.
'''
h, status = cv2.findHomography(pts_src, pts_dst)
'''
The calculated homography can be used to warp
the source image to destination. Size is the
size (width,height) of im_dst
'''
im_dst = cv2.warpPerspective(im_src, h, size)

```

## Application

- 图像矫正
- 图像扫描
- 虚拟广告牌

## Reference

- [Homography Examples using OpenCV ( Python / C ++ ) | Learn OpenCV](https://www.learnopencv.com/homography-examples-using-opencv-python-c/)
- [Opencv日常之Homography_liuphahaha的专栏-CSDN博客](https://blog.csdn.net/liuphahaha/article/details/50719275)
- 🌟homography变换推导：[单应性(homography)变换的推导 - flyinsky518 - 博客园](https://www.cnblogs.com/ml-cv/p/5871052.html)
- [单应性变换-Homography - 知乎](https://zhuanlan.zhihu.com/p/145170405)
- [平面标定（Homography变换） - 知乎](https://zhuanlan.zhihu.com/p/60482480)