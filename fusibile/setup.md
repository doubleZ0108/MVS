# fusibile环境配置

```
mkdir build && cd build
cmake ..
make
```

进行编译即可，会自动导出后续fusibile被调用时的路径

一些坑记录在下面：

-----


【**运行转化点云时报错**：Error: no kernel image is available for execution on the device】

- **原因**：主要是不同显卡的CUDA架构问题，根据[Matching CUDA arch and CUDA gencode for various NVIDIA architectures](https://arnon.dk/matching-sm-architectures-arch-and-gencode-for-various-nvidia-cards/)找到自己的GPU，再到fusibile的`CMakeLists.txt`中修改`-gencode arch=compute_70,code=sm_70`为自己的型号即可

- **解决方案**：在`CMakeList`中将第10行中的`set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS};-O3 --use_fast_math --ptxas-options=-v -std=c++11 --compiler-options -Wall -gencode arch=compute_75,code=sm_75)`更改为正确的版本
    > 具体版本仍不太知道该如何查找
    > 
    > 参考：[Error: no kernel image is available for execution on the device · Issue #9 · kysucix/fusibile · GitHub](https://github.com/YoYo000/MVSNet/issues/28)

-----

【OpenCV报错】

```bash
CMake Error at CMakeLists.txt:4 (find_package):
  By not providing "FindOpenCV.cmake" in CMAKE_MODULE_PATH this project has
  asked CMake to find a package configuration file provided by "OpenCV", but
  CMake did not find one.

  Could not find a package configuration file provided by "OpenCV" with any
  of the following names:

    OpenCVConfig.cmake
    opencv-config.cmake

  Add the installation prefix of "OpenCV" to CMAKE_PREFIX_PATH or set
  "OpenCV_DIR" to a directory containing one of the above files.  If "OpenCV"
  provides a separate development package or SDK, be sure it has been
  installed.

-- Configuring incomplete, errors occurred!
```

- **原因**：OpenCV安装问题
- **解决办法**：重新安装OpenCV，[Ubuntu配置OpenCV终极解决方案](https://zhuanlan.zhihu.com/p/368573848)，可以通过[linux下查看opencv安装路径以及版本号](https://blog.csdn.net/zhenguo26/article/details/79627232)查看是否安装成功

-----

【报错：fatal error: GL/gl.h: No such file or directory】

- **原因**：系统中缺少OpenGl库
- **解决方案**：

```bash
apt-get install mesa-common-dev
apt-get install libgl1-mesa-dev
```

-----

【报错：**`#error -- unsupported GNU version! gcc versions later than 6 are not supported`**】

```
sudo update-alternatives --install /usr/bin/g++ g++ /usr/bin/g++-6 10
sudo update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-6 10
```