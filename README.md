# README

## 项目概述

本项目旨在实现基于 U-Net 算法的 CT 肿瘤图像分割。通过训练和测试 U-Net 模型，实现对 CT 图像中肿瘤区域的自动分割。

## 文件夹结构

- **train**：存储患者的 CT 肿瘤影像文件夹。
- **label**：存储与 train 文件夹对应的肿瘤掩膜。
- **evaluation**：用于存放网络测试的结果。
- **weights**：用于保存训练好的网络权重。
- **u_net.py**：项目的主执行文件，包含了实现 U-Net 算法的代码。

## 环境配置

### 下载 Conda

推荐使用 Conda 管理项目环境。可以从以下链接下载 Conda：

[Conda 官方下载地址](https://anaconda.org/anaconda/conda)

或者使用清华镜像源：

[清华镜像源](https://pypi.tuna.tsinghua.edu.cn/simple/)

如果对安装conda过程有疑问，可以参考 B 站教程：

[PyTorch 深度学习快速入门教程](https://www.bilibili.com/video/BV1hE411t7RN/?spm_id_from=333.337.search-card.all.click&vd_source=a02c88013ffaa4f9661de7f810e6dfcc)

### 安装 TensorFlow

1. **判断是否具备独立显卡**
   - 如果没有独立显卡，请使用以下命令安装 TensorFlow CPU 版本：
     ```bash
     pip install tensorflow
     ```
   - 如果有独立显卡，请注意确保 CUDA 和 cuDNN 的版本与 TensorFlow 兼容。具体版本匹配关系可以参考以下教程：
     [TensorFlow-gpu 保姆级安装教程](https://blog.csdn.net/weixin_43412762/article/details/129824339)

2. **配置 GPU 环境（无独立显卡可忽略）**
   - 根据上述教程安装 TensorFlow GPU 版本：
     ```bash
     pip install tensorflow-gpu==2.7.0 -i https://pypi.mirrors.ustc.edu.cn/simple
     ```

3. **解决依赖问题**
   - 如果在安装过程中遇到依赖问题，可以使用以下命令安装指定版本的 protobuf：
     ```bash
     pip install protobuf==3.19.0 -i https://pypi.tuna.tsinghua.edu.cn/simple --trusted-host pypi.tuna.tsinghua.edu.cn
     ```

4. **测试 TensorFlow 安装**
   - 进入 Python 环境，运行以下代码测试 TensorFlow 是否安装成功：
     ```python
     import tensorflow as tf
     print(tf.__version__)
     print(tf.test.gpu_device_name())
     ```

### 其他建议

如果想使用服务器运行项目，可以租用相关服务器，并使用 FileZilla 传输文件。

## 运行项目

1. **激活 Conda 环境**
   - 如果使用 Conda 管理环境，请确保已激活相应的项目环境。

2. **运行主文件**
   - 确保 `u_net.py` 文件中的 `unet.train` 函数未被注释，然后运行主文件。

3. **性能参考**
   - 作者在使用不同硬件环境下的运行性能参考：
     - 笔记本 CPU（12 代 i7）：平均一个 epoch 运行 270 秒。
     - 笔记本 GPU（3060）：平均一个 epoch 运行 36 秒。
     - 云服务器 GPU（3090）：平均一个 epoch 运行 10 秒。

## 注意事项

如果在运行过程中遇到问题，可以根据报错信息使用 Conda 或 pip 命令安装相关依赖包。若下载速度较慢，可以考虑开启 VPN。

希望本项目对您的研究有所帮助！
