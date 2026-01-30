# 作业3：两层神经网络实现MNIST手写数字分类

## 项目概述

本项目实现了一个两层神经网络，用于MNIST手写数字分类任务。通过前向传播和反向传播算法，训练模型并评估其性能，同时生成训练过程的可视化结果。

## 项目结构

```
作业3/
├── results/             # 结果文件夹
│   ├── accuracy_curve.png       # 准确率曲线
│   ├── predictions_seed42.png   # 模型预测结果
│   └── training_loss.png        # 训练损失曲线
├── week1_homework3.py   # 主代码文件
└── README.md            # 项目说明文件
```

## 代码结构

### 1. Utils 工具类

提供神经网络所需的各种工具函数：
- `one_hot_encode(x, output_dim)`: 对标签进行独热编码
- `relu(x)`: ReLU激活函数
- `relu_derivative(x)`: ReLU激活函数的导数
- `softmax(x)`: Softmax激活函数
- `cross_entropy_loss(y_pred, y_true)`: 交叉熵损失函数

### 2. TwoLayerNetwork 两层神经网络类

实现了一个包含一个隐藏层的神经网络：
- `__init__(...)`: 初始化网络参数
- `load_mnist_data()`: 加载并预处理MNIST数据集
- `forward(x)`: 前向传播计算
- `backward(x, y_true, z1, a1, z2, a2)`: 反向传播计算梯度
- `update_param(dw1, db1, dw2, db2)`: 更新网络参数
- `predict(x)`: 对输入数据进行预测
- `accuracy(x, y_true_onehot)`: 计算模型准确率
- `train(x_train, x_test, y_train_onehot, y_test_onehot)`: 训练模型并记录训练过程

### 3. Visualizer 可视化类

用于生成训练过程和结果的可视化：
- `plot_history(history, save_dir)`: 绘制训练损失曲线和准确率曲线
- `show_predictions(network, x_test, y_test_onehot, n, seed, save_dir)`: 展示模型对测试数据的预测结果

## 技术实现

1. **数据处理**:
   - 加载MNIST数据集并进行归一化处理
   - 对标签进行独立热编码
   - 随机划分训练集和测试集

2. **网络结构**:
   - 输入层: 784个神经元（对应28x28像素的图像）
   - 隐藏层: 128个神经元，使用ReLU激活函数
   - 输出层: 10个神经元，使用Softmax激活函数（对应10个数字类别）

3. **训练方法**:
   - 使用小批量梯度下降法（batch size=64）
   - 学习率: 0.05
   - 训练轮数: 50轮
   - 每轮训练后计算训练准确率和测试准确率

4. **参数初始化**:
   - 使用He初始化方法初始化权重矩阵
   - 偏置项初始化为0

### 生成的结果文件

1. **training_loss.png**: 训练损失曲线，展示训练过程中损失值的变化
2. **accuracy_curve.png**: 准确率曲线，展示训练准确率和测试准确率的变化
3. **predictions_seed42.png**: 模型对测试数据的预测结果，展示12个测试样本的预测结果与真实标签

## 如何运行

1. 确保安装了必要的依赖库：
   - numpy
   - scikit-learn
   - matplotlib

2. 运行主脚本：
   ```bash
   python week1_homework3.py
   ```

3. 运行完成后，结果会保存在`results`文件夹中

## 性能评估

- **最终测试准确率**: 约97.51%
- **训练时间**: 取决于硬件配置，通常在几分钟内完成
- **模型大小**: 约0.98MB（参数数量：784×128 + 128×10 = 101,888）

## 依赖项

- Python 3.7+
- numpy
- scikit-learn
- matplotlib
