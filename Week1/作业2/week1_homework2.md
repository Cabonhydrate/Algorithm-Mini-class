# 作业二：神经网络反向传播实现说明

## 项目结构

本目录包含以下文件：

- **backward_coding.py**：神经网络反向传播算法的核心实现
- **overall_process.py**：完整神经网络类实现（包含前向传播、反向传播、训练等功能）

## backward_coding.py 详细说明

### 功能概述

`backward_coding.py` 实现了神经网络的反向传播算法，是深度学习中模型训练的核心部分。该文件主要负责计算网络中各层参数的梯度，为参数更新提供依据。

### 核心函数

#### 1. `_backward_single_layer(da, a_prev, w, z, activation, a_current=None, da_is_dz=False)`

- **功能**：计算单个层的反向传播梯度
- **参数**：
  - `da`：当前层激活值的梯度
  - `a_prev`：上一层的激活值（当前层的输入）
  - `w`：当前层的权重
  - `z`：当前层的线性输出
  - `activation`：当前层的激活函数
  - `a_current`：当前层的激活值
  - `da_is_dz`：是否直接提供dz
- **返回值**：
  - `dw`：权重梯度
  - `db`：偏置梯度
  - `da_prev`：传递给上一层的梯度

#### 2. `backward(y_true, caches)`

- **功能**：完整的反向传播过程，从输出层反向计算所有层的梯度
- **参数**：
  - `y_true`：真实标签
  - `caches`：前向传播过程中缓存的中间值
- **返回值**：
  - `grads`：各层参数的梯度字典

#### 3. `update_params(grads, learning_rate)`

- **功能**：根据梯度更新网络参数
- **参数**：
  - `grads`：各层参数的梯度
  - `learning_rate`：学习率

#### 4. `activation_derivative(activation, a=None, z=None)`

- **功能**：计算激活函数的导数
- **参数**：
  - `activation`：激活函数名称
  - `a`：激活值
  - `z`：线性输出
- **返回值**：
  - 激活函数的导数

### 支持的激活函数

- `sigmoid`：S型激活函数，常用于二分类输出层
- `relu`：修正线性单元，常用于隐藏层
- `tanh`：双曲正切函数，可用于隐藏层
- `none`/`linear`：线性激活函数，常用于回归任务
- `softmax`：softmax函数，仅支持在输出层使用

### 实现特点

1. **批量梯度计算**：考虑了批次大小对梯度的影响，通过除以样本数进行归一化
2. **链式法则**：严格按照链式法则实现反向传播
3. **激活函数导数**：针对不同激活函数实现了相应的导数计算
4. **分类任务优化**：对于分类任务（sigmoid/softmax激活）直接计算dz，提高效率
5. **错误处理**：对不支持的激活函数提供明确的错误信息

## overall_process.py 简要说明

`overall_process.py` 实现了一个完整的神经网络类，包含以下功能：

- 基础工具函数：各种激活函数、损失函数及其导数
- 神经网络类：支持自定义网络结构、激活函数和损失函数
- 前向传播：计算网络输出
- 反向传播：计算参数梯度
- 训练功能：支持批量/小批量/随机梯度下降
- 预测功能：根据训练好的模型进行预测

该文件提供了一个完整的神经网络训练框架。

## 简易使用示例

### 反向传播示例

```python
# 假设已有前向传播的缓存caches和真实标签y_true
# 计算梯度
grads = backward(y_true, caches)
# 更新参数
update_params(grads, learning_rate=0.01)
```

### 完整神经网络使用示例

```python
# 创建神经网络模型
model = NeuralNetwork(
    layer_dims=[784, 128, 64, 10],  # 输入层784，隐藏层128和64，输出层10
    activation_hidden="relu",        # 隐藏层使用ReLU
    activation_output="softmax",     # 输出层使用Softmax
    loss_type="cross_entropy"        # 使用交叉熵损失
)

# 训练模型
model.train(x_train, y_train, epochs=1000, learning_rate=0.01)

# 预测
predictions = model.predict(x_test)
```


## 注意事项

- 确保输入数据的维度正确：(特征维度, 样本数)
- 对于分类任务，标签应采用one-hot编码
- 选择合适的学习率，过大可能导致训练不稳定，过小可能导致收敛缓慢
- 对于深层网络，建议使用ReLU等激活函数以缓解梯度消失问题
