# ResNet18 CIFAR-10 训练项目

本项目使用ResNet18模型在CIFAR-10数据集上进行从头训练，并实现了详细的评估指标计算和日志记录功能。

## 训练结果

### 最佳模型性能

- **准确率**：0.8737
- **精确率**：0.8736
- **召回率**：0.8737
- **F1分数**：0.8735

### 评估输出

评估过程中会显示详细的指标，包括：
- **混淆矩阵**：显示每个类别的预测情况
- **总体指标**：准确率、精确率、召回率、F1分数
- **每个类别的指标**：TP、FN、FP、TN、精确率、召回率、F1分数

### 输出文件

训练和评估过程会生成以下输出文件（我把生成的输出文件也都上传了，老师可以直接查看）：

#### 日志文件（`logs/`目录）
- `training_20260206_120236.log`：训练过程日志
- `evaluating_20260206_132455.log`：评估过程日志
- `metrics_20260206_120236.json`：训练评估指标（JSON格式）
- `metrics_eval_20260206_132455.json`：模型评估结果指标（JSON格式）

#### 模型权重（`checkpoints/`目录）
- `best_train.pth`：最佳模型权重
- `last.pth`：最后一轮模型权重
- `ckpt_epoch_*.pth`：周期性保存的模型权重（每10轮）

#### 可视化结果（`visualization/outputs/`目录）
- `confusion_matrix.png`：混淆矩阵可视化
- `confusion_matrix_norm.png`：归一化混淆矩阵可视化
- `summary.txt`：评估结果摘要

## 项目结构

```
大作业/
├── train.py              # 训练脚本
├── evaluate.py           # 评估脚本
├── README.md             # 项目说明
├── models/               # 模型目录
│   ├── __init__.py
│   └── ResNet18.py       # ResNet18模型定义
├── data/                 # 数据目录
│   ├── data_loader.py    # 数据加载器
│   └── cifar-10-batches-py/  # CIFAR-10数据集
├── utils/                # 工具目录
│   ├── __init__.py
│   ├── metrics.py        # 评估指标计算
│   └── logger.py         # 日志记录
├── config/               # 配置目录
│   ├── __init__.py
│   └── config.py         # 配置文件
├── logs/                 # 日志文件
├── checkpoints/          # 模型权重文件
└── visualization/        # 可视化工具
    ├── __init__.py
    ├── plot_eval.py      # 评估结果可视化
    └── outputs/          # 可视化输出
```

## 功能说明

1. **数据加载**：自动下载并预处理CIFAR-10数据集，支持数据增强
2. **模型训练**：使用ResNet18模型从头训练，支持学习率调度
3. **评估指标**：计算详细的评估指标，包括：
   - TP、FN、FP、TN
   - 准确率（Accuracy）
   - 精确率（Precision）
   - 召回率（Recall）
   - F1分数
   - 混淆矩阵
4. **日志记录**：将训练过程和评估结果保存到日志文件
5. **模型保存**：自动保存最佳模型权重
6. **结果可视化**：生成混淆矩阵等可视化结果

## 环境要求

- Python 3.x
- PyTorch
- NumPy

## 使用方法

### 1. 训练模型

运行训练脚本，开始训练ResNet18模型：

```bash
python train.py
```

训练过程中会自动：
- 下载并预处理CIFAR-10数据集
- 训练模型100个epoch
- 每轮评估模型性能
- 保存最佳模型权重
- 记录训练和评估过程到日志文件

### 2. 评估模型

使用评估脚本评估训练好的模型：

```bash
python evaluate.py --model_path checkpoints/best_train.pth
```

### 3. 可视化评估结果

使用可视化脚本生成评估结果的可视化：

```bash
python visualization/plot_eval.py --metrics_json logs/metrics_eval_20260206_132455.json
```

### 4. 配置参数

可以在`config/config.py`文件中修改训练参数：

- `epochs`：训练轮数
- `batch_size`：批次大小
- `learning_rate`：学习率
- `lr_scheduler`：学习率调度策略
- 等其他参数



## 评估指标说明

在评估时，会显示以下指标：

1. **混淆矩阵**：显示每个类别的预测情况
2. **总体指标**：准确率、精确率、召回率、F1分数
3. **每个类别的指标**：
   - TP（真正例）
   - FN（假负例）
   - FP（假正例）
   - TN（真负例）
   - 精确率
   - 召回率
   - F1分数

## 注意事项

1. 首次运行会自动下载CIFAR-10数据集，可能需要一些时间
2. 训练过程中会占用较多的GPU内存（如果使用GPU）
3. 日志文件和模型权重会占用一定的磁盘空间
4. 评估结果的可视化需要运行可视化脚本

