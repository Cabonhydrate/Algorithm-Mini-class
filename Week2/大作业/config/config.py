# 导入必要的库
import torch

# 训练配置
config = {
    # 模型配置
    'model': 'ResNet18',
    'num_classes': 10,
    'channels': 3,
    
    # 训练配置
    'epochs': 80,
    'batch_size': 128,
    'learning_rate': 0.1,
    'momentum': 0.9,
    'weight_decay': 5e-4,
    'lr_scheduler': {
        'type': 'MultiStepLR',
        'milestones': [40, 60, 75],
        'gamma': 0.1
    },
    
    # 数据配置
    'data': {
        'dataset': 'CIFAR10',
        'num_workers': 4,
        'pin_memory': True
    },
    
    # 设备配置
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
    
    # 日志配置
    'log_dir': 'logs',
    'save_every': 10,        # 每10个epoch保存一次周期checkpoint
    'keep_last_ckpts': 5,    # 周期checkpoint只保留最近5份

    
    # 评估配置
    'eval_interval': 1
}

# 确保配置的设备是可用的
if config['device'] == 'cuda' and not torch.cuda.is_available():
    print("CUDA不可用，将使用CPU")
    config['device'] = 'cpu'
