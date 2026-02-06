import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader


def get_cifar10_dataloaders(batch_size=128, num_workers=4):
    """
    获取CIFAR-10数据集的数据加载器
    
    Args:
        batch_size: 批次大小
        num_workers: 数据加载的线程数
    
    Returns:
        train_loader: 训练数据加载器
        test_loader: 测试数据加载器
    """
    # 训练集的变换（包含数据增强）
    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])
    
    # 测试集的变换（仅归一化）
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])
    
    # 检查数据集是否已经存在
    import os
    dataset_exists = os.path.exists('./data/cifar-10-batches-py')
    
    # 加载训练集
    train_dataset = torchvision.datasets.CIFAR10(
        root='./data',
        train=True,
        download=not dataset_exists,
        transform=train_transform
    )
    
    # 加载测试集
    test_dataset = torchvision.datasets.CIFAR10(
        root='./data',
        train=False,
        download=not dataset_exists,
        transform=test_transform
    )
    
    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, test_loader


if __name__ == "__main__":
    # 测试数据加载器
    train_loader, test_loader = get_cifar10_dataloaders(batch_size=32)
    print(f"训练集批次数量: {len(train_loader)}")
    print(f"测试集批次数量: {len(test_loader)}")
    
    # 查看数据集中的类别
    classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
    print(f"CIFAR-10类别: {classes}")
