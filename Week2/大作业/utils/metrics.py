import torch
import numpy as np


def calculate_metrics(preds, targets, num_classes=10):
    """
    计算分类评估指标
    
    Args:
        preds: 模型预测的类别
        targets: 真实类别
        num_classes: 类别数量
    
    Returns:
        metrics: 包含各种评估指标的字典
    """
    # 将数据转换为numpy数组
    preds_np = preds.cpu().numpy()
    targets_np = targets.cpu().numpy()
    
    # 计算混淆矩阵
    cm = np.zeros((num_classes, num_classes), dtype=int)
    for t, p in zip(targets_np, preds_np):
        cm[t, p] += 1
    
    # 计算TP、FN、FP、TN
    TP = cm.diagonal()
    FN = cm.sum(axis=1) - TP
    FP = cm.sum(axis=0) - TP
    TN = cm.sum() - (TP + FN + FP)
    
    # 计算准确率
    accuracy = TP.sum() / cm.sum()
    
    # 计算每个类别的精确率、召回率、F1分数
    class_precision = np.zeros(num_classes)
    class_recall = np.zeros(num_classes)
    class_f1 = np.zeros(num_classes)
    
    for i in range(num_classes):
        if TP[i] + FP[i] > 0:
            class_precision[i] = TP[i] / (TP[i] + FP[i])
        else:
            class_precision[i] = 0.0
        
        if TP[i] + FN[i] > 0:
            class_recall[i] = TP[i] / (TP[i] + FN[i])
        else:
            class_recall[i] = 0.0
        
        if class_precision[i] + class_recall[i] > 0:
            class_f1[i] = 2 * (class_precision[i] * class_recall[i]) / (class_precision[i] + class_recall[i])
        else:
            class_f1[i] = 0.0
    
    # 计算宏平均指标
    precision = np.mean(class_precision)
    recall = np.mean(class_recall)
    f1 = np.mean(class_f1)
    
    # 计算每个类别的指标
    class_metrics = {}
    for i in range(num_classes):
        class_metrics[f'class_{i}'] = {
            'TP': int(TP[i]),
            'FN': int(FN[i]),
            'FP': int(FP[i]),
            'TN': int(TN[i]),
            'precision': float(class_precision[i]),
            'recall': float(class_recall[i]),
            'f1': float(class_f1[i])
        }
    
    # 构建总指标字典
    metrics = {
        'accuracy': float(accuracy),
        'precision': float(precision),
        'recall': float(recall),
        'f1': float(f1),
        'confusion_matrix': cm.tolist(),
        'class_metrics': class_metrics
    }
    
    return metrics


def print_metrics(metrics):
    """
    打印评估指标
    
    Args:
        metrics: 包含评估指标的字典
    """
    print("\n" + "="*60)
    print("评估指标")
    print("="*60)
    print(f"准确率: {metrics['accuracy']:.4f}")
    print(f"精确率: {metrics['precision']:.4f}")
    print(f"召回率: {metrics['recall']:.4f}")
    print(f"F1分数: {metrics['f1']:.4f}")
    print("\n混淆矩阵:")
    for row in metrics['confusion_matrix']:
        print(' '.join(f'{cell:4d}' for cell in row))
    print("\n每个类别的指标:")
    for class_idx, class_metric in metrics['class_metrics'].items():
        print(f"\n{class_idx}:")
        print(f"  TP: {class_metric['TP']}, FN: {class_metric['FN']}, FP: {class_metric['FP']}, TN: {class_metric['TN']}")
        print(f"  精确率: {class_metric['precision']:.4f}, 召回率: {class_metric['recall']:.4f}, F1: {class_metric['f1']:.4f}")
    print("="*60 + "\n")
