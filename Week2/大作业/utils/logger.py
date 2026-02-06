import os
import time
import json
from datetime import datetime


class Logger:
    """
    日志记录器，用于记录训练和评估过程
    """
    
    def __init__(self, log_dir='logs'):
        """
        初始化日志记录器
        
        Args:
            log_dir: 日志保存目录
        """
        self.log_dir = log_dir
        os.makedirs(self.log_dir, exist_ok=True)
        
        # 创建日志文件
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.log_file = os.path.join(self.log_dir, f'training_{timestamp}.log')
        self.metrics_file = os.path.join(self.log_dir, f'metrics_{timestamp}.json')
        
        # 初始化日志数据
        self.logs = []
        self.metrics = {
            'training': [],
            'evaluation': []
        }
        
        # 写入日志头
        with open(self.log_file, 'w', encoding='utf-8') as f:
            f.write(f'训练日志 - {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}\n')
            f.write('=' * 80 + '\n')
    
    def log(self, message):
        """
        记录日志信息
        
        Args:
            message: 日志信息
        """
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        log_entry = f'[{timestamp}] {message}'
        
        # 打印到控制台
        print(log_entry)
        
        # 写入日志文件
        with open(self.log_file, 'a', encoding='utf-8') as f:
            f.write(log_entry + '\n')
        
        # 保存到内存
        self.logs.append(log_entry)
    
    def log_training(self, epoch, loss, accuracy):
        """
        记录训练过程
        
        Args:
            epoch:  epoch数
            loss: 损失值
            accuracy: 准确率
        """
        message = f'Epoch {epoch}: Loss = {loss:.4f}, Accuracy = {accuracy:.4f}'
        self.log(message)
        
        # 保存到metrics
        self.metrics['training'].append({
            'epoch': epoch,
            'loss': loss,
            'accuracy': accuracy,
            'timestamp': datetime.now().isoformat()
        })
        
        # 保存metrics到文件
        self._save_metrics()
    
    def log_evaluation(self, epoch, metrics):
        """
        记录评估过程
        
        Args:
            epoch: epoch数
            metrics: 评估指标
        """
        # 记录主要指标
        message = f'Evaluation - Epoch {epoch}: Accuracy = {metrics["accuracy"]:.4f}, F1 = {metrics["f1"]:.4f}, Precision = {metrics["precision"]:.4f}, Recall = {metrics["recall"]:.4f}'
        self.log(message)
        
        # 记录详细指标
        self.log('详细评估指标:')
        self.log(f'混淆矩阵: {metrics["confusion_matrix"]}')
        
        # 保存到metrics
        self.metrics['evaluation'].append({
            'epoch': epoch,
            'metrics': metrics,
            'timestamp': datetime.now().isoformat()
        })
        
        # 保存metrics到文件
        self._save_metrics()
    
    def _save_metrics(self):
        """
        保存metrics到JSON文件
        """
        with open(self.metrics_file, 'w', encoding='utf-8') as f:
            json.dump(self.metrics, f, indent=2, ensure_ascii=False)
    
    def log_config(self, config):
        """
        记录配置信息
        
        Args:
            config: 配置字典
        """
        self.log('配置信息:')
        for key, value in config.items():
            self.log(f'{key}: {value}')
        self.log('')
    
    def get_log_file(self):
        """
        获取日志文件路径
        
        Returns:
            log_file: 日志文件路径
        """
        return self.log_file
    
    def get_metrics_file(self):
        """
        获取metrics文件路径
        
        Returns:
            metrics_file: metrics文件路径
        """
        return self.metrics_file
