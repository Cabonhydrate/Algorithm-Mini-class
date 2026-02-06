import torch
import argparse
import os
from torch.cuda.amp import autocast

# 导入模块
from models import ResNet18
from data.data_loader import get_cifar10_dataloaders
from utils import calculate_metrics, print_metrics
from config import config

# 直接用你给的 Logger（确保 logger.py 在工程根目录，或在 utils 里按路径改 import）
from utils.logger import Logger


@torch.no_grad()
def evaluate_model(model_path, device, log_dir="logs"):
    """
    评估模型，并将结果保存到 log_dir
    """
    logger = Logger(log_dir=log_dir)

    logger.log(f"使用设备: {device}")
    logger.log("")

    # 加载数据
    logger.log("加载CIFAR-10数据集...")
    _, test_loader = get_cifar10_dataloaders(
        batch_size=config["batch_size"],
        num_workers=config["data"]["num_workers"],
    )
    logger.log(f"测试集大小: {len(test_loader.dataset)}")
    logger.log("")

    # 初始化模型
    logger.log("初始化ResNet18模型...")
    model = ResNet18(num_classes=config["num_classes"], channels=config["channels"]).to(device)

    # 加载权重
    logger.log(f"加载模型权重从 {model_path}")
    try:
        state = torch.load(model_path, map_location=device, weights_only=True)
    except TypeError:
        # 兼容旧版 PyTorch（没有 weights_only 参数）
        state = torch.load(model_path, map_location=device)
    # 兼容两种保存方式：纯 state_dict 或 checkpoint dict
    if isinstance(state, dict) and "model" in state:
        model.load_state_dict(state["model"])
    else:
        model.load_state_dict(state)
    model.eval()
    logger.log("模型加载完成!")
    logger.log("")

    # 评估
    logger.log("开始评估模型...")
    logger.log("=" * 80)

    correct = 0
    total = 0
    all_preds = []
    all_targets = []

    use_amp = (device != "cpu")

    for batch_idx, (inputs, targets) in enumerate(test_loader):
        inputs = inputs.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        with autocast(enabled=use_amp):
            outputs = model(inputs)

        _, predicted = outputs.max(1)
        bs = targets.size(0)
        total += bs
        correct += predicted.eq(targets).sum().item()

        all_preds.append(predicted.detach().cpu())
        all_targets.append(targets.detach().cpu())

        if (batch_idx + 1) % 50 == 0:
            logger.log(f"Batch {batch_idx + 1}/{len(test_loader)}")

    all_preds = torch.cat(all_preds, dim=0) if len(all_preds) else torch.empty(0, dtype=torch.long)
    all_targets = torch.cat(all_targets, dim=0) if len(all_targets) else torch.empty(0, dtype=torch.long)

    metrics = calculate_metrics(all_preds, all_targets, config["num_classes"])

    logger.log("")
    logger.log("评估结果:")
    logger.log("=" * 80)
    print_metrics(metrics)  # 控制台保留原来的详细输出

    # ✅ 把 metrics 也写入文件（复用 Logger 的 JSON 保存逻辑）
    # 因为 evaluate 没有 epoch 概念，这里用 epoch=0 占位即可
    logger.log_evaluation(epoch=0, metrics=metrics)

    logger.log("评估总结:")
    logger.log(f"模型路径: {model_path}")
    logger.log(f'准确率: {metrics["accuracy"]:.4f}')
    logger.log(f'F1分数: {metrics["f1"]:.4f}')
    logger.log(f'精确率: {metrics["precision"]:.4f}')
    logger.log(f'召回率: {metrics["recall"]:.4f}')
    logger.log("=" * 80)

    logger.log(f"日志文件: {logger.get_log_file()}")
    logger.log(f"指标文件: {logger.get_metrics_file()}")


def main():
    parser = argparse.ArgumentParser(description="评估ResNet18模型")
    parser.add_argument("--model_path", type=str, required=True, help="模型路径")
    parser.add_argument("--log_dir", type=str, default="logs", help="日志保存目录")
    args = parser.parse_args()

    if not os.path.exists(args.model_path):
        print(f"模型路径不存在: {args.model_path}")
        return

    device = config["device"]
    evaluate_model(args.model_path, device, log_dir=args.log_dir)


if __name__ == "__main__":
    main()
