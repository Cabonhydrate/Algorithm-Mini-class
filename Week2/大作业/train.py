import os
import random
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.amp import autocast, GradScaler
import glob

# 导入你的模块（保持工程结构不变）
from models import ResNet18
from data.data_loader import get_cifar10_dataloaders
from utils import Logger
from config import config



def set_seed(seed: int = 42, deterministic: bool = False):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # CIFAR10 输入尺寸固定，benchmark 通常更快
    cudnn.benchmark = not deterministic
    cudnn.deterministic = deterministic


def train_epoch(model, train_loader, criterion, optimizer, device, scaler: GradScaler):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for batch_idx, (inputs, targets) in enumerate(train_loader):
        inputs = inputs.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        # 使用新的autocast方式以避免弃用警告
        if device != "cpu":
            with autocast('cuda'):
                outputs = model(inputs)
                loss = criterion(outputs, targets)
        else:
            outputs = model(inputs)
            loss = criterion(outputs, targets)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        # 更严谨的统计：按样本数加权
        bs = inputs.size(0)
        running_loss += loss.item() * bs

        _, predicted = outputs.max(1)
        total += bs
        correct += predicted.eq(targets).sum().item()

        if (batch_idx + 1) % 100 == 0:
            print(f"Batch {batch_idx + 1}/{len(train_loader)}, Loss: {loss.item():.4f}")

    avg_loss = running_loss / max(total, 1)
    accuracy = correct / max(total, 1)
    return avg_loss, accuracy


def save_checkpoint(path, epoch, model, optimizer, scheduler, scaler, best_train_acc):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict() if scheduler is not None else None,
            "scaler": scaler.state_dict() if scaler is not None else None,
            "best_train_acc": best_train_acc,
        },
        path,
    )
    
def prune_checkpoints(save_dir: str, pattern: str, keep_last: int):
    """
    删除多余的周期性 checkpoint，仅保留最近 keep_last 个
    pattern 例子: "ckpt_epoch_*.pth"
    """
    paths = sorted(
        glob.glob(os.path.join(save_dir, pattern)),
        key=os.path.getmtime
    )
    if keep_last <= 0:
        return
    for p in paths[:-keep_last]:
        try:
            os.remove(p)
        except OSError:
            pass


def try_resume(resume_path, model, optimizer, scheduler, scaler, device):
    """
    自动从 checkpoints/last.pth 恢复（如果存在）
    """
    if not resume_path or not os.path.exists(resume_path):
        return 0, 0.0

    ckpt = torch.load(resume_path, map_location=device, weights_only=True)
    model.load_state_dict(ckpt["model"])
    optimizer.load_state_dict(ckpt["optimizer"])

    if scheduler is not None and ckpt.get("scheduler") is not None:
        scheduler.load_state_dict(ckpt["scheduler"])

    if scaler is not None and ckpt.get("scaler") is not None:
        scaler.load_state_dict(ckpt["scaler"])

    start_epoch = int(ckpt.get("epoch", 0))
    best_train_acc = float(ckpt.get("best_train_acc", 0.0))
    return start_epoch, best_train_acc


def main():
    set_seed(config.get("seed", 42), deterministic=config.get("deterministic", False))

    logger = Logger(config["log_dir"])
    logger.log("配置信息:")
    for k, v in config.items():
        logger.log(f"{k}: {v}")
    logger.log("")

    # Data
    logger.log("加载CIFAR-10数据集...")
    train_loader, _ = get_cifar10_dataloaders(
        batch_size=config["batch_size"],
        num_workers=config["data"]["num_workers"],
    )
    logger.log(f"训练集大小: {len(train_loader.dataset)}")
    logger.log("")

    # Model
    device = config["device"]
    model = ResNet18(num_classes=config["num_classes"], channels=config["channels"]).to(device)
    logger.log(f"模型已加载到 {device}")

    logger.log("")

    # Loss / Optim / Scheduler
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(
        model.parameters(),
        lr=config["learning_rate"],
        momentum=config["momentum"],
        weight_decay=config["weight_decay"],
    )
    scheduler = optim.lr_scheduler.MultiStepLR(
        optimizer,
        milestones=config["lr_scheduler"]["milestones"],
        gamma=config["lr_scheduler"]["gamma"],
    )

    # 使用新的GradScaler初始化方式以避免弃用警告
    if device != "cpu":
        scaler = GradScaler('cuda')
    else:
        scaler = GradScaler('cpu')

    # Checkpoints
    save_dir = "checkpoints"
    os.makedirs(save_dir, exist_ok=True)
    last_ckpt_path = os.path.join(save_dir, "last.pth")

    # Resume
    start_epoch, best_train_acc = try_resume(
        resume_path=last_ckpt_path if config.get("auto_resume", True) else None,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        scaler=scaler,
        device=device,
    )
    if start_epoch > 0:
        logger.log(f"已从 {last_ckpt_path} 恢复：epoch={start_epoch}, best_train_acc={best_train_acc:.4f}")
        logger.log("")

    logger.log("开始训练...")
    logger.log("=" * 80)

    for epoch in range(start_epoch + 1, config["epochs"] + 1):
        logger.log(f"Epoch {epoch}/{config['epochs']}")
        logger.log("-" * 80)

        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device, scaler)

        # 记录
        logger.log_training(epoch, train_loss, train_acc)

        # “best” 这里只能基于 train 指标（真正 best 你用 evaluate.py 再挑）
        if train_acc > best_train_acc:
            best_train_acc = train_acc
            best_path = os.path.join(save_dir, "best_train.pth")
            torch.save(model.state_dict(), best_path)
            logger.log(f"更新 best_train 到 {best_path}")

        # 周期保存（用于回溯），并只保留最近 K 份
        save_every = config.get("save_every", 10)          # 每10轮存一次
        keep_last = config.get("keep_last_ckpts", 5)       # 只保留最近5份

        if save_every > 0 and (epoch % save_every == 0):
            ckpt_path = os.path.join(save_dir, f"ckpt_epoch_{epoch}.pth")
            save_checkpoint(ckpt_path, epoch, model, optimizer, scheduler, scaler, best_train_acc)
            prune_checkpoints(save_dir, "ckpt_epoch_*.pth", keep_last)
            logger.log(f"保存周期 checkpoint 到 {ckpt_path}（保留最近 {keep_last} 份）")

        # 每轮都保存 last（断点续训用）
        save_checkpoint(
            last_ckpt_path,
            epoch,
            model,
            optimizer,
            scheduler,
            scaler,
            best_train_acc,
        )

        scheduler.step()
        logger.log(f"学习率更新为: {optimizer.param_groups[0]['lr']}")
        logger.log("")

    logger.log("训练完成!")
    logger.log(f"best_train_acc: {best_train_acc:.4f}")
    logger.log("=" * 80)


if __name__ == "__main__":
    main()
