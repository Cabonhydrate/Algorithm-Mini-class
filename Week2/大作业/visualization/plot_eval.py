import os
import json
import argparse

import numpy as np
import matplotlib.pyplot as plt


def _ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)


def _to_np(x):
    # confusion_matrix 可能是 list[list[int]] / list[int] 等
    return np.array(x)


def plot_confusion_matrix(cm, class_names, save_path, normalize=False):
    cm = _to_np(cm)
    if normalize:
        with np.errstate(divide="ignore", invalid="ignore"):
            cm = cm / cm.sum(axis=1, keepdims=True)
            cm = np.nan_to_num(cm)

    plt.figure()
    plt.imshow(cm)
    plt.title("Confusion Matrix" + (" (Normalized)" if normalize else ""))
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.xticks(range(len(class_names)), class_names, rotation=45, ha="right")
    plt.yticks(range(len(class_names)), class_names)

    # 标注数值（可选：矩阵很大时会密）
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            val = cm[i, j]
            txt = f"{val:.2f}" if normalize else f"{int(val)}"
            plt.text(j, i, txt, ha="center", va="center", fontsize=8)

    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()


def plot_per_class_bars(per_class, class_names, save_path, key):
    """
    per_class: dict like {"class_0": {"precision":..., "recall":..., "f1":...}, ...}
    key: "precision"/"recall"/"f1"
    """
    vals = []
    for i in range(len(class_names)):
        k = f"class_{i}"
        vals.append(per_class.get(k, {}).get(key, 0.0))

    plt.figure()
    plt.bar(range(len(class_names)), vals)
    plt.title(f"Per-class {key.upper()}")
    plt.xlabel("Class")
    plt.ylabel(key)
    plt.xticks(range(len(class_names)), class_names, rotation=45, ha="right")
    plt.ylim(0.0, 1.0)
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()


def plot_training_curves(training_list, out_dir):
    """
    training_list: metrics["training"] list of dicts with epoch/loss/accuracy
    """
    if not training_list:
        return

    epochs = [d.get("epoch", i + 1) for i, d in enumerate(training_list)]
    losses = [d.get("loss", 0.0) for d in training_list]
    accs = [d.get("accuracy", 0.0) for d in training_list]

    plt.figure()
    plt.plot(epochs, losses)
    plt.title("Training Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "train_loss.png"), dpi=200)
    plt.close()

    plt.figure()
    plt.plot(epochs, accs)
    plt.title("Training Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.ylim(0.0, 1.0)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "train_acc.png"), dpi=200)
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--metrics_json", required=True, help="logs/metrics_*.json 路径")
    parser.add_argument("--out_dir", default="visualization/outputs", help="图片输出目录")
    parser.add_argument("--num_classes", type=int, default=10, help="类别数（CIFAR10=10）")
    parser.add_argument("--class_names", default="", help="可选：逗号分隔的类别名，不填则用 class_0..")
    args = parser.parse_args()

    _ensure_dir(args.out_dir)

    with open(args.metrics_json, "r", encoding="utf-8") as f:
        m = json.load(f)

    # 1) 训练曲线（如果 metrics.json 里有 training）
    training_list = m.get("training", [])
    plot_training_curves(training_list, args.out_dir)

    # 2) 评估结果：取 evaluation 里最后一次
    eval_list = m.get("evaluation", [])
    if not eval_list:
        raise RuntimeError("metrics.json 里没有 evaluation 记录。请先运行 evaluate.py 生成评估日志。")

    last_eval = eval_list[-1].get("metrics", {})

    cm = last_eval.get("confusion_matrix", None)
    per_class = last_eval.get("per_class_metrics", {})  # 取决于你 calculate_metrics 返回结构

    # 类别名
    if args.class_names.strip():
        class_names = [x.strip() for x in args.class_names.split(",")]
    else:
        class_names = [f"class_{i}" for i in range(args.num_classes)]

    # 3) 混淆矩阵图
    if cm is not None:
        plot_confusion_matrix(
            cm, class_names, os.path.join(args.out_dir, "confusion_matrix.png"), normalize=False
        )
        plot_confusion_matrix(
            cm, class_names, os.path.join(args.out_dir, "confusion_matrix_norm.png"), normalize=True
        )

    # 4) 每类指标柱状图（如果存在 per_class_metrics）
    # 你的 print 输出里确实有每类 precision/recall/f1，所以大概率 calculate_metrics 也返回了
    if isinstance(per_class, dict) and len(per_class) > 0:
        plot_per_class_bars(per_class, class_names, os.path.join(args.out_dir, "per_class_precision.png"), "precision")
        plot_per_class_bars(per_class, class_names, os.path.join(args.out_dir, "per_class_recall.png"), "recall")
        plot_per_class_bars(per_class, class_names, os.path.join(args.out_dir, "per_class_f1.png"), "f1")

    # 5) 汇总写个文本
    summary_path = os.path.join(args.out_dir, "summary.txt")
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write(f"metrics_json: {args.metrics_json}\n")
        for k in ["accuracy", "precision", "recall", "f1"]:
            if k in last_eval:
                f.write(f"{k}: {last_eval[k]:.4f}\n")

    print(f"[OK] 输出完成：{args.out_dir}")


if __name__ == "__main__":
    main()
