import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torch.optim.lr_scheduler import CosineAnnealingLR
import numpy as np
import matplotlib.pyplot as plt
import os
import json
from datetime import datetime
import pandas as pd

from dataset import FusionDataset
from model import FusionModel
from utils import evaluate

def main():

    # =======================
    # Config
    # =======================

    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    CSV_PATH = os.path.join(BASE_DIR, "datasets", "scenario33", "scenario33_processed.csv")
    RESULTS_DIR = os.path.join(BASE_DIR, "results")
    BATCH_SIZE = 32
    EPOCHS = 80
    LR = 1e-5
    NUM_CLASSES = 64
    BACKBONE = "resnet34"   # "resnet34" / "vit" / "swin"
    RADAR_BACKBONE = "resnet34"  # "resnet34" / "vit" / "swin"

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"{BACKBONE}_{run_timestamp}"
    run_dir = os.path.join(RESULTS_DIR, run_name)
    curves_dir = os.path.join(run_dir, "curves")
    os.makedirs(curves_dir, exist_ok=True)

    BEST_MODEL_PATH = os.path.join(run_dir, "best_model.pth")
    METRICS_CSV_PATH = os.path.join(run_dir, "epoch_metrics.csv")
    SUMMARY_JSON_PATH = os.path.join(run_dir, "summary.json")


    # =======================
    # Dataset
    # =======================

    full_dataset = FusionDataset(CSV_PATH, mode="train", strong_aug=True)

    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size

    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

    # 把 val dataset 的 transform 改为验证模式
    val_dataset.dataset.mode = "val"

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=2,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=2,
        pin_memory=True
    )


    # =======================
    # Model
    # =======================

    model = FusionModel(
        num_classes=NUM_CLASSES,
        backbone=BACKBONE,
        radar_backbone=RADAR_BACKBONE,
    ).to(DEVICE)


    # ===== 冻结 backbone 前几层（小数据推荐）=====
    if BACKBONE == "resnet34":
        for name, param in model.cnn.named_parameters():
            if "layer1" in name or "layer2" in name:
                param.requires_grad = False

    elif BACKBONE == "swin":
        for name, param in model.cnn.named_parameters():
            if "layers.0" in name or "layers.1" in name:
                param.requires_grad = False

    elif BACKBONE == "vit":
        for name, param in model.cnn.named_parameters():
            if "encoder.layers.0" in name:
                param.requires_grad = False


    # =======================
    # Loss & Optimizer
    # =======================

    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LR,
        weight_decay=1e-4
    )

    scheduler = CosineAnnealingLR(
        optimizer,
        T_max=EPOCHS
    )


    # =======================
    # Training Loop
    # =======================

    best_top1 = 0
    best_epoch = -1
    
    # 初始化记录list用于保存曲线与评估数据
    train_loss_history = []
    val_loss_history = []
    top1_history = []
    top3_history = []
    top5_history = []
    lr_history = []
    epochs_list = []
    metrics_records = []

    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0

        for image, geo, radar, label in train_loader:
            image = image.to(DEVICE)
            geo = geo.to(DEVICE)
            radar = radar.to(DEVICE)
            label = label.to(DEVICE)

            optimizer.zero_grad()

            output = model(image, geo, radar)
            loss = criterion(output, label)

            loss.backward()

            # ===== Gradient Clipping（防止 Transformer 爆炸）=====
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            optimizer.step()

            total_loss += loss.item()

        avg_train_loss = total_loss / len(train_loader)

        # =======================
        # Validation
        # =======================
        val_loss, top1, top3, top5 = evaluate(model, val_loader, DEVICE, criterion)

        scheduler.step()
        current_lr = optimizer.param_groups[0]["lr"]
        
        # 记录指标
        epochs_list.append(epoch + 1)
        train_loss_history.append(avg_train_loss)
        val_loss_history.append(val_loss)
        top1_history.append(top1)
        top3_history.append(top3)
        top5_history.append(top5)
        lr_history.append(current_lr)

        metrics_records.append({
            "epoch": epoch + 1,
            "train_loss": avg_train_loss,
            "val_loss": val_loss,
            "top1": top1,
            "top3": top3,
            "top5": top5,
            "lr": current_lr
        })

        print(f"Epoch {epoch+1}/{EPOCHS}")
        print(f"Train Loss: {avg_train_loss:.4f} | Val Loss: {val_loss:.4f}")
        print(f"Val Top1: {top1:.4f} | Top3: {top3:.4f} | Top5: {top5:.4f}")
        print(f"LR: {current_lr:.6e}")
        print("-" * 40)

        # 保存最佳模型
        if top1 > best_top1:
            best_top1 = top1
            best_epoch = epoch + 1
            torch.save(model.state_dict(), BEST_MODEL_PATH)
            print("🔥 Saved Best Model")


    print(f"Training Finished. Best Top1: {best_top1:.4f} at epoch {best_epoch}")

    # =======================
    # 保存评估数据
    # =======================
    metrics_df = pd.DataFrame(metrics_records)
    metrics_df.to_csv(METRICS_CSV_PATH, index=False)

    summary = {
        "run_name": run_name,
        "backbone": BACKBONE,
        "radar_backbone": RADAR_BACKBONE,
        "csv_path": CSV_PATH,
        "batch_size": BATCH_SIZE,
        "epochs": EPOCHS,
        "lr": LR,
        "device": str(DEVICE),
        "best_epoch": best_epoch,
        "best_top1": best_top1,
        "best_top3": float(metrics_df["top3"].max()),
        "best_top5": float(metrics_df["top5"].max()),
        "lowest_train_loss": float(metrics_df["train_loss"].min()),
        "lowest_val_loss": float(metrics_df["val_loss"].min()),
        "best_model_path": BEST_MODEL_PATH,
        "metrics_csv_path": METRICS_CSV_PATH
    }

    with open(SUMMARY_JSON_PATH, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(f"✅ Metrics saved to {METRICS_CSV_PATH}")
    print(f"✅ Summary saved to {SUMMARY_JSON_PATH}")
    
    
    # =======================
    # 绘制并保存更详细曲线
    # =======================

    # 1. Loss 对比曲线（Train vs Val）
    plt.figure(figsize=(11, 6))
    plt.plot(epochs_list, train_loss_history, "-", color="#1f77b4", linewidth=2.2, marker="o", markersize=4, label="Train Loss")
    plt.plot(epochs_list, val_loss_history, "-", color="#ff7f0e", linewidth=2.2, marker="s", markersize=4, label="Val Loss")
    if best_epoch > 0:
        plt.axvline(best_epoch, color="#2ca02c", linestyle="--", linewidth=1.6, label=f"Best Epoch = {best_epoch}")
    plt.xlabel("Epoch", fontsize=12)
    plt.ylabel("Loss", fontsize=12)
    plt.title("Training and Validation Loss", fontsize=14, fontweight="bold")
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=11)
    plt.tight_layout()
    loss_curve_path = os.path.join(curves_dir, "loss_curve_detailed.png")
    plt.savefig(loss_curve_path, dpi=300)
    print(f"✅ Loss curve saved to {loss_curve_path}")
    plt.close()

    # 2. Top-K Accuracy 曲线（标注最佳 Top-1）
    plt.figure(figsize=(11, 6))
    plt.plot(epochs_list, top1_history, "-", color="#1f77b4", linewidth=2.2, marker="o", label="Top-1", markersize=4)
    plt.plot(epochs_list, top3_history, "-", color="#2ca02c", linewidth=2.2, marker="s", label="Top-3", markersize=4)
    plt.plot(epochs_list, top5_history, "-", color="#d62728", linewidth=2.2, marker="^", label="Top-5", markersize=4)
    if best_epoch > 0:
        best_top1_value = top1_history[best_epoch - 1]
        best_top3_value = top3_history[best_epoch - 1]
        best_top5_value = top5_history[best_epoch - 1]
        plt.scatter([best_epoch], [best_top1_value], color="#1f77b4", s=55, zorder=6)
        plt.scatter([best_epoch], [best_top3_value], color="#2ca02c", s=55, zorder=6)
        plt.scatter([best_epoch], [best_top5_value], color="#d62728", s=55, zorder=6)
        plt.annotate(
            (
                f"Best Epoch={best_epoch}\n"
                f"Top-1={best_top1_value:.4f}\n"
                f"Top-3={best_top3_value:.4f}\n"
                f"Top-5={best_top5_value:.4f}"
            ),
            xy=(best_epoch, best_top1_value),
            xytext=(best_epoch + 1, min(1.0, best_top1_value + 0.08)),
            arrowprops=dict(arrowstyle="->", color="#333333", lw=1.2),
            fontsize=10
        )
    plt.xlabel("Epoch", fontsize=12)
    plt.ylabel("Accuracy", fontsize=12)
    plt.title("Validation Top-K Accuracy", fontsize=14, fontweight="bold")
    plt.ylim(0, 1.0)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=11, loc="lower right")
    plt.tight_layout()
    topk_curve_path = os.path.join(curves_dir, "topk_curve_detailed.png")
    plt.savefig(topk_curve_path, dpi=300)
    print(f"✅ Top-K curve saved to {topk_curve_path}")
    plt.close()

    # 3. 三联图：Loss / Top-1 / Learning Rate
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    axes[0].plot(epochs_list, train_loss_history, color="#1f77b4", linewidth=2, marker="o", markersize=3)
    axes[0].plot(epochs_list, val_loss_history, color="#ff7f0e", linewidth=2, marker="s", markersize=3)
    axes[0].set_title("Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(epochs_list, top1_history, color="#2ca02c", linewidth=2, marker="o", markersize=3)
    axes[1].set_title("Top-1 Accuracy")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].set_ylim(0, 1.0)
    axes[1].grid(True, alpha=0.3)

    axes[2].plot(epochs_list, lr_history, color="#9467bd", linewidth=2, marker="o", markersize=3)
    axes[2].set_title("Learning Rate")
    axes[2].set_xlabel("Epoch")
    axes[2].set_ylabel("LR")
    axes[2].grid(True, alpha=0.3)

    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, ["Train Loss", "Val Loss"], loc="upper center", ncol=2, fontsize=10)
    fig.suptitle("Training Dashboard", fontsize=15, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    dashboard_curve_path = os.path.join(curves_dir, "training_dashboard.png")
    plt.savefig(dashboard_curve_path, dpi=300)
    print(f"✅ Dashboard figure saved to {dashboard_curve_path}")
    plt.close()

    # 4. 额外保存 npz，方便后续快速分析
    npz_path = os.path.join(run_dir, "metrics_arrays.npz")
    np.savez(
        npz_path,
        epoch=np.array(epochs_list),
        train_loss=np.array(train_loss_history),
        val_loss=np.array(val_loss_history),
        top1=np.array(top1_history),
        top3=np.array(top3_history),
        top5=np.array(top5_history),
        lr=np.array(lr_history)
    )
    print(f"✅ Numpy metrics saved to {npz_path}")

if __name__ == "__main__":
    main()