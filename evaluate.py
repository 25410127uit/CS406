"""
evaluate.py — Đánh giá chi tiết model đã train: confusion matrix,
classification report (precision/recall/F1 theo từng lớp), learning curves.

Cách chạy:
    python evaluate.py --model efficientnet
"""

import argparse
import json
import os

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
)

import config
from data_preparation import get_datasets


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate trained model")
    parser.add_argument("--model", type=str, default="efficientnet",
                         choices=config.SUPPORTED_MODELS)
    parser.add_argument("--checkpoint", type=str, default=None,
                         help="Đường dẫn model cụ thể; mặc định dùng "
                              "<model>_best.keras trong saved_models/")
    return parser.parse_args()


def plot_confusion_matrix(cm, class_names, out_path):
    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(class_names)))
    ax.set_yticks(range(len(class_names)))
    ax.set_xticklabels(class_names, rotation=45, ha="right")
    ax.set_yticklabels(class_names)
    ax.set_xlabel("Dự đoán")
    ax.set_ylabel("Thực tế")
    ax.set_title("Confusion Matrix")

    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], "d"), ha="center", va="center",
                     color="white" if cm[i, j] > thresh else "black")

    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_learning_curves(history_path, out_path):
    if not os.path.exists(history_path):
        print(f"Không tìm thấy {history_path}, bỏ qua learning curves.")
        return
    with open(history_path) as f:
        hist = json.load(f)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    axes[0].plot(hist["accuracy"], label="train")
    axes[0].plot(hist["val_accuracy"], label="val")
    axes[0].set_title("Accuracy")
    axes[0].set_xlabel("Epoch")
    axes[0].legend()

    axes[1].plot(hist["loss"], label="train")
    axes[1].plot(hist["val_loss"], label="val")
    axes[1].set_title("Loss")
    axes[1].set_xlabel("Epoch")
    axes[1].legend()

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main():
    args = parse_args()
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)

    ckpt = args.checkpoint or os.path.join(config.MODEL_DIR, f"{args.model}_best.keras")
    if not os.path.exists(ckpt):
        raise FileNotFoundError(
            f"Không tìm thấy checkpoint: {ckpt}\n"
            f"Hãy chạy 'python train.py --model {args.model}' trước để tạo checkpoint này."
        )
    print(f"Đang tải model: {ckpt}")
    model = tf.keras.models.load_model(ckpt, compile=False)
    model.compile(
        optimizer="adam",
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )

    _, _, test_ds = get_datasets()

    y_true, y_pred = [], []
    for images, labels in test_ds:
        preds = model.predict(images, verbose=0)
        y_true.extend(np.argmax(labels.numpy(), axis=1))
        y_pred.extend(np.argmax(preds, axis=1))

    y_true, y_pred = np.array(y_true), np.array(y_pred)

    # Aggregate metrics used by the CS406 comparison table.
    eval_dict = model.evaluate(test_ds, verbose=0, return_dict=True)
    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_precision": float(
            precision_score(y_true, y_pred, average="macro", zero_division=0)
        ),
        "macro_recall": float(
            recall_score(y_true, y_pred, average="macro", zero_division=0)
        ),
        "macro_f1": float(
            f1_score(y_true, y_pred, average="macro", zero_division=0)
        ),
        "loss": float(eval_dict["loss"]),
        "checkpoint": os.path.basename(ckpt),
        "n_test": int(len(y_true)),
        "metrics_version": 2,
    }
    result_path = os.path.join(
        config.OUTPUT_DIR, f"{args.model}_test_results.json"
    )
    with open(result_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)
    print("Test metrics:", metrics)
    print(f"Đã lưu test metrics: {result_path}")

    # Luôn truyền labels=[0..n_classes-1] tường minh cho classification_report VÀ
    # confusion_matrix. Nếu không, sklearn tự suy ra danh sách lớp từ chính
    # y_true/y_pred: nếu 1 lớp nào đó (vd lớp ít ảnh nhất) không xuất hiện trong
    # tập test của lần chạy này, sklearn sẽ tự BỎ QUA lớp đó — khiến số lớp thực
    # tế < 10, làm target_names=config.CLASS_NAMES (luôn đủ 10 tên) bị LỆCH HÀNG
    # so với các lớp còn lại, và confusion matrix trả về nhỏ hơn 10x10 khiến
    # plot_confusion_matrix() (giả định đúng 10x10) vẽ sai/lỗi.
    all_labels = list(range(len(config.CLASS_NAMES)))

    # --- Classification report ---
    report = classification_report(
        y_true, y_pred, labels=all_labels, target_names=config.CLASS_NAMES,
        digits=4, zero_division=0,
    )
    print(report)
    report_path = os.path.join(config.OUTPUT_DIR, f"{args.model}_classification_report.txt")
    with open(report_path, "w") as f:
        f.write(report)

    # --- Confusion matrix ---
    cm = confusion_matrix(y_true, y_pred, labels=all_labels)
    cm_path = os.path.join(config.OUTPUT_DIR, f"{args.model}_confusion_matrix.png")
    plot_confusion_matrix(cm, config.CLASS_NAMES, cm_path)
    print(f"Đã lưu confusion matrix: {cm_path}")

    # --- Learning curves ---
    history_path = os.path.join(config.OUTPUT_DIR, f"{args.model}_history.json")
    curves_path = os.path.join(config.OUTPUT_DIR, f"{args.model}_learning_curves.png")
    plot_learning_curves(history_path, curves_path)
    print(f"Đã lưu learning curves: {curves_path}")


if __name__ == "__main__":
    main()
