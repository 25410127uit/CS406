"""
train.py — Huấn luyện model theo 2 giai đoạn (khuyến nghị cho transfer learning
trên dataset nhỏ, xem số lượng ảnh thực tế mỗi tập tại config.py/DATA_DIR sau khi
chạy Mục 4 của notebook — dataset Durian gốc có ~5000 ảnh trước khi dedup):

  Giai đoạn 1: freeze backbone, chỉ train classification head (LR lớn hơn)
  Giai đoạn 2: fine-tune bằng cách mở khóa N layer cuối của backbone (LR nhỏ)

Cách chạy:
    python train.py --model efficientnet --epochs 30 --batch_size 32
"""

import argparse
import json
import os

import tensorflow as tf

import config
from data_preparation import get_datasets, compute_class_weights
from models import build_model, unfreeze_top_layers


def parse_args():
    parser = argparse.ArgumentParser(description="Train durian leaf disease classifier")
    parser.add_argument("--model", type=str, default="efficientnet",
                         choices=config.SUPPORTED_MODELS)
    parser.add_argument("--epochs", type=int, default=config.EPOCHS_HEAD + config.EPOCHS_FINE_TUNE)
    parser.add_argument("--batch_size", type=int, default=config.BATCH_SIZE)
    parser.add_argument("--use_class_weights", action="store_true",
                         help="Bật class weighting để xử lý mất cân bằng nhẹ giữa các lớp")
    return parser.parse_args()


def get_callbacks(model_name):
    os.makedirs(config.MODEL_DIR, exist_ok=True)
    ckpt_path = os.path.join(config.MODEL_DIR, f"{model_name}_best.keras")
    return [
        tf.keras.callbacks.ModelCheckpoint(
            ckpt_path, monitor="val_accuracy", save_best_only=True, verbose=1
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=6, restore_best_weights=True
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=3, min_lr=1e-7, verbose=1
        ),
        tf.keras.callbacks.CSVLogger(
            os.path.join(config.OUTPUT_DIR, f"{model_name}_training_log.csv"),
            append=True,  # callback này được tái sử dụng cho cả 2 giai đoạn fit() —
                           # append=False (mặc định) sẽ ghi đè, xoá mất log Giai đoạn 1
                           # ngay khi Giai đoạn 2 (fine-tune) bắt đầu
        ),
    ]


def main():
    args = parse_args()
    config.BATCH_SIZE = args.batch_size
    tf.keras.utils.set_random_seed(config.SEED)
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)
    os.makedirs(config.MODEL_DIR, exist_ok=True)

    # CSVLogger (trong get_callbacks) dùng append=True để nối log Giai đoạn 1 +
    # Giai đoạn 2 trong CÙNG một lần chạy — nên cần xoá log cũ ở đây để mỗi lần
    # chạy `train.py` mới (train lại từ đầu) bắt đầu với file log sạch, không bị
    # nối lẫn vào kết quả của lần chạy trước.
    old_log = os.path.join(config.OUTPUT_DIR, f"{args.model}_training_log.csv")
    if os.path.exists(old_log):
        os.remove(old_log)

    print(f"=== Đang tải dữ liệu từ {config.DATA_DIR} ===")
    train_ds, val_ds, test_ds = get_datasets()

    class_weights = compute_class_weights() if args.use_class_weights else None
    if class_weights:
        print("Class weights:", class_weights)

    print(f"=== Xây dựng model: {args.model} ===")
    model, _ = build_model(args.model, freeze_backbone=True)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(config.LEARNING_RATE_HEAD),
        loss="categorical_crossentropy",
        metrics=["accuracy", tf.keras.metrics.Precision(name="precision"),
                 tf.keras.metrics.Recall(name="recall")],
    )
    model.summary()

    callbacks = get_callbacks(args.model)

    # --- Giai đoạn 1: train classification head ---
    print("\n=== GIAI ĐOẠN 1: Train head (backbone freeze) ===")
    history_head = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=config.EPOCHS_HEAD,
        class_weight=class_weights,
        callbacks=callbacks,
    )

    # --- Giai đoạn 2: fine-tune ---
    print("\n=== GIAI ĐOẠN 2: Fine-tune (mở khóa lớp cuối backbone) ===")
    model = unfreeze_top_layers(model, config.UNFREEZE_LAST_N_LAYERS)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(config.LEARNING_RATE_FINE_TUNE),
        loss="categorical_crossentropy",
        metrics=["accuracy", tf.keras.metrics.Precision(name="precision"),
                 tf.keras.metrics.Recall(name="recall")],
    )
    history_finetune = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=args.epochs,
        initial_epoch=history_head.epoch[-1] + 1,
        class_weight=class_weights,
        callbacks=callbacks,
    )

    # --- Nạp lại checkpoint tốt nhất trước khi đánh giá/lưu model cuối cùng ---
    # Lý do: EarlyStopping(restore_best_weights=True) ở trên phục hồi trọng số
    # theo val_loss tốt nhất, nhưng ModelCheckpoint (get_callbacks) lại lưu
    # "<model>_best.keras" theo val_accuracy tốt nhất — 2 tiêu chí khác nhau nên
    # 2 bộ trọng số có thể KHÁC NHAU. Nếu không đồng bộ ở đây, test_results.json
    # (dùng để dựng bảng so sánh accuracy) sẽ được tính trên model theo val_loss,
    # trong khi evaluate.py / gradcam.py / app.py đều load "_best.keras" (theo
    # val_accuracy) để vẽ confusion matrix / Grad-CAM — dẫn tới số liệu trong
    # báo cáo không khớp với hình minh họa. Nạp lại "_best.keras" ở đây để mọi
    # script sau đó (evaluate.py, gradcam.py, app.py, và bảng so sánh trong
    # notebook) đều dùng CHUNG một bộ trọng số.
    best_ckpt_path = os.path.join(config.MODEL_DIR, f"{args.model}_best.keras")
    if os.path.exists(best_ckpt_path):
        print(f"\nNạp lại checkpoint tốt nhất (theo val_accuracy): {best_ckpt_path}")
        model = tf.keras.models.load_model(best_ckpt_path)
    else:
        print("\nCảnh báo: không tìm thấy checkpoint '_best.keras' — "
              "dùng trọng số hiện tại trong bộ nhớ (có thể chỉ khớp val_loss, "
              "không khớp val_accuracy tốt nhất).")

    # --- Lưu model cuối cùng (đồng bộ với checkpoint tốt nhất) + lịch sử training ---
    final_path = os.path.join(config.MODEL_DIR, f"{args.model}_final.keras")
    model.save(final_path)
    print(f"Đã lưu model tại: {final_path}")

    combined_history = {}
    for k in history_head.history:
        combined_history[k] = history_head.history[k] + history_finetune.history.get(k, [])
    with open(os.path.join(config.OUTPUT_DIR, f"{args.model}_history.json"), "w") as f:
        json.dump(combined_history, f, indent=2)

    # --- Đánh giá nhanh trên tập test ---
    print("\n=== Đánh giá trên tập test ===")
    results = model.evaluate(test_ds, return_dict=True)
    print(results)
    with open(os.path.join(config.OUTPUT_DIR, f"{args.model}_test_results.json"), "w") as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()
