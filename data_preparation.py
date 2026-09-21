"""
data_preparation.py — Xây dựng tf.data pipeline cho tập train/val/test,
kèm data augmentation phù hợp với đặc điểm bệnh lá (giữ nguyên hình dạng
vết bệnh, chỉ augment ánh sáng / góc chụp / lật ảnh).
"""

import tensorflow as tf
from tensorflow.keras import layers

import config


def build_augmentation_pipeline():
    """
    Augmentation nhẹ nhàng: xoay, lật, zoom nhẹ, thay đổi độ sáng/tương phản.
    Tránh augment quá mạnh (ví dụ méo hình) vì hình dạng vết bệnh
    (lesion shape) là đặc trưng quan trọng để phân biệt các lớp.

    Lưu ý: factor của RandomRotation là tỉ lệ của 2*pi radian (360 độ), KHÔNG
    phải độ trực tiếp — factor=0.15 nghĩa là xoay tối đa ±54 độ, vốn khá mạnh
    và có thể làm biến dạng/che khuất vùng tổn thương nhỏ trên lá (đi ngược
    với mục tiêu "augment nhẹ" nêu trên). Dùng factor=0.05 (~ ±18 độ) để augment
    thực sự nhẹ, phù hợp với ảnh chụp thực địa vốn đã có góc chụp đa dạng sẵn.
    """
    return tf.keras.Sequential(
        [
            layers.RandomFlip("horizontal_and_vertical"),
            layers.RandomRotation(0.05),          # ~ ±18 độ tối đa
            layers.RandomZoom(0.1),
            layers.RandomContrast(0.15),
            layers.RandomBrightness(0.15),
        ],
        name="augmentation",
    )


def _make_dataset(directory, shuffle, augment=False):
    ds = tf.keras.utils.image_dataset_from_directory(
        directory,
        labels="inferred",
        label_mode="categorical",
        class_names=config.CLASS_NAMES,
        image_size=config.IMG_SIZE,
        batch_size=config.BATCH_SIZE,
        shuffle=shuffle,
        seed=config.SEED,
    )

    if augment:
        aug = build_augmentation_pipeline()
        ds = ds.map(lambda x, y: (aug(x, training=True), y),
                    num_parallel_calls=tf.data.AUTOTUNE)

    # Cache + prefetch để tăng tốc training
    return ds.prefetch(tf.data.AUTOTUNE)


def get_datasets():
    """Trả về (train_ds, val_ds, test_ds) đã sẵn sàng cho model.fit()."""
    train_ds = _make_dataset(config.TRAIN_DIR, shuffle=True, augment=True)
    val_ds = _make_dataset(config.VAL_DIR, shuffle=False, augment=False)
    test_ds = _make_dataset(config.TEST_DIR, shuffle=False, augment=False)
    return train_ds, val_ds, test_ds


def compute_class_weights():
    """
    Tính class weight để xử lý mất cân bằng giữa các lớp — số lượng ảnh/lớp
    có thể chênh lệch đáng kể trong dataset ảnh thực địa (raw), khác với các
    dataset đã được chọn lọc/cân bằng sẵn.
    """
    import os
    # Cùng danh sách file/thư mục rác hệ thống được loại trừ ở mọi nơi khác trong
    # pipeline (vd. .ipynb_checkpoints do Colab/Jupyter tự sinh) — nếu không lọc,
    # đếm bằng len(os.listdir(...)) có thể tính nhầm các mục rác này là ảnh thật,
    # làm class weight bị lệch nhẹ so với số ảnh thật sự trong mỗi lớp.
    valid_ext = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    counts = {}
    for cls in config.CLASS_NAMES:
        cls_dir = os.path.join(config.TRAIN_DIR, cls)
        if os.path.isdir(cls_dir):
            counts[cls] = sum(
                1 for f in os.listdir(cls_dir)
                if not f.startswith(".")
                and os.path.isfile(os.path.join(cls_dir, f))
                and os.path.splitext(f)[1].lower() in valid_ext
            )
    if not counts:
        return None

    total = sum(counts.values())
    n_classes = len(counts)
    class_weights = {
        i: total / (n_classes * counts[cls])
        for i, cls in enumerate(config.CLASS_NAMES)
        if cls in counts
    }
    return class_weights


if __name__ == "__main__":
    train_ds, val_ds, test_ds = get_datasets()
    print("Train batches:", tf.data.experimental.cardinality(train_ds).numpy())
    print("Val batches:", tf.data.experimental.cardinality(val_ds).numpy())
    print("Test batches:", tf.data.experimental.cardinality(test_ds).numpy())
    print("Class weights:", compute_class_weights())
