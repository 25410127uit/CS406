"""
models.py — Định nghĩa các kiến trúc transfer learning:
EfficientNetB0, ResNet50, MobileNetV2. Tất cả dùng chung 1 classification
head để so sánh công bằng giữa các backbone.
"""

import tensorflow as tf
from tensorflow.keras import layers, Model
from tensorflow.keras.applications import (
    EfficientNetB0,
    ResNet50,
    MobileNetV2,
)

import config


BACKBONE_REGISTRY = {
    "efficientnet": (EfficientNetB0, tf.keras.applications.efficientnet.preprocess_input),
    "resnet50": (ResNet50, tf.keras.applications.resnet50.preprocess_input),
    "mobilenet": (MobileNetV2, tf.keras.applications.mobilenet_v2.preprocess_input),
}


def build_model(model_name: str, num_classes: int = config.NUM_CLASSES,
                 img_size=config.IMG_SIZE, freeze_backbone: bool = True):
    """
    Tạo model transfer learning.

    Args:
        model_name: 'efficientnet' | 'resnet50' | 'mobilenet'
        freeze_backbone: True để freeze toàn bộ backbone (giai đoạn train head)

    Returns:
        (keras.Model, preprocess_input_fn)
    """
    if model_name not in BACKBONE_REGISTRY:
        raise ValueError(f"Model '{model_name}' không hỗ trợ. "
                          f"Chọn trong {list(BACKBONE_REGISTRY.keys())}")

    backbone_cls, preprocess_fn = BACKBONE_REGISTRY[model_name]

    base_model = backbone_cls(
        include_top=False,
        weights="imagenet",
        input_shape=(*img_size, 3),
    )
    base_model.trainable = not freeze_backbone

    inputs = tf.keras.Input(shape=(*img_size, 3))
    x = preprocess_fn(inputs)
    x = base_model(x, training=False if freeze_backbone else None)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dropout(0.2)(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)

    model = Model(inputs, outputs, name=f"{model_name}_durian_classifier")
    model.backbone = base_model  # tham chiếu tiện cho việc unfreeze khi fine-tune
    return model, preprocess_fn


def unfreeze_top_layers(model, n_layers: int = config.UNFREEZE_LAST_N_LAYERS):
    """Mở khóa N layer cuối của backbone để fine-tune."""
    base_model = model.backbone
    base_model.trainable = True
    for layer in base_model.layers[:-n_layers]:
        layer.trainable = False
    for layer in base_model.layers[-n_layers:]:
        if isinstance(layer, tf.keras.layers.BatchNormalization):
            layer.trainable = False
        else:
            layer.trainable = True
    return model


if __name__ == "__main__":
    for name in config.SUPPORTED_MODELS:
        m, _ = build_model(name)
        print(f"--- {name} ---")
        print("Tổng số tham số:", m.count_params())
