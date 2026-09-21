"""
gradcam.py — Trực quan hóa Grad-CAM để kiểm tra model có thực sự "nhìn"
vào vùng tổn thương trên lá hay không (giải thích được / explainability),
rất hữu ích cho phần thảo luận trong báo cáo đồ án.

Cách chạy:
    python gradcam.py --model efficientnet --image path/to/leaf.jpg
"""

import argparse
import os

import matplotlib
import matplotlib.cm as cm
import numpy as np
import tensorflow as tf

import config
from models import BACKBONE_REGISTRY


def parse_args():
    parser = argparse.ArgumentParser(description="Grad-CAM visualization")
    parser.add_argument("--model", type=str, default="efficientnet",
                         choices=config.SUPPORTED_MODELS)
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--image", type=str, required=True,
                         help="Đường dẫn ảnh lá cần trực quan hóa")
    parser.add_argument("--last_conv_layer", type=str, default=None,
                         help="Tên layer conv cuối trong backbone; nếu để trống "
                              "script sẽ tự tìm layer conv cuối cùng")
    return parser.parse_args()


def _output_ndim(layer):
    """Số chiều (ndim) của output layer, hoặc None nếu không xác định được.

    Keras 3 (TF >= 2.16) đã bỏ thuộc tính `layer.output_shape` cũ (tf.keras 2) —
    layer giờ chỉ có `.output` (một KerasTensor); phải lấy shape qua `.output.shape`.
    Một số layer (vd. InputLayer chưa build, layer nhiều inbound node) có thể
    raise AttributeError/ValueError khi truy cập `.output` — bỏ qua an toàn.
    """
    try:
        shape = layer.output.shape
    except (AttributeError, ValueError):
        return None
    return None if shape is None else len(shape)


def find_last_conv_layer(model):
    """Tự động tìm layer convolution cuối cùng trong model (bao gồm bên trong backbone)."""
    for layer in reversed(model.layers):
        if isinstance(layer, tf.keras.Model):
            # backbone lồng bên trong — tìm đệ quy
            for sub_layer in reversed(layer.layers):
                if _output_ndim(sub_layer) == 4:
                    return layer.name, sub_layer.name
        elif _output_ndim(layer) == 4:
            return None, layer.name
    raise ValueError("Không tìm thấy conv layer nào trong model.")


def make_gradcam_heatmap(img_array, model, last_conv_layer_name, backbone_name=None,
                          preprocess_fn=None):
    if backbone_name:
        backbone = model.get_layer(backbone_name)
        last_conv_layer = backbone.get_layer(last_conv_layer_name)

        # backbone được gọi như 1 sub-model bên trong `model` (base_model(x) trong
        # models.py). Lưu ý: KHÔNG thể dựng `Model(model.inputs, backbone.input)` để
        # "lấy lại" tensor đã tiền xử lý từ graph của `model` — Keras 3 không trace
        # được qua ranh giới model lồng theo cách đó (lỗi KeyError khi gọi model).
        # Thay vào đó, áp dụng thẳng `preprocess_fn` (đã biết chính xác từ
        # models.BACKBONE_REGISTRY, tương ứng với đúng backbone đang dùng) lên ảnh gốc.
        #
        # backbone.output cũng KHÔNG dùng được trực tiếp làm "predictions": nó chỉ là
        # feature map thô, CHƯA qua GlobalAveragePooling2D + Dense ở đầu phân loại
        # (các layer đó nằm ngoài backbone, trong `model`) — nên vẫn cần tách riêng
        # classifier_model để chạy tiếp qua các layer đó, tái sử dụng đúng trọng số
        # đã train (không tạo layer mới / không mất trọng số).
        conv_model = tf.keras.models.Model(backbone.inputs, last_conv_layer.output)

        backbone_idx = model.layers.index(backbone)
        classifier_input = tf.keras.Input(shape=last_conv_layer.output.shape[1:])
        x = classifier_input
        for layer in model.layers[backbone_idx + 1:]:
            x = layer(x)
        classifier_model = tf.keras.models.Model(classifier_input, x)

        preprocessed = preprocess_fn(img_array) if preprocess_fn is not None else img_array
        with tf.GradientTape() as tape:
            conv_outputs = conv_model(preprocessed)
            tape.watch(conv_outputs)
            predictions = classifier_model(conv_outputs)
            pred_index = tf.argmax(predictions[0])
            class_channel = predictions[:, pred_index]
        grads = tape.gradient(class_channel, conv_outputs)
    else:
        grad_model = tf.keras.models.Model(
            model.inputs, [model.get_layer(last_conv_layer_name).output, model.output]
        )
        with tf.GradientTape() as tape:
            conv_outputs, predictions = grad_model(img_array)
            pred_index = tf.argmax(predictions[0])
            class_channel = predictions[:, pred_index]
        grads = tape.gradient(class_channel, conv_outputs)

    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0) / (tf.math.reduce_max(heatmap) + 1e-8)
    return heatmap.numpy(), int(pred_index.numpy())


def overlay_heatmap(img_path, heatmap, out_path, alpha=0.4):
    img = tf.keras.utils.load_img(img_path, target_size=config.IMG_SIZE)
    img = tf.keras.utils.img_to_array(img)

    heatmap = np.uint8(255 * heatmap)
    # matplotlib.cm.get_cmap() bị deprecated từ matplotlib 3.7 và có thể bị gỡ hẳn ở
    # bản mới hơn (matplotlib.colormaps[...] là API thay thế chính thức) — thử API mới
    # trước, fallback về API cũ nếu chạy trên matplotlib cũ hơn chưa có colormaps.
    try:
        jet = matplotlib.colormaps["jet"]
    except AttributeError:
        jet = cm.get_cmap("jet")
    jet_colors = jet(np.arange(256))[:, :3]
    jet_heatmap = jet_colors[heatmap]
    jet_heatmap = tf.keras.utils.array_to_img(jet_heatmap)
    jet_heatmap = jet_heatmap.resize((img.shape[1], img.shape[0]))
    jet_heatmap = tf.keras.utils.img_to_array(jet_heatmap)

    superimposed = jet_heatmap * alpha + img
    superimposed = tf.keras.utils.array_to_img(superimposed)
    superimposed.save(out_path)


def main():
    args = parse_args()
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)

    ckpt = args.checkpoint or os.path.join(config.MODEL_DIR, f"{args.model}_best.keras")
    if not os.path.exists(ckpt):
        raise FileNotFoundError(
            f"Không tìm thấy checkpoint: {ckpt}\n"
            f"Hãy chạy 'python train.py --model {args.model}' trước để tạo checkpoint này."
        )
    model = tf.keras.models.load_model(ckpt)

    img = tf.keras.utils.load_img(args.image, target_size=config.IMG_SIZE)
    img_array = tf.keras.utils.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)

    backbone_name, last_conv_layer_name = (
        (None, args.last_conv_layer) if args.last_conv_layer
        else find_last_conv_layer(model)
    )
    # Preprocess_fn đúng với backbone đang dùng (tra theo --model, giống hệt lúc
    # train trong models.py) — bắt buộc cho Grad-CAM khi model có backbone lồng.
    _, preprocess_fn = BACKBONE_REGISTRY[args.model]

    heatmap, pred_class_idx = make_gradcam_heatmap(
        img_array, model, last_conv_layer_name, backbone_name, preprocess_fn
    )

    out_path = os.path.join(
        config.OUTPUT_DIR,
        f"gradcam_{args.model}_{os.path.basename(args.image)}"
    )
    overlay_heatmap(args.image, heatmap, out_path)

    print(f"Dự đoán: {config.CLASS_NAMES[pred_class_idx]}")
    print(f"Đã lưu ảnh Grad-CAM tại: {out_path}")


if __name__ == "__main__":
    main()
