import os
import numpy as np
import pandas as pd
import streamlit as st
import tensorflow as tf
from PIL import Image
import matplotlib
import matplotlib.pyplot as plt
from image_processing import hsv_histogram, sobel, canny

st.set_page_config(page_title="Durian Disease AI", page_icon="🌿", layout="wide", initial_sidebar_state="collapsed")

MODEL_FILES = {
    "EfficientNetB0": "efficientnet_best.keras",
    "ResNet50": "resnet50_best.keras",
    "MobileNetV2": "mobilenet_best.keras",
}
CLASS_NAMES = [
    "Anthracnose", "Canker", "Fruit Rot", "Mealybug Infestation",
    "Pink Disease", "Sooty Mold", "Stem Blight",
    "Stem Cracking Gummosis", "Thrips Disease", "Yellow Leaf",
]
IMG_SIZE = (224, 224)

DISEASE_INFO = {
    "Anthracnose": ("Bệnh thán thư", "Thường biểu hiện bằng các vùng tổn thương sẫm màu trên mô cây; vùng bệnh có thể khô hoặc hoại tử.",
                    ["Đốm/vùng nâu đến đen.", "Vết bệnh có thể lan rộng.", "Mô bị ảnh hưởng có thể khô hoặc hoại tử."]),
    "Canker": ("Bệnh loét", "Tổn thương dạng loét thường xuất hiện trên thân hoặc cành và làm mô tại khu vực bị bệnh suy yếu.",
               ["Vết loét trên thân/cành.", "Vỏ cây tại vùng bệnh có thể đổi màu.", "Tổn thương có thể mở rộng."]),
    "Fruit Rot": ("Bệnh thối trái", "Tình trạng mô trái bị tổn thương và thối; vùng bệnh có thể lan sang phần còn lại của trái.",
                  ["Vùng nâu hoặc sẫm màu trên trái.", "Mô trái có thể mềm/phân hủy.", "Vết bệnh có thể tăng kích thước."]),
    "Mealybug Infestation": ("Rệp sáp", "Tình trạng cây bị rệp sáp gây hại; côn trùng thường tập trung thành cụm với lớp sáp trắng.",
                             ["Cụm trắng giống bông/sáp.", "Có thể tập trung ở cành, cuống, lá hoặc trái.", "Cây bị hại nặng có thể sinh trưởng kém."]),
    "Pink Disease": ("Bệnh nấm hồng", "Bệnh thường ảnh hưởng cành hoặc thân và có thể tạo lớp màu hồng trên bề mặt vùng bị bệnh.",
                     ["Mảng/lớp màu hồng trên cành hoặc thân.", "Vỏ cây có thể bị tổn thương.", "Cành bị ảnh hưởng nặng có thể suy yếu hoặc khô."]),
    "Sooty Mold": ("Bệnh bồ hóng", "Biểu hiện đặc trưng là lớp màu đen giống bồ hóng phủ trên bề mặt lá hoặc bộ phận khác của cây.",
                   ["Lớp đen giống muội than.", "Có thể phủ một phần hoặc phần lớn bề mặt.", "Che phủ nhiều có thể làm giảm ánh sáng tới lá."]),
    "Stem Blight": ("Bệnh cháy thân", "Tổn thương trên thân hoặc cành có thể dẫn đến khô và chết mô tại khu vực bị ảnh hưởng.",
                    ["Vùng nâu/sẫm trên thân hoặc cành.", "Mô có dấu hiệu khô/chết.", "Vùng tổn thương có thể lan dọc thân/cành."]),
    "Stem Cracking Gummosis": ("Nứt thân và chảy nhựa", "Thân hoặc vỏ cây xuất hiện vết nứt kèm hiện tượng tiết/chảy nhựa tại vùng tổn thương.",
                               ["Vỏ hoặc thân xuất hiện vết nứt.", "Có thể có nhựa tiết ra.", "Vùng xung quanh có thể đổi màu hoặc hoại tử."]),
    "Thrips Disease": ("Tổn thương do bọ trĩ", "Bọ trĩ có thể gây tổn thương bề mặt mô non, làm lá hoặc bộ phận bị hại biến dạng/đổi màu.",
                       ["Bề mặt lá có thể bạc/đổi màu.", "Lá non có thể biến dạng.", "Tổn thương thường tập trung ở mô non."]),
    "Yellow Leaf": ("Vàng lá", "Lá mất dần màu xanh và chuyển vàng. Đây là biểu hiện hình ảnh có thể liên quan đến nhiều nguyên nhân.",
                    ["Một phần hoặc toàn bộ phiến lá chuyển vàng.", "Màu xanh giảm rõ rệt.", "Có thể đi kèm sinh trưởng kém."]),
}

st.markdown("""
<style>
.block-container{max-width:1220px;padding-top:2rem;padding-bottom:4rem}
#MainMenu,footer{visibility:hidden}
header[data-testid="stHeader"]{background:transparent}
.hero{padding:32px 38px;border-radius:22px;background:linear-gradient(135deg,#0d5132,#198754);color:white;margin-bottom:28px;box-shadow:0 10px 30px rgba(0,0,0,.12)}
.hero-title{font-size:37px;font-weight:800}.hero-subtitle{margin-top:7px;color:#e9fff2;font-size:16px}
.result-card{padding:26px;border:1px solid #e5e7eb;border-radius:18px;background:white;box-shadow:0 6px 22px rgba(0,0,0,.06)}
.result-label{font-size:12px;font-weight:700;color:#64748b}.result-name{color:#16834b;font-size:30px;font-weight:800;margin-top:7px}
.result-vi{font-size:17px;color:#475569;margin-bottom:15px}.info-card{padding:22px 24px;border-radius:16px;border:1px solid #e5e7eb;background:#fafafa;margin-top:12px}
.warning-card{padding:15px 18px;border-radius:12px;background:#fff8e6;border:1px solid #f4d58d;margin-top:20px}
[data-testid="stImage"] img{border-radius:16px}
</style>
<div class="hero"><div class="hero-title">🌿 Durian Disease AI</div>
<div class="hero-subtitle">Nhận diện bệnh sầu riêng bằng Transfer Learning và giải thích mô hình bằng Grad-CAM</div></div>
""", unsafe_allow_html=True)

@st.cache_resource(show_spinner=False)
def load_model(name):
    path = os.path.join("saved_models", MODEL_FILES[name])
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Không tìm thấy checkpoint: {path}")
    return tf.keras.models.load_model(path, compile=False)

# Prediction uses original RGB; research preprocessing is not silently applied here.
def prepare_image(image):
    image = image.convert("RGB").resize(IMG_SIZE)
    return np.expand_dims(np.asarray(image, dtype=np.float32), 0)

def predict(model, image):
    p = np.asarray(model.predict(prepare_image(image), verbose=0)[0], dtype=np.float32)
    if p.min() < 0 or p.max() > 1 or not np.isclose(p.sum(), 1, atol=1e-3):
        p = tf.nn.softmax(p).numpy()
    return p

def find_backbone(model):
    candidates = []
    for layer in model.layers:
        if isinstance(layer, tf.keras.Model):
            count = 0
            for sub in layer.layers:
                try:
                    if len(sub.output.shape) == 4:
                        count += 1
                except Exception:
                    pass
            if count:
                candidates.append((count, layer))
    return max(candidates, key=lambda x: x[0])[1] if candidates else None

def find_last_conv_layer(backbone):
    for layer in reversed(backbone.layers):
        try:
            if len(layer.output.shape) == 4:
                return layer
        except Exception:
            pass
    return None

def make_gradcam_heatmap(model, image, class_index):
    backbone = find_backbone(model)
    if backbone is None:
        raise RuntimeError("Không tìm thấy CNN backbone.")
    conv = find_last_conv_layer(backbone)
    if conv is None:
        raise RuntimeError("Không tìm thấy feature map 4D cuối.")

    feature_model = tf.keras.Model(backbone.input, [conv.output, backbone.output])
    head_input = tf.keras.Input(shape=backbone.output.shape[1:])
    x = head_input
    found = False
    for layer in model.layers:
        if layer is backbone:
            found = True
            continue
        if found:
            x = layer(x)
    head_model = tf.keras.Model(head_input, x)

    with tf.GradientTape() as tape:
        conv_out, backbone_out = feature_model(prepare_image(image), training=False)
        tape.watch(conv_out)
        pred = head_model(backbone_out, training=False)
        score = pred[:, class_index]
    grads = tape.gradient(score, conv_out)
    if grads is None:
        raise RuntimeError("Không tính được gradient.")
    weights = tf.reduce_mean(grads, axis=(0,1,2))
    heatmap = tf.reduce_sum(conv_out[0] * weights, axis=-1)
    heatmap = tf.maximum(heatmap, 0)
    maximum = tf.reduce_max(heatmap)
    heatmap = tf.where(maximum > 0, heatmap / maximum, heatmap)
    return heatmap.numpy(), backbone.name, conv.name

def overlay_gradcam(image, heatmap, alpha=.42):
    hm = np.uint8(np.clip(heatmap, 0, 1) * 255)
    rgb = matplotlib.colormaps["jet"](np.arange(256))[:, :3]
    colored = Image.fromarray(np.uint8(rgb[hm] * 255)).resize(image.size)
    return Image.blend(image.convert("RGB"), colored, alpha)

a,b = st.columns([1,2])
with a:
    model_name = st.selectbox("Mô hình", list(MODEL_FILES), index=1)
with b:
    uploaded = st.file_uploader("Ảnh cần phân tích", type=["jpg","jpeg","png"])

if uploaded is None:
    st.info("👆 Chọn một ảnh sầu riêng để bắt đầu phân tích.")
    st.stop()

image = Image.open(uploaded).convert("RGB")
try:
    with st.spinner(f"Đang phân tích bằng {model_name}..."):
        model = load_model(model_name)
        probabilities = predict(model, image)
except Exception as e:
    st.error(f"Không thể chạy mô hình: {e}")
    st.stop()

idx = int(np.argmax(probabilities))
confidence = float(probabilities[idx])
predicted_class = CLASS_NAMES[idx]
vi_name, description, symptoms = DISEASE_INFO[predicted_class]

left,right = st.columns([1.05,1], gap="large")
with left:
    st.subheader("📷 Ảnh đầu vào")
    st.image(image, use_container_width=True)
with right:
    st.subheader("🔎 Kết quả chẩn đoán")
    st.markdown(f"""<div class="result-card"><div class="result-label">MÔ HÌNH DỰ ĐOÁN</div>
    <div class="result-name">{predicted_class}</div><div class="result-vi">{vi_name}</div>
    <div>Độ tin cậy: <strong>{confidence:.2%}</strong></div></div>""", unsafe_allow_html=True)
    st.progress(confidence)
    st.markdown("### Top 3 dự đoán")
    top3=np.argsort(probabilities)[-3:][::-1]
    st.dataframe(pd.DataFrame({"Bệnh":[CLASS_NAMES[i] for i in top3],
                               "Xác suất":[f"{probabilities[i]:.2%}" for i in top3]}),
                 hide_index=True,use_container_width=True)
    st.caption(f"Model: {model_name} • Input: 224×224 RGB • 10 classes")

st.markdown("---")
st.header(f"🦠 {vi_name}")
st.markdown(f"**Tên lớp:** {predicted_class}")
st.markdown("### Bệnh/tình trạng này là gì?")
st.markdown(f'<div class="info-card">{description}</div>', unsafe_allow_html=True)
st.markdown("### 🔍 Dấu hiệu nhận biết")
for s in symptoms:
    st.markdown(f"- {s}")

st.markdown("### 📊 Mức độ chắc chắn")
if confidence >= .90:
    st.success(f"Độ tin cậy của mô hình cao: {confidence:.2%}.")
elif confidence >= .70:
    st.warning(f"Độ tin cậy tương đối: {confidence:.2%}. Nên xem thêm Top-3.")
else:
    st.warning(f"Độ tin cậy thấp: {confidence:.2%}. Các lớp có thể có biểu hiện hình ảnh tương tự.")

st.markdown("---")
st.header("🖼️ Phân tích ảnh — CS406")
st.caption("Histogram HSV, Sobel và Canny chỉ dùng để quan sát đặc trưng; CNN vẫn dự đoán từ ảnh RGB gốc.")
with st.expander("Xem Histogram HSV, Sobel và Canny", expanded=False):
    rgb_np = np.asarray(image.convert("RGB"), dtype=np.uint8)
    hist = hsv_histogram(rgb_np, bins=8)
    fig, ax = plt.subplots(figsize=(10, 3))
    ax.plot(hist)
    ax.set_title("HSV Histogram (8 × 8 × 8 bins)")
    ax.set_xlabel("Histogram bin")
    ax.set_ylabel("Normalized frequency")
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)
    e1, e2 = st.columns(2)
    with e1:
        st.markdown("**Sobel**")
        st.image(sobel(rgb_np), use_container_width=True)
    with e2:
        st.markdown("**Canny**")
        st.image(canny(rgb_np), use_container_width=True)

st.markdown("---")
st.header("🔬 Explainability — Grad-CAM")
st.caption(f"Vùng nóng thể hiện khu vực ảnh ảnh hưởng nhiều đến dự đoán {predicted_class}.")
try:
    with st.spinner("Đang tạo Grad-CAM..."):
        heatmap, backbone_name, conv_name = make_gradcam_heatmap(model, image, idx)
        overlay = overlay_gradcam(image, heatmap)
    g1,g2=st.columns(2,gap="large")
    with g1:
        st.markdown("### 📷 Ảnh gốc")
        st.image(image,use_container_width=True)
    with g2:
        st.markdown("### 🔥 Grad-CAM")
        st.image(overlay,use_container_width=True)
    with st.expander("⚙️ Thông tin Grad-CAM"):
        st.write("Model:", model_name)
        st.write("Backbone:", backbone_name)
        st.write("Feature layer:", conv_name)
        st.write("Predicted class:", predicted_class)
        st.write("Confidence:", f"{confidence:.4%}")
except Exception as e:
    st.warning("Không tạo được Grad-CAM cho checkpoint này.")
    st.code(str(e))

st.markdown("""<div class="warning-card"><strong>⚠️ Lưu ý</strong><br><br>
Kết quả là dự đoán của mô hình học sâu dựa trên hình ảnh, phục vụ học tập/nghiên cứu CS406 và hỗ trợ nhận diện; không nên xem là chẩn đoán chuyên môn độc lập.
</div>""",unsafe_allow_html=True)
st.markdown("---")
st.caption(f"CS406 — Xử lý ảnh và ứng dụng • Durian Disease Classification • {model_name}")
