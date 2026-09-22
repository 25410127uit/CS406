# Phân loại bệnh lá sầu riêng bằng Deep Learning (CS406)

Đồ án: Phân loại bệnh trên lá cây sầu riêng sử dụng Transfer Learning (CNN).

## 1. Dataset

**Image Dataset of Ten Durian Diseases Captured in Real-Field Conditions from a Family Orchard in Vinh Long, Vietnam**
- Nguồn: Mendeley Data — https://data.mendeley.com/datasets/mhjwyb5p48/1
- DOI: 10.17632/mhjwyb5p48.1
- ~4.000 ảnh JPEG **gốc (raw)**, chụp thực địa (nhiễu, ánh sáng không đều, góc chụp đa dạng), **10 lớp bệnh** trên lá/thân/quả/rễ — KHÔNG có lớp "khỏe mạnh". Kèm theo bản đã crop dạng PNG.
- Dataset gốc **chưa chia sẵn** `train/validation/test` theo tỉ lệ cố định — notebook Colab đi kèm (`notebooks/CS406_Main.ipynb`) tự tải, tự dò cấu trúc, **khử trùng lặp/data leakage TRƯỚC** (ảnh raw + bản crop của cùng 1 tấm dễ bị trùng), rồi mới tự chia stratified 70/15/15 trên dữ liệu đã sạch.

### Cách tải dữ liệu
1. Cách khuyến nghị: chạy notebook Colab `notebooks/CS406_Main.ipynb` — Mục 4 tự tải dữ liệu (link tải trực tiếp cụ thể xem biến `MENDELEY_ZIP_URL` trong cell; endpoint `/public-api/` chính thức của Mendeley yêu cầu access token nên không dùng trực tiếp được), tự khử trùng lặp, tự chia tập, và in ra tên các lớp thực tế phát hiện được.
2. Cách thủ công: truy cập link Mendeley ở trên, bấm "Download all files" (file .zip), giải nén ra 1 thư mục tạm (10 thư mục lớp, ảnh còn phẳng — CHƯA chia tập), sau đó:
   ```bash
   # Bước A — khử trùng lặp TRƯỚC khi chia tập (quan trọng, xem giải thích ở trên)
   python dedup.py --mode flat --data-dir /đường/dẫn/thư/mục/vừa/giải/nén
   # Bước B — tự chia stratified 70/15/15 vào data/train, data/validation, data/test
   # (có thể dùng lại logic chia tập trong notebook Mục 4, hoặc script chia tập tuỳ ý)
   ```
   rồi đưa vào `data/` theo cấu trúc:

```
data/
├── train/
│   ├── <tên_lớp_1>/
│   ├── <tên_lớp_2>/
│   └── ... (10 thư mục lớp)
├── validation/
│   └── ... (cùng cấu trúc)
└── test/
    └── ... (cùng cấu trúc)
```

> **Không cần sửa `config.py` thủ công:** `CLASS_NAMES` được `config.py` **tự động dò** từ tên các thư mục con thực tế trong `data/train/` mỗi khi import — miễn `data/` đã có đúng cấu trúc trên. `DEFAULT_CLASS_NAMES` trong `config.py` chỉ là danh sách dự phòng dùng tạm khi `data/` chưa tồn tại.

## 2. Cài đặt môi trường

```bash
python -m venv venv
source venv/bin/activate         # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## 3. Cấu trúc project

```
durian-leaf-cnn/
├── config.py           # Cấu hình chung (đường dẫn, hyperparameters)
├── dedup.py             # Khử trùng lặp / data leakage (chạy TRƯỚC khi chia tập)
├── data_preparation.py # Data loader + augmentation
├── models.py            # Định nghĩa các kiến trúc transfer learning
├── train.py             # Script huấn luyện
├── evaluate.py           # Đánh giá: confusion matrix, classification report
├── gradcam.py            # Trực quan hóa Grad-CAM
├── requirements.txt
├── outputs/              # Nơi lưu biểu đồ, kết quả đánh giá
└── saved_models/         # Nơi lưu model đã train (.h5 / .keras)
```

## 4. Chạy huấn luyện

Huấn luyện 1 model (mặc định EfficientNetB0):
```bash
python train.py --model efficientnet --epochs 30 --batch_size 32
```

Các lựa chọn `--model`: `efficientnet`, `resnet50`, `mobilenet`

Train cả 3 model để so sánh:
```bash
for m in efficientnet resnet50 mobilenet; do
    python train.py --model $m --epochs 30
done
```

## 5. Đánh giá kết quả

```bash
python evaluate.py --model efficientnet
```
Kết quả xuất ra `outputs/`: confusion matrix (.png), classification report (.txt/.csv), learning curves.

## 6. Trực quan hóa Grad-CAM

```bash
python gradcam.py --model efficientnet --image path/to/leaf.jpg
```

## 7. Xử lý ảnh truyền thống

`image_processing.py` chứa các kỹ thuật dùng trong đồ án:
- HSV Histogram: phân tích đặc trưng màu.
- Gaussian Blur, Bilateral Filter, Sharpening: preprocessing experiment với ResNet50.
- Sobel, Canny: phân tích biên.

GLCM, SVM, SIFT và SURF không còn thuộc pipeline chính.

## 8. Ứng dụng demo (Streamlit)

Sau khi đã có ít nhất 1 model deep learning (`saved_models/<model>_best.keras`)
:

```bash
streamlit run app.py
```

Ứng dụng cho phép:
- Tải ảnh lá sầu riêng lên
- Chọn model để dự đoán: EfficientNetB0 / ResNet50 / MobileNetV2
- Xem **lớp bệnh dự đoán + confidence score** và phân phối xác suất theo từng lớp
- Xem **Histogram HSV, Sobel, Canny** để phân tích ảnh\n- Xem **Grad-CAM** để giải thích vùng ảnh mô hình chú ý

## 9. Quy trình thực nghiệm đề xuất cho báo cáo

1. Train 3 kiến trúc (EfficientNetB0, ResNet50, MobileNetV2) với cùng hyperparameters → so sánh accuracy/F1 trên tập test.
2. Ablation: có augmentation vs không augmentation.
3. Ablation: freeze toàn bộ backbone vs fine-tune một phần (unfreeze N layer cuối).
4. Phân tích confusion matrix — xác định các cặp lớp dễ nhầm lẫn (ví dụ các bệnh có triệu chứng màu sắc/vết tổn thương gần giống nhau).
5. Grad-CAM trên vài ảnh mỗi lớp để kiểm chứng model học đúng vùng tổn thương, không học nền/artefact.
6. (Tùy chọn) So sánh với kết quả bài báo gốc dùng cùng dataset (K-Fold CV + Bayesian Optimization) để benchmark.

## 10. Tài liệu tham khảo dataset

Nguyen, T. (2025). Image Dataset of Ten Durian Diseases Captured in Real-Field Conditions from a Family Orchard in Vinh Long, Vietnam. *Mendeley Data*, V1. DOI: 10.17632/mhjwyb5p48.1


## CS406 final compatibility notes
- Canonical 10-class mapping is shared by training, evaluation and Streamlit.
- Distributed experiment archive: 5,451 usable images
  (4,154 train / 649 validation / 648 test).
- Published description reports 5,452; that number is publication metadata only.
- Main checkpoints: efficientnet_best.keras, resnet50_best.keras,
  mobilenet_best.keras.
- evaluate.py writes accuracy, macro_precision, macro_recall, macro_f1,
  loss, checkpoint and n_test to outputs/<model>_test_results.json.


## CS406 final experiment convention

- Dataset root is supplied through `DURIAN_DATA_DIR`.
- Output/model roots are supplied through `DURIAN_OUTPUT_DIR` and `DURIAN_MODEL_DIR`.
- Official distributed split is preserved: 4,154 train / 649 validation / 648 test = 5,451 usable images.
- Main CNN comparison: EfficientNetB0, ResNet50, MobileNetV2.
- Preprocessing training experiment: Original, Gaussian Blur, Bilateral Filter, Sharpening with ResNet50.
- Sobel/Canny and HSV Histogram are analysis/visualization only.

