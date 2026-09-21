"""
dedup.py — Khử trùng lặp / data leakage cho dataset ảnh bệnh lá sầu riêng.

Bối cảnh: dataset gốc (Mendeley) có cả ảnh raw và bản đã crop của cùng 1 tấm ảnh.
Nếu 2 bản của cùng 1 ảnh rơi vào 2 tập khác nhau (vd. raw ở train, crop ở test),
model sẽ "nhìn thấy" ảnh test ngay trong lúc train -> data leakage, accuracy trên
test bị thổi phồng ảo. Notebook Colab đi kèm (`DurianLeafCS406.ipynb`) đã xử lý
việc này tự động ở Mục 4, nhưng logic đó chỉ nằm trong notebook — script này là
bản tương đương để dùng khi chạy pipeline cục bộ (`pip install -r requirements.txt`
rồi chạy `train.py` trực tiếp, không qua Colab).

CÁCH DÙNG ĐÚNG THỨ TỰ (quan trọng — xem README Mục 1 để hiểu vì sao thứ tự này
quan trọng đối với tỉ lệ train/validation/test):

    # 1) Nếu dữ liệu CHƯA chia tập (mỗi lớp 1 thư mục phẳng chứa toàn bộ ảnh):
    #    chạy dedup TRƯỚC, rồi mới tự chia 70/15/15 (script/notebook khác).
    python dedup.py --mode flat --data-dir raw_data

    # 2) Nếu dữ liệu ĐÃ được chia sẵn train/validation/test từ nguồn khác (không
    #    qua bước 1 ở trên) — dùng làm tuyến phòng vệ cuối, quét leakage NGANG các
    #    tập và chỉ xoá ở validation/test (giữ nguyên bản ở train). Lưu ý: cách này
    #    có thể làm lệch tỉ lệ split ban đầu vì chỉ xoá được ở 2/3 tập.
    python dedup.py --mode split --data-dir data
"""

import argparse
import hashlib
import os
from collections import defaultdict

from PIL import Image
import imagehash

VALID_EXT = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
IGNORE_ENTRIES = {".ipynb_checkpoints", ".DS_Store", "Thumbs.db", "desktop.ini"}
PHASH_THRESHOLD = 4  # Hamming distance tối đa để coi là "trùng lặp gần đúng"


def _is_real_dir(root, name):
    return (not name.startswith(".") and name not in IGNORE_ENTRIES
            and os.path.isdir(os.path.join(root, name)))


def _class_dirs(root):
    return sorted(d for d in os.listdir(root) if _is_real_dir(root, d))


def _md5_of(fpath):
    h = hashlib.md5()
    with open(fpath, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def _phash_of(fpath):
    try:
        with Image.open(fpath) as im:
            return imagehash.phash(im.convert("RGB"))
    except Exception:
        return None


def dedup_flat(data_dir):
    """Khử trùng lặp trên pool ảnh PHẲNG theo từng lớp, TRƯỚC khi chia tập.

    Chỉ giữ 1 bản đại diện mỗi nhóm trùng lặp (bản đầu tiên theo tên file đã
    sort, để kết quả xác định/reproducible), xoá các bản còn lại. Sau khi chạy
    xong, thư mục data_dir sẵn sàng để chia stratified train/validation/test mà
    không còn rủi ro 1 ảnh (raw+crop) lọt vào 2 tập khác nhau.

    Returns: (n_hashed, removed_log) — removed_log là list các (lớp, tên file, loại).
    """
    classes = _class_dirs(data_dir)
    if not classes:
        raise ValueError(f"Không tìm thấy thư mục lớp nào trong: {data_dir}")

    removed_log = []
    n_hashed = 0

    for cls in classes:
        cls_dir = os.path.join(data_dir, cls)
        fnames = sorted(
            f for f in os.listdir(cls_dir)
            if f.lower().endswith(VALID_EXT) and f not in IGNORE_ENTRIES
            and os.path.isfile(os.path.join(cls_dir, f))
        )
        records = [[f, os.path.join(cls_dir, f), _md5_of(os.path.join(cls_dir, f)), None]
                   for f in fnames]
        for r in records:
            r[3] = _phash_of(r[1])
        n_hashed += len(records)

        keep = {r[1] for r in records}

        # 1) Trùng lặp CHÍNH XÁC (md5), trong cùng lớp -> chỉ giữ bản đầu
        by_md5 = defaultdict(list)
        for r in records:
            by_md5[r[2]].append(r)
        for group in by_md5.values():
            if len(group) > 1:
                for r in group[1:]:
                    if r[1] in keep:
                        keep.discard(r[1])
                        removed_log.append((cls, r[0], "exact"))

        # 2) Trùng lặp GẦN ĐÚNG (perceptual hash), vd cặp raw/crop cùng 1 ảnh gốc
        remaining = [r for r in records if r[1] in keep and r[3] is not None]
        for i in range(len(remaining)):
            if remaining[i][1] not in keep:
                continue
            for j in range(i + 1, len(remaining)):
                if remaining[j][1] not in keep:
                    continue
                if remaining[i][3] - remaining[j][3] <= PHASH_THRESHOLD:
                    keep.discard(remaining[j][1])
                    removed_log.append((cls, remaining[j][0], "near"))

        for r in records:
            if r[1] not in keep and os.path.exists(r[1]):
                os.remove(r[1])

    return n_hashed, removed_log


def dedup_split(data_dir, splits=("train", "validation", "test")):
    """Quét/xoá leakage GIỮA CÁC TẬP đã chia sẵn — tuyến phòng vệ cuối.

    Chỉ tính là leakage khi trùng lặp xảy ra GIỮA 2 tập khác nhau (trùng trong
    cùng 1 tập thì bỏ qua). Với mọi leakage phát hiện được, xoá bản ở
    validation/test, giữ bản ở train.

    Nhận diện tên thư mục split KHÔNG phân biệt hoa/thường — dataset gốc (Mendeley,
    xem README) có thể dùng "Train"/"Test"/"Validation" viết hoa chữ đầu, trong khi
    Linux/Colab phân biệt hoa thường ở tên thư mục.

    LƯU Ý: cách này có thể làm lệch tỉ lệ split ban đầu (xem README Mục 1) —
    ưu tiên dùng dedup_flat() TRƯỚC khi chia tập nếu có thể.

    Returns: (exact_leaks, near_leak_pairs, removed_log).
    """
    SPLIT_ALIASES = {"train": "train", "validation": "validation", "val": "validation", "test": "test"}
    real_dirs = {d.lower(): d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))}
    split_dirs = {}
    for s in splits:
        canonical = SPLIT_ALIASES.get(s.lower(), s.lower())
        real_name = real_dirs.get(canonical)
        if real_name is None:
            raise ValueError(
                f"Không tìm thấy thư mục tập '{s}' tại: {data_dir} "
                f"(đã thử không phân biệt hoa/thường, các thư mục con hiện có: {sorted(real_dirs.values())})"
            )
        split_dirs[s] = os.path.join(data_dir, real_name)

    records = []  # [split, class, filepath, md5, phash]
    for split, split_dir in split_dirs.items():
        for cls in _class_dirs(split_dir):
            cls_dir = os.path.join(split_dir, cls)
            for fname in sorted(os.listdir(cls_dir)):
                fpath = os.path.join(cls_dir, fname)
                if (fname in IGNORE_ENTRIES or fname.startswith(".")
                        or not os.path.isfile(fpath)):
                    continue
                records.append([split, cls, fpath, _md5_of(fpath), None])
    for r in records:
        r[4] = _phash_of(r[2])

    by_md5 = defaultdict(list)
    for r in records:
        by_md5[r[3]].append(r)
    exact_leaks = [recs for recs in by_md5.values() if len({r[0] for r in recs}) > 1]

    by_class = defaultdict(list)
    for r in records:
        if r[4] is not None:
            by_class[r[1]].append(r)
    near_leak_pairs = []
    for cls, recs in by_class.items():
        for i in range(len(recs)):
            for j in range(i + 1, len(recs)):
                ri, rj = recs[i], recs[j]
                if ri[0] == rj[0]:
                    continue
                if ri[4] - rj[4] <= PHASH_THRESHOLD:
                    near_leak_pairs.append((ri, rj))

    removed_log = []
    for group in exact_leaks:
        keep = next((r for r in group if r[0] == "train"), group[0])
        for r in group:
            if r is not keep and r[0] in ("validation", "test") and os.path.exists(r[2]):
                os.remove(r[2])
                removed_log.append((r[0], r[1], os.path.basename(r[2])))
    for ri, rj in near_leak_pairs:
        for r in (ri, rj):
            if r[0] in ("validation", "test") and os.path.exists(r[2]):
                os.remove(r[2])
                removed_log.append((r[0], r[1], os.path.basename(r[2])))

    return exact_leaks, near_leak_pairs, removed_log


def main():
    parser = argparse.ArgumentParser(
        description="Khử trùng lặp / data leakage cho dataset ảnh bệnh lá sầu riêng"
    )
    parser.add_argument("--mode", choices=["flat", "split"], required=True,
                         help="'flat': dedup TRƯỚC khi chia tập (khuyến nghị). "
                              "'split': dedup SAU khi đã chia tập (tuyến phòng vệ cuối).")
    parser.add_argument("--data-dir", default="data",
                         help="'flat': thư mục chứa các lớp phẳng, chưa chia tập. "
                              "'split': thư mục cha chứa train/validation/test.")
    args = parser.parse_args()

    if args.mode == "flat":
        print(f"Đang khử trùng lặp trên pool phẳng: {args.data_dir}")
        n_hashed, removed = dedup_flat(args.data_dir)
        n_exact = sum(1 for _, _, k in removed if k == "exact")
        n_near = sum(1 for _, _, k in removed if k == "near")
        print(f"Đã hash {n_hashed} ảnh.")
        print(f"Trùng lặp chính xác (MD5, cùng lớp)   : {n_exact} file đã xoá")
        print(f"Trùng lặp gần đúng (perceptual hash)  : {n_near} file đã xoá")
        if removed:
            print("Ví dụ vài file đã xoá:")
            for cls, fname, kind in removed[:10]:
                print(f"  [{cls}] {fname} ({kind})")
        print(f"Tổng ảnh còn lại (sẵn sàng để chia tập): {n_hashed - len(removed)}")
    else:
        print(f"Đang quét leakage giữa các tập trong: {args.data_dir}")
        exact_leaks, near_leak_pairs, removed = dedup_split(args.data_dir)
        print(f"Trùng lặp chính xác (MD5) giữa các tập : {len(exact_leaks)} nhóm")
        print(f"Trùng lặp gần đúng giữa các tập        : {len(near_leak_pairs)} cặp")
        if not exact_leaks and not near_leak_pairs:
            print("Không phát hiện leakage.")
        else:
            print(f"Đã xoá {len(removed)} file ở validation/test (giữ bản ở train).")
            print("LƯU Ý: tỉ lệ train/validation/test có thể đã lệch sau bước này.")
            if removed:
                print("Ví dụ vài file đã xoá:")
                for split, cls, fname in removed[:10]:
                    print(f"  [{split}/{cls}] {fname}")


if __name__ == "__main__":
    main()
