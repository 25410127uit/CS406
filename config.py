"""
config.py — Cấu hình chung cho pipeline CS406.
Có thể chạy local hoặc Colab. Notebook ưu tiên truyền đường dẫn qua biến môi trường.
"""

import os

PROJECT_DIR = os.environ.get(
    "DURIAN_PROJECT_DIR",
    os.path.dirname(os.path.abspath(__file__))
)

DATA_DIR = os.environ.get(
    "DURIAN_DATA_DIR",
    os.path.join(PROJECT_DIR, "data")
)
TRAIN_DIR = os.path.join(DATA_DIR, "train")
VAL_DIR = os.path.join(DATA_DIR, "validation")
TEST_DIR = os.path.join(DATA_DIR, "test")

OUTPUT_DIR = os.environ.get(
    "DURIAN_OUTPUT_DIR",
    os.path.join(PROJECT_DIR, "outputs")
)
MODEL_DIR = os.environ.get(
    "DURIAN_MODEL_DIR",
    os.path.join(PROJECT_DIR, "saved_models")
)

DEFAULT_CLASS_NAMES = [
    "anthracnose_disease",
    "canker_disease",
    "fruit_rot",
    "mealybug_infestation",
    "pink_disease",
    "sooty_mold",
    "stem_blight",
    "stem_cracking_gummosis",
    "thrips_disease",
    "yellow_leaf",
]
CANONICAL_CLASS_NAMES = list(DEFAULT_CLASS_NAMES)

def _detect_class_names():
    if os.path.isdir(TRAIN_DIR):
        found = sorted(
            d for d in os.listdir(TRAIN_DIR)
            if os.path.isdir(os.path.join(TRAIN_DIR, d))
            and not d.startswith(".")
        )
        expected = sorted(CANONICAL_CLASS_NAMES)
        if found and found != expected:
            raise RuntimeError(
                "TRAIN_DIR class folders do not match canonical CS406 classes.\n"
                f"Found: {found}\nExpected: {expected}"
            )
    return list(CANONICAL_CLASS_NAMES)

CLASS_NAMES = _detect_class_names()
NUM_CLASSES = len(CLASS_NAMES)

if NUM_CLASSES != 10:
    raise RuntimeError(f"CS406 expects exactly 10 classes, got {NUM_CLASSES}.")

IMG_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS_HEAD = 10
EPOCHS_FINE_TUNE = 20
LEARNING_RATE_HEAD = 1e-3
LEARNING_RATE_FINE_TUNE = 1e-5
UNFREEZE_LAST_N_LAYERS = 30
SEED = 42

SUPPORTED_MODELS = ["efficientnet", "resnet50", "mobilenet"]
