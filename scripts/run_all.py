# scripts/run_all.py

import os
import re
import math
import random
import json
import shutil
from pathlib import Path

import numpy as np
import cv2
from tqdm import tqdm
import tensorflow as tf

USE_KERAS_ADAMW = True

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

DATASET_DIR = r"C:/Users/User/Downloads/DIT5411-Cheung Man Tung/data"

WORKDIR = Path("./workdir")
RAW_SPLIT_DIR = WORKDIR / "split"
AUG_TRAIN_DIR = WORKDIR / "processed/train_aug"
TEST_DIR = WORKDIR / "processed/test"
RESULTS_DIR = WORKDIR / "results"
CKPT_DIR = WORKDIR / "checkpoints"

IMG_SIZE = 64
MIN_PER_CLASS_TRAIN = 5
TARGET_PER_CLASS_TRAIN = 200
BATCH_SIZE = 128
EPOCHS = 40

def natural_key(s):
    ss = str(s)
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", ss)]

def ensure_clean(dirpath: Path):
    if dirpath.exists():
        shutil.rmtree(dirpath)
    dirpath.mkdir(parents=True, exist_ok=True)

def is_image_path(p: Path):
    return p.suffix.lower() in [".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"]

def iter_all_images_under_cleaned(root: Path):
    for cleaned_dir in sorted(root.glob("cleaned_data*"), key=natural_key):
        if not cleaned_dir.is_dir():
            continue
        # 平坦層
        for p in sorted(cleaned_dir.glob("*"), key=natural_key):
            if p.is_file() and is_image_path(p):
                yield p
        # 遞迴層
        for p in cleaned_dir.rglob("*"):
            if p.is_file() and is_image_path(p):
                yield p

def extract_class_from_filename(path: Path):
    name = path.stem
    if "_" in name:
        return name.split("_", 1)[0]
    return name

# -------- Unicode-safe I/O for OpenCV on Windows --------

def imread_unicode(path: Path, flags=cv2.IMREAD_GRAYSCALE):
    # 使用 numpy.fromfile + cv2.imdecode 以避免 Unicode 路徑問題
    try:
        data = np.fromfile(str(path), dtype=np.uint8)
        if data.size == 0:
            return None
        img = cv2.imdecode(data, flags)
        return img
    except Exception:
        return None

def imwrite_unicode(path: Path, img, params=None):
    # 使用 cv2.imencode + tofile，避免 Unicode 路徑問題
    ext = path.suffix.lower()
    if ext == "":
        ext = ".png"
        path = Path(str(path) + ext)
    # 默認 png
    fmt = ext
    if not fmt.startswith("."):
        fmt = "." + fmt
    try:
        ok, buf = cv2.imencode(fmt, img, params or [])
        if not ok:
            return False
        buf.tofile(str(path))
        return True
    except Exception:
        return False

def split_train_test_from_filenames(src_root: Path, dst_root: Path, min_train=40):
    train_root = dst_root / "train_raw"
    test_root = dst_root / "test_raw"
    ensure_clean(dst_root)
    train_root.mkdir(parents=True, exist_ok=True)
    test_root.mkdir(parents=True, exist_ok=True)

    by_class = {}
    seen = set()
    for p in iter_all_images_under_cleaned(src_root):
        key = str(p.resolve())
        if key in seen:
            continue
        seen.add(key)
        cname = extract_class_from_filename(p)
        by_class.setdefault(cname, []).append(p)

    kept_stats = []
    class_names = []

    for cname, paths in sorted(by_class.items(), key=lambda x: natural_key(x[0])):
        paths = [p for p in sorted(paths, key=natural_key)]
        if len(paths) < min_train:
            kept_stats.append((cname, len(paths), "SKIP(<min_train)"))
            continue
        class_names.append(cname)
        kept_stats.append((cname, len(paths), "KEEP"))
        (train_root / cname).mkdir(parents=True, exist_ok=True)
        (test_root / cname).mkdir(parents=True, exist_ok=True)
        for i, p in enumerate(paths):
            dst_dir = (train_root / cname) if i < min_train else (test_root / cname)
            # 直接 copy 原檔名（保留 Unicode），後續用 unicode-safe 讀法
            shutil.copy2(p, dst_dir / p.name)

    print("Class collection stats:")
    for cname, cnt, tag in kept_stats:
        print(f"  - {cname}: {cnt} images -> {tag}")

    return class_names, train_root, test_root

def read_gray(path: Path):
    im = imread_unicode(path, cv2.IMREAD_GRAYSCALE)
    if im is None:
        raise ValueError(f"Read fail: {path}")
    # 自動二值 + 反相，變白字黑底
    _, th = cv2.threshold(im, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    im = 255 - th
    h, w = im.shape
    scale = IMG_SIZE / max(h, w) if max(h, w) > 0 else 1.0
    new_w = max(int(w * scale), 1)
    new_h = max(int(h * scale), 1)
    im_resized = cv2.resize(im, (new_w, new_h), interpolation=cv2.INTER_AREA)
    canvas = np.zeros((IMG_SIZE, IMG_SIZE), dtype=np.uint8)
    y0 = (IMG_SIZE - im_resized.shape[0]) // 2
    x0 = (IMG_SIZE - im_resized.shape[1]) // 2
    canvas[y0:y0 + im_resized.shape[0], x0:x0 + im_resized.shape[1]] = im_resized
    return canvas

def augment_once(img: np.ndarray):
    H, W = img.shape
    imgf = img.astype(np.float32) / 255.0
    angle = np.random.uniform(-12, 12)
    scale = np.random.uniform(0.9, 1.1)
    tx = np.random.uniform(-0.06, 0.06) * W
    ty = np.random.uniform(-0.06, 0.06) * H
    shear = np.random.uniform(-0.2, 0.2)

    M = cv2.getRotationMatrix2D((W / 2, H / 2), angle, scale)
    M[:, 2] += [tx, ty]
    img1 = cv2.warpAffine(
        imgf, M, (W, H),
        flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0.0
    )
    M_shear = np.array([[1, shear, 0], [0, 1, 0]], dtype=np.float32)
    img2 = cv2.warpAffine(
        img1, M_shear, (W, H),
        flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0.0
    )

    if np.random.rand() < 0.5:
        k = np.random.choice([1, 2])
        kernel = np.ones((k, k), np.uint8)
        img3 = cv2.dilate((img2 * 255).astype(np.uint8), kernel, iterations=1)
    else:
        kernel = np.ones((1, 1), np.uint8)
        img3 = cv2.erode((img2 * 255).astype(np.uint8), kernel, iterations=1)

    if np.random.rand() < 0.3:
        noise = np.random.normal(0, 0.02, img3.shape).astype(np.float32)
        img3 = np.clip(img3 / 255.0 + noise, 0, 1.0)
        img3 = (img3 * 255).astype(np.uint8)

    return img3

def build_augmented_dataset(class_names, train_src: Path, test_src: Path, aug_dst: Path, test_dst: Path):
    ensure_clean(aug_dst.parent)
    aug_dst.mkdir(parents=True, exist_ok=True)
    test_dst.mkdir(parents=True, exist_ok=True)

    read_errors = []

    # 測試集
    for c in tqdm(class_names, desc="Prepare test"):
        (test_dst / c).mkdir(parents=True, exist_ok=True)
        for p in sorted((test_src / c).glob("*"), key=natural_key):
            if not is_image_path(p):
                continue
            try:
                proc = read_gray(p)
                ok = imwrite_unicode((test_dst / c) / (Path(p).stem + ".png"), proc)
                if not ok:
                    read_errors.append(str(p))
            except Exception:
                read_errors.append(str(p))

    # 訓練集
    for c in tqdm(class_names, desc="Augment train"):
        src_c = train_src / c
        dst_c = aug_dst / c
        dst_c.mkdir(parents=True, exist_ok=True)
        base = []
        for p in sorted(src_c.glob("*"), key=natural_key):
            if not is_image_path(p):
                continue
            try:
                base.append(read_gray(p))
            except Exception:
                read_errors.append(str(p))
        count = 0
        for i, im in enumerate(base):
            imwrite_unicode(dst_c / f"orig_{i:03d}.png", im)
            count += 1
        i = 0
        while count < TARGET_PER_CLASS_TRAIN and len(base) > 0:
            im = base[i % len(base)]
            aug = augment_once(im)
            imwrite_unicode(dst_c / f"aug_{count:04d}.png", aug)
            count += 1
            i += 1

    if read_errors:
        print(f"Warning: {len(read_errors)} files failed to process (read or write). Example:")
        for s in read_errors[:10]:
            print("  -", s)

AUTO = tf.data.AUTOTUNE

def decode_image(path):
    img = tf.io.read_file(path)
    img = tf.io.decode_png(img, channels=1)
    img = tf.image.convert_image_dtype(img, tf.float32)
    return img

def build_dataset(root: Path, class_names, batch, shuffle=False):
    files, labels = [], []
    for i, c in enumerate(class_names):
        class_dir = Path(root) / c
        if not class_dir.exists():
            continue
        for p in sorted(class_dir.glob("*.png"), key=natural_key):
            files.append(str(p))
            labels.append(i)

    if len(files) == 0:
        raise RuntimeError(
            f"No images found under {root}. "
            f"Classes discovered: {class_names}. "
            f"Check DATASET_DIR and MIN_PER_CLASS_TRAIN."
        )

    files_ds = tf.data.Dataset.from_tensor_slices(tf.constant(files, dtype=tf.string))
    labels_ds = tf.data.Dataset.from_tensor_slices(tf.constant(labels, dtype=tf.int32))
    ds = tf.data.Dataset.zip((files_ds, labels_ds))

    if shuffle:
        ds = ds.shuffle(buffer_size=len(files), seed=SEED, reshuffle_each_iteration=True)

    def _map(f, y):
        x = decode_image(f)
        x = tf.image.resize(x, (IMG_SIZE, IMG_SIZE))
        return x, tf.one_hot(y, depth=len(class_names))

    ds = ds.map(_map, num_parallel_calls=AUTO).batch(batch).prefetch(AUTO)
    return ds, len(files)

def build_cnn_baseline(num_classes, img_size):
    inputs = tf.keras.Input(shape=(img_size, img_size, 1))
    x = inputs
    for filters in [32, 64, 128]:
        x = tf.keras.layers.Conv2D(filters, 3, padding="same", use_bias=False)(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.ReLU()(x)
        x = tf.keras.layers.MaxPool2D()(x)
    x = tf.keras.layers.Conv2D(256, 3, padding="same", use_bias=False)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dropout(0.3)(x)
    outputs = tf.keras.layers.Dense(num_classes, activation="softmax")(x)
    return tf.keras.Model(inputs, outputs, name="cnn_baseline")

def conv_bn_relu(x, filters, kernel=3, stride=1):
    x = tf.keras.layers.Conv2D(filters, kernel, strides=stride, padding="same", use_bias=False)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)
    return x

def res_block(x, filters, down=False):
    shortcut = x
    stride = 2 if down else 1
    x = conv_bn_relu(x, filters, 3, stride)
    x = tf.keras.layers.Conv2D(filters, 3, padding="same", use_bias=False)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    if down or shortcut.shape[-1] != filters:
        shortcut = tf.keras.layers.Conv2D(filters, 1, strides=stride, padding="same", use_bias=False)(shortcut)
        shortcut = tf.keras.layers.BatchNormalization()(shortcut)
    x = tf.keras.layers.Add()([x, shortcut])
    x = tf.keras.layers.ReLU()(x)
    return x

def build_resnet_small(num_classes, img_size):
    inputs = tf.keras.Input(shape=(img_size, img_size, 1))
    x = conv_bn_relu(inputs, 32, 3, 1)
    x = res_block(x, 32, down=False)
    x = res_block(x, 64, down=True)
    x = res_block(x, 64, down=False)
    x = res_block(x, 128, down=True)
    x = res_block(x, 128, down=False)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dropout(0.3)(x)
    outputs = tf.keras.layers.Dense(num_classes, activation="softmax")(x)
    return tf.keras.Model(inputs, outputs, name="resnet_small")

def get_optimizer(steps_per_epoch):
    lr_sched = tf.keras.optimizers.schedules.CosineDecayRestarts(
        initial_learning_rate=1e-3,
        first_decay_steps=max(steps_per_epoch * 5, 1),
        t_mul=2.0,
        m_mul=0.8,
        alpha=1e-5,
    )
    if USE_KERAS_ADAMW:
        opt = tf.keras.optimizers.AdamW(learning_rate=lr_sched, weight_decay=1e-4)
    else:
        import tensorflow_addons as tfa
        opt = tfa.optimizers.AdamW(learning_rate=lr_sched, weight_decay=1e-4)
    return opt

def compile_and_train(model, train_ds, val_ds, steps_per_epoch, name):
    opt = get_optimizer(steps_per_epoch)
    model.compile(
        optimizer=opt,
        loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.1),
        metrics=[
            tf.keras.metrics.TopKCategoricalAccuracy(k=1, name="top1"),
            tf.keras.metrics.TopKCategoricalAccuracy(k=5, name="top5"),
        ],
    )
    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(CKPT_DIR / f"{name}_best.keras"),
            monitor="val_top1",
            mode="max",
            save_best_only=True,
        ),
        tf.keras.callbacks.EarlyStopping(monitor="val_top1", mode="max", patience=8, restore_best_weights=True),
        tf.keras.callbacks.CSVLogger(str(RESULTS_DIR / f"{name}_log.csv")),
    ]
    hist = model.fit(train_ds, validation_data=val_ds, epochs=EPOCHS, callbacks=callbacks)
    return hist

def evaluate_and_save(model, val_ds, class_names, name):
    res = model.evaluate(val_ds, return_dict=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_DIR / f"{name}_eval.json", "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False, indent=2)
    with open(RESULTS_DIR / "class_names.json", "w", encoding="utf-8") as f:
        json.dump(class_names, f, ensure_ascii=False, indent=2)
    model.save(str(RESULTS_DIR / f"{name}_final.keras"))
    print(name, res)

def main():
    WORKDIR.mkdir(exist_ok=True)

    class_names, train_raw, test_raw = split_train_test_from_filenames(
        Path(DATASET_DIR), RAW_SPLIT_DIR, MIN_PER_CLASS_TRAIN
    )
    print(f"Classes: {len(class_names)}")
    if len(class_names) == 0:
        raise RuntimeError(
            "No classes were collected. "
            "Check DATASET_DIR and that filenames follow '<class>_<index>.ext'."
        )

    build_augmented_dataset(class_names, train_raw, test_raw, AUG_TRAIN_DIR, TEST_DIR)

    train_ds, n_train = build_dataset(AUG_TRAIN_DIR, class_names, BATCH_SIZE, shuffle=True)
    test_ds, n_test = build_dataset(TEST_DIR, class_names, BATCH_SIZE, shuffle=False)
    steps = max(math.ceil(n_train / BATCH_SIZE), 1)
    print(f"Train samples: {n_train}, Test samples: {n_test}, Steps/epoch: {steps}")

    model1 = build_cnn_baseline(len(class_names), IMG_SIZE)
    compile_and_train(model1, train_ds, test_ds, steps, name="cnn_baseline")
    evaluate_and_save(model1, test_ds, class_names, "cnn_baseline")

    model2 = build_resnet_small(len(class_names), IMG_SIZE)
    compile_and_train(model2, train_ds, test_ds, steps, name="resnet_small")
    evaluate_and_save(model2, test_ds, class_names, "resnet_small")

if __name__ == "__main__":
    main()