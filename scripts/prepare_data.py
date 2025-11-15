import os, cv2, glob
from tqdm import tqdm

SRC_DIR = "data/raw"
DST_DIR = "data/normalized"
IMG_SIZE = 64

os.makedirs(DST_DIR, exist_ok=True)

classes = sorted([d for d in os.listdir(SRC_DIR) if os.path.isdir(os.path.join(SRC_DIR, d))])

for cls in tqdm(classes, desc="Normalize"):
    src_cls = os.path.join(SRC_DIR, cls)
    dst_cls = os.path.join(DST_DIR, cls)
    os.makedirs(dst_cls, exist_ok=True)
    for fp in glob.glob(os.path.join(src_cls, "*.png")):
        img = cv2.imread(fp, cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue
        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
        img = cv2.equalizeHist(img)
        cv2.imwrite(os.path.join(dst_cls, os.path.basename(fp)), img)
