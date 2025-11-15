import os, glob, shutil
from tqdm import tqdm

SRC = "data/normalized/FF"
TRAIN = "data/train/FF"
TEST = "data/test/FF"

os.makedirs(TRAIN, exist_ok=True)
os.makedirs(TEST, exist_ok=True)

files = sorted(glob.glob(os.path.join(SRC, "*.png")))
print("Total files:", len(files))

# 前 40 張放訓練
train_files = files[:40]
# 強制只取 10 張放測試
test_files = files[40:50]

for f in tqdm(train_files, desc="Copy Train"):
    shutil.copy(f, os.path.join(TRAIN, os.path.basename(f)))

for f in tqdm(test_files, desc="Copy Test"):
    shutil.copy(f, os.path.join(TEST, os.path.basename(f)))
