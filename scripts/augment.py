import os, cv2, glob
import numpy as np
from tqdm import tqdm

SRC = "data/train/FF"
DST = "data/augmented/FF"
os.makedirs(DST, exist_ok=True)

files = glob.glob(os.path.join(SRC, "*.png"))

def augment(img):
    augmented = []
    
    augmented.append(cv2.flip(img, 1))
    
    for angle in [10, -10, 20, -20]:
        M = cv2.getRotationMatrix2D((32, 32), angle, 1.0)
        augmented.append(cv2.warpAffine(img, M, (64, 64)))
    
    noise = np.random.normal(0, 10, img.shape).astype(np.uint8)
    augmented.append(cv2.add(img, noise))
    return augmented

count = 0
for f in tqdm(files, desc="Augment"):
    img = cv2.imread(f, cv2.IMREAD_GRAYSCALE)
    if img is None:
        continue
    cv2.imwrite(os.path.join(DST, f"aug_{count}.png"), img)
    count += 1
    for aug in augment(img):
        cv2.imwrite(os.path.join(DST, f"aug_{count}.png"), aug)
        count += 1

print("Total augmented images:", count)
