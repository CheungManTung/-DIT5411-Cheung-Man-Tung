import os, glob, cv2
import numpy as np
from tensorflow.keras.models import load_model


MODEL_PATH = "models/ff_model.h5"

TEST_DIR = "data/test/FF"

IMG_SIZE = 64


model = load_model(MODEL_PATH)


files = sorted(glob.glob(os.path.join(TEST_DIR, "*.png")))
x_test = []
for f in files:
    img = cv2.imread(f, cv2.IMREAD_GRAYSCALE)  
    if img is None:
        continue
    img = img / 255.0                          
    x_test.append(img.reshape(IMG_SIZE, IMG_SIZE, 1)) 
x_test = np.array(x_test)


if len(x_test) == 0:
    print("⚠️ No test images found in data/test/FF/")
    exit(0)


preds = model.predict(x_test, verbose=0)

pred_labels = (preds > 0.5).astype(int).flatten()


true_labels = np.zeros(len(x_test), dtype=int)


accuracy = np.mean(pred_labels == true_labels)
print(f"✅ Test Accuracy: {accuracy * 100:.2f}% (on {len(x_test)} images)")


errors = []
for i, f in enumerate(files):
    if pred_labels[i] != true_labels[i]:
        errors.append(os.path.basename(f))

if errors:
    print("❌ Misclassified files:")
    for name in errors:
        print(" -", name)
else:
    print("🎉 All test samples classified correctly")
