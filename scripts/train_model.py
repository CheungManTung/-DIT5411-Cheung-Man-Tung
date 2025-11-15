import os, glob, cv2
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.callbacks import EarlyStopping


TRAIN_DIR = "data/augmented/FF"
TEST_DIR = "data/test/FF"
IMG_SIZE = 64

def load_images(path, label):
    files = glob.glob(os.path.join(path, "*.png"))
    data, labels = [], []
    for f in files:
        img = cv2.imread(f, cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue
        img = img / 255.0
        data.append(img.reshape(IMG_SIZE, IMG_SIZE, 1))
        labels.append(label)
    return np.array(data), np.array(labels)


x_train, y_train = load_images(TRAIN_DIR, 0)
x_test, y_test = load_images(TEST_DIR, 0)


os.makedirs("models", exist_ok=True)


model = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(IMG_SIZE, IMG_SIZE, 1)),
    MaxPooling2D((2,2)),
    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D((2,2)),
    Flatten(),
    Dense(64, activation='relu'),
    Dense(1, activation='sigmoid') 
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])


model.fit(
    x_train, y_train,
    epochs=10,
    batch_size=16,
    validation_data=(x_test, y_test),
    callbacks=[EarlyStopping(patience=3, restore_best_weights=True)]
)


model.save("models/ff_model.h5")
print("✅ models is save：models/ff_model.h5")
