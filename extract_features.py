import os
import cv2
import numpy as np
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.models import Model

DATASET_PATH = "101_ObjectCategories"

images = []
originals = []

for folder in os.listdir(DATASET_PATH):
    folder_path = os.path.join(DATASET_PATH, folder)
    if not os.path.isdir(folder_path):
        continue

    for img_name in os.listdir(folder_path)[:5]:
        img_path = os.path.join(folder_path, img_name)
        img = cv2.imread(img_path)

        if img is None:
            continue

        img = cv2.resize(img, (224, 224))
        images.append(img)
        originals.append(img)

images = np.array(images)

base_model = ResNet50(weights="imagenet", include_top=False, pooling="avg")
model = Model(inputs=base_model.input, outputs=base_model.output)

images = preprocess_input(images)
features = model.predict(images, batch_size=16)

np.save("features.npy", features)
np.save("images.npy", np.array(originals))

print("✅ Feature extraction completed successfully")
