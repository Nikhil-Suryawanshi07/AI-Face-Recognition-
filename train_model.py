import os
import cv2
import numpy as np
from skimage.feature import hog
from sklearn.svm import LinearSVC
import joblib

data = []
labels = []

def process_folder(folder, label):
    for file in os.listdir(folder):
        path = os.path.join(folder, file)
        img = cv2.imread(path)

        if img is None:
            continue

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray = cv2.resize(gray, (128,128))

        features = hog(gray,
                       orientations=9,
                       pixels_per_cell=(8,8),
                       cells_per_block=(2,2))

        data.append(features)
        labels.append(label)

process_folder("dataset/real", 1)
process_folder("dataset/fake", 0)

print("Training model...")
model = LinearSVC()
model.fit(data, labels)

joblib.dump(model, "model.pkl")
print("Model saved as model.pkl")
