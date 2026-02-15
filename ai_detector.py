import cv2
import numpy as np
import joblib
from skimage.feature import hog

# Load trained model
model = joblib.load("model.pkl")

def detect_ai(image_path):
    img = cv2.imdecode(np.fromfile(image_path, dtype=np.uint8), cv2.IMREAD_COLOR)

    if img is None:
        return "Invalid Image"

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, (128,128))

    features = hog(gray,
                   orientations=9,
                   pixels_per_cell=(8,8),
                   cells_per_block=(2,2))

    prediction = model.predict([features])[0]

    if prediction == 1:
        return "Real Image"
    else:
        return "AI Generated"
