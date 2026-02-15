import cv2
import numpy as np

face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")


def extract_face(image_path):
    img = cv2.imread(image_path)
    if img is None:
        return None

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    if len(faces) == 0:
        return None

    (x, y, w, h) = faces[0]
    face = gray[y:y+h, x:x+w]
    face = cv2.resize(face, (100, 100))
    return face.flatten()


def group_faces(image_paths, threshold=3000):
    encodings = []
    valid_paths = []

    for path in image_paths:
        face = extract_face(path)
        if face is not None:
            encodings.append(face)
            valid_paths.append(path)

    groups = []

    for i, enc in enumerate(encodings):
        placed = False

        for group in groups:
            ref_enc = encodings[group[0]]
            dist = np.linalg.norm(enc - ref_enc)

            if dist < threshold:
                group.append(i)
                placed = True
                break

        if not placed:
            groups.append([i])

    result = []
    for group in groups:
        result.append([valid_paths[i] for i in group])

    return result
