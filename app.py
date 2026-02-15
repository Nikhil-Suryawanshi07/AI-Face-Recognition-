from flask import Flask, render_template, request, send_from_directory
import os
from ai_detector import detect_ai
from face_grouping import group_faces

# FIRST create app (ye sabse pehle hona chahiye)
app = Flask(__name__)

UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)


@app.route("/")
def home():
    return render_template("index.html", images=None, detections=None, groups=None)


@app.route("/upload", methods=["POST"])
def upload():
    files = request.files.getlist("files")

    file_paths = []
    file_names = []
    detection_results = []

    # Save files
    for file in files:
        path = os.path.join(UPLOAD_FOLDER, file.filename)
        file.save(path)

        file_paths.append(path)
        file_names.append(file.filename)

    # AI Detection
    for path in file_paths:
        result = detect_ai(path)
        detection_results.append({
            "name": os.path.basename(path),
            "result": result
        })

    # Face grouping
    groups = group_faces(file_paths)

    grouped_people = []
    for g in groups:
        grouped_people.append([os.path.basename(x) for x in g])

    return render_template(
        "index.html",
        images=file_names,
        detections=detection_results,
        groups=grouped_people
    )


# Serve uploaded images to browser
@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(UPLOAD_FOLDER, filename)


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)

