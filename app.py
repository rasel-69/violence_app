from flask import Flask, request, render_template
from werkzeug.utils import secure_filename
import os
import cv2
from ultralytics import YOLO

app = Flask(__name__)

# Upload folder config
UPLOAD_FOLDER = "static/uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

# Load YOLO model
model = YOLO("best.pt")

import json

def process_video(video_path, json_output_path):
    cap = cv2.VideoCapture(video_path)
    detection_results = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        results = model(frame)
        detections = results[0].boxes.data.cpu().numpy() if results[0].boxes is not None else []

        violence_detected = False
        for det in detections:
            cls = int(det[5])
            if results[0].names[cls].lower() == "violence":
                violence_detected = True
                break

        detection_results.append("Violence" if violence_detected else "Non-Violence")

    cap.release()

    # Save detections to JSON
    with open(json_output_path, "w") as f:
        json.dump(detection_results, f)

    return detection_results


@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        file = request.files.get("file")
        if not file or file.filename == "":
            return render_template("index.html", message="No file uploaded")

        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
        file.save(file_path)

        # JSON path to store frame-by-frame results
        json_path = os.path.join(app.config["UPLOAD_FOLDER"], f"{filename}_detections.json")

        # Process video and save per-frame detection
        detection_results = process_video(file_path, json_path)

        return render_template("index.html",
                               input_video=file_path,
                               detection_results=json.dumps(detection_results))  # pass to JS
    return render_template("index.html")

if __name__ == "__main__":
    app.run(debug=True)
