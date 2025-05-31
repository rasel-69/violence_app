from flask import Flask, request, render_template
from werkzeug.utils import secure_filename
import os
import cv2
from ultralytics import YOLO
import json
import smtplib
import requests
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.image import MIMEImage

app = Flask(__name__)

# Upload folder config
UPLOAD_FOLDER = "static/uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

# Load YOLO model
model = YOLO("best.pt")

# Email config
EMAIL_SENDER = "rasel4897981@gmail.com"
EMAIL_PASSWORD = "kkwsehrfcejnoahg"  # App password
EMAIL_RECEIVER = "bentimo498@gmail.com"

# Google Maps Static API Key
GOOGLE_MAPS_API_KEY = "AIzaSyCF5g9mbR-q0v2T90xWAr4t9JWfPLP_aGo"  # Replace with your key

def send_email_notification(video_filename, latitude, longitude):
    subject = "🚨 Violence Detected in Your Area"
    body_text = f"""
Alert: Violence has been detected in your Area.

Location:
Latitude: {latitude}
Longitude: {longitude}

See the map image below for the exact location.
"""

    msg = MIMEMultipart("related")
    msg["Subject"] = subject
    msg["From"] = EMAIL_SENDER
    msg["To"] = EMAIL_RECEIVER

    # Create the alternative part for HTML and plain text
    msg_alternative = MIMEMultipart("alternative")
    msg.attach(msg_alternative)

    # HTML body with embedded image (linked with CID)
    body_html = body_text.replace('\n', '<br>')
    html_body = f"""
    <html>
      <body>
        <p>{body_html}</p>
        <img src="cid:map_image">
      </body>
    </html>
    """

    msg_alternative.attach(MIMEText(body_text, "plain"))
    msg_alternative.attach(MIMEText(html_body, "html"))

    try:
        # Fetch map image from Google Static Maps API
        map_url = (
            f"https://maps.googleapis.com/maps/api/staticmap?"
            f"center={latitude},{longitude}&zoom=15&size=600x300"
            f"&markers=color:red%7C{latitude},{longitude}&key={GOOGLE_MAPS_API_KEY}"
        )
        map_response = requests.get(map_url)

        if map_response.status_code == 200:
            image = MIMEImage(map_response.content)
            image.add_header("Content-ID", "<map_image>")
            image.add_header("Content-Disposition", "inline", filename="map.png")
            msg.attach(image)
        else:
            print("Failed to fetch map image from Google Static Maps API.")

        # Send the email
        with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
            server.login(EMAIL_SENDER, EMAIL_PASSWORD)
            server.send_message(msg)
            print("Email with map sent successfully!")
    except Exception as e:
        print(f"Error sending email: {e}")


def process_video(video_path, json_output_path):
    cap = cv2.VideoCapture(video_path)
    detection_results = []
    violence_found = False

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
                violence_found = True
                break

        if violence_detected:
            confidence = float(det[4])  # Confidence score
            detection_results.append({"label": "Violence", "confidence": round(confidence * 100, 2)})
        else:
            confidence = float(det[4])
            detection_results.append({"label": "Non-Violence", "confidence": round(confidence * 200, 2)})


    cap.release()

    with open(json_output_path, "w") as f:
        json.dump(detection_results, f)

    return detection_results, violence_found

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        file = request.files.get("file")
        if not file or file.filename == "":
            return render_template("index.html", message="No file uploaded")

        # Location from hidden form fields
        latitude = request.form.get("latitude")
        longitude = request.form.get("longitude")

        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
        file.save(file_path)

        json_path = os.path.join(app.config["UPLOAD_FOLDER"], f"{filename}_detections.json")

        detection_results, violence_found = process_video(file_path, json_path)


        if violence_found and latitude and longitude:
            send_email_notification(filename, latitude, longitude)

        return render_template("index.html",
                               input_video=file_path,
                               detection_results=json.dumps(detection_results))
    return render_template("index.html")

if __name__ == "__main__":
    app.run(debug=True)
