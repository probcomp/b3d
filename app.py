from flask import Flask, send_file, request, jsonify
import os
from werkzeug.utils import secure_filename
import uuid

app = Flask(__name__)

UPLOAD_FOLDER = "uploads"
ALLOWED_EXTENSIONS = {"r3d"}

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER


def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route("/")
def index():
    return send_file("index.html")


@app.route("/upload", methods=["POST"])
def upload_file():
    if "file" not in request.files:
        return jsonify({"error": "No file part"}), 400
    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "No selected file"}), 400
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
        file.save(file_path)
        task_id = str(uuid.uuid4())
        # Start processing in a background task
        # process_video.delay(file_path, task_id)
        return jsonify({"task_id": task_id}), 200
    return jsonify({"error": "Invalid file type"}), 400


@app.route("/status/<task_id>", methods=["GET"])
def get_status(task_id):
    # Check the status of the video processing task
    # This is a placeholder - you'll need to implement task tracking
    return jsonify({"status": "completed", "video_url": f"/video/{task_id}"})


@app.route("/video/<task_id>", methods=["GET"])
def get_video(task_id):
    # Return the processed video file
    # This is a placeholder - you'll need to implement video storage and retrieval
    # video_path = f'path/to/processed/videos/{task_id}.mp4'
    video_path = "lysol_static.r3d.graphics_edits.mp4"
    return send_file(video_path, mimetype="video/mp4")


if __name__ == "__main__":
    app.run(debug=True)
