from flask import Flask, send_file, request, jsonify
import os
from werkzeug.utils import secure_filename
import uuid
import scripts.acquire_object_model as acquire
from threading import Thread

app = Flask(__name__)

UPLOAD_FOLDER = "uploads"
OUTPUT_FOLDER = "converted"
ALLOWED_EXTENSIONS = {"r3d"}

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["OUTPUT_FOLDER"] = OUTPUT_FOLDER

# Ensure the upload and output directories exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)


def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route("/")
def index():
    return send_file("index.html")


def convert_to_mp4(input_path, output_path, task_id):
    try:
        acquire.acquire(input_path, output_path)
        # After conversion is done, you might want to update a status in a database
        print(f"Conversion completed for task {task_id}")
    except Exception as e:
        print(f"Conversion failed for task {task_id}: {str(e)}")


@app.route("/upload", methods=["POST"])
def upload_file():
    if "file" not in request.files:
        return jsonify({"error": "No file part"}), 400
    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "No selected file"}), 400
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        input_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
        file.save(input_path)

        task_id = str(uuid.uuid4())
        output_filename = f"{task_id}.mp4"
        output_path = os.path.join(app.config["OUTPUT_FOLDER"], output_filename)

        # Start conversion in a background thread
        Thread(target=convert_to_mp4, args=(input_path, output_path, task_id)).start()

        return jsonify({"task_id": task_id}), 200
    return jsonify({"error": "Invalid file type"}), 400


@app.route("/status/<task_id>", methods=["GET"])
def get_status(task_id):
    output_path = os.path.join(app.config["OUTPUT_FOLDER"], f"{task_id}.mp4")
    if os.path.exists(output_path):
        return jsonify({"status": "completed", "video_url": f"/video/{task_id}"})
    else:
        return jsonify({"status": "processing"})


@app.route("/video/<task_id>", methods=["GET"])
def get_video(task_id):
    video_path = os.path.join(app.config["OUTPUT_FOLDER"], f"{task_id}.mp4")
    if os.path.exists(video_path):
        return send_file(video_path, mimetype="video/mp4")
    else:
        return jsonify({"error": "Video not found"}), 404


if __name__ == "__main__":
    app.run(debug=True)
