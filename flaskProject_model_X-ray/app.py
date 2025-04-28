from flask import Flask, request, render_template, jsonify
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from PIL import Image
import os

app = Flask(__name__)
model = load_model("X-ray-covid_model_3_v2.h5")

# Thư mục lưu ảnh upload
UPLOAD_FOLDER = "static/uploads"
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

class_labels = ["Bacterial", "COVID-19", "Lung Opacity", "Normal", "Viral"]

# Hàm tiền xử lý ảnh
def preprocess_image(image_path):
    img = Image.open(image_path).convert("L")
    img = img.resize((256, 256))
    img = np.array(img) / 255.0
    img = np.expand_dims(img, axis=0)
    return img

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/upload", methods=["POST"])
def upload_file():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "No selected file"}), 400

    file_path = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(file_path)

    return jsonify({"image_url": file_path.replace("\\", "/")})

@app.route("/predict", methods=["POST"])
def predict():
    data = request.json
    image_path = data.get("image_path")

    if not image_path:
        return jsonify({"error": "No image path provided"}), 400

    img = preprocess_image(image_path)
    predictions = model.predict(img)[0]
    confidence = {class_labels[i]: round(float(predictions[i]) * 100, 2) for i in range(len(class_labels))}

    return jsonify({"confidence": confidence})

if __name__ == "__main__":
    app.run(debug=True)
