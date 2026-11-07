import os
import json
import numpy as np
from PIL import Image
import onnxruntime as ort
from flask import Flask, request, jsonify

app = Flask(__name__)

MODEL_PATH = "champion_model.onnx"
METADATA_PATH = "model_metadata.json"

threshold = 0.5
if os.path.exists(METADATA_PATH):
    with open(METADATA_PATH, "r") as f:
        meta = json.load(f)
        threshold = meta.get("best_threshold", meta.get("threshold", 0.5))

session = ort.InferenceSession(MODEL_PATH)
input_name = session.get_inputs()[0].name

def preprocess_image(image_file):
    img = Image.open(image_file).convert("RGB")
    img = img.resize((224, 224))
    arr = np.array(img, dtype=np.float32) / 255.0
    return np.expand_dims(arr, axis=0)

@app.route("/", methods=["GET"])
@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "healthy", "runtime": "onnx-cpu"}), 200

@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400
    
    file = request.files["file"]
    tensor = preprocess_image(file)
    outputs = session.run(None, {input_name: tensor})
    raw_score = float(outputs[0][0][0])
    
    prediction = "RealArt" if raw_score >= threshold else "AiArtData"
    confidence = raw_score if prediction == "RealArt" else (1.0 - raw_score)
    
    return jsonify({
        "status": "success",
        "prediction": prediction,
        "confidence": round(confidence * 100, 2),
        "raw_score": round(raw_score, 4),
        "threshold_applied": threshold
    }), 200

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))