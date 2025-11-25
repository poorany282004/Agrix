from flask import Flask, render_template, request, jsonify, send_from_directory, url_for
from werkzeug.utils import secure_filename
import os
from PIL import Image
import numpy as np
import tensorflow as tf

# ----------------------------------
# CONFIG
# ----------------------------------
UPLOAD_FOLDER = "uploads/images"
DISEASE_DATA_FOLDER = "static/diseases"
MODEL_PATH = "saved_model/agrix_model.keras"
LABELS_PATH = "saved_model/labels.txt"

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

# ----------------------------------
# LOAD MODEL
# ----------------------------------
model = None
class_names = []

if os.path.exists(MODEL_PATH):
    print("✅ Loading model...")
    model = tf.keras.models.load_model(MODEL_PATH)

    if os.path.exists(LABELS_PATH):
        with open(LABELS_PATH, "r") as f:
            class_names = [l.strip() for l in f.readlines()]
    else:
        class_names = sorted(os.listdir("dataset/train"))
else:
    print("⚠ No model found. Running fallback only")

# ----------------------------------
# FALLBACK
# ----------------------------------
def fallback_predict():
    return "healthy", 0.5

# ----------------------------------
# LOAD TEXT CONTENT
# ----------------------------------
def load_disease_content(label):
    file_path = os.path.join(DISEASE_DATA_FOLDER, f"{label}.txt")
    if not os.path.exists(file_path):
        file_path = os.path.join(DISEASE_DATA_FOLDER, "default.txt")

    about, symptoms, prevention = "", [], []
    section = None

    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            txt = line.strip()

            if txt.startswith("ABOUT:"):
                section = "ABOUT"; continue
            if txt.startswith("SYMPTOMS:"):
                section = "SYMPTOMS"; continue
            if txt.startswith("PREVENTION:"):
                section = "PREVENTION"; continue

            if section == "ABOUT":
                about += txt + " "
            elif section == "SYMPTOMS" and txt:
                symptoms.append(txt)
            elif section == "PREVENTION" and txt:
                prevention.append(txt)

    return about.strip(), symptoms, prevention

# ----------------------------------
# STAGE LOGIC
# ----------------------------------
def find_stage(prob):
    if prob < 0.40:
        return "Healthy"
    elif prob < 0.70:
        return "Partially Affected"
    return "Fully Affected"

# ----------------------------------
# ROUTES
# ----------------------------------
@app.route("/")
def home():
    return render_template("cover.html")

@app.route("/scan")
def scan():
    return render_template("scan.html")

@app.route("/uploads/<path:filename>")
def serve_uploaded(filename):
    return send_from_directory(UPLOAD_FOLDER, filename)

# ----------------------------------
# PREDICT → return HTML result page
# ----------------------------------
@app.route("/predict", methods=["POST"])
def predict():
    file = request.files.get("image")
    if not file:
        return "No file uploaded", 400

    filename = secure_filename(file.filename)
    filepath = os.path.join(UPLOAD_FOLDER, filename)
    file.save(filepath)

    # MODEL PREDICTION
    if model:
        img = Image.open(filepath).convert("RGB").resize((224, 224))
        arr = np.array(img) / 255.0
        arr = arr.reshape(1, 224, 224, 3)

        preds = model.predict(arr)
        idx = int(np.argmax(preds))
        pred_label = class_names[idx]
        pred_prob = float(preds[0][idx])
    else:
        pred_label, pred_prob = fallback_predict()

    stage = find_stage(pred_prob)
    about, symptoms, prevention = load_disease_content(pred_label)

    # Render FINAL HTML directly
    return render_template(
        "result.html",
        image_url=url_for("serve_uploaded", filename=filename),
        disease_name=pred_label.replace("_", " ").title(),
        stage=stage,
        description=about,
        symptoms=symptoms,
        prevention=prevention,
        probability=round(pred_prob, 3)
    )

# ----------------------------------
# RUN SERVER
# ----------------------------------
if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
