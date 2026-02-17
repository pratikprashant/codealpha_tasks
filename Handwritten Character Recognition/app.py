from flask import Flask, render_template, request
import numpy as np
from tensorflow.keras.models import load_model
from PIL import Image
import os

app = Flask(__name__)

UPLOAD_FOLDER = "static/uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

model = load_model("emnist_cnn.h5")

mapping = {}
with open("emnist-balanced-mapping.txt") as f:
    for line in f:
        key, val = line.strip().split()
        mapping[int(key)] = chr(int(val))

def preprocess_image(image_path):
    # Load image
    img = Image.open(image_path).convert("L")  # grayscale
    img = img.resize((28, 28))

    # Convert to numpy
    img = np.array(img)

    # Normalize
    img = img / 255.0

    # Fix EMNIST orientation
    img = np.rot90(img, k=1)
    img = np.fliplr(img)

    # Add batch & channel dimensions
    img = img.reshape(1, 28, 28, 1)

    return img

@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    image_path = None

    if request.method == "POST":
        if "file" not in request.files:
            return render_template("index.html")

        file = request.files["file"]

        if file.filename == "":
            return render_template("index.html")

        # Save uploaded image
        image_path = os.path.join(app.config["UPLOAD_FOLDER"], file.filename)
        file.save(image_path)

        # Preprocess & predict
        img = preprocess_image(image_path)
        preds = model.predict(img)
        class_id = np.argmax(preds)

        prediction = mapping[class_id]

    return render_template(
        "index.html",
        prediction=prediction,
        image_path=image_path
    )

if __name__ == "__main__":
    app.run(debug=True)
