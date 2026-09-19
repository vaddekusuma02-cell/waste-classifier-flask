
from flask import Flask, request, render_template_string
import tensorflow as tf
import numpy as np
import os

app = Flask(__name__)

# Load the trained model
model = tf.keras.models.load_model("waste_classifier.keras")

# Class names
classes = ["organic", "plastic"]

# Web page
html = """
<!DOCTYPE html>
<html>
<head>
    <title>Waste Classifier</title>

    <style>
        body {
            font-family: Arial;
            text-align: center;
            margin-top: 80px;
        }

        .box {
            width: 500px;
            margin: auto;
            padding: 30px;
            border: 2px solid #333;
            border-radius: 15px;
        }

        button {
            padding: 10px 25px;
            font-size: 16px;
            cursor: pointer;
        }

        h1 {
            margin-bottom: 30px;
        }
    </style>
</head>

<body>

<div class="box">

    <h1>♻️ Waste Classifier</h1>

    <form method="POST" enctype="multipart/form-data">

        <input type="file" name="image" accept="image/*" required>

        <br><br>

        <button type="submit">Predict</button>

    </form>

    {% if prediction %}

        <h2>Prediction: {{ prediction }}</h2>

        <h3>Confidence: {{ confidence }}%</h3>

        <h3>Guidance:</h3>

        <p>{{ guidance }}</p>

    {% endif %}

</div>

</body>
</html>
"""

@app.route("/", methods=["GET", "POST"])
def home():

    prediction = None
    confidence = None
    guidance = None

    if request.method == "POST":

        image = request.files["image"]

        # Save uploaded image temporarily
        image_path = "uploaded_image.jpg"
        image.save(image_path)

        # Load image
        img = tf.keras.utils.load_img(
            image_path,
            target_size=(180, 180)
        )

        # Convert image into numbers
        img_array = tf.keras.utils.img_to_array(img)

        # Add extra dimension
        img_array = np.expand_dims(img_array, axis=0)

        # Make prediction
        result = model.predict(img_array, verbose=0)

        # Get class
        class_index = np.argmax(result[0])

        prediction = classes[class_index]

        # Get confidence
        confidence = round(result[0][class_index] * 100, 1)

        # Give guidance
        if prediction == "plastic":

            guidance = (
                "Place the plastic item in the appropriate "
                "dry/recyclable waste collection. "
                "Clean and dry it before recycling when possible."
            )

        else:

            guidance = (
                "Place the organic item in the wet/organic waste bin. "
                "Composting is also an option where suitable."
            )

    return render_template_string(
        html,
        prediction=prediction,
        confidence=confidence,
        guidance=guidance
    )


# Start the web application
if __name__ == "__main__":
    app.run(debug=True)