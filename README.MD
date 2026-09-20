# ♻️ Waste Classifier

A machine learning-based web application that classifies waste images into two categories:

- Organic Waste
- Plastic Waste

The application uses a Convolutional Neural Network (CNN) built with TensorFlow and provides a simple Flask web interface for uploading an image and getting a prediction.

## Features

- Upload a waste image through a web browser
- Classify the image as Organic or Plastic
- Display prediction confidence
- Provide basic waste disposal guidance
- Simple Flask web interface

## Technologies Used

- Python
- TensorFlow
- Keras
- NumPy
- Flask
- HTML
- CSS

## Project Structure

```text
waste-classifier-flask/
│
├── predict_waste.py
├── train_waste.py
├── waste_classifier.keras
├── README.md
└── .gitignore