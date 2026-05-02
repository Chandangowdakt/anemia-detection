# AI-Based Anemia Detection System

An intelligent web application that detects anemia from
conjunctiva (inner eyelid) eye images using Deep Learning.

## Features
- EfficientNetB0 transfer learning model
- 86.9% validation accuracy, 75% anaemic recall
- GradCAM explainable AI visualization
- Self-learning pipeline (auto-improves from new data)
- Admin dashboard with prediction analytics
- Image validation (rejects non-eye images)

## Tech Stack
- Python, Flask, TensorFlow, OpenCV
- EfficientNetB0 (ImageNet pretrained)
- GradCAM for explainability

## How to Run Locally
pip install -r requirements.txt
python app.py

## Model
Trained on 401 conjunctiva images (anaemic + non-anaemic)
Two-stage training with class weights

## Disclaimer
This is a screening tool only. Not a medical diagnosis.
Always consult a doctor for clinical diagnosis.
