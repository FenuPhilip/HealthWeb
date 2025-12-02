from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import joblib
import pandas as pd
import os
import numpy as np


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, '..', 'model', 'disease_predictor_model.joblib')
SYMPTOMS_PATH = os.path.join(BASE_DIR, '..', 'model', 'all_symptoms.txt')
FRONTEND_PATH = os.path.join(BASE_DIR, '..', 'frontend')


app = Flask(__name__)
CORS(app)  

#symptoms and models load chiyyaan 
model = None
all_symptoms = []

try:
    model = joblib.load(MODEL_PATH)
    with open(SYMPTOMS_PATH, 'r') as f:
        all_symptoms = [line.strip() for line in f.readlines()]
except FileNotFoundError:
    print(f"Error: Model or symptoms file not found.")
    print(f"Attempted to load model from: {MODEL_PATH}")
    print("Please ensure you have run the training script (train.py) from the root directory first.")
    model = None 

# --- API Routes ---

@app.route('/')
def serve_index():
    """Serves the main index.html file from the frontend folder."""
    return send_from_directory(FRONTEND_PATH, 'index.html')

@app.route('/symptoms', methods=['GET'])
def get_symptoms():
    """Endpoint to get the list of all symptoms."""
    if not all_symptoms:
        return jsonify({"error": "Symptoms list not available. Please train the model first."}), 500
    return jsonify(all_symptoms)

@app.route('/predict', methods=['POST'])
def predict():
    """Endpoint to predict the top 3 diseases based on symptoms."""
    if model is None:
        return jsonify({"error": "Model not loaded. Please train the model first."}), 500

    data = request.get_json()
    if not data or 'symptoms' not in data:
        return jsonify({"error": "Invalid input. 'symptoms' key is required."}), 400

    symptoms = data['symptoms']
    if not isinstance(symptoms, list) or len(symptoms) == 0:
        return jsonify({"error": "'symptoms' must be a non-empty list."}), 400

    try:
        symptom_string = " ".join(symptoms)
        
        # all class prob
        probabilities = model.predict_proba([symptom_string])[0]
        
        # top 3 prob
        top_indices = np.argsort(probabilities)[-3:][::-1]
        
        predictions = []
        for i in top_indices:
            predictions.append({
                "disease": model.classes_[i],
                "probability": f"{probabilities[i]:.2f}"
            })

        return jsonify(predictions)

    except Exception as e:
        return jsonify({"error": f"An error occurred during prediction: {str(e)}"}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000)

