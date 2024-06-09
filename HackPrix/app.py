from flask import Flask, request, jsonify, render_template
import pandas as pd
import numpy as np
import pickle

app = Flask(__name__)

# Load models and encoders
model_files = {
    'ExtraTrees': r'C:\Users\rohit\Desktop\hackprix\model\ExtraTrees.pkl',
    # Add other models if needed
}

symptom_severity_path = r'C:\Users\rohit\Desktop\hackprix\Symptom-severity.csv'
symptom_description_path = r'C:\Users\rohit\Desktop\hackprix\symptom_Description.csv'
symptom_precaution_path = r'C:\Users\rohit\Desktop\hackprix\symptom_precaution.csv'

def load_models():
    models = {}
    for model_name, model_file in model_files.items():
        try:
            with open(model_file, 'rb') as f:
                models[model_name] = pickle.load(f)
        except FileNotFoundError:
            app.logger.error(f"Model file not found for {model_name}: {model_file}")
    return models

def load_data():
    try:
        df_severity = pd.read_csv(symptom_severity_path)
        symptom_list = df_severity['Symptom'].tolist()
        df_desc = pd.read_csv(symptom_description_path)
        df_prec = pd.read_csv(symptom_precaution_path)
        return df_severity, symptom_list, df_desc, df_prec
    except FileNotFoundError as e:
        app.logger.error(f"Data file not found: {str(e)}")
        return None, None, None, None

models = load_models()
df_severity, symptom_list, df_desc, df_prec = load_data()

if df_severity is None or df_desc is None or df_prec is None:
    raise RuntimeError("Failed to load essential data files.")

@app.route('/')
def index():
    return render_template('main.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json
        symptoms = data.get('symptoms', [])

        if not symptoms:
            return jsonify({"error": "No symptoms provided"}), 400

        # Convert symptoms to the input format
        input_vector = pd.Series([0] * len(symptom_list), index=symptom_list)
        for symptom in symptoms:
            if symptom in input_vector.index:
                input_vector[symptom] = 1
        
        input_vector = input_vector.to_numpy().reshape(1, -1)

        # Predict using the model
        model = models.get('ExtraTrees')
        if model is None:
            return jsonify({"error": "Model not found"}), 500

        prediction_proba = model.predict_proba(input_vector)
        top_indices = np.argsort(prediction_proba[0])[-5:][::-1]
        top_diseases = model.classes_[top_indices]
        top_probabilities = prediction_proba[0][top_indices]

        predictions = []
        for disease, prob in zip(top_diseases, top_probabilities):
            disease_desc = df_desc[df_desc['Disease'] == disease]['Description'].values[0]
            precautions = df_prec[df_prec['Disease'] == disease].values[0][1:].tolist()
            predictions.append({
                "disease": disease,
                "probability": prob,
                "description": disease_desc,
                "precautions": precautions
            })

        return jsonify({"predictions": predictions})

    except Exception as e:
        app.logger.error(f"Prediction error: {str(e)}")
        return jsonify({"error": "Internal server error"}), 500

if __name__ == '__main__':
    app.run(debug=True)
