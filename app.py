from flask import Flask, request, jsonify, render_template
import joblib
import numpy as np
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module='sklearn')

app = Flask(__name__)
# Load the saved model
model = joblib.load('svm_model_original.pkl')
scaler = joblib.load('scaler.pkl')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json(force=True)
    fertilizer_labels = ['10-26-26', '14-35-14', '17-17-17', '20-20', '28-28', 'DAP', 'Urea']
    # Extract features from request
    features = data.get('features')

    if features is None:
        return jsonify({'error': 'Missing "features" in request body'}), 400

    # Convert to numpy array and reshape if necessary
    input_data = np.array(features).reshape(1, -1)
    scaled_features = scaler.transform(input_data)
    # Predict
    prediction = model.predict(scaled_features)
    predicted_index = int(prediction[0])
    predicted_name = fertilizer_labels[predicted_index]

    return jsonify({'prediction': predicted_name})

if __name__ == '__main__':
    app.run(debug=True)
