from flask import Flask, request, render_template, jsonify
import joblib
import numpy as np

app = Flask(__name__)
model = joblib.load('heart_disease_model.pkl')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/contact')
def contact():
    return render_template('contact.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.form.to_dict()
    features = [float(data['Age']), int(data['Sex']), int(data['ChestPain']), float(data['RestBP']), float(data['Chol']),
                int(data['Fbs']), int(data['RestECG']), float(data['MaxHR']), int(data['ExAng']),
                float(data['Oldpeak']), int(data['Slope']), float(data['Ca']), float(data['Thal'])]
    prediction = model.predict([features])[0]
    output = 'Yes' if prediction == 1 else 'No'
    return jsonify({'prediction_text': f'Heart Disease Prediction: {output}'})

if __name__ == '__main__':
    app.run(debug=True)
