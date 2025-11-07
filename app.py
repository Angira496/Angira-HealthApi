from flask import Flask, render_template, request, jsonify
import pandas as pd
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer

app = Flask(__name__)

# Load data and model
df = pd.read_csv("health_dataset.csv")
df['Symptoms'] = df['Symptoms'].str.lower().str.split(',').apply(lambda x: [sym.strip() for sym in x])
df['Symptoms_str'] = df['Symptoms'].apply(lambda x: ' '.join(x))

vectorizer = TfidfVectorizer()
vectorizer.fit(df['Symptoms_str'])

model = joblib.load("rf_model.sav")

@app.route('/')
def home():
    return render_template('index.html')

@app.route('https://angira-healthapi.onrender.com/predict', methods=['POST'])
def predict():
    try:
        symptoms = request.form['symptoms']
        symptoms_list = [s.strip().lower() for s in symptoms.split(',')]
        symptoms_str = ' '.join(symptoms_list)
        X_new = vectorizer.transform([symptoms_str])
        prediction = model.predict(X_new)[0]

        info = df[df['Disease Name'].str.lower() == prediction.lower()].iloc[0].to_dict()

        return render_template('index.html',
                               prediction=prediction,
                               disease_type=info.get('Type of Disease'),
                               doctor=info.get('Doctor'),
                               remedies=info.get('Remedies'))
    except Exception as e:
        return render_template('index.html', error=str(e))

import os

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))  # Render provides PORT automatically
    app.run(host='0.0.0.0', port=port, debug=False)


