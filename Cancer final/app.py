
from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

# Load model and scaler
model = pickle.load(open('cancer_model.pkl', 'rb'))
scaler = pickle.load(open('scaler.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/hospitals')
def hospitals():
    return render_template('hospitals.html')

@app.route('/prevention')
def prevention():
    return render_template('prevention.html')

@app.route('/awareness')
def awareness():
    return render_template('awareness.html')

@app.route('/stages')
def stages():
    return render_template('stages.html')
    
@app.route('/predict', methods=['POST'])
def predict():
    features = [float(x) for x in request.form.values()]
    final_features = scaler.transform([features])
    prediction = model.predict(final_features)

    result = "High Risk of Lung Cancer" if prediction[0] == 1 else "Low Risk of Lung Cancer"
    return render_template('index.html', prediction_text=result)

if __name__ == "__main__":
    app.run(debug=True)
