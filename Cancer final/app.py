from flask import Flask, render_template, request
import numpy as np
from model_pipeline import get_trained_pipeline

app = Flask(__name__)

# Load trained pipeline (Random Forest inside)
pipeline = get_trained_pipeline()


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
    try:
        # Get input values from form
        features = [float(x) for x in request.form.values()]
        final_features = np.array([features])

        # Prediction only (no probability)
        prediction = pipeline.predict(final_features)[0]

        # Clean result output
        if prediction == 1:
            result = "High Risk of Lung Cancer"
        else:
            result = "Low Risk of Lung Cancer"

        return render_template('index.html', prediction_text=result)

    except Exception as e:
        return render_template('index.html', prediction_text=f"Error: {str(e)}")


if __name__ == "__main__":
    app.run(debug=True)