import os
import numpy as np
import pandas as pd
import pickle
from flask import Flask, request, render_template

# Initialize Flask app
app = Flask(__name__)

# Set path where this script is running
base_path = os.path.dirname(__file__)

# Load model, encoder, and scaler safely using full paths
model = pickle.load(open(os.path.join(base_path, "model.pkl"), "rb"))
encoder = pickle.load(open(os.path.join(base_path, "encoder.pkl"), "rb"))
scaler = pickle.load(open(os.path.join(base_path, "scale.pkl"), "rb"))

# Route: Home Page
@app.route('/')
def home():
    return render_template("index.html")

# Route: Prediction
@app.route('/predict', methods=["POST"])
def predict():
    try:
        # Get form values from index.html
        holiday = request.form['holiday']
        temp = float(request.form['temp'])
        rain = float(request.form['rain'])
        snow = float(request.form['snow'])
        weather = request.form['weather']
        year = int(request.form['year'])
        month = int(request.form['month'])
        day = int(request.form['day'])
        hours = int(request.form['hours'])
        minutes = int(request.form['minutes'])
        seconds = int(request.form['seconds'])

        # Encode inputs
        holiday_encoded = encoder.transform([holiday])[0]
        weather_encoded = encoder.transform([weather])[0]

        # Prepare model input
        input_data = np.array([[holiday_encoded, temp, rain, snow, weather_encoded, year, month, day, hours, minutes, seconds]])
        input_scaled = scaler.transform(input_data)

        # Make prediction
        prediction = model.predict(input_scaled)[0]
        prediction = round(prediction, 2)

        # Logic: high or low traffic
        if prediction > 4500:
            return render_template("chance.html", prediction=prediction)
        else:
            return render_template("nochance.html", prediction=prediction)

    except Exception as e:
        return render_template("index.html", prediction_text=f"❌ Error: {str(e)}")

# Run the app
if __name__ == '__main__':
    app.run(debug=True)