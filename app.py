import os
import socket

from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

model  = pickle.load(open('cardio_model.pkl', 'rb'))
scaler = pickle.load(open('scaler.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        age_years   = float(request.form['age'])
        age_days    = age_years * 365
        gender      = float(request.form['gender'])
        height      = float(request.form['height'])
        weight      = float(request.form['weight'])
        ap_hi       = float(request.form['ap_hi'])
        ap_lo       = float(request.form['ap_lo'])
        cholesterol = float(request.form['cholesterol'])
        gluc        = float(request.form['gluc'])
        smoke       = float(request.form['smoke'])
        alco        = float(request.form['alco'])
        active      = float(request.form['active'])

        bmi = round(weight / ((height / 100) ** 2), 1)

        # Match training feature order: age (days), gender, height, weight,
        # ap_hi, ap_lo, cholesterol, gluc, smoke, alco, active, bmi
        features = np.array([[age_days, gender, height, weight,
                              ap_hi, ap_lo, cholesterol, gluc,
                              smoke, alco, active, bmi]], dtype=float)

        # The scaler was trained only on numeric columns:
        # ['age', 'height', 'weight', 'ap_hi', 'ap_lo', 'bmi']
        num_cols_idx = [0, 2, 3, 4, 5, 11]
        features_scaled = features.copy()
        features_scaled[:, num_cols_idx] = scaler.transform(features[:, num_cols_idx])

        prediction  = model.predict(features_scaled)[0]
        probability = model.predict_proba(features_scaled)[0]
        risk_percent    = round(probability[1] * 100, 1)
        risk_level      = "high" if prediction == 1 else "low"

        return render_template('result.html',
                               risk_level=risk_level,
                               risk_percent=risk_percent,
                               safe_percent=round(probability[0] * 100, 1),
                               age=int(age_years),
                               bmi=bmi,
                               ap_hi=int(ap_hi),
                               ap_lo=int(ap_lo))
    except Exception as e:
        return render_template('result.html',
                               risk_level="error",
                               risk_percent=0,
                               safe_percent=0,
                               error=str(e))

def _find_free_port(start: int = 5000, end: int = 5010) -> int:
    """Return the first free port in the given range.

    This is used to avoid port conflicts (e.g., macOS AirPlay Receiver reserves 5000).
    """
    for port in range(start, end + 1):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            try:
                sock.bind(('127.0.0.1', port))
                return port
            except OSError:
                continue
    raise RuntimeError(f"No free ports available in range {start}-{end}")


if __name__ == '__main__':
    port = int(os.environ.get('PORT') or os.environ.get('FLASK_RUN_PORT') or _find_free_port())
    app.run(host="0.0.0.0", port=port) 