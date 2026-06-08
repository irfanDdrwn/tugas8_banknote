"""
Aplikasi Web Klasifikasi Keaslian Uang Kertas
Menggunakan Algoritma Random Forest + Flask
Tugas 8 - Kecerdasan Buatan
"""

from flask import Flask, render_template, request, jsonify
import joblib
import numpy as np
import json
import os

app = Flask(__name__)

# ── Load model & scaler ──────────────────────────────────────────────────────
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH  = os.path.join(BASE_DIR, 'model', 'rf_model.pkl')
SCALER_PATH = os.path.join(BASE_DIR, 'model', 'scaler.pkl')
META_PATH   = os.path.join(BASE_DIR, 'model', 'model_meta.json')

model  = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)

with open(META_PATH) as f:
    meta = json.load(f)

# ── Routes ───────────────────────────────────────────────────────────────────
@app.route('/')
def index():
    return render_template('index.html', meta=meta)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        variance  = float(data['variance'])
        skewness  = float(data['skewness'])
        curtosis  = float(data['curtosis'])
        entropy   = float(data['entropy'])

        features = np.array([[variance, skewness, curtosis, entropy]])
        features_scaled = scaler.transform(features)

        prediction   = model.predict(features_scaled)[0]
        probability  = model.predict_proba(features_scaled)[0]

        hasil = {
            'label'      : 'ASLI' if prediction == 0 else 'PALSU',
            'kelas'      : int(prediction),
            'prob_asli'  : round(float(probability[0]) * 100, 2),
            'prob_palsu' : round(float(probability[1]) * 100, 2),
            'input'      : {
                'variance' : variance,
                'skewness' : skewness,
                'curtosis' : curtosis,
                'entropy'  : entropy
            }
        }
        return jsonify({'success': True, 'hasil': hasil})

    except (KeyError, ValueError) as e:
        return jsonify({'success': False, 'error': f'Input tidak valid: {str(e)}'})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/about')
def about():
    return render_template('about.html', meta=meta)

if __name__ == '__main__':
    app.run(debug=True, port=5000)
