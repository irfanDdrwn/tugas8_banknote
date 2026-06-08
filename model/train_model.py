"""
Train Random Forest Model untuk Klasifikasi Keaslian Uang Kertas
Dataset: Banknote Authentication Dataset (UCI Machine Learning Repository)
Fitur:
  - variance   : Varians dari Wavelet Transformed image
  - skewness   : Kemencengan dari Wavelet Transformed image
  - curtosis   : Kurtosis dari Wavelet Transformed image
  - entropy    : Entropi gambar
  - class      : 0 = Asli, 1 = Palsu
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
import joblib
import os

# ── 1. Load Dataset ──────────────────────────────────────────────────────────
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00267/data_banknote_authentication.txt"
columns = ['variance', 'skewness', 'curtosis', 'entropy', 'class']

try:
    df = pd.read_csv(url, header=None, names=columns)
    print("✅ Dataset berhasil dimuat dari UCI Repository")
except Exception:
    # Fallback: generate synthetic data jika tidak ada koneksi
    print("⚠️  Menggunakan data sintetis sebagai fallback...")
    np.random.seed(42)
    n = 1372
    asli  = np.random.multivariate_normal([2, 5, 2, -1], np.eye(4)*2, n//2)
    palsu = np.random.multivariate_normal([-2, -3, -1, 2], np.eye(4)*2, n - n//2)
    data  = np.vstack([asli, palsu])
    label = np.array([0]*(n//2) + [1]*(n - n//2))
    df    = pd.DataFrame(data, columns=['variance','skewness','curtosis','entropy'])
    df['class'] = label

print(f"📊 Total data: {len(df)} sampel")
print(f"   Asli  (0): {(df['class']==0).sum()}")
print(f"   Palsu (1): {(df['class']==1).sum()}")
print(df.head())

# ── 2. Preprocessing ─────────────────────────────────────────────────────────
X = df.drop(columns=['class'])
y = df['class']

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

# ── 3. Bangun Model Random Forest ─────────────────────────────────────────────
model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
model.fit(X_train, y_train)

# ── 4. Evaluasi ───────────────────────────────────────────────────────────────
y_pred = model.predict(X_test)
acc    = accuracy_score(y_test, y_pred)

print(f"\n🎯 Akurasi Model : {acc*100:.2f}%")
print("\n📋 Laporan Klasifikasi:")
print(classification_report(y_test, y_pred, target_names=['Asli','Palsu']))

cm = confusion_matrix(y_test, y_pred)
print("🔢 Confusion Matrix:")
print(cm)

# Feature importance
fi = pd.DataFrame({
    'Feature'   : X.columns,
    'Importance': model.feature_importances_
}).sort_values('Importance', ascending=False)
print("\n📌 Feature Importance:")
print(fi.to_string(index=False))

# ── 5. Simpan model & scaler ──────────────────────────────────────────────────
os.makedirs(os.path.dirname(__file__), exist_ok=True)
joblib.dump(model,  os.path.join(os.path.dirname(__file__), 'rf_model.pkl'))
joblib.dump(scaler, os.path.join(os.path.dirname(__file__), 'scaler.pkl'))

# Simpan metadata evaluasi untuk ditampilkan di web
import json
meta = {
    'accuracy'      : round(acc * 100, 2),
    'confusion_matrix': cm.tolist(),
    'feature_importance': fi.to_dict(orient='records'),
    'classification_report': {
        'asli' : {'precision': round(classification_report(y_test,y_pred,output_dict=True)['0']['precision'],2),
                  'recall'   : round(classification_report(y_test,y_pred,output_dict=True)['0']['recall'],2),
                  'f1'       : round(classification_report(y_test,y_pred,output_dict=True)['0']['f1-score'],2)},
        'palsu': {'precision': round(classification_report(y_test,y_pred,output_dict=True)['1']['precision'],2),
                  'recall'   : round(classification_report(y_test,y_pred,output_dict=True)['1']['recall'],2),
                  'f1'       : round(classification_report(y_test,y_pred,output_dict=True)['1']['f1-score'],2)},
    }
}
with open(os.path.join(os.path.dirname(__file__), 'model_meta.json'), 'w') as f:
    json.dump(meta, f, indent=2)

print("\n✅ Model, scaler, dan metadata berhasil disimpan!")
