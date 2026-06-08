# 🏦 BankNote.RF — Klasifikasi Keaslian Uang Kertas

> **Tugas 8 — Kecerdasan Buatan**  
> Implementasi Algoritma Random Forest untuk Klasifikasi Keaslian Uang Kertas Berbasis Web Menggunakan Flask  
> Teknik Informatika — Universitas Bale Bandung

---

## 📌 Deskripsi

Aplikasi web yang memanfaatkan algoritma **Random Forest** dari Scikit-Learn untuk mengklasifikasikan uang kertas sebagai **Asli** atau **Palsu** berdasarkan fitur statistik yang diekstrak melalui **Wavelet Transform**.

- **Akurasi Model:** 99.27%
- **Dataset:** Banknote Authentication (UCI ML Repository) — 1.372 sampel
- **Fitur Input:** Variance, Skewness, Curtosis, Entropy

---

## 🚀 Cara Menjalankan

### 1. Clone repository
```bash
git clone https://github.com/username/tugas8-banknote-rf.git
cd tugas8-banknote-rf
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Latih model (jalankan sekali)
```bash
python model/train_model.py
```

### 4. Jalankan aplikasi
```bash
python app.py
```

### 5. Buka browser
```
http://localhost:5000
```

---

## 📂 Struktur Folder

```
tugas8_banknote/
├── app.py                  # Aplikasi Flask utama
├── requirements.txt        # Daftar library
├── Procfile                # Konfigurasi Heroku
├── model/
│   ├── train_model.py      # Script pelatihan model
│   ├── rf_model.pkl        # Model Random Forest (generated)
│   ├── scaler.pkl          # StandardScaler (generated)
│   └── model_meta.json     # Metadata evaluasi (generated)
└── templates/
    ├── index.html          # Halaman utama (deteksi)
    └── about.html          # Halaman tentang model
```

---

## 🌐 Deployment ke Heroku

```bash
heroku create nama-aplikasi
git push heroku main
heroku open
```

---

## 🔬 Fitur Dataset

| Fitur | Deskripsi | Rentang |
|-------|-----------|---------|
| variance | Varians wavelet transformed image | -5 hingga 7 |
| skewness | Kemencengan distribusi | -14 hingga 13 |
| curtosis | Kurtosis distribusi | -18 hingga 18 |
| entropy | Entropi citra | -9 hingga 3 |

---

## 📊 Konfigurasi Model

```python
RandomForestClassifier(
    n_estimators = 100,
    max_depth    = 10,
    random_state = 42
)
```

---

## 🛠️ Teknologi

- **Python** — Bahasa pemrograman utama
- **Flask** — Framework web backend
- **Scikit-Learn** — Algoritma Random Forest
- **Pandas & NumPy** — Manajemen dataset
- **Joblib** — Serialisasi model
- **Bootstrap 5** — UI/UX frontend

---

## 👤 Informasi

- **Mata Kuliah:** Kecerdasan Buatan
- **Dosen:** Mohammad Bayu Anggara, S.Kom., M.Kom.
- **Institusi:** Teknik Informatika, Universitas Bale Bandung
