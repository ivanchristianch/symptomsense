# SymptomSense

Saran diagnosis awal dari daftar gejala, sebagai alat bantu keputusan untuk dokter (pendukung fitur auto-fill di SayMed/BPRC).
Final project Data Science & AI/ML Batch 42. **Bukan alat diagnosis**; dokter tetap pengambil keputusan.

## Arsitektur

```
Streamlit UI  --HTTP-->  FastAPI  -->  model (NumPy: DNN & Logistic Regression)
                            |
                            +--------->  SQLite (riwayat prediksi)
GitHub Actions: ruff + pytest pada setiap push / pull request
```

Bobot model dilatih di notebook (TensorFlow / scikit-learn) lalu diekspor ke `artifacts/*.npz`.
API menjalankan model dengan NumPy murni, jadi **tidak membutuhkan TensorFlow saat deployment**.
Kesamaan hasil dengan model asli diuji otomatis (`tests/test_model_service.py`, kasus golden).

## Struktur

| Folder | Isi |
|---|---|
| `app/` | FastAPI (`main.py`), layanan model (`model_service.py`), database (`db.py`) |
| `streamlit_app/app.py` | Antarmuka: Prediksi, Riwayat, Performa Model, Tentang |
| `artifacts/` | Bobot model, daftar gejala/kelas, metrik, kasus golden untuk tes |
| `tests/` | 15 tes: unit, API, dan end-to-end (UI -> API -> database) |
| `train/export_artifacts.py` | Mengekspor artefak dari output notebook (sekali jalan) |
| `.github/workflows/ci.yml` | CI (lint + tes) dan deploy hook opsional |

## Menjalankan secara lokal

```bash
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# terminal 1: API  (dokumentasi interaktif di http://localhost:8000/docs)
uvicorn app.main:app --port 8000

# terminal 2: antarmuka
streamlit run streamlit_app/app.py                      # API_URL default http://localhost:8000
```

Dengan Docker (belum diuji di lingkungan pembuatan): `docker compose up --build`.

## API

| Endpoint | Fungsi |
|---|---|
| `POST /predict` | Top-k diagnosis, penjelasan per gejala, peringatan spine cluster; menyimpan ke database |
| `GET /history?limit=50` | Riwayat prediksi tersimpan |
| `GET /stats` | Jumlah prediksi, per model, diagnosis teratas |
| `GET /models` | Perbandingan model dan metrik evaluasi |
| `GET /symptoms`, `GET /health` | Daftar 283 gejala; status layanan |

Contoh:

```bash
curl -X POST localhost:8000/predict -H 'Content-Type: application/json' \
  -d '{"symptoms":["back pain","low back pain","leg pain","hip pain"],"model":"dnn","top_k":3}'
```

```json
{
  "model": "dnn",
  "n_symptoms": 4,
  "top_predictions": [
    {"disease": "spondylitis", "probability": 0.254, "is_musculoskeletal": true, "in_spine_cluster": false},
    {"disease": "spondylolisthesis", "probability": 0.140, "is_musculoskeletal": true, "in_spine_cluster": true},
    {"disease": "chronic pain disorder", "probability": 0.133, "is_musculoskeletal": false, "in_spine_cluster": false}
  ],
  "explanation_for": "spondylitis",
  "explanation": [{"symptom": "leg pain", "contribution": 0.125}, "..."],
  "spine_cluster_warning": null,
  "disclaimer": "Alat bantu keputusan, bukan pengganti penilaian klinis dokter.",
  "saved_id": 1
}
```

`explanation` memakai *leave-one-symptom-out*: seberapa turun probabilitas diagnosis teratas jika satu gejala dibuang.
Gejala yang tidak dikenali atau daftar kosong ditolak dengan HTTP 422.

## Database

SQLite (`data/predictions.db`, path lewat env `DB_PATH`). Tabel `predictions`:
`id, created_at, model, symptoms (JSON), top1_disease, top1_probability, top_predictions (JSON)`.
Untuk produksi, ganti ke PostgreSQL dengan menyesuaikan `app/db.py`.

## Tes dan CI/CD

```bash
pip install -r requirements-dev.txt
ruff check . && pytest -q
```

GitHub Actions (`.github/workflows/ci.yml`): pada setiap push/PR menjalankan `ruff` dan `pytest`.
Push ke `main` memicu langkah deploy opsional bila secret `DEPLOY_HOOK_URL` diisi (mis. deploy hook Render); bila kosong, langkah dilewati.

### Mengunggah ke repository private GitHub
```bash
git init && git add . && git commit -m "SymptomSense final project"
git branch -M main
git remote add origin git@github.com:<username>/symptomsense.git    # buat repo PRIVATE di GitHub dulu
git push -u origin main
```
Tab **Actions** di GitHub akan menampilkan hasil CI (bukti pengerjaan untuk checkpoint).

## Model dan hasil (data test, 32.746 baris)

| Model | Accuracy | F1-weighted | Interpretabilitas |
|---|---|---|---|
| DNN (128-64, seed 42) | 0.8474 | 0.8512 | rendah |
| Logistic Regression | 0.8500 | 0.8507 | tinggi |
| Random Forest (40 pohon)* | 0.8231 | 0.8248 | sedang |
| XGBoost (57 putaran)* | 0.8303 | 0.8316 | sedang |

\* dibatasi memori/waktu pada lingkungan pembuatan; belum dituning penuh.
Aplikasi menyediakan DNN (model utama) dan Logistic Regression (pembanding dan penjelas).

## Keterbatasan

- Data berupa daftar gejala terstruktur (Kaggle *Symptoms To Diseases*), bukan transkripsi percakapan SayMed.
- Belum divalidasi pada pasien nyata; bukan perangkat medis.
- Daftar 23 kelas musculoskeletal dipilih lewat kata kunci dan perlu direview klinisi
  (mis. neuralgia, spinal stenosis, herniated disk berdekatan secara klinis tetapi tidak termasuk).
- Hasil training bergantung pada seed; angka bisa berbeda tipis jika dilatih ulang.
