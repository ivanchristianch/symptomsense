"""Layanan prediksi: DNN dan Logistic Regression dijalankan murni dengan NumPy.

Bobot dilatih di notebook (TensorFlow / scikit-learn), lalu diekspor ke artifacts/*.npz.
Dengan begitu API tidak membutuhkan TensorFlow saat deployment (lebih ringan & cepat start).
"""
import json
import os
from pathlib import Path

import numpy as np

DEFAULT_ARTIFACT_DIR = Path(__file__).resolve().parent.parent / "artifacts"


class UnknownSymptomError(ValueError):
    pass


class ModelService:
    def __init__(self, artifact_dir=None):
        art = Path(artifact_dir or os.getenv("ARTIFACT_DIR", DEFAULT_ARTIFACT_DIR))
        self.symptoms = json.loads((art / "symptoms.json").read_text())
        self.classes = json.loads((art / "classes.json").read_text())
        self.metrics = json.loads((art / "metrics.json").read_text())
        self._index = {s: i for i, s in enumerate(self.symptoms)}
        w = np.load(art / "dnn_weights.npz")
        self._dnn = [w[k] for k in ("W1", "b1", "W2", "b2", "W3", "b3")]
        lr = np.load(art / "logreg.npz")
        self._lr = (lr["coef"], lr["intercept"])
        self.msk = set(self.metrics["msk_classes"])
        self.spine = set(self.metrics["spine_cluster"])

    def vectorize(self, symptoms):
        unknown = [s for s in symptoms if s not in self._index]
        if unknown:
            raise UnknownSymptomError(f"Gejala tidak dikenali: {unknown}")
        x = np.zeros(len(self.symptoms), dtype=np.float32)
        for s in symptoms:
            x[self._index[s]] = 1.0
        return x

    def proba(self, model, X):
        """X: (n, n_gejala) -> probabilitas (n, n_kelas)."""
        if model == "dnn":
            W1, b1, W2, b2, W3, b3 = self._dnn
            h = np.maximum(X @ W1 + b1, 0)      # Dense(ReLU); Dropout tidak aktif saat inferensi
            h = np.maximum(h @ W2 + b2, 0)
            z = h @ W3 + b3
        elif model == "logreg":
            coef, b = self._lr
            z = X @ coef.T + b
        else:
            raise ValueError(f"Model tidak dikenal: {model}")
        z = z - z.max(axis=1, keepdims=True)    # softmax stabil secara numerik
        e = np.exp(z)
        return e / e.sum(axis=1, keepdims=True)

    def explain(self, model, x, class_idx):
        """Leave-one-symptom-out: seberapa turun probabilitas kelas jika satu gejala dibuang."""
        active = np.flatnonzero(x)
        if len(active) == 0:
            return []
        base = self.proba(model, x[None])[0, class_idx]
        X2 = np.repeat(x[None], len(active), axis=0)
        X2[np.arange(len(active)), active] = 0
        drop = base - self.proba(model, X2)[:, class_idx]
        out = [{"symptom": self.symptoms[j], "contribution": float(d)} for j, d in zip(active, drop)]
        return sorted(out, key=lambda r: abs(r["contribution"]), reverse=True)

    def predict(self, symptoms, model="dnn", top_k=3):
        x = self.vectorize(sorted(set(symptoms)))
        p = self.proba(model, x[None])[0]
        order = np.argsort(p)[::-1][:top_k]
        top = [{"disease": self.classes[i], "probability": float(p[i]),
                "is_musculoskeletal": self.classes[i] in self.msk,
                "in_spine_cluster": self.classes[i] in self.spine} for i in order]
        in_spine = [t["disease"] for t in top if t["in_spine_cluster"]]
        warning = None
        if len(in_spine) >= 2 or (top[0]["in_spine_cluster"] and top[0]["probability"] < 0.9):
            warning = ("Gejala ini mengarah ke kelompok diagnosis tulang belakang yang gejalanya sangat mirip "
                       "(degenerative disc disease, spondylosis, ankylosing spondylitis, spondylolisthesis, "
                       "chronic back pain). Pertimbangkan diagnosis banding, bukan hanya diagnosis teratas.")
        return {"model": model, "n_symptoms": int(x.sum()), "top_predictions": top,
                "explanation_for": top[0]["disease"],
                "explanation": self.explain(model, x, int(order[0])),
                "spine_cluster_warning": warning,
                "disclaimer": "Alat bantu keputusan, bukan pengganti penilaian klinis dokter."}
