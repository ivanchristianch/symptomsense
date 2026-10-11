"""Mengekspor artefak model dari output notebook ke folder artifacts/ (dijalankan SEKALI; butuh TensorFlow + scikit-learn).

Pemakaian:  python train/export_artifacts.py <folder_output_notebook>
Folder tersebut harus berisi: cleaned_data.csv, final_model_part3.keras, model_logreg_final.pkl,
part3_results.json, pred_DNN.npy, pred_LogReg.npy  (semuanya dihasilkan Bagian 3 notebook).
Hasil: dnn_weights.npz, logreg.npz, symptoms.json, classes.json, metrics.json, golden.json
"""
import json
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MultiLabelBinarizer
from tensorflow import keras

SRC = Path(sys.argv[1] if len(sys.argv) > 1 else ".")
OUT = Path(__file__).resolve().parent.parent / "artifacts"
OUT.mkdir(exist_ok=True)

MSK = ['ankylosing spondylitis', 'arthritis of the hip', 'bursitis', 'carpal tunnel syndrome', 'chronic back pain',
       'complex regional pain syndrome', 'degenerative disc disease', 'fibromyalgia', 'fracture of the arm',
       'fracture of the facial bones', 'fracture of the leg', 'fracture of the rib', 'gout', 'injury to the shoulder',
       'joint effusion', 'muscle spasm', 'osteoarthritis', 'rheumatoid arthritis', 'spondylitis', 'spondylolisthesis',
       'spondylosis', 'sprain or strain', 'tendinitis']
SPINE = ['degenerative disc disease', 'spondylosis', 'ankylosing spondylitis', 'spondylolisthesis', 'chronic back pain']

# data & split: identik dengan Bagian 2/3 notebook
df = pd.read_csv(SRC / "cleaned_data.csv")
df["symptom_list"] = df["symptom_text"].apply(lambda x: [s.strip().lower() for s in x.split(",")])
mlb = MultiLabelBinarizer()
X = mlb.fit_transform(df["symptom_list"]).astype("float32")
le = LabelEncoder()
y = le.fit_transform(df["diseases"])
X_tr, X_test, y_tr, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# bobot DNN & LogReg
dnn = keras.models.load_model(SRC / "final_model_part3.keras")
dense = [layer for layer in dnn.layers if layer.__class__.__name__ == "Dense"]
(W1, b1), (W2, b2), (W3, b3) = [d.get_weights() for d in dense]
np.savez(OUT / "dnn_weights.npz", W1=W1, b1=b1, W2=W2, b2=b2, W3=W3, b3=b3)
lr = pickle.load(open(SRC / "model_logreg_final.pkl", "rb"))
np.savez(OUT / "logreg.npz", coef=lr.coef_.astype("float32"), intercept=lr.intercept_.astype("float32"))

# verifikasi: implementasi NumPy == model asli
sub = X_test[:2000]
h = np.maximum(sub @ W1 + b1, 0)
h = np.maximum(h @ W2 + b2, 0)
z = h @ W3 + b3
z -= z.max(axis=1, keepdims=True)
p_np = np.exp(z) / np.exp(z).sum(axis=1, keepdims=True)
assert np.abs(p_np - dnn.predict(sub, verbose=0)).max() < 1e-4, "DNN NumPy != Keras"
zl = sub @ lr.coef_.T + lr.intercept_
zl -= zl.max(axis=1, keepdims=True)
assert np.abs(np.exp(zl) / np.exp(zl).sum(axis=1, keepdims=True) - lr.predict_proba(sub)).max() < 1e-4

json.dump(list(mlb.classes_), open(OUT / "symptoms.json", "w"))
json.dump(list(le.classes_), open(OUT / "classes.json", "w"))

# metrik untuk API/Streamlit
res = json.load(open(SRC / "part3_results.json"))
preds = {"dnn": np.load(SRC / "pred_DNN.npy"), "logreg": np.load(SRC / "pred_LogReg.npy")}
cls = list(le.classes_)
spine_idx = [cls.index(d) for d in SPINE]
metrics = {"msk_classes": MSK, "spine_cluster": SPINE, "comparison": res["comparison"],
           "robustness": res["robustness"], "seed_study": res["seed_study"],
           "dataset": {"rows": int(len(df)), "classes": len(cls), "symptoms": len(mlb.classes_),
                       "test": int(len(X_test))},
           "spine_confusion": {"labels": SPINE}, "msk_f1": {}}
for name, p in preds.items():
    cm = confusion_matrix(y_test, p, labels=range(len(cls)))
    metrics["spine_confusion"][name] = {"matrix": cm[np.ix_(spine_idx, spine_idx)].tolist(),
                                        "support": cm[spine_idx].sum(axis=1).tolist()}  # total kasus tiap penyakit
    f1s = f1_score(y_test, p, average=None, labels=range(len(cls)))
    metrics["msk_f1"][name] = {d: round(float(f1s[cls.index(d)]), 4) for d in MSK}
json.dump(metrics, open(OUT / "metrics.json", "w"), indent=1)

# golden cases untuk regression test
rng = np.random.default_rng(7)
spo, ddd = cls.index("spondylosis"), cls.index("degenerative disc disease")
hard = np.flatnonzero((y_test == spo) & (preds["dnn"] == ddd))
if len(hard) == 0:
    hard = np.flatnonzero((y_test == spo) & (preds["dnn"] != spo))
idx = [int(hard[0])] + [int(i) for i in rng.choice(len(X_test), 5, replace=False)]
golden = []
for i in idx:
    x = X_test[i][None]
    entry = {"symptoms": [mlb.classes_[j] for j in np.flatnonzero(x[0])]}
    for name, proba in (("dnn", dnn.predict(x, verbose=0)[0]), ("logreg", lr.predict_proba(x)[0])):
        k = int(np.argmax(proba))
        entry[name] = {"top1": cls[k], "p": float(proba[k])}
    golden.append(entry)
json.dump(golden, open(OUT / "golden.json", "w"), indent=1)
print("Artefak tersimpan di", OUT)
