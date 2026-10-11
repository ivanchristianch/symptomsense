"""SymptomSense — antarmuka Streamlit. Semua prediksi dilakukan lewat REST API (FastAPI)."""
import os

import altair as alt
import numpy as np
import pandas as pd
import requests
import streamlit as st

API_URL = os.getenv("API_URL", "http://localhost:8000")
MODEL_LABELS = {"dnn": "DNN (deep learning)", "logreg": "Logistic Regression (mudah dijelaskan)"}
EXAMPLE = ["back pain", "low back pain", "leg pain", "hip pain"]

st.set_page_config(page_title="SymptomSense", page_icon="🩺", layout="wide")


def api(method, path, **kwargs):
    try:
        r = requests.request(method, f"{API_URL}{path}", timeout=30, **kwargs)
    except requests.RequestException as exc:
        st.error(f"API tidak dapat dihubungi di {API_URL}. Pastikan server FastAPI berjalan. ({exc.__class__.__name__})")
        return None
    if r.status_code >= 400:
        st.error(f"API mengembalikan error {r.status_code}: {r.json().get('detail', r.text)}")
        return None
    return r.json()


@st.cache_data(ttl=300)
def load_symptoms():
    data = api("GET", "/symptoms")
    return data["symptoms"] if data else []


@st.cache_data(ttl=300)
def load_metrics():
    return api("GET", "/models")


def fill_example():
    st.session_state["symptoms"] = EXAMPLE


def page_predict():
    st.title("🩺 SymptomSense")
    st.caption("Saran diagnosis awal dari daftar gejala — alat bantu keputusan untuk dokter, bukan pengganti penilaian klinis.")
    left, right = st.columns([1, 1.4], gap="large")
    with left:
        st.multiselect("Pilih gejala pasien", load_symptoms(), key="symptoms", help="Ketik untuk mencari gejala.")
        st.button("Isi contoh kasus nyeri punggung", on_click=fill_example)
        model = st.selectbox("Model", list(MODEL_LABELS), format_func=MODEL_LABELS.get)
        top_k = st.slider("Jumlah kemungkinan diagnosis", 1, 5, 3)
        save = st.checkbox("Simpan ke riwayat", value=True)
        go = st.button("Prediksi", type="primary")
    with right:
        if not go:
            st.info("Pilih gejala di sebelah kiri, lalu tekan **Prediksi**.")
            return
        if not st.session_state.get("symptoms"):
            st.warning("Pilih minimal satu gejala.")
            return
        res = api("POST", "/predict", json={"symptoms": st.session_state["symptoms"], "model": model,
                                            "top_k": top_k, "save": save})
        if not res:
            return
        top = pd.DataFrame(res["top_predictions"])
        top["Probabilitas (%)"] = (top["probability"] * 100).round(1)
        top["Kelompok"] = top.apply(lambda r: "Tulang belakang" if r["in_spine_cluster"]
                                    else ("Musculoskeletal" if r["is_musculoskeletal"] else "Lainnya"), axis=1)
        st.subheader("Kemungkinan diagnosis")
        st.dataframe(top[["disease", "Probabilitas (%)", "Kelompok"]].rename(columns={"disease": "Diagnosis"}),
                     hide_index=True, use_container_width=True)
        st.altair_chart(alt.Chart(top).mark_bar().encode(
            x=alt.X("Probabilitas (%):Q", scale=alt.Scale(domain=[0, 100])),
            y=alt.Y("disease:N", sort="-x", title=None)).properties(height=40 * len(top) + 20),
            use_container_width=True)
        if res["spine_cluster_warning"]:
            st.warning(res["spine_cluster_warning"])
        st.subheader(f"Gejala yang mendorong prediksi: {res['explanation_for']}")
        exp = pd.DataFrame(res["explanation"])
        if not exp.empty:
            exp["arah"] = exp["contribution"].apply(lambda v: "menaikkan" if v > 0 else "menurunkan")
            st.altair_chart(alt.Chart(exp).mark_bar().encode(
                x=alt.X("contribution:Q", title="Perubahan probabilitas jika gejala dibuang"),
                y=alt.Y("symptom:N", sort="-x", title=None),
                color=alt.Color("arah:N", scale=alt.Scale(domain=["menaikkan", "menurunkan"],
                                                          range=["#1E2A32", "#E8734A"]))
            ).properties(height=36 * len(exp) + 30), use_container_width=True)
        st.caption(res["disclaimer"])
        if res.get("saved_id"):
            st.caption(f"Tersimpan di riwayat (id {res['saved_id']}).")


def page_history():
    st.title("Riwayat prediksi")
    stats = api("GET", "/stats")
    items = api("GET", "/history", params={"limit": 100})
    if not stats or items is None:
        return
    c1, c2 = st.columns(2)
    c1.metric("Total prediksi tersimpan", stats["total_predictions"])
    c2.write("Per model: " + (", ".join(f"{MODEL_LABELS.get(k, k)}: {v}" for k, v in stats["by_model"].items()) or "-"))
    rows = [{"id": i["id"], "waktu (UTC)": i["created_at"], "model": i["model"],
             "gejala": ", ".join(i["symptoms"]), "diagnosis teratas": i["top1_disease"],
             "probabilitas": round(i["top1_probability"], 3)} for i in items["items"]]
    if rows:
        st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)
    else:
        st.info("Belum ada prediksi tersimpan.")


def page_performance():
    st.title("Performa model")
    m = load_metrics()
    if not m:
        return
    st.subheader("Perbandingan model (data test, split sama untuk semua model)")
    cmp_df = pd.DataFrame(m["comparison"])
    st.dataframe(cmp_df, hide_index=True, use_container_width=True)
    long = cmp_df.melt(id_vars="Model", value_vars=["Accuracy", "F1-weighted"], var_name="Metrik", value_name="Nilai")
    st.altair_chart(alt.Chart(long).mark_bar().encode(
        x=alt.X("Model:N", sort=None), y=alt.Y("Nilai:Q", scale=alt.Scale(domain=[0.6, 0.9])),
        color="Metrik:N", xOffset="Metrik:N").properties(height=300), use_container_width=True)
    st.subheader("Ketahanan temuan 'spine cluster' di semua model")
    st.dataframe(pd.DataFrame(m["robustness"]), hide_index=True, use_container_width=True)
    st.subheader("Confusion matrix — spine cluster (% dari seluruh kasus aktual)")
    sc = m["spine_confusion"]
    which = st.radio("Model", ["dnn", "logreg"], format_func=MODEL_LABELS.get, horizontal=True)
    mat = np.array(sc[which]["matrix"], dtype=float)
    support = np.array(sc[which]["support"], dtype=float)
    ext = np.hstack([mat, (support - mat.sum(axis=1))[:, None]])          # kolom terakhir: tebakan ke kelas lain
    cols = sc["labels"] + ["kelas lain"]
    pct = pd.DataFrame((ext / support[:, None] * 100).round(0), index=sc["labels"], columns=cols)
    long = pct.reset_index().melt(id_vars="index", var_name="Prediksi", value_name="%").rename(columns={"index": "Aktual"})
    base = alt.Chart(long).encode(x=alt.X("Prediksi:N", sort=None), y=alt.Y("Aktual:N", sort=None))
    st.altair_chart(base.mark_rect().encode(color=alt.Color("%:Q", scale=alt.Scale(scheme="yelloworangered", domain=[0, 100])))
                    + base.mark_text().encode(text="%:Q"), use_container_width=True)
    st.caption("Persen dihitung dari seluruh kasus tiap penyakit di data test, termasuk tebakan ke penyakit di luar 5 penyakit ini.")
    st.subheader("Apakah V1 vs V3 benar-benar berbeda? (3 seed per konfigurasi)")
    seed = pd.DataFrame(m["seed_study"])
    st.dataframe(seed.groupby("Konfigurasi")["Test Accuracy"].agg(["mean", "std", "min", "max"]).round(4),
                 use_container_width=True)


def page_about():
    st.title("Tentang SymptomSense")
    st.markdown(
        "**Tujuan:** memberi saran diagnosis awal dari daftar gejala sebagai fitur pendukung auto-fill di SayMed/BPRC.\n\n"
        "**Data:** Kaggle *Symptoms To Diseases* (254 penyakit, 283 gejala unik, 163.727 baris setelah dedup).\n\n"
        "**Model:** DNN (deep learning) dibandingkan dengan Logistic Regression, Random Forest, dan XGBoost.\n\n"
        "**Keterbatasan:** data berupa daftar gejala terstruktur, bukan transkripsi percakapan; belum divalidasi pada "
        "pasien nyata; beberapa diagnosis (terutama kelompok tulang belakang) memiliki gejala yang sangat mirip. "
        "Dokter tetap pengambil keputusan akhir.")


PAGES = {"Prediksi": page_predict, "Riwayat": page_history, "Performa Model": page_performance, "Tentang": page_about}
choice = st.sidebar.radio("Menu", list(PAGES))
PAGES[choice]()
