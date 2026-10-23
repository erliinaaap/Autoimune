"""
app.py — Aplikasi Streamlit prediksi penyakit reumatik & autoimun
Model terbaik hasil eksperimen: Gradient Boosting (GBoost) pada cabang seleksi fitur RFE (10 fitur).

Jalankan:  streamlit run app.py
"""
import os
import numpy as np
import pandas as pd
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import streamlit as st
from lime.lime_tabular import LimeTabularExplainer

st.set_page_config(page_title="Diagnosis Penyakit Reumatik & Autoimun", page_icon="🩺", layout="wide")

BUNDLE_PATH = os.path.join(os.path.dirname(__file__), "model", "model_bundle.joblib")

DISEASE_NAMES = {
    "RA": "Rheumatoid Arthritis (RA)",
    "ReA": "Reactive Arthritis (ReA)",
    "AS": "Ankylosing Spondylitis (AS)",
    "SS": "Sjögren's Syndrome (SS)",
    "SLE": "Systemic Lupus Erythematosus (SLE)",
    "PA": "Psoriatic Arthritis (PA)",
    "N": "Normal (tidak terindikasi)",
}
NUMERIC_HELP = {
    "ESR": "Laju endap darah (Erythrocyte Sedimentation Rate)",
    "CRP": "C-Reactive Protein",
    "RF": "Rheumatoid Factor",
    "Anti-CCP": "Anti-cyclic citrullinated peptide",
    "C3": "Komplemen C3",
    "C4": "Komplemen C4",
}
CATEGORICAL_HELP = {
    "HLA-B27": "Antigen HLA-B27",
    "ANA": "Antinuclear antibody",
    "Anti-Ro": "Antibodi anti-Ro (SSA)",
    "Anti-La": "Antibodi anti-La (SSB)",
}


@st.cache_resource(show_spinner="Memuat model...")
def load_bundle():
    if not os.path.exists(BUNDLE_PATH):
        return None
    return joblib.load(BUNDLE_PATH)


@st.cache_resource(show_spinner=False)
def get_explainer(_bundle):
    return LimeTabularExplainer(
        _bundle["lime_background"], feature_names=_bundle["selected"],
        class_names=_bundle["class_order"], discretize_continuous=True,
        mode="classification", random_state=42)


bundle = load_bundle()

st.title("🩺 Prediksi Penyakit Reumatik & Autoimun")
st.caption("Model: Gradient Boosting (GBoost) + seleksi fitur RFE · Rekonstruksi Mahdi et al. (2025), PeerJ Computer Science 11:e3096")

if bundle is None:
    st.error("Berkas model belum ditemukan (`model_bundle.joblib`).")
    st.markdown("Jalankan perintah berikut sekali pada folder aplikasi, lalu muat ulang halaman:")
    st.code('python train_model.py --data "Rheumatic and Autoimmune Disease Dataset.xlsx"', language="bash")
    st.stop()

model, selected, classes = bundle["model"], bundle["selected"], bundle["class_order"]
modes, ranges, metrics = bundle["modes"], bundle["ranges"], bundle["metrics"]

st.warning("Aplikasi ini merupakan **alat bantu penelitian/edukasi**, bukan pengganti diagnosis dokter. "
           "Hasil prediksi harus dikonfirmasi oleh tenaga medis yang berkompeten.", icon="⚠️")

with st.sidebar:
    st.header("Ringkasan Model")
    st.write("**Algoritma:** Gradient Boosting (parameter default jurnal)")
    st.write(f"**Seleksi fitur:** RFE → {len(selected)} fitur")
    st.write(", ".join(selected))
    st.divider()
    st.metric("Akurasi (data uji)", f"{metrics['Accuracy']:.2f}%")
    st.metric("F1-macro (data uji)", f"{metrics['F1']:.2f}%")
    st.caption(f"Data latih {bundle['n_train']:,} → {bundle['n_train_balanced']:,} (setelah ADASYN); "
               f"data uji {bundle['n_test']:,} (distribusi asli).")

tab_diag, tab_model = st.tabs(["Diagnosis", "Tentang Model"])

# ------------------------------------------------------------------ TAB DIAGNOSIS
with tab_diag:
    st.subheader("Input data pasien")
    st.caption("Hanya 10 fitur terpilih RFE yang dipakai model. Gunakan satuan yang sama dengan dataset acuan. "
               "Bila hasil suatu pemeriksaan tidak tersedia, centang “Tidak tersedia” (diisi dengan rata-rata/modus data latih).")

    values, missing = {}, []
    num_feats = [f for f in selected if f in ranges]
    cat_feats = [f for f in selected if f in modes and f not in ranges]

    st.markdown("**Hasil laboratorium numerik**")
    cols = st.columns(3)
    for i, f in enumerate(num_feats):
        r = ranges[f]
        with cols[i % 3]:
            na = st.checkbox("Tidak tersedia", key=f"na_{f}")
            v = st.number_input(f"{f}", min_value=float(r["min"]), max_value=float(r["max"]) * 3,
                                value=float(round(r["mean"], 2)), step=0.1, disabled=na, key=f"in_{f}",
                                help=f"{NUMERIC_HELP.get(f, f)} · rentang data latih {r['min']:.4g}–{r['max']:.4g}")
            values[f] = r["mean"] if na else v
            if na: missing.append(f)

    st.markdown("**Hasil serologi/genetik (kategorik)**")
    cols = st.columns(4)
    for i, f in enumerate(cat_feats):
        with cols[i % 4]:
            c = st.selectbox(f, ["Negative", "Positive", "Tidak tersedia"], key=f"in_{f}",
                             format_func=lambda s: {"Negative": "Negatif", "Positive": "Positif"}.get(s, s),
                             help=CATEGORICAL_HELP.get(f, f))
            if c == "Tidak tersedia":
                c = modes[f]; missing.append(f)
            values[f] = 1.0 if c == "Positive" else 0.0

    if st.button("🔍 Prediksi", type="primary", width="stretch"):
        x = np.array([[values[f] for f in selected]], dtype=float)
        proba = model.predict_proba(x)[0]
        order = np.argsort(-proba)
        top = int(order[0])

        st.divider()
        st.subheader("Hasil prediksi")
        c1, c2 = st.columns([1, 1.4])
        with c1:
            st.metric("Prediksi utama", DISEASE_NAMES[classes[top]])
            st.metric("Probabilitas", f"{proba[top] * 100:.1f}%")
            second = int(order[1])
            st.write(f"Kemungkinan terdekat berikutnya: **{DISEASE_NAMES[classes[second]]}** ({proba[second] * 100:.1f}%)")
            if proba[top] < 0.60:
                st.info("Keyakinan model tergolong rendah (< 60%). Hasil perlu dibaca dengan hati-hati.")
            if classes[top] in ("AS", "ReA"):
                st.caption("Catatan: pada data uji, recall kelas AS dan ReA relatif lebih rendah dibandingkan kelas lain.")
            if missing:
                st.caption("Fitur diisi nilai pengganti: " + ", ".join(missing) + ".")
        with c2:
            dfp = pd.DataFrame({"Penyakit": [DISEASE_NAMES[c] for c in classes], "Probabilitas (%)": proba * 100})
            dfp = dfp.sort_values("Probabilitas (%)", ascending=True)
            fig, ax = plt.subplots(figsize=(6, 3.4))
            colors = ["#2a6fdb" if p == dfp["Probabilitas (%)"].max() else "#b9c6d8" for p in dfp["Probabilitas (%)"]]
            ax.barh(dfp["Penyakit"], dfp["Probabilitas (%)"], color=colors)
            for y, v in enumerate(dfp["Probabilitas (%)"]):
                ax.text(v + 0.8, y, f"{v:.1f}%", va="center", fontsize=8)
            ax.set_xlim(0, 110); ax.set_xlabel("Probabilitas (%)"); ax.tick_params(labelsize=8)
            ax.spines[["top", "right"]].set_visible(False)
            plt.tight_layout(); st.pyplot(fig); plt.close(fig)

        st.subheader("Penjelasan prediksi (XAI LIME)")
        with st.spinner("Menghitung penjelasan LIME..."):
            exp = get_explainer(bundle).explain_instance(
                x[0], model.predict_proba, num_features=len(selected), labels=(top,))
            items = exp.as_list(label=top)
        lime_df = pd.DataFrame(items, columns=["Kondisi fitur", "Bobot"]).iloc[::-1]
        fig, ax = plt.subplots(figsize=(7, 0.45 * len(lime_df) + 1))
        ax.barh(lime_df["Kondisi fitur"], lime_df["Bobot"],
                color=["#2e9d5b" if w > 0 else "#d9534f" for w in lime_df["Bobot"]])
        ax.axvline(0, color="k", lw=0.8)
        ax.set_xlabel(f"Kontribusi terhadap kelas {classes[top]}  (hijau mendukung, merah menentang)")
        ax.tick_params(labelsize=8); ax.spines[["top", "right"]].set_visible(False)
        plt.tight_layout(); st.pyplot(fig); plt.close(fig)
        st.caption("LIME menjelaskan keputusan model untuk pasien ini secara lokal; "
                   "bobot menunjukkan arah dan besar pengaruh kondisi fitur terhadap prediksi.")

        with st.expander("Data masukan yang dipakai model"):
            st.dataframe(pd.DataFrame([values], index=["Nilai"]).T, width="stretch")

# ------------------------------------------------------------------ TAB MODEL
with tab_model:
    st.subheader("Kinerja pada data uji asli")
    cols = st.columns(6)
    for col, (k, lbl) in zip(cols, [("Accuracy", "Akurasi"), ("Precision", "Presisi"), ("Recall", "Recall"),
                                    ("F1", "F1-macro"), ("Kappa", "Kappa"), ("MCC", "MCC")]):
        col.metric(lbl, f"{metrics[k]:.2f}%")

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Laporan klasifikasi per kelas**")
        rep = bundle["report"].loc[classes + ["macro avg"], ["precision", "recall", "f1-score", "support"]]
        st.dataframe(rep.style.format({"precision": "{:.3f}", "recall": "{:.3f}", "f1-score": "{:.3f}", "support": "{:.0f}"}),
                     width="stretch")
    with c2:
        st.markdown("**Matriks konfusi**")
        cm = bundle["confusion"]
        fig, ax = plt.subplots(figsize=(5, 4))
        ax.imshow(cm, cmap="Blues")
        ax.set_xticks(range(len(classes)), classes); ax.set_yticks(range(len(classes)), classes)
        ax.set_xlabel("Prediksi"); ax.set_ylabel("Sebenarnya")
        for i in range(len(classes)):
            for j in range(len(classes)):
                ax.text(j, i, cm[i, j], ha="center", va="center", fontsize=8,
                        color="white" if cm[i, j] > cm.max() / 2 else "black")
        plt.tight_layout(); st.pyplot(fig); plt.close(fig)

    st.markdown("**Feature importance global (model GBoost)**")
    imp = bundle["importance"].iloc[::-1]
    fig, ax = plt.subplots(figsize=(6, 3.2))
    ax.barh(imp.index, imp.values, color="#2a6fdb"); ax.tick_params(labelsize=8)
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout(); st.pyplot(fig); plt.close(fig)

    st.markdown("""
**Alur singkat model:** pembagian data 70:30 → imputasi (rata-rata/modus) dan encoding biner → ADASYN pada kelas ReA
(hanya data latih) → seleksi fitur RFECV (XGBoost, F1-macro) → pelatihan Gradient Boosting.
Model dipilih berdasarkan skor validasi silang (CV F1-macro), bukan data uji, dengan aturan *parsimony*
(fitur paling sedikit di antara alternatif yang setara).
""")
