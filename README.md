# Aplikasi Streamlit — Diagnosis Penyakit Reumatik & Autoimun

Model terbaik hasil eksperimen (notebook v3 Tuning): **Gradient Boosting (GBoost) + RFE, 10 fitur**
(ESR, CRP, RF, Anti-CCP, HLA-B27, ANA, Anti-Ro, Anti-La, C3, C4), hyperparameter default jurnal.
Kinerja data uji: akurasi 84,39%, F1-macro 83,71%.

## Cara menjalankan
1. `pip install -r requirements.txt`
2. Latih dan simpan model (sekali saja; butuh file dataset Harvard Dataverse):
   `python train_model.py --data "Rheumatic and Autoimmune Disease Dataset.xlsx"`
3. `streamlit run app.py`

Skrip pelatihan memakai SEED 42 dan alur yang sama dengan notebook, sehingga fitur terpilih RFECV dan
metrik uji seharusnya sama dengan hasil notebook (periksa keluaran di terminal).

## Deploy
Unggah folder ini ke GitHub (termasuk `model/model_bundle.joblib`) lalu hubungkan ke Streamlit Community Cloud
dengan berkas utama `app.py`.

Catatan: aplikasi ini alat bantu penelitian/edukasi, bukan pengganti diagnosis dokter.
