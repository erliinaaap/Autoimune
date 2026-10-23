"""
train_model.py
Melatih ulang MODEL TERBAIK hasil eksperimen (GBoost [RFE], default jurnal) dan menyimpannya
ke model/model_bundle.joblib agar dapat dipakai oleh app.py (Streamlit).

Pipeline identik dengan notebook "Rekonstruksi Jurnal Alur Penelitian v3 (Tuning)":
  split 70:30 stratified (seed 42) -> imputasi mean/mode + encoding biner (fit pada data latih)
  -> ADASYN manual (hanya ReA, data latih) -> RFECV (XGBoost, F1-macro, min 6 fitur; fit pada data latih ASLI)
  -> GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=3, min_samples_split=2)

Penggunaan:
    python train_model.py --data "Rheumatic and Autoimmune Disease Dataset.xlsx"
"""
import argparse, math, os, time, warnings
import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import OneHotEncoder
from sklearn.neighbors import NearestNeighbors
from sklearn.feature_selection import RFECV
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                             cohen_kappa_score, matthews_corrcoef, classification_report,
                             confusion_matrix)
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")

SEED = 42
TEST_SIZE = 0.30
ADASYN_K = 5
RFECV_MIN_FEATURES = 6
CLASS_ORDER = ["RA", "ReA", "AS", "SS", "SLE", "PA", "N"]
NUM_FEATURES = ["Age", "ESR", "CRP", "RF", "Anti-CCP", "C3", "C4"]
CAT_FEATURES = ["Gender", "HLA-B27", "ANA", "Anti-Ro", "Anti-La", "Anti-dsDNA", "Anti-Sm"]
FEATURES = ["Age", "Gender", "ESR", "CRP", "RF", "Anti-CCP", "HLA-B27", "ANA", "Anti-Ro",
            "Anti-La", "Anti-dsDNA", "Anti-Sm", "C3", "C4"]
class2id = {c: i for i, c in enumerate(CLASS_ORDER)}


def to_abbr(s):
    s = str(s)
    if "Rheumatoid" in s: return "RA"
    if "Reactive" in s:   return "ReA"
    if "Ankylosing" in s: return "AS"
    if "gren" in s:       return "SS"
    if "Lupus" in s:      return "SLE"
    if "Psoriatic" in s:  return "PA"
    if "Normal" in s:     return "N"
    raise ValueError(f"Label tidak dikenal: {s}")


def adasyn_manual(X, y, minority_label, n_target, k=5, seed=SEED):
    """ADASYN sesuai Pers. (4)-(8) jurnal (sama dengan notebook)."""
    rng = np.random.default_rng(seed)
    X = np.asarray(X, dtype=float); y = np.asarray(y)
    is_min = (y == minority_label)
    X_min = X[is_min]; n_min = len(X_min)
    G = int(n_target - n_min)
    nn_all = NearestNeighbors(n_neighbors=k + 1).fit(X)
    idx_all = nn_all.kneighbors(X_min, return_distance=False)[:, 1:]
    g = (y[idx_all] != minority_label).mean(axis=1)
    r = g / g.sum() if g.sum() > 0 else np.full(n_min, 1.0 / n_min)
    raw_cnt = r * G
    cnt = np.floor(raw_cnt).astype(int)
    remainder = G - cnt.sum()
    if remainder > 0:
        cnt[np.argsort(-(raw_cnt - cnt))[:remainder]] += 1
    nn_min = NearestNeighbors(n_neighbors=min(k + 1, n_min)).fit(X_min)
    idx_min = nn_min.kneighbors(X_min, return_distance=False)[:, 1:]
    synth = []
    for i in range(n_min):
        if cnt[i] == 0: continue
        z = rng.choice(idx_min[i], size=cnt[i])
        delta = rng.uniform(0, 1, size=(cnt[i], 1))
        synth.append(X_min[i] + delta * (X_min[z] - X_min[i]))
    X_syn = np.vstack(synth)
    y_syn = np.full(len(X_syn), minority_label)
    return np.vstack([X, X_syn]), np.concatenate([y, y_syn])


def balance_train(X, y, seed=SEED):
    """Mode 'rea_only': ReA dinaikkan ke jumlah kelas mayoritas."""
    counts = pd.Series(y).value_counts()
    return adasyn_manual(np.asarray(X, float), np.asarray(y), class2id["ReA"], int(counts.max()),
                         k=ADASYN_K, seed=seed)


def main(data_path, out_dir):
    t0 = time.time()
    raw = pd.read_excel(data_path)
    raw["Disease"] = raw["Disease"].map(to_abbr)
    print("Ukuran dataset:", raw.shape)

    # 1) Split 70:30 stratified pada data mentah
    train_df, test_df = train_test_split(raw, test_size=TEST_SIZE, stratify=raw["Disease"], random_state=SEED)
    y_train = train_df["Disease"].map(class2id).astype(int).values
    y_test = test_df["Disease"].map(class2id).astype(int).values

    # 2) Preprocessing (fit hanya pada data latih)
    means = {c: train_df[c].sum() / train_df[c].notna().sum() for c in NUM_FEATURES}
    modes = {c: train_df[c].value_counts().idxmax() for c in CAT_FEATURES}
    ohe = OneHotEncoder(drop="if_binary", sparse_output=False, dtype=float).fit(train_df[CAT_FEATURES].fillna(modes))

    def apply_prep(df):
        num = df[NUM_FEATURES].fillna(means)
        cat = df[CAT_FEATURES].fillna(modes)
        enc = pd.DataFrame(ohe.transform(cat), columns=CAT_FEATURES, index=df.index)
        return pd.concat([num, enc], axis=1)[FEATURES].astype(float)

    X_train, X_test = apply_prep(train_df), apply_prep(test_df)

    # 3) ADASYN (hanya data latih)
    Xb, y_bal = balance_train(X_train.values, y_train)
    X_bal = pd.DataFrame(Xb, columns=FEATURES)
    print(f"Data latih {len(y_train):,} -> {len(y_bal):,} setelah ADASYN")

    # 4) RFECV (XGBoost, F1-macro) pada data latih ASLI
    est = XGBClassifier(n_estimators=100, max_depth=4, learning_rate=0.1, random_state=SEED,
                        n_jobs=-1, verbosity=0, eval_metric="mlogloss")
    rfecv = RFECV(est, step=1, cv=StratifiedKFold(5, shuffle=True, random_state=SEED),
                  scoring="f1_macro", min_features_to_select=RFECV_MIN_FEATURES, n_jobs=1)
    rfecv.fit(X_train.values, y_train)
    selected = [f for f, s in zip(FEATURES, rfecv.support_) if s]
    print(f"RFECV memilih {len(selected)} fitur:", selected)

    # 5) Model terbaik: GBoost default jurnal pada fitur RFE
    model = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=3,
                                       min_samples_split=2, random_state=SEED)
    model.fit(X_bal[selected].values, y_bal)

    # 6) Evaluasi pada data uji asli
    y_pred = model.predict(X_test[selected].values)
    metrics = dict(
        Accuracy=100 * accuracy_score(y_test, y_pred),
        Precision=100 * precision_score(y_test, y_pred, average="macro", zero_division=0),
        Recall=100 * recall_score(y_test, y_pred, average="macro", zero_division=0),
        F1=100 * f1_score(y_test, y_pred, average="macro", zero_division=0),
        Kappa=100 * cohen_kappa_score(y_test, y_pred),
        MCC=100 * matthews_corrcoef(y_test, y_pred),
    )
    report = pd.DataFrame(classification_report(y_test, y_pred, target_names=CLASS_ORDER,
                                                output_dict=True, zero_division=0)).T
    cm = confusion_matrix(y_test, y_pred, labels=range(len(CLASS_ORDER)))
    print({k: round(v, 2) for k, v in metrics.items()})

    # 7) Simpan bundel
    rng = np.random.default_rng(SEED)
    n_bg = min(3000, len(X_bal))
    bg = X_bal[selected].iloc[rng.choice(len(X_bal), n_bg, replace=False)].values
    ranges = {c: dict(min=float(train_df[c].min()), max=float(train_df[c].max()),
                      mean=float(means[c])) for c in NUM_FEATURES}
    bundle = dict(
        model=model, selected=selected, class_order=CLASS_ORDER,
        means=means, modes=modes, ranges=ranges, lime_background=bg,
        metrics=metrics, report=report, confusion=cm,
        importance=pd.Series(model.feature_importances_, index=selected).sort_values(ascending=False),
        n_train=len(y_train), n_train_balanced=len(y_bal), n_test=len(y_test),
    )
    os.makedirs(out_dir, exist_ok=True)
    joblib.dump(bundle, os.path.join(out_dir, "model_bundle.joblib"), compress=3)
    print(f"Tersimpan di {out_dir}/model_bundle.joblib  ({time.time() - t0:.0f} detik)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="Rheumatic and Autoimmune Disease Dataset.xlsx")
    ap.add_argument("--out", default="model")
    a = ap.parse_args()
    main(a.data, a.out)
