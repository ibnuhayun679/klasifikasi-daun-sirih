"""
Klasifikasi Kesehatan Daun Sirih - Ekstraksi Fitur Warna + Decision Tree
Dataset : folder train dan test, masing-masing berisi 4 kelas:
          Healthy Leaf, Dried Leaf, Bacterial Leaf Disease, Fungal Brown Spot Disease

Mode (pilih salah satu):
  default  -> BINER  : Baik (Healthy Leaf) vs Buruk (3 kelas lainnya)   <- sesuai judul "Baik/Buruk"
  --empat  -> 4 KELAS: tiap folder jadi satu kelas

Jalankan:
  python klasifikasi_daun_sirih.py "D:\\train" "D:\\test"
  python klasifikasi_daun_sirih.py "D:\\train" "D:\\test" 200      (maks 200 foto/kelas, untuk uji cepat)
  python klasifikasi_daun_sirih.py "D:\\train" "D:\\test" --empat
  python klasifikasi_daun_sirih.py "D:\\train" "D:\\test" --ulang  (hitung ulang fitur)
Prediksi 1 foto setelah model jadi:
  python klasifikasi_daun_sirih.py --prediksi "D:\\foto_daun.jpg"

Butuh: pip install pandas scikit-learn matplotlib joblib pillow numpy
"""
import os
import sys

import joblib
import matplotlib
import numpy as np
import pandas as pd
from PIL import Image

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import (ConfusionMatrixDisplay, accuracy_score,
                             classification_report, confusion_matrix)
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.tree import DecisionTreeClassifier, export_text, plot_tree

UKURAN = (128, 128)  # semua foto diperkecil agar cepat dan seragam
EKSTENSI = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
FILE_MODEL = "model_daun_sirih.pkl"


# ------------------------------------------------------- EKSTRAKSI FITUR WARNA
def ekstrak_fitur(path):
    """Mengubah 1 foto menjadi 16 angka fitur warna."""
    img = Image.open(path).convert("RGB").resize(UKURAN)
    rgb = np.asarray(img, dtype=np.float32)
    hsv = np.asarray(img.convert("HSV"), dtype=np.float32)
    h, s, v = hsv[..., 0] * 360 / 255, hsv[..., 1] / 255, hsv[..., 2] / 255

    # abaikan latar putih/abu (saturasi rendah) dan area sangat gelap bila ada
    daun = (s > 0.15) & (v > 0.10)
    if daun.sum() < 50:
        daun = np.ones_like(s, dtype=bool)

    r, g, b = rgb[..., 0][daun], rgb[..., 1][daun], rgb[..., 2][daun]
    hh, ss, vv = h[daun], s[daun], v[daun]
    return {
        "R_mean": r.mean(), "G_mean": g.mean(), "B_mean": b.mean(),
        "R_std": r.std(), "G_std": g.std(), "B_std": b.std(),
        "H_mean": hh.mean(), "S_mean": ss.mean(), "V_mean": vv.mean(),
        "H_std": hh.std(), "S_std": ss.std(), "V_std": vv.std(),
        "rasio_hijau": ((hh >= 70) & (hh <= 170)).mean(),
        "rasio_kuning_coklat": ((hh >= 15) & (hh < 70)).mean(),
        "rasio_gelap": (vv < 0.25).mean(),
        "G_per_R": g.mean() / (r.mean() + 1e-6),
    }


def bangun_fitur(root, maks, nama_csv):
    if not os.path.isdir(root):
        sys.exit(f"Folder tidak ditemukan: {root}")
    data, gagal = [], 0
    for kelas in sorted(os.listdir(root)):
        folder = os.path.join(root, kelas)
        if not os.path.isdir(folder):
            continue
        files = []
        for dp, _, fs in os.walk(folder):
            files += [os.path.join(dp, f) for f in fs if f.lower().endswith(EKSTENSI)]
        files.sort()
        if maks:
            files = list(np.random.RandomState(42).permutation(files)[:maks])
        print(f"  {os.path.basename(root)}/{kelas}: {len(files)} foto")
        for i, p in enumerate(files, 1):
            try:
                f = ekstrak_fitur(p)
                f["kelas_asli"], f["file"] = kelas, os.path.basename(p)
                data.append(f)
            except Exception:
                gagal += 1  # file rusak dilewati
            if i % 200 == 0:
                print(f"    {i}/{len(files)}")
    if gagal:
        print(f"  Dilewati (rusak): {gagal}")
    df = pd.DataFrame(data)
    df.to_csv(nama_csv, index=False)
    return df


def muat_fitur(root, maks, nama_csv, ulang):
    if os.path.exists(nama_csv) and not ulang:
        print(f"Memakai fitur tersimpan: {nama_csv} (tambah --ulang untuk hitung ulang)")
        return pd.read_csv(nama_csv)
    print(f"Mengekstrak fitur dari {root} ...")
    return bangun_fitur(root, maks, nama_csv)


def beri_label(df, empat):
    if empat:
        return df["kelas_asli"]
    return df["kelas_asli"].apply(lambda k: "Baik" if "healthy" in k.lower() else "Buruk")


def prediksi(path_foto):
    paket = joblib.load(FILE_MODEL)
    f = pd.DataFrame([ekstrak_fitur(path_foto)])[paket["kolom"]]
    proba = paket["model"].predict_proba(f)[0]
    hasil = sorted(zip(paket["model"].classes_, proba), key=lambda t: -t[1])
    print(f"Hasil: {hasil[0][0]}")
    print("Peluang:", ", ".join(f"{c} {p:.0%}" for c, p in hasil))


# ------------------------------------------------------------------------ MAIN
def main():
    args = sys.argv[1:]
    if args and args[0] == "--prediksi":
        return prediksi(args[1])
    pos = [a for a in args if not a.startswith("--")]
    if len(pos) < 2:
        sys.exit(__doc__)
    dir_train, dir_test = pos[0], pos[1]
    maks = int(pos[2]) if len(pos) > 2 else None
    empat, ulang = "--empat" in args, "--ulang" in args
    mode = "4 kelas" if empat else "biner (Baik/Buruk)"
    print("Mode:", mode)

    tr = muat_fitur(dir_train, maks, "fitur_train.csv", ulang)
    te = muat_fitur(dir_test, maks, "fitur_test.csv", ulang)
    tr["label"], te["label"] = beri_label(tr, empat), beri_label(te, empat)

    print("\nJumlah data latih:", len(tr), "| data uji:", len(te))
    print("\nDistribusi data latih:\n", tr["label"].value_counts())
    print("\nDistribusi data uji:\n", te["label"].value_counts())
    kolom = [c for c in tr.columns if c not in ("kelas_asli", "file", "label")]
    print("\nRata-rata fitur per kelas asli:")
    print(tr.groupby("kelas_asli")[["G_mean", "H_mean", "S_mean", "V_mean",
                                    "rasio_hijau", "rasio_kuning_coklat",
                                    "rasio_gelap"]].mean().round(3).to_string())

    X_tr, y_tr, X_te, y_te = tr[kolom], tr["label"], te[kolom], te["label"]
    baseline = y_te.value_counts(normalize=True).max()

    grid = GridSearchCV(
        DecisionTreeClassifier(random_state=42),
        {"criterion": ["gini", "entropy"], "max_depth": [3, 4, 5, 6, 8, 10],
         "min_samples_leaf": [2, 5, 10]},
        cv=StratifiedKFold(5, shuffle=True, random_state=42), scoring="accuracy")
    grid.fit(X_tr, y_tr)
    model = grid.best_estimator_
    print("\nParameter terbaik:", grid.best_params_)
    print("Akurasi CV (data latih): %.3f" % grid.best_score_)

    pred = model.predict(X_te)
    print("Akurasi data uji       : %.3f  (baseline %.3f, %d data uji)"
          % (accuracy_score(y_te, pred), baseline, len(y_te)))
    print("\nClassification report:\n", classification_report(y_te, pred, zero_division=0))
    print("Confusion matrix (baris=sebenarnya, kolom=prediksi; urutan:",
          list(model.classes_), ")\n", confusion_matrix(y_te, pred, labels=model.classes_))

    imp = pd.Series(model.feature_importances_, index=kolom).sort_values(ascending=False)
    print("\nFitur paling berpengaruh:\n", imp[imp > 0].round(3).head(8))

    open("aturan_decision_tree_sirih.txt", "w", encoding="utf-8").write(
        export_text(model, feature_names=kolom))

    plt.figure(figsize=(22, 10))
    plot_tree(model, feature_names=kolom, class_names=list(model.classes_),
              filled=True, rounded=True, fontsize=7, max_depth=4)
    plt.tight_layout()
    plt.savefig("decision_tree_sirih.png", dpi=150)
    plt.figure(figsize=(6, 5))
    ConfusionMatrixDisplay.from_predictions(
        y_te, pred, labels=model.classes_, cmap="Greens", ax=plt.gca(),
        xticks_rotation=30)
    plt.tight_layout()
    plt.savefig("confusion_matrix_sirih.png", dpi=150)

    joblib.dump({"model": model, "kolom": kolom}, FILE_MODEL)
    print("\nTersimpan: decision_tree_sirih.png, confusion_matrix_sirih.png,",
          "aturan_decision_tree_sirih.txt,", FILE_MODEL)
    print('Coba prediksi foto: python klasifikasi_daun_sirih.py --prediksi "foto.jpg"')


if __name__ == "__main__":
    main()
