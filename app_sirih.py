import io

import joblib
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image

from klasifikasi_daun_sirih import ekstrak_fitur

st.set_page_config(page_title="Cek Kesehatan Daun Sirih", page_icon="🌿", layout="centered")

# ---------------------------------------------------------------- tampilan
# Antarmuka hitam-putih. Warna hanya dipakai untuk hasil:
# hijau = Baik, merah = Buruk, oranye = keyakinan rendah.
st.markdown(
    """
<style>
:root {
    --baik:#2e7d32; --buruk:#c62828; --waspada:#ef6c00;
    --bg:#ffffff; --card:#f3f5f2; --garis:#d9ded8; --teks:#1f2933; --redup:#5f6b66; --aksen:#1f2933;
}
.stApp { background: var(--bg); color: var(--teks); }
header[data-testid="stHeader"] { background: transparent; }
.block-container { max-width: 760px; padding-top: 2.2rem; }
h1, h2, h3, p, label, span, div { color: var(--teks); }
.judul { font-size: 2.2rem !important; font-weight: 800 !important; margin: 0; }
.sub { color: var(--redup); margin: .2rem 0 1.4rem; }
[data-testid="stFileUploader"] section {
    background: var(--card); border: 2px dashed #9aa5a0; border-radius: 14px; padding: 1.4rem;
}
[data-testid="stFileUploader"] section small, [data-testid="stFileUploader"] section span { color: var(--redup); }
[data-testid="stFileUploader"] button { background: var(--aksen); color: #ffffff; border: 0; font-weight: 700; }
[data-testid="stImage"] img { border-radius: 14px; border: 1px solid var(--garis); }
.kartu { background: var(--card); border: 1px solid var(--garis); border-radius: 14px; padding: 1.3rem 1.4rem; margin-bottom: 1rem; }
.kartu-kepala { display: flex; align-items: center; gap: .7rem; color: var(--redup); font-weight: 600; margin-bottom: .6rem; }
.ikon { width: 38px; height: 38px; border-radius: 10px; display: flex; align-items: center; justify-content: center;
        font-size: 1.2rem; font-weight: 800; color: #ffffff; }
.ikon.baik { background: var(--baik); }
.ikon.buruk { background: var(--buruk); }
.hasil { font-size: 2.6rem; font-weight: 800; line-height: 1.1; margin: .2rem 0 .5rem; }
.hasil.baik { color: var(--baik); }
.hasil.buruk { color: var(--buruk); }
.penjelasan { color: var(--redup); line-height: 1.55; }
.peringatan { margin-top: .8rem; padding: .6rem .8rem; border-radius: 10px; background: #fff4e5;
              border: 1px solid #f3cfa0; color: #8a4b08 !important; font-size: .9rem; }
.grid { display: grid; grid-template-columns: repeat(4, 1fr); gap: .8rem; }
@media (max-width: 640px) { .grid { grid-template-columns: repeat(2, 1fr); } }
.kotak { background: #ffffff; border: 1px solid var(--garis); border-radius: 10px; padding: .8rem .9rem; }
.kotak .k { color: var(--redup); font-size: .78rem; }
.kotak .v { font-size: 1.4rem; font-weight: 800; margin-top: .15rem; }
.bar { height: 6px; background: var(--garis); border-radius: 4px; margin-top: .5rem; overflow: hidden; }
.bar > div { height: 100%; border-radius: 4px; }
.judul-seksi { font-weight: 700; margin-bottom: .8rem; }
.catatan { color: var(--redup); font-size: .8rem; margin-top: .6rem; }
</style>
""",
    unsafe_allow_html=True,
)

st.markdown('<p class="judul">🌿 Cek Kesehatan Daun Sirih</p>', unsafe_allow_html=True)
st.markdown(
    '<p class="sub">Klasifikasi berdasarkan fitur warna dengan Decision Tree</p>',
    unsafe_allow_html=True,
)

AMBANG_YAKIN = 70  # di bawah ini hasil ditandai "kurang yakin"


# ---------------------------------------------------------------- model
@st.cache_resource
def muat_model():
    paket = joblib.load("model_daun_sirih.pkl")
    return paket["model"], paket["kolom"]


def jalankan_ekstraksi(berkas, gambar):
    """Memanggil ekstrak_fitur dengan jenis masukan yang diterimanya."""
    data = berkas.getvalue()

    def dari_berkas():
        berkas.seek(0)
        return ekstrak_fitur(berkas)

    def dari_bytes():
        return ekstrak_fitur(io.BytesIO(data))

    percobaan = [
        dari_berkas,
        dari_bytes,
        lambda: ekstrak_fitur(gambar),
        lambda: ekstrak_fitur(np.array(gambar)),
        lambda: ekstrak_fitur(np.array(gambar)[:, :, ::-1].copy()),  # BGR
    ]
    semua_galat = []
    for coba in percobaan:
        try:
            return coba()
        except Exception as e:  # noqa: BLE001
            semua_galat.append(f"{type(e).__name__}: {e}")
    raise RuntimeError(" | ".join(semua_galat))


def jadi_dict(hasil, kolom):
    if isinstance(hasil, pd.DataFrame):
        return hasil.iloc[0].to_dict()
    if isinstance(hasil, pd.Series):
        return hasil.to_dict()
    if isinstance(hasil, dict):
        return hasil
    return dict(zip(kolom, list(np.ravel(hasil))))


def adalah_baik(label):
    teks = str(label).strip().lower()
    if "baik" in teks:
        return True
    if "buruk" in teks:
        return False
    return teks in ("1", "1.0", "true")


def persen(nilai):
    """Rasio bisa 0-1 atau 0-100; jadikan 0-100."""
    nilai = float(nilai)
    return nilai * 100 if nilai <= 1.0 else nilai


def ambil(fitur, nama):
    for k, v in fitur.items():
        if str(k).lower() == nama.lower():
            return v
    return None


def kotak(judul, nilai):
    if nilai is None:
        return f'<div class="kotak"><div class="k">{judul}</div><div class="v">-</div></div>'
    lebar = max(0, min(100, nilai))
    return (
        f'<div class="kotak"><div class="k">{judul}</div>'
        f'<div class="v">{nilai:.1f}%</div>'
        f'<div class="bar"><div style="width:{lebar}%;background:#1f2933"></div></div></div>'
    )


# ---------------------------------------------------------------- halaman
try:
    model, kolom = muat_model()
except Exception as e:  # noqa: BLE001
    st.error(f"Model tidak bisa dimuat: {e}")
    st.stop()

berkas = st.file_uploader(
    "Pilih atau seret foto daun sirih ke sini",
    type=["jpg", "jpeg", "png", "webp"],
)

if berkas is None:
    st.markdown(
        '<p class="catatan">Gunakan foto satu daun dengan latar yang tidak terlalu ramai agar hasilnya lebih akurat.</p>',
        unsafe_allow_html=True,
    )
    st.stop()

gambar = Image.open(berkas).convert("RGB")

try:
    mentah = jalankan_ekstraksi(berkas, gambar)
    fitur = jadi_dict(mentah, kolom)
    X = pd.DataFrame([[fitur[k] for k in kolom]], columns=kolom)
    label = model.predict(X)[0]
    peluang = float(np.max(model.predict_proba(X)[0])) * 100
except Exception as e:  # noqa: BLE001
    st.error(f"Gambar tidak dapat diproses: {e}")
    st.stop()

baik = adalah_baik(label)
kelas = "baik" if baik else "buruk"
teks_hasil = "Baik" if baik else "Buruk"
ikon = "✓" if baik else "!"

hijau = ambil(fitur, "rasio_hijau")
coklat = ambil(fitur, "rasio_kuning_coklat")
gelap = ambil(fitur, "rasio_gelap")
h_mean = ambil(fitur, "H_mean")

hijau_p = persen(hijau) if hijau is not None else None
coklat_p = persen(coklat) if coklat is not None else None
gelap_p = persen(gelap) if gelap is not None else None

if baik:
    penjelasan = "Warna daun didominasi hijau dan bagian kuning/coklatnya kecil, sehingga daun tergolong sehat."
else:
    penjelasan = "Bagian kuning/coklat atau gelap cukup besar dibanding hijau, tanda daun mulai layu, menguning, atau rusak."
if hijau_p is not None and coklat_p is not None:
    penjelasan += f" (Hijau {hijau_p:.0f}%, kuning/coklat {coklat_p:.0f}%.)"

peringatan = ""
if peluang < AMBANG_YAKIN:
    peringatan = (
        f'<div class="peringatan">Keyakinan model rendah ({peluang:.0f}%). '
        "Coba foto ulang dengan cahaya cukup dan latar polos.</div>"
    )

kiri, kanan = st.columns([1, 1.2], gap="large")
with kiri:
    st.image(gambar, use_container_width=True)
with kanan:
    st.markdown(
        f"""
<div class="kartu">
  <div class="kartu-kepala"><div class="ikon {kelas}">{ikon}</div>Kesehatan Daun</div>
  <div class="hasil {kelas}">{teks_hasil}</div>
  <div class="penjelasan">{penjelasan}</div>
  {peringatan}
</div>
""",
        unsafe_allow_html=True,
    )

h_teks = f"{float(h_mean):.1f}" if h_mean is not None else "-"
st.markdown(
    f"""
<div class="kartu">
  <div class="judul-seksi">Detail Analisis</div>
  <div class="grid">
    {kotak("Rasio hijau", hijau_p)}
    {kotak("Rasio kuning/coklat", coklat_p)}
    {kotak("Rasio gelap", gelap_p)}
    <div class="kotak"><div class="k">Rata-rata hue (H)</div><div class="v">{h_teks}</div></div>
  </div>
  <div class="catatan">Tingkat keyakinan model: {peluang:.0f}%. Angka ini berasal dari proporsi data latih pada daun
  akhir pohon keputusan, bukan jaminan ketepatan.</div>
</div>
""",
    unsafe_allow_html=True,
)

with st.expander("Lihat semua fitur warna"):
    st.dataframe(pd.DataFrame([fitur]).T.rename(columns={0: "nilai"}), use_container_width=True)
