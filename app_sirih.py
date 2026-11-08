import joblib
import pandas as pd
import streamlit as st

from klasifikasi_daun_sirih import ekstrak_fitur

st.set_page_config(page_title="Cek Kesehatan Daun Sirih", page_icon="🌿", layout="wide")
st.title("🌿 Cek Kesehatan Daun Sirih")
st.caption("Klasifikasi berdasarkan fitur warna dengan Decision Tree")

paket = joblib.load("model_daun_sirih.pkl")
model = paket["model"]
kolom = paket["kolom"]

file = st.file_uploader("Pilih foto daun sirih", type=["jpg", "jpeg", "png", "webp"])

if file is not None:
    file.seek(0)
    fitur = pd.DataFrame([ekstrak_fitur(file)])[kolom]
    peluang_semua = model.predict_proba(fitur)[0]
    hasil = sorted(zip(model.classes_, peluang_semua), key=lambda t: -t[1])
    label, peluang = hasil[0]

    nilai = fitur.iloc[0]
    hijau = nilai["rasio_hijau"]
    kuning = nilai["rasio_kuning_coklat"]

    kol_foto, kol_hasil = st.columns([1, 2], gap="large")

    # Kiri: foto (diperkecil)
    with kol_foto:
        st.image(file, caption="Foto daun", width=300)

    # Kanan: hasil, peluang, keterangan
    with kol_hasil:
        m1, m2 = st.columns(2)
        m1.metric("Hasil", label)
        m2.metric("Peluang", f"{peluang:.0%}")

        if label == "Baik":
            if kuning <= 0.15:
                st.success(
                    f"Daun didominasi warna hijau ({hijau:.0%} area daun) dan hanya sedikit "
                    f"bagian kuning/coklat ({kuning:.0%}). Ini ciri daun sirih yang sehat "
                    f"dan layak digunakan."
                )
            else:
                st.success(
                    f"Daun tergolong baik, tetapi sudah ada bagian kuning/coklat yang "
                    f"cukup terlihat ({kuning:.0%}) di samping warna hijau ({hijau:.0%}). "
                    f"Sebaiknya dipakai segera sebelum kualitasnya menurun."
                )
        else:
            st.warning(
                f"Warna hijau daun hanya {hijau:.0%} dan bagian kuning/coklat mencapai "
                f"{kuning:.0%}. Ini menandakan daun mulai layu, menguning, atau rusak, "
                f"sehingga kualitasnya kurang baik."
            )

        if peluang < 0.70:
            st.info(
                "Model kurang yakin dengan hasil ini. Coba foto ulang dengan cahaya "
                "cukup dan latar polos agar hasilnya lebih akurat."
            )