"""
app.py - Web Streamlit "Panorama Lingkungan Kampus"
UTS Visi Komputer - Kelompok 2 (Tema 2)

Fitur wajib:
  1. Menerima upload gambar (3-5 foto yang saling overlap)
  2. Menampilkan gambar asli dan gambar hasil berdampingan
  3. Menampilkan informasi hasil (jumlah foto, keypoint, good match, inlier, waktu, ukuran, status)

Jalankan lokal:  streamlit run app.py
"""
import cv2
import pandas as pd
import streamlit as st

import pipeline as pl

st.set_page_config(page_title="Panorama Lingkungan Kampus", page_icon="🏛️", layout="wide")

# ------------------------------------------------------------------ Sidebar
with st.sidebar:
    st.header("⚙️ Pengaturan")
    metode = st.radio(
        "Metode fitur", ["SIFT", "ORB"], index=0,
        help="SIFT: lebih akurat & tahan perubahan cahaya. ORB: lebih cepat.",
    ).lower()
    sisi_maks = st.slider(
        "Sisi terpanjang citra (px)", 800, 1200, 1000, step=100,
        help="Foto diperkecil agar proses cepat. Rentang 800-1200 px sesuai ketentuan soal.",
    )
    balik = st.checkbox("Balik urutan (jika foto diambil kanan → kiri)", value=False)

    st.divider()
    st.subheader("Cara pakai")
    st.markdown(
        "1. Upload **3–5 foto** yang saling tumpang tindih (overlap 30–50%).\n"
        "2. Nama file harus **berurutan dari kiri ke kanan** (mis. `01.jpg`, `02.jpg`, `03.jpg`).\n"
        "3. Klik **Proses Panorama**."
    )
    st.caption("Pipeline: HSV (kanal V) → Grayscale + Hist. Equalization → SIFT/ORB → "
               "BFMatcher + Lowe's ratio → Homografi RANSAC → Warping → Crop kontur.")

# ------------------------------------------------------------------ Header
st.title("🏛️ Panorama Lingkungan Kampus")
st.write("Gabungkan beberapa foto kampus yang saling tumpang tindih menjadi satu citra panorama "
         "yang rapi dan bebas dari batas hitam.")

# ------------------------------------------------------------------ 1. Upload
files = st.file_uploader(
    "Upload foto (3–5 file, JPG/JPEG/PNG)",
    type=["jpg", "jpeg", "png"],
    accept_multiple_files=True,
)

if files:
    files = sorted(files, key=lambda f: pl.natural_sort_key(f.name))
    if balik:
        files = files[::-1]

    if len(files) < 2:
        st.warning("Upload minimal 3 foto (minimal 2 agar bisa digabung).")
    elif len(files) < 3 or len(files) > 5:
        st.info(f"Kamu mengunggah {len(files)} foto. Ketentuan tugas adalah 3–5 foto, "
                "tetapi aplikasi tetap akan mencoba memprosesnya.")

    st.caption("Urutan yang dipakai (kiri → kanan): " + "  →  ".join(f.name for f in files))

proses = st.button("🚀 Proses Panorama", type="primary", disabled=not files or len(files) < 2)

# ------------------------------------------------------------------ Proses
if proses:
    try:
        imgs = [pl.baca_citra_bytes(f.getvalue(), sisi_maks) for f in files]
    except ValueError as e:
        st.error(f"Gagal membaca gambar: {e}")
        st.stop()

    with st.spinner(f"Memproses {len(imgs)} foto dengan {metode.upper()} ..."):
        hasil = pl.jalankan_pipeline(imgs, metode)

    st.session_state["hasil"] = hasil
    st.session_state["imgs_asli"] = imgs
    st.session_state["nama"] = [f.name for f in files]

# ------------------------------------------------------------------ Tampilkan hasil
hasil = st.session_state.get("hasil")
if hasil is not None:
    imgs = st.session_state["imgs_asli"]
    nama = st.session_state["nama"]
    m = pl.ringkas_metrik(hasil)

    st.divider()
    if hasil["sukses"]:
        st.success(f"✅ {hasil['pesan']}")
    else:
        st.error(f"❌ Panorama gagal dibuat. {hasil['pesan']}")

    # ---------- 2. Gambar asli & hasil berdampingan ----------
    kol_asli, kol_hasil = st.columns(2, gap="large")

    with kol_asli:
        st.subheader(f"📷 Gambar Asli ({len(imgs)} foto)")
        grid = st.columns(2)
        for i, (im, nm) in enumerate(zip(imgs, nama)):
            with grid[i % 2]:
                st.image(im, channels="BGR", caption=f"{i + 1}. {nm}", use_container_width=True)

    with kol_hasil:
        st.subheader("🖼️ Hasil Panorama")
        if hasil["sukses"]:
            pano = hasil["panorama"]
            st.image(pano, channels="BGR", use_container_width=True,
                     caption=f"Panorama akhir ({metode.upper()}) - {pano.shape[1]} x {pano.shape[0]} px")
            ok, buf = cv2.imencode(".jpg", pano, [cv2.IMWRITE_JPEG_QUALITY, 95])
            if ok:
                st.download_button("⬇️ Unduh panorama (JPG)", buf.tobytes(),
                                   file_name=f"panorama_kampus_{metode}.jpg", mime="image/jpeg")
        else:
            st.warning("Belum ada panorama. Lihat pesan di atas dan tab "
                       "'Pencocokan fitur' di bawah untuk melihat pasangan yang bermasalah.")

    # ---------- 3. Informasi hasil ----------
    st.subheader("📊 Informasi Hasil")
    c = st.columns(6)
    c[0].metric("Jumlah foto", hasil["n_foto"])
    c[1].metric("Total keypoint", f"{m['total_kp']:,}")
    c[2].metric("Good matches", f"{m['total_good']:,}")
    c[3].metric("Inlier RANSAC", f"{m['total_inlier']:,}")
    c[4].metric("Waktu proses", f"{hasil['waktu']:.2f} s")
    if hasil["sukses"]:
        c[5].metric("Ukuran panorama", f"{hasil['panorama'].shape[1]}×{hasil['panorama'].shape[0]}")
    else:
        c[5].metric("Ukuran panorama", "-")

    st.caption(f"Metode: **{metode.upper()}** | Pasangan berhasil: "
               f"**{m['n_pasangan_ok']}/{m['n_pasangan']}** | "
               f"Status: **{'BERHASIL' if hasil['sukses'] else 'GAGAL'}**")

    # Tabel per pasangan
    baris = []
    for i in range(1, hasil["n_foto"]):
        p = hasil["pasangan"][i]
        ok = p["H"] is not None and p["n_inlier"] >= pl.MIN_INLIER
        baris.append({
            "Pasangan": f"{nama[i]} → {nama[i - 1]}",
            "Keypoint A": len(p["kpA"]),
            "Keypoint B": len(p["kpB"]),
            "Good matches": len(p["good"]),
            "Inlier": p["n_inlier"],
            "Rasio inlier": round(p["n_inlier"] / len(p["good"]), 2) if p["good"] else 0.0,
            "Status": "OK" if ok else "GAGAL",
        })
    st.dataframe(pd.DataFrame(baris), use_container_width=True, hide_index=True)

    # ---------- Hasil antara ----------
    st.subheader("🔍 Hasil Antara Pipeline")
    tabs = st.tabs(["1. Kecerahan (HSV-V)", "2. Gray + Equalize", "3. Keypoints",
                    "4. Pencocokan fitur", "5. Panorama mentah & kontur"])

    def baris_gambar(daftar, channels="BGR", prefix=""):
        kolom = st.columns(min(len(daftar), 3))
        for i, im in enumerate(daftar):
            with kolom[i % len(kolom)]:
                st.image(im, channels=channels, caption=f"{prefix}{i + 1}. {nama[i]}",
                         use_container_width=True)

    with tabs[0]:
        st.caption("Rata-rata kanal V disamakan terhadap foto pertama agar exposure konsisten.")
        baris_gambar(hasil["imgs_terang"])
    with tabs[1]:
        st.caption("Grayscale + Histogram Equalization untuk menonjolkan tekstur sebelum deteksi fitur.")
        baris_gambar(hasil["imgs_gray"], channels="GRAY")
    with tabs[2]:
        st.caption("Titik hijau = keypoint yang terdeteksi.")
        for i, im in enumerate(hasil["imgs_keypoint"]):
            st.image(im, channels="BGR", use_container_width=True,
                     caption=f"{nama[i]} - {hasil['n_kp'][i]:,} keypoint")
    with tabs[3]:
        st.caption("Garis = pasangan titik inlier hasil RANSAC (maks. 70 garis per pasangan).")
        for i, vis in enumerate(hasil["match_vis"], start=1):
            p = hasil["pasangan"][i]
            st.image(vis, channels="BGR", use_container_width=True,
                     caption=f"{nama[i]} → {nama[i - 1]} | good: {len(p['good'])}, inlier: {p['n_inlier']}")
    with tabs[4]:
        if hasil["sukses"]:
            st.image(hasil["pano_mentah"], channels="BGR", use_container_width=True,
                     caption="Panorama mentah (masih ada batas hitam)")
            st.image(hasil["vis_bbox"], channels="BGR", use_container_width=True,
                     caption="Kotak hijau = area hasil crop kontur (bebas batas hitam)")
        else:
            st.info("Panorama mentah tidak tersedia karena proses gagal.")

elif not files:
    st.info("👆 Upload foto terlebih dahulu untuk memulai.")
