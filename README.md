# Panorama Lingkungan Kampus dengan SIFT dan ORB

Proyek UTS Praktikum Visi Komputer: menyusun beberapa foto lingkungan kampus yang saling bertumpang tindih menjadi satu panorama. Aplikasi berbasis Streamlit memungkinkan pengguna mengunggah foto, memilih metode fitur, melihat tahapan pemrosesan dan metrik, lalu mengunduh panorama.

> **Catatan identitas:** README awal dan kode menyebut **Kelompok 2**, sedangkan sampul PDF laporan yang dilampirkan menyebut **Kelompok 3**. Nama dan NPM berikut disalin dari PDF laporan; pastikan nomor kelompok dan daftar anggota sesuai sebelum publikasi.

## Nama Anggota Tim dan NPM

| Nama | NPM |
|---|---|
| Nayla Nabila Syahel | 2408107010005 |
| Rahmatul Uliya | 2408107010012 |
| Abdan Syakura Bin Yasir | 2408107010014 |
| Andre Alfarizi | 2408107010020 |

Program Studi Informatika, Universitas Syiah Kuala, Banda Aceh (2026).

## Deskripsi

Aplikasi menerima 3–5 foto berurutan dari kiri ke kanan dengan area overlap yang disarankan sekitar 30–50%. Foto diproses dengan metode **SIFT** atau **ORB** untuk mendeteksi dan mencocokkan fitur. Aplikasi menampilkan foto masukan, panorama, keypoints, pasangan fitur, jumlah good matches dan inlier RANSAC, waktu proses, serta status stitching.

## Pipeline

```text
Foto berurutan (3–5)
        ↓
Resize proporsional dan normalisasi kecerahan
(HSV, penyesuaian kanal V terhadap foto pertama)
        ↓
Grayscale + Histogram Equalization
        ↓
Deteksi keypoints dan deskriptor (SIFT atau ORB)
        ↓
BFMatcher k-NN + Lowe's Ratio Test (rasio 0,75)
        ↓
Homografi dengan RANSAC (ambang reproyeksi 4 piksel)
        ↓
Warping ke bidang bersama (foto tengah sebagai anchor)
        ↓
Penggabungan, pencarian kontur, dan crop area hitam
        ↓
Panorama akhir + metrik dan berkas unduhan
```

RANSAC memakai minimal 15 good matches untuk menghitung homografi dan minimal 10 inlier agar pasangan dinyatakan valid. File `pipeline.py` memuat proses visi komputer, sedangkan `app.py` mengatur antarmuka Streamlit.

## Dataset

Dataset dikumpulkan sendiri oleh anggota menggunakan kamera ponsel di enam lokasi kampus. Foto diambil berurutan dari kiri ke kanan dengan overlap antarfoto. Folder `dataset/` berisi foto sumber, dikelompokkan berdasarkan lokasi.

| Kondisi dalam laporan | Lokasi | Foto |
|---|---:|---:|
| Ideal | 1 dan 5 | 8 |
| Cahaya | 3 dan 6 | 6 |
| Sudut | 2 | 5 |
| Sulit | 4 | 5 |
| **Jumlah** | **6 lokasi** | **24** |

Jumlah tersebut sesuai dengan file gambar pada folder dataset proyek.

## Hasil

Laporan menyatakan stitching berhasil pada seluruh enam lokasi (100%). Kesimpulan laporan menyebut SIFT menghasilkan kualitas visual rata-rata **4,8/5,0** dan rata-rata **2.357 inlier**, sementara ORB unggul dalam kecepatan dengan waktu yang dilaporkan **0,62 detik**. Angka ini merupakan hasil yang dicatat dalam laporan; waktu aktual dapat berbeda menurut komputer dan pengaturan.

Panorama akhir dan visualisasi proses tersedia di `output/`. Contoh hasil:

- `output/panorama_lokasi1_sift.jpg`
- `output/panorama_lokasi1_orb.jpg`
- `output/panorama_lokasi6_sift.jpg`
- `output/panorama_lokasi6_orb.jpg`

Laporan menilai SIFT lebih sesuai saat kualitas panorama menjadi prioritas, sedangkan ORB cocok ketika kecepatan pemrosesan lebih penting. Pada aplikasi, hasil juga bisa dicoba langsung memakai dataset atau foto lain.

## Eksperimen Pembanding

Eksperimen membandingkan SIFT dan ORB pada enam lokasi yang sama. Perbandingan mencakup jumlah keypoints, good matches, inlier RANSAC, waktu proses, keberhasilan stitching, dan penilaian visual (skala 1–5). Visualisasi ringkas tersedia pada [`output/perbandingan_sift_orb.png`](output/perbandingan_sift_orb.png), dan hasil akhir tiap metode tersimpan di folder `output/`.

| Aspek | SIFT | ORB |
|---|---|---|
| Hasil menurut laporan | Rata-rata 2.357 inlier; visual 4,8/5,0 | Waktu pemrosesan yang dilaporkan 0,62 detik; lebih cepat |
| Kelebihan yang diamati | Lebih tahan terhadap perubahan pencahayaan dan menghasilkan panorama visual lebih baik pada dataset ini | Pemrosesan lebih cepat dan efisien |
| Pertimbangan | Waktu proses relatif lebih lama | Inlier dapat turun pada perubahan kontras ekstrem |

Kesimpulan ini berlaku pada dataset dan kondisi eksperimen dalam laporan; hasil dapat berubah untuk foto dengan overlap, tekstur, pencahayaan, atau sudut yang berbeda.

## Cara Install di Laptop Lokal

Pastikan Python dan `pip` sudah tersedia. Dari terminal, masuk ke folder proyek, lalu jalankan:

```bash
python -m venv .venv
```

Aktifkan lingkungan virtual:

```bash
# macOS / Linux
source .venv/bin/activate

# Windows PowerShell
.venv\Scripts\Activate.ps1
```

Pasang dependensi dan jalankan aplikasi:

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
streamlit run app.py
```

Streamlit akan menampilkan alamat lokal di terminal (umumnya `http://localhost:8501`). Buka alamat tersebut di browser. Unggah foto JPG/JPEG/PNG yang saling overlap, urutkan nama file dari kiri ke kanan (misalnya `01.jpg`, `02.jpg`, `03.jpg`), pilih **SIFT** atau **ORB**, lalu klik **Proses Panorama**. Panorama yang berhasil dapat diunduh sebagai JPG.

## Struktur Folder

```text
.
├── app.py
├── pipeline.py
├── requirements.txt
├── UTS_Kelompok2.ipynb
├── dataset/
│   ├── lokasi1/ ... lokasi6/
└── output/
    ├── panorama_*.jpg
    ├── perbandingan_sift_orb.png
    └── lokasi1/ ... lokasi6/
```
