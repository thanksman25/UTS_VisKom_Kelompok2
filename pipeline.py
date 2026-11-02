"""
pipeline.py
Logika visi komputer untuk "Panorama Lingkungan Kampus" (UTS Visi Komputer - Kelompok 2).

Semua fungsi inti diambil dari notebook UTS_Kelompok2.ipynb, lalu dibungkus agar
bisa dipakai oleh web Streamlit (app.py):
  baca_citra -> samakan_kecerahan (HSV, kanal V) -> praproses_gray_eq (Hist. Equalization)
  -> deteksi (SIFT/ORB) -> matching (BFMatcher + Lowe's ratio) -> estimasi_homografi (RANSAC)
  -> stitching_warping (anchor tengah) -> segmentasi_kontur_crop (findContours + boundingRect)
"""
import re
import time

import cv2
import numpy as np

# ---------------- Parameter global (sama dengan notebook) ----------------
RASIO = 0.75       # ambang Lowe's Ratio Test
REPROJ = 4.0       # batas error reproyeksi RANSAC (piksel)
MIN_GOOD = 15      # minimal good matches sebelum hitung homografi
MIN_INLIER = 10    # minimal inlier RANSAC agar pasangan dianggap valid
MAX_PIKSEL_KANVAS = 16_000_000


def natural_sort_key(s):
    """Urutan natural: 1.jpg, 2.jpg, ..., 10.jpg."""
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", s)]


# ---------------- Tahap 1: baca & praproses (Praktikum 3) ----------------
def baca_citra_bytes(data: bytes, sisi_maks=1000):
    """Decode bytes hasil upload menjadi citra BGR, resize proporsional."""
    arr = np.frombuffer(data, np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("File tidak dapat dibaca sebagai gambar.")
    h, w = img.shape[:2]
    skala = sisi_maks / max(h, w)
    if skala < 1:
        img = cv2.resize(img, None, fx=skala, fy=skala, interpolation=cv2.INTER_AREA)
    return img


def samakan_kecerahan(imgs):
    """Samakan rata-rata kanal V (HSV) semua foto terhadap foto pertama."""
    ref_val = cv2.cvtColor(imgs[0], cv2.COLOR_BGR2HSV)[:, :, 2].mean()
    hasil = []
    for im in imgs:
        hsv = cv2.cvtColor(im, cv2.COLOR_BGR2HSV).astype(np.float32)
        gain = np.clip(ref_val / max(hsv[:, :, 2].mean(), 1.0), 0.7, 1.4)
        hsv[:, :, 2] = np.clip(hsv[:, :, 2] * gain, 0, 255)
        hasil.append(cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR))
    return hasil


def praproses_gray_eq(img):
    """Grayscale + Histogram Equalization."""
    return cv2.equalizeHist(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))


# ---------------- Tahap 2-3: fitur, matching, homografi (Praktikum 5) ----------------
def deteksi(gray, metode):
    det = cv2.SIFT_create() if metode.lower() == "sift" else cv2.ORB_create(nfeatures=5000)
    return det.detectAndCompute(gray, None)


def matching(desA, desB, metode, rasio=RASIO):
    if desA is None or desB is None:
        return []
    norm = cv2.NORM_L2 if metode.lower() == "sift" else cv2.NORM_HAMMING
    raw = cv2.BFMatcher(norm).knnMatch(desA, desB, k=2)
    return [m[0] for m in raw if len(m) == 2 and m[0].distance < rasio * m[1].distance]


def estimasi_homografi(kpA, kpB, good, reproj=REPROJ):
    if len(good) < MIN_GOOD:
        return None, None
    ptsA = np.float32([kpA[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    ptsB = np.float32([kpB[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
    H, mask = cv2.findHomography(ptsA, ptsB, cv2.RANSAC, reproj)
    if H is None:
        return None, None
    return H / H[2, 2], mask.ravel().astype(bool)


# ---------------- Tahap 4: warping & crop kontur (Praktikum 4 & 5) ----------------
def stitching_warping(imgs, pasangan):
    """Warp semua citra ke bidang bersama, anchor = foto tengah (n // 2)."""
    n = len(imgs)
    r = n // 2
    Hs = [None] * n
    Hs[r] = np.eye(3)

    for i in range(r + 1, n):
        if pasangan[i]["H"] is None:
            return None, None
        Hs[i] = Hs[i - 1] @ pasangan[i]["H"]
    for i in range(r - 1, -1, -1):
        if pasangan[i + 1]["H"] is None:
            return None, None
        Hs[i] = Hs[i + 1] @ np.linalg.inv(pasangan[i + 1]["H"])

    sudut = []
    for im, H in zip(imgs, Hs):
        h, w = im.shape[:2]
        c = np.float32([[0, 0], [w, 0], [w, h], [0, h]]).reshape(-1, 1, 2)
        sudut.append(cv2.perspectiveTransform(c, H))
    sem = np.concatenate(sudut)
    if not np.isfinite(sem).all():
        return None, None
    xmin, ymin = np.floor(sem.min(axis=0).ravel()).astype(int)
    xmax, ymax = np.ceil(sem.max(axis=0).ravel()).astype(int)
    W, Ht = int(xmax - xmin), int(ymax - ymin)
    if W <= 0 or Ht <= 0 or W * Ht > MAX_PIKSEL_KANVAS:
        return None, None

    T = np.array([[1, 0, -xmin], [0, 1, -ymin], [0, 0, 1]], np.float64)
    acc = np.zeros((Ht, W, 3), np.float32)
    cnt = np.zeros((Ht, W), np.float32)
    for im, H in zip(imgs, Hs):
        M = T @ H
        warp = cv2.warpPerspective(im, M, (W, Ht))
        m = cv2.warpPerspective(np.full(im.shape[:2], 255, np.uint8), M, (W, Ht))
        m = cv2.erode(m, np.ones((3, 3), np.uint8)) > 0
        acc[m] += warp[m]
        cnt[m] += 1
    pano = (acc / np.maximum(cnt, 1)[..., None]).astype(np.uint8)
    return pano, cnt > 0


def segmentasi_kontur_crop(pano, mask):
    """Cari kontur terbesar lalu pangkas sisa batas hitam."""
    biner = mask.astype(np.uint8) * 255
    cnts, _ = cv2.findContours(biner, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return pano, (0, 0, pano.shape[1], pano.shape[0])
    c = max(cnts, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(c)
    sub = mask[y:y + h, x:x + w]
    t, b, l, r = 0, h, 0, w
    while b - t > 1 and r - l > 1:
        s = sub[t:b, l:r]
        hitam = {
            "atas": int((~s[0]).sum()),
            "bawah": int((~s[-1]).sum()),
            "kiri": int((~s[:, 0]).sum()),
            "kanan": int((~s[:, -1]).sum()),
        }
        sisi, n = max(hitam.items(), key=lambda kv: kv[1])
        if n == 0:
            break
        if sisi == "atas":
            t += 1
        elif sisi == "bawah":
            b -= 1
        elif sisi == "kiri":
            l += 1
        else:
            r -= 1
    return pano[y + t:y + b, x + l:x + r], (x + l, y + t, r - l, b - t)


# ---------------- Pipeline lengkap untuk web ----------------
def jalankan_pipeline(imgs_asli, metode="sift"):
    """
    Jalankan seluruh pipeline pada daftar citra BGR (urut kiri -> kanan).
    Mengembalikan dict berisi hasil akhir, hasil antara, dan metrik untuk ditampilkan.
    """
    t0 = time.perf_counter()
    n = len(imgs_asli)
    hasil = dict(metode=metode, n_foto=n, sukses=False, pesan="", panorama=None,
                 pano_mentah=None, bbox=None, imgs_terang=None, imgs_gray=None,
                 imgs_keypoint=None, n_kp=[], pasangan=[], match_vis=[], waktu=0.0)

    imgs_terang = samakan_kecerahan(imgs_asli)
    imgs_gray = [praproses_gray_eq(im) for im in imgs_terang]
    fitur = [deteksi(g, metode) for g in imgs_gray]

    hasil["imgs_terang"] = imgs_terang
    hasil["imgs_gray"] = imgs_gray
    hasil["n_kp"] = [len(f[0]) for f in fitur]
    hasil["imgs_keypoint"] = [
        cv2.drawKeypoints(imgs_terang[i], fitur[i][0], None, color=(0, 255, 0))
        for i in range(n)
    ]

    pasangan = [None]
    for i in range(1, n):
        kpA, desA = fitur[i]
        kpB, desB = fitur[i - 1]
        good = matching(desA, desB, metode)
        H, inl = estimasi_homografi(kpA, kpB, good)
        n_inl = int(inl.sum()) if inl is not None else 0
        pasangan.append(dict(kpA=kpA, kpB=kpB, good=good, H=H, inlier=inl, n_inlier=n_inl))

        good_in = [m for m, k in zip(good, inl) if k] if inl is not None else good
        vis = cv2.drawMatches(imgs_terang[i], kpA, imgs_terang[i - 1], kpB, good_in[:70], None,
                              flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
        hasil["match_vis"].append(vis)

    hasil["pasangan"] = pasangan

    # Validasi tiap pasangan
    buruk = [i for i in range(1, n)
             if pasangan[i]["H"] is None or pasangan[i]["n_inlier"] < MIN_INLIER]
    if buruk:
        daftar = ", ".join(f"foto {i + 1}-{i} ({pasangan[i]['n_inlier']} inlier)" for i in buruk)
        hasil["pesan"] = (f"Pasangan foto berikut kurang cocok (minimal {MIN_INLIER} inlier): {daftar}. "
                          "Pastikan urutan foto kiri ke kanan benar dan overlap 30-50%.")
        hasil["waktu"] = time.perf_counter() - t0
        return hasil

    try:
        pano_mentah, mask = stitching_warping(imgs_terang, pasangan)
    except Exception as e:  # noqa: BLE001
        pano_mentah, mask = None, None
        hasil["pesan"] = f"Terjadi kesalahan saat warping: {e}"

    if pano_mentah is None:
        if not hasil["pesan"]:
            hasil["pesan"] = ("Warping gagal: homografi menghasilkan kanvas tidak valid atau terlalu besar. "
                              "Coba kurangi jumlah foto atau pakai foto dengan sudut lebih kecil.")
        hasil["waktu"] = time.perf_counter() - t0
        return hasil

    pano_crop, bbox = segmentasi_kontur_crop(pano_mentah, mask)
    vis_bbox = pano_mentah.copy()
    x, y, w, h = bbox
    cv2.rectangle(vis_bbox, (x, y), (x + w, y + h), (0, 255, 0), 4)

    hasil.update(sukses=True, pano_mentah=pano_mentah, panorama=pano_crop,
                 bbox=bbox, vis_bbox=vis_bbox, pesan="Panorama berhasil dibuat.")
    hasil["waktu"] = time.perf_counter() - t0
    return hasil


def ringkas_metrik(hasil):
    """Ambil angka ringkasan untuk ditampilkan sebagai 'informasi hasil'."""
    ps = hasil["pasangan"][1:]
    return dict(
        total_kp=int(sum(hasil["n_kp"])),
        total_good=int(sum(len(p["good"]) for p in ps)),
        total_inlier=int(sum(p["n_inlier"] for p in ps)),
        n_pasangan=len(ps),
        n_pasangan_ok=int(sum(p["H"] is not None and p["n_inlier"] >= MIN_INLIER for p in ps)),
    )
