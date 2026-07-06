import cv2
import numpy as np
import matplotlib.pyplot as plt


# =====================================================
# PARAMETER UTAMA
# =====================================================
image_path = "tomat.jpeg"   # ganti sesuai nama file gambar
WORK_WIDTH = 900            # ukuran kerja agar parameter lebih stabil

# Threshold klasifikasi warna pada ruang HSV OpenCV: H = 0-179
H_MATANG_MAX = 25           # merah-oranye; turunkan ke 20 jika kuning tidak dianggap matang
H_MERAH_ATAS_MIN = 165      # merah pada ujung atas rentang Hue
H_BELUM_MAX = 95            # hijau sampai kuning kehijauan

S_MIN_MATANG = 35
S_MIN_BELUM = 20
V_MIN = 45

# Parameter watershed berbasis distance transform
MIN_PEAK_DISTANCE_RATIO = 0.045
MIN_PEAK_VALUE_RATIO = 0.022

# Filter objek hasil watershed
MIN_OBJECT_AREA_RATIO = 0.0010
MAX_OBJECT_AREA_RATIO = 0.0500
MIN_CIRCULARITY = 0.12

# Klasifikasi tingkat kematangan
MATURE_RATIO_LIMIT = 0.50
LOW_COVERAGE_LIMIT = 0.35

# True jika objek yang terpotong di tepi gambar tidak ingin dihitung
IGNORE_BORDER_OBJECTS = False
BORDER_MARGIN = 3


# =====================================================
# FUNGSI KLASIFIKASI PIKSEL
# =====================================================
def klasifikasi_piksel(hsv_img, bgr_img):
    """
    Menghasilkan mask matang, mask belum matang, mask kandidat tomat,
    dan mask valid. Fungsi ini dipakai untuk visualisasi dan klasifikasi
    per objek agar logika threshold tidak terduplikasi.
    """
    H, S, V = cv2.split(hsv_img)
    B, G, R = cv2.split(bgr_img)

    Rf = R.astype(np.float32)
    Gf = G.astype(np.float32)
    Bf = B.astype(np.float32)

    # Highlight putih mengkilap diabaikan karena bukan warna asli tomat
    highlight = (V > 235) & (S < 35)

    # Piksel valid: cukup terang dan bukan highlight ekstrem
    valid = (V >= V_MIN) & (~highlight)

    # Tomat matang: merah sampai oranye
    matang = (
        (((H <= H_MATANG_MAX) | (H >= H_MERAH_ATAS_MIN)) &
         (S >= S_MIN_MATANG) &
         (V >= V_MIN))
    ) & valid

    # Tomat belum matang: hijau, kuning kehijauan, dan hijau pucat
    hijau_hsv = (
        ((H > H_MATANG_MAX) & (H <= H_BELUM_MAX) &
         (S >= S_MIN_BELUM) &
         (V >= V_MIN))
    )

    hijau_pucat = (
        (S < 75) &
        (V > 95) &
        (Gf >= 0.88 * Rf) &
        (Gf >= Bf + 5)
    )

    belum = (hijau_hsv | hijau_pucat) & valid & (~matang)

    # Piksel transisi ditambahkan agar permukaan tomat tidak terlalu berlubang
    transisi = (
        ((H > H_MATANG_MAX) & (H <= 38) &
         (S >= 25) &
         (V >= V_MIN))
    ) & valid

    kandidat = matang | belum | transisi

    return (
        (matang.astype(np.uint8) * 255),
        (belum.astype(np.uint8) * 255),
        (kandidat.astype(np.uint8) * 255),
        (valid.astype(np.uint8) * 255)
    )


# =====================================================
# FUNGSI PEMBUAT MARKER WATERSHED
# =====================================================
def buat_marker_watershed(mask_tomat):
    """
    Membuat marker foreground dari puncak distance transform.
    Marker ini menjadi perkiraan pusat masing-masing tomat.
    """
    h, w = mask_tomat.shape[:2]
    min_dim = min(h, w)

    dist = cv2.distanceTransform(mask_tomat, cv2.DIST_L2, 5)

    min_peak_distance = max(10, int(MIN_PEAK_DISTANCE_RATIO * min_dim))
    min_peak_value = max(6, int(MIN_PEAK_VALUE_RATIO * min_dim))

    kernel_peak = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (2 * min_peak_distance + 1, 2 * min_peak_distance + 1)
    )

    # Local maxima: titik yang nilainya sama dengan hasil dilasi lokal
    dist_dilated = cv2.dilate(dist, kernel_peak)
    peaks = ((dist == dist_dilated) & (dist >= min_peak_value) & (mask_tomat > 0))
    peaks = peaks.astype(np.uint8) * 255

    # Jangan dilakukan opening pada peak karena titik puncak bisa hilang
    num_peak, peak_labels, stats, centroids = cv2.connectedComponentsWithStats(peaks, 8)

    markers = np.zeros(mask_tomat.shape, dtype=np.int32)
    markers[mask_tomat == 0] = 1  # background pasti

    current_marker = 2

    for i in range(1, num_peak):
        area_peak = stats[i, cv2.CC_STAT_AREA]

        if area_peak < 1:
            continue

        cx, cy = centroids[i]
        cx = int(round(cx))
        cy = int(round(cy))

        if cx < 0 or cx >= w or cy < 0 or cy >= h:
            continue

        if dist[cy, cx] < min_peak_value:
            continue

        # Marker dibuat sebagai area kecil, bukan satu piksel
        r_marker = max(3, int(0.35 * dist[cy, cx]))
        cv2.circle(markers, (cx, cy), r_marker, current_marker, -1)
        current_marker += 1

    return markers, dist, peaks, current_marker


# =====================================================
# 1. MEMBACA CITRA
# =====================================================
img0 = cv2.imread(image_path)

if img0 is None:
    raise FileNotFoundError("Gambar tidak ditemukan. Periksa nama file dan lokasi gambar.")

h0, w0 = img0.shape[:2]

if w0 != WORK_WIDTH:
    scale = WORK_WIDTH / w0
    img = cv2.resize(img0, (WORK_WIDTH, int(h0 * scale)), interpolation=cv2.INTER_AREA)
else:
    img = img0.copy()

h, w = img.shape[:2]
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


# =====================================================
# 2. KONVERSI WARNA DAN PEMBUATAN MASK
# =====================================================
img_blur = cv2.GaussianBlur(img, (3, 3), 0)
hsv = cv2.cvtColor(img_blur, cv2.COLOR_BGR2HSV)

mask_matang, mask_belum, mask_tomat, mask_valid = klasifikasi_piksel(hsv, img_blur)

kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

mask_tomat = cv2.morphologyEx(mask_tomat, cv2.MORPH_OPEN, kernel, iterations=1)
mask_tomat = cv2.morphologyEx(mask_tomat, cv2.MORPH_CLOSE, kernel, iterations=2)

# Menghapus komponen kandidat yang terlalu kecil
num_cc, labels_cc, stats_cc, _ = cv2.connectedComponentsWithStats(mask_tomat, 8)
mask_bersih = np.zeros_like(mask_tomat)
min_area_awal = int(0.00025 * h * w)

for i in range(1, num_cc):
    if stats_cc[i, cv2.CC_STAT_AREA] >= min_area_awal:
        mask_bersih[labels_cc == i] = 255

mask_tomat = mask_bersih


# =====================================================
# 3. WATERSHED UNTUK MEMISAHKAN TOMAT BERTUMPUK
# =====================================================
markers, dist, peaks, current_marker = buat_marker_watershed(mask_tomat)

markers_ws = markers.copy()
cv2.watershed(img_blur, markers_ws)


# =====================================================
# 4. ANALISIS OBJEK DAN KLASIFIKASI KEMATANGAN
# =====================================================
output = img.copy()

jumlah_tomat = 0
jumlah_matang = 0
jumlah_belum = 0
jumlah_low_confidence = 0

data_objek = []

min_object_area = int(MIN_OBJECT_AREA_RATIO * h * w)
max_object_area = int(MAX_OBJECT_AREA_RATIO * h * w)

for marker_id in range(2, current_marker):
    component = np.zeros((h, w), dtype=np.uint8)
    component[(markers_ws == marker_id) & (mask_tomat > 0)] = 255

    area = cv2.countNonZero(component)

    if area < min_object_area or area > max_object_area:
        continue

    contours, _ = cv2.findContours(component, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if len(contours) == 0:
        continue

    contour = max(contours, key=cv2.contourArea)
    contour_area = cv2.contourArea(contour)
    perimeter = cv2.arcLength(contour, True)

    if perimeter == 0:
        continue

    circularity = 4 * np.pi * contour_area / (perimeter ** 2)

    x, y, bw, bh = cv2.boundingRect(contour)
    aspect_ratio = bw / bh if bh > 0 else 0

    if circularity < MIN_CIRCULARITY:
        continue

    if aspect_ratio < 0.30 or aspect_ratio > 3.00:
        continue

    if IGNORE_BORDER_OBJECTS:
        menyentuh_tepi = (
            x <= BORDER_MARGIN or y <= BORDER_MARGIN or
            x + bw >= w - BORDER_MARGIN or y + bh >= h - BORDER_MARGIN
        )

        if menyentuh_tepi:
            continue

    piksel_valid = np.count_nonzero((mask_valid > 0) & (component > 0))
    piksel_matang = np.count_nonzero((mask_matang > 0) & (component > 0))
    piksel_belum = np.count_nonzero((mask_belum > 0) & (component > 0))

    piksel_terklasifikasi = piksel_matang + piksel_belum

    coverage = piksel_terklasifikasi / max(piksel_valid, 1)
    rasio_matang = piksel_matang / max(piksel_terklasifikasi, 1)

    if rasio_matang >= MATURE_RATIO_LIMIT:
        status = "Matang"
        warna = (0, 0, 255)       # merah dalam BGR
        jumlah_matang += 1
    else:
        status = "Belum Matang"
        warna = (0, 255, 0)       # hijau dalam BGR
        jumlah_belum += 1

    confidence = "Tinggi"

    if coverage < LOW_COVERAGE_LIMIT:
        confidence = "Rendah"
        jumlah_low_confidence += 1
    elif 0.40 <= rasio_matang <= 0.60:
        confidence = "Sedang"

    jumlah_tomat += 1

    cv2.drawContours(output, [contour], -1, warna, 2)

    label = f"{jumlah_tomat} {'M' if status == 'Matang' else 'BM'}"

    if confidence == "Rendah":
        label += "?"

    posisi_teks = (x, y - 5)

    if y < 15:
        posisi_teks = (x, y + 15)

    cv2.putText(
        output,
        label,
        posisi_teks,
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        warna,
        1
    )

    data_objek.append({
        "ID": jumlah_tomat,
        "Status": status,
        "Confidence": confidence,
        "Area": int(area),
        "Circularity": round(float(circularity), 3),
        "Coverage": round(float(coverage), 3),
        "Rasio Matang": round(float(rasio_matang), 3),
        "Pixel Matang": int(piksel_matang),
        "Pixel Belum Matang": int(piksel_belum)
    })


# =====================================================
# 5. OUTPUT TERMINAL
# =====================================================
print("====================================")
print("HASIL DETEKSI TOMAT - WATERSHED")
print("====================================")
print("Jumlah total tomat         :", jumlah_tomat)
print("Jumlah tomat matang        :", jumlah_matang)
print("Jumlah tomat belum matang  :", jumlah_belum)
print("Objek confidence rendah    :", jumlah_low_confidence)
print("====================================")

for data in data_objek:
    print(data)


# =====================================================
# 6. SIMPAN HASIL
# =====================================================
cv2.imwrite("hasil_deteksi_tomat_watershed.jpg", output)
cv2.imwrite("mask_tomat_matang.jpg", mask_matang)
cv2.imwrite("mask_tomat_belum_matang.jpg", mask_belum)
cv2.imwrite("mask_kandidat_tomat.jpg", mask_tomat)
cv2.imwrite("marker_peak_watershed.jpg", peaks)

print("\nFile hasil disimpan sebagai:")
print("- hasil_deteksi_tomat_watershed.jpg")
print("- mask_tomat_matang.jpg")
print("- mask_tomat_belum_matang.jpg")
print("- mask_kandidat_tomat.jpg")
print("- marker_peak_watershed.jpg")


# =====================================================
# 7. VISUALISASI
# =====================================================
plt.figure(figsize=(15, 9))

plt.subplot(2, 3, 1)
plt.imshow(img_rgb)
plt.title("Citra Asli")
plt.axis("off")

plt.subplot(2, 3, 2)
plt.imshow(mask_matang, cmap="gray")
plt.title("Mask Tomat Matang")
plt.axis("off")

plt.subplot(2, 3, 3)
plt.imshow(mask_belum, cmap="gray")
plt.title("Mask Tomat Belum Matang")
plt.axis("off")

plt.subplot(2, 3, 4)
plt.imshow(mask_tomat, cmap="gray")
plt.title("Mask Kandidat Tomat")
plt.axis("off")

plt.subplot(2, 3, 5)
plt.imshow(markers_ws, cmap="nipy_spectral")
plt.title("Label Watershed")
plt.axis("off")

plt.subplot(2, 3, 6)
plt.imshow(cv2.cvtColor(output, cv2.COLOR_BGR2RGB))
plt.title(f"Total: {jumlah_tomat}, Matang: {jumlah_matang}, Belum: {jumlah_belum}")
plt.axis("off")

plt.tight_layout()
plt.show()