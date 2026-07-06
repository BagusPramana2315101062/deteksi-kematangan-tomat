# Deteksi Kematangan Buah Tomat

Project ini merupakan tugas pengolahan citra digital untuk mendeteksi jumlah buah tomat serta mengklasifikasikan tingkat kematangannya berdasarkan warna pada citra.

## Deskripsi Project

Program ini menggunakan Python dan OpenCV untuk memproses citra buah tomat. Tahapan utama yang digunakan meliputi pembacaan gambar, konversi warna ke ruang HSV, segmentasi warna, pemisahan objek menggunakan watershed, serta klasifikasi tomat matang dan belum matang.

## Fitur

- Mendeteksi objek buah tomat pada gambar
- Menghitung jumlah buah tomat
- Mengklasifikasikan tomat matang dan belum matang
- Menyimpan hasil deteksi dalam bentuk gambar
- Menampilkan mask hasil segmentasi warna

## Teknologi yang Digunakan

- Python
- OpenCV
- NumPy
- Matplotlib

## Struktur File

```text
deteksi-kematangan-tomat/
├── deteksi_tomat.py
├── tomat.jpeg
├── hasil_deteksi_tomat_watershed.jpg
├── marker_peak_watershed.jpg
├── mask_kandidat_tomat.jpg
├── mask_tomat_belum_matang.jpg
├── mask_tomat_matang.jpg
├── requirements.txt
├── .gitignore
└── README.md
```
