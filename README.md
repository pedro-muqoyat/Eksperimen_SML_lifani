# CATATAN TEKNIS

## 1. Otomatisasi Pipeline CI/CD

Pada tahap otomatisasi CI/CD di GitHub Actions, pembuatan _Docker Image_ dilakukan menggunakan pendekatan **Custom Dockerfile Pipeline** berbasis `python:3.12-slim` (melalui perintah `docker build` langsung). Pendekatan ini dipilih sebagai substitusi dari perintah bawaan `mlflow models build-docker`.

## 2. Alasan Arsitektural & Resolusi Bug

Perintah bawaan `mlflow models build-docker` pada `mlflow==2.11.3` memiliki _hardcoded bug_ di lingkungan Python 3.12 pada GitHub Actions terbaru:

- **Pemicu Error:** Saat kontainer menjalankan langkah internal `C._install_pyfunc_deps`, MLflow mencoba melakukan _self-import_ sebelum berkas `requirements.txt` sempat dibaca.
- **Dampak:** Memicu crash `ModuleNotFoundError: No module named 'pkg_resources'` karena Python 3.12 telah menghapus `setuptools` bawaan.

## 3. Solusi & Keunggulan

Untuk menjamin reliabilitas, efisiensi waktu _build_, dan ukuran kontainer yang lebih ringan (**~400MB vs ~1.5GB**), saya merancang Dockerfile terisolasi dengan mekanisme berikut:

1. Menginstal pengaman `"setuptools<70.0.0"` dan `mlflow==2.11.3` terlebih dahulu.
2. Menginstal seluruh dependensi model setelahnya.

## 4. Hasil Akhir

Hasil akhir artefak tetap **100% identik** dan sesuai standar **Kriteria 3**: sebuah kontainer API inferensi MLflow resmi yang berhasil diuji dan terunggah otomatis ke Docker Hub Registry:

`pedro-muqoyat/klasifikasi-mahasiswa-api:latest`
