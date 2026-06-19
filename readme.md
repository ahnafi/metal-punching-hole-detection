# Metal Punching Hole Detector

Sebuah aplikasi berbasis web (menggunakan Flask) untuk mendeteksi kecacatan (seperti *punching hole* atau *bending*) pada permukaan logam. Aplikasi ini memanfaatkan model *Deep Learning* (YOLO) dan tahapan *Computer Vision* (OpenCV) untuk melakukan deteksi secara otomatis.

## 🌟 Fitur Utama

- **Pemrosesan Citra Cerdas**: Menggunakan *Grayscale*, *Adaptive Binary Threshold*, dan *Opening* untuk memperjelas area yang dideteksi sebelum dilempar ke model YOLO.
- **Deteksi Objek (YOLO)**: Memanfaatkan model YOLO (`model.pt`) khusus untuk mencari titik-titik anomali pada logam.
- **Visualisasi Hasil**: Otomatis menggambar kotak pembatas (*bounding box*) dan label pada gambar hasil *thresholding* maupun gambar asli untuk kemudahan identifikasi visual.
- **Detail & Skor Akurasi Dinamis**: Menampilkan rincian deteksi di UI dengan indikator warna dinamis (Hijau, Kuning, Merah) berdasarkan tingkat akurasi (Confidence Score).

## 🛠️ Persyaratan Sistem

Pastikan Anda telah menginstal **Python 3.8+** di komputer Anda. Modul yang dibutuhkan meliputi:
- `Flask`
- `opencv-python` (`cv2`)
- `numpy`
- `ultralytics`
- `werkzeug`

Anda dapat menginstal semua *dependencies* menggunakan pip:
```bash
pip install Flask opencv-python numpy ultralytics werkzeug
```

## 🚀 Langkah-langkah Menjalankan Aplikasi

1. **Clone Repositori**  
   Unduh atau *clone* repositori ini ke komputer Anda.

2. **Siapkan Model YOLO**  
   Pastikan file model YOLO bernama `model.pt` sudah tersedia dan ditempatkan di dalam folder utama (*root*) project, sejajar dengan `main.py`.

3. **Jalankan Server Lokal**  
   Buka terminal/command prompt, arahkan ke folder direktori project ini, lalu jalankan skrip utama:
   ```bash
   python main.py
   ```

4. **Buka Web di Browser**  
   Server lokal Flask akan berjalan di port `5000`. Buka browser pilihan Anda dan kunjungi:
   [http://127.0.0.1:5000](http://127.0.0.1:5000)

5. **Mulai Lakukan Deteksi**  
   - Klik tombol **Pilih Gambar** dan unggah foto permukaan logam yang ingin diperiksa (hanya format `.png`, `.jpg`, atau `.jpeg` dengan ukuran maksimal 16MB).
   - Klik **Proses Gambar**.
   - Tunggu beberapa saat untuk model bekerja, rincian hasil deteksi beserta gambar lengkapnya akan langsung ditampilkan!

## 📁 Struktur Penting Direktori

- `main.py`: File program utama yang berisi logika web server Flask, urutan pemrosesan gambar OpenCV, dan inisialisasi inferensi YOLO.
- `templates/index.html`: File HTML untuk merender tampilan dan *interface* pengguna di halaman browser.
- `static/uploads/`: Folder penyimpanan sementara untuk gambar yang Anda unggah (juga menyimpan gambar asli yang telah ditimpa dengan *bounding box* YOLO).
- `static/results/`: Folder penyimpanan untuk menyimpan gambar hasil *image processing* yang juga sudah digambar *bounding box*.