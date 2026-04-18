# Sistem Pengenalan Bahasa Isyarat Indonesia (SIBI) dengan CNN

## 📋 Deskripsi Proyek

Proyek ini merupakan aplikasi **Convolutional Neural Network (CNN)** untuk mengenali huruf-huruf dalam Sistem Bahasa Isyarat Indonesia (SIBI). Tujuan utama adalah membangun model deep learning yang dapat mengidentifikasi gesture isyarat tangan untuk setiap huruf dalam alfabet SIBI dengan akurasi tinggi.

### Teknologi yang Digunakan:
- **Python 3.x** - Bahasa pemrograman utama
- **TensorFlow/Keras** - Framework untuk membangun dan melatih model CNN
- **NumPy** - Pemrosesan array dan data numerik
- **OpenCV** - Pengolahan citra dan preprocessing gambar
- **H5 Format** - Penyimpanan model terlatih

---

## 📦 Library yang Digunakan

| Library | Version | Fungsi |
|---------|---------|--------|
| **NumPy** | Latest | Manipulasi array, operasi numerik |
| **OpenCV (cv2)** | 4.x+ | Pembacaan gambar, preprocessing, resize citra |
| **TensorFlow** | 2.x+ | Deep learning framework utama |
| **Keras** | Built-in TF | Model Sequential, layer Conv2D, Dense, dll |
| **MediaPipe** | Latest | Hand detection dan landmark tracking untuk data collection |
| **Python OS** | Built-in | Manajemen file dan directory |

### Cara Install Library:
```bash
pip install numpy opencv-python tensorflow mediapipe
```

---

## 🎯 Tujuan Proyek

1. Mengembangkan model CNN yang dapat mengenali gesture isyarat untuk 24 huruf SIBI (A-Z kecuali J dan Z)
2. Mencapai akurasi tinggi dalam klasifikasi gesture isyarat
3. Membangun sistem yang dapat diintegrasikan untuk real-time sign language recognition
4. Mengumpulkan dan mengorganisir dataset SIBI berkualitas tinggi

---

## 📁 Struktur Proyek

```
SEMESTER 4/SISTEM BAHASA ISYARAT-PENGOLAHAN CITRA/
├── CNN.ipynb              # Notebook untuk training model CNN
├── collect_data.py        # Script untuk mengumpulkan data gesture
├── main.py                # Aplikasi utama (dalam pengerjaan)
├── docs.md                # Dokumentasi proyek (file ini)
├── dataset/
│   └── SIBI/              # Dataset isyarat untuk setiap huruf
│       ├── A/, B/, C/, ... ├── Y/  # Folder untuk masing-masing huruf
└── model/
    └── cnn_model.h5       # Model CNN terlatih
```

---

## � Sumber Data

**Dataset SIBI** diunduh dari:
- **Kaggle**: [SIBI Dataset](https://www.kaggle.com/datasets/alvinbintang/sibi-dataset)
- **Author**: Alvin Bintang
- **Deskripsi**: Dataset Sistem Bahasa Isyarat Indonesia yang berisi gesture tangan untuk setiap huruf alfabet

---

## �📊 Status Proyek: **DALAM PENGERJAAN** ⚠️

### ✅ Bagian yang Sudah Selesai:

1. **Model Training**
   - ✅ CNN model telah dibangun dan dilatih
   - ✅ Model disimpan dalam format `.h5` (cnn_model.h5)
   - ✅ Training dan validation dilakukan dalam `CNN.ipynb`
   - ✅ Metrics akurasi telah dihitung

2. **Data Preprocessing**
   - ✅ Normalisasi gambar
   - ✅ Resize gambar ke ukuran standard
   - ✅ Label mapping untuk setiap huruf SIBI

### 🔄 Bagian yang Masih Dikerjakan:

1. **Main Application**
   - ⏳ File `main.py` masih kosong - perlu implementasi aplikasi utama
   - ⏳ Real-time prediction dari webcam/kamera
   - ⏳ Interface untuk menampilkan hasil prediksi

2. **Testing & Validation**
   - ⏳ Unit testing untuk berbagai modul
   - ⏳ Testing dengan data baru (outside training set)
   - ⏳ Evaluation metrik yang lebih detail

3. **Dokumentasi & UI**
   - ⏳ GUI/Web interface untuk user interaction
   - ⏳ API documentation lengkap
   - ⏳ User manual

4. **Deployment**
   - ⏳ Optimasi model untuk production
   - ⏳ Setup untuk real-time inference
   - ⏳ Integration dengan aplikasi eksternal

---

## 🚀 Fitur Utama

- **Model CNN** yang dilatih untuk 24 huruf SIBI
- **Preprocessing otomatis** untuk normalisasi gambar input
- **Label mapping** untuk konversi prediksi ke label huruf
- **Model persistence** dengan penyimpanan H5

---

## 📈 Metrik Performa (dari training)

| Metrik | Status |
|--------|--------|
| Accuracy | ✅ Sudah dihitung |
| Training Loss | ✅ Tercatat di history |
| Validation Accuracy | ✅ Tersedia |
| Model Size | ~H5 format |

---

## 🔧 Langkah Selanjutnya (TODO)

- [ ] Implementasikan fungsi prediksi real-time di `main.py`
- [ ] Tambahkan support untuk webcam input
- [ ] Buat visualisasi hasil prediksi
- [ ] Test dengan data baru dari luar dataset
- [ ] Optimasi akurasi model jika diperlukan
- [ ] Deploy aplikasi ke environment production
- [ ] Buat interface user-friendly (CLI/GUI/Web)

---

## 📝 Catatan Pengembang

- Model sudah terlatih dan siap digunakan
- Dataset SIBI tersimpan terorganisir per huruf
- Training history tersimpan dalam variabel `history` 
- Perlu fokus pada development `main.py` untuk implementasi penuh

---

## 👨‍💻 Pengembang

| Aspek | Detail |
|-------|--------|
| **Nama** | Haidar Habibi Al Farisi |
| **Jurusan** | Ilmu Komputer |
| **Tahun** | Semester 4 |

---

**Last Updated**: April 2026  
**Status**: 🟡 Work In Progress (60% Complete)
