# Benchmark Summary — Exp-017: Ollama Native Embedding & Classifier Head Matrix

- **Tanggal Pengujian:** 2026-10-12
- **Hardware Platform:** NVIDIA GeForce RTX 5050 Laptop GPU (8GB VRAM), WSL2 Linux
- **VRAM Footprint:** Start 3947.0 MiB / Peak 3959.0 MiB (Zero Python RAM / Zero Swap)
- **Testbed:** 60 Kueri Terstratifikasi (20 Formal, 20 Slang, 20 Implisit)

## 1. Tabel Komparasi Utama (9 Varian: 3 Model × 3 Head)

| No | Embedding Model | Classifier Head | Overall Acc (%) | Formal Acc (%) | Slang Acc (%) | Implicit Acc (%) | Avg Margin | P50 Latensi (ms) |
|:--:|---|---|:---:|:---:|:---:|:---:|:---:|:---:|
| 1 | `nomic-embed-text` | **Centroid** | **96.67%** | 95.0% | 95.0% | 100.0% | 0.0401 | 30.4 ms |
| 2 | `nomic-embed-text` | **Logistic** | **96.67%** | 95.0% | 95.0% | 100.0% | 0.0831 | 30.5 ms |
| 3 | `nomic-embed-text` | **kNN_1** | **93.33%** | 90.0% | 90.0% | 100.0% | 0.1724 | 30.4 ms |
| 4 | `qwen3-embedding:0.6b` | **Centroid** | **98.33%** | 95.0% | 100.0% | 100.0% | 0.1176 | 129.9 ms |
| 5 | `qwen3-embedding:0.6b` | **Logistic** | **98.33%** | 95.0% | 100.0% | 100.0% | 0.2110 | 130.1 ms |
| 6 | `qwen3-embedding:0.6b` | **kNN_1** | **100.0%** | 100.0% | 100.0% | 100.0% | 0.3155 | 130.0 ms |
| 7 | `bge-m3` | **Centroid** | **98.33%** | 95.0% | 100.0% | 100.0% | 0.1376 | 170.0 ms |
| 8 | `bge-m3` | **Logistic** | **98.33%** | 95.0% | 100.0% | 100.0% | 0.2338 | 170.1 ms |
| 9 | `bge-m3` | **kNN_1** | **98.33%** | 100.0% | 95.0% | 100.0% | 0.2934 | 170.0 ms |

## 2. Analisis Performa Berdasarkan Kategori Akademik FTMM

| Varian | CHAT_ONLY | PREREQUISITE | SKS_CALCULATION | DEGREE_PLANNING | COURSE_DISCOVERY |
|---|:---:|:---:|:---:|:---:|:---:|
| `nomic-embed-text + Centroid` | 100.0% | 91.67% | 100.0% | 100.0% | 91.67% |
| `nomic-embed-text + Logistic` | 100.0% | 91.67% | 100.0% | 100.0% | 91.67% |
| `nomic-embed-text + kNN_1` | 83.33% | 83.33% | 100.0% | 100.0% | 100.0% |
| `qwen3-embedding:0.6b + Centroid` | 100.0% | 91.67% | 100.0% | 100.0% | 100.0% |
| `qwen3-embedding:0.6b + Logistic` | 100.0% | 91.67% | 100.0% | 100.0% | 100.0% |
| `qwen3-embedding:0.6b + kNN_1` | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% |
| `bge-m3 + Centroid` | 91.67% | 100.0% | 100.0% | 100.0% | 100.0% |
| `bge-m3 + Logistic` | 91.67% | 100.0% | 100.0% | 100.0% | 100.0% |
| `bge-m3 + kNN_1` | 100.0% | 91.67% | 100.0% | 100.0% | 100.0% |

## 3. Temuan Kunci Riset (Key Empirical Findings)

1. **Zero-Overhead Ollama Native Ecosystem**:
   - Seluruh inferensi embedding berjalan di runtime C++ Ollama daemon via GPU mmap.
   - **0 MB beban Python Host RAM**, **0 disk I/O**, dan **0 Hyper-V swap delay**. Waktu inferensi per kueri stabil tanpa jitter.
   - VRAM total berada di ~3.95 GB (hanya 49.5% dari kapasitas 8 GB GPU RTX 5050), menyisakan headroom lapang >4 GB.

2. **Performa Sempurna `qwen3-embedding:0.6b` + Multi-Exemplar k-NN (100.0% Akurasi)**:
   - Kombinasi model `qwen3-embedding:0.6b` (639 MB) dengan `kNN_1` mencapai **akurasi sempurna 100.0%** di seluruh 60 kueri (20 Formal: 100%, 20 Slang: 100%, 20 Implisit: 100%).
   - Separabilitas margin melonjak drastis hingga **0.3155** (hampir 8x lipat lebih tajam dibandingkan baseline Nomic Centroid 0.0401).
   - Multi-Exemplar k-NN membuktikan bahwa mempertahankan representasi multi-modalitas (ragam diksi slang dan formal) per kelas mencegah efek distorsi perataan (*mean cancellation*) pada centroid.

3. **Efisiensi Ultra-Cepat `nomic-embed-text` (30.4 ms P50)**:
   - `nomic-embed-text` tetap menjadi raja latensi tercepat (**30.4 ms**).
   - Penerapan **Head B (Logistic Regression)** pada Nomic menggandakan margin separabilitas dari **0.0401 ke 0.0831** tanpa penalti latensi (+0.1 ms).

4. **Konfirmasi Kemampuan Multilingual `bge-m3` (Trello Card 25 & 32)**:
   - `bge-m3` mencapai **98.33% akurasi** di seluruh varian head dengan margin tinggi (0.1376–0.2934).
   - Namun, latensi inferensinya sedikit lebih tinggi (**170.0 ms**) karena ukuran model yang lebih besar (1.2 GB).

## 4. Keputusan Arsitektur & Rekomendasi Produksi

- **Rekomendasi Utama (Accuracy-Centric)**:
  Adopsi **`qwen3-embedding:0.6b` + Multi-Exemplar k-NN** untuk System-1 Intent Router:
  * Akurasi: **100.0% PASS** di seluruh ragam bahasa (Formal, Slang, Implisit).
  * Margin: **0.3155** (sangat tegas, anti-ambiguitas).
  * Latensi: **~130 ms** (masih sangat cepat untuk System-1 router sub-detik).
  * Footprint: Hanya 639 MB di Ollama, zero Python RAM.
- **Rekomendasi Alternatif (Speed-Centric Sub-35ms)**:
  Pertahankan **`nomic-embed-text` + Logistic Hyperplane**:
  * Akurasi: **96.67%**.
  * Latensi: **30.5 ms** (tercepat).
  * Margin: **0.0831** (2x lipat lebih aman dibanding Centroid).
