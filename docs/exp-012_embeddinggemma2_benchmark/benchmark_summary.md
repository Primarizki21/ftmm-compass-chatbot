# Benchmark Summary — Exp-012: EmbeddingGemma-2 (270M) vs nomic-embed-text

- **Tanggal Pengujian:** 2026-10-12
- **Platform Hardware:** NVIDIA GeForce RTX 5050 Laptop GPU (8GB VRAM), WSL2 Ubuntu Linux
- **VRAM Baseline / Peak:** 814.0 MiB / 2415.0 MiB
- **Dataset Evaluasi:** 129 Ground Truth Queries (108 Formal Subset A + 21 Slang Subset B)

## 1. Perbandingan Kinerja Retrieval Keseluruhan (OVERALL — 129 Kueri)

| Metode Retrieval | Dimensi | Hit@1 (%) | Hit@3 (%) | Hit@5 (%) | MRR | NDCG@5 | Avg Latensi (ms) | P50 (ms) |
|---|---|---|---|---|---|---|---|---|
| **BM25 (Sparse Okapi)** | Sparse | 63.6% | 77.5% | 81.4% | 0.7018 | 0.6807 | 1.0 ms | 0.9 ms |
| **FAISS Dense (nomic-embed-text 768d)** | 768d | 50.4% | 76.0% | 82.2% | 0.6306 | 0.5924 | 30.7 ms | 28.4 ms |
| **FAISS Dense (EmbeddingGemma-2 768d)** | 768d | 80.6% | 85.3% | 87.6% | 0.8345 | 0.8137 | 96.6 ms | 59.4 ms |
| **Hybrid RRF (BM25 + nomic-embed-text)** | 768d | 61.2% | 80.6% | 83.7% | 0.7120 | 0.7007 | 28.0 ms | 27.4 ms |
| **Hybrid RRF (BM25 + EmbeddingGemma-2)** | 768d | 76.7% | 86.8% | 88.4% | 0.8161 | 0.8042 | 82.3 ms | 57.8 ms |
| **Matryoshka Dense (EmbeddingGemma-2 512d)** | 512d | 76.0% | 85.3% | 86.8% | 0.8067 | 0.7933 | 69.6 ms | 56.4 ms |
| **Matryoshka Dense (EmbeddingGemma-2 256d)** | 256d | 76.7% | 84.5% | 87.6% | 0.8050 | 0.7829 | 64.5 ms | 54.2 ms |
| **Matryoshka Dense (EmbeddingGemma-2 128d)** | 128d | 69.0% | 82.2% | 86.0% | 0.7553 | 0.7258 | 62.6 ms | 54.3 ms |

## 2. Analisis Performa Berdasarkan Register Bahasa

### A. Kueri Formal (Subset A — 108 Kueri)

| Metode Retrieval | Hit@1 (%) | Hit@3 (%) | Hit@5 (%) | MRR | NDCG@5 |
|---|---|---|---|---|---|
| **BM25 (Sparse Okapi)** | 66.7% | 82.4% | 86.1% | 0.7387 | 0.7279 |
| **FAISS Dense (nomic-embed-text 768d)** | 50.0% | 78.7% | 85.2% | 0.6429 | 0.6073 |
| **FAISS Dense (EmbeddingGemma-2 768d)** | 83.3% | 87.0% | 88.9% | 0.8560 | 0.8418 |
| **Hybrid RRF (BM25 + nomic-embed-text)** | 63.0% | 84.3% | 88.0% | 0.7394 | 0.7424 |
| **Hybrid RRF (BM25 + EmbeddingGemma-2)** | 79.6% | 88.9% | 90.7% | 0.8421 | 0.8417 |
| **Matryoshka Dense (EmbeddingGemma-2 512d)** | 77.8% | 87.0% | 88.0% | 0.8244 | 0.8176 |
| **Matryoshka Dense (EmbeddingGemma-2 256d)** | 79.6% | 86.1% | 88.9% | 0.8285 | 0.8120 |
| **Matryoshka Dense (EmbeddingGemma-2 128d)** | 72.2% | 85.2% | 89.8% | 0.7895 | 0.7647 |

### B. Kueri Informal / Slang Mahasiswa (Subset B — 21 Kueri)

| Metode Retrieval | Hit@1 (%) | Hit@3 (%) | Hit@5 (%) | MRR | NDCG@5 |
|---|---|---|---|---|---|
| **BM25 (Sparse Okapi)** | 47.6% | 52.4% | 57.1% | 0.5119 | 0.4376 |
| **FAISS Dense (nomic-embed-text 768d)** | 52.4% | 61.9% | 66.7% | 0.5675 | 0.5156 |
| **FAISS Dense (EmbeddingGemma-2 768d)** | 66.7% | 76.2% | 81.0% | 0.7238 | 0.6689 |
| **Hybrid RRF (BM25 + nomic-embed-text)** | 52.4% | 61.9% | 61.9% | 0.5714 | 0.4865 |
| **Hybrid RRF (BM25 + EmbeddingGemma-2)** | 61.9% | 76.2% | 76.2% | 0.6825 | 0.6117 |
| **Matryoshka Dense (EmbeddingGemma-2 512d)** | 66.7% | 76.2% | 81.0% | 0.7159 | 0.6683 |
| **Matryoshka Dense (EmbeddingGemma-2 256d)** | 61.9% | 76.2% | 81.0% | 0.6841 | 0.6330 |
| **Matryoshka Dense (EmbeddingGemma-2 128d)** | 52.4% | 66.7% | 66.7% | 0.5794 | 0.5258 |

## 3. Analisis Matryoshka Representation Learning (MRL Truncation)

Evaluasi degradasi akurasi terhadap pemangkasan dimensi pada EmbeddingGemma-2:

| Dimensi Vektor | Hit@1 (%) | Hit@3 (%) | Hit@5 (%) | MRR | Penghematan Storage Index |
|---|---|---|---|---|---|
| **768d (Full)** | 80.6% | 85.3% | 87.6% | 0.8345 | 0% (Baseline) |
| **512d** | 76.0% | 85.3% | 86.8% | 0.8067 | Hemat 33.3% |
| **256d** | 76.7% | 84.5% | 87.6% | 0.8050 | Hemat 66.7% |
| **128d** | 69.0% | 82.2% | 86.0% | 0.7553 | Hemat 83.3% |

## 4. Temuan Riset & Keputusan Arsitektur

1. **Dominasi Signifikan Dense Retrieval Murni**:
   - `EmbeddingGemma-2 (768d)` mengungguli `nomic-embed-text` secara telak di seluruh metrik Information Retrieval:
     - **Hit@1**: Meningkat dari **50.4% ke 80.6% (+30.2%)**.
     - **Hit@3**: Meningkat dari **76.0% ke 85.3% (+9.3%)**.
     - **MRR**: Melonjak dari **0.6306 ke 0.8345 (+32.3%)**.
     - **NDCG@5**: Melonjak dari **0.5924 ke 0.8137 (+37.4%)**.
   - Struktur asymmetric task instruction (`task: search result | query: ...` vs `title: none | text: ...`) memberikan representasi semantik mata kuliah FTMM yang jauh lebih diskriminatif.

2. **Keunggulan Drastis pada Kueri Slang / Informal Mahasiswa (Subset B)**:
   - Pada kueri informal/singkatan mahasiswa FTMM (21 sampel), Dense `nomic-embed-text` sebelumnya hanya mencapai Hit@3 61.9% dan MRR 0.5675.
   - `EmbeddingGemma-2` melonjakkan Hit@3 menjadi **76.2% (+14.3%)** dan MRR menjadi **0.7238 (+27.5%)**, membuktikan generalisasi semantik bahasa Indonesia non-baku yang jauh lebih kuat.

3. **Ketahanan Matryoshka Representation Learning (MRL)**:
   - Pemangkasan dimensi vektor ke **512d** mempertahankan **Hit@3 identik di 85.3%** dengan penghematan memori indeks sebesar 33.3%.
   - Pemangkasan ke **256d** hanya mengalami penurunan Hit@3 sebesar 0.8% (menjadi **84.5%**), namun memangkas kebutuhan RAM/disk indeks sebesar **66.7%**.
   - Bahkan pada dimensi terkecil **128d (Hit@3 82.2%)**, akurasinya masih jauh melampaui `nomic-embed-text` 768d penuh (76.0%).

4. **Profil Komputasi & Hardware (RTX 5050 Laptop GPU 8GB)**:
   - **VRAM Total**: 2.415,0 MiB (hanya $\approx 29.6\%$ dari kapasitas GPU 8 GB), memberikan headroom lapang $\ge 5,7\text{ GB}$ untuk model generator `qwen2.5:3b`.
   - **Inference Latency**: Rata-rata 96.3 ms (P50: 63.9 ms), sangat memadai untuk target sub-detik chatbot FTMM COMPASS.
   - **Keputusan Arsitektur**: `EmbeddingGemma-2 (768d)` direkomendasikan penuh sebagai *primary retriever backbone* menggantikan `nomic-embed-text`. Format Hybrid RRF (BM25 + EmbeddingGemma-2) mencapai puncak akurasi **Hit@3 86.8% dan Hit@5 88.4%**.
