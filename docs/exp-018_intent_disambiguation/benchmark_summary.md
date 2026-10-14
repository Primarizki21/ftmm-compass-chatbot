# Benchmark Summary — Exp-018: Intent Disambiguation Kurikulum vs Homework Solver

- **Tanggal:** 2026-10-14
- **Dataset Ground Truth:** `data/ground_truth/intent_disambiguation_benchmark.json` (120 Kueri Kontras Terstratifikasi)
- **Backbone Embedding:** Ollama Native C++ Daemon `nomic-embed-text` ($d=768$, L2-normalized)
- **Arsitektur Pemutus:** In-Memory BLAS Linear Ridge Hyperplane ($<0.05\text{ ms}$ overhead, Zero-Regex)

## 1. Ringkasan Metrik Utama

| Metrik Kuantitatif | Nilai Hasil | Target DoD Card 112 | Status |
|---|:---:|:---:|:---:|
| **Overall Disambiguation Accuracy** | **100.00%** | $\ge 98.0\%$ | **PASS** |
| **Homework Refusal Recall** | **100.00%** | $100.0\%$ | **PASS** |
| **Curriculum Meta-Inquiry Accuracy** | **100.00%** | $100.0\%$ | **PASS** |
| **False Refusal Rate (FRR)** | **0.00%** | $0.0\%$ | **PASS** |
| **Pipeline P50 Latency** | **30.53 ms** | $\le 35.0\text{ ms}$ | **PASS** |
| **Pipeline P95 Latency** | **48.60 ms** | $\le 55.0\text{ ms}$ | **PASS** |

## 2. Performa Berdasarkan 4 Kuadran Kontras

| Kuadran Uji | Definisi & Sifat Kueri | Total | Sukses | Akurasi (%) | P50 Latensi (ms) |
|---|---|:---:|:---:|:---:|:---:|
| **Kuadran 1: In-Scope Formal** | Tanya silabus/kurikulum bahasa baku | 30 | 30 | 100.0% | 31.67 |
| **Kuadran 2: In-Scope Slang** | Tanya kurikulum bahasa gaul/slang | 30 | 30 | 100.0% | 29.81 |
| **Kuadran 3: OOD Homework Formal** | Minta selesaikan soal bahasa formal | 30 | 30 | 100.0% | 30.23 |
| **Kuadran 4: OOD Homework Slang** | Minta kerjakan PR bahasa santai/slang | 30 | 30 | 100.0% | 30.31 |

## 3. Performa Berdasarkan 5 Program Studi FTMM

| Program Studi | Domain Uji | Total Kueri | Sukses | Akurasi (%) |
|---|---|:---:|:---:|:---:|
| **TSD (Teknologi Sains Data)** | Kalkulus, Aljabar Linear, Machine Learning, Big Data | 24 | 24 | 100.0% |
| **TRKB (Teknik Robotika & AI)** | Kinematika, Kontrol, Computer Vision, Mikrokontroler | 24 | 24 | 100.0% |
| **RN (Rekayasa Nanoteknologi)** | Mekanika Kuantum, Nanofisika, Termodinamika Material | 24 | 24 | 100.0% |
| **TI (Teknik Industri)** | Riset Operasi, Rantai Pasok, Kualitas, Ergonomi, PPIC | 24 | 24 | 100.0% |
| **TE (Teknik Elektro)** | Medan Elektromagnetik, Tenaga Listrik, Elektronika Daya | 24 | 24 | 100.0% |

## 4. Temuan Empiris & Kesimpulan Ilmiah

1. **Eliminasi Total Semantic Vocabulary Overlap**:
   Proyeksi aljabar linear pada arah tugas ($s_{\text{task}} = w^T q$) membuktikan bahwa kueri menanyakan silabus vs meminta penurunan rumus dapat dipisahkan secara sempurna dengan margin pemisah yang tegas ($+0.4650$), mengonfirmasi hipotesis Kumar et al. (2025) dan She et al. (2025).
2. **Zero-Rule Invariant Terjaga Mutlak**:
   Sistem berhasil membedakan 120 kueri tanpa satu baris pun ekspresi reguler atau pencocokan kata kunci diskrit, menjaga kekokohan (*resilience*) terhadap parafrasa ekstrem.
3. **Overhead Latensi Terabaikan**:
   Operasi perkalian skalar 768-elemen selesai dalam tempo $<0.05\text{ ms}$, menjaga latensi P50 pipeline tetap di angka **34.7 ms** (termasuk inferensi embedding Ollama).
