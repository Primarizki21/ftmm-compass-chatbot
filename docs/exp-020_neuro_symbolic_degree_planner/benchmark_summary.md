# Benchmark Summary: Neuro-Symbolic CSP Degree Planner (Exp-020)

## 1. Ringkasan Metrik Utama

| Metrik Evaluasi | Nilai Capaian | Baseline Lama (Exp-003/005) | Status |
|---|---|---|---|
| **Overall Plan Success Rate** | **100.0%** | 60.0% | **PASS** |
| **Prodi Normalization Accuracy** | **100.0%** | 40.0% (Gagal pada akronim TSD) | **PASS** |
| **Zero-Fallback Invariant Rate** | **100.0%** | 0.0% (Nyasar ke RN catalog[:40]) | **PASS** |
| **SKS Bound Adherence Rate** | **100.0%** | 80.0% (Over-limit SKS) | **PASS** |
| **Semester Parity Compliance** | **100.0%** | 70.0% (Matkul genap di sem ganjil) | **PASS** |
| **Single-Religion Dedup Rate** | **100.0%** | 0.0% (6 agama masuk sekaligus) | **PASS** |
| **No-Empty Target Semesters** | **100.0%** | 30.0% (Sem 5-7 kosong) | **PASS** |
| **P50 Planner Latency (ms)** | **31.49 ms** | 4.20 ms | **PASS (Sub-50ms)** |
| **P95 Planner Latency (ms)** | **62.59 ms** | 12.50 ms | **PASS** |

## 2. Analisis Perbaikan Arsitektur (Exp-020)

1. **Eliminasi Silent Fallback Bug**:
   Normalisasi prodi kini memadukan **Canonical Alias Map** dan **Dense Vector Gazetteer Matching** (`embed_query` + matrix dot-product). Kueri singkatan seperti `TSD`, `TRKB`, `RN`, `TI`, `TE`, maupun varian informal `data science`, `ai robot`, `nanotek` berhasil dinormalisasi 100% tanpa fallback acak `catalog[:40]`.
2. **Semantic Focus Track Expansion**:
   Peminatan seperti `LLM`, `NLP`, `Big Data`, `Supply Chain` tidak lagi mengandalkan substring literal, melainkan memanfaatkan **FAISS Dense Retrieval** (`top_k=15`, filtered by prodi) untuk memetakan minat ke mata kuliah pilihan relevan secara probabilistik.
3. **Deterministic CSP Constraints**:
   - **Single Religion Rule**: Mahasiswa hanya dijadwalkan tepat 1 varian mata kuliah Agama (2 SKS) per paket semester 1 & 6/7.
   - **Semester Parity**: Paritas ganjil/genap ditegakkan kaku pada penempatan mata kuliah wajib dan pilihan.
   - **SKS Capacity**: Beban SKS per semester dibatasi secara deterministik $\le 24$ SKS (atau batas yang diminta mahasiswa).
