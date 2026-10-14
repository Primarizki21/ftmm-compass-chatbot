# Experiment 020: Hardening Neuro-Symbolic CSP Degree Planner & Dense Gazetteer Normalizer

## 1. Deskripsi Riset & Problem Statement
Sebelumnya, fitur pembuatan rencana studi (`generate_degree_plan`) mengalami **kegagalan fatal (*silent fallback bug*)** ketika memproses kueri singkatan/slang mahasiswa (misalnya input `"TSD"` atau `"ai robot"`):
1. **Normalisasi String Rapuh:** Substring match `"tsd"` gagal mengenali string resmi kurikulum `"Teknologi Sains Data"`, memicu fallback darurat `catalog[:40]` yang secara keliru mengambil kurikulum Rekayasa Nanoteknologi (RN).
2. **Naive Track Matching:** Peminatan seperti `"LLM"` atau `"Data Science"` dicari dengan substring harfiah, sehingga tidak ada mata kuliah pilihan yang terplot.
3. **Semester 5–7 Kosong:** Truncation 40 matkul pertama RN tidak memuat mata kuliah semester 5–7, sehingga jadwal semester atas kosong (`[]`).
4. **Agama Redundan:** Pada semester awal, seluruh 6 varian agama dimasukkan serempak hingga memboroskan beban SKS.

## 2. Solusi Arsitektur (Neuro-Symbolic CSP Planner)
Eksperimen 020 mengimplementasikan arsitektur perencana hibrida yang menggabungkan:
1. **Dense Vector Gazetteer Normalizer:** Memadukan kamus alias kanonikal institusi dengan pencocokan vektor semantik kontinu (`nomic-embed-text` L2 dot product) pada ambang batas similarity $\ge 0.45$.
2. **Semantic Focus Track Expansion:** Peminatan mahasiswa (seperti `"LLM"`, `"NLP"`, `"Big Data"`) diekspansi secara semantik menggunakan FAISS Hybrid Retriever (`top_k=15`, filtered by target prodi).
3. **Deterministic Constraint Satisfaction Problem (CSP) Engine:**
   - **Single Religion Rule:** Membatasi mata kuliah Agama tepat 1 varian (2 SKS) per paket kurikulum.
   - **Semester Parity Enforcement:** Penempatan mata kuliah wajib dan pilihan wajib mematuhi paritas ganjil (1, 3, 5, 7) vs genap (2, 4, 6, 8).
   - **SKS Capacity Cap:** Menegakkan batas beban SKS per semester $\le 24$ SKS (atau batas kustom mahasiswa).
   - **Zero Fallback Invariant:** Mengeliminasi seluruh pemotongan acak `catalog[:40]`.

## 3. Hasil Kuantitatif & Benchmark

| Metrik Evaluasi | Nilai Capaian | Baseline Lama (Exp-003) | Status |
|---|---|---|---|
| **Overall Plan Success Rate** | **100.0%** (20/20) | 60.0% | **PASS** |
| **Prodi Normalization Accuracy** | **100.0%** (20/20) | 40.0% | **PASS** |
| **Zero-Fallback Invariant Rate** | **100.0%** (20/20) | 0.0% | **PASS** |
| **SKS Bound Adherence Rate** | **100.0%** (20/20) | 80.0% | **PASS** |
| **Semester Parity Compliance** | **100.0%** (20/20) | 70.0% | **PASS** |
| **Single-Religion Dedup Rate** | **100.0%** (20/20) | 0.0% | **PASS** |
| **No-Empty Target Semesters** | **100.0%** (20/20) | 30.0% | **PASS** |
| **P50 Planner Latency (ms)** | **31.49 ms** | 4.20 ms | **PASS** |
| **P95 Planner Latency (ms)** | **62.59 ms** | 12.50 ms | **PASS** |

## 4. Struktur Berkas Artefak
- `README.md`: Ringkasan ilmiah dan latar belakang riset.
- `benchmark_summary.md`: Rincian metrik kuantitatif dan analisis empiris.
- `benchmark_results.json`: Rekapitulasi metrik format JSON.
- `benchmark_metrics.csv`: Data telemetri tabular per skenario pengujian.
- `benchmark_raw_traces.jsonl`: Jejak eksekusi mentah per skenario (payload, counts, SKS totals).
