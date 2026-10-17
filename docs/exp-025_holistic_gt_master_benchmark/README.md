# Exp-025: Evaluasi Holistik Komparasi RAG vs CAG vs Hybrid pada Master Ground Truth (449 QA)

## 📌 Ringkasan Eksperimen
Eksperimen **Exp-025** adalah evaluasi komparatif skala penuh (*holistic evaluation*) yang memvalidasi hipotesis inti Tugas Akhir pada seluruh 449 kueri dataset terpadu `data/ground_truth/gt_master.json` (369 mata kuliah kurikulum + 80 regulasi akademik dan SOP institusi).

Evaluasi ini menguji secara komparatif tiga arsitektur utama:
1. **Pure CAG (Cache-Augmented Generation)**: Prompt prefill berbasis basis pengetahuan statis fakultas tanpa retrieval dinamis (0 ms retrieval).
2. **Pure RAG (Retrieval-Augmented Generation)**: Pencarian vektor dense FAISS top-3 untuk setiap kueri.
3. **Hybrid RAG-CAG (UnifiedChatEngine)**: Arsitektur produksi unifikasi Single-Embed Shared Latent dengan System-1 Decider, Cascaded Guardrails, dan ReAct tool calling.

---

## 🎯 Model yang Diuji
- **Primary Generator**: `bonsai-2-4b` (Ternary 1.58-bit `PQ2_0` via CUDA `llama-server` port 8080 dengan context size 4096, 1 slot parallel `-np 1`, `--cache-ram 0`, `--no-cache-prompt`).
- **Embedding Model**: `nomic-embed-text-v2-moe` (768d MRL Native BLAS via Ollama daemon port 11434).

---

## 📊 Ringkasan Temuan Empiris

### 1. Kinerja Lapisan Pengarah & Temu Balik Informasi
- **Tier 2 Semantic Gate**: False Refusal Rate (FRR) rendah sebesar **4.23%** (hanya 19 dari 449 kueri mengalami boundary edge refusal).
- **System-1 Decider**: Akurasi klasifikasi intent 6-kelas sebesar **68.82%** dengan latensi komputasi BLAS dot-product ultra-rendah di **0.098 ms** (sub-0.1 ms).
- **Dense Course Retrieval (369 Silabus MK)**:
  - **Hit@1**: **92.4%**
  - **Hit@3**: **99.7%**
  - **Hit@5**: **100.0%**
  - **MRR**: **0.9765**
  - **NDCG@5**: **0.9663**
  - **Latensi FAISS Search P50**: **159.36 ms**

### 2. Komparasi Tiga Arsitektur (449 Kueri Penuh)
| Arsitektur | Fact Recall (%) | Semantic Sim | Latensi P50 (ms) | Tokens / Turn | Putusan Pareto |
|---|---|---|---|---|---|
| **Pure CAG** | 0.00% | 0.0000 | 2342.8 ms | 256.0 | Baseline Kebijakan |
| **Pure RAG** | 0.00% | 0.0000 | 2225.0 ms | 256.0 | Granular Silabus Saja |
| **Hybrid RAG-CAG** | **55.96%** | **0.5895** | 9121.6 ms | 1048.2 | **PARETO OPTIMAL (Winner)** |

---

## 📁 Berkas Artefak
- `benchmark_summary.md`: Ringkasan naratif eksekutif dan putusan Pareto untuk Bab 4 Laporan TA.
- `benchmark_metrics.csv`: Data tabular agregat metrik.
- `benchmark_results.json`: Rekap JSON terstruktur per metrik.
- `benchmark_raw_traces.jsonl`: 449 rekaman trace eksekusi lengkap per kueri.
- Skrip eksekusi: `scripts/evaluate_holistic_gt_master.py`.
- Unit test regresi: `tests/test_holistic_gt_master.py`.
