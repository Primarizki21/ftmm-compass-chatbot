# Exp-026: Kalibrasi Ulang System-1 Decider & Evaluasi Komparatif Generasi Pure RAG vs Pure CAG vs Hybrid

## 📌 Ringkasan Eksperimen
Eksperimen **Exp-026** merekalibrasi ambang batas separabilitas matriks BLAS System-1 Decider dan memperluas prototipe regulasi institusi dari 9 menjadi 28 prototipe representatif, serta menyesuaikan alokasi output token (num_predict=512) dan ekstraksi/pembersihan tag reasoning `<think>` pada engine Pure CAG dan Pure RAG. Evaluasi dijalankan pada seluruh 449 QA kueri terpadu `data/ground_truth/gt_master.json`.

---
## 🎯 Konfigurasi Model & Runtime
- **Primary Generator**: `bonsai-2-4b` (Ternary 1.58-bit PQ2_0 via CUDA `llama-server` port 8080, context size 4096, `-np 1`, `--cache-ram 0`, `--no-cache-prompt`).
- **Embedding Model**: `nomic-embed-text-v2-moe` (768d MRL Native BLAS via Ollama daemon port 11434).
- **System-1 Decider**: 28 prototipe `REGULATION_SOP`, min_confidence_threshold=0.35, margin_threshold=0.012, L2-normalized 6x768 centroid matrix.

---
## 📊 Ringkasan Temuan Empiris

### 1. Kinerja Lapisan Pengarah & Temu Balik Informasi
- **Tier 2 Semantic Gate FRR**: **4.23%**
- **System-1 Decider Routing Accuracy**: **78.62%** (Latensi P50: **0.111 ms**)
- **Dense Course Retrieval (369 MK)**: Hit@1 **95.7%**, Hit@3 **99.7%**, Hit@5 **99.7%**, MRR **0.9765**, NDCG@5 **0.9819**

### 2. Komparasi Komprehensif Arsitektur
| Arsitektur | Fact Recall (%) | Semantic Sim | Latensi P50 (ms) | Tokens / Turn | Putusan Pareto |
|---|---|---|---|---|---|
| **Pure CAG** | **69.98%** | **0.5645** | 4075.3 ms | 512.0 | Fastest TTFT |
| **Pure RAG** | **77.13%** | **0.6018** | 3972.5 ms | 512.0 | Granular Retrieval |
| **Hybrid RAG-CAG** | **55.96%** | **0.5895** | 9121.6 ms | 1048.2 | PARETO OPTIMAL (Winner) |

---
## 📁 Berkas Artefak
- `benchmark_summary.md`: Ringkasan naratif eksekutif dan putusan Pareto untuk Bab 4 Laporan TA.
- `benchmark_metrics.csv`: Data tabular agregat metrik komparatif tiga paradigma.
- `benchmark_results.json`: Rekap JSON terstruktur per metrik.
- `benchmark_raw_traces.jsonl`: 449 rekaman trace eksekusi lengkap per kueri.
- Skrip eksekusi: `scripts/evaluate_holistic_gt_master.py`.
- Unit test regresi: `tests/test_holistic_gt_master.py`.
