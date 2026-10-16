# Exp-021: Two-Tier Native Embedding Benchmark (Task A Retrieval + Task B Routing)

## 1. Problem Statement & Tujuan Riset
Mengevaluasi secara kuantitatif performa empat arsitektur model embedding native pada daemon C++ Ollama:
1. `nomic-embed-text` (768d dense & 256d Matryoshka)
2. `nomic-embed-text-v2-moe` (768d MoE & 256d Matryoshka)
3. `qwen3-embedding:0.6b` (1024d high-capacity dense)
4. `bge-m3` (1024d cross-lingual multilingual)

## 2. Struktur Evaluasi Dua Tingkat (Two-Tier Design)
- **Tier 1 (Iso-Dimension Control):** Komparasi Nomic v1 vs Nomic v2 MoE pada dimensi identik (768d vs 256d).
- **Tier 2 (Frontier Pareto Analysis):** Pemenang Tier 1 vs Qwen3 (1024d) vs BGE-M3 (1024d).

## 3. Metrik Evaluasi
- **Task A (369 Mata Kuliah FTMM, 149 Kueri):** Hit@1, Hit@3, Hit@5, MRR, NDCG@5 lintas register Formal (108), Slang (21), dan Bilingual (20).
- **Task B (Router System-1, 60 Kueri):** Akurasi 6-kelas, per-class breakdown, dan margin pemisah Top-1 minus Top-2.
- **Telemetri Sistem:** Latensi P50/P95, GPU VRAM via nvidia-smi, dan verifikasi zero WSL2 disk swap.

Rincian hasil lengkap dapat dilihat pada `benchmark_summary.md` dan `benchmark_metrics.csv`.
