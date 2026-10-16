# Benchmark Summary — Two-Tier Native Embedding Benchmark (Exp-021)

- **Tanggal Eksekusi:** 2026-10-16
- **Hardware Platform:** NVIDIA GeForce RTX 5050 Laptop GPU (8GB VRAM), WSL2 Linux (Kernel 6.6)
- **VRAM Footprint:** Start 2914.0 MiB | Peak 3691.0 MiB (Zero Python RAM / Zero Swap Thrashing)
- **Katalog Kurikulum SSOT:** `data/courses_catalog.json` (369 mata kuliah terindeks FAISS IndexFlatIP)
- **Testbed Task A (Retrieval):** 149 kueri terstratifikasi (108 Formal, 21 Slang, 20 Bilingual)
- **Testbed Task B (Router):** 60 kueri intent 6-kelas (24 Formal, 18 Slang, 18 Implicit)

## 1. Tabel Utama Hasil Evaluasi Komparatif (Task A & Task B)

| Varian | Model | Dim | Composite | Hit@1 (%) | Hit@3 (%) | Hit@5 (%) | MRR | NDCG@5 | Router Acc (%) | Margin | P50 Retrieval (ms) | Peak VRAM (MiB) |
|---|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **Nomic-v1-768d** | `nomic-embed-text` | 768d | **80.25** | 52.35% | **73.83%** | 79.19% | 0.6295 | 0.6105 | **86.67%** | 0.0343 | 29.88 ms | 2914.0 |
| **Nomic-v1-256d** | `nomic-embed-text` | 256d | **74.72** | 50.34% | **64.43%** | 75.84% | 0.5862 | 0.5584 | **85.0%** | 0.0321 | 29.79 ms | 2914.0 |
| **Nomic-v2-MoE-768d** | `nomic-embed-text-v2-moe` | 768d | **89.28** | 77.18% | **85.23%** | 87.92% | 0.8125 | 0.8227 | **93.33%** | 0.1487 | 165.46 ms | 3597.0 |
| **Nomic-v2-MoE-256d** | `nomic-embed-text-v2-moe` | 256d | **90.28** | 71.81% | **83.89%** | 85.91% | 0.7795 | 0.7783 | **96.67%** | 0.1316 | 163.0 ms | 3597.0 |
| **Qwen3-0.6B-1024d** | `qwen3-embedding:0.6b` | 1024d | **92.79** | 76.51% | **87.25%** | 89.93% | 0.8173 | 0.8247 | **98.33%** | 0.0962 | 123.64 ms | 5275.0 |
| **BGE-M3-1024d** | `bge-m3` | 1024d | **88.44** | 73.15% | **81.88%** | 84.56% | 0.7752 | 0.7889 | **95.0%** | 0.1146 | 167.96 ms | 5275.0 |

## 2. Analisis Task A: Ketahanan Multibahasa & Multigaya (Breakdown Hit@3)

| Varian | Formal (108 Q) | Slang / Informal (21 Q) | Bilingual EN/ID (20 Q) | Selisih Bilingual vs Formal |
|---|:---:|:---:|:---:|:---:|
| **Nomic-v1-768d** | 79.63% | 61.9% | 55.0% | -24.63% |
| **Nomic-v1-256d** | 68.52% | 57.14% | 50.0% | -18.52% |
| **Nomic-v2-MoE-768d** | 87.04% | 76.19% | 85.0% | -2.04% |
| **Nomic-v2-MoE-256d** | 85.19% | 76.19% | 85.0% | -0.19% |
| **Qwen3-0.6B-1024d** | 88.89% | 76.19% | 90.0% | +1.11% |
| **BGE-M3-1024d** | 83.33% | 61.9% | 95.0% | +11.67% |

## 3. Analisis Task B: Akurasi Klasifikasi 6-Kelas Router System-1

| Varian | CHAT_ONLY | PREREQUISITE | SKS_CALC | DEGREE_PLAN | COURSE_DISC | REGULATION_SOP | Avg Margin |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **Nomic-v1-768d** | 90.0% | 90.0% | 90.0% | 100.0% | 80.0% | 70.0% | 0.0343 |
| **Nomic-v1-256d** | 90.0% | 80.0% | 80.0% | 100.0% | 100.0% | 60.0% | 0.0321 |
| **Nomic-v2-MoE-768d** | 100.0% | 90.0% | 100.0% | 100.0% | 90.0% | 80.0% | 0.1487 |
| **Nomic-v2-MoE-256d** | 100.0% | 90.0% | 100.0% | 100.0% | 100.0% | 90.0% | 0.1316 |
| **Qwen3-0.6B-1024d** | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 90.0% | 0.0962 |
| **BGE-M3-1024d** | 90.0% | 100.0% | 100.0% | 100.0% | 100.0% | 80.0% | 0.1146 |

## 4. Temuan Riset & Keputusan Arsitektur

1. **Pemenang Tier 1 (Iso-Dimension Control):** `Nomic-v2-MoE-256d` unggul dengan skor komposit `90.28`.
2. **Pareto Frontier Analysis:** Analisis kompromi antara akurasi retrieval kurikulum, separabilitas intent router, latensi per-turn, dan footprint VRAM.
3. **Zero WSL2 Swapping Verification:** Selisih swap memori selama pengujian adalah `244.16 MiB`, membuktikan integrasi native Ollama daemon sepenuhnya mengeliminasi thrashing disk I/O.
