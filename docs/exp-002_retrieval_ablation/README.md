# Eksperimen 002: Retrieval Ablation & Evaluation (Sparse vs Dense vs Hybrid RRF vs Reformulation)

- **Eksperimen ID:** `exp-002`
- **Tanggal:** 2026-10-03
- **Kartu Trello Terkait:**
  - Card 24: `[Nauval & Ario] Implementasi & evaluasi RAG Sparse (BM25)`
  - Card 25: `[Nauval & Ario] Implementasi & evaluasi RAG Dense (embedding similarity)`
  - Card 27: `[Nauval & Ario] Implementasi & evaluasi RAG Hybrid (Sparse + Dense)`
  - Card 28: `[Nauval & Ario] Implementasi & evaluasi RAG Hybrid + Query Reformulation`
  - Card 29: `[Nauval & Ario] Analisis hasil Tahap 1 — tentukan RAG terbaik`
- **Model:**
  - Embedding: `nomic-embed-text` (768 dimensi, Cosine IndexFlatIP FAISS)
  - Reformulasi: `qwen2.5:3b` via Ollama
- **Hardware:** NVIDIA GeForce RTX 5050 (8GB VRAM) / Linux WSL2
- **Testbed Dataset:** 129 sampel kueri (108 Formal dari Ground Truth Google Drive + 21 Informal slang mahasiswa FTMM)
- **Total Pengujian Inferensi:** 516 retrieval runs (129 sampel x 4 metode).

---

## 1. Hipotesis Penelitian
1. **H1 (Sparse vs Dense):** BM25 unggul pada kueri formal dengan kode MK eksak, tetapi mengalami degradasi parah pada kueri informal slang mahasiswa.
2. **H2 (Hybrid RRF Synergy):** Reciprocal Rank Fusion (RRF) menggabungkan kekuatan leksikal BM25 dan semantik FAISS, menghasilkan Hit Rate@k dan Recall yang melampaui kedua metode individual.
3. **H3 (Query Reformulation Latency Trade-off):** Reformulasi kueri meningkatkan akurasi kueri slang, namun menghasilkan overhead latensi inferensi LLM (>300 ms) yang signifikan untuk sistem real-time.

---

## 2. Ringkasan Hasil Utama

| Metode Retrieval | Hit@1 | Hit@3 | Hit@5 | MRR@5 | Recall@3 | Prec@3 | Avg Latensi (ms) | P95 Latensi (ms) |
|---|---|---|---|---|---|---|---|---|
| **BM25 (Sparse)** | **63.6%** | 77.5% | 81.4% | **0.7018** | 65.5% | **44.4%** | **0.8 ms** | **1.8 ms** |
| **FAISS (Dense)** | 50.4% | 76.0% | 82.2% | 0.6306 | 58.1% | 38.8% | 99.1 ms | 52.6 ms |
| **Hybrid RRF** | 45.7% | **81.4%** | **85.3%** | 0.6313 | 63.0% | 42.1% | 28.3 ms | 31.0 ms |
| **Hybrid + Reformulation** | 51.9% | **83.7%** | **88.4%** | 0.6698 | 64.3% | 43.9% | 315.6 ms | 465.3 ms |

### Analisis Ragam Bahasa (Formal vs Informal):
- **BM25 mengalami drop terbesar (-30.0%)** pada kueri informal (dari 82.4% ke 52.4%) karena singkatan mahasiswa ("alpro", "jarkom", "visdat", "metstat") tidak ada dalam teks resmi kurikulum.
- **FAISS Dense paling tahan kueri informal** dengan drop hanya **-16.8%** (dari 78.7% ke 61.9%), membuktikan dense embedding mampu menjembatani sinonim semantik.
- **Hybrid RRF memberikan Pareto frontier terbaik**: Hit@3 naik ke **81.4%** (melampaui BM25 murni dan Dense murni) dengan latensi hanya **28.3 ms**.

---

## 3. Keputusan Rekomendasi (Menutup Card 29 Trello)
1. **Pilihan Arsitektur Standar:** **Hybrid RRF ($k=60$)** ditetapkan sebagai retrieval engine baku pada FTMM COMPASS.
2. **Optimalisasi Kueri Informal:** Query Reformulation sebaiknya tidak dijalankan via LLM per-turn, melainkan digantikan dengan **leksikon singkatan/slang normalizer** lokal (rule-based O(1)) sebelum RRF untuk menjaga latensi <35 ms.

---

## 4. Berkas Artefak
- `benchmark_summary.md`: Tabel perbandingan lengkap dan analisis stratified formal-informal.
- `benchmark_metrics.csv`: Data mentah 516 baris pengukuran per kueri.
- `benchmark_results.json`: Rekap statistik agregat.
