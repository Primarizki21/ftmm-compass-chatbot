# FTMM COMPASS — Master Experiment Ledger

> **Klasifikasi:** Dokumen Publik Portofolio & Log Riset Ilmiah (Tracked in Git).
> **Fungsi:** Buku induk pencatatan kronologis seluruh eksperimen komparasi arsitektur chatbot FTMM COMPASS.
> **Aturan Agent:** Setiap agent **WAJIB MEMBACA** file ini sebelum merancang eksperimen baru, dan **WAJIB MENG-APPEND** hasil eksperimen baru ke tabel dan seksi detail di bawah ini.

---

## 1. Master Experiment Table

| Exp ID | Tanggal | Topik & Kartu Trello | Model Generatif & Embedding | Answer Correctness | Semantic Similarity | Fact Recall | Avg Retrieval (ms) | Avg Latensi Total (ms) | Konsistensi (Std Dev) | Keputusan & Temuan Utama | Path Artefak |
|---|---|---|---|---|---|---|---|---|---|---|---|
| **`exp-001`** | 2026-09-30 | Baseline Komparasi 3 Paradigma (Card 31, 32, 36) | `qwen2.5:3b` + `nomic-embed-text` | Pure RAG: 0.5903<br>Pure CAG: **0.7043**<br>Hybrid: 0.6836 | Pure RAG: 0.7831<br>Pure CAG: **0.8350**<br>Hybrid: 0.8078 | Pure RAG: 30.11%<br>Pure CAG: **50.83%**<br>Hybrid: 49.72% | Pure RAG: 34.3 ms<br>Pure CAG: **0.0 ms**<br>Hybrid: **3.5 ms** | Pure RAG: 2329.8 ms<br>Pure CAG: **1665.4 ms**<br>Hybrid: 2507.6 ms | Pure RAG: ±0.1825<br>Pure CAG: ±0.1718<br>Hybrid: **±0.1561** | **Hybrid RAG-CAG terbukti paling stabil** (varians terendah ±0.1561) dan paling tahan kueri slang/informal (delta hanya +0.0435). CAG tercepat (0 ms retrieval). | `docs/exp-001_baseline_cag_rag_hybrid/` |

---

## 2. Detail Log & Analisis per Eksperimen

### [exp-001] Baseline Komparasi 3 Paradigma: Pure RAG vs Pure CAG vs Hybrid RAG-CAG
- **Tanggal:** 2026-09-30
- **Kartu Trello:** Card 31 (CAG), Card 32 (Hybrid RAG+CAG), Card 33 (Subset A Formal), Card 34 (Subset B Informal), Card 36 (Analisis RQ1-RQ3).
- **Tujuan / Hipotesis:**
  Menguji hipotesis bahwa arsitektur Hybrid RAG-CAG (memadukan NLU router + pre-cached SOP rules + selective FAISS course retrieval) mampu mengungguli Pure RAG pada kueri informal tanpa menanggung penalti latensi retrieval di setiap turn dialog.
- **Setup Teknis (Low-Level Architecture):**
  - **Generator:** Ollama `qwen2.5:3b` running on local GPU NVIDIA RTX 5050 (8GB VRAM).
  - **Embedding:** `nomic-embed-text` (768 dimensi, Cosine IndexFlatIP FAISS).
  - **Corpus RAG:** 369 mata kuliah dari 5 prodi FTMM UNAIR (*Row-based & Column-aware Chunking*).
  - **Corpus CAG:** 8 SOP Administrasi & Regulasi Akademik (SKS, PKL, Skripsi, Yudisium) (*Schema-aware Chunking* ~811 token pre-cached system prompt).
  - **Evaluasi:** 90 run inferensi (15 kueri formal Subset A + 15 kueri informal Subset B $\times$ 3 arsitektur).
- **Temuan Kunci:**
  1. **RQ1 (Arsitektur Terbaik):**
     Hybrid RAG-CAG membuktikan konsistensi semantik terbaik (std dev $\pm 0.1561$) dibanding Pure CAG ($\pm 0.1718$) dan Pure RAG ($\pm 0.1825$).
  2. **RQ2 (Pengaruh Ragam Bahasa Formal vs Informal):**
     Pada kueri informal (slang/singkatan mahasiswa), Pure RAG mengalami penurunan skor sebesar **-0.0552** dan Pure CAG **-0.0746**. Hybrid RAG-CAG mencatat penurunan terkecil yaitu hanya **-0.0435** (dari 0.7053 ke 0.6618), membuktikan bahwa NLU Intent Classifier sukses menormalisasi maksud kueri sebelum menyentuh retrieval.
  3. **RQ3 (Trade-off Tiga Dimensi):**
     Pure CAG unggul pada latensi total (1665.4 ms) dan 0 ms retrieval overhead. Hybrid RAG-CAG memberikan titik temu optimal (*sweet spot*): rata-rata retrieval hanya 3.5 ms karena kueri regulasi diproses via CAG tanpa query ke FAISS.
- **Artefak:**
  - `docs/exp-001_baseline_cag_rag_hybrid/benchmark_summary.md`
  - `docs/exp-001_baseline_cag_rag_hybrid/benchmark_metrics.csv`
  - `docs/exp-001_baseline_cag_rag_hybrid/benchmark_results.json`
