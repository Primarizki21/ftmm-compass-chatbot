# FTMM COMPASS — Master Experiment Ledger

> **Klasifikasi:** Dokumen Publik Portofolio & Log Riset Ilmiah (Tracked in Git).
> **Fungsi:** Buku induk pencatatan kronologis seluruh eksperimen komparasi arsitektur chatbot FTMM COMPASS.
> **Aturan Agent:** Setiap agent **WAJIB MEMBACA** file ini sebelum merancang eksperimen baru, dan **WAJIB MENG-APPEND** hasil eksperimen baru ke tabel dan seksi detail di bawah ini.

---

## 1. Master Experiment Table

| Exp ID | Tanggal | Topik & Kartu Trello | Model Generatif & Embedding | Answer Correctness | Semantic Similarity | Fact Recall | Avg Retrieval (ms) | Avg Latensi Total (ms) | Konsistensi (Std Dev) | Keputusan & Temuan Utama | Path Artefak |
|---|---|---|---|---|---|---|---|---|---|---|---|
| **`exp-001`** | 2026-09-30 | Baseline Komparasi 3 Paradigma (Card 31, 32, 36) | `qwen2.5:3b` + `nomic-embed-text` | Pure RAG: 0.5903<br>Pure CAG: **0.7043**<br>Hybrid: 0.6836 | Pure RAG: 0.7831<br>Pure CAG: **0.8350**<br>Hybrid: 0.8078 | Pure RAG: 30.11%<br>Pure CAG: **50.83%**<br>Hybrid: 49.72% | Pure RAG: 34.3 ms<br>Pure CAG: **0.0 ms**<br>Hybrid: **3.5 ms** | Pure RAG: 2329.8 ms<br>Pure CAG: **1665.4 ms**<br>Hybrid: 2507.6 ms | Pure RAG: ±0.1825<br>Pure CAG: ±0.1718<br>Hybrid: **±0.1561** | **Hybrid RAG-CAG terbukti paling stabil** (varians terendah ±0.1561) dan paling tahan kueri slang/informal (delta hanya +0.0435). CAG tercepat (0 ms retrieval). | `docs/exp-001_baseline_cag_rag_hybrid/` |
| **`exp-002`** | 2026-10-03 | Retrieval Ablation & Evaluation (Card 24, 25, 27, 28, 29) | `nomic-embed-text` + `qwen2.5:3b` (reform) | BM25: Hit@3 77.5%<br>Dense: Hit@3 76.0%<br>Hybrid RRF: **81.4%**<br>Reform: **83.7%** | BM25: MRR 0.7018<br>Dense: MRR 0.6306<br>Hybrid RRF: 0.6313<br>Reform: **0.6698** | BM25: Recall@3 65.5%<br>Dense: 58.1%<br>Hybrid RRF: 63.0%<br>Reform: 64.3% | BM25: **0.8 ms**<br>Dense: 99.1 ms<br>Hybrid RRF: **28.3 ms**<br>Reform: 315.6 ms | BM25: 0.8 ms<br>Dense: 99.1 ms<br>Hybrid RRF: 28.3 ms<br>Reform: 315.6 ms | BM25: P95 1.8 ms<br>Dense: P95 52.6 ms<br>Hybrid RRF: **P95 31.0 ms**<br>Reform: P95 465.3 ms | **Hybrid RRF ($k=60$) terpilih sebagai backbone terbaik**: mengungguli BM25 & Dense murni (Hit@3 81.4%, Hit@5 85.3%) pada latensi sangat rendah (28.3 ms). BM25 drop -30.0% pada kueri slang. Menutup Card 29. | `docs/exp-002_retrieval_ablation/` |

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

---

### [exp-002] Retrieval Ablation & Evaluation: BM25 vs FAISS vs Hybrid RRF vs Query Reformulation
- **Tanggal:** 2026-10-03
- **Kartu Trello:** Card 24 (BM25 Sparse), Card 25 (Dense FAISS), Card 27 (Hybrid Sparse+Dense), Card 28 (Hybrid + Reformulation), Card 29 (Analisis Tahap 1 & Pemilihan RAG Terbaik).
- **Tujuan / Hipotesis:**
  Mengevaluasi secara kuantitatif performa Information Retrieval (IR) dari 4 varian backbone retrieval pada korpus 369 mata kuliah FTMM UNAIR. Menjawab pertanyaan apakah dense vector search saja cukup atau membutuhkan gabungan leksikal BM25 dan ekspansi kueri.
- **Setup Teknis (Low-Level Architecture):**
  - **Sparse Retriever:** Pure Python/NumPy BM25 Okapi ($k_1=1.5, b=0.75$) diindeks langsung pada 369 course chunks.
  - **Dense Retriever:** FAISS IndexFlatIP (Cosine Similarity) dengan embedding `nomic-embed-text` (768 dimensi, L2-normalized).
  - **Hybrid RRF:** Reciprocal Rank Fusion ($k=60$) menggabungkan peringkat sparse dan dense tanpa normalisasi skor manual.
  - **Hybrid Reformulation:** Ollama `qwen2.5:3b` satu-baris zero-shot normalizer kueri informal sebelum proses RRF.
  - **Dataset Pengujian:** 129 kueri terstratifikasi (108 kueri Formal dari Ground Truth Google Drive + 21 kueri Informal slang mahasiswa). Total 516 run evaluasi.
- **Temuan Kunci:**
  1. **Keunggulan Sinergi Hybrid RRF (Card 27 & 29):**
     Hybrid RRF mencatat Hit@3 sebesar **81.4%** dan Hit@5 sebesar **85.3%**, mengungguli Dense FAISS murni (Hit@3 76.0%) dan BM25 murni (Hit@3 77.5%).
  2. **Kerentanan BM25 terhadap Bahasa Informal (Card 24):**
     BM25 sangat cepat (0.8 ms) dan akurat pada kueri kode MK eksak (Hit@3 formal 82.4%), namun anjlok drastis ke **52.4% (-30.0%)** pada kueri informal slang ("alpro", "jarkom", "visdat", "metstat").
  3. **Ketahanan Dense terhadap Bahasa Informal (Card 25):**
     Dense FAISS mempertahankan performa lebih baik pada kueri informal (Hit@3 61.9%, drop hanya -16.8%), membuktikan dense embedding mampu mengenali sinonim konseptual.
  4. **Trade-off Latensi Query Reformulation (Card 28):**
     Query Reformulation menghasilkan Hit@3 tertinggi (83.7%), namun membebani latensi rata-rata **315.6 ms** (p95 465.3 ms) akibat inferensi LLM.
- **Keputusan Arsitektur & Tindak Lanjut:**
  - Memilih **Hybrid RRF** sebagai standar mesin retrieval FTMM COMPASS.
  - Menutup tugas Card 29 di Trello board dan menyiapkan migrasi engine untuk Sprint 6.
- **Artefak:**
  - `docs/exp-002_retrieval_ablation/README.md`
  - `docs/exp-002_retrieval_ablation/benchmark_summary.md`
  - `docs/exp-002_retrieval_ablation/benchmark_metrics.csv`
  - `docs/exp-002_retrieval_ablation/benchmark_results.json`
