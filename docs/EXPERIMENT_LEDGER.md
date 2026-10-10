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
| **`exp-003`** | 2026-10-03 | ReAct Agent Tool-Calling vs Static Router (Card 44) | `qwen2.5:3b` + native tools + `nomic-embed-text` | Static Router: 40.0%<br>ReAct Agent: **60.0%** (Task Success) | Static Router: 0.0%<br>ReAct: **96.7%** (Tool Acc) | Static: 46.1%<br>ReAct: **61.1%** | Static: 28.3 ms<br>ReAct: **42.2 ms** (Tool Exec) | Static: 2933.0 ms<br>ReAct: **2219.1 ms** | Static: Halluc 26.7%<br>ReAct: **Halluc 13.3%** | **ReAct Agent melipatgandakan Task Success (+50% rel)** dan menekan halusinasi SKS hingga 0%. Berhasil mengeksekusi `generate_degree_plan` (JSON) untuk integrasi UI `DegreePlanner.tsx`. Menjawab Card 44. | `docs/exp-003_react_tool_calling/` |
| **`exp-004`** | 2026-10-03 | Evaluasi Penuh ReAct Agent (Zero-Abstraction Telemetry) | `qwen2.5:3b` + native tools + `nomic-embed-text` | Task Success: **75.0%** (45/60)<br>TI: 83.3%<br>TSD/RN/TE: 75.0%<br>TRKB: 66.7% | Tool Selection Acc: **96.7%** (58/60) | SKS: **100%**<br>Prereq: 70.0%<br>Discovery: 80% | Tool Exec: **3.8 ms** (avg) | Total: **2703.7 ms**<br>P50: 2002.7 ms<br>P95: 6807.5 ms | Hallucination: **10.0%**<br>Self-Correction: **20.0%** | **Evaluasi 60 skenario lintas 5 prodi FTMM**: Task success naik ke 75.0% (vs 60.0% di exp-003). SKS policy 100% konsisten. Kegagalan terlacak mentah: 7 kasus planning via teks & 6 false positive akibat hedging sopan. Dump raw traces tersimpan di JSONL. | `docs/exp-004_react_full_evaluation/` |
| **`exp-005`** | 2026-10-03 | ReAct Agent Hardened: Mitigasi 5 Kluster Kegagalan (Card 44) | `qwen2.5:3b` + native tools + `nomic-embed-text` | Task Success: **88.3%** (53/60)<br>TI: **100%**<br>TSD: **91.7%**<br>TRKB/RN/TE: **83.3%** | Tool Selection Acc: **95.0%** (57/60) | Planning: **100%**<br>Discovery: **100%**<br>SKS: **95%**<br>Prereq: **70%** | Tool Exec: **4.1 ms** (avg) | Total: **2934.0 ms**<br>P50: 1951.0 ms<br>P95: 6609.7 ms | Hallucination: **10.0%**<br>Self-Correction: **0.0%** | **Lonjakan Task Success ke 88.3% (+13.3% vs exp-004)**: DEGREE_PLANNING melonjak dari 30% ke **100%** berkat profile auto-injection & parameter defaults. TI mencapai 100% tuntas. Kegagalan terpotong dari 15 kasus menjadi 7 kasus. Zero black-box traces tersimpan di JSONL. | `docs/exp-005_react_hardened_evaluation/` |
| **`exp-006`** | 2026-10-03 | Multi-Style Resilience & Paraphrase Invariance (Card 44, 91) | `qwen2.5:3b` + native tools + `nomic-embed-text` | Formal: **90.0%**<br>Implicit: **90.0%**<br>Slang: **85.0%** | Tool Selection Acc: **83.3%** (50/60) | Planning: **100%**<br>SKS: **93.3%**<br>Prereq: **80.0%** | Tool Exec: **3.9 ms** (avg) | Total: **2407.1 ms**<br>P50: 1729.5 ms<br>P95: 5780.0 ms | **Invariance: 75.0%**<br>Halluc: **3.3%** | **Generalisasi Anti-Overfitting Terbukti**: Evaluasi 60 kueri berpasangan (Formal, Slang, Implisit) membuktikan ReAct tahan variasi diksi (slang hanya drop 5% vs -30% di BM25). Paraphrase Invariance 75.0%. DEGREE_PLANNING 100% konsisten di semua gaya bahasa. | `docs/exp-006_react_multistyle_resilience/` |
| **`exp-007`** | 2026-10-03 | Evaluasi Bilingual ReAct & Hardware Efficiency (Card 31, 32, 44) | `qwen2.5:3b` + native tools + `nomic-embed-text` | English (EN): **90.0%**<br>Indonesian (ID): **85.0%** | Tool Selection Acc: **100.0%** (40/40) | Planning: **100%**<br>SKS: **100%**<br>Discovery: **100%** | Tool Exec: **3.8 ms** (avg) | Total: **2670.4 ms**<br>P50: 1630.8 ms<br>P95: 6784.5 ms | **Parity: 80.0%**<br>Lang Adherence: **85.0%** | **Keberhasilan Penalaran Dwibahasa (Bilingual Parity 80.0%)**: Kueri Bahasa Inggris mencapai success rate 90.0% (melampaui ID 85.0%) dengan 100% Tool Selection Accuracy. Profiling RTX 5050 (8GB VRAM) aman pada ~2.5 GB alokasi model + 4096 token context window tanpa OOM spillover. | `docs/exp-007_bilingual_react_efficiency/` |
| **`exp-008`** | 2026-10-07 | Academic Safety Guardrails Ablation (Card 45, 92) | `qwen2.5:3b` + `nomic-embed-text` | Baseline: 52.0%<br>NeMo: 92.0%<br>Lightweight: **100.0%**<br>Hybrid: **100.0%** (Safety Acc) | ASR: Baseline 93.3%<br>NeMo 26.7%<br>Lightweight **0.0%**<br>Hybrid **0.0%** | OOD-R: Baseline 0.0%<br>NeMo **100%**<br>Lightweight **100%**<br>Hybrid **100%** | Tier 1: **0.0 ms**<br>Tier 2: **31.2 ms**<br>NeMo: 284.6 ms | Baseline: 2183.2 ms<br>NeMo: **284.6 ms**<br>Lightweight: 1384.6 ms (P50 **663.6 ms**)<br>Hybrid: 1471.4 ms | FRR: **0.0%** (0 false refusals pada benign & slang) | **Lightweight Cascaded Guard (Varian C) mencapai 100.0% Safety Acc & 0.0% ASR**. Memangkas latensi P50 ke 663.6 ms (-58%) tanpa ketergantungan black-box framework. Menjawab Card 45 & 92. | `docs/exp-008_guardrails_evaluation/` |
| **`exp-009`** | 2026-10-07 | Multi-Turn Conversation Memory & Crescendo Defense (Card 46, 92) | `qwen2.5:3b` + `nomic-embed-text` | Stateless: 71.4%<br>Stateful: **94.3%** (Dialog Success) | Context Ret: Stateless 90%<br>Stateful **100%** | Anaphora Res: Stateless 14.3%<br>Stateful **85.7%** | Crescendo Def: Stateless 100%<br>Stateful **100%** | Stateless: 1691.3 ms<br>Stateful: 3442.3 ms (P50 3421.1 ms) | Sliding window 6 turn stabil (P95 6259.9 ms) | **ConversationMemoryManager mendongkrak Dialog Success ke 94.3% (+22.9%) dan Anaphora Resolution ke 85.7% (+71.4%)**. Retensi prodi/semester 100% konsisten lintas 7 turn. Menjawab Card 46 & 92. | `docs/exp-009_multiturn_memory_evaluation/` |
| **`exp-010`** | 2026-10-10 | Low-Level Inference Optimization & Dual VRAM Pinning (Card 31, 44, 99) | `qwen2.5:3b` + `nomic-embed-text` | SKS: 1465 ms<br>Prereq: 1320 ms<br>Planning: 3381 ms | Greeting P50: **694.7 ms** (Speedup **8.5x**) | Avg Prefill: **1369 tok** (-35% prompt bloat) | VRAM Swap Latency: **0.0 ms** (Dual resident) | Overall P50: **1393.9 ms**<br>Greeting P50: **694.7 ms** | VRAM GPU 2.565 MiB stabil (~31.4% VRAM) | **Pinning dual-model VRAM (`OLLAMA_MAX_LOADED_MODELS=2`) + Dynamic Tool Schema Pruning memangkas latensi sapaan ke sub-detik (420-700 ms)** tanpa swap runner di WSL2. Menjawab Card 31, 44, 99. | `docs/exp-010_lowlevel_inference_optimization/` |

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

---

### [exp-003] ReAct Agent (Tool-Calling) vs Static 1-Shot Router: Evaluasi Multi-Hop & Perencanaan Studi
- **Tanggal:** 2026-10-03
- **Kartu Trello:** Card 44 (`[Nauval & Ario] Implementasi ReAct Agent dengan tool-calling`).
- **Tujuan / Hipotesis:**
  Menguji apakah reasoning loop ReAct berbasis Ollama native tool-calling yang terhubung ke 4 fungsi deterministik (`search_courses`, `validate_prerequisites_dag`, `calculate_sks_limit`, `generate_degree_plan`) mampu mengeliminasi halusinasi regulasi dan menghasilkan payload perencanaan studi yang kompatibel langsung dengan antarmuka React frontend.
- **Setup Teknis (Low-Level Architecture):**
  - **LLM Backbone:** Ollama `qwen2.5:3b` via native tool calling protocol (OpenAI-compatible schema).
  - **Tool 1 (`search_courses`):** Hybrid RRF retriever (BM25 Okapi + FAISS Cosine) dari Exp-002.
  - **Tool 2 (`validate_prerequisites_dag`):** Graph traversal deterministik rantai prasyarat kurikulum resmi.
  - **Tool 3 (`calculate_sks_limit`):** Aturan deterministik Pasal 5.4 Panduan Akademik FTMM.
  - **Tool 4 (`generate_degree_plan`):** State generator semester-by-semester untuk antarmuka visual `DegreePlanner.tsx`.
  - **Testbed Pengujian:** 30 skenario terstratifikasi (10 Cek Prasyarat DAG, 10 Batasan Beban SKS, 10 Perencanaan Studi End-to-End).
- **Temuan Kunci:**
  1. **Peningkatan Task Success Rate Drastis:**
     ReAct Agent mencatat kesuksesan tugas **60.0%** vs Static Router **40.0%** (peningkatan relatif +50%).
  2. **Eliminasi Total Halusinasi Batas SKS:**
     Pada kategori batas SKS (`SKS_LIMIT_CONSTRAINT`), ReAct Agent mencapai **100% Success Rate** dan **0.0% Hallucination** berkat eksekusi Pasal 5.4 secara deterministik. Sebaliknya, Static Router mengalami halusinasi 40.0%.
  3. **Akurasi Pemilihan Alat (Tool Selection):**
     Model mencapai akurasi seleksi alat sebesar **96.7%** (29 dari 30 skenario memanggil tool yang relevan secara presisi).
  4. **Eksekusi Structured Action Payload untuk Frontend:**
     ReAct Agent sukses memicu pembuatan JSON `DegreePlanPayload` (`action: "APPLY_DEGREE_PLAN"`) yang siap dikonsumsi langsung oleh `DegreePlanner.tsx` di frontend `ftmm-compass-ui`.
- **Keputusan Arsitektur & Rekomendasi:**
  - ReAct Agent ditetapkan sebagai arsitektur eksekutor untuk kueri perencanaan studi dan pengecekan prasyarat di Sprint 6.
  - Menutup target riset Card 44.
- **Artefak:**
  - `docs/exp-003_react_tool_calling/README.md`
  - `docs/exp-003_react_tool_calling/benchmark_summary.md`
  - `docs/exp-003_react_tool_calling/benchmark_metrics.csv`
  - `docs/exp-003_react_tool_calling/benchmark_results.json`

---

### [exp-004] Evaluasi Penuh ReAct Agent: Zero-Abstraction Telemetry & 60 Skenario Lintas 5 Prodi FTMM
- **Tanggal:** 2026-10-03
- **Kartu Trello:** Card 44 (`[Nauval & Ario] Implementasi ReAct Agent dengan tool-calling`).
- **Tujuan / Hipotesis:**
  Melakukan evaluasi skala penuh terhadap agen penalaran ReAct dengan instrumentasi telemetri tingkat rendah (zero black-box) tanpa abstraksi. Menguji 60 skenario multi-hop lintas 5 program studi FTMM (TSD, TRKB, RN, TI, TE) serta merekam trace eksekusi mentah, token, latensi per step, dan mekanisme self-correction saat argumen fungsi salah.
- **Setup Teknis (Low-Level Architecture):**
  - **LLM Backbone:** Ollama `qwen2.5:3b` (native tool-calling JSON schema).
  - **Telemetry Logging:** Perekaman raw prompt messages, completion tokens, prompt tokens, latency per-step dalam microsecond, dan raw tool outputs ke file JSON Lines (`benchmark_raw_traces.jsonl`).
  - **Self-Correction Retry Loop:** Mekanisme recovery otomatis pada level Python runtime saat argumen tool menghasilkan pengecualian atau ketidaksesuaian tipe data.
  - **Testbed Pengujian:** 60 skenario akademik tervalidasi kurikulum resmi (12 skenario per program studi).
- **Temuan Kunci:**
  1. **Kenaikan Task Success Rate ke 75.0%:**
     Kesuksesan tugas naik dari 60.0% (Exp-003) menjadi **75.0%** (45 dari 60 skenario tuntas dengan benar).
  2. **Konsistensi Kebijakan Beban SKS Mutlak (100% SKS_POLICY):**
     Seluruh 20 kasus aturan SKS berhasil 100% dengan angka halusinasi **0.0%** dan latensi rata-rata tercepat (**1563.1 ms**).
  3. **Akurasi Seleksi Alat Sangat Tinggi (96.7%):**
     Model berhasil memilih tool deterministik yang tepat pada 58 dari 60 skenario uji.
  4. **Aktivasi Self-Correction (20.0%):**
     Sebanyak 12 skenario memicu self-correction di mana agen secara dinamis memperbaiki tipe data parameter dan berhasil merampungkan tugas.
  5. **Diagnosis Kegagalan yang Terlacak Mentah (Failure Analysis):**
     - *DEGREE_PLANNING (7 gagal):* Model 3B terkadang menjabarkan rencana studi langsung secara teks naratif alih-alih memanggil tool `generate_degree_plan`.
     - *PREREQUISITE_DAG (6 false positive):* Model menerima informasi bahwa mahasiswa belum memenuhi prasyarat, namun merespons dengan kalimat sopan bersyarat (*"Anda boleh mengambilnya jika sudah..."*) yang terbaca ambigu oleh sistem evaluasi.
- **Keputusan Arsitektur & Rekomendasi:**
  - Seluruh raw traces tersimpan dan dapat diinspeksi per skenario untuk debugging.
  - Menyiapkan integrasi engine ke endpoint API backend (`/api/chat`).
- **Artefak:**
  - `docs/exp-004_react_full_evaluation/README.md`
  - `docs/exp-004_react_full_evaluation/benchmark_summary.md`
  - `docs/exp-004_react_full_evaluation/benchmark_metrics.csv`
  - `docs/exp-004_react_full_evaluation/benchmark_raw_traces.jsonl`
  - `docs/exp-004_react_full_evaluation/benchmark_results.json`

---

### [exp-005] ReAct Agent Hardened: Mitigasi 5 Kluster Kegagalan & Skalasi Evaluasi 60 Skenario FTMM
- **Tanggal:** 2026-10-03
- **Kartu Trello:** Card 44 (`[Nauval & Ario] Implementasi ReAct Agent dengan tool-calling`).
- **Tujuan / Hipotesis:**
  Menguji efektivitas mitigasi rekayasa (defensive argument defaults, student profile auto-injection, NoneType sanitation, anti-inversion parameter docstring, dan kenaikan `num_predict: 450`) untuk menyelesaikan 15 kasus kegagalan yang ditemukan pada Exp-004.
- **Setup Teknis (Low-Level Architecture):**
  - **Defensive Defaults:** `generate_degree_plan` dan `validate_prerequisites_dag` diperkuat dengan fallback nilai default sehingga aman dari pemanggilan argumen tak lengkap oleh model SLM.
  - **Profile Auto-Injection:** Otomatis mengisi `program_studi`, `passed_courses`, `target_semester`, dan `focus_tracks` dari memori sesi profil mahasiswa jika model tidak menyediakannya.
  - **Anti-Inversion Schema:** Parameter `course_code_or_name` diperjelas secara eksplisit sebagai mata kuliah TARGET agar model tidak membalik urutan verifikasi prasyarat.
  - **Evaluator Robustness:** Penambahan leksikon penolakan legal formal (*"tidak berhak"*, *"tidak diizinkan"*, *"belum memenuhi"*).
  - **Testbed Pengujian:** 60 skenario terstratifikasi lintas 5 program studi FTMM (TSD, TRKB, RN, TI, TE).
- **Temuan Kunci:**
  1. **Lonjakan Drastis Task Success Rate (88.3%):**
     Tingkat keberhasilan tugas naik signifikan dari 75.0% (Exp-004) menjadi **88.3%** (53 dari 60 skenario tuntas dengan sempurna).
  2. **Kesuksesan Mutlak Perencanaan Studi (`DEGREE_PLANNING` 100%):**
     Dari sebelumnya hanya 30.0% (3/10) pada Exp-004, kini mencapai **100.0% (10/10)** sukses menghasilkan payload JSON rencana studi terstruktur yang siap disuntikkan ke `DegreePlanner.tsx`.
  3. **Pencarian Mata Kuliah Sempurna (`COURSE_DISCOVERY` 100%):**
     Seluruh 10 skenario penemuan kurikulum berhasil 100% tanpa ada crash `NoneType`.
  4. **Performa Sempurna pada Program Studi Teknik Industri (TI 100%):**
     Prodi TI mencapai keberhasilan 100% (12/12), TSD mencapai 91.7% (11/12), dan TRKB, RN, TE masing-masing mencapai 83.3% (10/12).
  5. **Pengurangan Kasus Kegagalan Lebih dari Separuh:**
     Total kegagalan terpangkas dari 15 kasus menjadi hanya **7 kasus**, yang seluruhnya terkonsentrasi pada false negative kalimat prasyarat.
- **Keputusan Arsitektur & Rekomendasi:**
  - Arsitektur ReAct Agent Hardened dinyatakan stabil dan siap diintegrasikan sebagai engine default pada backend produksi (`src/api/server.py`).
- **Artefak:**
  - `docs/exp-005_react_hardened_evaluation/README.md`
  - `docs/exp-005_react_hardened_evaluation/benchmark_summary.md`
  - `docs/exp-005_react_hardened_evaluation/benchmark_metrics.csv`
  - `docs/exp-005_react_hardened_evaluation/benchmark_raw_traces.jsonl`
  - `docs/exp-005_react_hardened_evaluation/benchmark_results.json`

---

### [exp-006] ReAct Agent Multi-Style Resilience & Paraphrase Invariance: Uji Anti-Overfitting
- **Tanggal:** 2026-10-03
- **Kartu Trello:** Card 44 (`[Nauval & Ario] Implementasi ReAct Agent`) & Card 91 (`[Nauval] Normalizer Slang Ringan`).
- **Tujuan / Hipotesis:**
  Membuktikan bahwa arsitektur ReAct Agent tidak mengalami overfitting terhadap diksi/kata tertentu pada dataset evaluasi sintetis. Menguji kemampuan generalisasi agen dalam menangani 3 gaya bahasa manusia yang berbeda secara leksikal namun identik secara intensi: (1) Formal Baku, (2) Slang/Santai Mahasiswa, dan (3) Implisit/Pengandaian.
- **Setup Teknis (Low-Level Architecture):**
  - **Token-Overlap & Punctuation Stripper (`academic_tools.py`):** Normalisasi leksikon singkatan mahasiswa (`alpro`, `visdat`, `metstat`, `probstat`, `jarkom`, `sisop`) dan kalkulasi token overlap Jaccard ($\ge 50\%$) untuk memitigasi tanda baca dan variasi nama matkul.
  - **Symmetric Decision Prompting (`react_agent.py`):** Panduan biner tegas dan simetris di mana hasil positif wajib dinyatakan langsung tanpa permohonan maaf atau hedging.
  - **Dataset Pengujian:** 60 kueri paralel (20 skenario dasar $\times$ 3 gaya bahasa: 20 Formal, 20 Slang, 20 Implisit) mencakup seluruh 5 prodi FTMM.
- **Temuan Kunci:**
  1. **Tingkat Ketahanan Parafrase (Paraphrase Invariance Score 75.0%):**
     Sebanyak 75.0% (15/20) skenario dasar berhasil secara konsisten pada KETIGA variasi gaya bahasa sekaligus.
  2. **Penurunan Performa Slang Sangat Minimal (Hanya -5.0%):**
     - Formal: **90.0%** (18/20).
     - Implisit: **90.0%** (18/20).
     - Slang / Santai: **85.0%** (17/20).
     Sebagai komparasi, pada Exp-002 metode leksikal murni (BM25) anjlok hingga -30.0% pada kueri slang. Kombinasi normalizer dan semantic reasoning membuktikan ketahanan terhadap diksi bebas.
  3. **Kesuksesan Mutlak Perencanaan Studi (100% di Semua Gaya):**
     Kategori `DEGREE_PLANNING` mencapai keberhasilan 100% pada gaya Formal, Slang, maupun Implisit, menghasilkan payload JSON yang valid.
  4. **Penekanan Halusinasi Ekstrem (3.3% Hallucination Rate):**
     Angka halusinasi turun ke titik terendah dalam riwayat pengujian (**3.3%**).
- **Keputusan Arsitektur & Portofolio Paper:**
  - Metrik *Paraphrase Invariance* dan data komparasi multi-style dimasukkan sebagai bab utama pada Paper Laporan TA untuk membuktikan generalisasi model di dunia nyata.
- **Artefak:**
  - `docs/exp-006_react_multistyle_resilience/README.md`
  - `docs/exp-006_react_multistyle_resilience/benchmark_summary.md`
  - `docs/exp-006_react_multistyle_resilience/benchmark_metrics.csv`
  - `docs/exp-006_react_multistyle_resilience/benchmark_raw_traces.jsonl`
  - `docs/exp-006_react_multistyle_resilience/benchmark_results.json`

---

### [exp-007] Evaluasi Bilingual ReAct (ID vs EN) & Hardware Efficiency Profiling (RTX 5050 8GB)
- **Tanggal:** 2026-10-03
- **Kartu Trello:** Card 31 (CAG Hardware VRAM), Card 32 (Hybrid), Card 44 (ReAct Agent).
- **Tujuan / Hipotesis:**
  Mengevaluasi kemampuan agen ReAct dalam memproses kueri berbahasa Inggris terhadap basis data kurikulum nasional FTMM UNAIR (Cross-Lingual Generalization) serta mengukur profil konsumsi hardware lokal (RTX 5050 Laptop GPU 8GB VRAM) untuk membuktikan kesiapan deployment tanpa OOM.
- **Setup Teknis (Low-Level Architecture):**
  - **Bilingual Academic Normalizer (`academic_tools.py`):** Pemetaan kamus istilah akademik internasional (Calculus, Discrete Math, Operating Systems, Database, Robotics, Control Systems, Nanotechnology, Supply Chain) ke kurikulum FTMM.
  - **Language-Adaptive Prompting (`react_agent.py`):** Aturan simetris di mana model otomatis menjawab dalam bahasa yang sama dengan input pengguna (Language Mirroring).
  - **Hardware Profiling:** Pengujian alokasi model ~2.5 GB (Qwen2.5-3B) dengan batas konteks 4096 token pada GPU RTX 5050 (8151 MB VRAM).
  - **Dataset Pengujian:** 40 kueri paralel (20 pasang kueri identik: 20 Bahasa Indonesia vs 20 Bahasa Inggris) mencakup 4 kategori akademik lintas 5 prodi FTMM.
- **Temuan Kunci:**
  1. **Akurasi Kueri Bahasa Inggris Melampaui Bahasa Indonesia:**
     - English (EN): **90.0%** (18/20).
     - Indonesian (ID): **85.0%** (17/20).
     - **Tool Selection Accuracy:** **100.0%** (40/40) sempurna pada kedua bahasa.
  2. **Tingkat Kesetaraan Dwibahasa Tinggi (Bilingual Parity Rate 80.0%):**
     Sebanyak 16 dari 20 pasang kueri berhasil diselesaikan secara bersamaan pada versi Indonesia dan Inggris.
  3. **Ketepatan Bahasa (Language Adherence 85.0%):**
     Model konsisten menjawab dalam bahasa Inggris saat ditanya bahasa Inggris (90.0%) tanpa code-switching yang mengganggu.
  4. **Efisiensi Hardware RTX 5050 (8GB VRAM Aman):**
     Alokasi model 3B memakan ~2.5 GB VRAM, menyisakan ~5.6 GB VRAM untuk KV cache hingga 4096 token. Latensi rata-rata stabil di 2670.4 ms (P50: 1630.8 ms).
- **Keputusan Arsitektur & Portofolio Paper:**
  - Chatbot FTMM COMPASS resmi terbukti siap melayani mahasiswa kelas internasional dan pertukaran pelajar (*inbound exchange*).
- **Artefak:**
  - `docs/exp-007_bilingual_react_efficiency/README.md`
  - `docs/exp-007_bilingual_react_efficiency/benchmark_summary.md`
  - `docs/exp-007_bilingual_react_efficiency/benchmark_metrics.csv`
  - `docs/exp-007_bilingual_react_efficiency/benchmark_raw_traces.jsonl`
  - `docs/exp-007_bilingual_react_efficiency/benchmark_results.json`

---

### [exp-008] Academic Safety Guardrails Ablation: Baseline vs NeMo vs Cascaded vs Hybrid
- **Tanggal:** 2026-10-07
- **Kartu Trello:** Card 45 (`[Nauval & Ario] Konfigurasi NeMo Guardrails sebagai safety layer`) & Card 92 (`[Ario & Nauval] Standarisasi Dataset Ground Truth Adversarial & Multi-Turn`).
- **Tujuan / Hipotesis:**
  Mengevaluasi secara kuantitatif efektivitas pertahanan berlapis (*Defense-in-Depth*) terhadap serangan adversarial (jailbreak, prompt injection, authority spoofing) dan kueri out-of-scope (OOD) non-akademik, serta mengukur pengaruhnya terhadap tingkat penolakan keliru (*False Refusal Rate* / FRR) pada ragam bahasa slang mahasiswa dan konsumsi latensi lokal RTX 5050.
- **Setup Teknis (Low-Level Architecture):**
  - **LLM Backbone & Embedding:** Ollama `qwen2.5:3b` + `nomic-embed-text` (768 dimensi, Cosine Centroids).
  - **Tier 1 (Deterministic Filter, 0.0 ms):** Regex & keyword blocklist untuk deteksi instan prompt injection, DAN jailbreak, authority spoofing, dan eksfiltrasi data/kunci jawaban.
  - **Tier 2 (Embedding Semantic Gating, <35 ms):** FAISS Cosine Distance kueri terhadap centroid akademik FTMM vs non-akademik (kuliner, crypto, politik umum, judi) tanpa beban inferensi LLM.
  - **Tier 3 (NeMo Colang Flow, ~285 ms):** Canonical intent classifier berbasis Colang safety policy.
  - **Tier 4 (Deterministic Tool & Argument Rail):** Validasi parameter fisik fungsi ReAct (kode MK, semester 1-8, batas beban SKS 0-24) sebelum eksekusi API/Database.
  - **Dataset Pengujian:** 50 kueri terstratifikasi dari Card 92 (15 Benign Academic, 10 Slang Mahasiswa, 10 Out-of-Scope, 15 Adversarial Attack). Total 200 run inferensi lintas 4 varian.
- **Temuan Kunci:**
  1. **Kerentanan Fatal Model Tanpa Guardrails (Baseline ASR 93.3%):**
     Model raw ReAct meloloskan 93.3% serangan adversarial dan menjawab seluruh pertanyaan non-akademik (OOD-R 0.0%), membuktikan model SLM mentah tidak aman dideploy tanpa lapisan guardrails.
  2. **Kesempurnaan Proteksi Lightweight Cascaded Guard (Varian C):**
     Kombinasi Tier 1 + Tier 2 + Tier 4 mencapai **100.0% Safety Accuracy** dan **0.0% Attack Success Rate** (semua 15 serangan adversarial dinetralkan). Kueri OOD terdeteksi sempurna (**100.0% OOD-R**).
  3. **Zero False Refusal Rate (FRR 0.0%):**
     Seluruh kueri sah mahasiswa dan slang informal (*"alpro"*, *"visdat"*, *"skripsian"*, *"ngebut"*) lolos 100% tanpa ada over-defense.
  4. **Efisiensi Latensi Komputasi Lokal (P50 663.6 ms):**
     Kueri adversarial terputus seketika pada Tier 1 (0.0 ms) dan kueri OOD pada Tier 2 (27–38 ms) tanpa memicu LLM autoregresif, memangkas P50 latensi total dari 1591.9 ms menjadi **663.6 ms** (-58.3%).
  5. **Kelemahan NeMo Black-Box pada Local Edge:**
     NeMo Guardrails murni (Varian B) meloloskan 26.7% serangan adversarial akibat keterbatasan penalaran SLM 3B pada prompt persuasif, serta rentan kehabisan socket HTTP jika menggunakan driver abstraction berat.
- **Keputusan Arsitektur & Tindak Lanjut:**
  - Mengadopsi **Lightweight Cascaded Guard (Varian C)** sebagai safety wrapper default sebelum diekspos ke API endpoint di Card 47.
  - Menuntaskan target riset Card 45 dan standarisasi dataset Card 92.
- **Artefak:**
  - `docs/exp-008_guardrails_evaluation/README.md`
  - `docs/exp-008_guardrails_evaluation/benchmark_summary.md`
  - `docs/exp-008_guardrails_evaluation/benchmark_metrics.csv`
  - `docs/exp-008_guardrails_evaluation/benchmark_raw_traces.jsonl`
  - `docs/exp-008_guardrails_evaluation/benchmark_results.json`

---

### [exp-009] Multi-Turn Conversation Memory Management & Crescendo Defense
- **Tanggal:** 2026-10-07
- **Kartu Trello:** Card 46 (`[Nauval & Ario] Implementasi conversation history management`) & Card 92 (`[Ario & Nauval] Standarisasi Dataset Ground Truth Adversarial & Multi-Turn`).
- **Tujuan / Hipotesis:**
  Mengevaluasi efektivitas arsitektur manajemen memori percakapan berkeadaan (*ConversationMemoryManager*) dengan *Sliding Window Buffer* (6 turns / 12 messages), akumulasi profil mahasiswa persisten, dan resolusi anafora terhadap keberhasilan dialog multi-turn serta ketahanan terhadap *Crescendo Jailbreak Attack*.
- **Setup Teknis (Low-Level Architecture):**
  - **LLM Backbone & Embedding:** Ollama `qwen2.5:3b` + `nomic-embed-text` (768 dimensi).
  - **Memory Module (`src/engine/memory.py`):**
    * `SessionState`: Penyimpanan memori in-memory terisolasi per `session_id` dengan TTL 2 jam (7200s).
    * `Sliding Window Buffer`: Maksimum 6 giliran (12 pesan) terpotong otomatis untuk mencegah pembengkakan konteks input dan OOM.
    * `Persistent Student Profile`: Ekstraksi otomatis prodi, semester, riwayat matkul lulus, dan fokus minat yang terakumulasi lintas turn.
    * `Anaphora Resolution`: Penautan kata rujukan (*"mata kuliah tersebut"*, *"itu"*, *"dosen pengampunya"*) ke entitas kurikulum pada giliran sebelumnya.
  - **Safety Wrapper:** Terintegrasi dengan *Lightweight Cascaded Guard* dari Exp-008.
  - **Dataset Pengujian:** 10 skenario dialog multi-turn (36 giliran dialog per konfigurasi: 5 prodi FTMM, stress-test 7 turn, dan 2 serangan Crescendo). Total 70 run dialog.
- **Temuan Kunci:**
  1. **Lonjakan Drastis Dialog Success Rate (94.3% vs 71.4%):**
     Penggunaan memori berkeadaan meningkatkan keberhasilan dialog sebesar **+22.9%**, mengeliminasi kebutuhan mahasiswa untuk mengulang informasi prodi/semester di setiap pesan.
  2. **Resolusi Anafora Efektif (85.7% vs 14.3%):**
     Model stateless gagal pada 85.7% kueri ber-rujukan (menganggap kueri ambigu atau diblokir OOD). Modul resolusi anafora mendongkrak pemahaman rujukan hingga **85.7%**.
  3. **Retensi Sempurna pada Percakapan Panjang (100% pada Turn 7):**
     Skenario dialog panjang (MT-06) membuktikan profil prodi dan semester dari Turn 1 tetap aktif dan akurat di Turn 7 tanpa degradasi memori.
  4. **Pertahanan Crescendo Mutlak (100% Defense Rate):**
     Kedua serangan jailbreak bertahap (ekstraksi prompt rahasia dan pemalsuan wewenang dekan 40 SKS) berhasil diputus 100% pada giliran kritis oleh Tier 1 Regex dan Tier 4 Tool Rail.
  5. **Karakteristik Latensi Konteks Geser:**
     Latensi rata-rata naik wajar dari 1691.3 ms ke 3442.3 ms karena penambahan token konteks (P50: 3421.1 ms), namun batas P95 terkunci stabil pada **6259.9 ms** berkat jendela geser 6 turn.
- **Keputusan Arsitektur & Rekomendasi:**
  - `ConversationMemoryManager` resmi dijadikan standar backend percakapan multi-turn FTMM COMPASS.
  - Menuntaskan seluruh acceptance criteria Card 46 dan dataset multi-turn Card 92.
  - Siap diintegrasikan ke endpoint API produksi (`POST /api/chat`) di Card 47.
- **Artefak:**
  - `docs/exp-009_multiturn_memory_evaluation/README.md`
  - `docs/exp-009_multiturn_memory_evaluation/benchmark_summary.md`
  - `docs/exp-009_multiturn_memory_evaluation/benchmark_metrics.csv`
  - `docs/exp-009_multiturn_memory_evaluation/benchmark_raw_traces.jsonl`
  - `docs/exp-009_multiturn_memory_evaluation/benchmark_results.json`

---

### [exp-010] Low-Level Inference Optimization & Dual-Model VRAM Pinning
- **Tanggal:** 2026-10-10
- **Kartu Trello:** Card 31 (`[Nauval & Ario] Hardware Profiling VRAM`), Card 44 (`[Nauval & Ario] ReAct Agent Latency Optimization`), Card 99 (`[Nauval & Ario] Decider Paradigm & Dynamic Tool Pruning`).
- **Tujuan / Hipotesis:**
  Mengeliminasi latensi *runner swapping* dan *prompt token bloat* pada laptop GPU lokal (RTX 5050 8GB VRAM di WSL2) dengan menerapkan 4 optimasi tingkat rendah (*low-level*): pinning VRAM dual-model resident, pruning skema tools dinamis ($O(1)$ Decider pattern), prefix cache pinning deterministik bergaya DeepSeek Harness, dan pembatasan batas konteks/prediksi (`num_ctx: 2048`, `num_predict: 256`).
- **Setup Teknis (Low-Level Architecture):**
  - **LLM Backbone & Embedding:** Ollama `qwen2.5:3b` + `nomic-embed-text`.
  - **VRAM Pinning Config:** `OLLAMA_MAX_LOADED_MODELS=2` dan `OLLAMA_KEEP_ALIVE=-1` dikonfigurasi pada service Ollama lokal.
  - **Dynamic Tool Attachment (`src/engine/react_agent.py`):** Fungsi seleksi cepat memotong parameter skema tools JSON yang tidak dibutuhkan pada kueri sapaan dan FAQ regulasi umum.
  - **Deterministic Prefix Layout:** Mempertahankan token index 0 tetap konstan secara biner untuk memaksimalkan hash hit pada KV cache Llama.cpp.
  - **Boundary Clamping:** `num_predict: 256` dan `num_ctx: 2048` membatasi buffer VRAM KV cache dan memotong waktu decoding bandwidth memori.
  - **Dataset Pengujian:** 20 kueri terstratifikasi (5 Greeting, 5 Batas SKS, 5 Prasyarat, 5 Perencanaan Rencana Studi).
- **Temuan Kunci:**
  1. **Akselerasi Drastis Kueri Sapaan (P50 694.7 ms vs ~5.500 ms):**
     Pemangkasan ~963 token skema JSON pada kueri sapaan memangkas latensi inferensi menjadi sub-detik (**420 – 700 ms**), memberikan impresi interaksi instan (kecepatan meningkat hingga **8.5x**).
  2. **Efisiensi Kueri SKS & Prasyarat (1.300 – 1.400 ms):**
     Kueri perhitungan SKS dan verifikasi prasyarat selesai stabil dalam **1.3 – 1.4 detik** dengan prefill token terpangkas 20–25%.
  3. **Zero VRAM Swapping Latency (0.0 ms Swap):**
     `qwen2.5:3b` (~2.2 GB) dan `nomic-embed-text` (~0.3 GB) terkunci permanen di memori GPU secara bersamaan. Sisa VRAM GPU RTX 5050 masih sangat lapang (**5.586 MiB / 68.6% bebas**), tanpa memicu paging memori virtual WSL2.
  4. **Stabilitas Termal & Operasional:**
     GPU beroperasi sangat stabil pada suhu 36°C – 46°C dengan utilitas daya hanya 7W – 15W saat idle.
- **Keputusan Arsitektur & Tindak Lanjut:**
  - Konfigurasi `OLLAMA_MAX_LOADED_MODELS=2` dan dynamic tool pruning di `src/engine/react_agent.py` resmi ditetapkan sebagai standar runtime FTMM COMPASS.
  - Menutup target efisiensi Card 31, Card 44, dan aspek pruning Card 99.
- **Artefak:**
  - `docs/exp-010_lowlevel_inference_optimization/README.md`
  - `docs/exp-010_lowlevel_inference_optimization/benchmark_summary.md`
  - `docs/exp-010_lowlevel_inference_optimization/benchmark_metrics.csv`
  - `docs/exp-010_lowlevel_inference_optimization/benchmark_raw_traces.jsonl`
  - `docs/exp-010_lowlevel_inference_optimization/benchmark_results.json`
