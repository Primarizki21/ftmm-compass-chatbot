# Eksperimen 011: System-1 Decision Classifier & Semantic Tool Schema Pruning

- **Eksperimen ID:** `exp-011`
- **Tanggal:** 2026-10-10
- **Kartu Trello:** Card 99 (`[Nauval & Ario] Implementasi System-1 Decision Classifier (Jev/Decider Paradigm) untuk Intent Routing & Tool Dispatching`)
- **Model:** Embedding `nomic-embed-text` (768 dimensi, Cosine Centroid BLAS Matrix Dot-Product) pada GPU lokal NVIDIA RTX 5050 (8GB VRAM)
- **Tujuan:** Mengimplementasikan dan mengevaluasi paradigma System-1 Decision Classifier (Jev/Decider) berbasis representasi embedding non-autoregresif untuk memangkas latensi pemilihan tools dan mereduksi token skema JSON sebelum menyentuh generator.

---

## 1. Hasil Kuantitatif Benchmark (60 Kueri Terstratifikasi)

| Dimensi Evaluasi | Segmen | Jumlah Kueri | Akurasi (%) [↑] | Latensi Rata-rata (ms) [↓] | P50 (ms) [↓] |
|---|---|---|---|---|---|
| **Gaya Bahasa** | **Formal Baku** | 20 | **95.0%** | 34.9 ms | 34.1 ms |
| **Gaya Bahasa** | **Slang Mahasiswa** | 20 | **95.0%** | 33.3 ms | 31.3 ms |
| **Gaya Bahasa** | **Implisit / Ambigu** | 20 | **100.0%** | 33.8 ms | 34.3 ms |
| **Kategori Tugas**| `CHAT_ONLY` (0 tools) | 12 | **100.0%** | 37.6 ms | 37.5 ms |
| **Kategori Tugas**| `PREREQUISITE` (2 tools) | 12 | **91.7%** | 32.8 ms | 33.5 ms |
| **Kategori Tugas**| `SKS_CALCULATION` (1 tool)| 12 | **100.0%** | 32.1 ms | 30.8 ms |
| **Kategori Tugas**| `DEGREE_PLANNING` (2 tools)| 12 | **100.0%** | 34.4 ms | 33.4 ms |
| **Kategori Tugas**| `COURSE_DISCOVERY` (1 tool)| 12 | **91.7%** | 30.9 ms | 31.0 ms |
| **OVERALL SUMMARY**| **Semua Kueri** | **60** | **96.7% (58/60)** | **34.0 ms** | **33.2 ms** |

---

## 2. Temuan Kunci Ilmiah

1. **Kecepatan Ekstrem Sub-35ms (95% Lebih Cepat dari LLM Generatif):**
   - Pemilihan tools berlangsung dalam **34.0 ms** (waktu embedding 33.9 ms + waktu dot product matriks BLAS 0.05 ms).
   - Menghilangkan latensi 300–800 ms yang biasanya dihabiskan oleh LLM autoregresif hanya untuk mengeluarkan nama fungsi JSON.

2. **Penghematan Token Prefill Signifikan (~675 Token per Kueri):**
   - Kueri sapaan/chitchat dieksekusi dengan **0 tools** (menghemat 963 token definisi skema JSON).
   - Kueri SKS hanya memuat 1 tool (menghemat ~770 token).
   - Matriks self-attention transformer pada layer ReAct Agent tereduksi secara drastis, memangkas latensi prefill di RTX 5050.

3. **Ketahanan terhadap Diksi Implisit & Slang (100% Implisit):**
   - Kalimat implisit tanpa kata kunci formal (*"Kira-kira saya bakal terganjal matkul lama gak ya?"*) terpetakan 100% akurat ke kategori `PREREQUISITE`.
   - Mengatasi kelemahan pencocokan berbasis regex kaku.

4. **Zero VRAM Footprint Overhead:**
   - Memanfaatkan model embedding `nomic-embed-text` yang sudah resident di GPU VRAM (323 MB), tanpa perlu memuat model decider terpisah berukuran 1–2 GB.

---

## 3. Keputusan Arsitektur & Tindak Lanjut

- `System1Decider` resmi diintegrasikan ke dalam `src/engine/react_agent.py` sebagai lapisan awal pemangkasan skema tools sebelum LLM generator dipanggil.
- Menuntaskan seluruh acceptance criteria pada **Trello Card 99** (Decider Paradigm).
- Seluruh pipeline inferensi (`System1Decider` + `CascadedGuard` + `MemoryManager` + `ReActAgent`) resmi tuntas dan siap diexpose ke API FastAPI di **Card 47**.

---

## 4. Berkas Artefak Eksperimen

1. `benchmark_raw_traces.jsonl`: 60 rekam jejak klasifikasi instan (kueri, gaya bahasa, skor dot product, tools terpilih, latensi).
2. `benchmark_metrics.csv`: Data matriks evaluasi per dimensi dan segmen.
3. `benchmark_summary.md`: Tabel ringkasan komparasi performa decider.
4. `benchmark_results.json`: Rekapitulasi metrik JSON mesin.
