# Eksperimen 003: ReAct Agent (Tool-Calling) vs Static 1-Shot Router

- **Eksperimen ID:** `exp-003`
- **Tanggal:** 2026-10-03
- **Kartu Trello Terkait:** Card 44 (`[Nauval & Ario] Implementasi ReAct Agent dengan tool-calling`)
- **Model:** Ollama `qwen2.5:3b` pada GPU lokal NVIDIA RTX 5050 (8GB VRAM)
- **Tujuan Riset:** Menguji efektivitas reasoning loop ReAct (Reasoning + Acting) yang dipandu 4 tools akademik deterministik vs Static 1-Shot Router untuk menyelesaikan permasalahan multi-hop dan perencanaan studi akademik FTMM.

---

## 1. Arsitektur Tools Deterministik (Zero Black-Box)
1. `search_courses`: Pencarian silabus & mata kuliah berbasis Hybrid RRF (BM25 + FAISS Cosine).
2. `validate_prerequisites_dag`: Verifikasi rantai prasyarat kurikulum berbasis graf ketergantungan (DAG) terstruktur.
3. `calculate_sks_limit`: Penghitungan batas beban studi SKS semester sesuai Pasal 5.4 Panduan Akademik FTMM.
4. `generate_degree_plan`: Penyusunan roadmap semester terstruktur yang kompatibel 100% dengan `DegreePlanner.tsx`.

---

## 2. Tabel Ringkasan Temuan Utama (30 Skenario Pengujian)

| Arsitektur | Task Success Rate | Fact Recall | Hallucination Rate | Tool Selection Acc | Avg Steps | Avg Latensi (ms) |
|---|---|---|---|---|---|---|
| **ReAct Agent (Tool-Calling)** | **60.0%** | **61.1%** | **13.3%** | **96.7%** | 2.0 | **2219.1 ms** |
| **Static 1-Shot Router** | 40.0% | 46.1% | 26.7% | 0.0% | 1.0 | 2933.0 ms |

### Breakdown per Kategori:
- **Batasan Beban SKS (`SKS_LIMIT_CONSTRAINT`):**
  - ReAct Agent mencatat **100% Success Rate** dan **0.0% Hallucination**, sedangkan Static Router mengalami halusinasi sebesar 40.0% akibat menebak angka di luar aturan resmi.
- **Validasi Prasyarat (`PREREQUISITE_CHECK`):**
  - ReAct Agent berhasil memverifikasi rantai prasyarat via `validate_prerequisites_dag` dengan akurasi seleksi alat 100%.
- **Eksekusi Rencana Studi (`STUDY_PLANNING_ACTION`):**
  - ReAct Agent berhasil menghasilkan payload terstruktur `plan_payload` (JSON) yang siap dikonsumsi langsung oleh antarmuka visual React `DegreePlanner.tsx`. Static Router hanya menghasilkan teks naratif tanpa kemampuan transfer state ke UI.

---

## 3. Implikasi untuk Paper Tugas Akhir
1. Pembuktian bahwa *deterministic tool-calling* merupakan kebutuhan mutlak untuk domain akademik tinggi yang tidak menoleransi halusinasi regulasi.
2. Latensi rata-rata ReAct Agent (2.2 detik) masih berada dalam batas interaksi obrolan nyaman (<3 detik).

---

## 4. Berkas Artefak
- `benchmark_summary.md`: Laporan komparasi lengkap dan analisis breakdown per kategori.
- `benchmark_metrics.csv`: Data mentah 60 run inferensi (30 skenario x 2 arsitektur).
- `benchmark_results.json`: Rekapitulasi metrik statistik agregat.
