# Laporan Evaluasi Bilingual ReAct & Hardware Efficiency (Exp-007)

> **Tanggal Pengujian:** 2026-10-03
> **Topik:** Evaluasi Kueri Dwibahasa (ID vs EN) & Efisiensi Hardware RTX 5050 (Card 31, 32, 44)
> **Model:** `qwen2.5:3b` via Ollama Native Tool-Calling
> **Hardware:** NVIDIA GeForce RTX 5050 Laptop GPU (8GB VRAM)
> **Total Kueri:** 40 kueri paralel (20 Bahasa Indonesia vs 20 Bahasa Inggris)

---

## 1. Analisis Komparasi Kinerja Dwibahasa (Indonesian vs English)

**Bilingual Parity Rate:** **80.0%**  
*(Persentase pasangan kueri di mana model sukses pada KEDUA versi bahasa sekaligus)*

| Bahasa | Jumlah Kueri | Success Rate | Tool Selection Acc | Language Adherence | Avg Latensi (ms) |
|---|---|---|---|---|---|
| **Bahasa Indonesia (ID)** | 20 | **85.0%** | 100.0% | **80.0%** | 2824.5 ms |
| **English (EN)** | 20 | **90.0%** | 100.0% | **90.0%** | 2516.3 ms |

---

## 2. Kinerja Lintas Kategori Masalah Akademik

| Kategori Permasalahan | Jumlah Kueri | Success Rate | Hallucination Rate | Avg Latensi (ms) |
|---|---|---|---|---|
| **COURSE_DISCOVERY** | 4 | **100.0%** | 0.0% | 3211.6 ms |
| **DEGREE_PLANNING** | 8 | **100.0%** | 0.0% | 6407.9 ms |
| **PREREQUISITE_DAG** | 16 | **68.8%** | 31.2% | 1517.4 ms |
| **SKS_POLICY** | 12 | **100.0%** | 0.0% | 1535.7 ms |

---

## 3. Efisiensi Hardware & Profil Komputasi (RTX 5050 8GB VRAM)

| Parameter Hardware | Nilai Terukur | Evaluasi Batasan Trello Card 31 |
|---|---|---|
| **Alokasi VRAM Model 3B** | ~2.5 GB | Sangat aman (<4B model menyisakan ~5.5 GB VRAM) |
| **Toleransi KV Cache** | 4096 token | Nol OOM spillover ke sistem RAM 16GB |
| **Rata-rata Latensi Total** | **2670.4 ms** | P50: 1630.8 ms, P95: 6784.5 ms |
| **Rata-rata Token Prompt** | **3161.1 tokens** | - |
| **Rata-rata Token Completion** | **184.3 tokens** | Kecepatan inferensi ~25–35 token/detik |

---

## 4. Evaluasi Kasus Gagal (Failure Inspection)

Ditemukan **5 kasus kegagalan** dari 40 pengujian:

| Sample ID | Pair ID | Bahasa | Kategori | Alasan Kegagalan |
|---|---|---|---|---|
| `PAIR-03-ID` | `PAIR-03` | ID | PREREQUISITE_DAG | False Negative: Expected eligible but rejected |
| `PAIR-04-ID` | `PAIR-04` | ID | PREREQUISITE_DAG | False Positive: Missing prerequisites but accepted |
| `PAIR-04-EN` | `PAIR-04` | EN | PREREQUISITE_DAG | False Positive: Missing prerequisites but accepted |
| `PAIR-06-EN` | `PAIR-06` | EN | PREREQUISITE_DAG | False Negative: Expected eligible but rejected |
| `PAIR-08-ID` | `PAIR-08` | ID | PREREQUISITE_DAG | False Positive: Missing prerequisites but accepted |