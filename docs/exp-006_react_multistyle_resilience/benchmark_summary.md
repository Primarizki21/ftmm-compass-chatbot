# Laporan Evaluasi Multi-Style Resilience & Anti-Overfitting (Exp-006)

> **Tanggal Pengujian:** 2026-10-03
> **Topik:** Uji Invariansi Parafrase & Ketahanan Diksi Santai/Slang (Card 44 & Card 91 Trello)
> **Model:** `qwen2.5:3b` via Ollama Native Tool-Calling
> **Total Kueri:** 60 kueri berpasangan (20 Formal, 20 Slang, 20 Implisit)

---

## 1. Analisis Invariansi Gaya Bahasa (Generalization vs Overfitting)

**Paraphrase Invariance Score:** **75.0%**  
*(Persentase skenario dasar di mana model berhasil konsisten pada KETIGA variasi gaya: Formal, Slang, dan Implisit)*

| Gaya Bahasa Kueri | Jumlah Kueri | Success Rate | Tool Selection Acc | Hallucination Rate | Avg Latensi (ms) |
|---|---|---|---|---|---|
| **FORMAL** | 20 | **90.0%** | 85.0% | 0.0% | 2489.6 ms |
| **SLANG** | 20 | **85.0%** | 70.0% | 10.0% | 2343.8 ms |
| **IMPLICIT** | 20 | **90.0%** | 95.0% | 0.0% | 2387.8 ms |

---

## 2. Performa Lintas Kategori Masalah Akademik

| Kategori Permasalahan | Jumlah Kueri | Success Rate | Hallucination Rate | Avg Latensi (ms) |
|---|---|---|---|---|
| **DEGREE_PLANNING** | 15 | **100.0%** | 0.0% | 5674.8 ms |
| **PREREQUISITE_DAG** | 30 | **80.0%** | 3.3% | 1306.4 ms |
| **SKS_POLICY** | 15 | **93.3%** | 6.7% | 1340.7 ms |

---

## 3. Performa Lintas 5 Program Studi FTMM

| Program Studi | Jumlah Kueri | Success Rate | Tool Selection Acc | Avg Latensi (ms) |
|---|---|---|---|---|
| **RN** | 12 | **83.3%** | 66.7% | 2497.1 ms |
| **TE** | 12 | **100.0%** | 91.7% | 2591.0 ms |
| **TI** | 12 | **75.0%** | 66.7% | 2200.0 ms |
| **TRKB** | 12 | **91.7%** | 100.0% | 2500.3 ms |
| **TSD** | 12 | **91.7%** | 91.7% | 2247.1 ms |

---

## 4. Evaluasi Kasus Gagal (Failure Inspection)

Ditemukan **7 kasus kegagalan** dari 60 pengujian:

| Sample ID | Base ID | Gaya | Prodi | Kategori | Alasan Kegagalan |
|---|---|---|---|---|---|
| `TSD-PREREQ-NEG-FORMAL` | `TSD-PREREQ-NEG` | formal | TSD | PREREQUISITE_DAG | Jawaban tidak tegas menyatakan status kelayakan prasyarat |
| `TRKB-PREREQ-NEG-IMPLICIT` | `TRKB-PREREQ-NEG` | implicit | TRKB | PREREQUISITE_DAG | Jawaban tidak tegas menyatakan status kelayakan prasyarat |
| `RN-PREREQ-POS-SLANG` | `RN-PREREQ-POS` | slang | RN | PREREQUISITE_DAG | False Negative: Seharusnya eligible/boleh tapi model menolak |
| `RN-SKS-SLANG` | `RN-SKS` | slang | RN | SKS_POLICY | Aturan SKS meleset: Seharusnya 20 SKS |
| `TI-PREREQ-NEG-FORMAL` | `TI-PREREQ-NEG` | formal | TI | PREREQUISITE_DAG | Jawaban tidak tegas menyatakan status kelayakan prasyarat |
| `TI-PREREQ-NEG-SLANG` | `TI-PREREQ-NEG` | slang | TI | PREREQUISITE_DAG | Jawaban tidak tegas menyatakan status kelayakan prasyarat |
| `TI-PREREQ-NEG-IMPLICIT` | `TI-PREREQ-NEG` | implicit | TI | PREREQUISITE_DAG | Jawaban tidak tegas menyatakan status kelayakan prasyarat |