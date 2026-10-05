# Laporan Evaluasi ReAct Agent Hardened (Exp-005)

> **Tanggal Pengujian:** 2026-10-03
> **Topik:** Evaluasi ReAct Agent Hardened (Mitigasi 5 Kluster Kesalahan Exp-004)
> **Model:** `qwen2.5:3b` via Ollama Native Tool-Calling
> **Total Skenario:** 60 skenario terstratifikasi lintas 5 program studi FTMM (TSD, TRKB, RN, TI, TE)

---

## 1. Ringkasan Metrik Kunci (Zero-Abstraction Telemetry)

| Metrik Evaluasi | Exp-005 (Hardened) | Exp-004 (Sebelum Perbaikan) | Peningkatan |
|---|---|---|---|
| **Overall Task Success Rate** | **88.3%** | 75.0% | **+13.3%** |
| **Tool Selection Accuracy** | **95.0%** | 96.7% | **+-1.7%** |
| **Hallucination Rate** | **10.0%** | 10.0% | **0.0%** |
| **Self-Correction Triggered** | **0.0%** | 20.0% | -20.0% |
| **Rata-rata Reasoning Steps** | **2.0 steps** | 2.1 steps | - |
| **Rata-rata Latensi Total** | **2934.0 ms** | 2703.7 ms | P50: 1951.0 ms, P95: 6609.7 ms |
| **Rata-rata Token Prompt** | **2759.9 tokens** | 2533.6 tokens | - |
| **Rata-rata Token Completion** | **219.1 tokens** | 195.5 tokens | - |

---

## 2. Performa Lintas 5 Program Studi FTMM

| Program Studi | Jumlah Skenario | Success Rate | Tool Selection Acc | Avg Latensi (ms) |
|---|---|---|---|---|
| **RN** | 12 | **83.3%** | 100.0% | 2857.2 ms |
| **TE** | 12 | **83.3%** | 91.7% | 2923.6 ms |
| **TI** | 12 | **100.0%** | 100.0% | 2513.5 ms |
| **TRKB** | 12 | **83.3%** | 83.3% | 2967.1 ms |
| **TSD** | 12 | **91.7%** | 100.0% | 3408.8 ms |

---

## 3. Performa Berdasarkan Kategori Masalah Akademik

| Kategori Permasalahan | Jumlah Skenario | Success Rate | Hallucination Rate | Avg Latensi (ms) |
|---|---|---|---|---|
| **COURSE_DISCOVERY** | 10 | **100.0%** | 0.0% | 3734.5 ms |
| **DEGREE_PLANNING** | 10 | **100.0%** | 0.0% | 6400.9 ms |
| **PREREQUISITE_DAG** | 20 | **70.0%** | 25.0% | 2308.7 ms |
| **SKS_POLICY** | 20 | **95.0%** | 5.0% | 1425.8 ms |

---

## 4. Diagnosis & Failure Analysis Pasca Hardening

Ditemukan **7 kasus kegagalan** dari 60 skenario (turun dari 15 kasus pada Exp-004):

| Sample ID | Prodi | Kategori | Status | Alasan Kegagalan / Diagnosis |
|---|---|---|---|---|
| `TSD-04` | TSD | PREREQUISITE_DAG | HALLUCINATED | False Negative: Seharusnya eligible/boleh tapi model menolak |
| `TRKB-02` | TRKB | PREREQUISITE_DAG | HALLUCINATED | False Negative: Seharusnya eligible/boleh tapi model menolak |
| `TRKB-06` | TRKB | SKS_POLICY | HALLUCINATED | Aturan SKS meleset: Seharusnya 20 SKS |
| `RN-02` | RN | PREREQUISITE_DAG | HALLUCINATED | False Negative: Seharusnya eligible/boleh tapi model menolak |
| `RN-03` | RN | PREREQUISITE_DAG | HALLUCINATED | False Negative: Seharusnya eligible/boleh tapi model menolak |
| `TE-02` | TE | PREREQUISITE_DAG | HALLUCINATED | False Negative: Seharusnya eligible/boleh tapi model menolak |
| `TE-03` | TE | PREREQUISITE_DAG | FAILED | Jawaban tidak tegas menyatakan status kelayakan prasyarat |

---

## 5. Berkas Artefak Eksekusi (Dapat Diinspeksi Mentah)
- `benchmark_raw_traces.jsonl`: Setiap baris berisi full dump pesan input, response model, pemanggilan tool, dan latensi per step.
- `benchmark_metrics.csv`: Data mentah 60 baris tabel telemetri.
- `benchmark_results.json`: Nilai agregat metrik.