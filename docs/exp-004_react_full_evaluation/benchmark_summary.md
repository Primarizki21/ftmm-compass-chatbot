# Laporan Evaluasi Penuh ReAct Agent (Exp-004)

> **Tanggal Pengujian:** 2026-10-03
> **Topik:** Evaluasi Menyeluruh ReAct Tool-Calling Lintas 5 Program Studi FTMM (Card 44 Trello)
> **Model:** `qwen2.5:3b` via Ollama Native Tool-Calling
> **Total Skenario:** 60 skenario terstratifikasi (TSD, TRKB, RN, TI, TE)

---

## 1. Ringkasan Metrik Kunci (Zero-Abstraction Telemetry)

| Metrik Evaluasi | Nilai Capaian | Keterangan & Target |
|---|---|---|
| **Overall Task Success Rate** | **75.0%** | Proporsi skenario yang terselesaikan dengan benar |
| **Tool Selection Accuracy** | **96.7%** | Ketepatan agen memilih tool deterministik |
| **Hallucination Rate** | **10.0%** | Rasio pelanggaran regulasi atau jawaban keliru |
| **Self-Correction Triggered** | **20.0%** | Rasio agen memicu recovery saat argumen error |
| **Rata-rata Reasoning Steps** | **2.1 steps** | Turn interaksi (LLM -> Tool -> LLM) |
| **Rata-rata Latensi Total** | **2703.7 ms** | P50: 2002.7 ms, P95: 6807.5 ms |
| **Rata-rata Token Prompt** | **2533.6 tokens** | Akumulasi prompt input per percakapan |
| **Rata-rata Token Completion** | **195.5 tokens** | Akumulasi token inferensi model |

---

## 2. Performa Lintas 5 Program Studi FTMM

| Program Studi | Jumlah Skenario | Success Rate | Tool Selection Acc | Avg Latensi (ms) |
|---|---|---|---|---|
| **RN** | 12 | **75.0%** | 100.0% | 2700.4 ms |
| **TE** | 12 | **75.0%** | 100.0% | 2680.7 ms |
| **TI** | 12 | **83.3%** | 91.7% | 2659.1 ms |
| **TRKB** | 12 | **66.7%** | 100.0% | 3014.1 ms |
| **TSD** | 12 | **75.0%** | 91.7% | 2464.2 ms |

---

## 3. Performa Berdasarkan Kategori Masalah Akademik

| Kategori Permasalahan | Jumlah Skenario | Success Rate | Hallucination Rate | Avg Latensi (ms) |
|---|---|---|---|---|
| **COURSE_DISCOVERY** | 10 | **80.0%** | 0.0% | 4168.0 ms |
| **DEGREE_PLANNING** | 10 | **30.0%** | 0.0% | 4416.5 ms |
| **PREREQUISITE_DAG** | 20 | **70.0%** | 30.0% | 2255.9 ms |
| **SKS_POLICY** | 20 | **100.0%** | 0.0% | 1563.1 ms |

---

## 4. Diagnosis & Failure Analysis (Low-Level Error Inspection)

Ditemukan **15 kasus kegagalan** dari 60 skenario:

| Sample ID | Prodi | Kategori | Status | Alasan Kegagalan / Diagnosis |
|---|---|---|---|---|
| `TSD-09` | TSD | COURSE_DISCOVERY | FAILED | Matkul yang dicari tidak muncul di jawaban |
| `TSD-11` | TSD | DEGREE_PLANNING | FAILED | Gagal memicu action PLAN_GENERATED atau plan_payload kosong |
| `TSD-12` | TSD | DEGREE_PLANNING | FAILED | Gagal memicu action PLAN_GENERATED atau plan_payload kosong |
| `TRKB-01` | TRKB | PREREQUISITE_DAG | HALLUCINATED | False Positive: Belum memenuhi prasyarat tapi model mengizinkan |
| `TRKB-04` | TRKB | PREREQUISITE_DAG | HALLUCINATED | False Positive: Belum memenuhi prasyarat tapi model mengizinkan |
| `TRKB-10` | TRKB | COURSE_DISCOVERY | FAILED | Matkul yang dicari tidak muncul di jawaban |
| `TRKB-11` | TRKB | DEGREE_PLANNING | FAILED | Gagal memicu action PLAN_GENERATED atau plan_payload kosong |
| `RN-04` | RN | PREREQUISITE_DAG | HALLUCINATED | False Positive: Belum memenuhi prasyarat tapi model mengizinkan |
| `RN-11` | RN | DEGREE_PLANNING | FAILED | Gagal memicu action PLAN_GENERATED atau plan_payload kosong |
| `RN-12` | RN | DEGREE_PLANNING | FAILED | Gagal memicu action PLAN_GENERATED atau plan_payload kosong |
| `TI-01` | TI | PREREQUISITE_DAG | HALLUCINATED | False Positive: Belum memenuhi prasyarat tapi model mengizinkan |
| `TI-11` | TI | DEGREE_PLANNING | FAILED | Gagal memicu action PLAN_GENERATED atau plan_payload kosong |
| `TE-03` | TE | PREREQUISITE_DAG | HALLUCINATED | False Negative: Seharusnya eligible/boleh tapi model menolak |
| `TE-04` | TE | PREREQUISITE_DAG | HALLUCINATED | False Positive: Belum memenuhi prasyarat tapi model mengizinkan |
| `TE-11` | TE | DEGREE_PLANNING | FAILED | Gagal memicu action PLAN_GENERATED atau plan_payload kosong |

---

## 5. Berkas Artefak Eksekusi (Dapat Diinspeksi Mentah)
- `benchmark_raw_traces.jsonl`: Setiap baris berisi full dump pesan input, response model, pemanggilan tool, dan latensi per step.
- `benchmark_metrics.csv`: Data mentah 60 baris tabel telemetri.
- `benchmark_results.json`: Nilai agregat metrik.