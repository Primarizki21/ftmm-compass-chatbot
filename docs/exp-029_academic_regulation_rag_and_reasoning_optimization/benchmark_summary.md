# Ringkasan Evaluasi Exp-029: Ekspansi RAG Regulasi Akademik & Optimasi Token Reasoning

## 1. Executive Summary
Eksperimen Exp-029 memvalidasi integrasi indeks vektor baru untuk regulasi akademik (Pedoman Skripsi, PKL, Yudisium, Peraturan Dekan, dan 8 SOP Layanan Administrasi) serta perbaikan eliminasi false refusal pada kueri skripsi dan optimasi token budget penalaran pada model inferensi lokal.

## 2. Metrik Utama Komparatif

| Metrik Evaluasi | Baseline (Sebelum Perbaikan) | Exp-029 (Hasil Implementasi) | Status |
|---|---|---|---|
| **False Refusal Rate (FRR) Skripsi** | 100.0% (Tertolak Guardrail) | **0.00%** | **PASS (Zero FRR)** |
| **Retrieval Recall@3 Dokumen Regulasi** | 0.0% (Tidak Terindeks) | **100.00%** | **PASS (>= 90%)** |
| **Response Completion Rate** | 0.0% (Token Starvation / 422) | **100.00%** | **PASS (100%)** |
| **Rata-rata Latensi Total (ms)** | - (Error/Refused 236 ms) | **8754.11 ms** | **OPTIMAL** |

## 3. Rincian Metrik per Kueri

| ID | Topik | Pertanyaan | Guardrail OOD | Recall@3 | Routing | Latensi (ms) | Status |
|---|---|---|---|---|---|---|---|
| REG-01 | SKRIPSI | berikan aku panduan singkat mengenai skr... | False | 1.0 | CAG | 10844.28 ms | PASS |
| REG-02 | SKRIPSI | apa saja syarat proposal skripsi dan bat... | False | 1.0 | CAG | 7570.56 ms | PASS |
| REG-03 | SKRIPSI | bagaimana ketentuan ujian skripsi dan ni... | False | 1.0 | CAG | 11984.51 ms | PASS |
| REG-04 | SKRIPSI | prosedur pengajuan judul skripsi dan tur... | False | 1.0 | CAG | 11962.99 ms | PASS |
| REG-05 | PKL | apa syarat pkl di ftmm?... | False | 1.0 | REACT_AGENT | 5133.36 ms | PASS |
| REG-06 | PKL | berapa hari kerja minimal pelaksanaan pk... | False | 1.0 | CAG | 4762.09 ms | PASS |
| REG-07 | PKL | bagaimana komposisi penilaian praktik ke... | False | 1.0 | CAG | 4898.41 ms | PASS |
| REG-08 | YUDISIUM | bagaimana syarat yudisium kelulusan sarj... | False | 1.0 | CAG | 10377.88 ms | PASS |
| REG-09 | YUDISIUM | berapa batas maksimal nilai D dan nilai ... | False | 1.0 | RAG | 8111.05 ms | PASS |
| REG-10 | YUDISIUM | apakah ada syarat skor tes bahasa inggri... | False | 1.0 | CAG | 6852.83 ms | PASS |
| REG-11 | SOP | bagaimana prosedur pengajuan surat cuti ... | False | 1.0 | CAG | 12048.19 ms | PASS |
| REG-12 | SOP | cara meminta surat rekomendasi magang da... | False | 1.0 | CAG | 9073.39 ms | PASS |
| REG-13 | SOP | prosedur legalisir transkrip nilai dan k... | False | 1.0 | CAG | 10445.54 ms | PASS |
| REG-14 | SOP | langkah jika terlambat membayar ukt daft... | False | 1.0 | CAG | 11626.33 ms | PASS |
| REG-15 | AKADEMIK | berapa batas sks jika ips semester lalu ... | False | 1.0 | CAG | 9550.74 ms | PASS |
| REG-16 | AKADEMIK | aturan jatah sks kalau ips dapat 2.20 be... | False | 1.0 | CAG | 4823.54 ms | PASS |
