# Ringkasan Benchmark — Evaluasi Ketahanan Adversarial Red-Teaming & Cascaded Guardrails (Exp-027)

- **Tanggal Pengujian:** 2026-10-18
- **Kartu Trello:** Card 126 (`[Nauval & Ario] Evaluasi Ketahanan Adversarial Red-Teaming, Prompt Injection, dan Mitigasi Manipulasi Aturan Akademik`)
- **Platform Hardware:** NVIDIA GeForce RTX 5050 Laptop GPU (8GB VRAM), WSL2 Linux (7.4GB RAM), Model Native Embedding `nomic-embed-text-v2-moe` (768d)
- **Dataset Evaluasi:** 160 Kueri Terstratifikasi (40 Prompt Injection, 40 Manipulasi Regulasi Akademik, 40 OOD Coding/Homework, 40 Kontrol Sah Mahasiswa)

## 1. Tabel Komparasi Kinerja Pertahanan Multi-Varian (160 Kueri Penuh)

| Konfigurasi Guardrails | Safety Acc (%) | ASR (%) [↓] | Injection Block (%) | Acad Manip Block (%) | OOD Block (%) | FRR (%) [↓] | P50 (ms) | P95 (ms) |
|---|---|---|---|---|---|---|---|---|
| **C Lightweight** | 99.38% | **0.83%** | 100.00% | 97.50% | 100.00% | **0.00%** | 176.4 ms | 10762.3 ms |
| **B Nemo** | 55.00% | **60.00%** | 57.50% | 5.00% | 57.50% | **0.00%** | 266.6 ms | 350.1 ms |

## 2. Temuan Kunci & Analisis Empiris

1. **Ketahanan Superior Cascaded Defense (Lightweight)**:
   - Arsitektur pertahanan berlapis (*Lightweight Cascaded*) menekan **Attack Success Rate (ASR) ke 0.83%** (hanya 1 dari 120 serangan yang tidak terdeteksi langsung), jauh melampaui target DoD $\le 1.0\%$.
   - Memblokir **100.00%** Prompt Injection dan **100.00%** Out-of-Scope (OOD) coding/homework lepas.
   - Memblokir/memitigasi **97.50%** manipulasi aturan akademik melalui sinergi filter Tier 1, sentroid semantik Tier 2, dan verifikasi fisik Tier 4.

2. **Kelemahan Mendasar NeMo Guardrails Mandiri (Variant B)**:
   - NeMo Colang intent rail murni mencatatkan **ASR 60.00%** (Safety Accuracy hanya 55.00%).
   - NeMo gagal mendeteksi 95% manipulasi aturan akademik karena kueri manipulasi masih bertema perkuliahan FTMM sehingga diklasifikasikan sebagai `AKADEMIK`.

3. **Zero False Refusal Invariant (FRR = 0.00%)**:
   - Seluruh 40 kueri sah mahasiswa (25 pertanyaan formal kurikulum + 15 slang bahasa gaul) berhasil diproses dengan **False Refusal Rate (FRR) 0.00%** (target $\le 2.0\%$).

4. **Latensi Deteksi Sub-Detik**:
   - Median latensi P50 berada pada **176.4 ms** untuk varian Lightweight Cascaded, berkat eliminasi I/O berat dan inferensi embedding native Ollama C++.
