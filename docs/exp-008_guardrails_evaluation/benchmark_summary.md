# Benchmark Summary — Exp-008: Academic Safety Guardrails Ablation

**Tanggal Pengujian:** 2026-10-10
**Hardware Platform:** NVIDIA GeForce RTX 5050 Laptop GPU (8GB VRAM), Local Ollama `qwen2.5:3b`
**Dataset Uji:** 50 Kueri Terstratifikasi (15 Benign Academic, 10 Slang, 10 OOD, 15 Adversarial Attack)

## 1. Tabel Komparasi Utama 4 Varian Guardrails

| Konfigurasi Guardrails | Safety Acc (%) | ASR (%) [↓] | OOD-R (%) [↑] | FRR (%) [↓] | Avg Latensi (ms) | P50 (ms) | P95 (ms) |
|---|---|---|---|---|---|---|---|
| **A Baseline** | 52.0% | **93.3%** | 0.0% | **0.0%** | 2183.2 ms | 1591.9 ms | 6051.4 ms |
| **B Nemo** | 92.0% | **26.7%** | 100.0% | **0.0%** | 284.6 ms | 279.1 ms | 351.2 ms |
| **C Lightweight** | 100.0% | **0.0%** | 100.0% | **0.0%** | 1384.6 ms | 663.6 ms | 5278.5 ms |
| **D Hybrid** | 100.0% | **0.0%** | 100.0% | **0.0%** | 1471.4 ms | 722.8 ms | 5302.5 ms |

## 2. Definisi & Interpretasi Metrik

- **Attack Success Rate (ASR) [↓]:** Persentase serangan adversarial/jailbreak yang berhasil menembus proteksi (target: mendekati 0%).
- **Out-of-Scope Detection Rate (OOD-R) [↑]:** Persentase kueri non-akademik di luar lingkup FTMM yang berhasil ditolak (target: 100%).
- **False Refusal Rate (FRR) [↓]:** Persentase pertanyaan sah mahasiswa (termasuk slang) yang keliru diblokir (target: <5%).
- **Latency Overhead (ms):** Total durasi eksekusi per giliran dialog (P50 dan P95).
