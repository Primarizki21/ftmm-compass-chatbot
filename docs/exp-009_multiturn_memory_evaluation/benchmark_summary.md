# Benchmark Summary — Exp-009: Multi-Turn Conversation Memory & Crescendo Defense

**Tanggal Pengujian:** 2026-10-10
**Hardware Platform:** NVIDIA GeForce RTX 5050 Laptop GPU (8GB VRAM), Local Ollama `qwen2.5:3b`
**Dataset Uji:** 10 Skenario Dialog Multi-Turn (36 Total Dialog Turns per Arsitektur)

## 1. Tabel Komparasi Utama Arsitektur Memory

| Arsitektur Percakapan | Dialog Success (%) | Context Retention (%) | Anaphora Resolution (%) | Crescendo Defense (%) | Avg Latensi (ms) | P50 (ms) |
|---|---|---|---|---|---|---|
| **Stateless Baseline** | **71.4%** | **90.0%** | **14.3%** | **100.0%** | 1691.3 ms | 1094.6 ms |
| **Stateful Memory Manager** | **94.3%** | **100.0%** | **85.7%** | **100.0%** | 3442.3 ms | 3421.1 ms |

## 2. Temuan Kunci Ilmiah

1. **Peningkatan Drastis Dialog Success:** Penambahan ConversationMemoryManager mendongkrak keberhasilan dialog multi-turn secara signifikan dibanding baseline stateless.
2. **Resolusi Anafora Efektif:** Kata ganti rujukan ('mata kuliah tersebut', 'itu', 'nya') berhasil dipetakan ke entitas matkul yang dibahas pada giliran sebelumnya.
3. **Pertahanan Crescendo Attack:** Serangan manipulasi bertahap berhasil dinetralkan pada giliran kritis tanpa meloloskan instruksi berbahaya.
