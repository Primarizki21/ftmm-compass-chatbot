# Eksperimen 007: Evaluasi Bilingual ReAct & Hardware Efficiency (RTX 5050)

- **Eksperimen ID:** `exp-007`
- **Tanggal:** 2026-10-03
- **Kartu Trello:** Card 31 (CAG VRAM), Card 32 (Hybrid), Card 44 (ReAct Agent)
- **Model:** Ollama `qwen2.5:3b` pada GPU lokal NVIDIA RTX 5050 (8GB VRAM)
- **Tujuan:** Menguji performa penalaran dwibahasa (Indonesian vs English) dan profil efisiensi hardware lokal pada 40 kueri paralel.

## Hasil Kunci:
- **Overall Task Success Rate:** 87.5%
- **Bilingual Parity Rate:** 80.0%
- **Language Adherence:** 85.0%
- **Tool Selection Accuracy:** 100.0%
- **Rata-rata Latensi Total:** 2670.4 ms (P95: 6784.5 ms)

## Berkas Artefak:
1. `benchmark_raw_traces.jsonl`: Jejak step-by-step mentah setiap turn dialog dan pemanggilan tools.
2. `benchmark_metrics.csv`: Data kuantitatif per kueri ID dan EN.
3. `benchmark_summary.md`: Tabel ringkasan komparasi dwibahasa dan analisis hardware.
4. `benchmark_results.json`: Rekap statistik JSON.
