# Eksperimen 005: ReAct Agent Hardened (Mitigasi 5 Kluster Kegagalan)

- **Eksperimen ID:** `exp-005`
- **Tanggal:** 2026-10-03
- **Kartu Trello:** Card 44 (`[Nauval & Ario] Implementasi ReAct Agent dengan tool-calling`)
- **Model:** Ollama `qwen2.5:3b` pada GPU lokal NVIDIA RTX 5050 (8GB VRAM)
- **Tujuan:** Menguji efektivitas hardening arsitektur ReAct (profile auto-injection, default parameters, NoneType sanitation, anti-inversion schema) pada 60 skenario 5 prodi FTMM.

## Hasil Kunci:
- **Task Success Rate:** 88.3% (vs 75.0% di Exp-004)
- **Tool Selection Accuracy:** 95.0%
- **Hallucination Rate:** 10.0% (vs 10.0% di Exp-004)
- **Rata-rata Latensi:** 2934.0 ms (P95: 6609.7 ms)
- **Rata-rata Langkah:** 2.0 steps

## Berkas Artefak:
1. `benchmark_raw_traces.jsonl`: Rekaman step-by-step mentah setiap turn dialog dan pemanggilan tools.
2. `benchmark_metrics.csv`: Data telemetri kuantitatif per skenario.
3. `benchmark_summary.md`: Analisis lengkap performa per prodi dan failure analysis.
4. `benchmark_results.json`: Format JSON rekap statistik.
