# Eksperimen 004: Evaluasi Penuh ReAct Agent (Zero-Abstraction Telemetry)

- **Eksperimen ID:** `exp-004`
- **Tanggal:** 2026-10-03
- **Kartu Trello:** Card 44 (`[Nauval & Ario] Implementasi ReAct Agent dengan tool-calling`)
- **Model:** Ollama `qwen2.5:3b` pada GPU lokal NVIDIA RTX 5050 (8GB VRAM)
- **Tujuan:** Evaluasi skala penuh ReAct Agent dengan instrumentasi data mentah tanpa abstraksi pada 60 skenario lintas 5 prodi FTMM.

## Hasil Kunci:
- **Task Success Rate:** 75.0%
- **Tool Selection Accuracy:** 96.7%
- **Hallucination Rate:** 10.0%
- **Rata-rata Latensi:** 2703.7 ms (P95: 6807.5 ms)
- **Rata-rata Langkah:** 2.1 steps

## Berkas Artefak:
1. `benchmark_raw_traces.jsonl`: Rekaman step-by-step mentah setiap turn dialog dan pemanggilan tools.
2. `benchmark_metrics.csv`: Data telemetri kuantitatif per skenario.
3. `benchmark_summary.md`: Analisis lengkap performa per prodi dan failure analysis.
4. `benchmark_results.json`: Format JSON rekap statistik.
