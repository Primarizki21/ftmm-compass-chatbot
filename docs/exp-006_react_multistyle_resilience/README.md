# Eksperimen 006: ReAct Agent Multi-Style Resilience & Paraphrase Invariance

- **Eksperimen ID:** `exp-006`
- **Tanggal:** 2026-10-03
- **Kartu Trello:** Card 44 (`[Nauval & Ario] Implementasi ReAct Agent`) & Card 91 (`[Nauval] Normalizer Slang`)
- **Model:** Ollama `qwen2.5:3b` pada GPU lokal NVIDIA RTX 5050 (8GB VRAM)
- **Tujuan:** Menguji ketahanan agen terhadap variasi diksi (Formal, Slang, Implisit) untuk membuktikan sistem tidak mengalami overfitting pada dataset sintetis.

## Hasil Kunci:
- **Overall Task Success Rate:** 88.3%
- **Paraphrase Invariance Score:** 75.0%
- **Tool Selection Accuracy:** 83.3%
- **Rata-rata Latensi Total:** 2407.1 ms (P95: 6078.6 ms)

## Berkas Artefak:
1. `benchmark_raw_traces.jsonl`: Jejak step-by-step mentah setiap turn dialog dan pemanggilan tools.
2. `benchmark_metrics.csv`: Data kuantitatif per kueri.
3. `benchmark_summary.md`: Tabel komparasi lengkap gaya bahasa dan analisis invariansi.
4. `benchmark_results.json`: Rekap statistik JSON.
