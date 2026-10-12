# Exp-016: Hardening Semantic Guardrails & Dual-Centroid Out-of-Scope Boundary

## Ringkasan Eksperimen
Eksperimen ini memperkuat lapisan pertahanan Tier 2 Semantic Gate dan ReAct Agent Engine guna mengatasi kebocoran kueri non-akademik umum (seperti `buatkan kode bubble sort`) tanpa bergantung pada framework NeMo Guardrails yang berat dan tanpa regex rapuh.

## Struktur Artefak
- `benchmark_summary.md`: Laporan komparasi metrik kuantitatif dan analisis ilmiah.
- `benchmark_metrics.csv`: Data metrik terstruktur dalam format CSV.
- `benchmark_results.json`: Dump metrik agregat JSON.
- `benchmark_raw_traces.jsonl`: Jejak eksekusi mentah dari seluruh 60 kueri pengujian.
