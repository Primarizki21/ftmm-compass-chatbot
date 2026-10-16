# Exp-024: Benchmark Komparasi SLM Ternary (Bonsai 2: 2B, 4B, 9B) vs Baseline Produksi Qwen2.5

Direktori ini berisi seluruh artefak riset empiris dan evaluasi kuantitatif komparasi efisiensi model Native Ternary (Bonsai 2: 2B, 4B, 9B format PQ2_0) vs model baseline produksi Qwen2.5 (1.5B, 3B, 7B format Q4_K_M) pada GPU NVIDIA GeForce RTX 5050 (8GB VRAM).

## Struktur Berkas Artefak

- `README.md`: Panduan direktori dan ringkasan eksekutif eksperimen.
- `benchmark_summary.md`: Laporan komprehensif metrik komparasi 6 model, breakdown per domain akademik, dan analisis Pareto multi-objective.
- `benchmark_metrics.csv`: Matriks data numerik per model untuk inspeksi spreadsheet dan visualisasi grafik.
- `benchmark_results.json`: Ringkasan agregat terstruktur format JSON.
- `benchmark_raw_traces.jsonl`: Log lengkap 300 kueri inferensi mentah (50 kueri $\times$ 6 model), mencakup token, TTFT, TPS, VRAM, dan status eksekusi alat fisik.

## Skrip & Pengujian

- Skrip Eksekusi Benchmark: `scripts/evaluate_ternary_bonsai.py`
- Unit Test Regresi: `tests/test_ternary_bonsai.py`
- Dataset Evaluasi: `data/ground_truth/slm_sizing_eval_dataset.json` (50 kueri terstratifikasi)
