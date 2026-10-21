# Exp-028: Benchmarking Komparatif Spektrum Model SLM dan LLM pada Master Dataset Akademik

## Ringkasan Eksperimen
Eksperimen ini menyajikan evaluasi komparatif empiris spektrum model bahasa Direct Instruct Non-Thinking pada master dataset akademik institusi FTMM COMPASS (449 butir soal: 369 silabus kurikulum 5 prodi dan 80 regulasi SOP fakultas).

## Spektrum Model yang Diuji
1. `qwen2.5:1.5b` (Small Language Model - Edge)
2. `qwen2.5:3b` (Small Language Model - Baseline Utama Produksi)
3. `llama3.2:3b` (Small Language Model - Compact Llama Architecture)
4. `qwen2.5:7b` (Compact Large Language Model - High Capacity Reference)

## File Artefak
- `benchmark_summary.md`: Laporan naratif lengkap komparasi model dan batas efisiensi Pareto.
- `benchmark_metrics.csv`: Matriks numerik lengkap seluruh metrik.
- `benchmark_results.json`: Hasil agregat terstruktur per subkategori kurikulum & regulasi.
- `benchmark_raw_traces.jsonl`: Trace mentah inferensi per kueri per model.
