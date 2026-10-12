# Exp-015: Evaluasi System-1 Decider dengan Centroid EmbeddingGemma-2 & MRL Truncation

Eksperimen evaluasi ilmiah kelayakan unifikasi embedding backbone (*Single-Model Residency*) pada System-1 Decision Classifier dengan menggantikan centroid `nomic-embed-text` menggunakan `google/embeddinggemma-2` (270M), serta menguji *task instruction prefix* (`task: classification`) dan *Matryoshka Representation Learning* (768d, 512d, 256d, 128d) terhadap akurasi tool routing dan separabilitas margin.

## File Artefak
- `benchmark_summary.md`: Laporan komprehensif metrik klasifikasi intent akademik, separabilitas margin, dan analisis ablasi MRL.
- `benchmark_metrics.csv`: Data kuantitatif terstruktur komparasi 5 varian lintas segmen gaya bahasa.
- `benchmark_results.json`: Dump hasil agregasi metrik per varian dan per kategori akademik.
- `benchmark_raw_traces.jsonl`: Log rekaman inferensi mentah per kueri (confidence, margin, predicted vs expected).
