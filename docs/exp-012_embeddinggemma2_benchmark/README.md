# Exp-012: Benchmark Evaluasi EmbeddingGemma-2 vs nomic-embed-text

Eksperimen komparasi empiris Information Retrieval (IR) antara model `google/embeddinggemma-2` (270M text backbone) terhadap baseline `nomic-embed-text` (768d) pada 369 mata kuliah FTMM UNAIR.

## File Artefak
- `benchmark_summary.md`: Laporan komprehensif metrik IR (Hit@1, Hit@3, Hit@5, MRR, NDCG@5) dan analisis ablasi MRL.
- `benchmark_metrics.csv`: Data mentah agregasi per metode dan per register bahasa.
- `benchmark_raw_traces.jsonl`: Log detail pencarian dan peringkat mata kuliah per kueri.
