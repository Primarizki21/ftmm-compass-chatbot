# Exp-029: Ekspansi RAG Regulasi Akademik, Eliminasi False Refusal Skripsi, dan Optimasi Token Reasoning

## Ringkasan Proyek
- **Tanggal Eksperimen:** 2026-10-20
- **PIC:** Nauval Aryasetya & Ario
- **Komponen yang Ditingkatkan:**
  1. `Tier2SemanticGate`: Penggantian anchor privilege escalation yang bertabrakan leksikal dengan kata 'skripsi'.
  2. `RegulationRetriever` (`ftmm_regulations.faiss`): Kompilasi 262 butir regulasi (Skripsi 152 butir, PKL 45 butir, Yudisium 24 butir, Akademik 33 butir, SOP 8 butir).
  3. `search_academic_regulations`: Tool OpenAI function-calling baru untuk pencarian semantik regulasi pada agent.
  4. `UnifiedChatEngine`: Ekspansi token budget reasoning model dari 300 menjadi 1536 token pada rute CAG/Regulasi.
  5. `ChatMessageItem`: Relaksasi batas minimal panjang konten riwayat untuk mencegah cascading HTTP 422 error.

## Artefak Eksperimen
- `benchmark_metrics.csv`
- `benchmark_results.json`
- `benchmark_summary.md`
- `benchmark_raw_traces.jsonl`
