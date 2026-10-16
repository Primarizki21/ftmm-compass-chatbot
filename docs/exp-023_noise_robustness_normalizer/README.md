# Eksperimen 023: Uji Ketahanan Perturbasi Typo/STT & Integrasi Levenshtein Normalizer

## Deskripsi Riset
Eksperimen ini menguji ketahanan sistem NLP FTMM Compass Chatbot terhadap variasi noise pengguna nyata: typo keyboard QWERTY, slang/akronim ekstrem mahasiswa, dan kesalahan transkripsi fonetik Speech-to-Text (STT).
Modul `AcademicNormalizer` mengintegrasikan kamus leksikal O(1), penanganan fonetik STT, dan Bounded Damerau-Levenshtein ($d \le 2$) dengan length-indexed candidate pruning.

## Artefak Evaluasi
- `benchmark_summary.md`: Laporan eksekutif ringkasan metrik komparasi.
- `benchmark_metrics.csv`: Data metrik tabuler terstruktur per subset.
- `benchmark_results.json`: Dump hasil evaluasi lengkap termasuk catatan per sampel.
- `benchmark_raw_traces.jsonl`: Log jejak transformasi normalizer dan evaluasi per sampel.
