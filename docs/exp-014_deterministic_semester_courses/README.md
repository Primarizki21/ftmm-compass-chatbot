# Eksperimen 014: Evaluasi Fitur Pencarian Mata Kuliah Deterministik per Semester

- **Eksperimen ID:** `exp-014`
- **Tanggal:** 2026-10-11
- **Kartu Trello:** Card 44 (`[Nauval & Ario] Implementasi ReAct Agent dengan tool-calling`) & Card 51 (`[Hans] Implementasi UI Module/Course Finder`)
- **Tujuan:** Menguji akurasi tool deterministik `get_courses_by_semester` pada 40 kombinasi (5 prodi x 8 semester) untuk mengeliminasi pemotongan top_k=3 semantik.

## Hasil Kunci:
- **Tool Selection Accuracy:** 100.0%
- **Parameter Extraction Accuracy:** 100.0%
- **Course Set-Match Recall:** 100.0%
- **Exact Set Match Rate:** 100.0%
- **Avg Tool Execution Latency:** 0.165 ms

## Artefak Berkas:
1. `benchmark_metrics.csv`: Data hasil pengujian 40 kasus uji per baris.
2. `benchmark_raw_traces.jsonl`: Jejak lengkap argumen dan kode matkul yang dikembalikan.
3. `benchmark_results.json`: Ringkasan metrik statistik teragregasi.
4. `benchmark_summary.md`: Tabel perbandingan komparatif dan temuan ilmiah.
