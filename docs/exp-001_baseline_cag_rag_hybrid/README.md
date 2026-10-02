# Eksperimen 001: Baseline Evaluasi Komparatif Pure RAG vs Pure CAG vs Hybrid RAG-CAG

- **Eksperimen ID:** 
- **Tanggal:** 2026-09-30
- **Model:** Ollama 
- **Embedding:**  (768 dimensi, Cosine FAISS IndexFlatIP)
- **Tujuan:** Menjawab Rumusan Masalah Tugas Akhir FTMM COMPASS:
  - **RQ1**: Arsitektur mana yang paling unggul secara keseluruhan?
  - **RQ2**: Bagaimana pengaruh ragam bahasa kueri (Formal vs Informal)?
  - **RQ3**: Bagaimana trade-off antara akurasi, latensi inferensi, dan konsistensi semantik?

## Berkas Artefak Eksperimen
1. : Laporan naratif dan tabel metrik perbandingan utama.
2. : Data mentah 90 kali pengujian inferensi (30 kueri x 3 arsitektur).

## Ringkasan Temuan
- **Hybrid RAG-CAG** membuktikan varians konsistensi terendah (±0.1561) dan paling tahan terhadap variasi slang mahasiswa (delta formal-informal hanya +0.0435).
- **Pure CAG** tercepat (0.0 ms retrieval, 1665.4 ms total), namun terbatas pada korpus yang muat di prompt context.
- **Pure RAG** memiliki retrieval overhead rata-rata 34.3 ms pada setiap turn dan paling rentan terhadap kueri informal (drop akurasi 0.6179 -> 0.5627).
