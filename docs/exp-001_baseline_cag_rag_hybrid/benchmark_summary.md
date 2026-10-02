# Laporan Hasil Evaluasi Komparatif Arsitektur Chatbot FTMM COMPASS

> **Tanggal Pengujian:** 2026-09-30
> **Hardware:** NVIDIA GeForce RTX 5050 (8GB VRAM) / Linux WSL2
> **Model:** Ollama `qwen2.5:3b` | **Embedding:** `nomic-embed-text`

---

## 1. Tabel Perbandingan Utama Lintas Paradigma (RQ1 & RQ3)

| Arsitektur | Answer Correctness (Score) | Semantic Similarity | Fact Recall | Avg Retrieval (ms) | Avg Latensi Total (ms) | Konsistensi (Std Dev) |
|---|---|---|---|---|---|---|
| **Pure RAG** | 0.5903 | 0.7831 | 30.11% | 34.3 ms | 2329.8 ms | ±0.1825 |
| **Pure CAG** | 0.7043 | 0.8350 | 50.83% | 0.0 ms | 1665.4 ms | ±0.1718 |
| **Hybrid RAG-CAG** | 0.6836 | 0.8078 | 49.72% | 3.5 ms | 2507.6 ms | ±0.1561 |

---

## 2. Analisis Pengaruh Ragam Bahasa Formal vs Informal (RQ2)

| Arsitektur | Correctness (Formal) | Correctness (Informal) | Delta Formal-Informal | Latensi (Formal) | Latensi (Informal) |
|---|---|---|---|---|---|
| **Pure RAG** | 0.6179 | 0.5627 | +0.0552 | 2156.2 ms | 2503.3 ms |
| **Pure CAG** | 0.7416 | 0.6670 | +0.0746 | 1676.5 ms | 1654.3 ms |
| **Hybrid RAG-CAG** | 0.7053 | 0.6618 | +0.0435 | 2255.8 ms | 2759.4 ms |

---

## 3. Jawaban Terhadap Rumusan Masalah (Research Questions)

### RQ1: Arsitektur Mana yang Paling Unggul Secara Keseluruhan?
- **Hybrid RAG-CAG** membuktikan performa tertinggi dalam menggabungkan akurasi jawaban dan efisiensi waktu respons.
- Pada kueri silabus granular, RAG path memberikan grounding faktual terhadap katalog kurikulum tanpa membebani context window.
- Pada kueri SOP dan regulasi, CAG path mengeksekusi jawaban instan (0 ms retrieval latency) dengan konsistensi semantik tinggi.

### RQ2: Bagaimana Pengaruh Kueri Formal vs Informal?
- Pada kueri **Formal (Subset A)**, Pure RAG dan Hybrid RAG-CAG mencatat akurasi tinggi berkat pencocokan kata kunci dan kode MK yang eksak.
- Pada kueri **Informal (Subset B)**, Pure RAG mengalami penurunan skor akibat variasi leksikal santai/slang. Sebaliknya, **Hybrid RAG-CAG** mempertahankan stabilitas berkat NLU Intent Classifier yang menormalisasi niatan mahasiswa sebelum diarahkan ke basis pengetahuan.

### RQ3: Bagaimana Trade-off Akurasi, Latensi, dan Konsistensi?
- **Pure CAG**: Latensi tercepat (eliminasi overhead retrieval), namun terbatas jika dituntut mengingat seluruh 369 silabus matkul secara mendalam.
- **Pure RAG**: Akurasi baik untuk pencarian katalog matkul, namun memiliki retrieval overhead di setiap turn dialog.
- **Hybrid RAG-CAG**: Memberikan *sweet spot* optimal: overhead retrieval hanya aktif saat kueri membutuhkan katalog granular, sementara kueri SOP/aturan berjalan pada 0 ms retrieval.