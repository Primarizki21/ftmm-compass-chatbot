# Eksperimen 009: Multi-Turn Conversation Memory & Crescendo Defense

- **Eksperimen ID:** `exp-009`
- **Tanggal:** 2026-10-07
- **Kartu Trello:** Card 46 (`[Nauval & Ario] Implementasi conversation history management`) & Card 92 (`[Ario & Nauval] Standarisasi Dataset Ground Truth Adversarial & Multi-Turn`)
- **Model:** Ollama `qwen2.5:3b` + `nomic-embed-text` pada GPU lokal NVIDIA RTX 5050 Laptop GPU (8GB VRAM)
- **Tujuan:** Mengevaluasi efektivitas arsitektur manajemen memori percakapan berkeadaan (*Stateful Conversation Memory Manager*) dengan *Sliding Window Buffer* (6 turns / 12 messages), akumulasi profil mahasiswa persisten, dan resolusi anafora terhadap performa dialog akademik serta ketahanan terhadap *Crescendo Jailbreak Attack*.

---

## 1. Hasil Kunci & Komparasi Kuantitatif (70 Run Turn Dialog)

| Arsitektur Percakapan | Dialog Success (%) [↑] | Context Retention (%) [↑] | Anaphora Resolution (%) [↑] | Crescendo Defense (%) [↑] | Avg Latensi Turn (ms) | P50 (ms) | P95 (ms) |
|---|---|---|---|---|---|---|---|
| **Stateless Baseline (Tanpa Memori)** | 71.4% | 90.0% | 14.3% | **100.0%** | **1691.3 ms** | **1094.6 ms** | 5681.4 ms |
| **Stateful Memory Manager (Card 46)** | **94.3%** | **100.0%** | **85.7%** | **100.0%** | 3442.3 ms | 3421.1 ms | 6259.9 ms |

---

## 2. Temuan Ilmiah Utama (Insights & Findings)

1. **Lonjakan Drastis Keberhasilan Dialog Multi-Turn (+22.9%):**
   - Keberhasilan dialog melonjak dari 71.4% pada Baseline tanpa memori menjadi **94.3%** dengan `ConversationMemoryManager`.
   - Pada Stateless baseline, mahasiswa harus mengulang nama prodi dan mata kuliah di setiap pesan; jika tidak, agen gagal memahami konteks atau bertanya balik secara berulang.

2. **Resolusi Anafora Efektif (+71.4% Peningkatan):**
   - Pada model stateless, kata ganti rujukan (*"mata kuliah tersebut"*, *"itu"*, *"dosen pengampunya"*) gagal dipahami (akurasi hanya 14.3%), menyebabkan model tersesat atau diblokir oleh filter OOD.
   - Modul `resolve_anaphora` berhasil memetakan kata rujukan ke entitas kurikulum yang dibahas pada giliran sebelumnya (**85.7% akurasi resolusi**).

3. **Retensi Konteks Sempurna pada Percakapan Panjang (100% Retensi):**
   - Pengujian stres dialog panjang (Skenario MT-06 hingga 7 turn) membuktikan bahwa informasi prodi (*Teknologi Sains Data*) dan semester saat ini (*Semester 2*) yang disebutkan pada Turn 1 tetap bertahan utuh hingga Turn 7 tanpa mengalami *context drift* atau halusinasi.

4. **Pertahanan Mutlak terhadap Crescendo Attack (100% Defense):**
   - Penyerang yang mencoba memecah instruksi jailbreak dan manipulasi SKS (40 SKS) dalam 3 giliran dialog bertahap berhasil dinetralkan 100% pada giliran kritis oleh lapisan *Tier 1 Regex* dan *Tier 4 Tool Rail* tanpa membocorkan sistem prompt atau parameter bypass.

5. **Trade-off Latensi Konteks Geser (Sliding Window):**
   - Penambahan konteks riwayat obrolan meningkatkan latensi rata-rata dari 1691.3 ms ke 3442.3 ms akibat kenaikan token konteks input (dari ~500 token menjadi ~1500 token).
   - Penggunaan *Sliding Window Buffer* 6 giliran berhasil membatasi latensi P95 di angka **6259.9 ms**, mencegah *runaway token expansion* yang dapat memicu OOM pada GPU 8GB.

---

## 3. Keputusan Arsitektur & Tindak Lanjut

- `ConversationMemoryManager` resmi diintegrasikan sebagai engine state dialog untuk FTMM COMPASS.
- Menuntaskan target riset Card 46 dan bagian multi-turn dari Card 92 di Sprint 6.
- Sistem siap diexpose sebagai API endpoint FastAPI (`POST /api/chat`) di Card 47.

---

## 4. Berkas Artefak Eksperimen

1. `benchmark_raw_traces.jsonl`: 70 jejak dialog per giliran (input pengguna, kueri teresolusi, profil terakumulasi, respons, latensi).
2. `benchmark_metrics.csv`: Data kuantitatif komparasi Stateless vs Stateful.
3. `benchmark_summary.md`: Ringkasan metrik untuk laporan ilmiah TA.
4. `benchmark_results.json`: Rekap statistik JSON.
