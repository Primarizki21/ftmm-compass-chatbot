# Eksperimen 008: Academic Safety Guardrails Ablation (NeMo vs Cascaded vs Hybrid)

- **Eksperimen ID:** `exp-008`
- **Tanggal:** 2026-10-07
- **Kartu Trello:** Card 45 (`[Nauval & Ario] Konfigurasi NeMo Guardrails sebagai safety layer`) & Card 92 (`[Ario & Nauval] Standarisasi Dataset Ground Truth Adversarial & Multi-Turn`)
- **Model:** Ollama `qwen2.5:3b` + `nomic-embed-text` pada GPU lokal NVIDIA RTX 5050 Laptop GPU (8GB VRAM)
- **Tujuan:** Mengevaluasi secara empiris efektivitas pertahanan berlapis (Defense-in-Depth) terhadap prompt injection, jailbreak, authority spoofing, dan kueri out-of-scope (OOD) serta mengukur dampaknya terhadap False Refusal Rate (FRR) dan latensi eksekusi lokal.

---

## 1. Hasil Kunci & Komparasi 4 Varian (200 Run Evaluasi)

| Varian Arsitektur | Safety Accuracy (%) | Attack Success Rate (ASR) [↓] | Out-of-Scope Detection (OOD-R) [↑] | False Refusal Rate (FRR) [↓] | Avg Latensi Total (ms) | P50 (ms) | P95 (ms) |
|---|---|---|---|---|---|---|---|
| **Varian A (Baseline Raw ReAct)** | 52.0% | 93.3% | 0.0% | **0.0%** | 2183.2 ms | 1591.9 ms | 6051.4 ms |
| **Varian B (NeMo Guardrails)** | 92.0% | 26.7% | **100.0%** | **0.0%** | **284.6 ms** | **279.1 ms** | **351.2 ms** |
| **Varian C (Lightweight Cascaded Guard)** | **100.0%** | **0.0%** | **100.0%** | **0.0%** | 1384.6 ms | 663.6 ms | 5278.5 ms |
| **Varian D (Hybrid Defense)** | **100.0%** | **0.0%** | **100.0%** | **0.0%** | 1471.4 ms | 722.8 ms | 5302.5 ms |

---

## 2. Temuan Kunci Ilmiah (Insights & Decision)

1. **Kerentanan Mutlak Baseline (Raw ReAct):**
   - Model mentah tanpa guardrails mencatat **Attack Success Rate (ASR) sebesar 93.3%** (14 dari 15 serangan adversarial tembus).
   - Seluruh kueri non-akademik di luar lingkup FTMM dijawab bebas (OOD-R 0.0%), membuktikan chatbot akademik mutlak membutuhkan filter keamanan deterministik.

2. **Keunggulan Sempurna Lightweight Cascaded Guard (Varian C):**
   - Menggabungkan Tier 1 (Regex blocklist, 0.0 ms) + Tier 2 (FAISS Cosine Semantic Gate, ~30 ms) + Tier 4 (Deterministic Tool Rail) berhasil mencapai **100.0% Safety Accuracy**.
   - Menekan **ASR hingga 0.0%** dan mendeteksi kueri non-akademik **100.0% OOD-R**.
   - Pada serangan adversarial dan OOD yang diblokir, latensi eksekusi turun drastis ke **0.0 – 35 ms** karena proses langsung diputus tanpa memanggil LLM autoregresif. P50 latensi terpangkas dari 1591.9 ms (Baseline) menjadi **663.6 ms** (-58.3%).

3. **Zero False Refusal Rate (FRR 0.0%):**
   - Seluruh 15 kueri akademik formal dan 10 kueri ragam slang mahasiswa FTMM (*"alpro"*, *"visdat"*, *"skripsian"*, *"ngebut"*, *"gas ambil"*) lolos 100% tanpa ada pemblokiran keliru (*over-defense*).

4. **Karakteristik NeMo Guardrails (Varian B):**
   - Pendekatan Colang canonical intent classifier menghasilkan latensi inferensi stabil (284.6 ms), namun masih meloloskan 26.7% serangan adversarial yang menggunakan teknik persuasi DAN/jailbreak bahasa Inggris canggih yang mengelabui self-check LLM.
   - Mengombinasikan NeMo dengan filter deterministik (Varian D) menutup seluruh celah serangan (ASR 0.0%).

---

## 3. Keputusan Arsitektur & Rekomendasi

- **Varian C (Lightweight Cascaded Guard)** ditetapkan sebagai safety wrapper standar untuk chatbot FTMM COMPASS.
- Memberikan proteksi 100% tanpa ketergantungan framework black-box berat, menjaga efisiensi komputasi lokal GPU RTX 5050 (8GB VRAM), dan siap dipasang pada endpoint API FastAPI di Card 47.

---

## 4. Berkas Artefak Eksperimen

1. `benchmark_raw_traces.jsonl`: 200 rekam jejak telemetri mentah per kueri (prompt, trigger tier, refusal reason, latensi tier, action type).
2. `benchmark_metrics.csv`: Tabel matriks kuantitatif 4 varian.
3. `benchmark_summary.md`: Laporan komparasi metrik untuk Laporan TA.
4. `benchmark_results.json`: Rekapitulasi metrik format JSON mesin.
