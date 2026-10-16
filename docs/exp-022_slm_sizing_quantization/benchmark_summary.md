# Benchmark Summary: Evaluasi Komparatif Sizing Model SLM & Kuantisasi Lokal

## 1. Matriks Performa & Telemetri Perangkat Keras

Pengujian dilakukan pada perangkat:
- **GPU**: NVIDIA GeForce RTX 5050 Laptop GPU (8151 MiB VRAM)
- **OS**: Linux WSL2 (Ubuntu 24.04 LTS), Kernel 6.6.87.2-microsoft
- **Inference Runtime**: Ollama C++ Daemon v0.32.5 (Zero Host RAM Python Allocation)
- **Kuantisasi**: GGUF Q4_K_M (Default Ollama)

| Model | Parameter | Tool Selection Acc (%) | Schema Compliance (%) | Param Error (%) | Physical Exec Success (%) | P50 Hardware TTFT (ms) | P95 Hardware TTFT (ms) | Mean TPS | Peak VRAM (MiB) | VRAM Delta (MiB) | Pareto Score | Pareto Frontier |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `qwen2.5:1.5b` | 1.54B | 34.00% | 36.00% | 66.00% | 34.00% | 123.6 ms | 133.1 ms | 157.1 | 4071 MiB | +1159 MiB | 51.23 | NO |
| `qwen2.5:3b` | 3.09B | 92.00% | 92.00% | 8.00% | 92.00% | 156.5 ms | 251.7 ms | 99.6 | 2914 MiB | +2 MiB | 89.81 | ⭐ YES |
| `llama3.2:3b` | 3.21B | 90.00% | 98.00% | 10.00% | 84.00% | 53.6 ms | 242.9 ms | 98.8 | 3136 MiB | +2315 MiB | 94.05 | ⭐ YES |
| `qwen2.5:7b` | 7.61B | 96.00% | 98.00% | 4.00% | 96.00% | 468.9 ms | 486.7 ms | 56.1 | 5346 MiB | +4525 MiB | 48.33 | ⭐ YES |

---

## 2. Rincian Akurasi per Kategori Kueri (50 Kueri Terstratifikasi)

| Model | SKS Calculation (10 Kueri) | Prerequisite DAG (10 Kueri) | Course Discovery (10 Kueri) | Degree Planning (10 Kueri) | Chitchat (10 Kueri) | Total Sukses (50 Kueri) |
|---|---|---|---|---|---|---|
| `qwen2.5:1.5b` | 1/10 (10%) | 0/10 (0%) | 4/10 (40%) | 2/10 (20%) | 10/10 (100%) | 17/50 (34.0%) |
| `qwen2.5:3b` | 10/10 (100%) | 7/10 (70%) | 10/10 (100%) | 9/10 (90%) | 10/10 (100%) | 46/50 (92.0%) |
| `llama3.2:3b` | 10/10 (100%) | 9/10 (90%) | 10/10 (100%) | 10/10 (100%) | 6/10 (60%) | 45/50 (90.0%) |
| `qwen2.5:7b` | 10/10 (100%) | 8/10 (80%) | 10/10 (100%) | 10/10 (100%) | 10/10 (100%) | 48/50 (96.0%) |

---

## 3. Analisis Ilmiah Trade-off Pareto & Pembahasan

### A. Dominasi Pareto dan Karakteristik Arsitektur
1. **Frontier Akurasi Tertinggi: `qwen2.5:7b` (Akurasi 96.00%, Eksekusi Sukses 96.00%)**
   - Model 7B mendominasi akurasi absolut dengan rasio kesalahan ekstraksi parameter terendah (**4.00%**). Ketaatan pada skema mencapai 98.00%.
   - Seluruh kueri SKS, Discovery, Degree Planning, dan Chitchat dijawab dengan ketaatan 100%.
   - **Trade-off:** Konsumsi VRAM mencapai **5346 MiB** (sekitar 65.6% dari total VRAM RTX 5050 8GB). Latensi prefill P50 TTFT berada pada **468.9 ms** dengan throughput 56.1 TPS, separuh dari kecepatan model 3B.

2. **Sweet Spot Efisiensi & Produksi: `qwen2.5:3b` (Akurasi 92.00%, Eksekusi Sukses 92.00%)**
   - Menghasilkan tingkat eksekusi fisik tertinggi di kelas model ringan (**92.00%**).
   - Memiliki ketaatan sempurna pada Chitchat (10/10, zero false positive tool-calling), SKS calculation (10/10), dan Course Discovery (10/10).
   - Throughput stabil pada **99.6 TPS** dengan P50 TTFT **156.5 ms** dan konsumsi VRAM sangat hemat (**2914 MiB**), menyisakan >5 GB VRAM untuk model embedding dan cache sistem.

3. **Frontier Latensi Terendah: `llama3.2:3b` (P50 TTFT 53.6 ms, Akurasi 90.00%)**
   - Llama-3.2 3B mencatat latensi Time-to-First-Token terendah (P50 TTFT **53.6 ms**).
   - Namun, Llama-3.2 3B menunjukkan kecenderungan *over-triggering tool* pada percakapan sapaan chitchat (hanya 6/10 chitchat yang berhasil diidentifikasi tanpa pemanggilan tool, 4 kueri memicu tool search_courses palsu).
   - Tingkat keberhasilan eksekusi fisik tool-nya lebih rendah (**84.00%**) karena beberapa argumen numerik diekstrak dalam bentuk string atau format non-standar.

4. **Kegagalan Fungsi Kritis pada Model 1.5B: `qwen2.5:1.5b` (Akurasi 34.00%, Error 66.00%)**
   - Model Qwen2.5 1.5B mengalami *catastrophic capability degradation* pada task function calling berstruktur.
   - Pada 66% kueri akademik, model 1.5B gagal memancarkan blok tool call dan malah menghasilkan teks halusinasi langsung dalam Bahasa Indonesia/Inggris tanpa data kurikulum yang valid.
   - Model 1.5B secara ilmiah **TIDAK LAYAK** digunakan sebagai generator utama ReAct Agent akademik.

---

## 4. Keputusan Rekayasa Sistem & Rekomendasi Arsitektur
1. **Mode Default Produksi (Recommended Deployment):**
   - Pertahankan **`qwen2.5:3b`** sebagai model default ReAct Agent untuk lingkungan interaktif mahasiswa karena memberikan rasio akurasi eksekusi tertinggi di kelasnya (92.00%), respon cepat (<160 ms TTFT, 99.6 TPS), dan memory footprint rendah (2.9 GB VRAM).
2. **Mode Penalaran Mendalam (High-Capacity Offline/Evaluation Mode):**
   - Gunakan **`qwen2.5:7b`** pada batch evaluation atau kueri perencanaan gelar kompleks karena mencapai akurasi 96.00% dan parameter extraction error terkecil (4.00%).
3. **Penyelarasan System-1 Decider Gating:**
   - Karena model generasi terkadang rentan *over-triggering* tool pada sapaan umum (terlihat pada Llama-3.2), pemisahan intent via **System-1 Decider Cosine Centroid Gating** (`src/engine/decider.py`) yang telah diuji pada Exp-011 dan Exp-015 adalah pertahanan wajib sebelum prompt masuk ke LLM.
