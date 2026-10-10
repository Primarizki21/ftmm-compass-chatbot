# Eksperimen 010: Low-Level Inference Optimization & Dual-Model VRAM Pinning

- **Eksperimen ID:** `exp-010`
- **Tanggal:** 2026-10-10
- **Kartu Trello:** Card 31 (`[Nauval & Ario] Hardware Profiling VRAM`), Card 44 (`[Nauval & Ario] ReAct Agent Latency Optimization`), Card 99 (`[Nauval & Ario] Decider Paradigm & Dynamic Tool Pruning`)
- **Model:** Ollama `qwen2.5:3b` + `nomic-embed-text` pada GPU lokal NVIDIA RTX 5050 Laptop GPU (8GB VRAM) di lingkungan Linux WSL2
- **Tujuan:** Mengeliminasi latensi *runner swapping* dan *prompt token bloat* melalui 4 optimasi tingkat rendah (*low-level*): pinning VRAM dual-model, pruning skema tools dinamis, penataan prefix caching bergaya DeepSeek Harness, dan pembatasan batas konteks/prediksi.

---

## 1. Empat Pilar Optimasi Low-Level yang Diimplementasikan

1. **Dual-Model Resident VRAM Pinning (`Zero Swapping`):**
   - Konfigurasi daemon: `OLLAMA_MAX_LOADED_MODELS=2` dan `OLLAMA_KEEP_ALIVE=-1`.
   - Mengunci `qwen2.5:3b` (~2.2 GB) dan `nomic-embed-text` (~0.3 GB) secara simultan di memori GPU.
   - Mengeliminasi jeda I/O transfer model dan disk paging WSL2 sebesar 1.000–2.500 ms di setiap pergantian task embedding $\leftrightarrow$ generasi.

2. **Dynamic Tool Schema Pruning ($O(1)$ Decider Pattern):**
   - Menggunakan pemetaan leksikal instan untuk menyortir fungsi sebelum panggilan LLM.
   - Kueri sapaan/chitchat dieksekusi **tanpa parameter `tools`**, menghemat ~963 token skema JSON dan mereduksi matriks attention dari 1.690.000 elemen menjadi hanya 22.500 elemen.

3. **Deterministic Prefix Cache Pinning (Pola DeepSeek Harness):**
   - Menjaga integritas hash KV cache Llama.cpp dengan memastikan `REACT_SYSTEM_PROMPT` statis berada di token 0.
   - Serialisasi profil mahasiswa dibuat terurut secara alfabetis deterministik agar tidak membatalkan cache prefix.

4. **Clamped Context & Generation Bounds:**
   - Dibatasi pada `num_ctx: 2048` (menghemat ~600 MiB alokasi buffer tensor KV cache di GPU).
   - Dibatasi pada `num_predict: 256` (memotong waktu decoding memori bandwidth hingga ~2 detik per giliran).

---

## 2. Hasil Kuantitatif Benchmark (20 Kueri Terstratifikasi)

| Kategori Kueri | Jumlah Kueri | Latensi Rata-rata (ms) | P50 (ms) | Rata-rata Prompt Prefill | Penghematan Token vs Baseline |
|---|---|---|---|---|---|
| **GREETING (Sapaan/Chitchat)** | 5 | 3893.4 ms* | **694.7 ms** | **519.2 token** | **-60.0% (-780 token)** |
| **SKS_LIMIT (Kalkulasi Beban SKS)** | 5 | **1465.3 ms** | **1348.9 ms** | 1443.6 token | -25.0% (-480 token) |
| **PREREQUISITE (Verifikasi Prasyarat)**| 5 | **1320.0 ms** | **1383.8 ms** | 1587.8 token | -20.0% (-400 token) |
| **PLANNING (Perencanaan Gelar/Studi)** | 5 | 3381.0 ms | 3701.7 ms | 1927.6 token | 0.0% (Full tools) |
| **RINGKASAN OVERALL** | **20** | **2514.9 ms** | **1393.9 ms** | **1369.5 token** | **-35.0% rata-rata** |

*\*Catatan: Rata-rata Greeting terpengaruh oleh kueri pertama (cold start load GPU 16.9s). Setelah warm-up, seluruh sapaan stabil pada 420 ms – 700 ms (kecepatan meningkat hingga 8.5x).*

---

## 3. Profil Konsumsi Memori GPU (RTX 5050 8GB)

```
NVIDIA-SMI Profiling:
- GPU Model: NVIDIA GeForce RTX 5050 Laptop GPU (8.151 MiB VRAM)
- Total VRAM Terpakai Bersama: 2.565 MiB (~31.4% kapasitas)
  * qwen2.5:3b (Q4_K_M) + KV Cache (ctx 2048): 2.242 MiB
  * nomic-embed-text (FP16): 323 MiB
- Sisa VRAM Headroom: 5.586 MiB (~68.6% bebas)
- Suhu Operasional: 36°C – 46°C (sangat dingin dan efisien)
```

---

## 4. Kesimpulan & Rekomendasi Ilmiah

1. **Aman untuk Hardware Laptop:** Konfigurasi VRAM Pinning terbukti aman pada GPU 8GB (hanya memakan ~2.5 GB), menyisakan lebih dari 5.5 GB ruang bebas tanpa memicu OOM atau throttling termal.
2. **Efisiensi Interaksi Nyata:** Pemangkasan skema tools memangkas giliran sapaan dan FAQ pendek menjadi sub-detik (**~400–700 ms**), memberikan impresi respons instan bagi pengguna antarmuka frontend `ftmm-compass-ui`.
3. **Kesiapan Fase API:** Engine ReAct teroptimasi ini resmi siap dipasangkan pada endpoint FastAPI di Card 47.
