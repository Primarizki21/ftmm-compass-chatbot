# [exp-022] Komparasi Model SLM Lokal & Kuantisasi (Sizing & Tool-Calling Benchmark)

## 1. Deskripsi & Latar Belakang Riset
Penelitian ini bertujuan untuk melakukan justifikasi ilmiah pemilihan arsitektur dan ukuran parameter Small Language Model (SLM) lokal untuk asisten akademik FTMM COMPASS pada perangkat keras laptop (*NVIDIA GeForce RTX 5050 Laptop GPU 8GB VRAM*).
Dalam arsitektur ReAct Agent, ketaatan pemanggilan alat (*tool-calling adherence*), kecepatan respon (*Time-to-First-Token* / TTFT), dan *throughput* (*Tokens per Second* / TPS) menjadi faktor penentu pengalaman pengguna mahasiswa dan keandalan sistemik penasihat akademik.

## 2. Kandidat Model & Konfigurasi Pengujian
Seluruh model dieksekusi secara lokal melalui daemon native C++ Ollama (v0.32.5) dengan kuantisasi standar Q4_K_M dan alokasi `num_ctx = 2048`, `temperature = 0.1`:
1. **`qwen2.5:1.5b` (Q4_K_M, ~986 MB)**: Edge ultra-lightweight architecture.
2. **`qwen2.5:3b` (Q4_K_M, ~1.9 GB)**: Model baseline produksi default saat ini.
3. **`llama3.2:3b` (Q4_K_M, ~2.0 GB)**: Pembanding arsitektur keluarga Llama pada skala 3B.
4. **`qwen2.5:7b` (Q4_K_M, ~4.7 GB)**: Model kapasitas penalaran tinggi untuk evaluasi batas atas GPU 8GB.

## 3. Dataset Uji Terstratifikasi
Evaluasi menggunakan 50 kueri uji akademik terstratifikasi (`data/ground_truth/slm_sizing_eval_dataset.json`) yang mencakup 5 domain fungsional:
1. `SKS_CALCULATION` (10 kueri): Kalkulasi batas maksimal SKS berdasarkan IPS mahasiswa (Pasal 5.4 Panduan Akademik FTMM).
2. `PREREQUISITE_DAG` (10 kueri): Pengecekan prasyarat berarah kurikulum resmi FTMM.
3. `COURSE_DISCOVERY` (10 kueri): Pencarian katalog semester deterministik dan silabus topik.
4. `DEGREE_PLANNING` (10 kueri): Sintesis perencanaan studi semester 1–8 untuk 5 program studi FTMM (TSD, TRKB, RN, TI, TE).
5. `CHITCHAT` (10 kueri): Sapaan dan percakapan umum tanpa pemanggilan tool (*zero false positive guard*).

## 4. Matriks Metrik Terukur
1. **Tool-Calling Adherence**:
   - *Tool Selection Accuracy (%):* Persentase kueri di mana model memilih fungsi tool yang tepat (atau tidak memanggil tool untuk chitchat).
   - *JSON Schema Compliance (%):* Persentase pemanggilan tool dengan struktur parameter valid JSON sesuai spesifikasi fungsi OpenAI format.
   - *Parameter Extraction Error Rate (%):* Tingkat kegagalan ekstraksi parameter (parameter hilang, tipe data salah, atau nilai IPS/semester meleset).
   - *Physical Execution Success Rate (%):* Persentase tool call yang berhasil dieksekusi secara nyata oleh runtime Python (`TOOL_REGISTRY`).
2. **Hardware & Inferensi**:
   - *Time-to-First-Token (TTFT, ms):* Diukur pada level hardware prompt prefill (`prompt_eval_duration`) dan persepsi klien streaming.
   - *Throughput (TPS):* Kecepatan pembangkitan token generasi per detik.
   - *Peak GPU VRAM Footprint (MiB):* Alokasi memori fisik GPU yang tercatat via `nvidia-smi`.
3. **Analisis Multi-Objective Pareto**:
   - Menghitung efisiensi kompromi antara Akurasi Adherence, Latensi TTFT, dan Konsumsi VRAM pada batasan 8GB VRAM.

## 5. Artefak Eksperimen
- `benchmark_metrics.csv`: Data kuantitatif lengkap per model.
- `benchmark_results.json`: Struktur data JSON lengkap metrik dan breakdown per kategori.
- `benchmark_raw_traces.jsonl`: 200 jejak rekaman inferensi empiris (50 kueri x 4 model).
- `benchmark_summary.md`: Laporan komparasi mendalam dan ringkasan temuan kunci.
