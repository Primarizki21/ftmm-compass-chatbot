# Exp-031: Evaluasi Penalaran Berbasis Profil Mahasiswa (Cohort Dependency) & Ketahanan Multi-Turn Dialog Drift

## Ringkasan Proyek
Eksperimen riset komparatif untuk mengukur kemampuan sistem FTMM COMPASS dalam memberikan saran akademik yang dipersonalisasi berdasarkan kohor angkatan mahasiswa (2021 vs 2024), semester berjalan, batas SKS berdasarkan IPS, dan prasyarat kelulusan, serta ketahanan memori sesi dalam menangani pengalihan topik (*topic drift*) dan pemulihan konteks (*dialogue recovery*).

## Spesifikasi Runtime
- **Generator LLM:** `llama3.2:3b`
- **Embedding:** `nomic-embed-text-v2-moe`
- **Protokol Eksekusi:** Decoupled Two-Stage Batch Evaluation (WSL2 Hyper-V Zero Swapping)
- **Peak VRAM:** $\le 7.000\text{ MiB}$

## Hasil Evaluasi Utama
- **Cohort Reasoning Overall Accuracy:** **92.5%**
- **Profile-Dependent Verdict Accuracy:** **90.0%**
- **Topic Drift Refusal Rate (Turn 3):** **93.33%**
- **In-Domain Recovery Rate (Turn 4):** **100.0%**
- **Anaphora Resolution Rate:** **84.62%**
- **Session State Retention:** **100.0%**

## Berkas Artefak
- `benchmark_summary.md`: Rekapitulasi naratif dan tabel komparatif hasil evaluasi.
- `benchmark_metrics.csv`: Data mentah kuantitatif per nomor soal dan per putaran percakapan.
- `benchmark_results.json`: Rekapitulasi metrik terstruktur dalam format JSON.
- `benchmark_raw_traces.jsonl`: Jejak prompt, respons lengkap model, dan telemetri latensi per kueri.
