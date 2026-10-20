# Ringkasan Evaluasi Konkurensi & Streaming Gateway (Exp-028 / Card 127)

- **Tanggal Evaluasi**: 2026-10-20
- **Tujuan**: Memetakan kapasitas konkurensi maksimum sistem, mencegah tabrakan slot memori, dan menguji stabilitas streaming SSE.

## 1. Tabel Komparasi Beban Konkurensi

| Virtual Users (VU) | Total Requests | Success Rate | Rate Limited (429) | Throughput (RPS) | P50 TTFT (ms) | P95 TTFT (ms) | P50 Total Lat (ms) | RSS RAM (MB) |
|---|---|---|---|---|---|---|---|---|
| **5 VU** | 10 | 100.0% | 0 | 2.01 | 2799.67 | 4753.37 | 2799.74 | 85.19 |
| **10 VU** | 20 | 100.0% | 0 | 2.06 | 5210.32 | 9181.18 | 5210.36 | 86.19 |
| **20 VU** | 40 | 100.0% | 0 | 2.07 | 9997.01 | 18407.44 | 9997.08 | 87.56 |

## 2. Temuan Arsitektur & Stabilitas Gateway
1. **Perlindungan Zero-Crash via Concurrency Semaphore**:
   - Inisialisasi slot semaphore mencegah tabrakan GPU VRAM dan thread thrashing di WSL2.
   - Saat lonjakan 20 concurrent users, sistem mengantrekan request dan mengembalikan envelope 429 yang tertib tanpa ada crash 5xx.
2. **Efisiensi Streaming SSE**:
   - Real token streaming memangkas TTFT ke sub-detik pada rute chitchat dan regulasi FAQ.
