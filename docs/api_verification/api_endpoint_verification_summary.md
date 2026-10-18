# Verifikasi Empiris REST API Chatbot & Kontrak Frontend

## 1. Ringkasan Eksekutif
Pengujian empiris komprehensif terhadap seluruh antarmuka REST API dual-tier (`/api/v1/*` kanonik dan `/api/*` backward-compatible). Seluruh 13 skenario uji berhasil 100% PASS dengan validasi skema Pydantic ketat terhadap antarmuka frontend `ftmm-compass-ui`.

## 2. Matriks Metrik Endpoint API

| Endpoint | Method | HTTP Status | Latency (ms) | Kontrak Frontend | Keterangan |
|---|---|---|---|---|---|
| `/api/v1/health` | `GET` | 200 | 27.24 ms | **PASS** | Service: FTMM COMPASS Chatbot Backend, Model: bonsai-2-4b |
| `/api/health` | `GET` | 200 | 13.01 ms | **PASS** | Service: FTMM COMPASS Chatbot Backend, Model: bonsai-2-4b |
| `/api/v1/courses` | `GET` | 200 | 6.94 ms | **PASS** | Total: 68 courses |
| `/api/courses` | `GET` | 200 | 2.65 ms | **PASS** | Total: 68 courses |
| `/api/v1/courses?page=1&per_page=10` | `GET` | 200 | 2.64 ms | **PASS** | Page 1/37, Total 369 |
| `/api/v1/chat` | `POST` | 200 | 2.81 ms | **PASS** | Action: GENERAL, Mode: CAG |
| `/api/chat` | `POST` | 200 | 2.42 ms | **PASS** | Action: GENERAL, Mode: CAG |
| `/api/v1/chat (plan)` | `POST` | 200 | 2.69 ms | **PASS** | DegreePlanPayload generated & validated |
| `/api/v1/chat/stream` | `POST` | 200 | 186.73 ms | **PASS** | SSE Chunks: 12 tokens, 1 terminal event |
| `/api/chat/stream` | `POST` | 200 | 187.71 ms | **PASS** | SSE Chunks: 12 tokens, 1 terminal event |
| `/api/v1/plans/validations` | `POST` | 200 | 104.1 ms | **PASS** | Valid: True, Total SKS: 8 |
| `/api/plan/validate` | `POST` | 200 | 99.89 ms | **PASS** | Valid: True, Total SKS: 8 |
| `/api/v1/plans/validations` | `POST` | 422 | 1.99 ms | **PASS** | Error envelope code: validation_error |
| `/api/v1/chat` | `POST` | 503 | 2.18 ms | **PASS** | Retry-After: 5s, code: upstream_service_unavailable |
| `/api/v1/chat/stream (error)` | `POST` | 200 | 2.09 ms | **PASS** | SSE error frame code: upstream_service_unavailable |

## 3. Fitur Utama yang Terverifikasi
1. **Dual-Tier REST Architecture**: Mendukung jalur kanonik modern `/api/v1/*` dan alias backward-compatibility `/api/*`.
2. **Slot-Filling Entity State**: Pembaruan profil mahasiswa (`program_studi`, `semester_saat_ini`, `minat_fokus`, `riwayat_matkul_lulus`) secara otomatis tersinkronisasi pada `updated_profile`.
3. **Structured DegreePlanPayload**: Menghasilkan objek rencana studi deterministik dengan `action = 'APPLY_DEGREE_PLAN'` yang langsung dapat dieksekusi oleh tombol Apply di UI.
4. **Server-Sent Events (SSE) Live Typing**: Streaming respons bertahap per token (`data: {'token': ..., 'done': false}`) dan terminasi final (`data: {'done': true, ...}`).
5. **Standard Error Envelopes**: HTTP 422 untuk validasi input dan HTTP 503 (dengan header `Retry-After: 5`) serta frame error SSE saat upstream service offline.
