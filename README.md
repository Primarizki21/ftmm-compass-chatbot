# FTMM COMPASS — AI Chatbot Backend

Backend service dan evaluation engine untuk academic advising chatbot **FTMM COMPASS** di Fakultas Teknologi Maju dan Multidisiplin (FTMM), Universitas Airlangga.

Microservice ini menyediakan layanan konsultasi akademik cerdas, pencarian mata kuliah kurikulum 5 program studi (TSD, TRKB, RN, TI, TE), validasi prasyarat (*prerequisite DAG*), dan sintesis rencana studi otomatis (*degree planner*). Layanan ini terhubung langsung ke antarmuka frontend di repositori `ftmm-compass-ui`.

---

## Arsitektur Inti (Core Architecture)

1. **Hybrid RAG-CAG (Retrieval-Augmented Generation + Cache-Augmented Generation)**:
   - **CAG (Cache-Augmented Generation)**: In-memory pre-warmed context untuk regulasi statis fakultas, batas beban SKS berdasarkan IPS, syarat kelulusan, dan SOP akademik (latensi retrieval 0 ms).
   - **RAG (Retrieval-Augmented Generation)**: Dense FAISS vector indexing dipadukan dengan BM25 Okapi dan Reciprocal Rank Fusion (RRF) untuk penelusuran silabus mata kuliah dan prasyarat kurikulum yang dinamis.
2. **System-1 Decider (Single-Embed Shared Latent Pipeline & 6-Class BLAS Dispatch)**:
   - Gating berbasis centroid semantik kontinu (<0.1 ms dot-product matrix) menggunakan 1 vektor laten per kueri untuk mengklasifikasikan intent (`CHAT_ONLY`, `PREREQUISITE`, `SKS_CALCULATION`, `DEGREE_PLANNING`, `COURSE_DISCOVERY`, `REGULATION_SOP`), memangkas skema tools yang tidak relevan, dan mengeliminasi halusinasi pemanggilan fungsi.
3. **Cascaded Defense-in-Depth Guardrails**:
   - **Tier 1 (0 ms)**: Deterministic Syntax Sanitizer untuk menangkal injeksi SQL diskrit dan tampering parameter kurikulum tanpa kerapuhan regex semantik.
   - **Tier 2 (~30 ms)**: Semantic Dual-Centroid Gate dengan kalibrasi margin cosine similarity untuk menolak kueri di luar lingkup akademik (OOD) dan adversarial prompt injection.
   - **Tier 3**: NeMo Colang safety flow untuk kepatuhan dialog institusi.
   - **Tier 4**: Physical Parameter Rail untuk memvalidasi batas legal argumen (semester 1–8, batas SKS 0–24).

---

## Tech Stack

- **Runtime & Package Manager**: Python >= 3.12, [`uv`](https://github.com/astral-sh/uv) (atau standar Python `venv` + `pip`)
- **Web Framework**: FastAPI, Uvicorn, Pydantic v2
- **Vector Search & Retrieval**: FAISS (`faiss-cpu`), Scikit-learn, NumPy
- **Generator LLM Produksi**: Ollama native `llama3.2:3b` (Port 11434)
- **Embedding Model Dense**: Ollama native `nomic-embed-text-v2-moe` (768 dimensi MRL)
- **Containerization**: Docker, Docker Compose

---

## Panduan Instalasi & Menjalankan (Cross-Platform Setup)

Sistem ini dirancang menggunakan **Topologi 2-Service Ringkas (FastAPI + Ollama)** yang dapat berjalan di **Windows, macOS (Apple Silicon M-Series / Intel), dan Linux / WSL2** tanpa memerlukan GPU NVIDIA khusus dan tanpa setup driver CUDA yang rumit.

```
+---------------------------+       HTTP REST / SSE       +-------------------------------+
|     FastAPI Backend       | <=========================> |         Ollama Daemon         |
|   `http://localhost:8000` |                             |    `http://localhost:11434`   |
| (Logika, Routing, FAISS)  |                             | (Llama 3.2 3B + Nomic Embed)  |
+---------------------------+                             +-------------------------------+
```

---

### Prasyarat Sistem (Prerequisites)

1. **Python**: Versi >= 3.12 terpasang di sistem.
2. **Ollama**: Aplikasi runtime model lokal.

---

### Langkah 1: Pasang Ollama & Unduh Model (Semua OS)

1. Unduh dan pasang Ollama dari situs resmi: **[https://ollama.com/download](https://ollama.com/download)**
   - **Windows**: Jalankan `OllamaSetup.exe`.
   - **macOS**: Unduh dan jalankan aplikasi Ollama (*otomatis mendukung akselerasi Apple Silicon Metal GPU*).
   - **Linux / WSL2**: Jalankan perintah `curl -fsSL https://ollama.com/install.sh | sh`.
2. Pastikan Ollama berjalan, lalu buka Terminal / Command Prompt / PowerShell dan unduh model yang dibutuhkan:
   ```bash
   # 1. Model generator utama (Llama 3.2 3B - ~2.0 GB)
   ollama pull llama3.2:3b

   # 2. Model embedding kurikulum (Nomic Embed Text v2 MoE - ~280 MB)
   ollama pull nomic-embed-text-v2-moe
   ```

---

### Langkah 2: Clone Repositori & Setup Konfigurasi (.env)

Buka terminal pada direktori kerja Anda:

```bash
git clone https://github.com/FTMM-COMPASS/ftmm-compass-chatbot.git
cd ftmm-compass-chatbot
```

Salin berkas konfigurasi lingkungan:
- **macOS / Linux / WSL2**:
  ```bash
  cp .env.example .env
  ```
- **Windows (PowerShell)**:
  ```powershell
  Copy-Item .env.example .env
  ```
- **Windows (Command Prompt / CMD)**:
  ```cmd
  copy .env.example .env
  ```

> **Catatan**: Konfigurasi default di `.env.example` sudah disetel langsung untuk Ollama native (`LLM_BACKEND=ollama`, `LLM_MODEL_NAME=llama3.2:3b`, `EMBED_MODEL_NAME=nomic-embed-text-v2-moe`). Anda tidak perlu mengubah isi file `.env`.

---

### Langkah 3: Install Dependensi Python

Pilih salah satu metode instalasi berikut:

#### Opsi A: Menggunakan `uv` (Sangat Cepat & Direkomendasikan)

Jika Anda ingin instalasi dependensi selesai dalam hitungan detik:

- **Windows (PowerShell)**:
  ```powershell
  powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
  uv sync
  ```
- **macOS (Terminal)**:
  ```bash
  brew install uv   # atau: curl -LsSf https://astral.sh/uv/install.sh | sh
  uv sync
  ```
- **Linux / WSL2 (Bash)**:
  ```bash
  curl -LsSf https://astral.sh/uv/install.sh | sh
  uv sync
  ```

#### Opsi B: Menggunakan Standar Python `venv` + `pip` (Tanpa Tool Tambahan)

Jika Anda tidak ingin memasang `uv`, Anda bisa menggunakan bawaan Python:

- **Windows (PowerShell)**:
  ```powershell
  # 1. Buat virtual environment
  python -m venv .venv

  # 2. Aktifkan virtual environment
  .venv\Scripts\Activate.ps1

  # 3. Pasang paket proyek
  pip install -e .
  ```
  *(Jika muncul error execution policy di PowerShell, jalankan: `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass` lalu ulangi aktivasi).*

- **macOS / Linux / WSL2 (Terminal)**:
  ```bash
  # 1. Buat virtual environment
  python3 -m venv .venv

  # 2. Aktifkan virtual environment
  source .venv/bin/activate

  # 3. Pasang paket proyek
  pip install -e .
  ```

---

### Langkah 4: Jalankan Backend FastAPI

Pastikan daemon Ollama aktif di latar belakang, lalu jalankan server:

- **Jika menggunakan `uv`**:
  ```bash
  uv run uvicorn src.api.server:app --host 0.0.0.0 --port 8000 --reload
  ```
- **Jika menggunakan virtual environment standar**:
  ```bash
  python -m uvicorn src.api.server:app --host 0.0.0.0 --port 8000 --reload
  ```

Backend kini aktif dan siap melayani permintaan di `http://localhost:8000`.

---

### Langkah 5: Verifikasi Status Layanan (Health Check)

Buka terminal baru untuk memverifikasi konektivitas dual-service:

- **macOS / Linux / WSL2**:
  ```bash
  curl -s http://localhost:8000/api/v1/health | jq .
  ```
- **Windows (PowerShell)**:
  ```powershell
  Invoke-RestMethod -Uri http://localhost:8000/api/v1/health | ConvertTo-Json
  ```

Contoh respon sukses:
```json
{
  "status": "healthy",
  "service": "FTMM COMPASS Chatbot Backend",
  "active_backend": "ollama",
  "ollama_online": true,
  "generator_model": "llama3.2:3b",
  "embedding_model": "nomic-embed-text-v2-moe"
}
```

---

## Pengujian & Regresi (Testing)

Jalankan test suite untuk memastikan seluruh logika guardrail, retrieval, planning, dan healthcheck berfungsi 100%:

- **Menggunakan `uv`**:
  ```bash
  uv run pytest tests/
  ```
- **Menggunakan virtual environment standar**:
  ```bash
  pytest tests/
  ```

---

## Docker Deployment (Opsional)

Jika Anda ingin menjalankan layanan secara terisolasi via Docker:

### 1. Mode Development di Linux / WSL2 (`network_mode: host`)
Pada Linux / WSL2, container mengakses daemon Ollama di host melalui loopback `127.0.0.1`:
```bash
docker compose up --build -d
```

### 2. Mode Produksi Multi-Container (`ftmm-network`)
Untuk deployment server di mana Ollama dan FastAPI sama-sama berjalan di dalam Docker:
```bash
docker network create ftmm-network 2>/dev/null || true
docker run -d --gpus=all -v ollama_data:/root/.ollama -p 11434:11434 --network ftmm-network --name ollama ollama/ollama
```

---

## Daftar Endpoint REST API (Dual-Tier Kanonik)

Layanan menyediakan endpoint RESTful kanonik (`/api/v1/*`) dan alias kompatibilitas (`/api/*`):

| HTTP Method | Endpoint | Deskripsi Fungsi |
|---|---|---|
| `GET` | `/api/v1/health` | Status probe kesehatan microservice dan konektivitas Ollama. |
| `GET` | `/api/v1/courses` | Katalog kurikulum 369 mata kuliah 5 prodi FTMM dengan filter paginasi. |
| `GET` | `/api/v1/courses/search` | Pencarian cerdas silabus berbasis FAISS Dense + BM25 RRF. |
| `POST` | `/api/v1/chat` | Endpoint percakapan akademik interaktif multi-turn. |
| `POST` | `/api/v1/chat/stream` | Server-Sent Events (SSE) streaming respons token langsung. |
| `POST` | `/api/v1/plans/validations` | Validasi aturan prasyarat kurikulum (DAG), beban batas SKS, dan paritas semester. |
| `POST` | `/api/v1/plans` | Sintesis rencana studi deterministik 144 SKS (8 semester) berbasis Neuro-Symbolic CSP. |

Dokumentasi skema request/response JSON selengkapnya dapat dilihat di [`docs/API_DOCUMENTATION.md`](docs/API_DOCUMENTATION.md).

---

## Catatan Deprecasi Model (Bonsai 1.58-bit Ternary)

Model eksperimental **Ternary-Bonsai-2-4B (1.58-bit PQ2_0)** via `llama-server` CUDA C++ sebelumnya digunakan pada fase ablasi riset (Exp-019 dan Exp-025). 

Per 21 Oktober 2026 (Exp-031), generator backend resmi **distandardisasi ke `llama3.2:3b` via Ollama native**:
1. **Zero-Friction Cross-Platform**: Dapat berjalan langsung di macOS (Apple Silicon Metal), Windows (CPU/GPU), dan Linux tanpa kompilasi C++ binary atau ketergantungan driver CUDA `libcudart.so.12`.
2. **Stabilitas Memori WSL2**: Mengeliminasi risiko disk thrashing dan swapping memori ke `.vhdx` pada lingkungan terbatas.
3. **Kinerja Penalaran Profil Tinggi**: Mencatatkan akurasi penalaran kohor (*cohort reasoning*) sebesar **92.50%** dan pemulihan konteks multi-turn sebesar **100.0%**.

Kode adapter untuk backend alternatif tetap tersimpan secara modular di `src/engine/llm_client.py` untuk kebutuhan riset akademik lanjutan.

---

## Buku Induk Eksperimen & Portofolio Riset

Seluruh riwayat eksperimen, metrik kuantitatif, log latensi, dan studi ablasi ilmiah (Exp-001 hingga Exp-031) didokumentasikan secara rinci di:
- **Buku Induk Eksperimen**: [`docs/EXPERIMENT_LEDGER.md`](docs/EXPERIMENT_LEDGER.md)
- **Folder Laporan Eksperimen**: [`docs/exp-*/`](docs/)
