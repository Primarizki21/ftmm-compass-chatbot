# FTMM COMPASS — AI Chatbot Backend

Backend service and evaluation engine for the **FTMM COMPASS** academic advising chatbot at Faculty of Advanced Technology and Multidisciplinary (FTMM), Universitas Airlangga.

This microservice provides high-precision academic advising, course discovery, prerequisite validation, and degree planning across all 5 FTMM study programs (TSD, TRKB, RN, TI, and TE). It interfaces with the frontend user interface at `ftmm-compass-ui`.

---

## Architecture Highlights

1. **Hybrid RAG-CAG (Retrieval-Augmented Generation + Cache-Augmented Generation)**:
   - **CAG (Cache-Augmented Generation)**: In-memory pre-warmed context for static faculty policies, SKS study load limits, graduation requirements, and institutional SOPs (0 ms retrieval latency).
   - **RAG (Retrieval-Augmented Generation)**: Dense FAISS vector indexing combined with BM25 Okapi and Reciprocal Rank Fusion (RRF) for dynamic, long-tail curriculum queries and syllabus exploration.
2. **System-1 Decider (Single-Embed Shared Latent Pipeline & 6-Class BLAS Dispatch)**:
   - Fast cosine centroid gating (<0.1 ms matrix dot-product inference) using a single L2-normalized latent vector per turn to route conversational intents (`CHAT_ONLY`, `PREREQUISITE`, `SKS_CALCULATION`, `DEGREE_PLANNING`, `COURSE_DISCOVERY`, `REGULATION_SOP`), prune unnecessary tool schemas, and prevent LLM tool-calling hallucinations.
3. **Cascaded Defense-in-Depth Guardrails**:
   - **Tier 1 (0 ms)**: Deterministic Syntax Filter targeting low-level syntax attacks (SQL injection syntax, parameter tampering exploits) with zero semantic regex brittleness.
   - **Tier 2 (~30 ms)**: Semantic Dual-Centroid Gate utilizing calibrated continuous cosine distance against multi-cluster academic vs non-academic prototypes to deflect out-of-domain (OOD) queries and semantic adversarial jailbreaks.
   - **Tier 3**: NeMo Colang input rails for dialog safety and institutional policy compliance.
   - **Tier 4**: Deterministic physical parameter bounds check (legal SKS limits 0–24, semesters 1–8).

---

## Tech Stack

- **Runtime & Package Manager**: Python >= 3.12, [`uv`](https://github.com/astral-sh/uv)
- **Web Framework**: FastAPI, Uvicorn, Pydantic v2
- **Vector Search & Retrieval**: FAISS (`faiss-cpu`), Scikit-learn, NumPy
- **Primary Generator LLM**: Bonsai-2-4B Ternary (PQ2_0 / 1.58-bit) via `llama-server` CUDA (Port 8080)
- **Embedding & Fallback LLM**: Ollama daemon (`nomic-embed-text-v2-moe`, `qwen2.5:3b` fallback) (Port 11434)
- **Containerization**: Docker, Docker Compose

---

## Getting Started & Local Setup

Sistem ini menggunakan **Topologi 3-Service** untuk memisahkan beban kerja secara optimal:
1. **FastAPI Backend (`:8000`)**: Logika aplikasi, state machine, guardrails, routing, dan API REST/SSE.
2. **Ollama Daemon (`:11434`)**: Melayani model embedding dense (`nomic-embed-text-v2-moe`) dan model cadangan (fallback).
3. **llama-server / llama.cpp (`:8080`)**: Melayani model generator utama **Bonsai-2-4B Ternary (1.58-bit PQ2_0)** secara native C++ CUDA dengan throughput tinggi dan footprint RAM sangat ramping (<780 MB Host RAM, <2.2 GB VRAM).

---

### Prasyarat Sistem (Prerequisites)

1. **Sistem Operasi**: Linux (Ubuntu 22.04+) atau Windows WSL2.
2. **Hardware**: GPU NVIDIA disarankan (VRAM >= 4 GB untuk full GPU offload), atau CPU fallback.
3. **Python & Package Manager**: Python >= 3.12 dan [`uv`](https://github.com/astral-sh/uv).
   ```bash
   curl -LsSf https://astral.sh/uv/install.sh | sh
   ```
4. **Ollama**: Terpasang di sistem (`/usr/local/bin/ollama` atau `~/.local/bin/ollama`).

---

### Panduan Setup Langkah Demi Langkah (Step-by-Step)

#### Langkah 1: Clone Repositori & Install Dependencies
```bash
git clone <repo-url>
cd ftmm-compass-chatbot
uv sync
cp .env.example .env
```

---

#### Langkah 2: Setup llama.cpp (`llama-server`) & Model Bonsai-2-4B

Model utama yang digunakan adalah **Ternary-Bonsai-2-4B** (bobot terkuantisasi 1.58-bit `PQ2_0`), yang dijalankan via binary `llama-server` yang mendukung kernel ternary.

##### Opsi A: Menggunakan Direktori Sibling `../Bonsai-demo` (Paling Praktis)
Jika Anda sudah meng-clone atau menjalankan repositori `Bonsai-demo` di folder yang bersebelahan (`../Bonsai-demo`), skrip runner kami akan **otomatis mendeteksi** binary dan model di direktori tersebut tanpa perlu konfigurasi tambahan.

##### Opsi B: Setup Standalone di dalam Repositori Ini
Jika Anda menjalankan proyek ini secara mandiri (*standalone*), siapkan binary dan model di direktori proyek:

1. **Unduh Binary `llama-server` Pre-built (PrismML CUDA)**:
   ```bash
   mkdir -p bin
   # Contoh unduh binary release Linux x86_64 CUDA 12:
   curl -L https://github.com/PrismML-Eng/llama.cpp/releases/download/prism-b10777-e3de715/llama-prism-b10777-e3de715-bin-ubuntu-x64-cuda-cu12.4.tar.gz | tar -xz -C bin/
   chmod +x bin/llama-server
   ```

2. **Unduh Bobot Model GGUF (~1.14 GB)**:
   ```bash
   mkdir -p models
   # Menggunakan huggingface-cli via uv:
   uv run huggingface-cli download prism-ml/Ternary-Bonsai-2-4B-gguf Ternary-Bonsai-2-4B-PQ2_0.gguf --local-dir ./models
   ```
   *(File model akan berada di `models/Ternary-Bonsai-2-4B-PQ2_0.gguf`).*

3. **Jalankan Service `llama-server`**:
   ```bash
   bash scripts/start_bonsai_service.sh
   # Atau jalankan di background:
   nohup bash scripts/start_bonsai_service.sh > /tmp/bonsai_server.log 2>&1 &
   ```

##### Mengapa Parameter Ini Digunakan? (Technical Rationale)
Skrip `scripts/start_bonsai_service.sh` menjalankan `llama-server` dengan parameter yang sudah disetel khusus untuk stabilitas WSL2 & VRAM terbatas:

| Flag Parameter | Nilai | Penjelasan Teknis |
|---|---|---|
| `-m` | `...PQ2_0.gguf` | Menunjuk ke bobot model ternary 1.58-bit yang sangat efisien. |
| `--port` | `8080` | Port default OpenAI-compatible API untuk Bonsai. |
| `-c` | `4096` | Ukuran context window 4.096 token, pas untuk percakapan multi-turn & schema tools. |
| `-ngl` | `99` | Offload seluruh layer (36 layer) ke GPU CUDA untuk latensi inferensi maksimal (<50 ms/token). |
| `-np` | `1` | Mengunci alokasi ke 1 slot sekuensial. Mencegah duplikasi alokasi KV-cache di memori. |
| `--cache-ram 0` | `0` | Mematikan alokasi cache prompt di RAM Host. |
| `--no-cache-prompt` | Aktif | Mencegah `llama-server` melakukan penulisan berulang ke Host RAM, mengeliminasi risiko disk thrashing / swapping ke `.vhdx` di WSL2. |

##### Verifikasi Service `llama-server`:
```bash
# 1. Cek status kesehatan server
curl -s http://localhost:8080/health | jq .
# Output yang diharapkan: {"status":"ok"}

# 2. Tes inferensi sederhana (OpenAI format)
curl -s http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "bonsai-2-4b", "messages": [{"role": "user", "content": "Halo! Jawab singkat."}]}' | jq .choices[0].message.content
```

---

#### Langkah 3: Setup Ollama Daemon (Embedding & Fallback)

Ollama melayani representasi vektor semantik dan bertindak sebagai cadangan jika `llama-server` tidak aktif.

1. **Jalankan daemon Ollama**:
   ```bash
   ollama serve
   ```
2. **Unduh model embedding dan model cadangan**:
   ```bash
   # Model embedding dense resmi (768d / 256d Matryoshka):
   ollama pull nomic-embed-text-v2-moe

   # Model fallback generator (opsional, jika fallback diaktifkan):
   ollama pull qwen2.5:3b
   ```

---

#### Langkah 4: Jalankan FastAPI Chatbot Backend

Setelah `llama-server` (`:8080`) dan Ollama (`:11434`) aktif, jalankan aplikasi FastAPI:

```bash
uv run uvicorn src.api.server:app --host 0.0.0.0 --port 8000 --reload
```

##### Verifikasi Kesehatan Dual-Tier Backend:
Lakukan pemanggilan ke endpoint health check backend:
```bash
curl -s http://localhost:8000/api/v1/health | jq .
```
Pastikan respon menunjukkan status sehat:
```json
{
  "status": "healthy",
  "service": "FTMM COMPASS Chatbot Backend",
  "active_backend": "bonsai",
  "bonsai_online": true,
  "ollama_online": true,
  "generator_model": "bonsai-2-4b",
  "embedding_model": "nomic-embed-text-v2-moe"
}
```

---

### Troubleshooting & FAQ Setup

1. **Error: `llama-server: error while loading shared libraries: libcudart.so.12`**:
   - **Penyebab**: Binary CUDA tidak menemukan file shared library runtime CUDA di `LD_LIBRARY_PATH`.
   - **Solusi**: Skrip `start_bonsai_service.sh` otomatis mencari di `~/.local/lib/ollama/cuda_v12`, `/usr/local/cuda/lib64`, dan direktori binary. Jika CUDA Anda berada di path lain, tambahkan ke `.env`:
     ```bash
     CUDA_LIB_DIR=/usr/local/cuda/lib64
     ```

2. **RAM WSL2 Tiba-tiba Penuh / Windows Terasa Berat**:
   - **Penyebab**: WSL2 default mengizinkan alokasi hingga 50% RAM sistem dan rentan melakukan swap disk.
   - **Solusi**: Parameter `-np 1`, `--cache-ram 0`, dan `--no-cache-prompt` pada skrip runner telah dirancang untuk membatasi konsumsi RAM hingga ~780 MB. Pastikan juga membatasi memory WSL2 di file `~/.wslconfig` jika diperlukan (`memory=8GB`).

3. **Ingin Berjalan Tanpa `llama-server` (Murni Menggunakan Ollama)**:
   - Jika perangkat Anda tidak memiliki GPU untuk menjalankan `llama-server`, Anda dapat beralih sepenuhnya ke generator Ollama.
   - Cukup ubah konfigurasi di file `.env`:
     ```ini
     LLM_BACKEND=ollama
     LLM_MODEL_NAME=qwen2.5:3b
     ```
   - Sistem akan secara otomatis mengarahkan seluruh inferensi ReAct generator ke Ollama tanpa membutuhkan `llama-server`.

---
## Docker Deployment (Multi-Container & ftmm-network)

Layanan ini dirancang untuk berjalan sebagai kontainer terisolasi yang terhubung ke jaringan bersama `ftmm-network` agar dapat diakses oleh API Gateway dari `ftmm-compass-ui`:

### 1. Inisialisasi Jaringan Host (Sekali Saja di Server Baru)
```bash
# Buat shared bridge network jika belum ada
docker network create ftmm-network 2>/dev/null || true

# Jika Ollama berjalan di host dengan firewall UFW aktif (misal Ubuntu/WSL2):
sudo ufw allow 11434/tcp
```

### 2. Jalankan Kontainer
```bash
docker compose up --build -d
```

### 3. URL Layanan & Healthcheck
- **Endpoint Internal (dalam `ftmm-network`):** `http://ftmm-compass-chatbot:8000`
- **Endpoint Publik Host:** `http://localhost:8000`
- **Healthcheck Probe:** `curl -s http://localhost:8000/api/v1/health | jq .` (memverifikasi status kesehatan ganda: `bonsai_online: true` dan `ollama_online: true`)
- **Interactive Swagger Docs:** `http://localhost:8000/docs`
- **Interactive ReDoc:** `http://localhost:8000/redoc`

### 4. Skenario Server Baru (Alternative Full-Containerized Ollama)
Di server VPS/Cloud tanpa instalasi Ollama lokal di host, daemon Ollama dapat dijalankan langsung di Docker dan digabungkan ke `ftmm-network`:
```bash
docker run -d --gpus=all -v ollama_data:/root/.ollama -p 11434:11434 --network ftmm-network --name ollama ollama/ollama
```
*(Cukup ubah `OLLAMA_HOST=http://ollama:11434` di `.env` tanpa perlu membuka port host UFW).*

---

## API Endpoints (Dual-Tier REST Architecture)

Layanan mengekspos endpoint kanonik (`/api/v1/*`) dan alias kompatibilitas (`/api/*`):
- `GET /api/v1/health` : Probe kesehatan layanan dual-tier (konektivitas Bonsai port 8080 dan Ollama port 11434).
- `GET /api/v1/courses` — Katalog 369 mata kuliah kurikulum 5 prodi FTMM dengan paginasi adaptif (`page`, `per_page`).
- `GET /api/v1/courses/search` — Pencarian semantik dan kata kunci silabus via Hybrid FAISS + BM25.
- `POST /api/v1/chat` — Dialog akademik interaktif dengan Intent Decider & Hybrid RAG-CAG engine.
- `POST /api/v1/chat/stream` — Real-time Server-Sent Events (SSE) streaming token dialog.
- `POST /api/v1/plans/validations` — Validasi rencana studi terhadap graf prasyarat (DAG) & paritas semester.
- `POST /api/v1/plans` — Sintesis rencana studi deterministik 8 semester otomatis.

Spesifikasi lengkap schema, Pydantic contracts, dan panduan integrasi tersedia di [`docs/API_DOCUMENTATION.md`](docs/API_DOCUMENTATION.md).
---

## Testing & Quality Assurance

Run the test suite with `pytest`:

```bash
uv run pytest tests/
```

---

## Empirical Research & Experiment Ledger

Rigorous empirical benchmark logs, latency profiles, ablation studies (Exp-001 through Exp-019), and architecture decision records are tracked in [`docs/EXPERIMENT_LEDGER.md`](docs/EXPERIMENT_LEDGER.md) and individual experiment folders under [`docs/exp-*/`](docs/).
