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
- **LLM & Embeddings**: Ollama (`qwen2.5:3b`, `nomic-embed-text`)
- **Containerization**: Docker, Docker Compose

---

## Getting Started

### Prerequisites

1. **Python >= 3.12** and **uv** installed:
   ```bash
   curl -LsSf https://astral.sh/uv/install.sh | sh
   ```
2. **Ollama** running locally with required models:
   ```bash
   ollama pull qwen2.5:3b
   ollama pull nomic-embed-text
   ```

### Installation & Local Run

1. Clone repository and install dependencies:
   ```bash
   git clone <repo-url>
   cd ftmm-compass-chatbot
   uv sync
   ```

2. Configure environment:
   ```bash
   cp .env.example .env
   ```

3. Run the development server:
   ```bash
   uv run python main.py
   ```
   Server will start at `http://localhost:8000`.

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
- **Healthcheck Probe:** `curl -s http://localhost:8000/api/v1/health | jq .` (memverifikasi `status: healthy` dan `ollama_online: true`)
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
- `GET /api/v1/health` — Probe kesehatan layanan dan konektivitas daemon Ollama.
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
