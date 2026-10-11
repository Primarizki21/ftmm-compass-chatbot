# FTMM COMPASS — AI Chatbot Backend

Backend service and evaluation engine for the **FTMM COMPASS** academic advising chatbot at Faculty of Advanced Technology and Multidisciplinary (FTMM), Universitas Airlangga.

This microservice provides high-precision academic advising, course discovery, prerequisite validation, and degree planning across all 5 FTMM study programs (TSD, TRKB, RN, TI, and TE). It interfaces with the frontend user interface at `ftmm-compass-ui`.

---

## Architecture Highlights

1. **Hybrid RAG-CAG (Retrieval-Augmented Generation + Cache-Augmented Generation)**:
   - **CAG (Cache-Augmented Generation)**: In-memory pre-warmed context for static faculty policies, SKS study load limits, graduation requirements, and institutional SOPs (0 ms retrieval latency).
   - **RAG (Retrieval-Augmented Generation)**: Dense FAISS vector indexing combined with BM25 Okapi and Reciprocal Rank Fusion (RRF) for dynamic, long-tail curriculum queries and syllabus exploration.
2. **System-1 Decider (Centroid Gating)**:
   - Fast cosine centroid gating (~30 ms) to route conversational queries, prune unnecessary tool schemas, and prevent LLM tool-calling hallucinations.
3. **Cascaded Defense-in-Depth Guardrails**:
   - **Tier 1 (0 ms)**: Deterministic regex filter blocking prompt injection, roleplay jailbreaks, and authority spoofing.
   - **Tier 2 (~30 ms)**: Semantic centroid gate deflecting out-of-domain / non-academic queries.
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

## Docker Deployment

To run as a containerized service with Docker Compose:

```bash
docker compose up --build -d
```

- API Base URL: `http://localhost:8000`
- Interactive Swagger docs: `http://localhost:8000/docs`
- Interactive ReDoc: `http://localhost:8000/redoc`

---

## API Endpoints

- `GET /api/health` — Service readiness and Ollama connectivity probe.
- `POST /api/chat` — Main conversation endpoint accepting `{ message, history, student_profile }` and returning structured responses, updated student profile, and action payloads (`DEGREE_PLAN` / `ANSWER`).

Full endpoint schemas and integration contracts are documented in [`docs/API_DOCUMENTATION.md`](docs/API_DOCUMENTATION.md).

---

## Testing & Quality Assurance

Run the test suite with `pytest`:

```bash
uv run pytest tests/
```

---

## Empirical Research & Experiment Ledger

Rigorous empirical benchmark logs, latency profiles, ablation studies (Exp-001 through Exp-011), and architecture decision records are tracked in [`docs/EXPERIMENT_LEDGER.md`](docs/EXPERIMENT_LEDGER.md) and individual experiment folders under [`docs/exp-*/`](docs/).
