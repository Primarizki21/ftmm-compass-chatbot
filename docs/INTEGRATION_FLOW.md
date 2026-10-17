# FTMM COMPASS — Arsitektur & Alur Integrasi End-to-End

> **Klasifikasi:** Dokumen Arsitektur Sistem & Spesifikasi Antarmuka (Tracked in Git).  
> **Aset Visual draw.io:** `docs/architecture/ftmm_compass_integration_flow.drawio.xml` (Bisa dibuka & diedit di [app.diagrams.net](https://app.diagrams.net/) atau VS Code Draw.io extension).  
> **Kartu Trello:** Kolom `📌 Project Hub & Quick Links` ➔ `📐 Arsitektur & Alur Integrasi End-to-End (draw.io)`.

---

## 1. Ringkasan Arsitektur Sistem Multi-Tier

Sistem **FTMM COMPASS** dirancang dengan pemisahan tanggung jawab yang tegas antar 4 layer (Tier):

```mermaid
graph LR
    Client[Mahasiswa / Browser] <--> FE[Frontend React<br/>Hans: ftmm-compass-ui]
    FE <--> GW[API Gateway Backend<br/>Mantra: FastAPI /api/v1]
    GW <--> AI[AI Engine Microservice<br/>Nauval & Ario: port 8000]
    GW <--> DB[(PostgreSQL Database<br/>Darryl: schema_v2_1_rc1)]
    AI <--> LocalEngine[Ternary Bonsai-2-4B + Nomic Embed v2 MoE]
    AI <--> VectorStore[(FAISS IndexFlatIP + BM25)]
```

| Layer | Komponen & Repository | Penanggung Jawab | Peran Utama |
|---|---|---|---|
| **Tier 1: Client / UI** | `ftmm-compass-ui/src/pages/Chatbot.tsx`<br>`DegreePlanner.tsx` | Hans (Frontend) | Antarmuka obrolan responsif, konsumsi SSE live typing, visualisasi roadmap semester 1–8. |
| **Tier 2: API Gateway** | `ftmm-compass-ui/backend` / FastAPI Core | Mantra (Backend) | Routing reverse proxy, validasi JWT Auth, rate limiting `slowapi`, SSE streaming relay (`/api/v1/chat/stream`), dan logging sesi via `ftmm-network`. |
| **Tier 3: AI Microservice** | `ftmm-compass-chatbot/src/` | Nauval & Ario (AI/NLP) | System-1 intent routing, Hybrid RAG-CAG, ReAct agent dengan deterministic tools, SSE streaming, validasi DAG (`POST /api/v1/plans/validations`), generate rencana studi (`POST /api/v1/plans`), dan Hybrid search (`GET /api/v1/courses/search`). |
| **Tier 4: Relational DB** | PostgreSQL 16 (`schema_v2_1_rc1.sql`) | Darryl (Database) | Relasi prasyarat matkul (`course_prerequisites` DAG), paritas ganjil/genap, dan persistensi rencana studi. |


### 1.1 Pipeline Internal AI Engine Microservice (Hasil Riset Exp-008 s.d. Exp-025)

AI Engine mengimplementasikan arsitektur *Cascaded Defense-in-Depth* dan *Single-Embed Shared Latent Pipeline*:

```mermaid
graph TD
    Query[Kueri Mahasiswa] --> T1[Tier 1: Deterministic Syntax Filter 0 ms<br/>SQL Injection & Parameter Tampering]
    T1 -- Blocked: Syntax Attack / Exploit --> RefusalT1[Refusal Keamanan Sintaksis: 0 ms]
    T1 -- Allowed --> Embed[Single-Embed Step: Ollama C++ Native<br/>1x Call -> L2-Normalized Shared Latent Vector]
    Embed --> T2[Tier 2: Semantic Dual-Centroid Gating ~30 ms<br/>OOD Detection & Adversarial Jailbreak Boundary]
    T2 -- Blocked: Out-of-Scope Non-Akademik / Jailbreak --> RefusalT2[Refusal Batasan Topik & Etika: ~30 ms]
    T2 -- Allowed --> Mem[Conversation Memory Manager<br/>Dense Gazetteer Linking & Latent Ellipsis Context]
    Mem --> Decider[6-Class System-1 Decider: BLAS Dot-Product <0.1 ms<br/>Shared Latent Vector Intent & Tool Dispatch]
    Decider -- CHAT_ONLY --> DirectLLM[ReAct Agent: Zero Tool Schema Overhead ~400-700 ms]
    Decider -- REGULATION_SOP --> CAGRoute[Direct CAG Route: Pre-cached Faculty SOP 0 ms Retrieval]
    Decider -- PREREQUISITE / SKS / PLAN / DISCOVERY --> ToolsLLM[ReAct Agent: Injeksi Tool Spesifik Saja]
    DirectLLM --> Out[Jawaban Akhir Teks]
    CAGRoute --> Out
    ToolsLLM --> T4[Tier 4: Deterministic Argument & Physical Tool Rail]
    T4 -- Parameter Valid --> Exec[Eksekusi Tool Database / DAG / Kurikulum]
    T4 -- Parameter Melanggar SKS/Semester --> RefusalT4[Refusal Parameter Pelanggaran Aturan]
    Exec --> Gen[Kompilasi Jawaban Akhir & DegreePlanPayload JSON]
    Gen --> Out
```

| Komponen Pipeline | Modul Sumber | Mekanisme & Target Latensi | Status Validasi |
|---|---|---|---|
| **Tier 1 Syntax Filter** | `src/guardrails/tier1_filter.py` | Deterministic Syntax Filter (0.0 ms) untuk cegah SQL syntax injection (`DROP TABLE`, `UNION SELECT`) & parameter tampering numerik. Zero semantic regex. | Tuntas (Exp-008, Exp-019, 0% ASR) |
| **Tier 2 Dual-Centroid Gating** | `src/guardrails/tier2_semantic_gate.py` | Semantic Dual-Centroid Gating (<35 ms) menggunakan L2-normalized shared latent vector untuk deteksi OOD, jailbreak adversarial, dan task-intent disambiguation sains vs homework solver. | Tuntas (Exp-016, Exp-018, Exp-019, 100% OOD-R, 100% Disambiguation) |
| **Memory Manager** | `src/engine/memory.py` | Dense Gazetteer Linking (369 kurikulum FTMM) + Latent Ellipsis & Structured Context Injection (prodi, semester, IPS, minat). | Tuntas (Exp-009, Exp-019, $\ge$95% Anaphora Success) |
| **System-1 Decider** | `src/engine/decider.py` | 6-Class System-1 Decider BLAS Dot-product (<0.1 ms inferensi matrix): `CHAT_ONLY`, `PREREQUISITE`, `SKS_CALCULATION`, `DEGREE_PLANNING`, `COURSE_DISCOVERY`, `REGULATION_SOP`. Memangkas token skema JSON dan redundansi embedding call. | Tuntas (Exp-011 96.7% Acc, Exp-015 95.0% Acc, Exp-017 Qwen3+kNN 100.0% Acc, Exp-019 Single-Embed Pipeline) |
| **ReAct Agent Engine** | `src/engine/react_agent.py` | Qwen2.5-3B VRAM Pinned (`OLLAMA_MAX_LOADED_MODELS=2`) + DeepSeek-style prefix cache. | Tuntas (Exp-010, P50 1.39s) |
| **CSP Degree Planner** | `src/engine/tools/academic_tools.py` | Neuro-Symbolic CSP Planner (`generate_degree_plan`) dengan Dense Gazetteer Normalizer, FAISS focus track expansion, penegakan paritas semester, batas SKS, dan dedup agama tunggal. | Tuntas (Exp-020, 100.0% Success, 0% Fallback) |
| **Tier 4 Argument Rail** | `src/guardrails/tier4_tool_rail.py` | Validasi batas legal semester (1–8) dan SKS (0–24) sebelum pemanggilan tool fisik. | Tuntas (Exp-008) |
| **Embedding Backbone** | `src/knowledge/retriever.py` & `src/engine/decider.py` | Two-Tier Native Embedding Benchmark: High-Capacity Mode `qwen3-embedding:0.6b` (1024d, Hit@3 87.25%, Router 98.33%) & Resource-Constrained Mode `nomic-embed-text-v2-moe` (256d Matryoshka, Hit@3 83.89%, Router 96.67%, Margin 0.1316, Peak VRAM 3.6 GB). | Tuntas (Exp-021) |
| **Academic Normalizer** | `src/engine/normalizer.py` | Pre-NLU Normalizer: Kamus leksikal O(1), Bounded Damerau-Levenshtein ($d \le 2$), dan penanganan noise fonetik STT. | Tuntas (Exp-023, F1 0.9333, Error Drop -58.33%, Latency 2.12 ms) |

## 2. Sequence Diagram Alur Integrasi End-to-End

```mermaid
sequenceDiagram
    autonumber
    actor User as Mahasiswa (Browser)
    participant FE as Frontend React (Hans: Chatbot.tsx)
    participant GW as API Gateway Backend (Mantra: /api/v1)
    participant AI as AI Engine (Nauval & Ario: port 8000)
    participant DB as PostgreSQL Database (Darryl)

    Note over User, AI: FASE 1: KONSULTASI & STREAMING PERCAKAPAN (SSE)
    User->>FE: Input pertanyaan ("Rekomendasikan matkul semester 3 TSD fokus ML")
    FE->>GW: POST /api/v1/chat/stream (Payload: message, history, student_profile)
    GW->>GW: Validasi JWT & Rate Limit (slowapi max 20 req/menit)
    GW->>AI: Proxy forward ke AI Engine (POST /api/chat/stream)
    AI->>AI: System-1 Intent Router -> ReAct Agent tool-calling (search_courses, calculate_sks_limit)
    loop Asynchronous Token Stream
        AI-->>GW: Yield event: token (kata-per-kata)
        GW-->>FE: SSE event: token (live typing effect)
        FE-->>User: Teks muncul mengalir huruf demi huruf
    end
    AI-->>GW: event: profile_update (prodi, semester, minat)
    GW-->>FE: Update badge profil mahasiswa di header UI
    AI-->>GW: event: plan (JSON DegreePlanPayload action: APPLY_DEGREE_PLAN)
    GW-->>FE: Render Widget Kartu "DegreePlanCardWidget"
    AI-->>GW: event: done (latency_ms, routing_mode)
    GW-->>FE: Selesai stream (koneksi tertutup rapi)

    Note over User, DB: FASE 2: PENERAPAN KE DEGREE PLANNER & VALIDASI DATABASE
    User->>FE: Klik tombol [ Terapkan ke Degree Planner ] pada kartu
    FE->>GW: POST /api/v1/planner/validate (Payload: program_studi, plan semester)
    GW->>DB: Query tabel courses & course_prerequisites
    DB-->>GW: Data relasi graf prasyarat & paritas semester
    GW->>GW: Validasi deterministik: Prerequisite DAG, Paritas Ganjil/Genap, Batas SKS Pasal 5.4
    GW->>DB: INSERT / UPDATE ke tabel study_plans (status: DRAFT)
    GW-->>FE: Response HTTP 200 { is_valid: true, violations: [] }
    FE->>FE: Alihkan tab ke DegreePlanner.tsx & Isi visual roadmap Semester 1-8
    FE-->>User: Visual roadmap terisi otomatis & siap diekspor (.ics / PDF)
```

---

## 3. Spesifikasi Kontrak Data Antar-Layer

### A. Kontrak Dialog Streaming (Tier 2 ➔ Tier 3 & Tier 1)
- **Path:** `POST /api/v1/chat/stream`
- **Request Body:**
  ```json
  {
    "message": "Buatkan rencana studi semester 3 prodi TSD peminatan Machine Learning",
    "history": [
      { "role": "user", "content": "Halo" },
      { "role": "assistant", "content": "Halo! Ada yang bisa saya bantu?" }
    ],
    "student_profile": {
      "program_studi": "Teknologi Sains Data",
      "semester_saat_ini": 2,
      "riwayat_matkul_lulus": ["Dasar Pemrograman", "Matematika I"],
      "minat_fokus": ["Machine Learning"],
      "target_kelulusan_semester": 8,
      "maks_sks_per_semester": 24,
      "is_confirmed_by_user": true
    }
  }
  ```

- **Format Server-Sent Events (SSE):**
  ```
  event: token
  data: {"token": "Tentu"}

  event: token
  data: {"token": ", berikut"}

  event: profile_update
  data: {"program_studi": "Teknologi Sains Data", "semester": 3, "minat_fokus": ["Machine Learning"]}

  event: plan
  data: {
    "action": "APPLY_DEGREE_PLAN",
    "summary": {
      "program_studi": "Teknologi Sains Data",
      "current_semester": 3,
      "target_semester": 8,
      "total_credits": 20,
      "focus_tracks": ["Machine Learning"]
    },
    "plan": {
      "3": [
        { "code": "SID201", "name": "Algoritma dan Struktur Data", "credits": 3 },
        { "code": "SID203", "name": "Statistika Matematika", "credits": 3 }
      ]
    }
  }

  event: done
  data: {"routing_mode": "HYBRID", "latency_ms": 1824.5}
  ```

### B. Kontrak Validasi Database Degree Planner (Tier 1 ➔ Tier 2 ➔ Tier 4)
- **Path:** `POST /api/v1/planner/validate`
- **Request Body:** Sesuai field `plan` dari `DegreePlanPayload`.
- **Response Body:**
  ```json
  {
    "is_valid": true,
    "violations": [],
    "total_sks": 144,
    "semester_sks": {
      "3": 20,
      "4": 20
    }
  }
  ```

---

## 4. Cara Membuka & Mengedit File Diagram draw.io

File diagram asli berada di `docs/architecture/ftmm_compass_integration_flow.drawio.xml`.
1. **Melalui Browser:**
   - Kunjungi [https://app.diagrams.net/](https://app.diagrams.net/).
   - Pilih *Open Existing Diagram* dan pilih file `docs/architecture/ftmm_compass_integration_flow.drawio.xml`.
2. **Melalui VS Code:**
   - Pasang ekstensi `Hediet.vscode-drawio`.
   - Buka file `docs/architecture/ftmm_compass_integration_flow.drawio.xml`, diagram akan langsung terbuka secara visual di editor.
