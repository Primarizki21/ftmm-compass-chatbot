# FTMM COMPASS — REST API Documentation

Spesifikasi resmi antarmuka pemrograman aplikasi (REST API) untuk backend AI Chatbot FTMM COMPASS. Dokumen ini bertindak sebagai *Single Source of Truth (SSOT)* kontrak komunikasi data antara Tier 3 (AI Microservice di repositori ini) dengan Tier 1 (Frontend React `ftmm-compass-ui`) dan Tier 2 (API Gateway FastAPI).

Menerapkan pola arsitektur **Dual-Tier REST API** (versi kanonik `/api/v1` dan alias kompatibilitas `/api`) sesuai standar `api-design`.

---

## 1. Overview & Service Discovery

Backend FTMM COMPASS Chatbot menyediakan endpoint REST berbasis HTTP/JSON dan Server-Sent Events (SSE) dengan dokumentasi OpenAPI interaktif bawaan.

### Base URL & Lingkungan Eksekusi
- **Docker Container (Port Publish Host):** `http://localhost:8000` (atau `http://127.0.0.1:8000`)
- **Docker Multi-Container Network (`ftmm-network`):** `http://ftmm-compass-chatbot:8000`
- **Native Host (`uv run`):** `http://127.0.0.1:8000`
- **Gateway Reverse Proxy (Repo UI):** `http://localhost:8001/api/v1/*` (diteruskan via `ftmm-network`)
- **Frontend Vite Dev Server (Repo UI):** `http://localhost:8443` (StrictPort)

### Interactive Documentation
Saat server aktif, dokumentasi interaktif dapat diakses langsung via browser:
- **Swagger UI:** `http://localhost:8000/docs`
- **ReDoc:** `http://localhost:8000/redoc`
- **OpenAPI JSON Schema:** `http://localhost:8000/openapi.json`

### Kebijakan CORS
Middleware CORS telah dikonfigurasi terbuka (`allow_origins=["*"]`, `allow_methods=["*"]`, `allow_headers=["*"]`) untuk memudahkan integrasi langsung dengan dev server frontend Vite di port `8443`, gateway di port `8001`, maupun staging server.

---

## 2. Matriks Rute Endpoint (Dual-Tier REST)

| Kategori | Canonical Endpoint (`/api/v1/`) | Legacy Alias (`/api/`) | HTTP Method | Keterangan |
|---|---|---|---|---|
| **Health** | `GET /api/v1/health` | `GET /api/health` | `GET` | Health check gateway & Ollama daemon connectivity |
| **Catalog** | `GET /api/v1/courses` | `GET /api/courses` | `GET` | Katalog 369 mata kuliah resmi 5 prodi FTMM (+ paginasi) |
| **Search** | `GET /api/v1/courses/search` | `GET /api/courses/search` | `GET` | Pencarian semantik & kata kunci silabus |
| **Chat** | `POST /api/v1/chat` | `POST /api/chat` | `POST` | Dialog akademik utama (Hybrid RAG-CAG) |
| **Streaming** | `POST /api/v1/chat/stream` | `POST /api/chat/stream` | `POST` | Server-Sent Events (SSE) streaming live tokens |
| **Plan Validation** | `POST /api/v1/plans/validations` | `POST /api/plan/validate` | `POST` | Validasi DAG prasyarat & paritas semester rencana studi |
| **Plan Generation** | `POST /api/v1/plans` | `POST /api/plan/generate` | `POST` | Pembuatan rencana studi deterministik otomatis |

---

## 3. Spesifikasi Endpoint

### 3.1 Endpoint Health Check

#### `GET /api/v1/health` (Alias: `GET /api/health`)
Digunakan oleh container healthcheck, reverse proxy gateway, dan frontend untuk memverifikasi kesiapan operasional backend serta status konektivitas ke daemon Ollama host.

- **Query Parameters:**
  - `strict` *(opsional, boolean, default: `false`)*: Jika bernilai `true`, endpoint akan melempar HTTP 503 jika daemon Ollama tidak dapat dihubungi.

##### Response (`200 OK`)
```json
{
  "status": "healthy",
  "service": "FTMM COMPASS Chatbot Backend",
  "ollama_online": true,
  "default_model": "qwen2.5:3b",
  "ollama": {
    "online": true,
    "host": "http://host.docker.internal:11434",
    "default_model": "qwen2.5:3b"
  }
}
```

##### Response Saat Terdegradasi (`strict=false`, `200 OK`)
```json
{
  "status": "degraded",
  "service": "FTMM COMPASS Chatbot Backend",
  "ollama_online": false,
  "default_model": "qwen2.5:3b",
  "ollama": {
    "online": false,
    "host": "http://host.docker.internal:11434",
    "default_model": "qwen2.5:3b"
  }
}
```

---

### 3.2 Endpoint Chat Dialog Utama

#### `POST /api/v1/chat` (Alias: `POST /api/chat`)
Endpoint dialog utama yang menerima pesan pengguna, riwayat percakapan (*sliding window*), dan status profil mahasiswa (*slot-filling memory*).

##### Request Payload (`ChatRequest`)
```json
{
  "message": "Saya mahasiswa semester 4 prodi TSD minat machine learning, rekomendasikan matkul semester 5",
  "history": [
    {
      "role": "user",
      "content": "Halo Compass AI!"
    },
    {
      "role": "assistant",
      "content": "Halo! Saya Compass AI, asisten akademik FTMM UNAIR. Ada yang bisa saya bantu?"
    }
  ],
  "session_id": "user-sess-001",
  "student_profile": {
    "program_studi": "Teknologi Sains Data",
    "semester_saat_ini": 4,
    "riwayat_matkul_lulus": ["SIA101", "SIA102"],
    "minat_fokus": ["Machine Learning"],
    "target_kelulusan_semester": 8,
    "maks_sks_per_semester": 24,
    "is_confirmed_by_user": false
  }
}
```

##### Response Payload (`ChatResponse`)
```json
{
  "reply": "Berdasarkan minat Machine Learning pada prodi Teknologi Sains Data semester 5, berikut adalah rekomendasi mata kuliah yang dapat Anda ambil...",
  "updated_profile": {
    "program_studi": "Teknologi Sains Data",
    "semester_saat_ini": 4,
    "riwayat_matkul_lulus": ["SIA101", "SIA102"],
    "minat_fokus": ["Machine Learning"],
    "target_kelulusan_semester": 8,
    "maks_sks_per_semester": 24,
    "is_confirmed_by_user": false
  },
  "plan_payload": {
    "action": "APPLY_DEGREE_PLAN",
    "summary": {
      "program_studi": "Teknologi Sains Data",
      "current_semester": 4,
      "target_semester": 8,
      "total_credits": 144,
      "focus_tracks": ["Machine Learning"],
      "note": "Rencana studi rekomendasi AI berdasarkan kurikulum resmi dan prasyarat FTMM."
    },
    "plan": {
      "5": [
        { "code": "SIA301", "name": "Pembelajaran Mesin Terapan", "credits": 3 },
        { "code": "SIA303", "name": "Pemrosesan Bahasa Alami", "credits": 3 }
      ]
    }
  },
  "action_type": "PLAN_GENERATED",
  "missing_slots": [],
  "routing_mode": "HYBRID",
  "latency_ms": 1420.80
}
```

---

### 3.3 Endpoint Real-Time Chat Streaming (SSE)

#### `POST /api/v1/chat/stream` (Alias: `POST /api/chat/stream`)
Menyediakan Server-Sent Events (SSE) streaming live token per token untuk rendering live typing di antarmuka obrolan React.

- **Content-Type:** `text/event-stream`
- **Format Event Stream:**
  ```text
  data: {"token": "Halo", "done": false}

  data: {"token": "!", "done": false}

  data: {"token": "", "done": true, "reply": "Halo!", "routing_mode": "DIRECT", "action_type": "GENERAL", "latency_ms": 420.5}
  ```

---

### 3.4 Endpoint Katalog Kurikulum Resmi

#### `GET /api/v1/courses` (Alias: `GET /api/courses`)
Single Source of Truth (SSOT) katalog kurikulum 5 program studi FTMM (369 mata kuliah).

- **Query Parameters:**
  - `program_studi` / `prodi` *(opsional, string)*: Filter prodi kanonik (`"Teknologi Sains Data"`, `"Teknik Elektro"`, dsb) atau alias (`"tsd"`, `"robotika"`).
  - `semester` *(opsional, integer 1–8)*: Filter semester sasaran.
  - `q` *(opsional, string)*: Pencarian semantik/silabus.
  - `page` *(opsional, integer >= 1)*: Halaman aktif (mengaktifkan mode paginasi).
  - `per_page` *(opsional, integer 1–100)*: Limit mata kuliah per halaman.

##### Response Mode Paginasi (`page` atau `per_page` diberikan):
```json
{
  "data": [
    {
      "id": "MK-001",
      "code": "SIA101",
      "name": "Pengantar Sains Data",
      "credits": 3,
      "semester": 1,
      "type": "Wajib",
      "parity": "odd",
      "desc": "Dasar konsep sains data...",
      "program": "Teknologi Sains Data",
      "prerequisites": []
    }
  ],
  "meta": {
    "total": 68,
    "page": 1,
    "per_page": 20,
    "total_pages": 4
  },
  "links": {
    "self": "/api/v1/courses?page=1&per_page=20",
    "next": "/api/v1/courses?page=2&per_page=20",
    "prev": null,
    "first": "/api/v1/courses?page=1&per_page=20",
    "last": "/api/v1/courses?page=4&per_page=20"
  },
  "program_studi": "Teknologi Sains Data"
}
```

##### Response Mode Katalog Utuh (Unpaginated):
```json
{
  "source": "chatbot_ssot",
  "program_studi": "Teknologi Sains Data",
  "total_courses": 68,
  "programs": [
    "S1 Rekayasa Nanoteknologi",
    "Teknik Elektro",
    "Teknik Industri",
    "Teknik Robotika dan Kecerdasan Buatan",
    "Teknologi Sains Data"
  ],
  "courses": [ ... ]
}
```

---

### 3.5 Endpoint Pencarian Mata Kuliah

#### `GET /api/v1/courses/search` (Alias: `GET /api/courses/search`)
Endpoint pencarian semantik dan kata kunci silabus terindeks.

- **Query Parameters:**
  - `q` *(wajib, string, minLength: 1)*: Kata kunci atau topik (misal `"machine learning"`).
  - `program_studi` / `prodi` *(opsional, string)*: Filter prodi.
  - `top_k` *(opsional, integer 1–20, default: `5`)*: Batas jumlah pencarian.

##### Response (`200 OK`):
```json
{
  "source": "hybrid_faiss_bm25",
  "query": "machine learning",
  "program_studi": "Teknologi Sains Data",
  "total_hits": 3,
  "results": [
    {
      "id": "MK-042",
      "code": "SIA301",
      "name": "Pembelajaran Mesin Terapan",
      "credits": 3,
      "semester": 5,
      "type": "Wajib",
      "parity": "odd",
      "program": "Teknologi Sains Data",
      "score": 0.942
    }
  ]
}
```

---

### 3.6 Endpoint Validasi Rencana Studi (Prerequisite DAG)

#### `POST /api/v1/plans/validations` (Alias: `POST /api/plan/validate`)
Memvalidasi rencana studi terhadap directed acyclic graph (DAG) prasyarat mata kuliah, aturan paritas semester ganjil/genap, dan batas maksimum beban SKS legal.

##### Request Payload (`PlanValidationRequest`):
```json
{
  "plan": {
    "5": [
      { "code": "SIA301", "name": "Pembelajaran Mesin Terapan", "credits": 3 },
      { "code": "SIA303", "name": "Pemrosesan Bahasa Alami", "credits": 3 }
    ]
  },
  "student_profile": {
    "program_studi": "Teknologi Sains Data",
    "semester_saat_ini": 4,
    "riwayat_matkul_lulus": ["SIA101", "SIA201"],
    "maks_sks_per_semester": 24
  }
}
```

##### Response Payload (`PlanValidationResponse`):
```json
{
  "is_valid": true,
  "violations": [],
  "warnings": [],
  "total_credits": 6,
  "total_sks_planned": 6,
  "details_per_semester": {
    "5": {
      "sks": 6,
      "courses_count": 2
    }
  }
}
```

---

### 3.7 Endpoint Pembuatan Rencana Studi Instan

#### `POST /api/v1/plans` (Alias: `POST /api/plan/generate`)
Menghasilkan draft rencana studi 8 semester deterministik yang langsung tervalidasi prasyarat DAG.

##### Request Payload: `StudentProfile`
##### Response Payload: `DegreePlanPayload`

---

## 4. Standar Penanganan Error & Status Codes (API Design)

Seluruh respons kegagalan API mengembalikan status HTTP semantik dan format envelope standar:

### Skema Error Response
```json
{
  "error": {
    "code": "upstream_service_unavailable",
    "message": "Layanan Ollama tidak dapat dihubungi. Pastikan daemon Ollama aktif pada host port 11434.",
    "details": [
      {
        "field": "ollama_connection",
        "message": "Failed to connect to Ollama daemon on host port 11434",
        "code": "connect_timeout"
      }
    ]
  }
}
```

### Matriks HTTP Status Codes
| Status Code | Code Identifier | Keterangan & Header |
|---|---|---|
| `200 OK` | - | Request berhasil diproses |
| `422 Unprocessable Entity` | `validation_error` | Payload JSON tidak sesuai tipe data Pydantic |
| `503 Service Unavailable` | `upstream_service_unavailable` | Daemon Ollama offline. Menyertakan header `Retry-After: 5` |
| `504 Gateway Timeout` | `upstream_timeout` | Proses inferensi melebihi batas waktu toleransi |

---

## 5. Contoh Integrasi Client

### Contoh Fetch TypeScript (Frontend UI)
```typescript
async function fetchCourses(page = 1, perPage = 20, prodi = "Teknologi Sains Data") {
  const url = `/api/v1/courses?prodi=${encodeURIComponent(prodi)}&page=${page}&per_page=${perPage}`;
  const response = await fetch(url, {
    headers: { "Accept": "application/json" }
  });

  if (!response.ok) {
    const errorBody = await response.json();
    throw new Error(errorBody.error?.message || `HTTP ${response.status}`);
  }

  return await response.json();
}
```

### Contoh cURL Test Suite
```bash
# 1. Health check kanonik
curl -s http://127.0.0.1:8000/api/v1/health | jq .

# 2. Paginasi katalog kurikulum
curl -s "http://127.0.0.1:8000/api/v1/courses?prodi=Teknologi%20Sains%20Data&page=1&per_page=5" | jq .

# 3. Dialog chat
curl -s -X POST http://127.0.0.1:8000/api/v1/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "Halo Compass AI", "history": []}' | jq .
```
