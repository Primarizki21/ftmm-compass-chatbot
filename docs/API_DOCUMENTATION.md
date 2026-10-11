# FTMM COMPASS — REST API Documentation

Spesifikasi resmi antarmuka pemrograman aplikasi (REST API) untuk backend AI Chatbot FTMM COMPASS. Dokumen ini bertindak sebagai *Single Source of Truth (SSOT)* kontrak komunikasi data antara Tier 3 (AI Microservice di repositori ini) dengan Tier 1 (Frontend React `ftmm-compass-ui`) dan Tier 2 (API Gateway FastAPI).

---

## 1. Overview & Service Discovery

Backend FTMM COMPASS Chatbot menyediakan endpoint REST berbasis HTTP/JSON dengan dokumentasi OpenAPI interaktif bawaan.

### Base URL & Lingkungan Eksekusi
- **Docker Container (Rekomendasi):** `http://localhost:8000`
- **Native Host (`uv run`):** `http://localhost:8000`
- **Vite Dev Server Proxy (Repo UI):** `http://localhost:5173/api/*` (diteruskan otomatis ke `http://localhost:8000`)

### Interactive Documentation
Saat server aktif, dokumentasi interaktif dapat diakses langsung via browser:
- **Swagger UI:** `http://localhost:8000/docs`
- **ReDoc:** `http://localhost:8000/redoc`
- **OpenAPI JSON Schema:** `http://localhost:8000/openapi.json`

### Kebijakan CORS
Middleware CORS telah dikonfigurasi terbuka (`allow_origins=["*"]`, `allow_methods=["*"]`, `allow_headers=["*"]`) untuk memudahkan integrasi langsung dengan dev server frontend Vite di port `5173` maupun staging server.

---

## 2. Endpoint Health Check

### `GET /api/health`
Digunakan oleh container healthcheck, reverse proxy gateway, dan frontend untuk memverifikasi kesiapan operasional backend serta status konektivitas ke daemon Ollama host.

#### Request
- **Method:** `GET`
- **Path:** `/api/health`
- **Headers:** Tidak memerlukan autentikasi khusus.

#### Response (`200 OK`)
```json
{
  "status": "healthy",
  "service": "FTMM COMPASS Chatbot Backend (Hybrid RAG-CAG)",
  "ollama_online": true,
  "default_model": "qwen2.5:3b"
}
```

#### Deskripsi Field Response
| Field | Tipe Data | Keterangan |
|---|---|---|
| `status` | `string` | Status kesehatan layanan (`"healthy"`). |
| `service` | `string` | Nama identitas mikroservis. |
| `ollama_online` | `boolean` | `true` jika backend berhasil menghubungi endpoint `/api/tags` pada daemon Ollama host. |
| `default_model` | `string` | Model bahasa lokal aktif yang dikonfigurasi (`"qwen2.5:3b"`). |

---

## 3. Endpoint Chat Dialog Utama

### `POST /api/chat`
Endpoint dialog utama yang menerima pesan pengguna, riwayat percakapan (*sliding window*), dan status profil mahasiswa (*slot-filling memory*). Endpoint ini mengembalikan jawaban terkurasi, pembaruan profil otomatis, serta struktur rencana studi (*degree plan*) jika kueri bertipe perencanaan akademik.

#### Request
- **Method:** `POST`
- **Path:** `/api/chat`
- **Content-Type:** `application/json`

#### Request Payload (`ChatRequest`)
```json
{
  "message": "Saya mahasiswa semester 4 prodi TSD minat machine learning, rekomendasikan matkul untuk semester 5",
  "history": [
    {
      "role": "user",
      "content": "Halo Compass AI!"
    },
    {
      "role": "assistant",
      "content": "Halo! Saya Compass AI, asisten akademik pintar di FTMM UNAIR. Ada yang bisa saya bantu?"
    }
  ],
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

#### Deskripsi Field Request
| Parameter | Tipe Data | Wajib | Keterangan |
|---|---|---|---|
| `message` | `string` | **Ya** | Pertanyaan atau pesan teks mahasiswa. |
| `history` | `array[ChatMessageItem]` | Tidak | Riwayat obrolan sebelumnya untuk konteks dialog multi-turn. |
| `history[].role` | `string` | Ya | Peran pengirim: `"user"`, `"assistant"`, atau `"system"`. |
| `history[].content` | `string` | Ya | Teks percakapan giliran tersebut. |
| `student_profile` | `object` | Tidak | State profil mahasiswa saat ini (default: objek kosong `{}`). |
| `student_profile.program_studi` | `string \| null` | Tidak | Prodi mahasiswa: `"Teknologi Sains Data"`, `"Teknik Robotika dan Kecerdasan Buatan"`, `"Rekayasa Nanoteknologi"`, `"Teknik Industri"`, atau `"Teknik Elektro"`. |
| `student_profile.semester_saat_ini`| `integer \| null` | Tidak | Angka semester mahasiswa saat ini (1–8). |
| `student_profile.riwayat_matkul_lulus` | `array[string]` | Tidak | Daftar kode MK yang telah diselesaikan. |
| `student_profile.minat_fokus` | `array[string]` | Tidak | Tag minat akademik (contoh: `["AI", "Robotics"]`). |
| `student_profile.target_kelulusan_semester` | `integer` | Tidak | Target lulus (default: `8`). |
| `student_profile.maks_sks_per_semester` | `integer` | Tidak | Beban SKS legal berdasarkan IPS (default: `24`). |
| `student_profile.is_confirmed_by_user` | `boolean` | Tidak | Flag konfirmasi profil oleh user (default: `false`). |

---

#### Response Payload (`ChatResponse`)

##### Contoh A: Respons Pertanyaan Kebijakan / SOP Umum (Mode CAG)
```json
{
  "reply": "Untuk mengambil Praktik Kerja Lapangan (PKL) di FTMM UNAIR, mahasiswa wajib telah menempuh beban studi kumulatif minimal 80 SKS tanpa nilai E pada mata kuliah prasyarat.",
  "updated_profile": {
    "program_studi": "Teknologi Sains Data",
    "semester_saat_ini": 4,
    "riwayat_matkul_lulus": ["SIA101", "SIA102"],
    "minat_fokus": ["Machine Learning"],
    "target_kelulusan_semester": 8,
    "maks_sks_per_semester": 24,
    "is_confirmed_by_user": false
  },
  "action_type": "GENERAL",
  "plan_payload": null,
  "routing_mode": "CAG",
  "latency_ms": 384.25
}
```

##### Contoh B: Respons Rekomendasi Rencana Studi (Mode HYBRID)
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
  "action_type": "PLAN_GENERATED",
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
        { "code": "SIA303", "name": "Pemrosesan Bahasa Alami", "credits": 3 },
        { "code": "SIA305", "name": "Sistem Temu Balik Informasi", "credits": 3 }
      ]
    }
  },
  "routing_mode": "HYBRID",
  "latency_ms": 1420.80
}
```

#### Deskripsi Field Response
| Field | Tipe Data | Keterangan |
|---|---|---|
| `reply` | `string` | Teks jawaban final dari asisten (format Markdown didukung). |
| `updated_profile` | `object` | State profil mahasiswa terkini setelah diekstrak oleh modul entity slot-filling. |
| `action_type` | `string` | Petunjuk aksi bagi UI frontend: `"GENERAL"`, `"NEED_INFO"`, `"INFO_CONFIRMED"`, atau `"PLAN_GENERATED"`. |
| `plan_payload` | `object \| null` | Berisi struktur rencana studi jika `action_type == "PLAN_GENERATED"`. Bernilai `null` pada kueri reguler. |
| `routing_mode` | `string` | Mode penalaran yang dieksekusi: `"CAG"`, `"RAG"`, `"HYBRID"`, atau `"GUARDED"`. |
| `latency_ms` | `float` | Total latensi pemrosesan turn dalam milidetik ($ms$). |

---

## 4. Spesifikasi Kontrak `DegreePlanPayload`

Jika `action_type` bernilai `"PLAN_GENERATED"`, objek `plan_payload` dikirimkan secara terstruktur agar komponen frontend `DegreePlanner.tsx` dapat merender kartu pratinjau rencana studi (*Degree Plan Card Widget*) dan mengeksekusi callback `onApplyPlan`.

```json
{
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
      {
        "code": "SIA301",
        "name": "Pembelajaran Mesin Terapan",
        "credits": 3
      }
    ]
  }
}
```

### Struktur Objek:
1. **`action`** (`string`): Wajib bernilai literal `"APPLY_DEGREE_PLAN"`.
2. **`summary`** (`object`): Ringkasan rencana studi (program studi, semester saat ini, semester target kelulusan, total SKS kumulatif, dan daftar fokus track).
3. **`plan`** (`map[semester, list[course]]`): Key berupa integer nomor semester tujuan (contoh: `5`), memetakan array objek mata kuliah dengan atribut `code`, `name`, dan `credits`.

---

## 5. Penanganan Error & Guardrail Refusal

Backend mengimplementasikan *Cascaded Defense-in-Depth* (Tier 1 s.d. Tier 4). Jika input melanggar batasan keamanan atau berada di luar cakupan akademik, API tetap merespons status `200 OK` dengan pesan penolakan formal dan `routing_mode: "GUARDED"` (atau `"REFUSED"`).

### Contoh Respons Penolakan (Guardrail Triggered)
```json
{
  "reply": "Permintaan tidak dapat diproses karena melanggar kebijakan keamanan sistem akademik FTMM UNAIR.",
  "updated_profile": { ... },
  "action_type": "GENERAL",
  "plan_payload": null,
  "routing_mode": "GUARDED",
  "latency_ms": 1.2
}
```

### Kategori Penolakan:
- **Tier 1 (Adversarial Attack / Jailbreak):** Ditolak instan dalam $\approx 0\text{ ms}$ oleh regex blocklist.
- **Tier 2 (Out-of-Scope Non-Akademik):** Ditolak dalam $\approx 30\text{ ms}$ oleh FAISS Centroid Semantic Gate (topik kuliner, crypto, politik, dsb).
- **Tier 4 (Argument Constraint Violation):** Ditolak jika parameter melanggar regulasi (misal meminta semester $>8$ atau beban SKS melebihi batas legal IPS).

### HTTP Error Codes
| Status Code | Arti | Skenario Terjadinya |
|---|---|---|
| `200 OK` | Sukses | Request berhasil diproses (termasuk respons penolakan ramah pengguna). |
| `422 Unprocessable Entity` | Validasi Schema Gagal | Payload JSON tidak sesuai tipe data Pydantic (misal `semester_saat_ini` berupa string huruf). |
| `500 Internal Server Error` | Server Exception | Terjadi kegagalan kritis pada backend atau dependensi host tidak merespons. |

---

## 6. Contoh Integrasi Client

### A. Contoh Fetch JavaScript / TypeScript (Frontend React)
```typescript
interface ChatPayload {
  message: string;
  history: Array<{ role: 'user' | 'assistant'; content: string }>;
  student_profile: Record<string, any>;
}

async function sendChatMessage(payload: ChatPayload) {
  const response = await fetch('http://localhost:8000/api/chat', {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify(payload),
  });

  if (!response.ok) {
    throw new Error(`HTTP error! status: ${response.status}`);
  }

  const data = await response.json();
  console.log('AI Reply:', data.reply);
  console.log('Routing Mode:', data.routing_mode);
  
  if (data.action_type === 'PLAN_GENERATED' && data.plan_payload) {
    console.log('Degree Plan Ready:', data.plan_payload);
  }

  return data;
}
```

### B. Contoh Pengujian via cURL

#### 1. Uji Healthcheck:
```bash
curl -X GET http://localhost:8000/api/health
```

#### 2. Uji Chat Sapaan (Chitchat):
```bash
curl -X POST http://localhost:8000/api/chat \
  -H "Content-Type: application/json" \
  -d '{
    "message": "Halo Compass AI!",
    "history": [],
    "student_profile": {}
  }'
```

#### 3. Uji Chat Regulasi Akademik (SOP PKL):
```bash
curl -X POST http://localhost:8000/api/chat \
  -H "Content-Type: application/json" \
  -d '{
    "message": "Berapa syarat SKS minimal untuk mengambil PKL di FTMM?",
    "history": [],
    "student_profile": {
      "program_studi": "Teknologi Sains Data",
      "semester_saat_ini": 4
    }
  }'
```
