# Panduan Standarisasi Istilah Teknis Visual & Dokumentasi (Anti-Translation Slop)

> **Klasifikasi:** Dokumen Panduan Standar Teknis & Gaya Bahasa (Tracked in Git).  
> **Fungsi:** Pedoman kanonik bagi seluruh agen AI dan pengembang dalam penulisan istilah teknis pada **flowchart visual, diagram arsitektur, poster, dan laporan resmi**.  
> **Aturan Wajib:** Seluruh agen **DILARANG KERAS** menerjemahkan istilah baku ilmu komputer, NLP, dan rekayasa perangkat lunak ke dalam Bahasa Indonesia secara harfiah atau kaku (*awkward literal translation*). Istilah teknis standar wajib dipertahankan dalam **Bahasa Inggris kanonik**.

---

## 1. Prinsip Utama (Core Principles)

1. **Keep Technical Terms in English (Bahasa Inggris Teknis Baku)**:
   Istilah-istilah baku industri dan literatur ilmiah AI/NLP (misal: *Cache-Augmented Generation*, *Shared Latent Vector*, *Dense Gazetteer Linking*, *Prefix Caching*, *Constraint Satisfaction Problem*) **WAJIB tetap ditulis dalam Bahasa Inggris**. Penerjemahan paksa ke Bahasa Indonesia menghasilkan istilah yang janggal, ambigu, dan tidak dikenal dalam komunitas ilmiah.
2. **Natural Technical Register (Bahasa Indonesia Pengantar yang Alami)**:
   Teks pengantar, verba tindakan, atau keterangan operasional boleh menggunakan Bahasa Indonesia formal, namun nomina teknis tetap dalam Bahasa Inggris.
   * *Contoh Benar*: "Sanitasi SQL Injection & parameter tampering"
   * *Contoh Salah (Kaku/Slop)*: "Pembersihan suntikan SQL & perusakan parameter"
3. **Zero Module Paths & Zero Exp-ID Invariant**:
   Diagram flowchart dan laporan resmi tidak boleh memuat path internal berkas (`src/...`, `.py`) maupun kode pelacak laboratorium (`Exp-XXX`). Komponen wajib dinamai murni berdasarkan peran arsitekturalnya.

---

## 2. Kamus Kanonik Istilah Teknis (Canonical Terminology Lexicon)

Gunakan tabel padanan ini sebagai acuan mutlak. Kolom **Istilah Terlarang** mencatat terjemahan harfiah yang dilarang digunakan oleh agen.

### 2.1 Arsitektur Inti, Embedding, & Retrieval

| Istilah Kanonik (Wajib Digunakan) | Istilah Terlarang (Awkward / Slop Translation) | Konteks & Definisi Arsitektur |
|---|---|---|
| **Cache-Augmented Generation (CAG)** | Generasi Ditambah Singgahan / Pembangkitan Berbasis Tembolok | Prapemanasan prompt sistem dengan regulasi statis (0 ms retrieval). |
| **Retrieval-Augmented Generation (RAG)** | Pembangkitan Ditambah Pengambilan | Pencarian informasi dinamis pada korpus kurikulum via FAISS/BM25. |
| **Hybrid RAG-CAG Pipeline** | Saluran Pipa Hibrida RAG-CAG | Penggabungan deterministik rute CAG regulasi dan RAG silabus. |
| **Single-Embed Shared Latent Vector** | Vektor Laten Bersama Satu Tempelan / Satu Benaman | 1x pemanggilan embedding yang digunakan bersama oleh Guardrail, Decider, dan Retriever. |
| **Shared Latent Intent Classification** | Klasifikasi Intent Semantik Ruang Laten / Niat Tersembunyi Bersama | Klasifikasi intent kueri menggunakan dot-product BLAS pada shared vector. |
| **Dense Gazetteer Linking** | Tautan Buku Geografi Padat / Penautan Kamus Padat | Pencocokan vektor representasi dense terhadap 369 katalog mata kuliah resmi. |
| **Reciprocal Rank Fusion (RRF)** | Penggabungan Peringkat Timbal Balik | Algoritma fusi peringkat sparse BM25 dan dense FAISS ($k=60$). |
| **Prefix Caching** | Penyimpanan Awalan / Tembolok Awalan | Mekanisme reuse KV cache prompt prefix untuk sapaan dan FAQ. |

---

### 2.2 Guardrails, Keamanan, & Filter

| Istilah Kanonik (Wajib Digunakan) | Istilah Terlarang (Awkward / Slop Translation) | Konteks & Definisi Arsitektur |
|---|---|---|
| **Deterministic Syntax Filter** | Filter Sintaksis Pasti / Penyaring Format Tetap | Tier 1: Pembersihan pola SQL injection murni dan parameter tamper. |
| **Semantic Dual-Centroid Gate** | Gerbang Titik Pusat Ganda Semantik | Tier 2: Batas margin cosine similarity kontinu untuk deteksi OOD. |
| **Out-of-Scope (OOD) Detection** | Deteksi Luar Lingkup / Penemuan Di Luar Cakupan | Pengidentifikasian kueri non-akademik dan non-FTMM. |
| **Adversarial Jailbreak** | Pembobolan Melawan / Penjebolan Sistem | Upaya manipulasi prompt untuk melewati guardrail keselamatan. |
| **Physical Parameter Rail** | Rel Parameter Fisik | Tier 4: Validasi batas numerik legal (semester 1-8, SKS 0-24). |
| **Error Envelope** | Amplop Kesalahan | Format standar JSON response untuk penolakan kueri (`{ error: { code, message } }`). |
| **False Refusal Rate (FRR)** | Tingkat Penolakan Keliru | Persentase kueri legal mahasiswa yang tertolak secara salah oleh guardrail. |
| **Attack Success Rate (ASR)** | Tingkat Keberhasilan Serangan | Persentase jailbreak atau injeksi yang berhasil menembus pertahanan sistem. |

---

### 2.3 Reasoning, Agent, & Perencanaan Studi (Degree Planning)

| Istilah Kanonik (Wajib Digunakan) | Istilah Terlarang (Awkward / Slop Translation) | Konteks & Definisi Arsitektur |
|---|---|---|
| **System-1 Decider** | Pengambil Keputusan Sistem Satu | Pengklasifikasi intent cepat berbasis BLAS dot-product ($<0.1\text{ ms}$). |
| **ReAct Agent (Tool-Calling)** | Agen Bereaksi (Pemanggilan Alat) | Pola reasoning *Reasoning + Acting* dengan native tool execution. |
| **Zero Tool Schema Overhead** | Tanpa Beban Skema Alat | Peniadaan skema JSON tool pada prompt kueri sapaan/FAQ. |
| **Neuro-Symbolic CSP Planner** | Perencana CSP Neuro-Simbolik / Pemecah Kepuasan Kendala | Sintesis rencana studi memadukan embedding semantik dan solver CSP deterministik. |
| **Constraint Satisfaction Problem (CSP)** | Masalah Kepuasan Kendala | Formulasi penempatan mata kuliah dengan batasan kapasitas SKS dan prasyarat. |
| **Topological Placement** | Penempatan Topologis | Pengurutan mata kuliah berdasarkan Directed Acyclic Graph (DAG) prasyarat. |
| **Semester Parity Adherence** | Kepatuhan Kesetaraan / Paritas Semester | Penegakan jadwal penawaran matkul hanya pada semester ganjil atau genap. |
| **Single Religion Dedup** | Penghapusan Duplikat Agama Tunggal | Pembatasan paket mata kuliah agama maksimal 1 agama per mahasiswa (2 SKS). |
| **Focus Track Expansion** | Ekspansi Jalur Fokus / Perluasan Minat | Pemetaan peminatan semantik non-literal (misal: "fokus LLM" $\to$ NLP SIC304). |
| **Latent Ellipsis Resolution** | Resolusi Elipsis Tersembunyi / Penyelesaian Titik-titik Laten | Pemulihan konteks tersirat mahasiswa tanpa kata ganti eksplisit. |

---

### 2.4 API Gateway, Streaming, & Antarmuka UI

| Istilah Kanonik (Wajib Digunakan) | Istilah Terlarang (Awkward / Slop Translation) | Konteks & Definisi Arsitektur |
|---|---|---|
| **Server-Sent Events (SSE) Streaming** | Aliran Peristiwa Kiriman Peladen | Pengiriman token jawaban mengalir secara real-time (`/api/v1/chat/stream`). |
| **Token Streaming** | Pengaliran Token | Pengiriman respons kata-per-kata untuk memicu efek mengetik (*live typing*). |
| **Live Typing Effect** | Efek Mengetik Langsung | Visualisasi teks muncul mengalir huruf demi huruf di browser. |
| **DegreePlanPayload JSON** | Muatan Rencana Gelar JSON | Kontrak data JSON terstruktur untuk merender roadmap semester 1-8 di frontend. |
| **Dual-Tier API Architecture** | Arsitektur API Tingkat Ganda | Penyediaan rute kanonik modern `/api/v1/*` dan backward-compatibility `/api/*`. |
---

## 3. Contoh Penerapan Baku pada Diagram Flowchart

Saat menyusun teks simpul, badge, dan label pada skrip flowchart (`scripts/generate_architecture_flowchart.py`):

* **Judul Simpul**:
  * Benar: `Single-Embed Step (Ollama C++ Native Daemon)`
  * Benar: `Tier 2: Semantic Dual-Centroid Gate`
  * Benar: `System-1 Decider (6-Class Intent Matrix)`
  * Benar: `Neuro-Symbolic CSP Planner`
  * Salah: *Langkah Satu Benaman*, *Gerbang Titik Pusat Semantik*, *Pengambil Keputusan Sistem-1*, *Perencana Masalah Kepuasan Kendala*.

* **Subjudul / Deskripsi Fungsional**:
  * Benar: `Shared Latent Intent Classification`
  * Benar: `Dense Gazetteer Linking & Latent Ellipsis Context`
  * Benar: `Zero Tool Schema Overhead (~400-700 ms) | DeepSeek-Style Prefix Caching`
  * Benar: `Pre-cached 8 SOP FTMM (~811 tok) | 0.0 ms Retrieval Overhead`
  * Benar: `Topological Placement, Semester Parity & Single Religion Dedup`
  * Benar: `Yielding SSE Event Tokens kata-per-kata & DegreePlanPayload JSON`

* **Pill Badges**:
  * Gunakan istilah ringkas baku: `0.0 ms Latency`, `~30 ms Latency`, `Fast Path`, `Zero-Retrieval`, `Tool-Calling`, `CSP Solver`, `Safety Invariant`, `SSE Streaming`.

---

## 4. Aturan Penegakan bagi Subagent & Evaluator

1. Setiap agent yang memodifikasi skrip diagram visual atau dokumentasi laporan **WAJIB membaca berkas ini terlebih dahulu**.
2. Jika terdeteksi istilah terjemahan harfiah yang kaku/aneh (*translation slop*), evaluator atau subagent reviewer **WAJIB menolak perubahan** (*Request Changes*) dan mengembalikan istilah ke bentuk Bahasa Inggris kanonikal sesuai tabel di atas.
