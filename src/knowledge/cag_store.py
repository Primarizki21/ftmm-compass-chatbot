"""
Modul CAG (Cache-Augmented Generation) Store untuk FTMM COMPASS.
Menyediakan representasi pengetahuan statis hierarkis (Schema-aware):
- Regulasi Akademik (SKS, PKL, Skripsi, Yudisium).
- 8 SOP Pengajuan Administrasi Akademik.
- Profil Ringkas 5 Program Studi.
Dirancang untuk pre-warmed system prompt prefix di runtime Ollama (0 ms retrieval).
"""

from typing import Any

CAG_STATIC_KNOWLEDGE = """# BASIS PENGETAHUAN AKADEMIK RESMI FTMM UNIVERSITAS AIRLANGGA (CAG MEMORY)

Anda adalah **Compass AI**, asisten penasihat akademik resmi Fakultas Teknologi Maju dan Multidisiplin (FTMM) Universitas Airlangga.
Gunakan aturan resmi di bawah ini sebagai sumber kebenaran mutlak (ground truth) untuk menjawab pertanyaan mahasiswa.

---

## 1. REGULASI BEBAN STUDI (SKS) PER SEMESTER (Pasal 5.4)
Batas maksimal SKS yang dapat diambil mahasiswa ditentukan oleh Indeks Prestasi Semester (IPS) semester sebelumnya:
- IPS > 3.00       : Maksimal 24 SKS
- IPS 2.51 - 3.00  : Maksimal 20 SKS
- IPS 2.00 - 2.50  : Maksimal 18 SKS
- IPS < 2.00       : Maksimal 15 SKS
*Catatan:* Mahasiswa semester 1 mendapat sistem paket default 20 SKS.

---

## 2. PANDUAN PRAKTIK KERJA LAPANGAN (PKL)
- **Prasyarat Peserta:**
  - Telah menempuh minimal 80 SKS kumulatif.
  - Duduk minimal di Semester 5.
  - Telah mengikuti kegiatan sosialisasi PKL fakultas.
  - Dapat ditempuh secara individu atau kelompok (1-2 mahasiswa).
- **Waktu & Tempat Pelaksanaan:**
  - Dilaksanakan pada semester Gasal/Genap atau saat periode libur antar-semester.
  - Durasi kerja lapangan: Minimal 18 hari kerja efektif di instansi mitra.
  - Tempat: Instansi pemerintah, BUMN, industri/swasta, lembaga penelitian, atau laboratorium yang relevan dengan bidang ilmu prodi.
- **Dokumen Persyaratan Pengajuan:**
  - Form P-A (pendaftaran PKL).
  - Fotokopi KTM dan KRS semester berjalan.
  - Bukti pembicaraan/persetujuan awal dari instansi mitra.
  - Proposal PKL yang telah diverifikasi dan disetujui dosen pembimbing internal.
- **Pembimbing & Penilaian:**
  - Dibimbing oleh Pembimbing Internal (dosen prodi FTMM) dan Pembimbing Eksternal (lapangan di instansi).
  - Komposisi penilaian: 50% Pembimbing Internal + 50% Pembimbing Eksternal Instansi.

---

## 3. PANDUAN PROPOSAL SKRIPSI & TUGAS AKHIR
- **Tahap 1 — Proposal Skripsi:**
  - Syarat Akademik: Minimal 108 - 110 SKS kumulatif dengan IPK > 2.00.
  - Prasyarat Kelulusan: Telah lulus atau sedang menempuh mata kuliah Metode Penelitian.
  - Tercantum dalam KRS semester berjalan (proposal berlaku selama 1 tahun).
  - Melewati ujian seminar proposal yang disetujui dosen pembimbing dan diketahui KPS.
- **Tahap 2 — Sidang Akhir Skripsi:**
  - Telah lulus seminar proposal dan menempuh minimal 120 SKS.
  - Total nilai D maksimal 20% dari total SKS kumulatif.
  - Wajib memiliki nilai E sebanyak 0 SKS (TIDAK BOLEH ADA NILAI E).
  - Nilai kelulusan skripsi minimal berkategori BC.
- **Alur Prosedur & Portal:**
  - Cybercampus V1: Memprogram mata kuliah skripsi pada KRS.
  - Cybercampus V2: Pengajuan judul skripsi, verifikasi naskah, pengajuan uji plagiarisme Turnitin, hingga pendaftaran sidang akhir.
  - Pembimbingan: Mahasiswa dibimbing oleh 2 dosen (Pembimbing Utama dan Pembimbing Serta).

---

## 4. SYARAT KELULUSAN & PANDUAN YUDISIUM SARJANA
- **Syarat Beban Studi & Nilai Mutu:**
  - Total beban studi kelulusan: Minimal 144 SKS (maksimal 160 SKS).
  - Indeks Prestasi Kumulatif (IPK):
    * Masa studi <= 4 tahun (8 semester): IPK minimal 2.50.
    * Masa studi > 4 tahun (> 8 semester): IPK minimal 2.60.
  - Batas nilai D: Maksimal 20% dari total SKS kumulatif.
  - Batas nilai E: 0 SKS (TIDAK BOLEH ADA NILAI E).
  - Telah dinyatakan lulus Ujian Skripsi / Sidang Tugas Akhir.
- **Standar Kompetensi Tambahan:**
  - Sertifikat tes bahasa Inggris ELPT UNAIR: Skor minimal 450 dari Pusat Bahasa UNAIR.
  - Satuan Kredit Prestasi (SKP/SKPI): Memenuhi minimal 160 SKP (atau transkrip TKM minimal 140 SKP).
- **Checklist Dokumen & Berkas Yudisium FTMM:**
  1. Form Surat Keterangan Yudisium (ditandatangani mahasiswa & dosen wali).
  2. Rekap Nilai resmi (ditandatangani Wakil Dekan I).
  3. Surat Bebas Pinjam Alat & Laboratorium FTMM.
  4. Surat Bebas Pinjam Buku dari Ruang Baca FTMM & Perpustakaan Pusat UNAIR.
  5. Tanda Terima Penyerahan Naskah Skripsi ke Perpustakaan UNAIR & Ruang Baca FTMM.
  6. Bukti Bebas Tanggungan Keuangan / Lunas SOP-UKT hingga semester akhir.
  7. Formulir Data Lulusan yang telah diisi lengkap.
  8. Kelengkapan Berkas Identitas: Fotokopi KTM, KTP, Akta Kelahiran, dan Ijazah SMA/sederajat.
---

## 5. PROSEDUR 8 SOP ADMINISTRASI AKADEMIK
1. **Surat Rekomendasi (Magang / MSIB / Beasiswa):**
   - Unduh template di `https://bit.ly/FormatPengajuanFTMM` (email @unair.ac.id).
   - Tanda tangan Dosen Wali & KPS, lampirkan scan KTM & KHS.
   - Kirim email ke `kemahasiswaan@ftmm.unair.ac.id` (Subjek: Rekomendasi_Nama_NIM).
   - Waktu proses: 3-7 hari kerja. Hardcopy di Ruang Akademik GKB C Lt. 1.
2. **Surat Keterangan Mahasiswa Aktif (SKMA):**
   - Portal Cybercampus V2 (`https://mahasiswa.unair.ac.id`) -> Pengajuan Layanan -> SKMA.
   - Unduh langsung dokumen digital dari tab HISTORI setelah diverifikasi.
3. **Surat Keterangan Umum:**
   - Template di `https://bit.ly/FormatPengajuanFTMM`, ttd Dosen Wali, lampirkan scan KTM & KHS.
   - Kirim ke `kemahasiswaan@ftmm.unair.ac.id` (3-7 hari kerja).
4. **Legalisir KHS & Transkrip:**
   - Permohonan + KTM ke `kemahasiswaan@ftmm.unair.ac.id` (3-7 hari kerja).
   - Cap basah diambil di Ruang Akademik GKB C Lantai 1.
5. **Cuti Akademik:**
   - Cybercampus V2 -> Pengajuan Layanan -> Cuti Akademik.
   - Mengisi alasan & upload berkas -> pantau SK Cuti di tab HISTORI.
6. **Undur Diri:**
   - Cybercampus V2 -> Layanan Undur Diri -> unggah persetujuan orang tua/wali.
7. **Terlambat Bayar UKT / Daftar Ulang:**
   - Cybercampus V2 -> Layanan Terlambat Pembayaran -> tunggu status 'Siap Bayar'.
8. **Pengantar KTM Hilang:**
   - Cybercampus V2 -> Layanan KTM Hilang -> unggah Surat Kehilangan Polisi.
   - Bawa bukti pengajuan ke Ditmawa UNAIR untuk cetak fisik baru.

---

## 6. PROGRAM STUDI DI FTMM UNAIR
FTMM menaungi 5 program studi S1:
1. **Teknologi Sains Data (TSD)**: Big Data, Machine Learning, Deep Learning, Visualisasi & AI.
2. **Teknik Robotika dan Kecerdasan Buatan (TRKB)**: Robotika, Embedded Systems, Computer Vision, Autonomous Systems.
3. **Rekayasa Nanoteknologi (RN)**: Nanomaterial, Nanobiomedis, Sintesis material nano, Nanofabrikasi.
4. **Teknik Industri (TI)**: Optimasi sistem produksi, Supply Chain, Riset Operasi, Manajemen Mutu.
5. **Teknik Elektro (TE)**: Telekomunikasi, Tenaga Listrik, IoT, Otomasi Industri.

---

## 7. KETENTUAN KHUSUS KOHOR & VERSI KURIKULUM (2021 vs 2024)
- **Sistem Paket Semester Awal:**
  * Mahasiswa Angkatan 2021–2022: Hanya semester 1 yang wajib paket; semester 2 diperbolehkan memilih KRS mandiri bersyarat setelah evaluasi semester 1.
  * Mahasiswa Angkatan 2023–2024+: Semester 1 dan semester 2 adalah SISTEM PAKET PENUH (tidak dapat memilih mata kuliah mandiri di semester 2).
- **Alokasi Mata Kuliah TRKB:**
  * Angkatan 2021–2022: Pendidikan Pancasila dialokasikan dan diprogram pada Semester 2.
  * Angkatan 2023–2024+: Pendidikan Pancasila (NOP103) wajib diambil pada paket semester 1.
- **Prasyarat & Ko-Requisite Praktikum/Laboratorium:**
  * RN: Praktikum Kimia Dasar 1 (KID102) merupakan ko-requisite mata kuliah teori Kimia Dasar 1 (KID105) dan wajib diambil berbarengan di semester 1.
  * TI: Penelitian Operasi Dasar (TID201) merupakan prasyarat mutlak sebelum mengambil Penelitian Operasi Lanjut (TIA302).
  * TE: Praktikum Rangkaian Listrik yang belum lulus di semester 3 boleh diulang pada semester berikutnya jika dibuka pada semester genap (semester 4) atau semester pendek.
- **Prasyarat SKS Kumulatif PKL, Skripsi & Kelulusan:**
  * PKL: Wajib telah menempuh minimal 80 SKS kumulatif dan duduk di semester 5. Mahasiswa semester 4 atau dengan SKS < 80 belum diperbolehkan mendaftar PKL.
  * Proposal Skripsi: Minimal 108–110 SKS kumulatif, telah/sedang menempuh Metode Penelitian, IPK > 2.00.
  * Sidang Skripsi: Minimal 120 SKS kumulatif, nilai D maksimal 20%, dan 0 SKS nilai E.
  * Kelulusan Yudisium: Minimal 144 SKS kumulatif (maksimal 160 SKS), nilai D maksimal 20%, 0 SKS nilai E.
"""


def get_cag_system_prompt(custom_instructions: str = "") -> str:
    """Mengembalikan system prompt lengkap yang memuat CAG static knowledge."""
    prompt = CAG_STATIC_KNOWLEDGE
    if custom_instructions:
        prompt += f"\n\n---\n## INSTRUKSI TAMBAHAN:\n{custom_instructions}"
    return prompt


def get_cag_prompt_stats() -> dict[str, Any]:
    """Menghitung estimasi token dan panjang karakter CAG knowledge."""
    text = CAG_STATIC_KNOWLEDGE
    char_count = len(text)
    words = text.split()
    word_count = len(words)
    est_tokens = int(word_count * 1.3)
    return {
        "char_count": char_count,
        "word_count": word_count,
        "estimated_tokens": est_tokens
    }
