"""
Dataset Evaluasi Stratifikasi FTMM COMPASS (Subset A Formal vs Subset B Informal).
Mendukung Card 33, 34, dan 36 Trello.
Disimpan ke data/evaluation_dataset.json (terlindungi oleh .gitignore).
"""

import json
from pathlib import Path
from typing import Any

DATASET_PATH = Path("data/evaluation_dataset.json")

SUBSET_A_FORMAL = [
    {
        "id": "A-01",
        "category": "sks_limit",
        "formality": "formal",
        "question": "Berapa batas beban SKS maksimal yang boleh diprogram mahasiswa FTMM jika memperoleh IPS 3.25 pada semester sebelumnya?",
        "ground_truth": "Berdasarkan Pasal 5.4 Panduan Akademik FTMM, mahasiswa dengan IPS di atas 3.00 berhak memprogram beban studi maksimal 24 SKS.",
        "key_facts": ["24 SKS", "IPS > 3.00", "Pasal 5.4"]
    },
    {
        "id": "A-02",
        "category": "pkl",
        "formality": "formal",
        "question": "Apa saja syarat akademik untuk dapat mengajukan mata kuliah Praktik Kerja Lapangan (PKL) di FTMM?",
        "ground_truth": "Syarat akademik PKL FTMM adalah telah menempuh minimal 80 SKS kumulatif, duduk minimal di semester 5, dan durasi kerja lapangan minimal 18 hari kerja efektif.",
        "key_facts": ["80 SKS", "semester 5", "18 hari kerja"]
    },
    {
        "id": "A-03",
        "category": "skripsi",
        "formality": "formal",
        "question": "Berapa syarat beban SKS kumulatif dan IPK minimal untuk memprogram Proposal Skripsi di FTMM?",
        "ground_truth": "Mahasiswa dapat memprogram Proposal Skripsi setelah menempuh minimal 108 hingga 110 SKS dengan IPK minimal di atas 2.00 dan telah atau sedang menempuh Metode Penelitian.",
        "key_facts": ["108-110 SKS", "IPK > 2.00", "Metode Penelitian"]
    },
    {
        "id": "A-04",
        "category": "yudisium",
        "formality": "formal",
        "question": "Berapa ketentuan persentase maksimal nilai D dan nilai E yang diperbolehkan untuk syarat kelulusan yudisium sarjana FTMM?",
        "ground_truth": "Ketentuan yudisium sarjana FTMM menyatakan bahwa total nilai D maksimal 20% dari total SKS kumulatif, dan wajib memiliki 0 SKS nilai E (tidak boleh ada nilai E).",
        "key_facts": ["maksimal 20% nilai D", "0 SKS nilai E", "tidak boleh ada nilai E"]
    },
    {
        "id": "A-05",
        "category": "sop_rekomendasi",
        "formality": "formal",
        "question": "Bagaimana prosedur resmi pengajuan Surat Rekomendasi magang atau beasiswa di FTMM UNAIR?",
        "ground_truth": "Unduh format permohonan di bit.ly/FormatPengajuanFTMM, minta persetujuan Dosen Wali dan KPS, lampirkan scan KTM & KHS, kirim email ke kemahasiswaan@ftmm.unair.ac.id dengan subjek Rekomendasi_Nama_NIM (proses 3-7 hari kerja).",
        "key_facts": ["bit.ly/FormatPengajuanFTMM", "kemahasiswaan@ftmm.unair.ac.id", "Dosen Wali", "3-7 hari kerja"]
    },
    {
        "id": "A-06",
        "category": "sop_skma",
        "formality": "formal",
        "question": "Bagaimana alur pengajuan Surat Keterangan Mahasiswa Aktif (SKMA) melalui portal Cybercampus V2?",
        "ground_truth": "Akses portal Cybercampus V2 di mahasiswa.unair.ac.id, pilih menu pengajuan layanan Surat Keterangan Mahasiswa Aktif, lengkapi alasan pengajuan, simpan dan kirim, lalu pantau serta unduh berkas di menu HISTORI.",
        "key_facts": ["Cybercampus V2", "mahasiswa.unair.ac.id", "HISTORI"]
    },
    {
        "id": "A-07",
        "category": "syllabus",
        "formality": "formal",
        "question": "Apa saja materi dan topik bahasan utama dalam mata kuliah SIA104 Pengantar Pemrograman?",
        "ground_truth": "Mata kuliah SIA104 Pengantar Pemrograman membahas computational thinking, perancangan algoritma dasar, dan implementasi pemrograman prosedural menggunakan bahasa Python.",
        "key_facts": ["computational thinking", "algoritma dasar", "Python", "prosedural"]
    },
    {
        "id": "A-08",
        "category": "prerequisite",
        "formality": "formal",
        "question": "Apa prasyarat yang harus ditempuh sebelum mengambil mata kuliah MAS104 Metode Statistika?",
        "ground_truth": "Prasyarat mata kuliah MAS104 Metode Statistika adalah mata kuliah Pengantar Statistika (MAS120).",
        "key_facts": ["Pengantar Statistika", "MAS120"]
    },
    {
        "id": "A-09",
        "category": "syllabus",
        "formality": "formal",
        "question": "Berapa SKS beban mata kuliah SID206 Pengantar Basis Data dan apa saja materi pembelajarannya?",
        "ground_truth": "Mata kuliah SID206 Pengantar Basis Data berbobot 4 SKS, mencakup pemodelan ERD, normalisasi basis data, sintaks SQL DDL/DML, dan querying data relasional.",
        "key_facts": ["4 SKS", "ERD", "normalisasi", "SQL"]
    },
    {
        "id": "A-10",
        "category": "prerequisite",
        "formality": "formal",
        "question": "Mata kuliah apa yang menjadi prasyarat untuk memprogram mata kuliah SIA206 Algoritma Pemrograman II?",
        "ground_truth": "Prasyarat mata kuliah SIA206 Algoritma Pemrograman II adalah Algoritma Pemrograman I (SIA109).",
        "key_facts": ["Algoritma Pemrograman I", "SIA109"]
    },
    {
        "id": "A-11",
        "category": "syllabus",
        "formality": "formal",
        "question": "Apa yang dipelajari pada mata kuliah MAT110 Matematika Lanjut di program studi Teknologi Sains Data?",
        "ground_truth": "Mata kuliah MAT110 Matematika Lanjut mencakup materi deret tak hingga, persamaan diferensial, integral lipat, dan geometri ruang.",
        "key_facts": ["deret tak hingga", "persamaan diferensial", "integral lipat", "geometri ruang"]
    },
    {
        "id": "A-12",
        "category": "sop_cuti",
        "formality": "formal",
        "question": "Bagaimana alur pengajuan Cuti Akademik resmi bagi mahasiswa FTMM?",
        "ground_truth": "Pengajuan cuti dilakukan secara daring via Cybercampus V2 pada menu layanan Cuti Akademik dengan mengunggah berkas pendukung, kemudian memantau penerbitan SK Cuti pada tab HISTORI.",
        "key_facts": ["Cybercampus V2", "Cuti Akademik", "HISTORI"]
    },
    {
        "id": "A-13",
        "category": "yudisium",
        "formality": "formal",
        "question": "Berapa batas minimal SKS kelulusan untuk meraih gelar sarjana di lingkungan FTMM UNAIR?",
        "ground_truth": "Total beban studi minimal untuk kelulusan program sarjana di FTMM UNAIR adalah 144 SKS (maksimal 160 SKS).",
        "key_facts": ["144 SKS", "160 SKS"]
    },
    {
        "id": "A-14",
        "category": "skripsi",
        "formality": "formal",
        "question": "Apa prasyarat untuk menempuh mata kuliah Skripsi (PNJ499) di FTMM?",
        "ground_truth": "Prasyarat mengambil mata kuliah Skripsi adalah telah lulus ujian Proposal Skripsi dan telah menempuh beban studi minimal 120 SKS.",
        "key_facts": ["Proposal Skripsi", "120 SKS"]
    },
    {
        "id": "A-15",
        "category": "prodi",
        "formality": "formal",
        "question": "Sebutkan 5 program studi sarjana yang bernaung di bawah Fakultas Teknologi Maju dan Multidisiplin UNAIR!",
        "ground_truth": "Lima program studi sarjana FTMM UNAIR adalah: Teknologi Sains Data (TSD), Teknik Robotika dan Kecerdasan Buatan (TRKB), Rekayasa Nanoteknologi (RN), Teknik Industri (TI), dan Teknik Elektro (TE).",
        "key_facts": ["Teknologi Sains Data", "Teknik Robotika", "Rekayasa Nanoteknologi", "Teknik Industri", "Teknik Elektro"]
    },
]

SUBSET_B_INFORMAL = [
    {
        "id": "B-01",
        "category": "sks_limit",
        "formality": "informal",
        "question": "kalo semester lalu ipk ku 3.4 dapet jatah sks berapa ya?",
        "ground_truth": "Berdasarkan aturan akademik FTMM, jika IPS di atas 3.00, kamu berhak mengambil maksimal 24 SKS di semester berikutnya.",
        "key_facts": ["24 SKS", "IPS > 3.00"]
    },
    {
        "id": "B-02",
        "category": "pkl",
        "formality": "informal",
        "question": "minimal udah lulus berapa sks bro biar bisa daftar pkl magang?",
        "ground_truth": "Untuk bisa mengajukan PKL di FTMM, minimal harus sudah menempuh 80 SKS dan duduk minimal di semester 5.",
        "key_facts": ["80 SKS", "semester 5"]
    },
    {
        "id": "B-03",
        "category": "skripsi",
        "formality": "informal",
        "question": "mau nanya sempro ta di ftmm syarat sks nya mentok minimal berapa?",
        "ground_truth": "Syarat mengambil Proposal Skripsi adalah minimal sudah menempuh 108 sampai 110 SKS dengan IPK minimal di atas 2.00.",
        "key_facts": ["108-110 SKS", "IPK > 2.00"]
    },
    {
        "id": "B-04",
        "category": "yudisium",
        "formality": "informal",
        "question": "kalo ada nilai d sama e pas mau yudisium lulus sarjana gimana aturannya?",
        "ground_truth": "Nilai D maksimal boleh 20% dari total SKS kumulatif, sedangkan nilai E sama sekali tidak boleh ada (wajib 0 SKS).",
        "key_facts": ["maksimal 20% nilai D", "0 SKS nilai E", "tidak boleh ada nilai E"]
    },
    {
        "id": "B-05",
        "category": "sop_rekomendasi",
        "formality": "informal",
        "question": "cara minta surat rekomendasi buat msib magang kemana ya ngirimnya?",
        "ground_truth": "Download template di bit.ly/FormatPengajuanFTMM, minta tanda tangan dosen wali dan KPS, lalu kirim email ke kemahasiswaan@ftmm.unair.ac.id.",
        "key_facts": ["bit.ly/FormatPengajuanFTMM", "kemahasiswaan@ftmm.unair.ac.id", "Dosen Wali"]
    },
    {
        "id": "B-06",
        "category": "sop_skma",
        "formality": "informal",
        "question": "skma buat daftar beasiswa ngurusnya lewat mana sih?",
        "ground_truth": "Ajukan lewat portal Cybercampus V2 (mahasiswa.unair.ac.id) di menu pengajuan layanan SKMA, nanti berkasnya bisa diunduh di tab HISTORI.",
        "key_facts": ["Cybercampus V2", "mahasiswa.unair.ac.id", "HISTORI"]
    },
    {
        "id": "B-07",
        "category": "syllabus",
        "formality": "informal",
        "question": "matkul pengantar pemrograman sia104 belajar apa aja sih?",
        "ground_truth": "Matkul Pengantar Pemrograman (SIA104) mempelajari computational thinking, dasar-dasar algoritma, dan pemrograman Python prosedural.",
        "key_facts": ["computational thinking", "algoritma dasar", "Python"]
    },
    {
        "id": "B-08",
        "category": "prerequisite",
        "formality": "informal",
        "question": "kalo mau ambil metode statistika harus lulus matkul apa dulu?",
        "ground_truth": "Prasyarat untuk mengambil Metode Statistika adalah harus lulus mata kuliah Pengantar Statistika.",
        "key_facts": ["Pengantar Statistika"]
    },
    {
        "id": "B-09",
        "category": "syllabus",
        "formality": "informal",
        "question": "basis data sid206 berapa sks ya dan materinya ngapain aja?",
        "ground_truth": "Pengantar Basis Data (SID206) berbobot 4 SKS, materinya tentang ERD, normalisasi tabel, dan perintah query SQL.",
        "key_facts": ["4 SKS", "ERD", "normalisasi", "SQL"]
    },
    {
        "id": "B-10",
        "category": "prerequisite",
        "formality": "informal",
        "question": "prasyarat alpro 2 tuh harus kelar apa dulu?",
        "ground_truth": "Untuk mengambil Alpro 2 (Algoritma Pemrograman II), prasyaratnya harus sudah lulus Algoritma Pemrograman I.",
        "key_facts": ["Algoritma Pemrograman I"]
    },
    {
        "id": "B-11",
        "category": "syllabus",
        "formality": "informal",
        "question": "matematika lanjut mat110 di tsd itu ngebahas materi apa aja ya?",
        "ground_truth": "Matematika Lanjut di TSD mempelajari deret tak hingga, persamaan diferensial, integral lipat, dan geometri ruang.",
        "key_facts": ["deret tak hingga", "persamaan diferensial", "integral lipat"]
    },
    {
        "id": "B-12",
        "category": "sop_cuti",
        "formality": "informal",
        "question": "gimana cara ngajuin cuti kuliah di kampus ftmm?",
        "ground_truth": "Buka Cybercampus V2, pilih menu Cuti Akademik, isi alasan dan upload berkas pendukung, lalu tunggu SK Cuti terbit di tab Histori.",
        "key_facts": ["Cybercampus V2", "Cuti Akademik"]
    },
    {
        "id": "B-13",
        "category": "yudisium",
        "formality": "informal",
        "question": "buat lulus sarjana di ftmm minimal harus ngumpulin berapa sks sih?",
        "ground_truth": "Syarat lulus sarjana FTMM minimal harus mengumpulkan 144 SKS.",
        "key_facts": ["144 SKS"]
    },
    {
        "id": "B-14",
        "category": "skripsi",
        "formality": "informal",
        "question": "syarat biar bisa ambil skripsi matkul apa dan berapa sks?",
        "ground_truth": "Syarat memprogram Skripsi adalah sudah lulus Proposal Skripsi dan menempuh minimal 120 SKS.",
        "key_facts": ["Proposal Skripsi", "120 SKS"]
    },
    {
        "id": "B-15",
        "category": "prodi",
        "formality": "informal",
        "question": "di ftmm ada prodi apa aja sih jurusannya?",
        "ground_truth": "Prodi di FTMM ada 5: Sains Data (TSD), Robotika dan AI (TRKB), Nanoteknologi (RN), Teknik Industri (TI), dan Teknik Elektro (TE).",
        "key_facts": ["Sains Data", "Robotika", "Nanoteknologi", "Teknik Industri", "Teknik Elektro"]
    },
]


def build_and_save_dataset() -> dict[str, Any]:
    """Menyimpan dataset evaluasi lengkap ke disk."""
    DATASET_PATH.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "metadata": {
            "title": "FTMM COMPASS Hybrid RAG-CAG Evaluation Dataset",
            "total_samples": len(SUBSET_A_FORMAL) + len(SUBSET_B_INFORMAL),
            "subset_a_count": len(SUBSET_A_FORMAL),
            "subset_b_count": len(SUBSET_B_INFORMAL),
        },
        "subset_a_formal": SUBSET_A_FORMAL,
        "subset_b_informal": SUBSET_B_INFORMAL,
    }
    with open(DATASET_PATH, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    return payload


def load_dataset() -> dict[str, Any]:
    """Memuat dataset evaluasi dari disk atau membuatnya jika belum ada."""
    if not DATASET_PATH.exists():
        return build_and_save_dataset()
    with open(DATASET_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


if __name__ == "__main__":
    data = build_and_save_dataset()
    print(f"Dataset berhasil disimpan ke {DATASET_PATH}!")
    print(f"  - Subset A (Formal): {data['metadata']['subset_a_count']} sampel")
    print(f"  - Subset B (Informal): {data['metadata']['subset_b_count']} sampel")
