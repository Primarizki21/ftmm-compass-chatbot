#!/usr/bin/env python3
"""
Build Retrieval Complexity Benchmark Dataset (Li & Iwata 2026 Taxonomy).
arXiv:2606.02983:
1. Lexical Match (30 queries) - Exact keyword / code overlap
2. Semantic Paraphrase (30 queries) - Colloquial / slang / concept without direct code
3. Multi-Evidence Synthesis (30 queries) - >= 2 distinct course chunks required
4. Out-of-Scope (30 queries) - Non-FTMM programs to evaluate Refusal Rate
Total: 120 queries stratified across TSD, TRKB, RN, TI, TE.
"""

import json
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("build_retrieval_gt")

OUTPUT_PATH = Path("data/ground_truth/retrieval_complexity_benchmark.json")

# 1. LEXICAL MATCH (30 queries: 6 per prodi)
LEXICAL_QUERIES = [
    # TSD
    {"prodi": "TSD", "query": "Apa saja materi pokok mata kuliah Pengantar Statistika MAS120 di TSD?", "kode": ["MAS120"], "nama": ["Pengantar Statistika"]},
    {"prodi": "TSD", "query": "Berapa SKS beban mata kuliah Metode Statistika MAS104 semester 2?", "kode": ["MAS104"], "nama": ["Metode Statistika"]},
    {"prodi": "TSD", "query": "Mata kuliah Algoritma Pemrograman I SIA107 di semester berapa?", "kode": ["SIA107"], "nama": ["Algoritma Pemrograman I"]},
    {"prodi": "TSD", "query": "Berapa SKS mata kuliah Matematika Diskrit MAL204 di prodi Sains Data?", "kode": ["MAL204"], "nama": ["Matematika Diskrit"]},
    {"prodi": "TSD", "query": "Silabus mata kuliah Pengantar Kecerdasan Buatan SIC203 membahas apa saja?", "kode": ["SIC203"], "nama": ["Pengantar Kecerdasan Buatan"]},
    {"prodi": "TSD", "query": "Mata kuliah Statistika Nonparametrik MAS245 ditawarkan pada semester berapa?", "kode": ["MAS245"], "nama": ["Statistika Nonparametrik"]},
    # TRKB
    {"prodi": "TRKB", "query": "Apa materi praktikum Eksperimen Robotika I RKS204 di TRKB?", "kode": ["RKS204"], "nama": ["Eksperimen Robotika I"]},
    {"prodi": "TRKB", "query": "Berapa SKS mata kuliah Sensor dan Aktuator RKS202 semester 4?", "kode": ["RKS202"], "nama": ["Sensor dan Aktuator"]},
    {"prodi": "TRKB", "query": "Mata kuliah Mobile Robot RKS301 di TRKB mempelajari apa?", "kode": ["RKS301"], "nama": ["Mobile Robot"]},
    {"prodi": "TRKB", "query": "Kapan mahasiswa TRKB memprogram Robot Manipulator RKS302?", "kode": ["RKS302"], "nama": ["Robot Manipulator"]},
    {"prodi": "TRKB", "query": "Apa saja topik dalam mata kuliah Kecerdasan Buatan MAT302 TRKB?", "kode": ["MAT302"], "nama": ["Kecerdasan Buatan"]},
    {"prodi": "TRKB", "query": "Berapa SKS beban mata kuliah Eksperimen Robotika II RKS304?", "kode": ["RKS304"], "nama": ["Eksperimen Robotika II"]},
    # RN
    {"prodi": "RN", "query": "Berapa SKS mata kuliah Kimia Dasar KID105 di Rekayasa Nanoteknologi?", "kode": ["KID105"], "nama": ["Kimia Dasar"]},
    {"prodi": "RN", "query": "Apa materi dalam mata kuliah Fisika Dasar 1 FID101 semester 1?", "kode": ["FID101"], "nama": ["Fisika Dasar 1"]},
    {"prodi": "RN", "query": "Berapa SKS beban Praktik Kimia Dasar 1 KID102 di prodi RN?", "kode": ["KID102"], "nama": ["Praktikum Kimia Dasar 1"]},
    {"prodi": "RN", "query": "Kapan jadwal semester mata kuliah Fisika Dasar 2 FID104 di RN?", "kode": ["FID104"], "nama": ["Fisika Dasar 2"]},
    {"prodi": "RN", "query": "Mata kuliah Kimia Dasar 2 KID103 mempelajari topik apa saja?", "kode": ["KID103"], "nama": ["Kimia Dasar 2"]},
    {"prodi": "RN", "query": "Berapa SKS beban mata kuliah Praktikum Fisika Dasar 1 FID102?", "kode": ["FID102"], "nama": ["Praktikum Fisika Dasar 1"]},
    # TI
    {"prodi": "TI", "query": "Apa silabus pokok mata kuliah Ergonomi TIA201 di Teknik Industri?", "kode": ["TIA201"], "nama": ["Ergonomi"]},
    {"prodi": "TI", "query": "Berapa SKS mata kuliah Penelitian Operasi Dasar TID201 semester 4?", "kode": ["TID201"], "nama": ["Penelitian Operasi Dasar"]},
    {"prodi": "TI", "query": "Mata kuliah Otomasi Industri TIU201 membahas apa di Teknik Industri?", "kode": ["TIU201"], "nama": ["Otomasi Industri"]},
    {"prodi": "TI", "query": "Kapan mata kuliah Statistik Industri TID202 ditawarkan di kurikulum TI?", "kode": ["TID202"], "nama": ["Statistik Industri"]},
    {"prodi": "TI", "query": "Berapa SKS beban mata kuliah Simulasi Komputer TIK301?", "kode": ["TIK301"], "nama": ["Simulasi Komputer"]},
    {"prodi": "TI", "query": "Mata kuliah Penelitian Operasi Lanjut TIA302 berada di semester berapa?", "kode": ["TIA302"], "nama": ["Penelitian Operasi Lanjut"]},
    # TE
    {"prodi": "TE", "query": "Apa isi materi mata kuliah Rangkaian Listrik TBE201 di Teknik Elektro?", "kode": ["TBE201"], "nama": ["Rangkaian Listrik"]},
    {"prodi": "TE", "query": "Berapa SKS mata kuliah Sistem Tenaga Listrik TEE202 semester 4?", "kode": ["TEE202"], "nama": ["Sistem Tenaga Listrik"]},
    {"prodi": "TE", "query": "Mata kuliah Medan Elektromagnetik TBO201 mempelajari apa?", "kode": ["TBO201"], "nama": ["Medan Elektromagnetik"]},
    {"prodi": "TE", "query": "Kapan jadwal pengambilan mata kuliah Sinyal dan Sistem TRS203 di TE?", "kode": ["TRS203"], "nama": ["Sinyal dan Sistem"]},
    {"prodi": "TE", "query": "Berapa SKS beban Dasar Sistem dan Jaringan Telekomunikasi TET201?", "kode": ["TET201"], "nama": ["Dasar Sistem dan Jaringan Telekomunikasi"]},
    {"prodi": "TE", "query": "Apa materi utama praktikum Praktikum Rangkaian Listrik TBE202?", "kode": ["TBE202"], "nama": ["Praktikum Rangkaian Listrik"]},
]

# 2. SEMANTIC PARAPHRASE (30 queries: 6 per prodi)
PARAPHRASE_QUERIES = [
    # TSD
    {"prodi": "TSD", "query": "matkul koding paling dasar buat anak data science belajar apa ya?", "kode": ["SIA107"], "nama": ["Algoritma Pemrograman I"]},
    {"prodi": "TSD", "query": "di sains data belajar uji hipotesis dan sebaran probabilitas di matkul apa?", "kode": ["MAS104"], "nama": ["Metode Statistika"]},
    {"prodi": "TSD", "query": "kuliah yang ngajarin logika matematika sama teori graf di semester 3 namanya apa?", "kode": ["MAL204"], "nama": ["Matematika Diskrit"]},
    {"prodi": "TSD", "query": "kalo mau belajar pengenalan AI dan agen cerdas di semester 4 ambil apa bro?", "kode": ["SIC203"], "nama": ["Pengantar Kecerdasan Buatan"]},
    {"prodi": "TSD", "query": "matkul statistika yang datanya bebas distribusi atau tanpa asumsi normalitas itu apa?", "kode": ["MAS245"], "nama": ["Statistika Nonparametrik"]},
    {"prodi": "TSD", "query": "pelajaran statistika awal banget di semester 1 tsd itu bahasannya seputar apa?", "kode": ["MAS120"], "nama": ["Pengantar Statistika"]},
    # TRKB
    {"prodi": "TRKB", "query": "matkul yang ngebahas cara kerja komponen deteksi fisik dan penggerak motorik robot apa ya?", "kode": ["RKS202"], "nama": ["Sensor dan Aktuator"]},
    {"prodi": "TRKB", "query": "kalo mau bikin robot beroda yang bisa jalan otonom di lab robotika itu di matkul apa?", "kode": ["RKS301"], "nama": ["Mobile Robot"]},
    {"prodi": "TRKB", "query": "praktikum hands-on pertama kali ngerakit kit robot di semester genap namanya apa?", "kode": ["RKS204"], "nama": ["Eksperimen Robotika I"]},
    {"prodi": "TRKB", "query": "studi tentang lengan robot industri dan kinematika gerak sendi dipelajari dimana?", "kode": ["RKS302"], "nama": ["Robot Manipulator"]},
    {"prodi": "TRKB", "query": "dasar algoritma cerdas kayak search tree dan logika fuzzy buat robotik itu matkul apa?", "kode": ["MAT302"], "nama": ["Kecerdasan Buatan"]},
    {"prodi": "TRKB", "query": "lab lanjutan robotika buat implementasi sistem navigasi cerdas semester 5 apa ya?", "kode": ["RKS304"], "nama": ["Eksperimen Robotika II"]},
    # RN
    {"prodi": "RN", "query": "kuliah sains atom dan reaksi kimia tingkat awal buat maba nanotek apa ya?", "kode": ["KID105"], "nama": ["Kimia Dasar"]},
    {"prodi": "RN", "query": "matkul pengantar gerak mekanika dan hukum newton pas semester satu namanya apa?", "kode": ["FID101"], "nama": ["Fisika Dasar 1"]},
    {"prodi": "RN", "query": "kegiatan laboratorium nyampur larutan kimia pertama kali di semester 1 apa namanya?", "kode": ["KID102"], "nama": ["Praktikum Kimia Dasar 1"]},
    {"prodi": "RN", "query": "lanjutan teori fisika gelombang dan termodinamika di semester dua nanoteknologi apa?", "kode": ["FID104"], "nama": ["Fisika Dasar 2"]},
    {"prodi": "RN", "query": "matkul reaksi senyawa anorganik dan termokimia lanjutan semester 2 apa kak?", "kode": ["KID103"], "nama": ["Kimia Dasar 2"]},
    {"prodi": "RN", "query": "praktikum ukur osilasi pendulum sama momen inersia semester satu RN itu apa?", "kode": ["FID102"], "nama": ["Praktikum Fisika Dasar 1"]},
    # TI
    {"prodi": "TI", "query": "kuliah perancangan stasiun kerja biar ga bikin pegel pekerja pabrik namanya apa?", "kode": ["TIA201"], "nama": ["Ergonomi"]},
    {"prodi": "TI", "query": "matkul pemodelan matematika linier programming buat maksimalkan profit industri apa ya?", "kode": ["TID201"], "nama": ["Penelitian Operasi Dasar"]},
    {"prodi": "TI", "query": "teknologi PLC dan ban berjalan di lantai produksi industri manufaktur dibahas di matkul apa?", "kode": ["TIU201"], "nama": ["Otomasi Industri"]},
    {"prodi": "TI", "query": "pengolahan data sampel cacat produk pabrik pakai distribusi probabilitas di TI apa namanya?", "kode": ["TID202"], "nama": ["Statistik Industri"]},
    {"prodi": "TI", "query": "bikin model simulasi antrian pabrik pakai software arena atau promodel di semester 6 apa?", "kode": ["TIK301"], "nama": ["Simulasi Komputer"]},
    {"prodi": "TI", "query": "teknik riset operasi nonlinier sama antrian stok logistik tingkat lanjut matkul apa?", "kode": ["TIA302"], "nama": ["Penelitian Operasi Lanjut"]},
    # TE
    {"prodi": "TE", "query": "analisis hukum kirchoff arus dan tegangan ac dc di semester 3 elektro apa ya?", "kode": ["TBE201"], "nama": ["Rangkaian Listrik"]},
    {"prodi": "TE", "query": "studi transmisi gardu induk sama distribusi trafo listrik tegangan tinggi matkul apa?", "kode": ["TEE202"], "nama": ["Sistem Tenaga Listrik"]},
    {"prodi": "TE", "query": "persamaan maxwell sama gelombang elektromagnetik rambat di ruang hampa elektro apa?", "kode": ["TBO201"], "nama": ["Medan Elektromagnetik"]},
    {"prodi": "TE", "query": "analisis transformasi fourier dan laplace untuk representasi sinyal waktu kontinyu apa?", "kode": ["TRS203"], "nama": ["Sinyal dan Sistem"]},
    {"prodi": "TE", "query": "dasar protokol komunikasi kabel fiber optik dan transmisi data telekomunikasi apa?", "kode": ["TET201"], "nama": ["Dasar Sistem dan Jaringan Telekomunikasi"]},
    {"prodi": "TE", "query": "praktik solder komponen resistor kapasitor di breadboard lab elektro semester 3 apa?", "kode": ["TBE202"], "nama": ["Praktikum Rangkaian Listrik"]},
]

# 3. MULTI-EVIDENCE SYNTHESIS (30 queries: 6 per prodi)
MULTI_EVIDENCE_QUERIES = [
    # TSD
    {
        "prodi": "TSD",
        "query": "Bandingkan prasyarat dan bobot SKS antara Pengantar Statistika dengan Metode Statistika di TSD",
        "kode": ["MAS120", "MAS104"],
        "nama": ["Pengantar Statistika", "Metode Statistika"],
        "required_synthesis": "Membandingkan SKS dan prasyarat MAS120 (Sem 1, 3 SKS, tanpa syarat) vs MAS104 (Sem 2, 3 SKS, syarat MAS120)",
    },
    {
        "prodi": "TSD",
        "query": "Bagaimana keterkaitan materi Matematika dengan Matematika Diskrit di kurikulum Sains Data?",
        "kode": ["MAT108", "MAL204"],
        "nama": ["Matematika", "Matematika Diskrit"],
        "required_synthesis": "Membahas kontinuitas kalkulus MAT108 semester 2 terhadap struktur diskrit MAL204 semester 3",
    },
    {
        "prodi": "TSD",
        "query": "Apa perbedaan fokus pembelajaran antara Algoritma Pemrograman I dan Pengantar Kecerdasan Buatan?",
        "kode": ["SIA107", "SIC203"],
        "nama": ["Algoritma Pemrograman I", "Pengantar Kecerdasan Buatan"],
        "required_synthesis": "SIA107 fokus ke sintaks Python prosedural dasar, sedangkan SIC203 fokus ke penalaran agen cerdas",
    },
    {
        "prodi": "TSD",
        "query": "Jelaskan urutan semester pengambilan Metode Statistika dan Statistika Nonparametrik di TSD",
        "kode": ["MAS104", "MAS245"],
        "nama": ["Metode Statistika", "Statistika Nonparametrik"],
        "required_synthesis": "MAS104 diambil di semester 2 sebagai dasar parametrik, disusul MAS245 di semester 3 untuk analisis nonparametrik",
    },
    {
        "prodi": "TSD",
        "query": "Berapa total akumulasi SKS jika mahasiswa mengambil Pengantar Statistika dan Algoritma Pemrograman I?",
        "kode": ["MAS120", "SIA107"],
        "nama": ["Pengantar Statistika", "Algoritma Pemrograman I"],
        "required_synthesis": "Akumulasi SKS: MAS120 (3 SKS) + SIA107 (2 SKS) = total 5 SKS",
    },
    {
        "prodi": "TSD",
        "query": "Bandingkan cakupan materi antara Matematika Diskrit dan Pengantar Kecerdasan Buatan di TSD",
        "kode": ["MAL204", "SIC203"],
        "nama": ["Matematika Diskrit", "Pengantar Kecerdasan Buatan"],
        "required_synthesis": "MAL204 memberikan landasan teori graf dan logika proposisi yang digunakan pada search tree SIC203",
    },
    # TRKB
    {
        "prodi": "TRKB",
        "query": "Bandingkan materi dan bobot SKS antara Eksperimen Robotika I dengan Eksperimen Robotika II di TRKB",
        "kode": ["RKS204", "RKS304"],
        "nama": ["Eksperimen Robotika I", "Eksperimen Robotika II"],
        "required_synthesis": "RKS204 berbobot 1 SKS pada semester 4 (hardware dasar), RKS304 berbobot 2 SKS pada semester 5 (navigasi lanjutan)",
    },
    {
        "prodi": "TRKB",
        "query": "Bagaimana hubungan sinergi antara mata kuliah Sensor dan Aktuator dengan Mobile Robot di TRKB?",
        "kode": ["RKS202", "RKS301"],
        "nama": ["Sensor dan Aktuator", "Mobile Robot"],
        "required_synthesis": "RKS202 menyediakan pemahaman akuisisi data sensor yang langsung diimplementasikan pada platform otonom RKS301",
    },
    {
        "prodi": "TRKB",
        "query": "Apa perbedaan fokus perancangan antara Mobile Robot dengan Robot Manipulator di TRKB?",
        "kode": ["RKS301", "RKS302"],
        "nama": ["Mobile Robot", "Robot Manipulator"],
        "required_synthesis": "RKS301 berfokus pada navigasi robot beroda, sedangkan RKS302 berfokus pada kinematika lengan robot sendi",
    },
    {
        "prodi": "TRKB",
        "query": "Jelaskan integrasi materi Kecerdasan Buatan terhadap perancangan Mobile Robot di TRKB",
        "kode": ["MAT302", "RKS301"],
        "nama": ["Kecerdasan Buatan", "Mobile Robot"],
        "required_synthesis": "MAT302 mengajarkan algoritma path planning cerdas (A*, Q-learning) yang dieksekusi pada platform RKS301",
    },
    {
        "prodi": "TRKB",
        "query": "Berapa total SKS yang ditempuh jika mengambil Sensor dan Aktuator bersama Eksperimen Robotika I?",
        "kode": ["RKS202", "RKS204"],
        "nama": ["Sensor dan Aktuator", "Eksperimen Robotika I"],
        "required_synthesis": "RKS202 (3 SKS) + RKS204 (1 SKS) = total 4 SKS pada paket semester 4 TRKB",
    },
    {
        "prodi": "TRKB",
        "query": "Bandingkan prasyarat dan tingkat kesulitan antara Robot Manipulator dan Eksperimen Robotika II",
        "kode": ["RKS302", "RKS304"],
        "nama": ["Robot Manipulator", "Eksperimen Robotika II"],
        "required_synthesis": "Keduanya diambil bersamaan di semester 5, RKS302 menekankan teori kinematika dan RKS304 menekankan integrasi fisik",
    },
    # RN
    {
        "prodi": "RN",
        "query": "Bandingkan beban SKS dan silabus antara Fisika Dasar 1 dengan Fisika Dasar 2 di Rekayasa Nanoteknologi",
        "kode": ["FID101", "FID104"],
        "nama": ["Fisika Dasar 1", "Fisika Dasar 2"],
        "required_synthesis": "FID101 (Sem 1, 2 SKS, mekanika gerak) vs FID104 (Sem 2, 2 SKS, listrik magnet dan termodinamika)",
    },
    {
        "prodi": "RN",
        "query": "Bagaimana kesinambungan antara Kimia Dasar dan Kimia Dasar 2 di Rekayasa Nanoteknologi?",
        "kode": ["KID105", "KID103"],
        "nama": ["Kimia Dasar", "Kimia Dasar 2"],
        "required_synthesis": "KID105 di semester 1 (struktur atom dan ikatan kimia) menjadi prasyarat teoritis KID103 di semester 2 (kesetimbangan dan kinetika)",
    },
    {
        "prodi": "RN",
        "query": "Berapa total jam dan SKS gabungan kuliah teori Fisika Dasar 1 dan Praktikum Fisika Dasar 1?",
        "kode": ["FID101", "FID102"],
        "nama": ["Fisika Dasar 1", "Praktikum Fisika Dasar 1"],
        "required_synthesis": "FID101 (2 SKS teori) + FID102 (1 SKS praktikum) = total 3 SKS paket sains semester 1",
    },
    {
        "prodi": "RN",
        "query": "Berapa total SKS gabungan kuliah teori Kimia Dasar dan Praktikum Kimia Dasar 1 di RN?",
        "kode": ["KID105", "KID102"],
        "nama": ["Kimia Dasar", "Praktikum Kimia Dasar 1"],
        "required_synthesis": "KID105 (2 SKS teori) + KID102 (1 SKS praktikum) = total 3 SKS",
    },
    {
        "prodi": "RN",
        "query": "Jelaskan perbedaan fokus eksperimen di Praktikum Fisika Dasar 1 dibanding Praktik Kimia Dasar 1",
        "kode": ["FID102", "KID102"],
        "nama": ["Praktikum Fisika Dasar 1", "Praktikum Kimia Dasar 1"],
        "required_synthesis": "FID102 mengukur besaran mekanik dan gerak partikel, sedangkan KID102 menguji titrasi asam-basa dan reaksi larutan",
    },
    {
        "prodi": "RN",
        "query": "Bandingkan semester penawaran dan materi Fisika Dasar 2 terhadap Kimia Dasar 2 di kurikulum RN",
        "kode": ["FID104", "KID103"],
        "nama": ["Fisika Dasar 2", "Kimia Dasar 2"],
        "required_synthesis": "Keduanya berada di semester 2 masing-masing berbobot 2 SKS, menjadi pilar sains pembentuk nano-material",
    },
    # TI
    {
        "prodi": "TI",
        "query": "Bandingkan fokus metode antara Penelitian Operasi Dasar dengan Penelitian Operasi Lanjut di TI",
        "kode": ["TID201", "TIA302"],
        "nama": ["Penelitian Operasi Dasar", "Penelitian Operasi Lanjut"],
        "required_synthesis": "TID201 (Sem 4, linear programming dan simpleks) vs TIA302 (Sem 5, non-linear programming, integer programming, teori antrian)",
    },
    {
        "prodi": "TI",
        "query": "Bagaimana keterkaitan antara mata kuliah Ergonomi dengan Otomasi Industri di Teknik Industri?",
        "kode": ["TIA201", "TIU201"],
        "nama": ["Ergonomi", "Otomasi Industri"],
        "required_synthesis": "TIA201 (human factors dan antropometri) dipadukan dengan TIU201 (sistem otomasi) untuk perancangan stasiun kerja terintegrasi",
    },
    {
        "prodi": "TI",
        "query": "Jelaskan peran Statistik Industri sebagai prasyarat pemodelan Simulasi Komputer di TI",
        "kode": ["TID202", "TIK301"],
        "nama": ["Statistik Industri", "Simulasi Komputer"],
        "required_synthesis": "TID202 mengajarkan distribusi probabilitas empiris yang menjadi input parameter distribusi waktu kedatangan pada TIK301",
    },
    {
        "prodi": "TI",
        "query": "Berapa total SKS gabungan mata kuliah Penelitian Operasi Dasar dan Statistik Industri semester 4?",
        "kode": ["TID201", "TID202"],
        "nama": ["Penelitian Operasi Dasar", "Statistik Industri"],
        "required_synthesis": "TID201 (3 SKS) + TID202 (3 SKS) = total 6 SKS kuantitatif di semester 4 TI",
    },
    {
        "prodi": "TI",
        "query": "Bandingkan semester pengambilan dan SKS antara Ergonomi dan Simulasi Komputer di kurikulum TI",
        "kode": ["TIA201", "TIK301"],
        "nama": ["Ergonomi", "Simulasi Komputer"],
        "required_synthesis": "TIA201 berada di semester 3 (3 SKS), sedangkan TIK301 berada di semester 6 (3 SKS)",
    },
    {
        "prodi": "TI",
        "query": "Bagaimana integrasi antara Otomasi Industri dan Simulasi Komputer dalam optimasi lini produksi?",
        "kode": ["TIU201", "TIK301"],
        "nama": ["Otomasi Industri", "Simulasi Komputer"],
        "required_synthesis": "Siklus perakitan otomasi TIU201 divalidasi dan diuji bottleneck-nya melalui simulasi model diskrit TIK301",
    },
    # TE
    {
        "prodi": "TE",
        "query": "Bandingkan materi teori dan praktikum antara Rangkaian Listrik dengan Praktikum Rangkaian Listrik",
        "kode": ["TBE201", "TBE202"],
        "nama": ["Rangkaian Listrik", "Praktikum Rangkaian Listrik"],
        "required_synthesis": "TBE201 (3 SKS teori analisis simpul dan mesh) diuji langsung di laboratorium TBE202 (1 SKS perakitan fisik)",
    },
    {
        "prodi": "TE",
        "query": "Bagaimana kesinambungan antara mata kuliah Rangkaian Listrik terhadap Sistem Tenaga Listrik di TE?",
        "kode": ["TBE201", "TEE202"],
        "nama": ["Rangkaian Listrik", "Sistem Tenaga Listrik"],
        "required_synthesis": "TBE201 di semester 3 mendasari pemahaman analisis daya impedansi untuk transmisi jaringan TEE202 semester 4",
    },
    {
        "prodi": "TE",
        "query": "Jelaskan hubungan materi antara Medan Elektromagnetik dengan Sinyal dan Sistem di Teknik Elektro",
        "kode": ["TBO201", "TRS203"],
        "nama": ["Medan Elektromagnetik", "Sinyal dan Sistem"],
        "required_synthesis": "Keduanya di semester 4 (masing-masing 3 SKS), TBO201 mengkaji propagasi gelombang fisik, TRS203 mengkaji representasi matematisnya",
    },
    {
        "prodi": "TE",
        "query": "Berapa total SKS jika mengambil Dasar Sistem Telekomunikasi bersama Praktikum Sistem Telekomunikasi?",
        "kode": ["TET201", "TET202"],
        "nama": ["Dasar Sistem dan Jaringan Telekomunikasi", "Praktikum Sistem dan Jaringan Telekomunikasi"],
        "required_synthesis": "TET201 (2 SKS teori) + TET202 (1 SKS praktikum) = total 3 SKS telekomunikasi",
    },
    {
        "prodi": "TE",
        "query": "Bandingkan semester penawaran Rangkaian Listrik dan Sinyal Sistem di kurikulum Teknik Elektro",
        "kode": ["TBE201", "TRS203"],
        "nama": ["Rangkaian Listrik", "Sinyal dan Sistem"],
        "required_synthesis": "TBE201 ditawarkan di semester 3 (3 SKS), sedangkan TRS203 ditawarkan di semester 4 (3 SKS)",
    },
    {
        "prodi": "TE",
        "query": "Bagaimana kaitan antara Medan Elektromagnetik dengan Dasar Sistem Telekomunikasi di TE?",
        "kode": ["TBO201", "TET201"],
        "nama": ["Medan Elektromagnetik", "Dasar Sistem dan Jaringan Telekomunikasi"],
        "required_synthesis": "Persamaan perambatan gelombang TBO201 mendasari analisis redaman kabel dan propagasi antena pada TET201",
    },
]

# 4. OUT-OF-SCOPE (30 queries: Non-FTMM faculties & external subjects)
OUT_OF_SCOPE_QUERIES = [
    # Kedokteran & Kesehatan (FK, FKG, FKM, FF)
    {"domain": "Kedokteran", "query": "Apa saja materi mata kuliah Anatomi Tubuh Manusia dan Histologi di UNAIR?", "prodi_target": "FK"},
    {"domain": "Kedokteran", "query": "Berapa SKS beban studi mata kuliah Patologi Klinik dan Farmakologi Dasar?", "prodi_target": "FK"},
    {"domain": "Farmasi", "query": "Mata kuliah Formulasi Sediaan Cair dan Steril diajarkan di semester berapa?", "prodi_target": "FF"},
    {"domain": "Kedokteran Gigi", "query": "Apa prasyarat untuk mengambil mata kuliah Konservasi Gigi dan Bedah Mulut?", "prodi_target": "FKG"},
    {"domain": "Kesehatan Masyarakat", "query": "Berapa SKS mata kuliah Epidemiologi Penyakit Menular di FKM UNAIR?", "prodi_target": "FKM"},
    {"domain": "Keperawatan", "query": "Kapan jadwal stase keperawatan gawat darurat dan medikal bedah?", "prodi_target": "FKP"},
    # Hukum & Ilmu Sosial (FH, FISIP)
    {"domain": "Hukum", "query": "Bagaimana silabus mata kuliah Hukum Tata Negara dan Hukum Acara Pidana di FH?", "prodi_target": "FH"},
    {"domain": "Hukum", "query": "Berapa SKS mata kuliah Hukum Perdata Internasional di Fakultas Hukum UNAIR?", "prodi_target": "FH"},
    {"domain": "Ilmu Sosial", "query": "Apa yang dipelajari dalam mata kuliah Teori Hubungan Internasional dan Diplomasi?", "prodi_target": "FISIP"},
    {"domain": "Ilmu Sosial", "query": "Mata kuliah Kebijakan Publik dan Administrasi Negara semester berapa?", "prodi_target": "FISIP"},
    {"domain": "Komunikasi", "query": "Berapa SKS beban mata kuliah Jurnalistik Investigasi dan Penyiaran Radio?", "prodi_target": "FISIP"},
    {"domain": "Sosiologi", "query": "Apa prasyarat mata kuliah Sosiologi Pembangunan dan Konflik Sosial?", "prodi_target": "FISIP"},
    # Ekonomi & Bisnis (FEB)
    {"domain": "Ekonomi", "query": "Apa saja materi dalam silabus Akuntansi Keuangan Menengah di FEB UNAIR?", "prodi_target": "FEB"},
    {"domain": "Ekonomi", "query": "Berapa SKS mata kuliah Perpajakan Lanjutan dan Akuntansi Sektor Publik?", "prodi_target": "FEB"},
    {"domain": "Manajemen", "query": "Kapan jadwal semester mata kuliah Manajemen Portofolio dan Analisis Investasi?", "prodi_target": "FEB"},
    {"domain": "Ekonomi Islam", "query": "Apa saja yang dipelajari pada mata kuliah Fiqih Muamalah dan Perbankan Syariah?", "prodi_target": "FEB"},
    {"domain": "Ekonomi", "query": "Berapa SKS beban studi mata kuliah Teori Ekonomi Makro Lanjutan di FEB?", "prodi_target": "FEB"},
    {"domain": "Manajemen", "query": "Apa prasyarat untuk memprogram mata kuliah Riset Pemasaran Strategis?", "prodi_target": "FEB"},
    # Humaniora & Bahasa (FIB)
    {"domain": "Sastra", "query": "Apa saja topik bahasan dalam mata kuliah Pengantar Linguistik Sastra Inggris?", "prodi_target": "FIB"},
    {"domain": "Sastra Jepang", "query": "Kapan jadwal mata kuliah Tata Bahasa Jepang Tingkat Mahir (Kanji N2)?", "prodi_target": "FIB"},
    {"domain": "Ilmu Sejarah", "query": "Berapa SKS mata kuliah Sejarah Kolonialisme dan Pergerakan Nasional di FIB?", "prodi_target": "FIB"},
    {"domain": "Sastra", "query": "Apa silabus mata kuliah Filologi dan Kodikologi Manuskrip Nusantara?", "prodi_target": "FIB"},
    {"domain": "Sastra Indonesia", "query": "Mata kuliah Fonologi dan Sintaksis Bahasa Indonesia semester berapa?", "prodi_target": "FIB"},
    {"domain": "Sastra", "query": "Apa prasyarat untuk mengambil mata kuliah Kritik Sastra Kontemporer?", "prodi_target": "FIB"},
    # Psikologi & Kedokteran Hewan (FPsi, FKH)
    {"domain": "Psikologi", "query": "Apa materi praktikum dalam mata kuliah Asesmen Psikologi dan Tes Proyektif?", "prodi_target": "FPsi"},
    {"domain": "Psikologi", "query": "Berapa SKS beban studi mata kuliah Psikologi Klinis Anak dan Remaja?", "prodi_target": "FPsi"},
    {"domain": "Psikologi", "query": "Kapan jadwal semester mata kuliah Psikometrika dan Konstruksi Alat Ukur?", "prodi_target": "FPsi"},
    {"domain": "Kedokteran Hewan", "query": "Apa saja materi mata kuliah Ilmu Bedah Khusus Veteriner di FKH UNAIR?", "prodi_target": "FKH"},
    {"domain": "Kedokteran Hewan", "query": "Berapa SKS mata kuliah Patologi Anatomi Veteriner dan Nekropsi Hewan?", "prodi_target": "FKH"},
    {"domain": "Kedokteran Hewan", "query": "Apa prasyarat untuk mengikuti stase koasistensi Klinik Reproduksi Hewan?", "prodi_target": "FKH"},
]


def assemble_benchmark() -> list[dict]:
    benchmark_dataset = []
    idx = 1

    # 1. Lexical Match
    for item in LEXICAL_QUERIES:
        benchmark_dataset.append({
            "id": f"RC-LEX-{idx:03d}",
            "complexity_tier": "LEXICAL_MATCH",
            "program_studi": item["prodi"],
            "query": item["query"],
            "target_kode_mk": item["kode"],
            "target_nama_mk": item["nama"],
            "gold_chunks_count": len(item["kode"]),
            "is_out_of_scope": False,
            "expected_action": "RETRIEVE_SINGLE",
            "required_evidence_fields": ["sks", "semester", "silabus"],
        })
        idx += 1

    # 2. Semantic Paraphrase
    for item in PARAPHRASE_QUERIES:
        benchmark_dataset.append({
            "id": f"RC-PAR-{idx:03d}",
            "complexity_tier": "SEMANTIC_PARAPHRASE",
            "program_studi": item["prodi"],
            "query": item["query"],
            "target_kode_mk": item["kode"],
            "target_nama_mk": item["nama"],
            "gold_chunks_count": len(item["kode"]),
            "is_out_of_scope": False,
            "expected_action": "RETRIEVE_SINGLE",
            "required_evidence_fields": ["silabus", "topik"],
        })
        idx += 1

    # 3. Multi-Evidence Synthesis
    for item in MULTI_EVIDENCE_QUERIES:
        benchmark_dataset.append({
            "id": f"RC-MUL-{idx:03d}",
            "complexity_tier": "MULTI_EVIDENCE_SYNTHESIS",
            "program_studi": item["prodi"],
            "query": item["query"],
            "target_kode_mk": item["kode"],
            "target_nama_mk": item["nama"],
            "gold_chunks_count": len(item["kode"]),
            "is_out_of_scope": False,
            "expected_action": "RETRIEVE_MULTI_SYNTHESIS",
            "required_synthesis": item["required_synthesis"],
            "required_evidence_fields": ["sks", "semester", "prasyarat", "silabus"],
        })
        idx += 1

    # 4. Out-of-Scope
    for item in OUT_OF_SCOPE_QUERIES:
        benchmark_dataset.append({
            "id": f"RC-OOD-{idx:03d}",
            "complexity_tier": "OUT_OF_SCOPE",
            "program_studi": None,
            "target_faculty": item["prodi_target"],
            "query": item["query"],
            "target_kode_mk": [],
            "target_nama_mk": [],
            "gold_chunks_count": 0,
            "is_out_of_scope": True,
            "expected_action": "REFUSE_OUT_OF_SCOPE",
            "refusal_reason": f"Kueri kurikulum di luar mandat 5 prodi FTMM ({item['domain']} / {item['prodi_target']})",
        })
        idx += 1

    return benchmark_dataset


def main():
    dataset = assemble_benchmark()
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(dataset, f, indent=2, ensure_ascii=False)

    logger.info("✓ Berhasil membuat %s dengan %d kueri terstratifikasi", OUTPUT_PATH, len(dataset))
    counts = {}
    for d in dataset:
        counts[d["complexity_tier"]] = counts.get(d["complexity_tier"], 0) + 1
    for tier, count in counts.items():
        logger.info("  - %-25s : %3d kueri", tier, count)


if __name__ == "__main__":
    main()
