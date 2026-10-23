#!/usr/bin/env python3
"""
scripts/build_hyde_eval_dataset.py

Membangun dataset evaluasi terstratifikasi untuk benchmark Hypothetical Document Embeddings (HyDE)
dan Adaptive Query Expansion pada FTMM COMPASS.
Memuat 80 butir pengujian terbagi ke dalam 4 kategori komparatif:
  1. EXPLICIT_LEXICAL (20 kueri)
  2. SLANG_INFORMAL (20 kueri)
  3. THEMATIC_ABSTRACT (20 kueri)
  4. CROSS_DISCIPLINARY (20 kueri)

Melakukan validasi integritas terhadap data/courses_catalog.json dan
menyinkronkan tab HYDE_EVAL ke Google Spreadsheet Ground Truth NLP (1jR0Hj3NBBvUv3NDfbU_-pPDIWB7EaDfJk5E7-7SoKSM).
"""

import json
import logging
import urllib.request
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from scripts.export_gt_master_and_literatur import (
    SPREADSHEET_ID,
    execute_sheets_api,
    get_access_token,
    get_all_sheets_meta,
)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

ROOT_DIR = Path(__file__).resolve().parent.parent
CATALOG_PATH = ROOT_DIR / "data" / "courses_catalog.json"
OUTPUT_PATH = ROOT_DIR / "data" / "ground_truth" / "hyde_eval_dataset.json"

RAW_QUERIES = [
    # -------------------------------------------------------------
    # 1. EXPLICIT_LEXICAL (20 Queries)
    # -------------------------------------------------------------
    {
        "id": "HYDE-LEX-001",
        "category": "EXPLICIT_LEXICAL",
        "program_studi": "TSD",
        "query": "Apa saja materi yang dipelajari pada mata kuliah Pengantar Basis Data SID206 di Sains Data?",
        "target_kode_mk": ["SID206"],
        "target_nama_mk": ["Pengantar Basis Data"],
        "expected_hypo_keywords": ["basis data", "SQL", "relasional", "DBMS", "normalisasi", "tabel"],
        "rationale": "Kueri eksplisit menyebut kode dan nama matkul. Baseline direct dense dan BM25 diprediksi sangat kuat."
    },
    {
        "id": "HYDE-LEX-002",
        "category": "EXPLICIT_LEXICAL",
        "program_studi": "TSD",
        "query": "Berapa SKS beban mata kuliah Data Mining I MAS207 di program studi Teknologi Sains Data?",
        "target_kode_mk": ["MAS207"],
        "target_nama_mk": ["Data Mining I"],
        "expected_hypo_keywords": ["data mining", "klasifikasi", "clustering", "association rule", "SKS"],
        "rationale": "Kueri eksplisit kode dan nama matkul data mining."
    },
    {
        "id": "HYDE-LEX-003",
        "category": "EXPLICIT_LEXICAL",
        "program_studi": "TSD",
        "query": "Kapan jadwal semester pengambilan mata kuliah Analisis Data Spasial MAS317 di TSD?",
        "target_kode_mk": ["MAS317"],
        "target_nama_mk": ["Analisis Data Spasial"],
        "expected_hypo_keywords": ["spasial", "geografis", "GIS", "pemetaan", "autokorelasi spasial"],
        "rationale": "Kueri leksikal eksplisit analisis spasial."
    },
    {
        "id": "HYDE-LEX-004",
        "category": "EXPLICIT_LEXICAL",
        "program_studi": "TSD",
        "query": "Silabus mata kuliah Teknologi Big Data SID307 mencakup materi komputasi apa saja?",
        "target_kode_mk": ["SID307"],
        "target_nama_mk": ["Teknologi Big Data"],
        "expected_hypo_keywords": ["big data", "hadoop", "spark", "mapreduce", "distributed computing"],
        "rationale": "Kueri eksplisit kode dan nama big data."
    },
    {
        "id": "HYDE-LEX-005",
        "category": "EXPLICIT_LEXICAL",
        "program_studi": "TRKB",
        "query": "Apa materi praktikum dalam mata kuliah Sensor dan Aktuator RKS202 di TRKB?",
        "target_kode_mk": ["RKS202"],
        "target_nama_mk": ["Sensor dan Aktuator"],
        "expected_hypo_keywords": ["sensor", "aktuator", "transduser", "sinyal", "motor"],
        "rationale": "Kueri eksplisit sensor dan aktuator robotika."
    },
    {
        "id": "HYDE-LEX-006",
        "category": "EXPLICIT_LEXICAL",
        "program_studi": "TRKB",
        "query": "Berapa SKS beban mata kuliah Sistem Tertanam dan Internet of Things (IoT) RKS201?",
        "target_kode_mk": ["RKS201"],
        "target_nama_mk": ["Sistem Tertanam dan Internet of Things (IoT)"],
        "expected_hypo_keywords": ["embedded system", "IoT", "mikrokontroler", "protokol jaringan", "SKS"],
        "rationale": "Kueri eksplisit sistem tertanam IoT."
    },
    {
        "id": "HYDE-LEX-007",
        "category": "EXPLICIT_LEXICAL",
        "program_studi": "TRKB",
        "query": "Mata kuliah Mobile Robot RKS301 di Teknik Robotika membahas topik apa saja?",
        "target_kode_mk": ["RKS301"],
        "target_nama_mk": ["Mobile Robot"],
        "expected_hypo_keywords": ["mobile robot", "kinematika", "navigasi", "SLAM", "odometri"],
        "rationale": "Kueri eksplisit mobile robot TRKB."
    },
    {
        "id": "HYDE-LEX-008",
        "category": "EXPLICIT_LEXICAL",
        "program_studi": "TRKB",
        "query": "Kapan jadwal semester mata kuliah Visi Komputer RKK303 di kurikulum TRKB?",
        "target_kode_mk": ["RKK303"],
        "target_nama_mk": ["Visi Komputer"],
        "expected_hypo_keywords": ["visi komputer", "computer vision", "segmentasi citra", "deteksi objek"],
        "rationale": "Kueri eksplisit visi komputer."
    },
    {
        "id": "HYDE-LEX-009",
        "category": "EXPLICIT_LEXICAL",
        "program_studi": "RN",
        "query": "Apa materi pokok dalam mata kuliah Nanosains Kuantum RND204 di Rekayasa Nanoteknologi?",
        "target_kode_mk": ["RND204"],
        "target_nama_mk": ["Nanosains Kuantum"],
        "expected_hypo_keywords": ["kuantum", "mekanika kuantum", "titik kuantum", "keadaan partikel"],
        "rationale": "Kueri eksplisit nanosains kuantum RN."
    },
    {
        "id": "HYDE-LEX-010",
        "category": "EXPLICIT_LEXICAL",
        "program_studi": "RN",
        "query": "Berapa SKS beban mata kuliah Komputasi Nanoteknologi 1 FIT214 di program studi RN?",
        "target_kode_mk": ["FIT214"],
        "target_nama_mk": ["Komputasi Nanoteknologi 1"],
        "expected_hypo_keywords": ["komputasi", "nanoteknologi", "simulasi molekuler", "DFT", "SKS"],
        "rationale": "Kueri eksplisit komputasi nanoteknologi."
    },
    {
        "id": "HYDE-LEX-011",
        "category": "EXPLICIT_LEXICAL",
        "program_studi": "RN",
        "query": "Mata kuliah Kinetika dan Termodinamika RND202 ditawarkan pada semester berapa?",
        "target_kode_mk": ["RND202"],
        "target_nama_mk": ["Kinetika dan Termodinamika"],
        "expected_hypo_keywords": ["kinetika", "termodinamika", "entalpi", "energi bebas", "reaksi"],
        "rationale": "Kueri eksplisit kinetika dan termodinamika."
    },
    {
        "id": "HYDE-LEX-012",
        "category": "EXPLICIT_LEXICAL",
        "program_studi": "RN",
        "query": "Apa saja teknik yang diajarkan pada Teknik Karakterisasi Material 1 RNA202?",
        "target_kode_mk": ["RNA202"],
        "target_nama_mk": ["Teknik Karakterisasi Material 1"],
        "expected_hypo_keywords": ["karakterisasi", "SEM", "XRD", "spektroskopi", "morfologi"],
        "rationale": "Kueri eksplisit karakterisasi material."
    },
    {
        "id": "HYDE-LEX-013",
        "category": "EXPLICIT_LEXICAL",
        "program_studi": "TI",
        "query": "Apa materi pokok mata kuliah Ergonomi TIA201 di Teknik Industri FTMM?",
        "target_kode_mk": ["TIA201"],
        "target_nama_mk": ["Ergonomi"],
        "expected_hypo_keywords": ["ergonomi", "antropometri", "beban kerja", "biomekanika", "fisiologi"],
        "rationale": "Kueri eksplisit ergonomi industri."
    },
    {
        "id": "HYDE-LEX-014",
        "category": "EXPLICIT_LEXICAL",
        "program_studi": "TI",
        "query": "Berapa SKS beban mata kuliah Otomasi Industri TIU201 di kurikulum TI?",
        "target_kode_mk": ["TIU201"],
        "target_nama_mk": ["Otomasi Industri"],
        "expected_hypo_keywords": ["otomasi", "industri", "PLC", "sensor pabrik", "SKS"],
        "rationale": "Kueri eksplisit otomasi industri."
    },
    {
        "id": "HYDE-LEX-015",
        "category": "EXPLICIT_LEXICAL",
        "program_studi": "TI",
        "query": "Silabus Penelitian Operasi Dasar TID201 di Teknik Industri mencakup topik apa?",
        "target_kode_mk": ["TID201"],
        "target_nama_mk": ["Penelitian Operasi Dasar"],
        "expected_hypo_keywords": ["penelitian operasi", "linear programming", "simpleks", "optimasi"],
        "rationale": "Kueri eksplisit penelitian operasi dasar."
    },
    {
        "id": "HYDE-LEX-016",
        "category": "EXPLICIT_LEXICAL",
        "program_studi": "TI",
        "query": "Kapan jadwal pengambilan mata kuliah Manajemen Rantai Pasok TIU302 di TI?",
        "target_kode_mk": ["TIU302"],
        "target_nama_mk": ["Manajemen Rantai Pasok"],
        "expected_hypo_keywords": ["rantai pasok", "supply chain", "logistik", "distribusi", "inventory"],
        "rationale": "Kueri eksplisit manajemen rantai pasok."
    },
    {
        "id": "HYDE-LEX-017",
        "category": "EXPLICIT_LEXICAL",
        "program_studi": "TE",
        "query": "Apa materi yang dipelajari pada mata kuliah Medan Elektromagnetik TBO201 di Elektro?",
        "target_kode_mk": ["TBO201"],
        "target_nama_mk": ["Medan Elektromagnetik"],
        "expected_hypo_keywords": ["medan elektromagnetik", "persamaan maxwell", "gelombang", "fluks listrik"],
        "rationale": "Kueri eksplisit medan elektromagnetik."
    },
    {
        "id": "HYDE-LEX-018",
        "category": "EXPLICIT_LEXICAL",
        "program_studi": "TE",
        "query": "Berapa SKS beban mata kuliah Sistem Kontrol TES203 di Teknik Elektro?",
        "target_kode_mk": ["TES203"],
        "target_nama_mk": ["Sistem Kontrol"],
        "expected_hypo_keywords": ["sistem kontrol", "feedback", "PID", "kestabilan", "SKS"],
        "rationale": "Kueri eksplisit sistem kontrol TE."
    },
    {
        "id": "HYDE-LEX-019",
        "category": "EXPLICIT_LEXICAL",
        "program_studi": "TE",
        "query": "Mata kuliah Sistem Tenaga Listrik TEE202 di semester 4 membahas apa saja?",
        "target_kode_mk": ["TEE202"],
        "target_nama_mk": ["Sistem Tenaga Listrik"],
        "expected_hypo_keywords": ["sistem tenaga", "aliran daya", "transmisi", "gardu induk", "trafo"],
        "rationale": "Kueri eksplisit sistem tenaga listrik."
    },
    {
        "id": "HYDE-LEX-020",
        "category": "EXPLICIT_LEXICAL",
        "program_studi": "TE",
        "query": "Kapan jadwal semester mata kuliah Sinyal dan Sistem TRS203 di Teknik Elektro?",
        "target_kode_mk": ["TRS203"],
        "target_nama_mk": ["Sinyal dan Sistem"],
        "expected_hypo_keywords": ["sinyal dan sistem", "fourier", "laplace", "konvolusi", "filter"],
        "rationale": "Kueri eksplisit sinyal dan sistem."
    },

    # -------------------------------------------------------------
    # 2. SLANG_INFORMAL (20 Queries)
    # -------------------------------------------------------------
    {
        "id": "HYDE-SLANG-001",
        "category": "SLANG_INFORMAL",
        "program_studi": "TSD",
        "query": "gue pengen bisa ngoprek sql sama bikin database relasional yang bener di prodi data, matkulnya apa ya?",
        "target_kode_mk": ["SID206"],
        "target_nama_mk": ["Pengantar Basis Data"],
        "expected_hypo_keywords": ["basis data", "SQL", "relasional", "tabel", "skema database"],
        "rationale": "Kueri slang santai tanpa menyebut nama resmi matkul. HyDE mentransformasikan kueri ke silabus SQL/DBMS."
    },
    {
        "id": "HYDE-SLANG-002",
        "category": "SLANG_INFORMAL",
        "program_studi": "TSD",
        "query": "ada matkul yang ngajarin clustering sama klasifikasi algoritma pohon keputusan gak sih di tsd?",
        "target_kode_mk": ["MAS207"],
        "target_nama_mk": ["Data Mining I"],
        "expected_hypo_keywords": ["data mining", "decision tree", "clustering", "k-means", "algoritma"],
        "rationale": "Kueri informal menanyakan algoritma machine learning/mining tanpa kata 'Data Mining I'."
    },
    {
        "id": "HYDE-SLANG-003",
        "category": "SLANG_INFORMAL",
        "program_studi": "TSD",
        "query": "kalo mau belajar olah data peta gis sama koordinat satelit pake r atau python itu matkul apa namanya?",
        "target_kode_mk": ["MAS317"],
        "target_nama_mk": ["Analisis Data Spasial"],
        "expected_hypo_keywords": ["spasial", "geografis", "pemetaan", "GIS", "shapefile"],
        "rationale": "Kueri slang mahasiswa mengacu pada peta GIS dan data koordinat satelit."
    },
    {
        "id": "HYDE-SLANG-004",
        "category": "SLANG_INFORMAL",
        "program_studi": "TSD",
        "query": "pengen paham hadoop spark sama arsitektur data gede banget di semester 4, ada gak?",
        "target_kode_mk": ["SID307"],
        "target_nama_mk": ["Teknologi Big Data"],
        "expected_hypo_keywords": ["big data", "hadoop", "spark", "pemrosesan paralel", "data terdistribusi"],
        "rationale": "Kueri informal menggunakan 'data gede banget' dan framework Hadoop/Spark."
    },
    {
        "id": "HYDE-SLANG-005",
        "category": "SLANG_INFORMAL",
        "program_studi": "TRKB",
        "query": "bro matkul trkb yang ngebahas sensor ultrasonik lidar sama motor servo itu yang mana ya?",
        "target_kode_mk": ["RKS202"],
        "target_nama_mk": ["Sensor dan Aktuator"],
        "expected_hypo_keywords": ["sensor", "aktuator", "transduser", "motor servo", "pengukuran sinyal"],
        "rationale": "Kueri santai menyebut komponen fisik (ultrasonik, lidar, servo) tanpa nama matkul resmi."
    },
    {
        "id": "HYDE-SLANG-006",
        "category": "SLANG_INFORMAL",
        "program_studi": "TRKB",
        "query": "kalo mau ngoding esp32 sama arduino biar nyambung cloud iot di semester 4 matkulnya apa?",
        "target_kode_mk": ["RKS201"],
        "target_nama_mk": ["Sistem Tertanam dan Internet of Things (IoT)"],
        "expected_hypo_keywords": ["sistem tertanam", "internet of things", "mikrokontroler", "IoT", "komunikasi nirkabel"],
        "rationale": "Kueri slang hardware developer (ESP32, Arduino, Cloud IoT)."
    },
    {
        "id": "HYDE-SLANG-007",
        "category": "SLANG_INFORMAL",
        "program_studi": "TRKB",
        "query": "matkul yang seru buat balapan robot beroda sama navigasi slam di lab trkb apa tuh?",
        "target_kode_mk": ["RKS301"],
        "target_nama_mk": ["Mobile Robot"],
        "expected_hypo_keywords": ["mobile robot", "robot beroda", "SLAM", "navigasi otonom", "lintasan"],
        "rationale": "Kueri slang 'robot beroda' dan 'balapan' memetakan ke Mobile Robot."
    },
    {
        "id": "HYDE-SLANG-008",
        "category": "SLANG_INFORMAL",
        "program_studi": "TRKB",
        "query": "gua pengen belajar deteksi objek yolo sama opencv buat kamera robot, ada matkulnya gak?",
        "target_kode_mk": ["RKK303"],
        "target_nama_mk": ["Visi Komputer"],
        "expected_hypo_keywords": ["visi komputer", "deteksi objek", "citra digital", "kamera", "deep learning visual"],
        "rationale": "Kueri slang praktis (YOLO, OpenCV) memetakan ke Visi Komputer."
    },
    {
        "id": "HYDE-SLANG-009",
        "category": "SLANG_INFORMAL",
        "program_studi": "RN",
        "query": "di rn ada gak sih matkul yang ngajarin fisika partikel kuantum level atomik?",
        "target_kode_mk": ["RND204"],
        "target_nama_mk": ["Nanosains Kuantum"],
        "expected_hypo_keywords": ["kuantum", "nanosains", "partikel", "struktur atom", "tingkat energi"],
        "rationale": "Kueri slang 'partikel kuantum level atomik' memetakan ke Nanosains Kuantum."
    },
    {
        "id": "HYDE-SLANG-010",
        "category": "SLANG_INFORMAL",
        "program_studi": "RN",
        "query": "matkul simulasi molekul pake dft sama software komputasi di rn apa ya bro?",
        "target_kode_mk": ["FIT214"],
        "target_nama_mk": ["Komputasi Nanoteknologi 1"],
        "expected_hypo_keywords": ["komputasi", "simulasi", "DFT", "pemodelan atom", "nanoteknologi"],
        "rationale": "Kueri slang DFT dan software komputasi memetakan ke Komputasi Nanoteknologi 1."
    },
    {
        "id": "HYDE-SLANG-011",
        "category": "SLANG_INFORMAL",
        "program_studi": "RN",
        "query": "belajar reaksi kimia panas entalpi sama laju reaksi material nano di semester 4 apa namanya?",
        "target_kode_mk": ["RND202"],
        "target_nama_mk": ["Kinetika dan Termodinamika"],
        "expected_hypo_keywords": ["kinetika", "termodinamika", "entalpi", "kesetimbangan", "laju reaksi"],
        "rationale": "Kueri informal menanyakan konsep panas entalpi dan laju reaksi material."
    },
    {
        "id": "HYDE-SLANG-012",
        "category": "SLANG_INFORMAL",
        "program_studi": "RN",
        "query": "kalo mau nyoba alat sem tem xrd buat ngeliat struktur material nano itu praktikum di matkul apa?",
        "target_kode_mk": ["RNA202"],
        "target_nama_mk": ["Teknik Karakterisasi Material 1"],
        "expected_hypo_keywords": ["karakterisasi", "SEM", "TEM", "XRD", "analisis struktur"],
        "rationale": "Kueri slang mahasiswa menyebutkan instrumen SEM/TEM/XRD."
    },
    {
        "id": "HYDE-SLANG-013",
        "category": "SLANG_INFORMAL",
        "program_studi": "TI",
        "query": "matkul ti yang ngitung beban kerja fisik antropometri sama posisi duduk biar ga pegel apa ya?",
        "target_kode_mk": ["TIA201"],
        "target_nama_mk": ["Ergonomi"],
        "expected_hypo_keywords": ["ergonomi", "antropometri", "beban fisik", "postur kerja", "efisiensi manusia"],
        "rationale": "Kueri slang bahasa santai 'biar ga pegel' dan 'antropometri' memetakan ke Ergonomi."
    },
    {
        "id": "HYDE-SLANG-014",
        "category": "SLANG_INFORMAL",
        "program_studi": "TI",
        "query": "pengen belajar plc sama konveyor pabrik otomatis di ti, ngambil matkul apa sob?",
        "target_kode_mk": ["TIU201"],
        "target_nama_mk": ["Otomasi Industri"],
        "expected_hypo_keywords": ["otomasi", "pabrik", "PLC", "sistem manufaktur otomatis", "konveyor"],
        "rationale": "Kueri slang praktis konveyor dan PLC memetakan ke Otomasi Industri."
    },
    {
        "id": "HYDE-SLANG-015",
        "category": "SLANG_INFORMAL",
        "program_studi": "TI",
        "query": "matkul linear programming metode simpleks buat maksimasi profit pabrik itu namanya apa di ti?",
        "target_kode_mk": ["TID201"],
        "target_nama_mk": ["Penelitian Operasi Dasar"],
        "expected_hypo_keywords": ["penelitian operasi", "linear programming", "simpleks", "optimalisasi", "profit"],
        "rationale": "Kueri informal menyebut metode simpleks dan optimasi profit tanpa nama resmi matkul."
    },
    {
        "id": "HYDE-SLANG-016",
        "category": "SLANG_INFORMAL",
        "program_studi": "TI",
        "query": "kalo mau belajar distribusi barang logistik gudang sama bullwhip effect itu matkul apa ya?",
        "target_kode_mk": ["TIU302"],
        "target_nama_mk": ["Manajemen Rantai Pasok"],
        "expected_hypo_keywords": ["rantai pasok", "logistik", "gudang", "distribusi", "supply chain"],
        "rationale": "Kueri slang logistik gudang dan bullwhip effect memetakan ke Manajemen Rantai Pasok."
    },
    {
        "id": "HYDE-SLANG-017",
        "category": "SLANG_INFORMAL",
        "program_studi": "TE",
        "query": "matkul elektro yang pusing ngitung persamaan maxwell gelombang radio antena apa tuh?",
        "target_kode_mk": ["TBO201"],
        "target_nama_mk": ["Medan Elektromagnetik"],
        "expected_hypo_keywords": ["medan elektromagnetik", "maxwell", "gelombang radio", "fluks magnet", "propagasi"],
        "rationale": "Kueri slang ekspresif 'pusing ngitung persamaan maxwell gelombang radio' memetakan ke Medan Elektromagnetik."
    },
    {
        "id": "HYDE-SLANG-018",
        "category": "SLANG_INFORMAL",
        "program_studi": "TE",
        "query": "belajar bikin kestabilan sistem pake diagram bode sama pid controller di te matkul apa?",
        "target_kode_mk": ["TES203"],
        "target_nama_mk": ["Sistem Kontrol"],
        "expected_hypo_keywords": ["sistem kontrol", "PID", "diagram bode", "umpan balik", "kestabilan"],
        "rationale": "Kueri informal menyebut diagram bode dan PID memetakan ke Sistem Kontrol."
    },
    {
        "id": "HYDE-SLANG-019",
        "category": "SLANG_INFORMAL",
        "program_studi": "TE",
        "query": "matkul yang ngebahas gardu induk trafo transmisi pln tegangan tinggi itu yang mana ya?",
        "target_kode_mk": ["TEE202"],
        "target_nama_mk": ["Sistem Tenaga Listrik"],
        "expected_hypo_keywords": ["tenaga listrik", "transmisi", "gardu induk", "tegangan tinggi", "distribusi daya"],
        "rationale": "Kueri slang 'trafo gardu induk tegangan tinggi' memetakan ke Sistem Tenaga Listrik."
    },
    {
        "id": "HYDE-SLANG-020",
        "category": "SLANG_INFORMAL",
        "program_studi": "TE",
        "query": "matkul yang ngoprek transformasi fourier laplace filter suara sama sinyal diskrit namanya apa?",
        "target_kode_mk": ["TRS203"],
        "target_nama_mk": ["Sinyal dan Sistem"],
        "expected_hypo_keywords": ["sinyal dan sistem", "transformasi fourier", "laplace", "filter digital", "konvolusi"],
        "rationale": "Kueri slang filter suara dan transformasi fourier memetakan ke Sinyal dan Sistem."
    },

    # -------------------------------------------------------------
    # 3. THEMATIC_ABSTRACT (20 Queries)
    # -------------------------------------------------------------
    {
        "id": "HYDE-THEM-001",
        "category": "THEMATIC_ABSTRACT",
        "program_studi": "TSD",
        "query": "Bagaimana kurikulum FTMM memfasilitasi mahasiswa yang ingin berkarir sebagai database engineer?",
        "target_kode_mk": ["SID206"],
        "target_nama_mk": ["Pengantar Basis Data"],
        "expected_hypo_keywords": ["basis data", "SQL", "skema database", "manajemen basis data", "integritas data"],
        "rationale": "Kueri profil karir abstrak 'database engineer' yang membutuhkan pemetaan kurikulum."
    },
    {
        "id": "HYDE-THEM-002",
        "category": "THEMATIC_ABSTRACT",
        "program_studi": "TSD",
        "query": "Mata kuliah apa yang mendalami penemuan pola tersembunyi dari dataset transaksi skala besar?",
        "target_kode_mk": ["MAS207"],
        "target_nama_mk": ["Data Mining I"],
        "expected_hypo_keywords": ["data mining", "ekstraksi pola", "analisis asosiasi", "pengelompokan data"],
        "rationale": "Kueri konsep teoretis penemuan pola tersembunyi memetakan ke Data Mining I."
    },
    {
        "id": "HYDE-THEM-003",
        "category": "THEMATIC_ABSTRACT",
        "program_studi": "TSD",
        "query": "Kajian analisis data geografis dan fenomena spasial lingkungan dipelajari pada perkuliahan apa?",
        "target_kode_mk": ["MAS317"],
        "target_nama_mk": ["Analisis Data Spasial"],
        "expected_hypo_keywords": ["data spasial", "geospasial", "pemodelan wilayah", "koordinat", "GIS"],
        "rationale": "Kueri tematik analisis spasial lingkungan memetakan ke Analisis Data Spasial."
    },
    {
        "id": "HYDE-THEM-004",
        "category": "THEMATIC_ABSTRACT",
        "program_studi": "TSD",
        "query": "Pondasi infrastruktur komputasi terdistribusi untuk pemrosesan volume data bervolume masif diajarkan di mata kuliah apa?",
        "target_kode_mk": ["SID307"],
        "target_nama_mk": ["Teknologi Big Data"],
        "expected_hypo_keywords": ["big data", "komputasi terdistribusi", "pemrosesan skala besar", "cluster"],
        "rationale": "Kueri abstrak infrastruktur komputasi masif memetakan ke Big Data."
    },
    {
        "id": "HYDE-THEM-005",
        "category": "THEMATIC_ABSTRACT",
        "program_studi": "TRKB",
        "query": "Perangkat keras transduser dan penggerak fisik pada sistem otomasi cerdas dipelajari di mana?",
        "target_kode_mk": ["RKS202"],
        "target_nama_mk": ["Sensor dan Aktuator"],
        "expected_hypo_keywords": ["sensor", "aktuator", "transduser", "konversi energi", "antarmuka fisik"],
        "rationale": "Kueri formal abstrak transduser dan penggerak fisik memetakan ke Sensor dan Aktuator."
    },
    {
        "id": "HYDE-THEM-006",
        "category": "THEMATIC_ABSTRACT",
        "program_studi": "TRKB",
        "query": "Integrasi mikroprosesor pengendali cerdas dengan arsitektur jaringan nirkabel global dibahas dalam mata kuliah apa?",
        "target_kode_mk": ["RKS201"],
        "target_nama_mk": ["Sistem Tertanam dan Internet of Things (IoT)"],
        "expected_hypo_keywords": ["sistem tertanam", "internet of things", "mikrokontroler", "konektivitas nirkabel"],
        "rationale": "Kueri konseptual integrasi mikroprosesor dan IoT nirkabel."
    },
    {
        "id": "HYDE-THEM-007",
        "category": "THEMATIC_ABSTRACT",
        "program_studi": "TRKB",
        "query": "Perancangan wahana tanpa awak otonom darat dengan kemampuan navigasi mandiri ada di mata kuliah apa?",
        "target_kode_mk": ["RKS301"],
        "target_nama_mk": ["Mobile Robot"],
        "expected_hypo_keywords": ["mobile robot", "navigasi mandiri", "perencanaan lintasan", "otonom"],
        "rationale": "Kueri abstrak perancangan wahana darat otonom memetakan ke Mobile Robot."
    },
    {
        "id": "HYDE-THEM-008",
        "category": "THEMATIC_ABSTRACT",
        "program_studi": "TRKB",
        "query": "Ekstraksi fitur visual citra digital untuk pengenalan pola lingkungan cerdas diajarkan pada mata kuliah apa?",
        "target_kode_mk": ["RKK303"],
        "target_nama_mk": ["Visi Komputer"],
        "expected_hypo_keywords": ["visi komputer", "ekstraksi fitur", "citra digital", "pengenalan visual"],
        "rationale": "Kueri tematik ekstraksi fitur visual citra digital memetakan ke Visi Komputer."
    },
    {
        "id": "HYDE-THEM-009",
        "category": "THEMATIC_ABSTRACT",
        "program_studi": "RN",
        "query": "Fenomena mekanika gelombang dan probabilitas keadaan partikel skala nanometer diajarkan di mana?",
        "target_kode_mk": ["RND204"],
        "target_nama_mk": ["Nanosains Kuantum"],
        "expected_hypo_keywords": ["nanosains kuantum", "mekanika gelombang", "fungsi gelombang", "skala nano"],
        "rationale": "Kueri tematik mekanika gelombang skala nanometer memetakan ke Nanosains Kuantum."
    },
    {
        "id": "HYDE-THEM-010",
        "category": "THEMATIC_ABSTRACT",
        "program_studi": "RN",
        "query": "Pemodelan berbasis prinsip pertama simulasi atomik material fungsional dipelajari pada mata kuliah apa?",
        "target_kode_mk": ["FIT214"],
        "target_nama_mk": ["Komputasi Nanoteknologi 1"],
        "expected_hypo_keywords": ["komputasi nanoteknologi", "simulasi atomik", "first-principles", "DFT"],
        "rationale": "Kueri pemodelan atomik first-principles memetakan ke Komputasi Nanoteknologi 1."
    },
    {
        "id": "HYDE-THEM-011",
        "category": "THEMATIC_ABSTRACT",
        "program_studi": "RN",
        "query": "Analisis transfer energi termal dan kesetimbangan fase material skala nano dibahas pada mata kuliah apa?",
        "target_kode_mk": ["RND202"],
        "target_nama_mk": ["Kinetika dan Termodinamika"],
        "expected_hypo_keywords": ["kinetika", "termodinamika", "transfer energi", "kesetimbangan fase"],
        "rationale": "Kueri konsep termodinamika fase material memetakan ke Kinetika dan Termodinamika."
    },
    {
        "id": "HYDE-THEM-012",
        "category": "THEMATIC_ABSTRACT",
        "program_studi": "RN",
        "query": "Metodologi analisis difraksi dan mikroskopi elektron untuk verifikasi morfologi partikel diajarkan di mana?",
        "target_kode_mk": ["RNA202"],
        "target_nama_mk": ["Teknik Karakterisasi Material 1"],
        "expected_hypo_keywords": ["karakterisasi material", "difraksi", "mikroskopi elektron", "morfologi partikel"],
        "rationale": "Kueri metodologi analisis difraksi memetakan ke Teknik Karakterisasi Material 1."
    },
    {
        "id": "HYDE-THEM-013",
        "category": "THEMATIC_ABSTRACT",
        "program_studi": "TI",
        "query": "Optimalisasi postur tubuh manusia dan interaksi stasiun kerja industri manufaktur dipelajari di mata kuliah apa?",
        "target_kode_mk": ["TIA201"],
        "target_nama_mk": ["Ergonomi"],
        "expected_hypo_keywords": ["ergonomi", "postur kerja", "antropometri", "lingkungan kerja"],
        "rationale": "Kueri tematik optimalisasi interaksi manusia-mesin memetakan ke Ergonomi."
    },
    {
        "id": "HYDE-THEM-014",
        "category": "THEMATIC_ABSTRACT",
        "program_studi": "TI",
        "query": "Perancangan sistem kendali terprogram untuk lini perakitan pabrik cerdas modern ada di mata kuliah apa?",
        "target_kode_mk": ["TIU201"],
        "target_nama_mk": ["Otomasi Industri"],
        "expected_hypo_keywords": ["otomasi industri", "kendali terprogram", "lini perakitan", "pabrik cerdas"],
        "rationale": "Kueri konseptual lini perakitan cerdas memetakan ke Otomasi Industri."
    },
    {
        "id": "HYDE-THEM-015",
        "category": "THEMATIC_ABSTRACT",
        "program_studi": "TI",
        "query": "Formulasi matematis alokasi sumber daya terbatas untuk optimalisasi sistem produksi dipelajari di mana?",
        "target_kode_mk": ["TID201"],
        "target_nama_mk": ["Penelitian Operasi Dasar"],
        "expected_hypo_keywords": ["penelitian operasi", "optimasi matematis", "alokasi sumber daya", "program linier"],
        "rationale": "Kueri formulasi alokasi sumber daya memetakan ke Penelitian Operasi Dasar."
    },
    {
        "id": "HYDE-THEM-016",
        "category": "THEMATIC_ABSTRACT",
        "program_studi": "TI",
        "query": "Koordinasi jaringan distribusi dan manajemen persediaan dari hulu ke hilir dipelajari dalam mata kuliah apa?",
        "target_kode_mk": ["TIU302"],
        "target_nama_mk": ["Manajemen Rantai Pasok"],
        "expected_hypo_keywords": ["rantai pasok", "distribusi", "persediaan", "supply chain management"],
        "rationale": "Kueri koordinasi distribusi hulu ke hilir memetakan ke Manajemen Rantai Pasok."
    },
    {
        "id": "HYDE-THEM-017",
        "category": "THEMATIC_ABSTRACT",
        "program_studi": "TE",
        "query": "Hukum perambatan gelombang radio dan kopling medan frekuensi tinggi dipelajari di mana?",
        "target_kode_mk": ["TBO201"],
        "target_nama_mk": ["Medan Elektromagnetik"],
        "expected_hypo_keywords": ["medan elektromagnetik", "gelombang radio", "perambatan", "frekuensi tinggi"],
        "rationale": "Kueri tematik perambatan gelombang radio memetakan ke Medan Elektromagnetik."
    },
    {
        "id": "HYDE-THEM-018",
        "category": "THEMATIC_ABSTRACT",
        "program_studi": "TE",
        "query": "Analisis respon transien dan kestabilan umpan balik lingkar tertutup dibahas dalam mata kuliah apa?",
        "target_kode_mk": ["TES203"],
        "target_nama_mk": ["Sistem Kontrol"],
        "expected_hypo_keywords": ["sistem kontrol", "respon transien", "kestabilan", "lingkar tertutup", "PID"],
        "rationale": "Kueri respon transien dan kestabilan umpan balik memetakan ke Sistem Kontrol."
    },
    {
        "id": "HYDE-THEM-019",
        "category": "THEMATIC_ABSTRACT",
        "program_studi": "TE",
        "query": "Infrastruktur konversi energi dan aliran daya listrik kapasitas besar dikaji pada mata kuliah apa?",
        "target_kode_mk": ["TEE202"],
        "target_nama_mk": ["Sistem Tenaga Listrik"],
        "expected_hypo_keywords": ["sistem tenaga listrik", "aliran daya", "konversi energi", "kapasitas daya"],
        "rationale": "Kueri konversi energi daya kapasitas besar memetakan ke Sistem Tenaga Listrik."
    },
    {
        "id": "HYDE-THEM-020",
        "category": "THEMATIC_ABSTRACT",
        "program_studi": "TE",
        "query": "Representasi domain frekuensi dan dekomposisi sinyal informasi analog-digital dipelajari di mana?",
        "target_kode_mk": ["TRS203"],
        "target_nama_mk": ["Sinyal dan Sistem"],
        "expected_hypo_keywords": ["sinyal dan sistem", "domain frekuensi", "dekomposisi sinyal", "transformasi"],
        "rationale": "Kueri representasi domain frekuensi memetakan ke Sinyal dan Sistem."
    },

    # -------------------------------------------------------------
    # 4. CROSS_DISCIPLINARY (20 Queries)
    # -------------------------------------------------------------
    {
        "id": "HYDE-CROSS-001",
        "category": "CROSS_DISCIPLINARY",
        "program_studi": "TRKB",
        "query": "Penerapan teknik computer vision dan deep learning pada wahana robot penjelajah otonom",
        "target_kode_mk": ["RKS301", "RKK303"],
        "target_nama_mk": ["Mobile Robot", "Visi Komputer"],
        "expected_hypo_keywords": ["mobile robot", "visi komputer", "navigasi visual", "deteksi rintangan"],
        "rationale": "Kueri sintetis menghubungkan platform robot bergerak dengan persepsi visi komputer."
    },
    {
        "id": "HYDE-CROSS-002",
        "category": "CROSS_DISCIPLINARY",
        "program_studi": "TRKB",
        "query": "Kombinasi sensor aktuator fisik dengan sistem mikroprosesor terhubung jaringan IoT",
        "target_kode_mk": ["RKS202", "RKS201"],
        "target_nama_mk": ["Sensor dan Aktuator", "Sistem Tertanam dan Internet of Things (IoT)"],
        "expected_hypo_keywords": ["sensor", "aktuator", "sistem tertanam", "IoT", "mikrokontroler"],
        "rationale": "Kueri multi-matkul menggabungkan sensor-aktuator dengan sistem tertanam IoT."
    },
    {
        "id": "HYDE-CROSS-003",
        "category": "CROSS_DISCIPLINARY",
        "program_studi": "TSD",
        "query": "Penggunaan algoritma data mining cerdas pada infrastruktur komputasi data berskala masif",
        "target_kode_mk": ["MAS207", "SID307"],
        "target_nama_mk": ["Data Mining I", "Teknologi Big Data"],
        "expected_hypo_keywords": ["data mining", "big data", "hadoop", "pemodelan skala besar"],
        "rationale": "Kueri menggabungkan algoritma analitik mining dengan platform big data terdistribusi."
    },
    {
        "id": "HYDE-CROSS-004",
        "category": "CROSS_DISCIPLINARY",
        "program_studi": "TE-TRKB",
        "query": "Integrasi sistem kontrol kestabilan umpan balik pada pergerakan lengan robot manipulator",
        "target_kode_mk": ["TES203", "RKS302"],
        "target_nama_mk": ["Sistem Kontrol", "Robot Manipulator"],
        "expected_hypo_keywords": ["sistem kontrol", "robot manipulator", "kinematika lengan", "umpan balik PID"],
        "rationale": "Kueri lintas prodi elektro dan robotika: kontrol umpan balik pada robot manipulator."
    },
    {
        "id": "HYDE-CROSS-005",
        "category": "CROSS_DISCIPLINARY",
        "program_studi": "RN",
        "query": "Penerapan pemodelan komputasi simulasi numerik untuk menganalisis sifat nanosains kuantum material",
        "target_kode_mk": ["FIT214", "RND204"],
        "target_nama_mk": ["Komputasi Nanoteknologi 1", "Nanosains Kuantum"],
        "expected_hypo_keywords": ["komputasi nanoteknologi", "nanosains kuantum", "simulasi atomik", "keadaan partikel"],
        "rationale": "Kueri sintesis metode komputasi numerik dengan teori nanosains kuantum."
    },
    {
        "id": "HYDE-CROSS-006",
        "category": "CROSS_DISCIPLINARY",
        "program_studi": "TI-TRKB",
        "query": "Otomasi industri lini pabrik terintegrasi dengan pemantauan sensor dan aktuator cerdas",
        "target_kode_mk": ["TIU201", "RKS202"],
        "target_nama_mk": ["Otomasi Industri", "Sensor dan Aktuator"],
        "expected_hypo_keywords": ["otomasi industri", "sensor", "aktuator", "lini perakitan", "pengukuran"],
        "rationale": "Kueri lintas disiplin teknik industri manufaktur dan sensor robotika."
    },
    {
        "id": "HYDE-CROSS-007",
        "category": "CROSS_DISCIPLINARY",
        "program_studi": "TE",
        "query": "Sistem transmisi energi tenaga listrik kapasitas besar yang diatur oleh sistem kontrol otomatis",
        "target_kode_mk": ["TEE202", "TES203"],
        "target_nama_mk": ["Sistem Tenaga Listrik", "Sistem Kontrol"],
        "expected_hypo_keywords": ["sistem tenaga listrik", "sistem kontrol", "transmisi daya", "pengaturan kestabilan"],
        "rationale": "Kueri penggabungan sistem tenaga listrik dengan sistem kontrol otomatis."
    },
    {
        "id": "HYDE-CROSS-008",
        "category": "CROSS_DISCIPLINARY",
        "program_studi": "TRKB-TI",
        "query": "Pengolahan citra visi komputer untuk inspeksi cacat mutu pada lini manufaktur industri",
        "target_kode_mk": ["RKK303", "TIA303"],
        "target_nama_mk": ["Visi Komputer", "Penjaminan dan Pengendalian Mutu"],
        "expected_hypo_keywords": ["visi komputer", "pengendalian mutu", "inspeksi visual", "cacat produk"],
        "rationale": "Kueri sintesis visi komputer untuk quality assurance di teknik industri."
    },
    {
        "id": "HYDE-CROSS-009",
        "category": "CROSS_DISCIPLINARY",
        "program_studi": "TI-TSD",
        "query": "Analisis rantai pasok logistik global yang dioptimalkan menggunakan teknik penambangan data spasial",
        "target_kode_mk": ["TIU302", "MAS317"],
        "target_nama_mk": ["Manajemen Rantai Pasok", "Analisis Data Spasial"],
        "expected_hypo_keywords": ["manajemen rantai pasok", "analisis data spasial", "logistik geospasial", "rute distribusi"],
        "rationale": "Kueri sintesis optimasi logistik rantai pasok dengan analisis spasial pemetaan wilayah."
    },
    {
        "id": "HYDE-CROSS-010",
        "category": "CROSS_DISCIPLINARY",
        "program_studi": "RN-TRKB",
        "query": "Karakterisasi material fungsional nanoteknologi untuk pengembangan komponen sensor transduser baru",
        "target_kode_mk": ["RNA202", "RKS202"],
        "target_nama_mk": ["Teknik Karakterisasi Material 1", "Sensor dan Aktuator"],
        "expected_hypo_keywords": ["karakterisasi material", "sensor", "aktuator", "nanoteknologi", "transduser nano"],
        "rationale": "Kueri lintas rekayasa nanoteknologi dan instrumentasi sensor robotika."
    },
    {
        "id": "HYDE-CROSS-011",
        "category": "CROSS_DISCIPLINARY",
        "program_studi": "TE",
        "query": "Pengembangan sistem kendali umpan balik berbasis analisis representasi sinyal dan sistem informasi",
        "target_kode_mk": ["TES203", "TRS203"],
        "target_nama_mk": ["Sistem Kontrol", "Sinyal dan Sistem"],
        "expected_hypo_keywords": ["sistem kontrol", "sinyal dan sistem", "umpan balik", "laplace", "tanggapan frekuensi"],
        "rationale": "Kueri sinergis pemrosesan sinyal dengan perancangan pengendali umpan balik."
    },
    {
        "id": "HYDE-CROSS-012",
        "category": "CROSS_DISCIPLINARY",
        "program_studi": "TSD-TE",
        "query": "Infrastruktur teknologi data skala besar untuk analisis keandalan aliran transmisi tenaga listrik cerdas",
        "target_kode_mk": ["SID307", "TEE202"],
        "target_nama_mk": ["Teknologi Big Data", "Sistem Tenaga Listrik"],
        "expected_hypo_keywords": ["teknologi big data", "sistem tenaga listrik", "smart grid", "data aliran daya"],
        "rationale": "Kueri smart grid: komputasi big data untuk manajemen tenaga listrik."
    },
    {
        "id": "HYDE-CROSS-013",
        "category": "CROSS_DISCIPLINARY",
        "program_studi": "TI",
        "query": "Formulasi optimasi matematika penelitian operasi untuk perancangan stasiun kerja ergonomis industri",
        "target_kode_mk": ["TID201", "TIA201"],
        "target_nama_mk": ["Penelitian Operasi Dasar", "Ergonomi"],
        "expected_hypo_keywords": ["penelitian operasi", "ergonomi", "optimasi stasiun kerja", "beban kerja"],
        "rationale": "Kueri kombinasi optimasi matematis dengan perancangan ergonomi stasiun kerja."
    },
    {
        "id": "HYDE-CROSS-014",
        "category": "CROSS_DISCIPLINARY",
        "program_studi": "RN",
        "query": "Simulasi komputasi ab-initio untuk memprediksi kinetika termodinamika fase material nano",
        "target_kode_mk": ["FIT214", "RND202"],
        "target_nama_mk": ["Komputasi Nanoteknologi 1", "Kinetika dan Termodinamika"],
        "expected_hypo_keywords": ["komputasi nanoteknologi", "kinetika dan termodinamika", "simulasi fase", "energi bebas"],
        "rationale": "Kueri pemodelan komputasi untuk termodinamika material kuantum."
    },
    {
        "id": "HYDE-CROSS-015",
        "category": "CROSS_DISCIPLINARY",
        "program_studi": "TRKB",
        "query": "Pemanfaatan visi komputer pada robot mobile untuk pemetaan lingkungan dan lokalisasi simultan SLAM",
        "target_kode_mk": ["RKK303", "RKS301"],
        "target_nama_mk": ["Visi Komputer", "Mobile Robot"],
        "expected_hypo_keywords": ["visi komputer", "mobile robot", "visual SLAM", "deteksi landmark"],
        "rationale": "Kueri visual SLAM pada robot bergerak."
    },
    {
        "id": "HYDE-CROSS-016",
        "category": "CROSS_DISCIPLINARY",
        "program_studi": "TSD-TRKB",
        "query": "Integrasi sistem basis data terstruktur dengan pemantauan perangkat keras embedded IoT di lapangan",
        "target_kode_mk": ["SID206", "RKS201"],
        "target_nama_mk": ["Pengantar Basis Data", "Sistem Tertanam dan Internet of Things (IoT)"],
        "expected_hypo_keywords": ["pengantar basis data", "sistem tertanam", "IoT database", "penyimpanan telemetri"],
        "rationale": "Kueri telemetri data IoT yang disimpan ke database relasional."
    },
    {
        "id": "HYDE-CROSS-017",
        "category": "CROSS_DISCIPLINARY",
        "program_studi": "TE-TRKB",
        "query": "Analisis perambatan medan gelombang elektromagnetik untuk sistem pengisian daya nirkabel robot mobile",
        "target_kode_mk": ["TBO201", "RKS301"],
        "target_nama_mk": ["Medan Elektromagnetik", "Mobile Robot"],
        "expected_hypo_keywords": ["medan elektromagnetik", "mobile robot", "wireless charging", "induksi magnetik"],
        "rationale": "Kueri pengisian daya nirkabel berbasis elektromagnetik pada robot otonom."
    },
    {
        "id": "HYDE-CROSS-018",
        "category": "CROSS_DISCIPLINARY",
        "program_studi": "RN-TE",
        "query": "Karakterisasi material nanolitografi untuk fabrikasi komponen rangkaian elektronika semikonduktor",
        "target_kode_mk": ["RND206", "TPB203"],
        "target_nama_mk": ["Teknik dan Fabrikasi Nanolitografi", "Rangkaian Elektronika"],
        "expected_hypo_keywords": ["nanolitografi", "rangkaian elektronika", "fabrikasi semikonduktor", "komponen mikro"],
        "rationale": "Kueri fabrikasi nanolitografi dengan desain rangkaian elektronika."
    },
    {
        "id": "HYDE-CROSS-019",
        "category": "CROSS_DISCIPLINARY",
        "program_studi": "TI-TSD",
        "query": "Penerapan algoritma optimasi kombinatorik pada perencanaan dan pengendalian produksi manufaktur pabrik",
        "target_kode_mk": ["MAA209", "TIA304"],
        "target_nama_mk": ["Optimasi Kombinatorik", "Perencanaan dan Pengendalian Produksi"],
        "expected_hypo_keywords": ["optimasi kombinatorik", "perencanaan produksi", "penjadwalan manufaktur", "PPC"],
        "rationale": "Kueri penjadwalan produksi dengan algoritma optimasi kombinatorik."
    },
    {
        "id": "HYDE-CROSS-020",
        "category": "CROSS_DISCIPLINARY",
        "program_studi": "TE",
        "query": "Analisis spektrum domain frekuensi sinyal untuk perancangan sistem telekomunikasi cerdas",
        "target_kode_mk": ["TRS203", "TET201"],
        "target_nama_mk": ["Sinyal dan Sistem", "Dasar Sistem dan Jaringan Telekomunikasi"],
        "expected_hypo_keywords": ["sinyal dan sistem", "telekomunikasi", "spektrum frekuensi", "modulasi sinyal"],
        "rationale": "Kueri pemrosesan sinyal untuk sistem jaringan telekomunikasi."
    },
]


def validate_dataset_against_catalog(dataset: list[dict], catalog_path: Path):
    with open(catalog_path, "r", encoding="utf-8") as f:
        catalog = json.load(f)

    catalog_lookup = {c["kode_mk"]: c["nama_mk"] for c in catalog}

    errors = []
    for item in dataset:
        for kode, nama in zip(item["target_kode_mk"], item["target_nama_mk"]):
            if kode not in catalog_lookup:
                errors.append(f"[{item['id']}] Kode '{kode}' not found in catalog!")
            elif catalog_lookup[kode] != nama:
                errors.append(
                    f"[{item['id']}] Kode '{kode}' name mismatch: expected '{catalog_lookup[kode]}', got '{nama}'"
                )

    if errors:
        for err in errors:
            logger.error("VALIDATION ERROR: %s", err)
        raise ValueError(f"Catalog validation failed with {len(errors)} error(s)!")
    logger.info("✓ All %d dataset queries validated 100%% against %s courses in catalog.", len(dataset), len(catalog))


def sync_to_google_sheets(dataset: list[dict]):
    token = get_access_token()
    sheet_title = "HYDE_EVAL"
    logger.info("Checking Google Spreadsheet tab '%s'...", sheet_title)

    meta_map = get_all_sheets_meta(token)
    sheet_id = None
    if sheet_title in meta_map:
        sheet_id = meta_map[sheet_title]["sheetId"]
        logger.info("Sheet '%s' already exists (sheetId=%s). Clearing...", sheet_title, sheet_id)
        endpoint = f"/values/{sheet_title}!A1:Z500:clear"
        execute_sheets_api("POST", endpoint, token, body={})
    else:
        logger.info("Creating new tab '%s'...", sheet_title)
        create_req = {
            "requests": [
                {
                    "addSheet": {
                        "properties": {
                            "title": sheet_title,
                            "gridProperties": {"frozenRowCount": 1},
                            "tabColor": {"red": 0.2, "green": 0.5, "blue": 0.8},
                        }
                    }
                }
            ]
        }
        res = execute_sheets_api("POST", ":batchUpdate", token, body=create_req)
        sheet_id = res["replies"][0]["addSheet"]["properties"]["sheetId"]
        logger.info("Created tab '%s' (sheetId=%s).", sheet_title, sheet_id)

    header = [
        "ID",
        "Kategori Pengujian",
        "Program Studi",
        "Kueri Mahasiswa (Query)",
        "Target Kode MK",
        "Target Nama MK",
        "Kata Kunci Hipotesis (HyDE Keywords)",
        "Rasional Ilmiah & Hipotesis",
    ]
    rows = [header]
    for q in dataset:
        rows.append([
            q["id"],
            q["category"],
            q["program_studi"],
            q["query"],
            ", ".join(q["target_kode_mk"]),
            ", ".join(q["target_nama_mk"]),
            ", ".join(q["expected_hypo_keywords"]),
            q["rationale"],
        ])

    endpoint = f"/values/{sheet_title}!A1:append?valueInputOption=RAW&insertDataOption=OVERWRITE"
    execute_sheets_api("POST", endpoint, token, body={"values": rows})

    # Styling header
    format_req = {
        "requests": [
            {
                "repeatCell": {
                    "range": {"sheetId": sheet_id, "startRowIndex": 0, "endRowIndex": 1},
                    "cell": {
                        "userEnteredFormat": {
                            "backgroundColor": {"red": 0.15, "green": 0.35, "blue": 0.6},
                            "textFormat": {"foregroundColor": {"red": 1.0, "green": 1.0, "blue": 1.0}, "bold": True},
                            "horizontalAlignment": "CENTER",
                        }
                    },
                    "fields": "userEnteredFormat(backgroundColor,textFormat,horizontalAlignment)",
                }
            },
            {
                "autoResizeDimensions": {
                    "dimensions": {"sheetId": sheet_id, "dimension": "COLUMNS", "startIndex": 0, "endIndex": 8}
                }
            },
        ]
    }
    execute_sheets_api("POST", ":batchUpdate", token, body=format_req)
    logger.info("✓ Uploaded %d rows to Google Spreadsheet '%s' successfully.", len(rows), sheet_title)


def main():
    logger.info("Compiling HyDE Evaluation Dataset (80 queries)...")
    validate_dataset_against_catalog(RAW_QUERIES, CATALOG_PATH)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(RAW_QUERIES, f, indent=2, ensure_ascii=False)
    logger.info("✓ Saved dataset to %s (%d items).", OUTPUT_PATH, len(RAW_QUERIES))

    sync_to_google_sheets(RAW_QUERIES)
    logger.info("Dataset compilation & Google Sheets sync complete!")


if __name__ == "__main__":
    main()
