#!/usr/bin/env python3
"""
Build Cross-Cohort Profile Matrix Benchmark Dataset (hoBIT, Kim et al. 2026).
arXiv:2608.26604:
Profile-grounded academic advising evaluation across 5 prodi and 4 cohorts (2021, 2022, 2023, 2024).
Tests whether the system resolves profile-dependent questions correctly without
the student having to specify their cohort year in the prompt text.
Total: 80 queries (16 per prodi).
"""

import json
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("build_cohort_gt")

OUTPUT_PATH = Path("data/ground_truth/cohort_profile_benchmark.json")

# Define base scenarios per prodi (4 per prodi × 4 cohort/style variations = 16 per prodi = 80 total)
COHORT_TEMPLATES = [
    # ========================== 1. TSD ==========================
    {
        "prodi": "Teknologi Sains Data",
        "prodi_code": "TSD",
        "domain": "SKS_PACKAGE_POLICY",
        "query_templates": [
            {
                "style": "verification_binary",
                "query": "Apakah saya boleh memilih mata kuliah sendiri dan tidak mengambil sistem paket di semester 2?",
                "cohort_rules": {
                    2021: {"verdict": True, "reason": "Kurikulum 2021 mengizinkan pemilihan KRS mandiri bersyarat setelah evaluasi semester 1."},
                    2024: {"verdict": False, "reason": "Berdasarkan Panduan Pendidikan 2024, semester 1 dan semester 2 adalah sistem paket penuh."},
                },
            },
            {
                "style": "first_person",
                "query": "Berapa beban SKS maksimal yang boleh saya ambil di semester depan kalau IPS saya 2.85?",
                "cohort_rules": {
                    2021: {"verdict": "Maksimal 20 SKS sesuai aturan rentang IPS 2.50-2.99 pada Panduan Akademik 2021.", "reason": "Rentang beban SKS IPS 2.50-2.99 adalah 20 SKS."},
                    2024: {"verdict": "Maksimal 20 SKS sesuai Pasal 5.4 Panduan Akademik FTMM.", "reason": "Rentang beban SKS IPS 2.50-2.99 adalah 20 SKS."},
                },
            },
            {
                "style": "verification_binary",
                "query": "Apakah saya yang sudah menempuh 108 SKS boleh langsung memprogram Proposal Skripsi semester depan?",
                "cohort_rules": {
                    2021: {"verdict": True, "reason": "Kurikulum TSD 2021 menetapkan ambang batas syarat Proposal Skripsi minimal 108 SKS kumulatif."},
                    2024: {"verdict": False, "reason": "Kurikulum TSD 2024 mensyaratkan minimal 110 SKS kumulatif lulus dan telah lulus Metodologi Penelitian."},
                },
            },
            {
                "style": "formal",
                "query": "Apa syarat kelulusan mata kuliah Praktik Kerja Lapangan (PKL) terkait batas minimal perolehan SKS?",
                "cohort_rules": {
                    2021: {"verdict": "Minimal 80 SKS kumulatif dan duduk di semester 5.", "reason": "Syarat PKL standar FTMM 80 SKS."},
                    2024: {"verdict": "Minimal 80 SKS kumulatif, duduk di semester 5, dan durasi kerja lapangan minimal 18 hari kerja efektif.", "reason": "Syarat formal PKL 2024 dengan kepatuhan durasi 18 hari kerja."},
                },
            },
        ],
    },
    # ========================== 2. TRKB ==========================
    {
        "prodi": "Teknik Robotika dan Kecerdasan Buatan",
        "prodi_code": "TRKB",
        "domain": "CURRICULUM_REVISION_MANDATORY",
        "query_templates": [
            {
                "style": "verification_binary",
                "query": "Apakah mata kuliah Pancasila wajib diprogram pada semester 1?",
                "cohort_rules": {
                    2021: {"verdict": False, "reason": "Pada kurikulum TRKB 2021, Pendidikan Pancasila dialokasikan pada semester 2."},
                    2024: {"verdict": True, "reason": "Pada kurikulum TRKB 2024, Pendidikan Pancasila (NOP103) wajib diambil pada paket semester 1."},
                },
            },
            {
                "style": "verification_binary",
                "query": "Apakah saya boleh mengajukan seminar PKL jika logbook baru terisi 12 hari kerja?",
                "cohort_rules": {
                    2021: {"verdict": False, "reason": "Minimal kerja lapangan adalah 18 hari kerja efektif."},
                    2024: {"verdict": False, "reason": "Minimal kerja lapangan adalah 18 hari kerja efektif dan wajib verifikasi pembimbing."},
                },
            },
            {
                "style": "first_person",
                "query": "Kalau SKS kumulatif saya saat ini 112 tapi nilai D saya ada 25%, apakah saya boleh mendaftar yudisium?",
                "cohort_rules": {
                    2021: {"verdict": False, "reason": "Batas toleransi nilai D maksimal 20% dari total SKS kumulatif dan nilai E wajib 0 SKS."},
                    2024: {"verdict": False, "reason": "Maksimal nilai D adalah 20% dan total kelulusan sarjana wajib minimal 144 SKS."},
                },
            },
            {
                "style": "formal",
                "query": "Apakah mata kuliah Eksperimen Robotika I memiliki prasyarat kelulusan mata kuliah lain?",
                "cohort_rules": {
                    2021: {"verdict": "Prasyarat Eksperimen Robotika I adalah Rangkaian Listrik dan Dasar Elektronika.", "reason": "Prasyarat laboratorium kurikulum 2021."},
                    2024: {"verdict": "Prasyarat Eksperimen Robotika I adalah Rangkaian Elektronika dan Pemrograman.", "reason": "Prasyarat laboratorium kurikulum 2024."},
                },
            },
        ],
    },
    # ========================== 3. RN ==========================
    {
        "prodi": "S1 Rekayasa Nanoteknologi",
        "prodi_code": "RN",
        "domain": "SCIENCE_LAB_PREREQUISITE",
        "query_templates": [
            {
                "style": "verification_binary",
                "query": "Apakah Praktikum Kimia Dasar 1 wajib diambil berbarengan dengan teori Kimia Dasar di semester 1?",
                "cohort_rules": {
                    2021: {"verdict": True, "reason": "KID102 (praktikum) merupakan ko-requisite mata kuliah teori KID105."},
                    2024: {"verdict": True, "reason": "KID102 dan KID105 adalah satu kesatuan paket semester 1 RN."},
                },
            },
            {
                "style": "verification_binary",
                "query": "Apakah saya yang angkatan lama boleh mengambil Skripsi jika baru menempuh 115 SKS?",
                "cohort_rules": {
                    2021: {"verdict": False, "reason": "Mata kuliah Skripsi (PNJ499) mensyaratkan minimal 120 SKS kumulatif."},
                    2024: {"verdict": False, "reason": "Syarat Skripsi adalah lulus Proposal Skripsi dan minimal 120 SKS kumulatif."},
                },
            },
            {
                "style": "first_person",
                "query": "Berapa SKS minimal yang harus saya selesaikan untuk bisa lulus sarjana di Nanoteknologi?",
                "cohort_rules": {
                    2021: {"verdict": "Minimal 144 SKS kumulatif (maksimal 160 SKS).", "reason": "Standar kelulusan sarjana SN-Dikti FTMM."},
                    2024: {"verdict": "Minimal 144 SKS kumulatif dengan 0 SKS nilai E.", "reason": "Standar kelulusan sarjana FTMM 2024."},
                },
            },
            {
                "style": "formal",
                "query": "Bagaimana ketentuan konversi nilai magang MBKM ke dalam mata kuliah pilihan di RN?",
                "cohort_rules": {
                    2021: {"verdict": "Konversi magang MBKM maksimal hingga 20 SKS melalui persetujuan KPS.", "reason": "Aturan MBKM angkatan 2021."},
                    2024: {"verdict": "Konversi magang wajib melalui verifikasi form konversi di awal semester dan disahkan tim kurikulum.", "reason": "Aturan MBKM angkatan 2024."},
                },
            },
        ],
    },
    # ========================== 4. TI ==========================
    {
        "prodi": "Teknik Industri",
        "prodi_code": "TI",
        "domain": "OPERATIONS_AND_PREREQUISITE",
        "query_templates": [
            {
                "style": "verification_binary",
                "query": "Apakah mata kuliah Penelitian Operasi Lanjut boleh saya ambil tanpa lulus Penelitian Operasi Dasar?",
                "cohort_rules": {
                    2021: {"verdict": False, "reason": "TID201 merupakan prasyarat mutlak untuk TIA302."},
                    2024: {"verdict": False, "reason": "TIA302 mewajibkan kelulusan TID201 (Penelitian Operasi Dasar)."},
                },
            },
            {
                "style": "verification_binary",
                "query": "Apakah mahasiswa yang memiliki IPS 3.40 berhak memprogram hingga 24 SKS?",
                "cohort_rules": {
                    2021: {"verdict": True, "reason": "IPS >= 3.00 berhak mengambil batas maksimal 24 SKS."},
                    2024: {"verdict": True, "reason": "Pasal 5.4 menetapkan IPS >= 3.00 berhak memprogram hingga 24 SKS."},
                },
            },
            {
                "style": "first_person",
                "query": "Kalau saya belum lulus mata kuliah Statistik Industri, apakah saya bisa memprogram Simulasi Komputer?",
                "cohort_rules": {
                    2021: {"verdict": False, "reason": "Statistik Industri (TID202) adalah prasyarat untuk Simulasi Komputer (TIK301)."},
                    2024: {"verdict": False, "reason": "Simulasi Komputer mewajibkan kelulusan Statistik Industri."},
                },
            },
            {
                "style": "formal",
                "query": "Kapan mahasiswa Teknik Industri diperkenankan mulai mengajukan mata kuliah PKL?",
                "cohort_rules": {
                    2021: {"verdict": "Mulai semester 5 dengan syarat perolehan minimal 80 SKS kumulatif.", "reason": "Aturan PKL FTMM."},
                    2024: {"verdict": "Mulai semester 5 dengan syarat perolehan minimal 80 SKS kumulatif dan persetujuan dosen wali.", "reason": "Aturan PKL FTMM 2024."},
                },
            },
        ],
    },
    # ========================== 5. TE ==========================
    {
        "prodi": "Teknik Elektro",
        "prodi_code": "TE",
        "domain": "ELECTRICAL_CURRICULUM_DAG",
        "query_templates": [
            {
                "style": "verification_binary",
                "query": "Apakah Praktikum Rangkaian Listrik boleh diambil di semester 4 jika di semester 3 belum lulus?",
                "cohort_rules": {
                    2021: {"verdict": True, "reason": "Praktikum dapat diulang pada semester berikutnya jika dibuka pada semester genap/SP."},
                    2024: {"verdict": False, "reason": "TBE202 hanya ditawarkan pada paritas semester ganjil (semester 3)."},
                },
            },
            {
                "style": "verification_binary",
                "query": "Apakah saya berhak mengambil 24 SKS jika IPS semester lalu saya 2.45?",
                "cohort_rules": {
                    2021: {"verdict": False, "reason": "IPS 2.00-2.49 dibatasi maksimal 18 SKS."},
                    2024: {"verdict": False, "reason": "Berdasarkan Pasal 5.4 Panduan Akademik, IPS 2.00-2.49 dibatasi maksimal 18 SKS."},
                },
            },
            {
                "style": "first_person",
                "query": "Saya mau ambil Sistem Tenaga Listrik semester 4, apa harus lulus Rangkaian Listrik dulu?",
                "cohort_rules": {
                    2021: {"verdict": True, "reason": "TBE201 (Rangkaian Listrik) adalah prasyarat TEE202 (Sistem Tenaga Listrik)."},
                    2024: {"verdict": True, "reason": "Rangkaian Listrik adalah prasyarat mutlak untuk Sistem Tenaga Listrik."},
                },
            },
            {
                "style": "formal",
                "query": "Berapa batas maksimal persentase nilai D untuk syarat kelulusan yudisium sarjana Teknik Elektro?",
                "cohort_rules": {
                    2021: {"verdict": "Maksimal 20% dari total SKS kelulusan, dan wajib 0 SKS nilai E.", "reason": "Panduan Yudisium FTMM."},
                    2024: {"verdict": "Maksimal 20% dari total SKS kelulusan, dan wajib 0 SKS nilai E.", "reason": "Panduan Yudisium FTMM 2024."},
                },
            },
        ],
    },
]


def assemble_cohort_benchmark() -> list[dict]:
    dataset = []
    bench_id = 1
    cohorts = [2021, 2022, 2023, 2024]

    for prodi_group in COHORT_TEMPLATES:
        prodi_name = prodi_group["prodi"]
        prodi_code = prodi_group["prodi_code"]
        domain = prodi_group["domain"]

        for tmpl in prodi_group["query_templates"]:
            style = tmpl["style"]
            query_text = tmpl["query"]
            cohort_rules = tmpl["cohort_rules"]

            for cohort_year in cohorts:
                # Determine applicable rule; map 2022 to 2021 baseline or 2023 to 2024 baseline
                rule_key = 2024 if cohort_year >= 2023 else 2021
                rule_data = cohort_rules[rule_key]

                # Determine contrast cohort
                contrast_year = 2021 if cohort_year >= 2023 else 2024
                contrast_data = cohort_rules[contrast_year]

                sem_default = 1 if "semester 1" in query_text.lower() else (4 if "semester 4" in query_text.lower() else 5)
                completed_sks_default = 18 if sem_default <= 2 else (75 if sem_default == 4 else 112)

                dataset.append({
                    "id": f"CPB-{bench_id:03d}",
                    "prodi": prodi_code,
                    "program_studi": prodi_name,
                    "domain": domain,
                    "style": style,
                    "query": query_text,
                    "session_profile": {
                        "program_studi": prodi_name,
                        "prodi_code": prodi_code,
                        "angkatan": cohort_year,
                        "semester_saat_ini": sem_default,
                        "completed_sks": completed_sks_default,
                    },
                    "expected_verdict": rule_data["verdict"],
                    "verdict_reason": rule_data["reason"],
                    "contrast_cohort": contrast_year,
                    "contrast_verdict": contrast_data["verdict"],
                    "contrast_reason": contrast_data["reason"],
                    "is_profile_dependent": rule_data["verdict"] != contrast_data["verdict"],
                    "curriculum_version": f"Kurikulum {rule_key}",
                })
                bench_id += 1

    return dataset


def main():
    dataset = assemble_cohort_benchmark()
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(dataset, f, indent=2, ensure_ascii=False)

    logger.info("✓ Berhasil membuat %s dengan %d skenario kohort", OUTPUT_PATH, len(dataset))
    by_prodi = {}
    for d in dataset:
        by_prodi[d["prodi"]] = by_prodi.get(d["prodi"], 0) + 1
    for p, c in by_prodi.items():
        logger.info("  - %-10s : %2d skenario", p, c)


if __name__ == "__main__":
    main()
