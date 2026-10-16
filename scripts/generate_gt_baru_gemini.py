#!/usr/bin/env python3
"""
Generate complete 369 course ground truth dataset matching sheet GT style.
Covers 100% of courses in data/courses_catalog.json (297 MK Wajib + 72 MK Pilihan).
Output: data/ground_truth/gt_baru_courses.json
"""

import json
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("generate_gt_baru")

CATALOG_PATH = Path("data/courses_catalog.json")
OUTPUT_PATH = Path("data/ground_truth/gt_baru_courses.json")


def clean_text(text: str) -> str:
    """Normalize whitespace and newlines in text."""
    if not text:
        return ""
    lines = [line.strip() for line in text.replace("\r", "\n").split("\n") if line.strip()]
    return " ".join(lines)


def build_question_and_answer(course: dict, idx: int) -> dict:
    kode_mk = course.get("kode_mk", "").strip()
    nama_mk = course.get("nama_mk", "").strip()
    sks = course.get("sks", 0)
    sem = course.get("semester", 1)
    prodi = course.get("program_studi", "FTMM").strip()
    jenis = course.get("jenis", "wajib").strip()
    is_pilihan = "pilihan" in jenis.lower()
    source_type = "Silabus Pilihan" if is_pilihan else "Silabus Wajib"

    prasyarat_list = course.get("prasyarat", [])
    if isinstance(prasyarat_list, list) and prasyarat_list:
        clean_prasyarat = [p.strip() for p in prasyarat_list if str(p).strip()]
        prasyarat_str = f"memiliki prasyarat kelulusan mata kuliah {', '.join(clean_prasyarat)}"
    else:
        prasyarat_str = "tidak ada prasyarat"

    deskripsi = clean_text(course.get("deskripsi", ""))
    if not deskripsi:
        deskripsi = f"kajian komprehensif konsep teoritis dan aplikasi praktis mata kuliah {nama_mk} dalam bidang {prodi}"

    cp_list = course.get("capaian_pembelajaran", [])
    if isinstance(cp_list, list) and cp_list:
        clean_cp = clean_text(cp_list[0])
    elif isinstance(cp_list, str) and cp_list.strip():
        clean_cp = clean_text(cp_list)
    else:
        clean_cp = f"mampu menguasai kompetensi dasar dan terapan dari mata kuliah {nama_mk} secara profesional dan beretika"

    # Formulate question based on index mod to maintain natural variation in sheet GT style
    var_type = idx % 3
    if var_type == 0:
        question = f"Pak/Bu, saya mau tanya, secara singkat mata kuliah {nama_mk} ({kode_mk}) di prodi {prodi} itu membahas apa saja dan berapa SKS?"
    elif var_type == 1:
        question = f"Pak/Bu, kalau saya mau mengambil mata kuliah {nama_mk} ({kode_mk}) di semester {sem}, berapa beban studinya, apakah ada prasyarat, dan apa capaian pembelajarannya?"
    else:
        question = f"Permisi Pak/Bu, untuk mata kuliah {nama_mk} ({kode_mk}) di kurikulum {prodi}, apakah ini mata kuliah wajib atau pilihan, berada di semester berapa, dan apa saja materi pokoknya?"

    # Formulate authoritative answer strictly adhering to sheet GT pattern
    status_matkul = "mata kuliah pilihan" if is_pilihan else "mata kuliah wajib"
    answer = (
        f"Berdasarkan silabus, mata kuliah {nama_mk} ({kode_mk}) merupakan {status_matkul} pada program studi {prodi}, "
        f"berada di semester {sem}, berbobot {sks} SKS, {prasyarat_str}, dan secara garis besar membahas {deskripsi}. "
        f"Capaian pembelajaran lulusan (CPMK) mata kuliah ini adalah mahasiswa {clean_cp}."
    )

    return {
        "id": str(idx + 1),
        "question": question,
        "answer": answer,
        "source_id": kode_mk,
        "source_title": nama_mk,
        "source_section": "Deskripsi Mata Kuliah/Silabus",
        "source_type": source_type,
        "program_studi": prodi,
        "kode_mk": kode_mk,
        "nama_mk": nama_mk,
        "sks": sks,
        "semester": sem,
        "jenis": jenis,
        "prasyarat": prasyarat_list,
    }


def generate_all_courses_gt():
    if not CATALOG_PATH.exists():
        raise FileNotFoundError(f"Catalog not found at {CATALOG_PATH}")

    with open(CATALOG_PATH, "r", encoding="utf-8") as f:
        courses = json.load(f)

    logger.info("Loaded %d courses from %s", len(courses), CATALOG_PATH)
    gt_entries = []

    for idx, c in enumerate(courses):
        entry = build_question_and_answer(c, idx)
        gt_entries.append(entry)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(gt_entries, f, indent=2, ensure_ascii=False)

    logger.info("✓ Successfully generated %d ground truth entries -> %s", len(gt_entries), OUTPUT_PATH)

    prodi_counts = {}
    type_counts = {}
    for item in gt_entries:
        prodi_counts[item["program_studi"]] = prodi_counts.get(item["program_studi"], 0) + 1
        type_counts[item["source_type"]] = type_counts.get(item["source_type"], 0) + 1

    logger.info("Distribution by Program Studi:")
    for p, count in prodi_counts.items():
        logger.info("  - %-38s : %3d MK", p, count)

    logger.info("Distribution by Course Type:")
    for t, count in type_counts.items():
        logger.info("  - %-25s : %3d MK", t, count)


if __name__ == "__main__":
    generate_all_courses_gt()
