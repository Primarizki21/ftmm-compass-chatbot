#!/usr/bin/env python3
"""
Generate unified master ground truth dataset (GT_MASTER).
Combines:
1. 369 course Q&As from data/ground_truth/gt_baru_courses.json (100% catalog coverage).
2. 80 institutional regulations, academic guide, and SOP Q&As from data/ground_truth/google_drive_gt.json.
Total: 449 authoritative Q&As.
Output: data/ground_truth/gt_master.json
"""

import json
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("generate_gt_master")

GT_BARU_PATH = Path("data/ground_truth/gt_baru_courses.json")
GT_LAMA_PATH = Path("data/ground_truth/google_drive_gt.json")
OUTPUT_PATH = Path("data/ground_truth/gt_master.json")

REQUIRED_COLUMNS = [
    "id",
    "question",
    "answer",
    "source_id",
    "source_title",
    "source_section",
    "source_type",
]


def clean_str(val) -> str:
    if val is None:
        return ""
    lines = [line.strip() for line in str(val).replace("\r", "\n").split("\n") if line.strip()]
    return " ".join(lines)


def is_course_entry(entry: dict) -> bool:
    """Detect whether entry is a course syllabus query or general institutional SOP/regulation."""
    sec = str(entry.get("source_section", "")).lower()
    sid = str(entry.get("source_id", "")).strip()

    # Match 6-character institutional course codes e.g. AGI101, TSD101, MAL204
    is_mk_code = len(sid) == 6 and sid[:3].isalpha() and sid[3:].isdigit()
    course_keywords = ["silabus", "semester", "prasyarat", "beban studi", "cpl", "deskripsi"]
    return is_mk_code or any(k in sec for k in course_keywords)


def build_gt_master():
    if not GT_BARU_PATH.exists():
        raise FileNotFoundError(f"Missing {GT_BARU_PATH}")
    if not GT_LAMA_PATH.exists():
        raise FileNotFoundError(f"Missing {GT_LAMA_PATH}")

    with open(GT_BARU_PATH, "r", encoding="utf-8") as f:
        courses_data = json.load(f)
    with open(GT_LAMA_PATH, "r", encoding="utf-8") as f:
        lama_data = json.load(f)

    logger.info("Loaded %d course entries from %s", len(courses_data), GT_BARU_PATH)
    logger.info("Loaded %d raw entries from %s", len(lama_data), GT_LAMA_PATH)

    # Extract non-course entries from gt_lama (regulasi, panduan, sop)
    sop_reg_entries = [d for d in lama_data if not is_course_entry(d)]
    logger.info("Extracted %d non-course regulations & SOP entries from gt_lama", len(sop_reg_entries))

    master_entries = []
    current_id = 1

    # Part 1: All 369 courses from GT_BARU
    for item in courses_data:
        entry = {
            "id": str(current_id),
            "question": clean_str(item.get("question")),
            "answer": clean_str(item.get("answer")),
            "source_id": clean_str(item.get("source_id")),
            "source_title": clean_str(item.get("source_title")),
            "source_section": clean_str(item.get("source_section", "Deskripsi Mata Kuliah/Silabus")),
            "source_type": clean_str(item.get("source_type", "Silabus Wajib")),
            "domain": "course_curriculum",
        }
        # Keep additional metadata if available
        if "program_studi" in item:
            entry["program_studi"] = item["program_studi"]
        if "sks" in item:
            entry["sks"] = item["sks"]
        if "semester" in item:
            entry["semester"] = item["semester"]

        master_entries.append(entry)
        current_id += 1

    # Part 2: All 80 SOP and regulation entries from GT_LAMA
    for item in sop_reg_entries:
        stype = clean_str(item.get("source_type"))
        if not stype or stype in ["None", "null"]:
            stype = "Regulasi Akademik"
        elif "sop" in stype.lower() or "pengajuan" in stype.lower():
            stype = "SOP Layanan"
        else:
            stype = "Regulasi Akademik"

        sec = clean_str(item.get("source_section"))
        title = clean_str(item.get("source_title"))
        if not title:
            title = sec if sec else "Peraturan Akademik FTMM"

        entry = {
            "id": str(current_id),
            "question": clean_str(item.get("question")),
            "answer": clean_str(item.get("answer")),
            "source_id": clean_str(item.get("source_id")),
            "source_title": title,
            "source_section": sec,
            "source_type": stype,
            "domain": "institutional_regulation",
        }
        master_entries.append(entry)
        current_id += 1

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(master_entries, f, indent=2, ensure_ascii=False)

    logger.info("✓ Successfully generated %d master ground truth entries -> %s", len(master_entries), OUTPUT_PATH)

    domain_counts = {}
    type_counts = {}
    for r in master_entries:
        d = r.get("domain", "unknown")
        domain_counts[d] = domain_counts.get(d, 0) + 1
        t = r.get("source_type", "unknown")
        type_counts[t] = type_counts.get(t, 0) + 1

    logger.info("Domain Breakdown:")
    for d, c in domain_counts.items():
        logger.info("  - %-30s : %4d items", d, c)
    logger.info("Source Type Breakdown:")
    for t, c in type_counts.items():
        logger.info("  - %-30s : %4d items", t, c)


if __name__ == "__main__":
    build_gt_master()
