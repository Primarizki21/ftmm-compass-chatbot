#!/usr/bin/env python3
"""
Sync Ground Truth & Knowledge Base from Google Spreadsheet SSOT.
Spreadsheet ID: 1jR0Hj3NBBvUv3NDfbU_-pPDIWB7EaDfJk5E7-7SoKSM
Exports each designated tab to its corresponding JSON in data/ground_truth/.
"""

import csv
import io
import json
import logging
import urllib.error
import urllib.request
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("sync_drive_gt")

SPREADSHEET_ID = "1jR0Hj3NBBvUv3NDfbU_-pPDIWB7EaDfJk5E7-7SoKSM"
OUTPUT_DIR = Path("data/ground_truth")

SHEET_CONFIGS = [
    {
        "gid": "532639148",
        "output_file": "google_drive_gt.json",
        "description": "Master Ground Truth QA (Kurikulum & Panduan)",
        "expected_min_rows": 200,
    },
    {
        "gid": "0",
        "output_file": "google_drive_sheet1.json",
        "description": "General Institutional QA (Single-hop & Multi-hop)",
        "expected_min_rows": 50,
    },
    {
        "gid": "1017899153",
        "output_file": "google_drive_sop.json",
        "description": "Alur Pengajuan Administrasi Akademik",
        "expected_min_rows": 8,
    },
    {
        "gid": "508469847",
        "output_file": "panduanakademik.json",
        "description": "Buku Panduan Pendidikan FTMM",
        "expected_min_rows": 30,
    },
    {
        "gid": "224932205",
        "output_file": "panduanpkl.json",
        "description": "Buku Panduan Praktik Kerja Lapangan",
        "expected_min_rows": 40,
    },
    {
        "gid": "1105412692",
        "output_file": "panduanskripsi.json",
        "description": "Buku Panduan Skripsi & Tugas Akhir FTMM",
        "expected_min_rows": 100,
    },
    {
        "gid": "1174137261",
        "output_file": "panduanyudisium.json",
        "description": "Buku Panduan Yudisium Kelulusan",
        "expected_min_rows": 20,
    },
    {
        "gid": "1377454251",
        "output_file": "gt_baru_courses.json",
        "description": "369 Mata Kuliah FTMM Ground Truth Silabus",
        "expected_min_rows": 369,
    },
    {
        "gid": "1428033810",
        "output_file": "gt_master.json",
        "description": "Master Ground Truth QA (369 MK + 80 Regulasi/SOP)",
        "expected_min_rows": 440,
    },
    {
        "gid": "1603934730",
        "output_file": "literatur.json",
        "description": "Kompilasi Literatur Akademik 5 Kolom Terstruktur",
        "expected_min_rows": 30,
    },
]


def fetch_csv_from_google_sheet(spreadsheet_id: str, gid: str) -> list[dict[str, str]]:
    url = f"https://docs.google.com/spreadsheets/d/{spreadsheet_id}/export?format=csv&gid={gid}"
    logger.info("Fetching CSV from %s (gid=%s)...", url, gid)
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"})
    with urllib.request.urlopen(req, timeout=30) as resp:
        if resp.status != 200:
            raise RuntimeError(f"Failed to fetch CSV: HTTP {resp.status}")
        raw_bytes = resp.read()
        text = raw_bytes.decode("utf-8", errors="replace")

    reader = csv.DictReader(io.StringIO(text))
    rows: list[dict[str, str]] = []
    for r in reader:
        # Strip all whitespace from keys and values
        clean_row = {k.strip(): v.strip() for k, v in r.items() if k is not None}
        if any(clean_row.values()):
            rows.append(clean_row)
    return rows


def sync_all_sheets(spreadsheet_id: str = SPREADSHEET_ID, output_dir: Path = OUTPUT_DIR) -> dict[str, int]:
    output_dir.mkdir(parents=True, exist_ok=True)
    summary: dict[str, int] = {}

    for cfg in SHEET_CONFIGS:
        gid = cfg["gid"]
        out_name = cfg["output_file"]
        desc = cfg["description"]
        min_rows = cfg["expected_min_rows"]
        out_path = output_dir / out_name

        try:
            rows = fetch_csv_from_google_sheet(spreadsheet_id, gid)
            if len(rows) < min_rows:
                logger.warning(
                    "Tab '%s' (%s) returned %d rows, expected at least %d",
                    desc,
                    out_name,
                    len(rows),
                    min_rows,
                )
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(rows, f, indent=2, ensure_ascii=False)
            logger.info("✓ Synced %d rows -> %s (%s)", len(rows), out_path, desc)
            summary[out_name] = len(rows)
        except Exception as exc:
            logger.error("Failed to sync %s (gid=%s): %s", out_name, gid, exc)
            raise

    return summary


if __name__ == "__main__":
    logger.info("Starting synchronization with Google Spreadsheet SSOT...")
    results = sync_all_sheets()
    logger.info("=" * 60)
    logger.info("Synchronization Summary:")
    for file_name, count in results.items():
        logger.info("  - %-30s : %4d items", file_name, count)
    logger.info("=" * 60)
