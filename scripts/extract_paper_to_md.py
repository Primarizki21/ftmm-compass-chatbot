#!/usr/bin/env python3
"""Script to extract and structure PDF academic papers into agent-readable Markdown.

Supports digital extraction via PyMuPDF with automatic fallback to Tesseract OCR
for scanned pages.
"""

from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
from pathlib import Path
import pymupdf


def is_scanned_page(text: str, char_threshold: int = 50) -> bool:
    return len(text.strip()) < char_threshold


def ocr_page_with_tesseract(page: pymupdf.Page) -> str:
    pix = page.get_pixmap(dpi=150)
    img_path = f"/tmp/ocr_tmp_{os.getpid()}.png"
    pix.save(img_path)
    try:
        res = subprocess.run(
            ["tesseract", img_path, "stdout", "--oem", "1", "-l", "eng+ind"],
            capture_output=True,
            text=True,
            check=True,
        )
        return res.stdout
    except Exception as e:
        sys.stderr.write(f"Tesseract OCR fallback error: {e}\n")
        return ""
    finally:
        if os.path.exists(img_path):
            os.remove(img_path)


def clean_text_blocks(doc: pymupdf.Document) -> tuple[str, list[dict]]:
    full_text_parts = []
    page_summaries = []

    for page_num, page in enumerate(doc, start=1):
        text = page.get_text("text")
        if is_scanned_page(text):
            sys.stderr.write(f"Page {page_num} appears scanned, invoking OCR...\n")
            text = ocr_page_with_tesseract(page)

        cleaned_text = re.sub(r"\n{3,}", "\n\n", text).strip()
        full_text_parts.append(f"<!-- Page {page_num} -->\n{cleaned_text}")
        page_summaries.append({"page": page_num, "char_count": len(cleaned_text)})

    return "\n\n---\n\n".join(full_text_parts), page_summaries


def extract_metadata_and_structure(doc: pymupdf.Document, raw_text: str) -> dict:
    meta = doc.metadata or {}
    first_pages_text = "\n".join([doc[i].get_text("text") for i in range(min(2, len(doc)))])

    title = meta.get("title") or ""
    if not title or len(title) < 5 or title.endswith(".pdf"):
        first_lines = [line.strip() for line in first_pages_text.split("\n") if line.strip() and not line.lower().startswith("applied artificial")]
        title = first_lines[0] if first_lines else "Untitled Document"

    author = meta.get("author") or ""
    if not author or author.lower() in ["unknown", "user", "admin"]:
        author_match = re.search(r"\n([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+(?:,\s*[A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)*(?:\s*(?:&|and)\s*[A-Z][a-z]+(?:\s+[A-Z][a-z]+)+))\n", first_pages_text)
        author = author_match.group(1).strip() if author_match else "Unknown"

    abstract_match = re.search(
        r"(?i)\babstract\b[:\s\n]*(.*?)(?=\n\s*(?:(?:1[\.\s]+)?introduction|keywords|index terms|article history|\Z))",
        first_pages_text,
        re.DOTALL,
    )
    abstract = abstract_match.group(1).strip() if abstract_match else "Abstract not explicitly isolated."
    abstract = re.sub(r"\s+", " ", abstract)

    return {
        "title": title,
        "author": author,
        "total_pages": len(doc),
        "abstract": abstract,
    }


def convert_pdf_to_markdown(pdf_path: Path, output_path: Path | None = None) -> Path:
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    if output_path is None:
        output_path = pdf_path.with_suffix(".md")

    doc = pymupdf.open(str(pdf_path))
    raw_content, pages_meta = clean_text_blocks(doc)
    info = extract_metadata_and_structure(doc, raw_content)
    doc.close()

    md_lines = [
        "---",
        f"title: \"{info['title']}\"",
        f"source_file: \"{pdf_path.name}\"",
        f"pages: {info['total_pages']}",
        f"author: \"{info['author']}\"",
        "extraction_engine: \"PyMuPDF (fitz) + Tesseract OCR Fallback\"",
        "---",
        "",
        f"# {info['title']}",
        "",
        "## Executive Metadata & Abstract",
        f"- **File Source**: `{pdf_path.name}`",
        f"- **Total Pages**: {info['total_pages']}",
        f"- **Author(s)**: {info['author']}",
        "",
        "### Abstract",
        info["abstract"],
        "",
        "## Structural Content (Page-by-Page Extraction)",
        "",
        raw_content,
        "",
    ]

    output_path.write_text("\n".join(md_lines), encoding="utf-8")
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract academic PDF to Markdown")
    parser.add_argument("pdf_files", nargs="*", help="PDF file(s) to convert")
    parser.add_argument("--all-in-papers", action="store_true", help="Process all PDFs in papers/ directory")
    args = parser.parse_args()

    files_to_process: list[Path] = []
    if args.all_in_papers:
        papers_dir = Path("papers")
        if papers_dir.exists():
            files_to_process.extend(sorted(papers_dir.glob("*.pdf")))
            files_to_process.extend(sorted(papers_dir.glob("*.PDF")))

    for f in args.pdf_files:
        p = Path(f)
        if p.exists() and p not in files_to_process:
            files_to_process.append(p)

    if not files_to_process:
        print("No PDF files specified. Use --all-in-papers or provide file paths.")
        sys.exit(1)

    print(f"Found {len(files_to_process)} PDF(s) to process.")
    for pdf_path in files_to_process:
        print(f"Extracting: {pdf_path} ...")
        out_md = convert_pdf_to_markdown(pdf_path)
        print(f"  -> Generated: {out_md} ({out_md.stat().st_size:,} bytes)")


if __name__ == "__main__":
    main()
