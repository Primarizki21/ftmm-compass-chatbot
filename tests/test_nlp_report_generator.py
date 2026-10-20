"""
Tests for NLP Final Project Report Generator (FTMM COMPASS).
Verifies XML structure fidelity, typography, page setup, table geometries,
content completeness, and APA references compliance.
"""

import os
import zipfile
import xml.etree.ElementTree as ET
import pytest

from scripts.report_generator.data_loader import (
    load_comparison_metrics,
    load_material_stats,
    load_error_analysis_cases,
    load_apa_references,
)
from scripts.report_generator.xml_builder import (
    build_paragraph,
    build_heading,
    build_title,
    build_table,
    build_image_drawing,
    build_document_xml,
)
from scripts.report_generator.generator import NLPReportGenerator

NS = {
    "w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main",
    "wp": "http://schemas.openxmlformats.org/drawingml/2006/wordprocessingDrawing",
}


def test_data_loader_metrics_and_sources():
    metrics = load_comparison_metrics()
    assert len(metrics) >= 3
    arch_names = [m["architecture"] for m in metrics]
    assert any("CAG" in a for a in arch_names)
    assert any("RAG" in a for a in arch_names)
    assert any("Hybrid" in a for a in arch_names)

    stats = load_material_stats()
    assert stats["gt_count"] == 449
    assert stats["courses_count"] == 369

    cases = load_error_analysis_cases()
    assert len(cases) == 5
    for c in cases:
        assert "aspek" in c and "komentar" in c and "label_aktual" in c and "prediksi_model" in c and "analisis_kesalahan" in c

    refs = load_apa_references(min_refs=30)
    assert len(refs) >= 30
    # Assert alphabetical ordering
    first_words = [r.split()[0].lower().replace("(", "") for r in refs]
    assert first_words == sorted(first_words)


def test_xml_builder_primitives():
    # Paragraph
    wrapper = f'<root xmlns:w="{NS["w"]}" xmlns:wp="{NS["wp"]}">'
    p_xml = build_paragraph("Test text", jc="center", sz=24, bold=True)
    root = ET.fromstring(f"{wrapper}{p_xml}</root>")
    assert root.find(".//w:p", NS) is not None

    # Table
    headers = ["No", "Method", "P", "R", "F1"]
    rows = [["1", "Method A", "0.9", "0.8", "0.85"]]
    widths = [619, 6322, 709, 709, 992]  # Sum = 9351
    t_xml = build_table(headers, rows, widths, table_width=9351)
    t_root = ET.fromstring(f"{wrapper}{t_xml}</root>")
    assert t_root.find(".//w:tbl", NS) is not None
    tblW = t_root.find(".//w:tblW", NS)
    assert tblW.attrib[f"{{{NS['w']}}}w"] == "9351"

    # Image
    img_xml = build_image_drawing(rel_id="rId5", cx=4400000, cy=6559700)
    img_root = ET.fromstring(f"{wrapper}{img_xml}</root>")
    assert img_root.find(".//wp:inline", NS) is not None


def test_nlp_report_generator_end_to_end(tmp_path):
    output_docx = str(tmp_path / "test_report.docx")
    generator = NLPReportGenerator()
    generator.generate(output_path=output_docx)

    assert os.path.exists(output_docx)
    assert os.path.getsize(output_docx) > 50000  # Should be substantial document with embedded image

    with zipfile.ZipFile(output_docx) as z:
        assert "word/document.xml" in z.namelist()
        assert "word/media/image1.png" in z.namelist()
        doc_xml = z.read("word/document.xml").decode("utf-8")

    root = ET.fromstring(doc_xml)

    # 1. Page Size & Margins (US Letter & 1.0 inch)
    sectPr = root.find(".//w:sectPr", NS)
    assert sectPr is not None
    pgSz = sectPr.find("w:pgSz", NS)
    assert pgSz.attrib[f"{{{NS['w']}}}w"] == "12240"
    assert pgSz.attrib[f"{{{NS['w']}}}h"] == "15840"

    pgMar = sectPr.find("w:pgMar", NS)
    assert pgMar.attrib[f"{{{NS['w']}}}top"] == "1440"
    assert pgMar.attrib[f"{{{NS['w']}}}bottom"] == "1440"
    assert pgMar.attrib[f"{{{NS['w']}}}left"] == "1440"
    assert pgMar.attrib[f"{{{NS['w']}}}right"] == "1440"

    # 2. Required Headings
    headings = []
    for p in root.findall(".//w:p", NS):
        sz = p.find(".//w:sz", NS)
        b = p.find(".//w:b", NS)
        if sz is not None and sz.attrib.get(f"{{{NS['w']}}}val") == "28" and b is not None:
            t = "".join([elem.text for elem in p.findall(".//w:t", NS) if elem.text])
            headings.append(t.strip())

    expected_headings = [
        "Introduction",
        "Related Works",
        "Methods",
        "Material",
        "Methodology",
        "Result",
        "Discussion",
        "Conclusion",
        "References",
    ]
    for exp in expected_headings:
        assert exp in headings, f"Heading '{exp}' not found in {headings}"

    # 3. Tables (Table 1 and Table 2)
    tables = root.findall(".//w:tbl", NS)
    assert len(tables) >= 2
    for tbl in tables:
        tblW = tbl.find(".//w:tblW", NS)
        assert tblW.attrib[f"{{{NS['w']}}}w"] == "9351"
        cols = [int(c.attrib[f"{{{NS['w']}}}w"]) for c in tbl.findall(".//w:gridCol", NS)]
        assert sum(cols) == 9351

    # 4. Embedded Drawing
    drawings = root.findall(".//w:drawing", NS)
    assert len(drawings) >= 1

    # 5. References with hanging indent
    refs = [
        p for p in root.findall(".//w:p", NS)
        if p.find(".//w:ind", NS) is not None
        and p.find(".//w:ind", NS).attrib.get(f"{{{NS['w']}}}hanging") == "720"
    ]
    assert len(refs) >= 30
