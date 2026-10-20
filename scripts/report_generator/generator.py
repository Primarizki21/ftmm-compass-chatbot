"""
Main Report Generator coordinator for NLP Final Project Report (FTMM COMPASS).
Orchestrates data intake from SSOT sources, document structure synthesis,
and OpenXML .docx packaging.
"""

import os
from typing import Any

from scripts.report_generator.data_loader import (
    load_comparison_metrics,
    load_material_stats,
    load_error_analysis_cases,
    load_apa_references,
)
from scripts.report_generator.content_builder import generate_full_report_xml
from scripts.report_generator.xml_builder import (
    build_document_xml,
    package_docx,
)


class NLPReportGenerator:
    """Coordinates generation of the final NLP course report in OpenXML format."""

    def __init__(
        self,
        template_path: str = "Format_Tugas_Laporan_NLP.docx",
        flowchart_image_path: str = "docs/architecture/ftmm_compass_pipeline_flowchart.png",
    ):
        self.template_path = template_path
        self.flowchart_image_path = flowchart_image_path

    def generate(self, output_path: str = "Laporan_Tugas_Besar_NLP_FTMM_COMPASS.docx") -> str:
        """Executes full report generation and writes output .docx file."""
        print(f"[1/4] Loading SSOT data and metrics...")
        metrics = load_comparison_metrics()
        material_stats = load_material_stats()
        error_cases = load_error_analysis_cases()
        references = load_apa_references(min_refs=30)

        print(f"      - Evaluated architectures loaded: {len(metrics)}")
        print(f"      - Material ground truth entries: {material_stats.get('gt_count', 0)}")
        print(f"      - Courses catalog entries: {material_stats.get('courses_count', 0)}")
        print(f"      - Qualitative error cases: {len(error_cases)}")
        print(f"      - APA English references: {len(references)}")

        assert len(references) >= 30, f"References count ({len(references)}) is less than required 30!"

        print(f"[2/4] Synthesizing Scholarly Content & Tables XML...")
        body_xml = generate_full_report_xml(
            metrics=metrics,
            material_stats=material_stats,
            error_cases=error_cases,
            references=references,
        )

        print(f"[3/4] Wrapping in OpenXML Document Root with US Letter Page Setup...")
        full_document_xml = build_document_xml(body_xml)

        print(f"[4/4] Packaging .docx archive from template '{self.template_path}'...")
        if not os.path.exists(self.template_path):
            raise FileNotFoundError(f"Template '{self.template_path}' not found!")

        package_docx(
            template_path=self.template_path,
            output_path=output_path,
            document_xml_content=full_document_xml,
            image_path=self.flowchart_image_path,
        )

        print(f"[SUCCESS] Report generated successfully at: {output_path}")
        return output_path
