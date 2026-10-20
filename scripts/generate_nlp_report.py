#!/usr/bin/env python3
"""
Entry point for automated NLP Final Project Report generation for FTMM COMPASS.
Usage:
    uv run python scripts/generate_nlp_report.py
    uv run python scripts/generate_nlp_report.py --output Laporan_Tugas_Besar_NLP_FTMM_COMPASS.docx
"""

import argparse
import sys
import os

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from scripts.report_generator.generator import NLPReportGenerator


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Automated OpenXML .docx report generator for NLP Final Project (FTMM COMPASS)."
    )
    parser.add_argument(
        "--template",
        default="Format_Tugas_Laporan_NLP.docx",
        help="Path to input .docx template (default: Format_Tugas_Laporan_NLP.docx)",
    )
    parser.add_argument(
        "--output",
        default="Laporan_Tugas_Besar_NLP_FTMM_COMPASS.docx",
        help="Path to output .docx file (default: Laporan_Tugas_Besar_NLP_FTMM_COMPASS.docx)",
    )
    parser.add_argument(
        "--flowchart",
        default="docs/architecture/ftmm_compass_pipeline_flowchart.png",
        help="Path to pipeline flowchart image (default: docs/architecture/ftmm_compass_pipeline_flowchart.png)",
    )
    args = parser.parse_args()

    generator = NLPReportGenerator(
        template_path=args.template,
        flowchart_image_path=args.flowchart,
    )
    generator.generate(output_path=args.output)
    return 0


if __name__ == "__main__":
    sys.exit(main())
