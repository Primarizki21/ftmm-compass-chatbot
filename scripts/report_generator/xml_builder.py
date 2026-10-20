"""
XML Builder for OpenXML WordprocessingML (.docx).
Constructs 100% compliant OpenXML structures matching Format_Tugas_Laporan_NLP.docx:
- Page Size: US Letter (12240 x 15840 DXA)
- Margins: 1.0 inch symmetric (1440 DXA)
- Typography: Calibri 12pt (sz=24), Heading Calibri Light 14pt Bold (sz=28), Title 16pt (sz=32, center)
- Tables: TableGrid full-width 9351 DXA, cell padding internal, clear shading
"""

import os
import zipfile
import xml.etree.ElementTree as ET
from xml.sax.saxutils import escape

# Standard namespaces
NS_W = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
NS_R = "http://schemas.openxmlformats.org/officeDocument/2006/relationships"
NS_WP = "http://schemas.openxmlformats.org/drawingml/2006/wordprocessingDrawing"
NS_A = "http://schemas.openxmlformats.org/drawingml/2006/main"
NS_PIC = "http://schemas.openxmlformats.org/drawingml/2006/picture"
NS_W14 = "http://schemas.microsoft.com/office/word/2010/wordml"
NS_WP14 = "http://schemas.microsoft.com/office/word/2010/wordprocessingDrawing"


def xml_escape(text: str) -> str:
    """Escapes XML entities while preserving unicode characters."""
    if not text:
        return ""
    return escape(str(text), entities={'"': "&quot;", "'": "&apos;"})


def build_run(
    text: str,
    font: str = "Calibri",
    sz: int = 24,
    bold: bool = False,
    italic: bool = False,
    color: str | None = None,
) -> str:
    """Constructs a single <w:r> run element."""
    escaped_text = xml_escape(text)
    # Check if text contains leading/trailing spaces
    space_attr = ' xml:space="preserve"' if (text.startswith(" ") or text.endswith(" ") or "  " in text) else ""
    
    rPr_parts = []
    if font:
        rPr_parts.append(f'<w:rFonts w:ascii="{font}" w:hAnsi="{font}" w:cs="{font}"/>')
    if bold:
        rPr_parts.append("<w:b/><w:bCs/>")
    if italic:
        rPr_parts.append("<w:i/><w:iCs/>")
    if sz:
        rPr_parts.append(f'<w:sz w:val="{sz}"/><w:szCs w:val="{sz}"/>')
    if color:
        rPr_parts.append(f'<w:color w:val="{color}"/>')

    rPr_xml = f"<w:rPr>{''.join(rPr_parts)}</w:rPr>" if rPr_parts else ""
    return f"<w:r>{rPr_xml}<w:t{space_attr}>{escaped_text}</w:t></w:r>"


def build_paragraph(
    text_or_runs: str | list[str],
    style: str = "Normal",
    jc: str = "left",
    spacing_before: int = 0,
    spacing_after: int = 120,
    line_spacing: int = 276,  # 1.15 line pitch
    hanging_indent: int = 0,
    left_indent: int = 0,
    font: str = "Calibri",
    sz: int = 24,
    bold: bool = False,
    italic: bool = False,
) -> str:
    """Constructs a <w:p> paragraph element."""
    pPr_parts = []
    if style and style != "Normal":
        pPr_parts.append(f'<w:pStyle w:val="{style}"/>')
    if jc and jc != "left":
        pPr_parts.append(f'<w:jc w:val="{jc}"/>')
    
    # Indentation
    if hanging_indent > 0 or left_indent > 0:
        ind_attrs = []
        if left_indent > 0:
            ind_attrs.append(f'w:left="{left_indent}"')
        if hanging_indent > 0:
            ind_attrs.append(f'w:hanging="{hanging_indent}"')
        pPr_parts.append(f'<w:ind {" ".join(ind_attrs)}/>')

    # Spacing
    pPr_parts.append(
        f'<w:spacing w:before="{spacing_before}" w:after="{spacing_after}" '
        f'w:line="{line_spacing}" w:lineRule="auto"/>'
    )

    # Paragraph default run properties
    pPr_parts.append(
        f'<w:rPr><w:rFonts w:ascii="{font}" w:hAnsi="{font}" w:cs="{font}"/>'
        f'<w:sz w:val="{sz}"/><w:szCs w:val="{sz}"/></w:rPr>'
    )

    pPr_xml = f"<w:pPr>{''.join(pPr_parts)}</w:pPr>"

    if isinstance(text_or_runs, list):
        runs_xml = "".join(text_or_runs)
    elif isinstance(text_or_runs, str):
        runs_xml = build_run(text_or_runs, font=font, sz=sz, bold=bold, italic=italic)
    else:
        runs_xml = ""

    return f"<w:p>{pPr_xml}{runs_xml}</w:p>"


def build_heading(title: str, is_sub: bool = False) -> str:
    """Constructs a Chapter or Subsection heading in Calibri Light 14pt Bold (sz=28)."""
    style = "ListParagraph" if is_sub else "Normal"
    before = 240 if is_sub else 320
    after = 120
    return build_paragraph(
        text_or_runs=title,
        style=style,
        jc="left",
        spacing_before=before,
        spacing_after=after,
        line_spacing=276,
        font="Calibri Light",
        sz=28,
        bold=True,
    )


def build_title(title: str) -> str:
    """Constructs Document Main Title: 16pt (sz=32), center, bold."""
    return build_paragraph(
        text_or_runs=title,
        style="Normal",
        jc="center",
        spacing_before=0,
        spacing_after=180,
        line_spacing=280,
        font="Calibri",
        sz=32,
        bold=True,
    )


def build_author(author_text: str, affiliation_text: str | None = None) -> str:
    """Constructs Author and Affiliation lines (center, 12pt)."""
    p1 = build_paragraph(
        text_or_runs=author_text,
        style="Normal",
        jc="center",
        spacing_before=60,
        spacing_after=60,
        line_spacing=240,
        font="Calibri",
        sz=24,
        bold=False,
    )
    if affiliation_text:
        p2 = build_paragraph(
            text_or_runs=affiliation_text,
            style="Normal",
            jc="center",
            spacing_before=0,
            spacing_after=240,
            line_spacing=240,
            font="Calibri",
            sz=22,
            italic=True,
        )
        return f"{p1}{p2}"
    return p1


def build_caption(caption_text: str) -> str:
    """Constructs Table or Figure caption: 10pt (sz=20), center."""
    return build_paragraph(
        text_or_runs=caption_text,
        style="Normal",
        jc="center",
        spacing_before=140,
        spacing_after=120,
        line_spacing=240,
        font="Calibri",
        sz=20,
        bold=False,
    )


def build_table(
    headers: list[str],
    rows: list[list[str]],
    col_widths: list[int],
    table_width: int = 9351,
    header_fill: str = "F2F2F2",
) -> str:
    """
    Constructs an OpenXML Table with TableGrid style, full-width 9351 DXA,
    internal cell padding, clear shading, and column widths.
    """
    assert sum(col_widths) == table_width, f"Column widths sum ({sum(col_widths)}) != table_width ({table_width})"

    tblPr = (
        f'<w:tblPr>'
        f'<w:tblStyle w:val="TableGrid"/>'
        f'<w:tblW w:w="{table_width}" w:type="dxa"/>'
        f'<w:tblCellMar>'
        f'<w:top w:w="120" w:type="dxa"/>'
        f'<w:bottom w:w="120" w:type="dxa"/>'
        f'<w:left w:w="150" w:type="dxa"/>'
        f'<w:right w:w="150" w:type="dxa"/>'
        f'</w:tblCellMar>'
        f'<w:tblLook w:val="04A0" w:firstRow="1" w:lastRow="0" w:firstColumn="1" w:lastColumn="0" w:noHBand="0" w:noVBand="1"/>'
        f'</w:tblPr>'
    )

    gridCols = "".join([f'<w:gridCol w:w="{w}"/>' for w in col_widths])
    tblGrid = f'<w:tblGrid>{gridCols}</w:tblGrid>'

    # Header Row
    header_cells_xml = []
    for h_text, w in zip(headers, col_widths):
        tcPr = (
            f'<w:tcPr>'
            f'<w:tcW w:w="{w}" w:type="dxa"/>'
            f'<w:shd w:val="clear" w:color="auto" w:fill="{header_fill}"/>'
            f'</w:tcPr>'
        )
        tc_p = build_paragraph(
            text_or_runs=h_text,
            style="Normal",
            jc="center",
            spacing_before=40,
            spacing_after=40,
            line_spacing=240,
            font="Calibri",
            sz=20,
            bold=True,
        )
        header_cells_xml.append(f'<w:tc>{tcPr}{tc_p}</w:tc>')

    header_row_xml = f'<w:tr><w:trPr><w:tblHeader/></w:trPr>{"".join(header_cells_xml)}</w:tr>'

    # Body Rows
    body_rows_xml = []
    for r_idx, row in enumerate(rows):
        cells_xml = []
        for c_idx, (c_text, w) in enumerate(zip(row, col_widths)):
            tcPr = f'<w:tcPr><w:tcW w:w="{w}" w:type="dxa"/><w:shd w:val="clear" w:color="auto" w:fill="auto"/></w:tcPr>'
            # First column or numeric column centered, long text left-aligned
            is_num_col = (c_idx == 0) or (len(str(c_text).strip()) <= 6 and str(c_text).strip().replace('.', '').replace('%', '').isdigit())
            jc = "center" if is_num_col else "left"
            tc_p = build_paragraph(
                text_or_runs=str(c_text),
                style="Normal",
                jc=jc,
                spacing_before=40,
                spacing_after=40,
                line_spacing=240,
                font="Calibri",
                sz=20,
                bold=False,
            )
            cells_xml.append(f'<w:tc>{tcPr}{tc_p}</w:tc>')
        body_rows_xml.append(f'<w:tr>{"".join(cells_xml)}</w:tr>')

    return f'<w:tbl>{tblPr}{tblGrid}{header_row_xml}{"".join(body_rows_xml)}</w:tbl>'


def build_image_drawing(
    rel_id: str = "rId5",
    cx: int = 4400000,
    cy: int = 6559700,
    doc_pr_id: int = 1547877788,
    doc_pr_name: str = "Picture 1",
) -> str:
    """
    Constructs a WordprocessingML inline DrawingML element referencing an image relation.
    Dimensions cx and cy are in EMUs (1 inch = 914,400 EMUs).
    """
    return (
        f'<w:p>'
        f'<w:pPr><w:jc w:val="center"/><w:spacing w:before="180" w:after="80"/></w:pPr>'
        f'<w:r>'
        f'<w:drawing>'
        f'<wp:inline distT="0" distB="0" distL="0" distR="0">'
        f'<wp:extent cx="{cx}" cy="{cy}"/>'
        f'<wp:effectExtent l="0" t="0" r="0" b="0"/>'
        f'<wp:docPr id="{doc_pr_id}" name="{doc_pr_name}"/>'
        f'<wp:cNvGraphicFramePr><a:graphicFrameLocks xmlns:a="{NS_A}" noChangeAspect="1"/></wp:cNvGraphicFramePr>'
        f'<a:graphic xmlns:a="{NS_A}">'
        f'<a:graphicData uri="http://schemas.openxmlformats.org/drawingml/2006/picture">'
        f'<pic:pic xmlns:pic="{NS_PIC}">'
        f'<pic:nvPicPr>'
        f'<pic:cNvPr id="{doc_pr_id}" name="{doc_pr_name}"/>'
        f'<pic:cNvPicPr/>'
        f'</pic:nvPicPr>'
        f'<pic:blipFill rotWithShape="1">'
        f'<a:blip xmlns:r="{NS_R}" r:embed="{rel_id}"/>'
        f'<a:stretch><a:fillRect/></a:stretch>'
        f'</pic:blipFill>'
        f'<pic:spPr bwMode="auto">'
        f'<a:xfrm><a:off x="0" y="0"/><a:ext cx="{cx}" cy="{cy}"/></a:xfrm>'
        f'<a:prstGeom prst="rect"><a:avLst/></a:prstGeom>'
        f'</pic:spPr>'
        f'</pic:pic>'
        f'</a:graphicData>'
        f'</a:graphic>'
        f'</wp:inline>'
        f'</w:drawing>'
        f'</w:r>'
        f'</w:p>'
    )


def build_code_block(code_lines: list[str]) -> str:
    """Constructs a formatted pseudocode / algorithm block with shaded background and monospaced font."""
    p_elements = []
    for line in code_lines:
        ind = len(line) - len(line.lstrip(" "))
        left_indent = 360 + (ind * 140)
        p = build_paragraph(
            text_or_runs=line.strip(),
            style="Normal",
            jc="left",
            spacing_before=20,
            spacing_after=20,
            line_spacing=240,
            left_indent=left_indent,
            font="Consolas",
            sz=19,
            bold=False,
        )
        p_elements.append(p)
    return "".join(p_elements)


def build_document_xml(body_content_xml: str) -> str:
    """Wraps body elements in a valid WordprocessingML document root with US Letter page setup."""
    sectPr = (
        f'<w:sectPr>'
        f'<w:pgSz w:w="12240" w:h="15840"/>'
        f'<w:pgMar w:top="1440" w:right="1440" w:bottom="1440" w:left="1440" w:header="708" w:footer="708" w:gutter="0"/>'
        f'<w:cols w:space="708"/>'
        f'<w:docGrid w:linePitch="360"/>'
        f'</w:sectPr>'
    )
    return (
        f'<?xml version="1.0" encoding="UTF-8" standalone="yes"?>\n'
        f'<w:document xmlns:w="{NS_W}" xmlns:r="{NS_R}" xmlns:wp="{NS_WP}" '
        f'xmlns:a="{NS_A}" xmlns:pic="{NS_PIC}" xmlns:w14="{NS_W14}" xmlns:wp14="{NS_WP14}">\n'
        f'<w:body>\n'
        f'{body_content_xml}\n'
        f'{sectPr}\n'
        f'</w:body>\n'
        f'</w:document>'
    )


def package_docx(
    template_path: str,
    output_path: str,
    document_xml_content: str,
    image_path: str | None = None,
) -> None:
    """
    Assembles a valid .docx ZIP package by cloning template parts and
    replacing word/document.xml and word/media/image1.png.
    """
    with zipfile.ZipFile(template_path, "r") as z_in:
        with zipfile.ZipFile(output_path, "w", compression=zipfile.ZIP_DEFLATED) as z_out:
            for item in z_in.infolist():
                if item.filename == "word/document.xml":
                    z_out.writestr("word/document.xml", document_xml_content.encode("utf-8"))
                elif item.filename == "word/media/image1.png" and image_path and os.path.exists(image_path):
                    with open(image_path, "rb") as img_f:
                        z_out.writestr("word/media/image1.png", img_f.read())
                else:
                    z_out.writestr(item.filename, z_in.read(item.filename))
