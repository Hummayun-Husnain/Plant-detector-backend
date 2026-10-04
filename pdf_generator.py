"""PDF report generator for the AI agronomic reporting pipeline."""

import os
import re
import shutil
import tempfile
from datetime import datetime, timezone
from typing import List
from xml.sax.saxutils import escape

from PIL import Image as PILImage
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.pdfgen import canvas as rl_canvas
from reportlab.platypus import (
    Image as ReportImage,
    KeepTogether,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

# --------------------------------------------------------------------------- #
# Theme
# --------------------------------------------------------------------------- #

BRAND = "Plant Detector"
REPORT_TITLE = "Agronomic Assessment Report"

C_PRIMARY = colors.HexColor("#1B5E3A")
C_PRIMARY_DARK = colors.HexColor("#0F3D26")
C_ACCENT = colors.HexColor("#3E8E5E")
C_ACCENT_LIGHT = colors.HexColor("#A7F3D0")
C_BG_SOFT = colors.HexColor("#F1F8F4")
C_TEXT = colors.HexColor("#1F2937")
C_MUTED = colors.HexColor("#6B7280")
C_BORDER = colors.HexColor("#E5E7EB")
C_ROW_ALT = colors.HexColor("#F9FAFB")
C_WHITE = colors.white

PAGE_W, PAGE_H = A4
MARGIN_X = 0.7 * inch
MARGIN_TOP = 1.05 * inch
MARGIN_BOTTOM = 0.85 * inch
CONTENT_W = PAGE_W - 2 * MARGIN_X


# --------------------------------------------------------------------------- #
# Styles
# --------------------------------------------------------------------------- #

def _build_styles() -> dict:
    base = getSampleStyleSheet()
    s = {}
    s["title"] = ParagraphStyle(
        "RTitle", parent=base["Title"], fontName="Helvetica-Bold", fontSize=24,
        leading=28, textColor=C_PRIMARY_DARK, alignment=TA_LEFT, spaceAfter=2,
    )
    s["subtitle"] = ParagraphStyle(
        "RSubtitle", parent=base["Normal"], fontName="Helvetica", fontSize=10.5,
        leading=14, textColor=C_MUTED, alignment=TA_LEFT,
    )
    s["h1"] = ParagraphStyle(
        "RH1", parent=base["Heading1"], fontName="Helvetica-Bold", fontSize=14,
        leading=18, textColor=C_PRIMARY_DARK, spaceBefore=0, spaceAfter=0,
    )
    s["h2"] = ParagraphStyle(
        "RH2", parent=base["Heading2"], fontName="Helvetica-Bold", fontSize=11.5,
        leading=15, textColor=C_PRIMARY, spaceBefore=8, spaceAfter=4,
    )
    s["body"] = ParagraphStyle(
        "RBody", parent=base["BodyText"], fontName="Helvetica", fontSize=10,
        leading=14.5, textColor=C_TEXT, alignment=TA_LEFT, spaceAfter=6,
    )
    s["bullet"] = ParagraphStyle("RBullet", parent=s["body"], spaceAfter=2, leading=14)
    s["small"] = ParagraphStyle(
        "RSmall", parent=s["body"], fontSize=8.5, leading=11.5, textColor=C_MUTED, spaceAfter=0,
    )
    s["caption"] = ParagraphStyle(
        "RCaption", parent=s["small"], alignment=TA_CENTER, fontSize=8, leading=10,
    )
    s["kpi_label"] = ParagraphStyle(
        "RKpiLabel", parent=s["body"], fontName="Helvetica-Bold", fontSize=7.5,
        leading=9, textColor=colors.HexColor("#D1FAE5"), spaceAfter=0,
    )
    s["kpi_value"] = ParagraphStyle(
        "RKpiValue", parent=s["body"], fontName="Helvetica-Bold", fontSize=26,
        leading=30, textColor=C_WHITE, spaceAfter=0,
    )
    s["meta_label"] = ParagraphStyle(
        "RMetaLabel", parent=s["body"], fontName="Helvetica-Bold", fontSize=7.5,
        leading=9, textColor=C_MUTED, spaceAfter=0,
    )
    s["meta_value"] = ParagraphStyle(
        "RMetaValue", parent=s["body"], fontSize=10, leading=13, textColor=C_TEXT, spaceAfter=0,
    )
    s["num"] = ParagraphStyle(
        "RNum", parent=s["body"], fontName="Helvetica-Bold", fontSize=8.5,
        leading=10, textColor=C_WHITE, alignment=TA_CENTER, spaceAfter=0,
    )
    s["table_cell"] = ParagraphStyle(
        "RTableCell", parent=s["body"], fontSize=9, leading=12.5, spaceAfter=0,
    )
    s["table_cell_muted"] = ParagraphStyle(
        "RTableCellMuted", parent=s["table_cell"], textColor=C_MUTED,
    )
    return s


# --------------------------------------------------------------------------- #
# Text helpers
# --------------------------------------------------------------------------- #

_BOLD_RE = re.compile(r"\*\*(.+?)\*\*")
_BULLET_RE = re.compile(r"^\s*(?:[-*\u2022]|\d+[.)])\s+(.*)$")


def _inline(text: str) -> str:
    """Escape XML and convert **bold** markdown the model may emit."""
    return _BOLD_RE.sub(r"<b>\1</b>", escape(str(text or "")))


def _text_block(text: str, styles: dict) -> list:
    """Render a free-text field as paragraphs, honouring blank-line breaks and
    simple '-' / '1.' bullet lines."""
    text = str(text or "").strip()
    if not text:
        return [Paragraph("<i>No information provided.</i>", styles["body"])]

    flow: list = []
    para_buf: List[str] = []
    bullets: List[str] = []

    def flush_para():
        if para_buf:
            flow.append(Paragraph(_inline(" ".join(para_buf)), styles["body"]))
            para_buf.clear()

    def flush_bullets():
        if bullets:
            # Plain paragraphs, not ListFlowable: a long bullet list stays
            # splittable across pages and cannot raise a layout error.
            for b in bullets:
                flow.append(Paragraph(
                    f"<font color='#3E8E5E'>&bull;</font>&nbsp;&nbsp;{_inline(b)}",
                    styles["bullet"],
                ))
            flow.append(Spacer(1, 4))
            bullets.clear()

    for raw in text.splitlines():
        line = raw.strip()
        if not line:
            flush_para()
            flush_bullets()
            continue
        m = _BULLET_RE.match(line)
        if m:
            flush_para()
            bullets.append(m.group(1))
        else:
            flush_bullets()
            para_buf.append(line)

    flush_para()
    flush_bullets()
    return flow


# --------------------------------------------------------------------------- #
# Page decorations: header band, footer, "Page X of Y"
# --------------------------------------------------------------------------- #

class _NumberedCanvas(rl_canvas.Canvas):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._saved_page_states = []

    def showPage(self):
        self._saved_page_states.append(dict(self.__dict__))
        self._startPage()

    def save(self):
        total = len(self._saved_page_states)
        for state in self._saved_page_states:
            self.__dict__.update(state)
            self.setFont("Helvetica", 8)
            self.setFillColor(C_MUTED)
            self.drawRightString(
                PAGE_W - MARGIN_X, MARGIN_BOTTOM - 0.38 * inch,
                f"Page {self._pageNumber} of {total}",
            )
            super().showPage()
        super().save()


def _make_page_decorator(job_id: str, generated_at: str):
    def _decorate(c: rl_canvas.Canvas, doc):
        c.saveState()
        band_h = 0.62 * inch

        c.setFillColor(C_PRIMARY_DARK)
        c.rect(0, PAGE_H - band_h, PAGE_W, band_h, stroke=0, fill=1)
        c.setFillColor(C_ACCENT)
        c.rect(0, PAGE_H - band_h - 3, PAGE_W, 3, stroke=0, fill=1)

        c.setFillColor(C_WHITE)
        c.setFont("Helvetica-Bold", 11)
        c.drawString(MARGIN_X, PAGE_H - band_h + 0.26 * inch, BRAND.upper())
        c.setFont("Helvetica", 8.5)
        c.setFillColor(C_ACCENT_LIGHT)
        c.drawString(MARGIN_X, PAGE_H - band_h + 0.11 * inch, REPORT_TITLE)

        # Keep the header reference short so a full UUID cannot run into the brand.
        ref = job_id if len(str(job_id)) <= 12 else f"{str(job_id)[:12]}..."
        c.setFillColor(C_WHITE)
        c.setFont("Helvetica", 8.5)
        c.drawRightString(PAGE_W - MARGIN_X, PAGE_H - band_h + 0.26 * inch, f"Ref {ref}")
        c.setFillColor(C_ACCENT_LIGHT)
        c.drawRightString(PAGE_W - MARGIN_X, PAGE_H - band_h + 0.11 * inch, generated_at)

        y = MARGIN_BOTTOM - 0.25 * inch
        c.setStrokeColor(C_BORDER)
        c.setLineWidth(0.6)
        c.line(MARGIN_X, y, PAGE_W - MARGIN_X, y)
        c.setFont("Helvetica", 8)
        c.setFillColor(C_MUTED)
        c.drawString(
            MARGIN_X, y - 0.13 * inch,
            f"{BRAND} \u00b7 AI-assisted agronomic report \u00b7 For advisory use only",
        )
        c.restoreState()

    return _decorate


# --------------------------------------------------------------------------- #
# Building blocks
# --------------------------------------------------------------------------- #

def _section_heading(number: int, text: str, styles: dict) -> Table:
    num_box = Table(
        [[Paragraph(f"{number:02d}", styles["num"])]],
        colWidths=[0.34 * inch], rowHeights=[0.26 * inch],
    )
    num_box.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), C_PRIMARY),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("LEFTPADDING", (0, 0), (-1, -1), 0),
        ("RIGHTPADDING", (0, 0), (-1, -1), 0),
        ("TOPPADDING", (0, 0), (-1, -1), 0),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 0),
    ]))
    t = Table(
        [[num_box, Paragraph(text, styles["h1"])]],
        colWidths=[0.45 * inch, CONTENT_W - 0.45 * inch],
    )
    t.setStyle(TableStyle([
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("LEFTPADDING", (0, 0), (-1, -1), 0),
        ("RIGHTPADDING", (0, 0), (-1, -1), 0),
        ("TOPPADDING", (0, 0), (-1, -1), 14),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
        ("LINEBELOW", (0, 0), (-1, -1), 0.8, C_ACCENT),
    ]))
    return t


def _keyframe_tile(path: str, caption: str, cell_w: float, styles: dict) -> Table:
    """One framed keyframe. The image is fitted inside a fixed window so every
    tile in the row is the same height, whether the frame is landscape or portrait."""
    frame_h = 1.42 * inch
    pad = 5
    max_w = cell_w - 2 * pad
    with PILImage.open(path) as im:
        w, h = im.size
    aspect = (h / w) if w else 0.75
    iw, ih = max_w, max_w * aspect
    if ih > frame_h:
        ih = frame_h
        iw = frame_h / aspect if aspect else max_w

    tile = Table(
        [[ReportImage(path, width=iw, height=ih)],
         [Paragraph(caption, styles["caption"])]],
        colWidths=[cell_w],
        rowHeights=[frame_h + 2 * pad, 16],
    )
    tile.setStyle(TableStyle([
        ("ALIGN", (0, 0), (-1, -1), "CENTER"),
        ("VALIGN", (0, 0), (0, 0), "MIDDLE"),
        ("VALIGN", (0, 1), (0, 1), "MIDDLE"),
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#F3F4F6")),
        ("BACKGROUND", (0, 1), (-1, 1), C_BG_SOFT),
        ("BOX", (0, 0), (-1, -1), 0.6, C_BORDER),
        ("LINEABOVE", (0, 1), (-1, 1), 0.4, C_BORDER),
        ("LEFTPADDING", (0, 0), (-1, -1), 0),
        ("RIGHTPADDING", (0, 0), (-1, -1), 0),
        ("TOPPADDING", (0, 0), (-1, -1), 0),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 0),
    ]))
    return tile


def _keyframe_gallery(keyframe_paths: List[str], styles: dict) -> list:
    if not keyframe_paths:
        return [Paragraph("<i>No keyframes were available for this video.</i>", styles["body"])]

    cols = 3 if len(keyframe_paths) >= 3 else len(keyframe_paths)
    gap = 0.12 * inch
    # Gap is its own column, so each tile is exactly cell_w wide and cannot
    # overflow the page the way padding-based gutters did.
    cell_w = (CONTENT_W - gap * (cols - 1)) / cols

    tiles = [
        _keyframe_tile(p, f"Frame {i + 1} of {len(keyframe_paths)}", cell_w, styles)
        for i, p in enumerate(keyframe_paths)
    ]

    col_widths = []
    for c in range(cols):
        col_widths.append(cell_w)
        if c != cols - 1:
            col_widths.append(gap)

    rows = []
    for start in range(0, len(tiles), cols):
        chunk = tiles[start:start + cols]
        row = []
        for c, tile in enumerate(chunk):
            if c:
                row.append("")
            row.append(tile)
        while len(row) < len(col_widths):
            row.append("")
        rows.append(row)

    grid = Table(rows, colWidths=col_widths, hAlign="LEFT")
    grid.setStyle(TableStyle([
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING", (0, 0), (-1, -1), 0),
        ("RIGHTPADDING", (0, 0), (-1, -1), 0),
        ("TOPPADDING", (0, 0), (-1, -1), 0),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
    ]))
    return [grid]


def _summary_band(plant_count: int, meta_rows: List[tuple], styles: dict) -> Table:
    """Plant-count panel and metadata, equal height, as one table.

    The count cell is spanned down every metadata row so the green panel
    matches the metadata card instead of sitting shorter beside it.
    """
    gap = 0.14 * inch
    left_w = 2.15 * inch
    label_w = 1.45 * inch
    value_w = CONTENT_W - left_w - gap - label_w

    count = Table(
        [[Paragraph("TOTAL PLANT COUNT", styles["kpi_label"])],
         [Paragraph(f"{int(plant_count):,}", styles["kpi_value"])]],
        colWidths=[left_w - 24],
    )
    count.setStyle(TableStyle([
        ("LEFTPADDING", (0, 0), (-1, -1), 0),
        ("RIGHTPADDING", (0, 0), (-1, -1), 0),
        ("TOPPADDING", (0, 0), (-1, -1), 0),
        ("BOTTOMPADDING", (0, 0), (0, 0), 2),
        ("BOTTOMPADDING", (0, 1), (0, 1), 0),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
    ]))

    data = []
    for i, (label, value) in enumerate(meta_rows):
        data.append([
            count if i == 0 else "",
            "",
            Paragraph(escape(label).upper(), styles["meta_label"]),
            Paragraph(escape(str(value)), styles["meta_value"]),
        ])

    # Column 1 is an empty gutter so the green panel does not bleed into the card.
    band = Table(data, colWidths=[left_w, gap, label_w, value_w], hAlign="LEFT")
    style = [
        ("SPAN", (0, 0), (0, -1)),
        ("BACKGROUND", (0, 0), (0, -1), C_PRIMARY),
        ("BACKGROUND", (2, 0), (-1, -1), C_BG_SOFT),
        ("BOX", (2, 0), (-1, -1), 0.6, C_BORDER),
        ("VALIGN", (0, 0), (0, -1), "MIDDLE"),
        ("VALIGN", (2, 0), (-1, -1), "MIDDLE"),
        ("LEFTPADDING", (0, 0), (0, -1), 12),
        ("RIGHTPADDING", (0, 0), (0, -1), 8),
        ("LEFTPADDING", (1, 0), (1, -1), 0),
        ("RIGHTPADDING", (1, 0), (1, -1), 0),
        ("LEFTPADDING", (2, 0), (2, -1), 10),
        ("RIGHTPADDING", (2, 0), (2, -1), 4),
        ("LEFTPADDING", (3, 0), (3, -1), 4),
        ("RIGHTPADDING", (3, 0), (3, -1), 10),
        ("TOPPADDING", (2, 0), (-1, -1), 7),
        ("BOTTOMPADDING", (2, 0), (-1, -1), 7),
        ("TOPPADDING", (0, 0), (0, -1), 8),
        ("BOTTOMPADDING", (0, 0), (0, -1), 8),
    ]
    last = len(data) - 1
    for r in range(last):
        style.append(("LINEBELOW", (2, r), (-1, r), 0.4, C_BORDER))
    band.setStyle(TableStyle(style))
    return band


# --------------------------------------------------------------------------- #
# Main entry point
# --------------------------------------------------------------------------- #

def _section_body(flowables: list) -> Table:
    """Put a section's paragraphs in one block with a left rule.

    One row per paragraph, so a long section can split between paragraphs
    instead of overflowing the page.
    """
    rows = [[item] for item in flowables] or [[""]]
    block = Table(rows, colWidths=[CONTENT_W])
    block.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), C_BG_SOFT),
        ("LINEBEFORE", (0, 0), (0, -1), 3, C_PRIMARY),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING", (0, 0), (-1, -1), 12),
        ("RIGHTPADDING", (0, 0), (-1, -1), 10),
        ("TOPPADDING", (0, 0), (0, 0), 8),
        ("BOTTOMPADDING", (0, -1), (-1, -1), 4),
        ("TOPPADDING", (0, 1), (-1, -1), 0),
        ("BOTTOMPADDING", (0, 0), (-1, -2), 0),
    ]))
    return block


def create_pdf(
    output_path: str,
    job_id: str,
    plant_count: int,
    insights: dict,
    keyframes: List[PILImage.Image],
) -> str:
    # English month names, not %b: a non-English server locale would emit
    # characters Helvetica cannot draw and the PDF build would fail.
    now = datetime.now(timezone.utc)
    month = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
             "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"][now.month - 1]
    generated_at = f"{now.day:02d} {month} {now.year}, {now:%H:%M} UTC"
    styles = _build_styles()
    insights = insights or {}

    doc = SimpleDocTemplate(
        output_path,
        pagesize=A4,
        leftMargin=MARGIN_X,
        rightMargin=MARGIN_X,
        topMargin=MARGIN_TOP,
        bottomMargin=MARGIN_BOTTOM,
        title=f"{REPORT_TITLE} \u2014 Job {job_id}",
        author=BRAND,
        subject="AI-assisted agronomic assessment from field video",
    )

    keyframe_dir = tempfile.mkdtemp(prefix="agro_keyframes_")
    keyframe_paths: List[str] = []
    for i, img in enumerate(keyframes or []):
        p = os.path.join(keyframe_dir, f"frame_{i}.jpg")
        rgb = img.convert("RGB")
        rgb.thumbnail((1400, 1400), PILImage.Resampling.LANCZOS)
        rgb.save(p, "JPEG", quality=88)
        keyframe_paths.append(p)

    story: list = []

    # Title block
    story.append(Paragraph(REPORT_TITLE, styles["title"]))
    story.append(Paragraph(
        f"Field assessment from video &nbsp;&middot;&nbsp; {generated_at}",
        styles["subtitle"],
    ))
    story.append(Spacer(1, 14))

    story.append(_summary_band(
        plant_count,
        [
            ("Job ID", job_id),
            ("Generated", generated_at),
            ("Source", "Video plant-count pipeline"),
            ("Keyframes analysed", str(len(keyframe_paths))),
        ],
        styles,
    ))

    # 01 Executive summary
    summary = _text_block(insights.get("executive_summary", ""), styles)
    story.append(KeepTogether([
        _section_heading(1, "Executive Summary", styles),
        Spacer(1, 8),
        _section_body(summary),
    ]))

    # 02 Keyframes
    story.append(_section_heading(2, "Representative Keyframes", styles))
    story.append(Spacer(1, 6))
    story.append(Paragraph(
        f"{len(keyframe_paths)} frames sampled evenly across the source video, in chronological order. "
        "These frames were provided to the vision model as the visual evidence for this report.",
        styles["small"],
    ))
    story.append(Spacer(1, 6))
    story.extend(_keyframe_gallery(keyframe_paths, styles))

    # 03-06 Insight sections
    sections = [
        (3, "Plant Health Assessment", insights.get("plant_health_assessment", "")),
        (4, "Nutrient Status & Deficiencies", insights.get("nutrient_deficiencies", "")),
        (5, "Stress, Disease & Pest Indicators", insights.get("stress_and_pest_indicators", "")),
        (6, "Actionable Recommendations", insights.get("actionable_recommendations", "")),
    ]
    for number, title, body in sections:
        block = _text_block(body, styles)
        story.append(Spacer(1, 4))
        story.append(KeepTogether([
            _section_heading(number, title, styles),
            Spacer(1, 8),
            _section_body(block),
        ]))

    # 07 Methodology & disclaimer
    story.append(KeepTogether([
        _section_heading(7, "Methodology & Disclaimer", styles),
        Spacer(1, 6),
        Paragraph("Methodology", styles["h2"]),
        Paragraph(
            f"Plants were detected and counted from the uploaded video by an object-detection and "
            f"tracking pipeline. {len(keyframe_paths)} representative keyframes were then sampled evenly "
            f"across the video and analysed by a multimodal vision-language model, which produced the "
            f"agronomic assessment presented in this report. The assessment is based solely on visual "
            f"evidence in those frames together with the detected plant count.",
            styles["body"],
        ),
    ]))
    story.append(Paragraph("Disclaimer", styles["h2"]))
    story.append(Paragraph(
        "This report is generated automatically with the assistance of artificial intelligence and is "
        "provided for advisory purposes only. Remote, image-based assessment cannot replace in-field "
        "scouting, tissue analysis or soil testing. Confirm all observations with a qualified agronomist "
        "and follow local regulations and product labels before applying any treatment.",
        styles["small"],
    ))

    decorate = _make_page_decorator(job_id, generated_at)
    try:
        doc.build(story, onFirstPage=decorate, onLaterPages=decorate, canvasmaker=_NumberedCanvas)
    finally:
        shutil.rmtree(keyframe_dir, ignore_errors=True)

    return output_path
