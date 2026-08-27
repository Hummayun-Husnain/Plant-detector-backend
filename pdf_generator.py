"""PDF report generator for the AI agronomic reporting pipeline."""

import os
import shutil
import tempfile
from datetime import datetime
from xml.sax.saxutils import escape
from typing import List

from PIL import Image as PILImage
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import (
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
    Image as ReportImage,
)


def _format_text(text: str) -> str:
    return escape(str(text or "")).replace("\n", "<br/>")


def _header_paragraph(text: str, style: ParagraphStyle) -> Paragraph:
    return Paragraph(text, style)


def _section_card(heading: str, body: str, styles: dict) -> Table:
    text = (
        f"<b><font color='#2e7d4f' size='11'>{heading}</font></b><br/><br/>"
        f"{_format_text(body)}"
    )
    right = Paragraph(text, styles["card_body"])
    left = Paragraph("", styles["card_gutter"])
    card = Table(
        [[left, right]],
        colWidths=[0.10 * inch, 6.3 * inch],
    )
    card.setStyle(
        TableStyle([
            ("BACKGROUND", (0, 0), (0, 0), colors.HexColor("#2e7d4f")),
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ("LEFTPADDING", (0, 0), (0, 0), 0),
            ("RIGHTPADDING", (0, 0), (0, 0), 0),
            ("LEFTPADDING", (1, 0), (1, 0), 10),
            ("RIGHTPADDING", (1, 0), (1, 0), 10),
            ("TOPPADDING", (1, 0), (1, 0), 10),
            ("BOTTOMPADDING", (1, 0), (1, 0), 10),
            ("BOX", (0, 0), (-1, -1), 0.5, colors.HexColor("#d1d5db")),
        ])
    )
    return card


def _image_flowable(path: str, max_width: float = 2.8 * inch) -> ReportImage:
    with PILImage.open(path) as img:
        w, h = img.size
        aspect = h / w if w else 1.0
        height = max_width * aspect
    return ReportImage(path, width=max_width, height=height)


def create_pdf(
    output_path: str,
    job_id: str,
    plant_count: int,
    insights: dict,
    keyframes: List[PILImage.Image],
) -> str:
    generated_at = datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC")

    doc = SimpleDocTemplate(
        output_path,
        pagesize=A4,
        rightMargin=0.55 * inch,
        leftMargin=0.55 * inch,
        topMargin=0.65 * inch,
        bottomMargin=0.65 * inch,
    )

    base = getSampleStyleSheet()
    styles = {
        "title": ParagraphStyle(
            "ReportTitle",
            parent=base["Title"],
            fontSize=28,
            textColor=colors.HexColor("#1a472a"),
            alignment=1,
            spaceAfter=4,
        ),
        "subtitle": ParagraphStyle(
            "ReportSubtitle",
            parent=base["Normal"],
            fontSize=10,
            textColor=colors.HexColor("#6b7280"),
            alignment=1,
            spaceAfter=18,
        ),
        "section_header": ParagraphStyle(
            "SectionHeader",
            parent=base["Heading2"],
            fontSize=13,
            textColor=colors.HexColor("#1a472a"),
            spaceAfter=8,
            spaceBefore=10,
        ),
        "card_body": ParagraphStyle(
            "CardBody",
            parent=base["BodyText"],
            fontSize=10,
            leading=14,
            textColor=colors.HexColor("#374151"),
        ),
        "card_gutter": ParagraphStyle(
            "CardGutter",
            parent=base["Normal"],
            fontSize=1,
        ),
        "label": ParagraphStyle(
            "Label",
            parent=base["BodyText"],
            fontSize=10,
            textColor=colors.HexColor("#6b7280"),
        ),
        "value": ParagraphStyle(
            "Value",
            parent=base["BodyText"],
            fontSize=10,
            textColor=colors.HexColor("#1f2937"),
        ),
        "footer": ParagraphStyle(
            "Footer",
            parent=base["Normal"],
            fontSize=8,
            textColor=colors.HexColor("#9ca3af"),
            alignment=1,
        ),
    }

    story = []

    # Top banner
    banner = Table(
        [[Paragraph("AGRONOMIC REPORT", styles["title"])]],
        colWidths=[6.5 * inch],
    )
    banner.setStyle(
        TableStyle([
            ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#eefdf4")),
            ("BOX", (0, 0), (-1, -1), 1.5, colors.HexColor("#2e7d4f")),
            ("TOPPADDING", (0, 0), (-1, -1), 14),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 14),
            ("ALIGN", (0, 0), (-1, -1), "CENTER"),
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ])
    )
    story.append(banner)
    story.append(Paragraph(f"Generated {generated_at} • Job {job_id}", styles["subtitle"]))
    story.append(Spacer(1, 6))

    # Metadata table
    meta_data = [
        [Paragraph("Job ID", styles["label"]), Paragraph(job_id, styles["value"])],
        [Paragraph("Generated", styles["label"]), Paragraph(generated_at, styles["value"])],
        [Paragraph("Source", styles["label"]), Paragraph("Video plant-count pipeline", styles["value"])],
    ]
    meta_table = Table(meta_data, colWidths=[1.5 * inch, 5.0 * inch])
    meta_table.setStyle(
        TableStyle([
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ("LINEBELOW", (0, 0), (-1, -2), 0.5, colors.HexColor("#e5e7eb")),
            ("TOPPADDING", (0, 0), (-1, -1), 6),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
        ])
    )
    story.append(meta_table)
    story.append(Spacer(1, 10))

    # Plant count badge
    count_badge = Table(
        [
            [Paragraph("TOTAL PLANT COUNT", styles["label"])],
            [Paragraph(f"<b><font size='26' color='white'>{plant_count}</font></b>", styles["value"])],
        ],
        colWidths=[2.4 * inch],
    )
    count_badge.setStyle(
        TableStyle([
            ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#2e7d4f")),
            ("ALIGN", (0, 0), (-1, -1), "CENTER"),
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ("TOPPADDING", (0, 0), (-1, -1), 10),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 10),
            ("BOX", (0, 0), (-1, -1), 1, colors.HexColor("#1a472a")),
        ])
    )
    story.append(count_badge)
    story.append(Spacer(1, 18))

    # Keyframes
    story.append(Paragraph("Representative Keyframes", styles["section_header"]))
    keyframe_dir = tempfile.mkdtemp(prefix="agro_keyframes_")
    keyframe_paths = []
    for i, img in enumerate(keyframes):
        p = os.path.join(keyframe_dir, f"frame_{i}.jpg")
        img.convert("RGB").save(p, "JPEG", quality=85)
        keyframe_paths.append(p)

    cols = 2
    flowables = [_image_flowable(p) for p in keyframe_paths]
    rows = []
    for i in range(0, len(flowables), cols):
        row = flowables[i : i + cols]
        while len(row) < cols:
            row.append(Paragraph("", styles["card_body"]))
        rows.append(row)

    if rows:
        keyframe_table = Table(
            rows,
            colWidths=[2.8 * inch] * cols,
            hAlign="CENTER",
        )
        keyframe_table.setStyle(
            TableStyle([
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("ALIGN", (0, 0), (-1, -1), "CENTER"),
                ("TOPPADDING", (0, 0), (-1, -1), 8),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
            ])
        )
        story.append(keyframe_table)
        story.append(Spacer(1, 12))

    # Agronomic insights
    story.append(Paragraph("AI-Generated Agronomic Insights", styles["section_header"]))
    story.append(Spacer(1, 4))

    sections = [
        ("Executive Summary", insights.get("executive_summary", "")),
        ("Plant Health Assessment", insights.get("plant_health_assessment", "")),
        ("Nutrient Deficiencies", insights.get("nutrient_deficiencies", "")),
        ("Stress & Pest Indicators", insights.get("stress_and_pest_indicators", "")),
        ("Actionable Recommendations", insights.get("actionable_recommendations", "")),
    ]

    for heading, text in sections:
        story.append(_section_card(heading, text, styles))
        story.append(Spacer(1, 12))

    story.append(Spacer(1, 20))
    story.append(
        Paragraph(
            "This report was generated automatically by the AI agronomic reporting pipeline.",
            styles["footer"],
        )
    )

    try:
        doc.build(story)
    finally:
        shutil.rmtree(keyframe_dir, ignore_errors=True)

    return output_path
