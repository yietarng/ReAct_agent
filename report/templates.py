from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY, TA_LEFT
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle


def build_styles():
    """Return a StyleSheet with all custom paragraph styles for the report."""
    styles = getSampleStyleSheet()

    styles.add(ParagraphStyle(
        "CoverTitle",
        fontSize=26,
        leading=32,
        alignment=TA_CENTER,
        textColor=colors.HexColor("#1a1a2e"),
        spaceAfter=18,
        fontName="Helvetica-Bold",
    ))

    styles.add(ParagraphStyle(
        "SubTitle",
        fontSize=14,
        leading=18,
        alignment=TA_CENTER,
        textColor=colors.HexColor("#16213e"),
        spaceAfter=12,
        fontName="Helvetica",
    ))

    styles.add(ParagraphStyle(
        "SectionHeader",
        fontSize=14,
        leading=18,
        textColor=colors.HexColor("#16213e"),
        spaceBefore=14,
        spaceAfter=6,
        fontName="Helvetica-Bold",
    ))

    styles.add(ParagraphStyle(
        "FieldLabel",
        fontSize=10,
        leading=13,
        fontName="Helvetica-Bold",
        textColor=colors.HexColor("#0f3460"),
        spaceBefore=8,
        spaceAfter=2,
    ))

    styles.add(ParagraphStyle(
        "Body",
        fontSize=10,
        leading=14,
        alignment=TA_JUSTIFY,
        fontName="Helvetica",
    ))

    styles.add(ParagraphStyle(
        "TOCEntry",
        fontSize=11,
        leading=16,
        alignment=TA_LEFT,
        fontName="Helvetica",
        spaceAfter=4,
    ))

    styles.add(ParagraphStyle(
        "CoverMeta",
        fontSize=10,
        leading=14,
        alignment=TA_CENTER,
        textColor=colors.HexColor("#555555"),
        fontName="Helvetica",
        spaceAfter=6,
    ))

    return styles
