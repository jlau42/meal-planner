"""Convert a saved meal-plan Markdown file to PDF.

Run:  uv run md_to_pdf.py plans/<file>.md   (writes <file>.pdf next to it)
"""

import re
import sys
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer

styles = getSampleStyleSheet()
H1 = ParagraphStyle("H1", parent=styles["Title"], fontSize=20, spaceAfter=6, alignment=0)
H2 = ParagraphStyle("H2", parent=styles["Heading2"], fontSize=14, spaceBefore=12, spaceAfter=4,
                    textColor=colors.HexColor("#2f5d3a"))
H3 = ParagraphStyle("H3", parent=styles["Heading3"], fontSize=11.5, spaceBefore=8, spaceAfter=2)
BODY = ParagraphStyle("Body", parent=styles["BodyText"], fontSize=10.5, leading=14)
BULLET = ParagraphStyle("Bullet", parent=BODY, leftIndent=16, bulletIndent=4, spaceAfter=1)
# The built-in PDF fonts have no empty-checkbox glyph, so borrow one from Windows'
# Segoe UI Symbol when available; otherwise fall back to a plain "[ ]".
_SYMBOL_FONT = Path(r"C:\Windows\Fonts\seguisym.ttf")
if _SYMBOL_FONT.exists():
    pdfmetrics.registerFont(TTFont("SegoeSymbol", str(_SYMBOL_FONT)))
    CHECK = ParagraphStyle("Check", parent=BULLET, leftIndent=20, bulletFontName="SegoeSymbol", bulletFontSize=11)
    BOX = "\u2610"
else:
    CHECK = ParagraphStyle("Check", parent=BULLET, leftIndent=22)
    BOX = "[  ]"


def inline(text: str) -> str:
    text = text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    text = re.sub(r"\[(.+?)\]\((https?://[^)\s]+)\)", r'<a href="\2" color="#1a5fb4">\1</a>', text)
    text = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", text)
    text = re.sub(r"(?<!\w)_(.+?)_(?!\w)", r"<i>\1</i>", text)
    return text


def convert(md_path: Path) -> Path:
    story = []
    for line in md_path.read_text(encoding="utf-8").splitlines():
        s = line.strip()
        if not s:
            story.append(Spacer(1, 4))
        elif s.startswith("### "):
            story.append(Paragraph(inline(s[4:]), H3))
        elif s.startswith("## "):
            story.append(Paragraph(inline(s[3:]), H2))
        elif s.startswith("# "):
            story.append(Paragraph(inline(s[2:]), H1))
        elif s.startswith("- [ ] ") or s.startswith("- [x] "):
            story.append(Paragraph(inline(s[6:]), CHECK, bulletText=BOX))
        elif s.startswith("- "):
            story.append(Paragraph(inline(s[2:]), BULLET, bulletText="•"))
        else:
            story.append(Paragraph(inline(s), BODY))

    pdf_path = md_path.with_suffix(".pdf")
    title = md_path.read_text(encoding="utf-8").splitlines()[0].lstrip("# ").strip()
    doc = SimpleDocTemplate(str(pdf_path), pagesize=letter, title=title,
                            leftMargin=0.8 * inch, rightMargin=0.8 * inch,
                            topMargin=0.7 * inch, bottomMargin=0.7 * inch)
    doc.build(story)
    return pdf_path


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit("usage: uv run md_to_pdf.py <plan.md> [more.md ...]")
    for arg in sys.argv[1:]:
        print(f"Wrote {convert(Path(arg))}")
