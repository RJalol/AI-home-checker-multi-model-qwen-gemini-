"""
report_generator.py
Turns the JSON analysis returned by HomeworkChecker into three deliverables:

  1. generate_html_report()   -> a styled, shareable .html file
  2. generate_pdf_report()    -> a printable .pdf file (pure-python, no
                                  system dependencies like wkhtmltopdf)
  3. generate_excel_gradebook() -> a .xlsx gradebook across many students

Each function returns the Path it wrote to.
"""

import html as html_lib
import re
from datetime import datetime
from pathlib import Path
from typing import List, Tuple

import markdown as md_lib
from jinja2 import Environment, FileSystemLoader, select_autoescape

from config import REPORTS_DIR, TEMPLATES_DIR, grade_for_score


def _ensure_blank_line_before_lists(text: str) -> str:
    """
    python-markdown only recognizes "- " lines as a list (and knows where
    a list ends) when blank lines separate them from surrounding
    paragraph text; otherwise a line right after a bullet gets treated as
    a lazy continuation of that same bullet instead of its own paragraph.
    Gemini's feedback often has no such blank lines, so we insert them.
    """
    lines = text.split("\n")
    out = []
    prev_was_list = False
    for line in lines:
        stripped = line.strip()
        is_list = stripped.startswith("- ") or stripped.startswith("* ")
        if out and out[-1].strip() != "":
            if is_list and not prev_was_list:
                out.append("")  # entering a list
            elif not is_list and prev_was_list and stripped != "":
                out.append("")  # leaving a list
        out.append(line)
        prev_was_list = is_list
    return "\n".join(out)


try:
    import nh3  # HTML sanitiser (Rust, tiny); see requirements.txt
except ImportError:  # pragma: no cover - exercised only on installs without nh3
    nh3 = None

# The only markup the report needs: what light markdown (**bold**, "- " bullets, `code`) produces.
# Everything else -- <script>, <img onerror=...>, <svg onload=...>, links, any attribute -- is removed.
_ALLOWED_TAGS = {
    "p", "br", "strong", "em", "b", "i", "u", "ul", "ol", "li", "code", "pre", "blockquote",
    "h1", "h2", "h3", "h4", "h5", "h6", "sub", "sup", "hr",
}


def _render_md(text: str) -> str:
    """
    Converts the light markdown Gemini is instructed to use in feedback
    text (**bold**, "- " bullets, `code`) into HTML for the report.
    Used for the HTML report (inserted with the |safe filter) -- so the result is SANITISED here:
    the text comes from a model that echoes student-written content, and a student can write
    markup into their homework. The PDF report has its own lightweight renderer below, since
    fpdf2 doesn't consume HTML/CSS the way a browser does.
    """
    if not text:
        return ""
    source = _ensure_blank_line_before_lists(str(text))
    if nh3 is None:
        # No sanitiser installed: fail SAFE by neutralising every raw tag before markdown runs.
        source = html_lib.escape(source)
    rendered = md_lib.markdown(source, extensions=["nl2br"])
    if nh3 is not None:
        rendered = nh3.clean(rendered, tags=_ALLOWED_TAGS, attributes={}, url_schemes=set(), link_rel=None)
    return rendered


def safe_filename_part(name: str, default: str = "student", max_len: int = 60) -> str:
    """A student name reduced to filename-safe characters: no path separators, dots or spaces."""
    cleaned = re.sub(r"[^\w\-]+", "_", str(name or ""), flags=re.UNICODE).strip("_")
    return cleaned[:max_len] or default


def _render_md_dict(result: dict) -> dict:
    """
    Returns a deep-ish copy of the grading result with every narrative
    text field pre-rendered from markdown to HTML, ready to insert into
    the Jinja2 template with |safe. Doesn't mutate the original dict, so
    callers can still use the raw result elsewhere (e.g. the Excel
    gradebook, which wants plain text, not HTML).
    """
    rendered = dict(result)
    rendered["questions"] = []
    for q in result.get("questions", []):
        rq = dict(q)
        for field in ("question", "student_answer", "expected_solution", "analysis", "correct_answer"):
            if field in rq:
                rq[field] = _render_md(rq[field])
        rq["mistakes"] = [_render_md(m) for m in q.get("mistakes", [])]
        rq["suggestions"] = [_render_md(s) for s in q.get("suggestions", [])]
        rendered["questions"].append(rq)

    summary = result.get("summary", {}) or {}
    rendered["summary"] = {
        key: [_render_md(item) for item in summary.get(key, [])]
        for key in ("strengths", "weaknesses", "topics_to_review", "recommendations")
    }
    return rendered


# ---------------------------------------------------------------------------
# HTML report
# ---------------------------------------------------------------------------
def generate_html_report(result: dict, student_name: str = "Student", out_name: str = None) -> Path:
    env = Environment(
        loader=FileSystemLoader(str(TEMPLATES_DIR)),
        autoescape=select_autoescape(["html"]),
    )
    template = env.get_template("report.html")

    rendered_result = _render_md_dict(result)
    for q in rendered_result["questions"]:
        q["grade_label"], q["grade_color"] = grade_for_score(q["score"])
    overall_label, overall_color = grade_for_score(result["overall_score"])

    html = template.render(
        student_name=student_name,
        generated_at=datetime.now().strftime("%Y-%m-%d %H:%M"),
        overall_score=result["overall_score"],
        grade=result.get("grade", overall_label),
        grade_color=overall_color,
        questions=rendered_result["questions"],
        summary=rendered_result["summary"],
    )

    out_name = out_name or f"report_{safe_filename_part(student_name)}_{datetime.now():%Y%m%d%H%M%S}.html"
    out_path = REPORTS_DIR / out_name
    out_path.write_text(html, encoding="utf-8")
    return out_path


# ---------------------------------------------------------------------------
# PDF report (pure python via fpdf2 -- no system deps required)
# ---------------------------------------------------------------------------
def generate_pdf_report(result: dict, student_name: str = "Student", out_name: str = None) -> Path:
    from fpdf import FPDF

    pdf = FPDF(format="A4")
    pdf.set_auto_page_break(auto=True, margin=15)

    # Helvetica (a core PDF font) only supports Latin-1 and crashes on
    # math symbols like √, ∫, ≠, ≤, π, etc. We ship a bundled Unicode
    # font (DejaVu Sans) so any symbol Gemini returns renders safely.
    font_dir = Path(__file__).resolve().parent / "static" / "fonts"
    regular_font = font_dir / "DejaVuSans.ttf"
    bold_font = font_dir / "DejaVuSans-Bold.ttf"
    use_unicode_font = regular_font.exists() and bold_font.exists()

    if use_unicode_font:
        pdf.add_font("DejaVu", "", str(regular_font))
        pdf.add_font("DejaVu", "B", str(bold_font))
        font_name = "DejaVu"
    else:
        # Fallback so the app still works if the font files are missing,
        # though unusual math symbols may then raise an encoding error.
        font_name = "Helvetica"

    pdf.add_page()

    def h1(text):
        pdf.set_font(font_name, "B", 16)
        pdf.multi_cell(0, 10, text, new_x="LMARGIN", new_y="NEXT")
        pdf.ln(1)

    def h2(text):
        pdf.set_font(font_name, "B", 12)
        pdf.multi_cell(0, 8, text, new_x="LMARGIN", new_y="NEXT")

    def body(text, size=10):
        pdf.set_font(font_name, "", size)
        pdf.multi_cell(0, 6, text, new_x="LMARGIN", new_y="NEXT")

    def _write_inline(line, size=10):
        """Writes one line, rendering **bold** spans inline and stripping
        `code` backticks (no monospace font bundled, so just de-marks it)."""
        line = re.sub(r"`([^`]+)`", r"\1", line)
        parts = re.split(r"(\*\*[^*]+\*\*)", line)
        for part in parts:
            if not part:
                continue
            if part.startswith("**") and part.endswith("**") and len(part) > 4:
                pdf.set_font(font_name, "B", size)
                pdf.write(6, part[2:-2])
            else:
                pdf.set_font(font_name, "", size)
                pdf.write(6, part)

    def markdown_lite(text, size=10):
        """
        Renders Gemini's light markdown (**bold**, "- " bullets, `code`)
        reasonably in the PDF without needing a full HTML/CSS engine --
        fpdf2's write_html() assumes browser-like CSS support we'd rather
        not depend on for a few simple markers.
        """
        if not text:
            return
        for raw_line in str(text).split("\n"):
            line = raw_line.strip()
            if not line:
                pdf.ln(2)
                continue
            if line.startswith(("- ", "* ")):
                pdf.set_x(pdf.l_margin + 4)
                pdf.set_font(font_name, "", size)
                pdf.write(6, "\u2022  ")
                _write_inline(line[2:], size)
                pdf.ln(6)
                pdf.set_x(pdf.l_margin)
            else:
                _write_inline(line, size)
                pdf.ln(6)

    def bullet_list(items, size=10):
        for item in items:
            pdf.set_x(pdf.l_margin)
            pdf.set_font(font_name, "", size)
            pdf.write(6, "\u2022  ")
            _write_inline(str(item), size)
            pdf.ln(6)

    overall_label, _ = grade_for_score(result["overall_score"])

    h1("Homework Grading Report")
    body(f"Student: {student_name}")
    body(f"Generated: {datetime.now():%Y-%m-%d %H:%M}")
    body(f"Overall Score: {result['overall_score']}/100  ({result.get('grade', overall_label)})")
    pdf.ln(4)

    for q in result["questions"]:
        pdf.set_draw_color(200, 200, 200)
        pdf.line(pdf.get_x(), pdf.get_y(), pdf.get_x() + 190, pdf.get_y())
        pdf.ln(3)

        h2(f"Question {q['question_number']}  -  Score: {q['score']}/100  ({q.get('difficulty', '')})")
        body("Question:")
        markdown_lite(q["question"])
        body("Student answer:")
        markdown_lite(q["student_answer"])
        body("Expected solution:")
        markdown_lite(q["expected_solution"])
        pdf.ln(1)
        body("Analysis:")
        markdown_lite(q["analysis"])

        if q.get("mistakes"):
            pdf.ln(1)
            body("Mistakes identified:")
            bullet_list(q["mistakes"])

        if q.get("error_categories"):
            body("Error categories: " + ", ".join(q["error_categories"]))

        if q.get("correct_answer"):
            pdf.ln(1)
            body("Correct answer:")
            markdown_lite(q["correct_answer"])

        if q.get("suggestions"):
            pdf.ln(1)
            body("Suggestions:")
            bullet_list(q["suggestions"])

        pdf.ln(4)

    summary = result.get("summary", {})
    pdf.add_page()
    h1("Final Summary")
    for label, key in [
        ("Strengths", "strengths"),
        ("Weaknesses", "weaknesses"),
        ("Topics to Review", "topics_to_review"),
        ("Study Recommendations", "recommendations"),
    ]:
        items = summary.get(key) or []
        if items:
            h2(label)
            bullet_list(items)
            pdf.ln(2)

    out_name = out_name or f"report_{safe_filename_part(student_name)}_{datetime.now():%Y%m%d%H%M%S}.pdf"
    out_path = REPORTS_DIR / out_name
    pdf.output(str(out_path))
    return out_path


# ---------------------------------------------------------------------------
# Excel gradebook (across one or many students/submissions)
# ---------------------------------------------------------------------------
def generate_excel_gradebook(
    records: List[Tuple[str, dict]],
    out_name: str = None,
) -> Path:
    """
    records: list of (student_name, result_dict) tuples.
    Produces a workbook with:
      - "Gradebook" sheet: one row per student with overall score/grade
      - "Question Detail" sheet: one row per question across all students
    """
    from openpyxl import Workbook
    from openpyxl.styles import Font, PatternFill, Alignment
    from openpyxl.utils import get_column_letter

    wb = Workbook()

    header_fill = PatternFill(start_color="1F2937", end_color="1F2937", fill_type="solid")
    header_font = Font(color="FFFFFF", bold=True)

    # --- Sheet 1: Gradebook overview ---
    ws1 = wb.active
    ws1.title = "Gradebook"
    headers1 = ["Student", "Overall Score", "Grade", "Questions Detected", "Generated At"]
    ws1.append(headers1)
    for col in range(1, len(headers1) + 1):
        cell = ws1.cell(row=1, column=col)
        cell.fill = header_fill
        cell.font = header_font
        cell.alignment = Alignment(horizontal="center")

    for student_name, result in records:
        label, _ = grade_for_score(result["overall_score"])
        ws1.append([
            student_name,
            result["overall_score"],
            result.get("grade", label),
            len(result.get("questions", [])),
            datetime.now().strftime("%Y-%m-%d %H:%M"),
        ])

    for i, width in enumerate([24, 14, 18, 18, 18], start=1):
        ws1.column_dimensions[get_column_letter(i)].width = width

    # --- Sheet 2: Per-question detail ---
    ws2 = wb.create_sheet("Question Detail")
    headers2 = [
        "Student", "Q#", "Score", "Difficulty", "Error Categories",
        "Mistakes", "Correct Answer",
    ]
    ws2.append(headers2)
    for col in range(1, len(headers2) + 1):
        cell = ws2.cell(row=1, column=col)
        cell.fill = header_fill
        cell.font = header_font
        cell.alignment = Alignment(horizontal="center")

    for student_name, result in records:
        for q in result.get("questions", []):
            ws2.append([
                student_name,
                q.get("question_number"),
                q.get("score"),
                q.get("difficulty"),
                ", ".join(q.get("error_categories", [])),
                "; ".join(q.get("mistakes", [])),
                q.get("correct_answer"),
            ])

    for i, width in enumerate([24, 6, 8, 12, 24, 40, 30], start=1):
        ws2.column_dimensions[get_column_letter(i)].width = width
    ws2.freeze_panes = "A2"
    ws1.freeze_panes = "A2"

    out_name = out_name or f"gradebook_{datetime.now():%Y%m%d%H%M%S}.xlsx"
    out_path = REPORTS_DIR / out_name
    wb.save(str(out_path))
    return out_path
