"""
hwchecker/pipeline/ocr.py
Document parsing helpers. There is no separate OCR call in this pipeline: a
scanned page / photo is handed straight to a vision-capable grading provider.
What this module does is tell the router which kind of input it is dealing
with -- needs_vision_ocr() is how router/grading_router.py decides that a
photo or PDF must go to a provider that can see (never the text-only GLM
tier) -- and extract plain text from DOCX/txt/md files without any model call.
"""

from pathlib import Path
from typing import Optional

TEXT_EXTRACTABLE_EXTENSIONS = {".docx", ".txt", ".md"}
IMAGE_EXTENSIONS = {".pdf", ".png", ".jpg", ".jpeg", ".webp"}


def needs_vision_ocr(file_path) -> bool:
    """True if this file needs a vision LLM to read (scanned/handwritten
    page or image), False if it can be parsed as plain text for $0."""
    return Path(file_path).suffix.lower() in IMAGE_EXTENSIONS


def extract_text_if_possible(file_path) -> Optional[str]:
    """
    Returns extracted plain text for DOCX/txt/md files ($0, no LLM
    call), or None if this file needs vision OCR instead (caller should
    then pass the raw file to a vision-capable provider, e.g.
    providers/gemini.py, rather than calling this again).
    """
    file_path = Path(file_path)
    ext = file_path.suffix.lower()

    if ext == ".docx":
        from utils import extract_text_from_docx

        return extract_text_from_docx(file_path)
    if ext in (".txt", ".md"):
        return file_path.read_text(encoding="utf-8", errors="replace")

    return None  # needs vision OCR -- see needs_vision_ocr()
