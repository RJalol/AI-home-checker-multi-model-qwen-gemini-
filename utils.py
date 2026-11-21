"""
utils.py
Small, dependency-light helper functions used across the app:
- saving uploads safely
- validating file types/sizes
- cleaning + validating the JSON that comes back from Gemini
- extracting text from .docx files (Gemini doesn't accept docx directly)
"""

import io
import json
import re
import uuid
from pathlib import Path
from typing import Optional, Tuple

from config import ALLOWED_EXTENSIONS, IMAGE_EXTENSIONS, MAX_FILE_SIZE_MB, MAX_IMAGE_SIDE_PX, UPLOAD_DIR


class UnsupportedFileError(Exception):
    pass


class FileTooLargeError(Exception):
    pass


class InvalidModelResponseError(Exception):
    pass


def validate_upload(filename: str, size_bytes: int) -> None:
    """Raises a descriptive error if the uploaded file fails basic checks."""
    ext = Path(filename).suffix.lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise UnsupportedFileError(
            f"'{ext}' is not supported. Allowed types: "
            f"{', '.join(sorted(ALLOWED_EXTENSIONS))}"
        )
    max_bytes = MAX_FILE_SIZE_MB * 1024 * 1024
    if size_bytes > max_bytes:
        raise FileTooLargeError(
            f"File is {size_bytes / (1024 * 1024):.1f} MB; "
            f"the limit is {MAX_FILE_SIZE_MB} MB."
        )


_HEIF_EXTENSIONS = {".heic", ".heif"}
_JPEG_QUALITY = 90


def normalize_image(file_bytes: bytes, ext: str) -> Tuple[bytes, str]:
    """
    Makes a phone photo safe to send to any AI provider. Returns (bytes, extension).

      * HEIC/HEIF (iPhone default) -> JPEG: no provider accepts HEIC.
      * EXIF orientation applied: phones store a sideways picture plus a "rotate me" tag that
        the vision APIs ignore, so a portrait homework page used to arrive rotated 90 degrees.
      * Longest side limited to MAX_IMAGE_SIDE_PX: smaller upload, fewer image tokens.

    A picture that needs none of this is returned untouched (no needless re-compression).
    Raises UnsupportedFileError for an unreadable image, or for HEIC when the optional
    pillow-heif package is missing. If Pillow itself is not installed, ordinary images are
    passed through unchanged.
    """
    ext = ext.lower()
    try:
        from PIL import Image, ImageOps
    except ImportError:
        if ext in _HEIF_EXTENSIONS:
            raise UnsupportedFileError(
                f"'{ext}' photos need the Pillow and pillow-heif packages (pip install Pillow pillow-heif). "
                "Alternatively upload the photo as JPEG or PNG."
            )
        return file_bytes, ext

    if ext in _HEIF_EXTENSIONS:
        try:
            import pillow_heif

            pillow_heif.register_heif_opener()
        except ImportError:
            raise UnsupportedFileError(
                f"'{ext}' photos need the pillow-heif package (pip install pillow-heif). "
                "Alternatively upload the photo as JPEG or PNG."
            )

    try:
        image = Image.open(io.BytesIO(file_bytes))
        image.load()
    except Exception as exc:
        raise UnsupportedFileError(f"The image could not be read ({exc.__class__.__name__}: {exc}).") from exc

    orientation = image.getexif().get(0x0112, 1) if hasattr(image, "getexif") else 1
    too_big = max(image.size) > MAX_IMAGE_SIDE_PX
    if ext not in _HEIF_EXTENSIONS and orientation in (0, 1) and not too_big:
        return file_bytes, ext

    image = ImageOps.exif_transpose(image)
    if max(image.size) > MAX_IMAGE_SIDE_PX:
        image.thumbnail((MAX_IMAGE_SIDE_PX, MAX_IMAGE_SIDE_PX), Image.LANCZOS)

    if image.mode in ("RGBA", "LA") or (image.mode == "P" and "transparency" in image.info):
        rgba = image.convert("RGBA")
        flattened = Image.new("RGB", rgba.size, (255, 255, 255))   # transparent -> white paper
        flattened.paste(rgba, mask=rgba.split()[-1])
        image = flattened
    elif image.mode != "RGB":
        image = image.convert("RGB")

    out = io.BytesIO()
    image.save(out, format="JPEG", quality=_JPEG_QUALITY, optimize=True)
    return out.getvalue(), ".jpg"


def check_upload_content(file_bytes: bytes, original_filename: str) -> None:
    """
    Rejects a file whose CONTENT does not match its extension, for the formats that are cheap to
    recognise: .pdf must contain the "%PDF-" header (the spec allows it anywhere in the first
    1024 bytes) and .docx must be a ZIP archive holding word/document.xml. Images are verified
    separately by actually decoding them (normalize_image); audio is left to the provider.
    Raises UnsupportedFileError.
    """
    ext = Path(original_filename).suffix.lower()
    if ext == ".pdf":
        if b"%PDF-" not in file_bytes[:1024]:
            raise UnsupportedFileError("The file is named .pdf but its content is not a PDF document.")
    elif ext == ".docx":
        import zipfile
        try:
            with zipfile.ZipFile(io.BytesIO(file_bytes)) as archive:
                if "word/document.xml" not in archive.namelist():
                    raise UnsupportedFileError("The file is named .docx but is not a Word document.")
        except zipfile.BadZipFile:
            raise UnsupportedFileError("The file is named .docx but is not a Word document.")


def save_uploaded_file(file_bytes: bytes, original_filename: str) -> Path:
    """Saves an uploaded file under a unique name and returns its path.

    Photos are normalised first (see normalize_image) and so may be stored with a different
    extension than the one uploaded (.heic -> .jpg). Raises UnsupportedFileError if a photo
    cannot be read/converted."""
    check_upload_content(file_bytes, original_filename)
    ext = Path(original_filename).suffix.lower()
    if ext in IMAGE_EXTENSIONS:
        file_bytes, ext = normalize_image(file_bytes, ext)
    safe_name = f"{uuid.uuid4().hex}{ext}"
    dest = UPLOAD_DIR / safe_name
    dest.write_bytes(file_bytes)
    return dest


def extract_text_from_docx(path: Path) -> str:
    """Extracts plain text (paragraphs + tables) from a .docx file."""
    import docx  # python-docx

    document = docx.Document(str(path))
    parts = []

    for para in document.paragraphs:
        if para.text.strip():
            parts.append(para.text)

    for table in document.tables:
        for row in table.rows:
            cells = [cell.text.strip() for cell in row.cells]
            if any(cells):
                parts.append(" | ".join(cells))

    return "\n".join(parts)


def strip_markdown_fences(text: str) -> str:
    """
    Gemini is instructed to return raw JSON, but models occasionally wrap
    output in ```json ... ``` anyway. This removes that wrapping safely.
    """
    cleaned = text.strip()
    cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
    cleaned = re.sub(r"\s*```$", "", cleaned)
    return cleaned.strip()


def parse_and_validate_json(raw_text: str) -> dict:
    """
    Parses the model's raw text response into a dict and checks that the
    minimum required keys are present. Raises InvalidModelResponseError
    with a helpful message if parsing/validation fails.
    """
    cleaned = strip_markdown_fences(raw_text)

    try:
        data = json.loads(cleaned)
    except json.JSONDecodeError as exc:
        raise InvalidModelResponseError(
            f"Model response was not valid JSON: {exc}\n---\n{raw_text[:500]}"
        ) from exc

    required_top_level = {"overall_score", "grade", "questions", "summary"}
    missing = required_top_level - data.keys()
    if missing:
        raise InvalidModelResponseError(
            f"Model JSON is missing required keys: {sorted(missing)}"
        )

    if not isinstance(data["questions"], list) or len(data["questions"]) == 0:
        raise InvalidModelResponseError(
            "Model JSON contains no detected questions ('questions' is empty)."
        )

    required_question_keys = {
        "question_number", "question", "student_answer", "expected_solution",
        "analysis", "mistakes", "error_categories", "correct_answer",
        "suggestions", "difficulty", "score",
    }
    for i, q in enumerate(data["questions"]):
        missing_q = required_question_keys - q.keys()
        if missing_q:
            raise InvalidModelResponseError(
                f"Question #{i + 1} in model JSON is missing keys: {sorted(missing_q)}"
            )

    return data


def guess_mime_type(path: Path) -> Optional[str]:
    ext = path.suffix.lower()
    return {
        ".pdf": "application/pdf",
        ".png": "image/png",
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".webp": "image/webp",
        ".heic": "image/heic",
        ".heif": "image/heif",
        ".mp3": "audio/mpeg",
        ".wav": "audio/wav",
        ".m4a": "audio/mp4",
        ".ogg": "audio/ogg",
    }.get(ext)
