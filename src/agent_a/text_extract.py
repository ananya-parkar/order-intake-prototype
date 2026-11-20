"""
A3 step 2 - turn any attachment into plain text (the "ground truth" A3 checks against).

    text PDF      -> pypdfium2 text layer, page by page
    scanned page  -> page has (almost) no text -> render at 300 DPI -> clean image -> Tesseract
    photo / PNG   -> clean image -> Tesseract
    Excel         -> openpyxl, one line per row, cells separated by tabs
    Word          -> python-docx, paragraphs and tables
    .txt / .csv   -> read as text

Libraries only READ the words. Working out which word is which field is the LLM's job (a3_extract).
Offline stand-in for Document AI: Tesseract also gives a confidence per word, kept in ocr_words.
"""
import os
import re
from dataclasses import dataclass, field
from datetime import date, datetime
from pathlib import Path
from typing import Optional

try:                                   # same .env as llm.py (TESSERACT_CMD lives there)
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).resolve().parents[2] / ".env")
except ImportError:
    pass

from .config import CFG
PDF_DPI = CFG.a3.pdf_dpi
MIN_PAGE_CHARS = CFG.a3.min_page_chars        # fewer characters than this = treat the page as a scan
MIN_OCR_WIDTH = CFG.a3.min_ocr_width          # enlarge smaller images before OCR


@dataclass
class ExtractResult:
    """The text read from one file, how it was read, and OCR confidence if OCR was used."""
    text: str = ""
    method: str = "none"               # pdf_text / ocr / excel / word / text / none
    pages: int = 0
    ocr_pages: int = 0
    ocr_words: list[tuple[str, float]] = field(default_factory=list)   # (word, confidence 0-1)
    ocr_confidence: Optional[float] = None                             # average over OCR words
    error: str = ""


# ---------------- OCR ----------------
def _tesseract():
    """Load pytesseract and point it at TESSERACT_CMD from .env if set."""
    import pytesseract
    cmd = os.getenv("TESSERACT_CMD")
    if cmd:
        pytesseract.pytesseract.tesseract_cmd = cmd
    return pytesseract


def clean_image(img):
    """Grey, upright, big enough, good contrast - most OCR errors come from the image, not Tesseract."""
    from PIL import ImageFilter, ImageOps
    img = ImageOps.exif_transpose(img)                 # phone photos: apply the camera's rotation
    img = img.convert("L")                             # greyscale
    if img.width < MIN_OCR_WIDTH:                      # enlarge small images (max 3x)
        scale = min(3.0, MIN_OCR_WIDTH / img.width)
        img = img.resize((int(img.width * scale), int(img.height * scale)))
    img = ImageOps.autocontrast(img, cutoff=1)         # dark text on white
    img = img.filter(ImageFilter.MedianFilter(3))      # remove speckle noise
    try:                                               # straighten pages turned 90/180 degrees
        osd = _tesseract().image_to_osd(img, output_type=_tesseract().Output.DICT)
        if osd.get("rotate"):
            img = img.rotate(-osd["rotate"], expand=True)
    except Exception:                                  # OSD data not installed or too little text
        pass
    return img


def ocr_image(img) -> tuple[str, list[tuple[str, float]]]:
    """OCR one image -> (text with line breaks, [(word, confidence 0-1)])."""
    pt = _tesseract()
    img = clean_image(img)
    data = pt.image_to_data(img, config="--psm 6", output_type=pt.Output.DICT)
    lines: dict[tuple, list[str]] = {}
    words = []
    for i, w in enumerate(data["text"]):
        w = (w or "").strip()
        conf = float(data["conf"][i])
        if not w or conf < 0:
            continue
        key = (data["block_num"][i], data["par_num"][i], data["line_num"][i])
        lines.setdefault(key, []).append(w)
        words.append((w, conf / 100))
    return "\n".join(" ".join(ws) for ws in lines.values()), words

def ocr(img) -> tuple[str, list[tuple[str, float]]]:
    """OCR one page with the engine set in config.yaml (Tesseract or Azure Document Intelligence)."""
    if CFG.ocr.provider == "azure_di":
        from PIL import ImageOps
        from .docintel import analyze_image
        return analyze_image(ImageOps.exif_transpose(img))     # DI cleans/rotates itself; just fix camera rotation
    return ocr_image(img)

def _finish_ocr(res: ExtractResult) -> None:
    """Average the OCR word confidences into one number for the document."""
    if res.ocr_words:
        res.ocr_confidence = round(sum(c for _, c in res.ocr_words) / len(res.ocr_words), 3)


# ---------------- readers ----------------
def read_pdf(path: Path) -> ExtractResult:
    """Read a PDF page by page; pages with almost no text are treated as scans and OCR'd."""
    import pypdfium2 as pdfium
    res = ExtractResult(method="pdf_text")
    pdf = pdfium.PdfDocument(str(path))
    try:
        parts = []
        res.pages = len(pdf)
        for i in range(len(pdf)):
            page = pdf[i]
            text = page.get_textpage().get_text_range().replace("\r\n", "\n").replace("\r", "\n")
            if len(text.strip()) < MIN_PAGE_CHARS:      # scanned page -> OCR it
                img = page.render(scale=PDF_DPI / 72).to_pil()
                text, words = ocr(img)
                res.ocr_words += words
                res.ocr_pages += 1
            parts.append(f"=== Page {i + 1} ===\n{text.strip()}")
        res.text = "\n\n".join(parts)
    finally:
        pdf.close()
    if res.ocr_pages:
        res.method = "ocr"
        _finish_ocr(res)
    return res


def read_image(path: Path) -> ExtractResult:
    """OCR an image (scan, phone photo, screenshot)."""
    from PIL import Image
    with Image.open(path) as img:
        text, words = ocr(img)
    res = ExtractResult(text=f"=== Page 1 ===\n{text}", method="ocr", pages=1, ocr_pages=1, ocr_words=words)
    _finish_ocr(res)
    return res


def _cell(v) -> str:
    """One Excel cell as text (dates as MM/DD/YYYY, 44000.0 -> 44000)."""
    if v is None:
        return ""
    if isinstance(v, (datetime, date)):
        return v.strftime("%m/%d/%Y")                   # write dates the way a PO shows them
    if isinstance(v, float) and v.is_integer():
        return str(int(v))
    return str(v).strip()


def read_excel(path: Path) -> ExtractResult:
    """Read every sheet of an Excel file (or a CSV) as tab-separated rows."""
    if path.suffix.lower() == ".csv":
        return read_text(path, method="excel")
    import openpyxl
    wb = openpyxl.load_workbook(path, data_only=True, read_only=True)   # data_only = formula results
    try:
        parts = []
        for ws in wb.worksheets:
            rows = []
            for row in ws.iter_rows(values_only=True):
                cells = [_cell(v) for v in row]
                while cells and not cells[-1]:
                    cells.pop()
                if any(cells):
                    rows.append("\t".join(cells))
            if rows:
                parts.append(f"=== Sheet {ws.title} ===\n" + "\n".join(rows))
        return ExtractResult(text="\n\n".join(parts), method="excel", pages=len(wb.worksheets))
    finally:
        wb.close()


def read_word(path: Path) -> ExtractResult:
    """Read a Word file: paragraphs and table rows."""
    import docx
    d = docx.Document(str(path))
    parts = [p.text for p in d.paragraphs if p.text.strip()]
    for t in d.tables:
        for row in t.rows:
            cells = []
            for c in row.cells:                         # merged cells repeat - keep each once
                txt = c.text.strip()
                if not cells or cells[-1] != txt:
                    cells.append(txt)
            if any(cells):
                parts.append("\t".join(cells))
    return ExtractResult(text="\n".join(parts), method="word", pages=1)


def read_text(path: Path, method: str = "text") -> ExtractResult:
    """Read a plain text file, trying common encodings."""
    raw = path.read_bytes()
    for enc in ("utf-8-sig", "cp1252", "latin-1"):
        try:
            return ExtractResult(text=raw.decode(enc), method=method, pages=1)
        except UnicodeDecodeError:
            continue
    return ExtractResult(error="could not decode text file")


READERS = {"pdf": read_pdf, "image": read_image, "excel": read_excel, "word": read_word, "text": read_text}


def extract(path: Path, kind: str) -> ExtractResult:
    """Read one attachment. Never raises: problems come back in .error."""
    reader = READERS.get(kind)
    if not reader:
        return ExtractResult(error=f"no reader for kind '{kind}'")
    if kind == "word" and path.suffix.lower() == ".doc":
        return ExtractResult(error="old .doc format not supported - ask for .docx or PDF")
    if kind == "excel" and path.suffix.lower() == ".xls":
        return ExtractResult(error="old .xls format not supported - ask for .xlsx or PDF")
    try:
        res = reader(path)
    except Exception as e:
        return ExtractResult(error=f"{type(e).__name__}: {e}")
    res.text = re.sub(r"[ \t]+\n", "\n", res.text).strip()
    if not res.text:
        res.error = res.error or "no text found"
    return res


if __name__ == "__main__":                              # quick look: python -m agent_a.text_extract <file> <kind>
    import sys
    r = extract(Path(sys.argv[1]), sys.argv[2])
    print(f"method={r.method} pages={r.pages} ocr_pages={r.ocr_pages} ocr_conf={r.ocr_confidence} error={r.error}\n")
    print(r.text[:3000])
    