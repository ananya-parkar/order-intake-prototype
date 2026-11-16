"""
A1 - Pick up the case.
Reads every .eml / .msg in a folder -> one Case per email -> output/<case_id>/case.json
Run from the src folder:
    python -m agent_a.a1_read --inbox ../data/samples --out ../output
"""
import argparse
import hashlib
import json
import mimetypes
import re
import shutil
from datetime import datetime, timezone
from email import policy
from email.parser import BytesParser
from email.utils import parseaddr, parsedate_to_datetime
from html.parser import HTMLParser
from pathlib import Path

from .models import Attachment, Case

ROOT = Path(__file__).resolve().parents[2]
SIGNATURE_MAX_BYTES = 30_000       # inline images smaller than this = logos
MAX_NESTING = 2                    # forwarded email inside forwarded email


# ---------- small helpers ----------
def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def safe_name(name: str) -> str:
    name = re.sub(r'[<>:"/\\|?*\x00-\x1f]', "_", name).strip(" .")
    return name[:150] or "unnamed"


def kind_of(filename: str, content_type: str) -> str:
    ext, ct = Path(filename).suffix.lower(), (content_type or "").lower()
    if ext == ".pdf" or ct == "application/pdf":                      return "pdf"
    if ext in {".xlsx", ".xlsm", ".xls", ".csv"} or "spreadsheet" in ct: return "excel"
    if ext in {".docx", ".doc"} or "word" in ct:                      return "word"
    if ext in {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"} or ct.startswith("image/"): return "image"
    if ext in {".eml", ".msg"} or ct == "message/rfc822":             return "email"
    if ext == ".txt" or ct.startswith("text/"):                       return "text"
    return "other"


class _HTMLToText(HTMLParser):
    """Turns an HTML-only email body into readable text (keeps table cells apart)."""
    def __init__(self):
        super().__init__(); self.parts, self.skip = [], 0
    def handle_starttag(self, tag, attrs):
        if tag in {"script", "style", "head"}: self.skip += 1
        elif tag in {"p", "div", "br", "tr", "li", "table", "hr"}: self.parts.append("\n")
        elif tag in {"td", "th"}: self.parts.append("\t")
    def handle_endtag(self, tag):
        if tag in {"script", "style", "head"} and self.skip: self.skip -= 1
    def handle_data(self, data):
        if not self.skip: self.parts.append(data)


def html_to_text(html: str) -> str:
    p = _HTMLToText(); p.feed(html)
    text = "".join(p.parts).replace("\xa0", " ")
    return re.sub(r"\n{3,}", "\n\n", re.sub(r"[ \t]+\n", "\n", text)).strip()


def save_attachment(folder: Path, name: str, data: bytes, ctype: str,
                    inline: bool, nested_from) -> Attachment:
    folder.mkdir(parents=True, exist_ok=True)
    name = safe_name(name)
    target, n = folder / name, 1
    while target.exists():                                   # same name twice
        target = folder / f"{Path(name).stem}_{n}{Path(name).suffix}"; n += 1
    target.write_bytes(data)
    kind = kind_of(name, ctype)
    return Attachment(
        filename=name, content_type=ctype or "application/octet-stream", kind=kind,
        size_bytes=len(data), saved_path=str(target.relative_to(ROOT)) if target.is_relative_to(ROOT) else str(target),
        is_inline=inline,
        likely_signature=(kind == "image" and inline and len(data) < SIGNATURE_MAX_BYTES),
        from_nested_email=nested_from,
    )


# ---------- .eml ----------
def eml_attachments(msg, folder, nested_from=None, depth=0) -> list[Attachment]:
    out = []
    for i, part in enumerate(msg.iter_attachments(), 1):
        ctype = part.get_content_type()
        inline = part.get_content_disposition() == "inline" or bool(part.get("Content-ID"))
        name = part.get_filename()

        if ctype == "message/rfc822":                        # forwarded email
            inner = part.get_content()
            name = name or f"{safe_name(str(inner.get('Subject', 'attached_email')))}.eml"
            out.append(save_attachment(folder, name, inner.as_bytes(), ctype, False, nested_from))
            if depth < MAX_NESTING:                          # pull files out of it too
                out += eml_attachments(inner, folder, name, depth + 1)
            continue

        data = part.get_payload(decode=True) or b""
        name = name or f"attachment_{i}{mimetypes.guess_extension(ctype) or ''}"
        out.append(save_attachment(folder, name, data, ctype, inline, nested_from))
    return out


def read_eml(path: Path, case: Case, folder: Path) -> None:
    msg = BytesParser(policy=policy.default).parsebytes(path.read_bytes())
    case.subject = str(msg.get("Subject", ""))
    case.from_name, case.from_address = parseaddr(str(msg.get("From", "")))
    case.to, case.cc = str(msg.get("To", "")), str(msg.get("Cc", ""))
    case.in_reply_to = str(msg.get("In-Reply-To", ""))
    if msg.get("Date"):
        try: case.received_at = parsedate_to_datetime(str(msg["Date"]))
        except (TypeError, ValueError): case.notes.append("Could not parse Date")

    plain = msg.get_body(preferencelist=("plain",))
    html = msg.get_body(preferencelist=("html",))
    case.body_text = plain.get_content() if plain else (html_to_text(html.get_content()) if html else "")
    case.attachments = eml_attachments(msg, folder)


# ---------- .msg (Outlook) ----------
def read_msg(path: Path, case: Case, folder: Path) -> None:
    import extract_msg                                       # only needed for .msg
    msg = extract_msg.openMsg(str(path))
    try:
        case.subject = msg.subject or ""
        case.from_name, case.from_address = parseaddr(msg.sender or "")
        case.to, case.cc = msg.to or "", msg.cc or ""
        d = msg.date
        case.received_at = d if isinstance(d, datetime) else None
        html = msg.htmlBody or b""
        html = html.decode("utf-8", "replace") if isinstance(html, bytes) else str(html)
        case.body_text = (msg.body or "").strip() or html_to_text(html)

        for i, att in enumerate(msg.attachments, 1):
            name = att.longFilename or att.shortFilename or f"attachment_{i}"
            data = att.data
            inline = bool(getattr(att, "cid", None))
            if isinstance(data, (bytes, bytearray)):
                ctype = mimetypes.guess_type(name)[0] or ""
                case.attachments.append(save_attachment(folder, name, bytes(data), ctype, inline, None))
            else:                                            # embedded Outlook message
                case.notes.append(f"Embedded Outlook message '{name}' not expanded yet")
    finally:
        msg.close()


# ---------- main step ----------
def run(inbox: Path, out_dir: Path) -> list[Case]:
    # "PO 123 (1).eml" sorts after "PO 123.eml" so the original wins the duplicate check
    files = sorted((p for p in inbox.iterdir() if p.suffix.lower() in {".eml", ".msg"}),
                   key=lambda p: (re.sub(r" \(\d+\)$", "", p.stem).lower(), len(p.name)))
    cases, seen = [], {}

    for path in files:
        raw = path.read_bytes()
        digest = sha256(raw)
        case = Case(case_id=f"CASE-{digest[:10]}", source_file=path.name,
                    source_path=str(path.resolve()), source_sha256=digest)

        if digest in seen:                                   # identical file already read
            case.case_id += "-DUP"
            case.status, case.duplicate_of = "duplicate_file", seen[digest]
        else:
            seen[digest] = case.case_id
            shutil.rmtree(out_dir / case.case_id, ignore_errors=True)   # re-run = start this case clean
            try:
                reader = read_eml if path.suffix.lower() == ".eml" else read_msg
                reader(path, case, out_dir / case.case_id / "attachments")
                case.status = "read"
            except Exception as e:                           # one bad file must not stop the run
                case.status = "manual_queue"
                case.notes.append(f"A1 could not read file: {e}")

        case_dir = out_dir / case.case_id
        case_dir.mkdir(parents=True, exist_ok=True)
        (case_dir / "case.json").write_text(case.model_dump_json(indent=2), encoding="utf-8")
        with (out_dir / "audit.jsonl").open("a", encoding="utf-8") as f:
            f.write(json.dumps({"ts": datetime.now(timezone.utc).isoformat(), "case_id": case.case_id,
                                "step": "A1", "event": case.status, "file": path.name}) + "\n")
        usable = [a for a in case.attachments if not a.likely_signature]
        print(f"{case.status:<14} {path.name[:55]:<55} {case.case_id}  "
              f"{len(usable)} attachment(s)" + (f"  (same as {case.duplicate_of})" if case.duplicate_of else ""))
        cases.append(case)
    return cases


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="A1 - read emails from a folder")
    ap.add_argument("--inbox", default=str(ROOT / "data" / "samples"))
    ap.add_argument("--out", default=str(ROOT / "output"))
    a = ap.parse_args()
    cases = run(Path(a.inbox), Path(a.out))
    print(f"\n{len(cases)} files: {sum(c.status == 'read' for c in cases)} read, "
          f"{sum(c.status == 'duplicate_file' for c in cases)} duplicate, "
          f"{sum(c.status == 'manual_queue' for c in cases)} failed")
    