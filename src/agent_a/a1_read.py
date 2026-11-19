"""
A1 - Pick up the case (Salesforce-shaped).

Today:  reads .eml / .msg files from a folder and builds the records that
        Salesforce Email-to-Case would create: Case, EmailMessage and
        CombinedAttachments, using Salesforce's real field names.
Later:  the same records come from the Salesforce REST API instead.

Output per email:
    <out>/CASE-<CaseNumber>/
        salesforce/case.json            <- SFCase          (never edited)
        salesforce/email_message.json   <- SFEmailMessage  (never edited)
        salesforce/attachments.json     <- [SFAttachment]  (never edited)
        attachments/                    <- the files
        case.json                       <- pipeline Case   (A2, A3... add to it)

Run from src/:
    python -m agent_a.a1_read --inbox ..\\data\\samples --out ..\\output
    python -m agent_a.a1_read --inbox ..\\data\\inbox --out ..\\output_real --accounts ..\\data\\private\\accounts.json
"""
import argparse
import hashlib
import json
import mimetypes
import re
import shutil
from dataclasses import dataclass, field
from datetime import datetime, timezone
from email import policy
from email.parser import BytesParser
from email.utils import getaddresses, parseaddr, parsedate_to_datetime
from html.parser import HTMLParser
from pathlib import Path
from typing import Optional

from .models import Attachment, Case
from .sf_adapter import from_salesforce
from .sf_models import (MAX_DESCRIPTION, MAX_TEXT_BODY, SFAccount, SFAttachment,
                        SFCase, SFContact, SFEmailMessage, fake_sf_id, file_type_for)

ROOT = Path(__file__).resolve().parents[2]
SIGNATURE_MAX_BYTES = 30_000       # inline images smaller than this = logos
MAX_NESTING = 2                    # forwarded email inside forwarded email
FIRST_CASE_NUMBER = 100_001        # -> "00100001"
MAX_SUBJECT = 255                  # Case.Subject limit in Salesforce
MAX_HEADERS = 32_000               # EmailMessage.Headers limit


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
        filename=target.name, content_type=ctype or "application/octet-stream", kind=kind,
        size_bytes=len(data),
        saved_path=(target.resolve().relative_to(ROOT).as_posix() if target.resolve().is_relative_to(ROOT)
                    else target.resolve().as_posix()),
        is_inline=inline,
        likely_signature=(kind == "image" and inline and len(data) < SIGNATURE_MAX_BYTES),
        from_nested_email=nested_from,
    )


# ---------- reading the email file (format-neutral result) ----------
@dataclass
class ParsedEmail:
    subject: str = ""
    from_name: str = ""
    from_address: str = ""
    to: str = ""
    cc: str = ""
    date: Optional[datetime] = None
    message_id: str = ""
    in_reply_to: str = ""
    text: str = ""
    html: str = ""
    headers: str = ""
    attachments: list[Attachment] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)


def eml_attachments(msg, folder, nested_from=None, depth=0) -> list[Attachment]:
    out = []
    for i, part in enumerate(msg.iter_attachments(), 1):
        ctype = part.get_content_type()
        inline = part.get_content_disposition() == "inline" or bool(part.get("Content-ID"))
        name = part.get_filename()

        if ctype == "message/rfc822":                        # forwarded email
            inner = part.get_content()
            name = name or f"{safe_name(str(inner.get('Subject', 'attached_email')))}.eml"
            saved = save_attachment(folder, name, inner.as_bytes(), ctype, False, nested_from)
            out.append(saved)
            if depth < MAX_NESTING:                          # pull files out of it too
                if not inner.is_multipart() and inner.get_content_maintype() not in ("text", "multipart"):
                    # The forwarded email IS a file (e.g. SAP sends a mail whose whole body is the PO PDF)
                    data = inner.get_payload(decode=True) or b""
                    fname = inner.get_filename() or f"{Path(saved.filename).stem}{mimetypes.guess_extension(inner.get_content_type()) or ''}"
                    out.append(save_attachment(folder, fname, data, inner.get_content_type(), False, saved.filename))
                else:
                    out += eml_attachments(inner, folder, saved.filename, depth + 1)
            continue

        data = part.get_payload(decode=True) or b""
        name = name or f"attachment_{i}{mimetypes.guess_extension(ctype) or ''}"
        out.append(save_attachment(folder, name, data, ctype, inline, nested_from))
    return out


def read_eml(path: Path, folder: Path) -> ParsedEmail:
    msg = BytesParser(policy=policy.default).parsebytes(path.read_bytes())
    p = ParsedEmail(subject=str(msg.get("Subject", "")),
                    to=str(msg.get("To", "")), cc=str(msg.get("Cc", "")),
                    message_id=str(msg.get("Message-ID", "")),
                    in_reply_to=str(msg.get("In-Reply-To", "")),
                    headers="\n".join(f"{k}: {v}" for k, v in msg.items()))
    p.from_name, p.from_address = parseaddr(str(msg.get("From", "")))
    if msg.get("Date"):
        try: p.date = parsedate_to_datetime(str(msg["Date"]))
        except (TypeError, ValueError): p.notes.append("Could not parse Date")

    plain = msg.get_body(preferencelist=("plain",))
    html = msg.get_body(preferencelist=("html",))
    p.html = html.get_content() if html else ""
    p.text = plain.get_content() if plain else html_to_text(p.html)
    p.attachments = eml_attachments(msg, folder)
    return p


def read_msg(path: Path, folder: Path) -> ParsedEmail:
    import extract_msg                                       # only needed for .msg
    msg = extract_msg.openMsg(str(path))
    try:
        p = ParsedEmail(subject=msg.subject or "", to=msg.to or "", cc=msg.cc or "",
                        message_id=getattr(msg, "messageId", "") or "",
                        in_reply_to=getattr(msg, "inReplyTo", "") or "",
                        headers=str(getattr(msg, "header", "") or ""))
        p.from_name, p.from_address = parseaddr(msg.sender or "")
        p.date = msg.date if isinstance(msg.date, datetime) else None
        html = msg.htmlBody or b""
        p.html = html.decode("utf-8", "replace") if isinstance(html, bytes) else str(html)
        p.text = (msg.body or "").strip() or html_to_text(p.html)

        for i, att in enumerate(msg.attachments, 1):
            name = att.longFilename or att.shortFilename or f"attachment_{i}"
            data = att.data
            inline = bool(getattr(att, "cid", None))
            if isinstance(data, (bytes, bytearray)):
                ctype = mimetypes.guess_type(name)[0] or ""
                p.attachments.append(save_attachment(folder, name, bytes(data), ctype, inline, None))
            else:                                            # embedded Outlook message
                p.notes.append(f"Embedded Outlook message '{name}' not expanded yet")
        return p
    finally:
        msg.close()


# ---------- Salesforce simulation ----------
def load_accounts(path: Path) -> dict[str, tuple[SFAccount, SFContact]]:
    """email (lower case) -> (Account, Contact). Missing file = no matches."""
    if not path.exists():
        print(f"NOTE: {path} not found - every case will be 'unknown_sender'\n")
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    lookup = {}
    for raw in data.get("accounts", []):
        acc = SFAccount(**raw)
        for c in acc.Contacts:
            lookup[c.Email.strip().lower()] = (acc, c)
    return lookup


class CaseNumbers:
    """Gives each email a stable 8-digit CaseNumber (same file -> same number on re-runs)."""
    def __init__(self, path: Path):
        self.path = path
        self.map = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}

    def get(self, key: str) -> str:
        if key not in self.map:
            nxt = max([int(v) for v in self.map.values()] + [FIRST_CASE_NUMBER - 1]) + 1
            self.map[key] = f"{nxt:08d}"
        return self.map[key]

    def save(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(self.map, indent=2), encoding="utf-8")


def build_sf_records(key: str, case_number: str, p: ParsedEmail,
                     match: Optional[tuple[SFAccount, SFContact]]):
    """What Email-to-Case would store for this email."""
    case_id = fake_sf_id("Case", key)
    email_id = fake_sf_id("EmailMessage", key)
    created = p.date or datetime.now(timezone.utc)

    # CombinedAttachments: only the files attached to the email itself.
    # Files we pulled out of a forwarded .eml are ours, not Salesforce's,
    # so they point to the Id of that .eml attachment.
    sf_atts, parent_id = [], None
    for i, a in enumerate(p.attachments):
        if a.from_nested_email is None:
            sf = SFAttachment(Id=fake_sf_id("ContentDocument", f"{key}:{i}"), ParentId=case_id,
                              Title=Path(a.filename).stem,
                              FileExtension=Path(a.filename).suffix.lstrip(".").lower() or None,
                              FileType=file_type_for(Path(a.filename).suffix),
                              ContentSize=a.size_bytes, CreatedDate=created)
            sf_atts.append(sf)
            a.sf_id = sf.Id
            if a.kind == "email":
                parent_id = sf.Id
        else:
            a.sf_id = parent_id

    to_addresses = [addr for _, addr in getaddresses([p.to]) if addr]
    email = SFEmailMessage(
        Id=email_id, ParentId=case_id, Subject=p.subject,
        FromAddress=p.from_address, FromName=p.from_name,
        ToAddress=p.to or None, CcAddress=p.cc or None,
        TextBody=p.text[:MAX_TEXT_BODY], HtmlBody=(p.html[:MAX_TEXT_BODY] or None),
        Headers=p.headers[:MAX_HEADERS] or None,
        MessageDate=p.date, MessageIdentifier=p.message_id or None,
        ThreadIdentifier=p.in_reply_to or None,
        HasAttachment=bool(sf_atts), CreatedDate=created,
    )

    acc, con = match if match else (None, None)
    case = SFCase(
        Id=case_id, CaseNumber=case_number,
        Subject=p.subject[:MAX_SUBJECT],
        Description=p.text[:MAX_DESCRIPTION],
        Email_Content_Truncated__c=len(p.text) > MAX_DESCRIPTION,
        CreatedDate=created, SourceId=email_id,
        SuppliedName=p.from_name or None, SuppliedEmail=p.from_address or None,
        Email_To_Case_Address__c=to_addresses[0] if to_addresses else None,
        OwnerId=None,                                        # queue not decided yet
        AccountId=acc.Id if acc else None,
        Account_Name__c=acc.Name if acc else None,
        SAP_Account_Number__c=acc.SAP_Account_Number__c if acc else None,
        BOT_Customer__c=acc.Is_BOT_Customer__c if acc else False,
        ContactId=con.Id if con else None,
        ContactEmail=con.Email if con else None,
        ContactPhone=con.Phone if con else None,
    )
    return case, email, sf_atts


def write_json(path: Path, data) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


# ---------- main step ----------
def run(inbox: Path, out_dir: Path, accounts_file: Path) -> list[Case]:
    # "PO 123 (1).eml" sorts after "PO 123.eml" so the original wins the duplicate check
    files = sorted((p for p in inbox.iterdir() if p.suffix.lower() in {".eml", ".msg"}),
                   key=lambda p: (re.sub(r" \(\d+\)$", "", p.stem).lower(), len(p.name)))
    accounts = load_accounts(accounts_file)
    numbers = CaseNumbers(out_dir / "_case_numbers.json")
    cases, seen = [], {}

    for path in files:
        raw = path.read_bytes()
        digest = sha256(raw)
        key = f"{digest}|{path.name}"                        # identical copies = separate cases, like Salesforce
        case_number = numbers.get(key)
        case_dir = out_dir / f"CASE-{case_number}"
        shutil.rmtree(case_dir, ignore_errors=True)          # re-run = start this case clean

        try:
            reader = read_eml if path.suffix.lower() == ".eml" else read_msg
            parsed = reader(path, case_dir / "attachments")
            match = accounts.get(parsed.from_address.strip().lower())
            sf_case, sf_email, sf_atts = build_sf_records(key, case_number, parsed, match)

            write_json(case_dir / "salesforce" / "case.json", sf_case.model_dump(mode="json"))
            write_json(case_dir / "salesforce" / "email_message.json", sf_email.model_dump(mode="json"))
            write_json(case_dir / "salesforce" / "attachments.json", [a.model_dump(mode="json") for a in sf_atts])

            case = from_salesforce(sf_case, sf_email, sf_atts, parsed.attachments,
                                   source_file=path.name, source_path=str(path.resolve()),
                                   source_sha256=digest, contact_name=match[1].Name if match else "")
            case.notes += parsed.notes
        except Exception as e:                               # one bad file must not stop the run
            case = Case(case_id=f"CASE-{case_number}", case_number=case_number, source_file=path.name,
                        source_path=str(path.resolve()), source_sha256=digest, status="manual_queue",
                        notes=[f"A1 could not read file: {e}"])

        if digest in seen and case.status == "read":         # identical file already read
            case.status, case.duplicate_of = "duplicate_file", seen[digest]
        else:
            seen.setdefault(digest, case.case_id)

        write_json(case_dir / "case.json", json.loads(case.model_dump_json()))
        with (out_dir / "audit.jsonl").open("a", encoding="utf-8") as f:
            f.write(json.dumps({"ts": datetime.now(timezone.utc).isoformat(), "case_id": case.case_id,
                                "step": "A1", "event": case.status, "file": path.name}) + "\n")
        usable = [a for a in case.attachments if not a.likely_signature]
        who = case.account_name or "unknown sender"
        print(f"{case.status:<14} {case.case_id}  {path.name[:45]:<45} {who[:28]:<28} "
              f"{len(usable)} att" + (f"  (same as {case.duplicate_of})" if case.duplicate_of else ""))
        cases.append(case)

    numbers.save()
    return cases


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="A1 - build Salesforce-shaped cases from email files")
    ap.add_argument("--inbox", default=str(ROOT / "data" / "samples"))
    ap.add_argument("--out", default=str(ROOT / "output"))
    ap.add_argument("--accounts", default=str(ROOT / "data" / "mock_accounts.json"),
                    help="customer lookup (real customers: data/private/accounts.json)")
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    cases = run(Path(a.inbox), out, Path(a.accounts))
    print(f"\n{len(cases)} files: {sum(c.status == 'read' for c in cases)} read, "
          f"{sum(c.status == 'duplicate_file' for c in cases)} duplicate, "
          f"{sum(c.status == 'manual_queue' for c in cases)} failed, "
          f"{sum('unknown_sender' in c.flags for c in cases)} unknown sender")