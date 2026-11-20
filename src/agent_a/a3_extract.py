"""
A3 - Read the PO.

For every case A2 marked "classified":
  1. pick the documents (attachments; the email body if there are none)
  2. read each one into plain text (text_extract) -> extracted/<file>.txt   = the ground truth
  3+4. ask the LLM: what kind of document is this, and if it is a PO, what are the fields
       (every field: value + the exact quote it came from + the model's confidence)
  5. check every field against the text (code): quote really there? value in the quote? format OK? OCR?
  6. cross-check numbers: qty x price / per = line total, sum of lines = PO total
  7. per-field confidence, missing fields, overall confidence, flags
  8. route: >=1 PO -> "extracted" (A4)   |   no PO / nothing readable -> "manual_queue"

Business validation (customer, ship-to, material, price, contract) is NOT done here - SAP does it in A4/A5.

Run from src/:
    python -m agent_a.a3_extract --out ..\\output --check ..\\data\\answer_keys
    python -m agent_a.a3_extract --out ..\\output_real
"""
import argparse
import json
import os
import re
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal, Optional

from pydantic import BaseModel

from . import llm
from .a2_classify import norm_po
from .clean import clean_email_text
from .config import CFG, ROOT
from .constants import (A3_SYSTEM_PROMPT, DATE_FIELDS, DOC_KINDS, KNOWN_UOM, LB_PER, MONTHS, NUM_FIELDS,
                        OCR_DIGIT_FIX)
from .models import Case, DocumentText, FieldValue, LineItem, PurchaseOrder, Source
from .text_extract import ExtractResult, extract

SAMPLES = (ROOT / CFG.paths.samples).resolve()
MAX_DOC_CHARS = CFG.a3.max_doc_chars          # text sent to the model per document (POs are at the start; T&Cs later)
OCR_LOW = CFG.a3.ocr_low
NO_QUOTE_CAP = CFG.a3.no_quote_cap
SUPPLIER_NAMES = CFG.a3.supplier_names

# ======================= what the model returns =======================
# Every value is a string ("" = not on the document); code turns numbers and dates into real values.
class XField(BaseModel):
    value: str
    quote: str          # exact text from the document that contains the value
    page: int           # from the "=== Page N ===" markers, 1 if none
    confidence: float   # the model's own confidence 0-1


class XLine(BaseModel):
    line_no: XField
    customer_material: XField
    supplier_material: XField
    description: XField
    quantity: XField
    uom: XField
    unit_price: XField
    price_per: XField
    line_total: XField
    delivery_date: XField


class XPO(BaseModel):
    po_number: XField
    po_date: XField
    requested_delivery_date: XField
    customer_name: XField
    buyer_name: XField
    buyer_phone: XField
    buyer_email: XField
    vendor_name: XField
    ship_to_name: XField
    ship_to_address: XField
    bill_to: XField
    currency: XField
    incoterms: XField
    payment_terms: XField
    notes: XField
    total: XField
    lines: list[XLine]


class XDoc(BaseModel):
    doc_type: Literal["purchase_order", "order_acknowledgement_form", "coa", "invoice", "bol",
                      "shipment_notice", "terms", "quote", "other"]
    purchase_orders: list[XPO]      # empty unless the document contains a purchase order

SYSTEM = A3_SYSTEM_PROMPT.format(supplier_names=SUPPLIER_NAMES)

def norm(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "")).strip().lower()

def nospace(s: str) -> str:
    return re.sub(r"\s+", "", (s or "")).lower()


def to_number(s) -> Optional[float]:
    if s is None or s == "":
        return None
    t = re.sub(r"[,$\s]|USD", "", str(s))
    try:
        return float(t)
    except ValueError:
        return None


def numbers_in(text: str) -> list[float]:
    out = []
    for m in re.finditer(r"\d[\d,]*(?:\.\d+)?|\.\d+", text or ""):
        n = to_number(m.group(0))
        if n is not None:
            out.append(n)
    return out


def ocr_fix_numbers(text: str) -> str:
    """'44OOO' -> '44000': fix look-alike letters only inside tokens that are mostly digits."""
    def fix(m):
        tok = m.group(0)
        digits = sum(ch.isdigit() for ch in tok)
        return tok.translate(OCR_DIGIT_FIX) if digits and digits >= len(tok) / 2 else tok
    return re.sub(r"[0-9OoQDlIi|SsBZzG.,]{2,}", fix, text or "")


def ocr_fix_uom(s: str) -> str:
    """OCR misreads in units: '1B' / 'IB' / 'L8' -> 'LB', 'BAG5OLB' -> 'BAG50LB'."""
    t = re.sub(r"(?<![A-Za-z])[1Il|]B\b", "LB", s or "")
    t = re.sub(r"\bL8\b", "LB", t)
    t = re.sub(r"(?<=\d)[Oo]|[Oo](?=\d)", "0", t)
    return t


def parse_iso(s: str) -> Optional[str]:
    try:
        return datetime.strptime((s or "").strip(), "%Y-%m-%d").date().isoformat()
    except ValueError:
        return None


def dates_in(text: str) -> list[str]:
    """Every date we can recognise in a piece of text, as YYYY-MM-DD, in order.
    03/08/2026 gives the US reading (March 8) first, then the day-first one."""
    t, found = text or "", []

    def add(y, m, d):
        try:
            y = int(y); y = y + 2000 if y < 100 else y
            iso = datetime(y, int(m), int(d)).date().isoformat()
            if iso not in found:
                found.append(iso)
        except ValueError:
            pass
    for y, m, d in re.findall(r"\b(\d{4})[-/.](\d{1,2})[-/.](\d{1,2})\b", t):
        add(y, m, d)
    for m, d, y in re.findall(r"\b(\d{1,2})[-/.](\d{1,2})[-/.](\d{2,4})\b", t):
        add(y, m, d)                                     # US style first (customers are US)
        add(y, d, m)                                     # and day-first, in case
    for mon, d, y in re.findall(r"\b([A-Za-z]{3})[a-z]*\.?\s+(\d{1,2}),?\s+(\d{4})\b", t):
        if mon.lower() in MONTHS: add(y, MONTHS[mon.lower()], d)
    for d, mon, y in re.findall(r"\b(\d{1,2})[\s-]+([A-Za-z]{3})[a-z]*\.?[\s-]+(\d{2,4})\b", t):
        if mon.lower() in MONTHS: add(y, MONTHS[mon.lower()], d)
    for y, m, d in re.findall(r"\b(20\d{2})(\d{2})(\d{2})\b", t):
        add(y, m, d)
    return found


def uom_ok(u: str) -> bool:
    head = re.match(r"[A-Za-z]+", (u or "").strip())
    return bool(head) and head.group(0).upper() in KNOWN_UOM


# ======================= step 5: check one field =======================
class DocCtx:
    """Everything about one document that the checks need."""
    def __init__(self, filename: str, method: str, text: str, ocr_words: list[tuple[str, float]]):
        self.filename, self.method, self.text = filename, method, text
        self.ocr_words = ocr_words
        self.n_text, self.ns_text = norm(text), nospace(text)
        self.n_fixed = norm(ocr_fix_numbers(text)) if method == "ocr" else self.n_text

    def quote_found(self, quote: str) -> bool:
        q = norm(quote)
        return bool(q) and (q in self.n_text or nospace(q) in self.ns_text
                            or (self.method == "ocr" and norm(ocr_fix_numbers(quote)) in self.n_fixed))

    def ocr_conf(self, quote: str) -> Optional[float]:
        """Average Tesseract confidence of the words in the quote (None = not OCR)."""
        if self.method != "ocr" or not self.ocr_words:
            return None
        wanted = {w.lower() for w in re.findall(r"\S+", quote or "")}
        confs = [c for w, c in self.ocr_words if w.lower() in wanted]
        if not confs:
            confs = [c for _, c in self.ocr_words]
        return sum(confs) / len(confs)


def check_field(name: str, x: XField, doc: DocCtx) -> FieldValue:
    raw = (x.value or "").strip()
    src = Source(file=doc.filename, page=x.page or 1, quote=(x.quote or "").strip())
    if not raw:
        return FieldValue(value=None, confidence=0.0, source=src)

    conf = max(0.0, min(1.0, x.confidence))
    checks: list[str] = []
    value: object = raw

    # c) format, and turn into a real number / date
    if name in NUM_FIELDS:
        n = to_number(raw if doc.method != "ocr" else ocr_fix_numbers(raw))
        if n is None:
            checks.append("bad_format"); conf *= 0.5
        else:
            value = n
    elif name in DATE_FIELDS:
        iso = parse_iso(raw) or next(iter(dates_in(raw)), None)      # first = US reading
        if iso is None:
            checks.append("bad_format"); conf *= 0.5
        else:
            value = iso
    elif name == "uom":
        if doc.method == "ocr" and ocr_fix_uom(raw) != raw:
            value = ocr_fix_uom(raw); checks.append("ocr_corrected")
        if not uom_ok(str(value)):
            checks.append("unknown_uom"); conf *= 0.7

    # a) is the quote really in the document?
    if not src.quote:
        checks.append("no_quote"); conf = min(conf, NO_QUOTE_CAP)
    elif not doc.quote_found(src.quote):
        checks.append("quote_not_found"); conf = min(conf, NO_QUOTE_CAP)
    else:
        # b) is the value really in its quote?
        if name in NUM_FIELDS and isinstance(value, float):
            if not any(abs(n - value) < 0.005 for n in numbers_in(src.quote)):
                fixed = numbers_in(ocr_fix_numbers(src.quote)) if doc.method == "ocr" else []
                if any(abs(n - value) < 0.005 for n in fixed):
                    checks.append("ocr_corrected"); conf *= 0.9          # e) 44OOO -> 44000
                else:
                    checks.append("value_not_in_quote"); conf *= 0.5
        elif name in DATE_FIELDS and isinstance(value, str):
            if value not in dates_in(src.quote):
                checks.append("value_not_in_quote"); conf *= 0.5
        else:
            ok = nospace(raw) in nospace(src.quote)
            if not ok and doc.method == "ocr":
                fix = ocr_fix_uom if name == "uom" else ocr_fix_numbers
                ok = nospace(fix(raw)) in nospace(fix(src.quote))
                if ok and "ocr_corrected" not in checks: checks.append("ocr_corrected")
            if not ok:                                   # long text (addresses): most words present?
                words = re.findall(r"\w+", raw.lower())
                hit = sum(w in norm(src.quote) for w in words) / max(1, len(words))
                if hit < 0.8:
                    checks.append("value_not_in_quote"); conf *= 0.5

    # d) text came from OCR -> trust it only as much as Tesseract did
    oc = doc.ocr_conf(src.quote)
    if oc is not None:
        conf *= oc
        if oc < OCR_LOW:
            checks.append("ocr_low")

    return FieldValue(value=value, confidence=round(conf, 3), source=src, checks=checks)

def weight_lb(qty: float, uom: str, description: str) -> Optional[float]:
    """Quantity in pounds, if we can tell. 400 BAG50LB -> 20000 LB."""
    u = (uom or "").upper().replace(" ", "")
    if u in LB_PER:
        return qty * LB_PER[u]
    m = re.search(r"(\d+(?:\.\d+)?)(LB|KG)", u) or re.search(r"(\d+(?:\.\d+)?)\s*(LB|KG)\b", (description or "").upper())
    if m and re.match(r"(BAG|SACK|TOTE|DRUM|PAIL|BB|SUPERSACK|BOX|CS|CASE)", u):
        return qty * float(m.group(1)) * LB_PER[m.group(2)]
    return None


def per_lb(price_per: str) -> Optional[float]:
    """'100 LB' -> 100, 'CWT' -> 100, 'LB' -> 1, 'KG' -> 2.20462 (pounds per pricing unit)."""
    p = (price_per or "").upper().replace("/", " ").strip()
    m = re.fullmatch(r"(\d+(?:\.\d+)?)?\s*(LBS?|CWT|KG|MT|TON|G)", p)
    if not m:
        return None
    return float(m.group(1) or 1) * LB_PER[m.group(2) if m.group(2) != "LBS" else "LB"]


def line_expected_total(line: LineItem) -> Optional[float]:
    q, p = line.quantity.value, line.unit_price.value
    if not isinstance(q, float) or not isinstance(p, float):
        return None
    uom = str(line.uom.value or "")
    per = str(line.price_per.value or "")
    if not per or per.upper().replace(" ", "") == uom.upper().replace(" ", "") or per.upper() in ("EA", "EACH", "UNIT"):
        return q * p                                     # priced per ordered unit
    lbs, unit_lb = weight_lb(q, uom, str(line.description.value or "")), per_lb(per)
    if lbs is not None and unit_lb:
        return lbs / unit_lb * p
    return None


def raise_conf(fv: FieldValue, note: str):
    if fv.value is not None and "quote_not_found" not in fv.checks and "no_quote" not in fv.checks:
        fv.confidence = round(max(fv.confidence, 0.9), 3)
        fv.checks.append(note)


def lower_conf(fv: FieldValue, note: str):
    if fv.value is not None:
        fv.confidence = round(fv.confidence * 0.7, 3)
        fv.checks.append(note)


def cross_check(po: PurchaseOrder) -> None:
    sum_lines, all_totals = 0.0, True
    for ln in po.lines:
        if isinstance(ln.unit_price.value, float) and ln.unit_price.value == 0:
            if "zero_price" not in po.flags:
                po.flags.append("zero_price")              # contract release: SAP prices it, reviewer sees it
        expected = line_expected_total(ln)
        total = ln.line_total.value
        if isinstance(total, float):
            sum_lines += total
        else:
            all_totals = False
        if expected is None or not isinstance(total, float):
            continue
        if abs(expected - total) <= max(0.01 * abs(total), 1.0):
            # the sum only works out with these values - so the unit and pricing unit are confirmed too
            for fv in (ln.quantity, ln.unit_price, ln.line_total, ln.uom, ln.price_per):
                raise_conf(fv, "math_confirmed")
        else:
            for fv in (ln.quantity, ln.unit_price, ln.line_total):
                lower_conf(fv, "math_mismatch")
            if "math_mismatch" not in po.flags:
                po.flags.append("math_mismatch")
    if po.lines and all_totals and isinstance(po.total.value, float):
        if abs(sum_lines - po.total.value) <= max(0.01 * abs(po.total.value), 1.0):
            raise_conf(po.total, "sum_confirmed")
        else:
            lower_conf(po.total, "sum_mismatch")
            po.flags.append("total_mismatch")


# ======================= step 7: summary =======================
def summarise(po: PurchaseOrder, a2_pos: list[str], doc: DocCtx, ocr_conf: Optional[float]) -> None:
    key: dict[str, FieldValue] = {"po_number": po.po_number, "po_date": po.po_date,
                                  "ship_to_address": po.ship_to_address}
    line_dates = [ln.delivery_date for ln in po.lines if ln.delivery_date.value]
    if po.requested_delivery_date.value or not line_dates:
        key["requested_delivery_date"] = po.requested_delivery_date
    else:
        key["requested_delivery_date"] = min(line_dates, key=lambda f: f.confidence)
    for i, ln in enumerate(po.lines, 1):
        mats = [f for f in (ln.customer_material, ln.supplier_material, ln.description) if f.value]
        key[f"lines[{i}].material"] = max(mats, key=lambda f: f.confidence) if mats else ln.customer_material
        key[f"lines[{i}].quantity"] = ln.quantity
        key[f"lines[{i}].uom"] = ln.uom

    po.missing_fields = [k for k, f in key.items() if f.value is None]
    if not po.lines:
        po.missing_fields.append("lines")
    po.overall_confidence = 0.0 if not po.lines else round(min(f.confidence for f in key.values()), 3)

    if ocr_conf is not None and ocr_conf < OCR_LOW:
        po.flags.append("low_ocr_confidence")
    if po.po_number.value and a2_pos:
        mine = norm_po(str(po.po_number.value))
        if not any(mine == p or (len(p) >= 4 and (p in mine or mine in p)) for p in a2_pos):
            po.flags.append("po_number_not_seen_by_a2")      # "31877" in the email = "GVS-31877" on the PO
    if any("quote_not_found" in f.checks or "no_quote" in f.checks for f in key.values()):
        po.flags.append("unverified_values")
    ours = [w.strip().split()[0].lower() for w in SUPPLIER_NAMES.split(",") if w.strip()]
    if po.vendor_name.value and not any(o in str(po.vendor_name.value).lower() for o in ours):
        po.flags.append("vendor_not_us")          # PO may be meant for another supplier - reviewer / SAP decide


# ======================= steps 1-8 for one case =======================
def build_po(x: XPO, doc: DocCtx, a2_pos: list[str], ocr_conf: Optional[float],
             email_date: Optional[datetime] = None) -> PurchaseOrder:
    head = {k: check_field(k, getattr(x, k), doc) for k in XPO.model_fields if k != "lines"}
    if doc.method == "email_body" and head["po_date"].value is None and email_date:
        # an order written in an email is dated when the email was sent
        head["po_date"] = FieldValue(value=email_date.date().isoformat(), confidence=0.8,
                                     source=Source(file="email_body", quote="email sent date"),
                                     checks=["from_email_date"])
    lines = [LineItem(**{k: check_field(k, getattr(xl, k), doc) for k in XLine.model_fields}) for xl in x.lines]
    lines = [ln for ln in lines if any(getattr(ln, k).value is not None
                                       for k in ("description", "customer_material", "quantity"))]
    po = PurchaseOrder(source_file=doc.filename, read_method=doc.method, lines=lines, **head)
    cross_check(po)
    summarise(po, a2_pos, doc, ocr_conf)
    return po


def find_file(case_dir: Path, a) -> Path:
    """The attachment's file. Its own case folder first - works whatever --out path A1 was given."""
    here = case_dir / "attachments" / a.filename
    if here.exists():
        return here
    p = Path(a.saved_path)
    for cand in (p, ROOT / p, Path.cwd() / p):
        if cand.exists():
            return cand
    return here


def pick_documents(case: Case) -> list:
    return [a for a in case.attachments if not a.likely_signature and a.kind in DOC_KINDS]


def write_text(case_dir: Path, name: str, text: str) -> str:
    p = case_dir / "extracted" / f"{name}.txt"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(text, encoding="utf-8")
    full = p.resolve()                                   # works for relative --out paths like ..\output
    return full.relative_to(ROOT).as_posix() if full.is_relative_to(ROOT) else full.as_posix()


def read_with_model(doc: DocCtx) -> tuple[XDoc, list[str]]:
    text = doc.text if len(doc.text) <= MAX_DOC_CHARS else doc.text[:MAX_DOC_CHARS] + "\n[...rest of document cut]"
    user = f"Document: {doc.filename}\nRead with: {doc.method}\n\n<<<\n{text}\n>>>"
    return llm.chat_json(SYSTEM, user, XDoc)


def process_document(case: Case, case_dir: Path, filename: str, res: ExtractResult) -> list[PurchaseOrder]:
    d = DocumentText(filename=filename, method=res.method if not res.error else "none", pages=res.pages,
                     chars=len(res.text), ocr_confidence=res.ocr_confidence, error=res.error)
    case.documents.append(d)
    if res.error or len(res.text.strip()) < 20:
        d.error = d.error or "too little text"
        return []
    d.text_path = write_text(case_dir, filename, res.text)
    ctx = DocCtx(filename, res.method, res.text, res.ocr_words)
    xdoc, findings = read_with_model(ctx)
    if findings:
        case.flags.append("guard_finding"); case.notes += findings
    d.doc_type = xdoc.doc_type
    if xdoc.doc_type not in ("purchase_order", "order_acknowledgement_form"):
        return []
    a2_pos = case.classification.po_numbers if case.classification else []
    return [build_po(x, ctx, a2_pos, res.ocr_confidence, case.received_at) for x in xdoc.purchase_orders]


def extract_case(case: Case, case_dir: Path) -> None:
    shutil.rmtree(case_dir / "extracted", ignore_errors=True)        # re-run = start clean
    case.documents, case.purchase_orders = [], []
    pos: list[PurchaseOrder] = []

    for a in pick_documents(case):                                   # steps 1-2
        pos += process_document(case, case_dir, a.filename, extract(find_file(case_dir, a), a.kind))

    if not pos and case.body_text.strip():                           # PO written in the email itself
        body = f"Subject: {case.subject}\n\n{clean_email_text(case.body_text)}"
        pos += process_document(case, case_dir, "email_body", ExtractResult(text=body, method="email_body", pages=1))

    best: dict[str, PurchaseOrder] = {}                              # same PO in two files -> keep the better read
    for po in pos:
        k = norm_po(str(po.po_number.value or "")) or f"{po.source_file}#{len(best)}"
        if k not in best or po.overall_confidence > best[k].overall_confidence:
            best[k] = po
    case.purchase_orders = list(best.values())

    if case.purchase_orders:                                         # step 8
        case.status = "extracted"
    else:
        case.status = "manual_queue"
        read = [f"{d.filename} ({d.doc_type or d.error or 'unread'})" for d in case.documents]
        case.notes.append(f"A3: no purchase order found in: {', '.join(read) or 'nothing readable'}")


# ======================= run + check =======================
def audit(out_dir: Path, case: Case) -> None:
    with (out_dir / "audit.jsonl").open("a", encoding="utf-8") as f:
        f.write(json.dumps({"ts": datetime.now(timezone.utc).isoformat(), "case_id": case.case_id, "step": "A3",
                            "event": case.status, "model": llm.model_name(),
                            "pos": [str(p.po_number.value) for p in case.purchase_orders],
                            "overall": [p.overall_confidence for p in case.purchase_orders],
                            "flags": [p.flags for p in case.purchase_orders]}) + "\n")


def a3_failed(c: Case) -> bool:
    return c.status == "manual_queue" and bool(c.notes) and c.notes[-1].startswith("A3 failed")


def is_order_case(c: Case) -> bool:
    """A case A2 sent to A3 (new order or possible duplicate) - whatever A3 did with it since."""
    return bool(c.classification) and c.classification.category in ("new_order", "duplicate") \
        and c.status in ("classified", "extracted", "manual_queue")


def run(out_dir: Path, retry_failed: bool = False, everything: bool = False) -> list[Case]:
    cases = [Case.model_validate_json(p.read_text(encoding="utf-8")) for p in out_dir.glob("*/case.json")]
    todo = sorted((c for c in cases if c.status == "classified" or (retry_failed and a3_failed(c))
                   or (everything and is_order_case(c))),
                  key=lambda c: c.case_id)
    for c in todo:
        c.notes = [n for n in c.notes if not n.startswith(("A3 failed", "A3: no purchase order"))]  # fresh start
    busy_in_a_row = 0

    allow_real = os.getenv("ALLOW_EXTERNAL_LLM_FOR_REAL_DATA", "false").strip().lower() == "true"
    if llm.is_external() and not allow_real:
        real = [c.source_file for c in todo if SAMPLES not in Path(c.source_path).resolve().parents]
        if real:
            raise SystemExit(f"Refusing to send non-synthetic documents to {llm.PROVIDER}: {real[:3]}... "
                             f"Set ALLOW_EXTERNAL_LLM_FOR_REAL_DATA=true in .env only if approved.")

    for case in todo:
        case_dir = out_dir / case.case_id
        try:
            extract_case(case, case_dir)
        except Exception as e:                                       # model down, etc. - never stop the run
            case.status = "manual_queue"
            case.notes.append(f"A3 failed: {type(e).__name__}: {e}")
            busy_in_a_row = busy_in_a_row + 1 if "busy" in str(e).lower() else 0
        else:
            busy_in_a_row = 0
        (case_dir / "case.json").write_text(case.model_dump_json(indent=2), encoding="utf-8")
        audit(out_dir, case)
        docs = ", ".join(f"{d.method}:{d.doc_type or d.error or '?'}" for d in case.documents)
        pos = "; ".join(f"{p.po_number.value} ({len(p.lines)} ln, conf {p.overall_confidence:.2f}"
                        f"{', ' + ' '.join(p.flags) if p.flags else ''}"
                        f"{', missing ' + ','.join(p.missing_fields) if p.missing_fields else ''})"
                        for p in case.purchase_orders)
        print(f"{case.case_id}  {case.status:<12} {pos or '-'}   [{docs}]")
        if case.status == "manual_queue" and case.notes:
            print(f"      reason: {case.notes[-1][:300]}")
        if busy_in_a_row >= 2:
            print("\nSTOPPED: the model provider keeps refusing (rate limit - for Groq free tier most likely the\n"
                  "daily token limit). Cases already done are saved. Later, continue with:\n"
                  "    python -m agent_a.a3_extract --out <same folder> --retry-failed")
            break
    return todo


def _key_date(s) -> Optional[str]:
    if not s:
        return None
    return next(iter(dates_in(str(s))), None)


def check(cases: list[Case], keys_dir: Path) -> None:
    """Compare extracted fields with the synthetic answer keys, field by field."""
    stats: dict[str, list] = {}           # field -> [right, total, conf_when_right, conf_when_wrong]
    by_method: dict[str, list] = {}

    def score(field: str, got: FieldValue, expected, same, method: str):
        ok = same(got.value, expected)
        s = stats.setdefault(field, [0, 0, [], []]); s[1] += 1
        m = by_method.setdefault(method, [0, 0]); m[1] += 1
        if ok:
            s[0] += 1; m[0] += 1; s[2].append(got.confidence)
        else:
            s[3].append(got.confidence)
            print(f"  MISS {field:<22} got {got.value!r:<28} expected {expected!r}  (conf {got.confidence:.2f})")

    eq_num = lambda a, b: (a is None and b is None) or (isinstance(a, float) and b is not None and abs(a - float(b)) < 0.01)
    eq_txt = lambda a, b: (a is None and not b) or (a is not None and b is not None and nospace(str(a)) == nospace(str(b)))
    eq_po = lambda a, b: a is not None and norm_po(str(a)) == norm_po(str(b))
    eq_date = lambda a, b: (a is None and not b) or (a is not None and a == _key_date(b))

    for c in cases:
        k = keys_dir / (Path(c.source_file).stem + ".json")
        if not k.exists():
            continue
        key = json.loads(k.read_text(encoding="utf-8"))
        exp_pos = key.get("purchase_orders", [])
        if not exp_pos or c.status not in ("extracted", "manual_queue"):
            continue
        print(f"{c.case_id} {c.source_file}")
        got_by_po = {norm_po(str(p.po_number.value or "")): p for p in c.purchase_orders}
        for e in exp_pos:
            p = got_by_po.get(norm_po(e["customer_po_number"]))
            s = stats.setdefault("po_found", [0, 0, [], []]); s[1] += 1
            if not p:
                print(f"  MISS po_found            {e['customer_po_number']} not extracted")
                continue
            s[0] += 1
            m = p.read_method
            score("po_number", p.po_number, e["customer_po_number"], eq_po, m)
            score("po_date", p.po_date, e.get("po_date"), eq_date, m)
            got_date = p.requested_delivery_date if p.requested_delivery_date.value else (
                p.lines[0].delivery_date if p.lines else p.requested_delivery_date)
            score("requested_delivery_date", got_date, e.get("requested_delivery_date"), eq_date, m)
            score("line_count", FieldValue(value=float(len(p.lines)), confidence=1.0), len(e["lines"]), eq_num, m)
            for i, el in enumerate(e["lines"]):
                if i >= len(p.lines):
                    break
                gl = p.lines[i]
                score("line.quantity", gl.quantity, el.get("qty"), eq_num, m)
                score("line.uom", gl.uom, el.get("uom"), eq_txt, m)
                score("line.unit_price", gl.unit_price, el.get("unit_price"), eq_num, m)
                score("line.customer_material", gl.customer_material, el.get("customer_material"), eq_txt, m)
                score("line.line_total", gl.line_total, el.get("line_total"), eq_num, m)

    if not stats:
        print("No answer keys matched."); return
    print(f"\n{'field':<24}{'accuracy':>10}   avg conf when right / wrong")
    tot_r = tot_n = 0
    for f, (r, n, cr, cw) in stats.items():
        tot_r += r; tot_n += n
        avg = lambda xs: f"{sum(xs) / len(xs):.2f}" if xs else "  - "
        print(f"{f:<24}{r:>4}/{n:<4}{r / n:>5.0%}    {avg(cr)} / {avg(cw)}")
    print(f"{'ALL FIELDS':<24}{tot_r:>4}/{tot_n:<4}{tot_r / tot_n:>5.0%}")
    print("\nby read method:", ", ".join(f"{m} {r}/{n}" for m, (r, n) in by_method.items()))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="A3 - read the PO: extract fields with per-field confidence")
    ap.add_argument("--out", default=str(ROOT / CFG.paths.output))
    ap.add_argument("--check", help="answer_keys folder to compare against (synthetic only)")
    ap.add_argument("--retry-failed", action="store_true",
                    help="also re-run cases where A3 failed before (e.g. rate limit); finished ones are skipped")
    ap.add_argument("--all", action="store_true",
                    help="re-run A3 on every order case from scratch (no need to re-run A1/A2)")
    a = ap.parse_args()
    print(f"Provider: {llm.PROVIDER}  model: {llm.model_name()}\n")
    done = run(Path(a.out), a.retry_failed, a.all)
    print(f"\nThis run: {len(done)} cases: {sum(c.status == 'extracted' for c in done)} extracted, "
          f"{sum(c.status == 'manual_queue' for c in done)} manual queue")
    if a.check:                                                       # score every case A3 has handled so far
        everything = [Case.model_validate_json(p.read_text(encoding="utf-8")) for p in Path(a.out).glob("*/case.json")]
        print()
        check([c for c in everything if c.documents], Path(a.check))