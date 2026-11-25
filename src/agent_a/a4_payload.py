"""
A4 - Build the SAP payload, and the review gate for it.

For every case A3 marked "extracted", one payload per PO, in the exact JSON the RPA sends to SAP via Azure:
  header: SAP_Account_Number  <- Salesforce (Case.SAP_Account_Number__c), 0000 + 6 digits (= ship-to)
          Sold_To_Party       <- SAP account without its 4 leading zeros
          Order_Reason 200, Order_Type TA, Sales_Org 1000, Distribution_Channel, Division <- fixed (config.yaml)
          PO_Number, PO_Date, Request_Date (ship date), contact_name  <- PO (A3)
          Created_On  <- today, US Central time, 'yyyy-MM-dd 00:00:00' (same as the RPA)
  Items:  one per PO line: PO_Item 10, 20, 30 ... + CUST_MAT22, Quantity, Pricing  <- PO (A3)
          Unit_of_Measure as SAP's code in capitals (mapping in config.yaml: a4.uom_map / a4.uom_prefix)
          Lines with quantity 0 are left out or sent, as config.yaml says (a4.zero_qty_lines: skip | send)
  Everything else stays "" (Created_By = "NULL"), as in the template.
  Material stays "": SAP resolves CUST_MAT22 through CMIR.

Review gate (next to every payload, in <PO_Number>.meta.json):
  review_status  ready_for_sap  - no problems found; the reviewer only confirms
                 needs_review   - one or more reasons below; the reviewer checks them
  reasons        [{code, severity (critical | warning | info), message}]
  In Phase 1 every case goes to a reviewer either way; the status tells them what to look at.

Output: <case>/payloads/<PO_Number>.json        the payload, exactly as the RPA sends it
        <case>/payloads/<PO_Number>.meta.json   the review gate for it
        case status "payload_built".

Run from src/:
    python -m agent_a.a4_payload --out ..\\output --check ..\\data\\sample_answer_keys
    python -m agent_a.a4_payload --out ..\\output --spec        (check saved payloads against the client's spec)
"""
import argparse
import copy
import json
import re
import time
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

from .config import CFG, ROOT
from .constants import (SAP_CRITICAL, SAP_ITEM as ITEM, SAP_MODERATE, SAP_REQUIRED_HEADER, SAP_REQUIRED_ITEM,
                        SAP_SPEC_HEADER, SAP_SPEC_ITEM, SAP_STATIC, SAP_TEMPLATE as TEMPLATE)
from .models import Case, LineItem, PurchaseOrder, SapPayload
from .trace import step_trace

READY, NEEDS_REVIEW = "ready_for_sap", "needs_review"
PER_ORDERED_UNIT = {"EA", "EACH", "UNIT", "UN"}          # "price per each" = per ordered unit -> no mismatch


def text(v) -> str:
    """None -> '', anything else -> trimmed string (every payload value is a string)."""
    return "" if v is None else str(v).strip()


def num(v) -> str:
    """44000.0 -> '44000', 47.75 -> '47.75'"""
    try:
        f = float(v)
    except (TypeError, ValueError):
        return text(v)
    return str(int(f)) if f.is_integer() else f"{f:.6f}".rstrip("0").rstrip(".")


def is_zero(v) -> bool:
    """True if the value is the number 0."""
    try:
        return float(v) == 0
    except (TypeError, ValueError):
        return False


def sold_to(account: str) -> str:
    """Sold_To_Party = SAP account number without its 4 leading zeros (0000104521 -> 104521)."""
    return account[4:] if account.startswith("0000") else account


def sap_uom(u) -> str:
    """Customer unit -> SAP unit code, ALL CAPS (bags -> BAG, Sack of 2000 -> SAC). Unknown units: just upper-case.
    The mapping lives in config.yaml (a4.uom_map exact spellings, a4.uom_prefix 'starts with')."""
    u = text(u).upper()
    if u in CFG.a4.uom_map:
        return CFG.a4.uom_map[u]
    return next((code for prefix, code in CFG.a4.uom_prefix.items() if u.startswith(prefix)), u)


def price_unit(per) -> str:
    """The pricing unit as a SAP code where we know one: 'per lb' -> 'LB', '/BAG' -> 'BAG', '1 LB' -> 'LB',
    'CWT' -> 'CWT', '100 LB' -> '100 LB'. '' if the PO shows none."""
    p = re.sub(r"^(PER\s+|/\s*)", "", text(per).upper()).strip()
    p = re.sub(r"^1\s+", "", p)
    return sap_uom(p) if p else ""


def us_date(iso: str, year_digits: int = 4) -> str:
    """'2026-09-18' -> '09/18/2026' (Request_Date) or '09/18/26' (PO_Date), as in the RPA payloads."""
    m = re.fullmatch(r"(\d{4})-(\d{2})-(\d{2})", iso or "")
    return f"{m[2]}/{m[3]}/{m[1][-year_digits:]}" if m else text(iso)


def today_central() -> date:
    """Today's date in US Central time (the RPA's time zone)."""
    try:
        from zoneinfo import ZoneInfo
        return datetime.now(ZoneInfo(CFG.a4.timezone)).date()
    except Exception:                                   # Windows without tzdata -> fixed UTC-6
        return (datetime.now(timezone.utc) - timedelta(hours=6)).date()


def created_on() -> str:
    """Same as the RPA: today's date in US Central time, 'yyyy-MM-dd 00:00:00'."""
    return today_central().strftime("%Y-%m-%d 00:00:00")


def request_date(po: PurchaseOrder) -> str:
    """Header ship/delivery date, else the earliest line date, as YYYY-MM-DD ('' if none)."""
    req = po.requested_delivery_date.value or min(
        (l.delivery_date.value for l in po.lines if l.delivery_date.value), default=None)
    return text(req)


def payload_lines(po: PurchaseOrder) -> tuple[list[LineItem], list[LineItem]]:
    """(lines that go into the payload, quantity-0 lines left out). config.yaml a4.zero_qty_lines: skip | send."""
    if CFG.a4.zero_qty_lines == "send":
        return list(po.lines), []
    return ([l for l in po.lines if not is_zero(l.quantity.value)],
            [l for l in po.lines if is_zero(l.quantity.value)])


def payload_paths(folder: Path) -> list[Path]:
    """The payload files in a payloads/ folder (not their .meta.json review files)."""
    if not folder.exists():
        return []
    return sorted(p for p in folder.glob("*.json") if not p.name.endswith(".meta.json"))


def build_payload(case: Case, po: PurchaseOrder) -> dict:
    """Fill the RPA template from the case (Salesforce) and one PO (A3); checked against SapPayload."""
    p = copy.deepcopy(TEMPLATE)
    p["SAP_Account_Number"] = text(case.sap_account_number)
    p["Sold_To_Party"] = sold_to(text(case.sap_account_number))
    p["Created_On"] = created_on()
    p["Order_Reason"] = CFG.a4.order_reason                    # fixed values from the client (config.yaml)
    p["Order_Type"] = CFG.a4.order_type
    p["Sales_Org"] = CFG.a4.sales_org
    p["Distribution_Channel"] = CFG.a4.distribution_channel
    p["Division"] = CFG.a4.division
    p["PO_Number"] = text(po.po_number.value)
    p["PO_Date"] = us_date(text(po.po_date.value), 2)          # MM/DD/YY, as the RPA sends it
    p["Request_Date"] = us_date(request_date(po))              # MM/DD/YYYY, as the RPA sends it
    p["contact_name"] = text(po.buyer_name.value) or case.contact_name   # RPA takes it from the PO
    kept, _ = payload_lines(po)
    for i, line in enumerate(kept):
        it = copy.deepcopy(ITEM)
        it["PO_Item"] = str((i + 1) * 10)
        it["CUST_MAT22"] = text(line.customer_material.value)
        it["Quantity"] = num(line.quantity.value)
        it["Unit_of_Measure"] = sap_uom(line.uom.value)
        it["Pricing"] = num(line.unit_price.value)
        p["Items"].append(it)
    return SapPayload.model_validate(p).model_dump(by_alias=True)     # wrong shape -> error, never saved


def missing(p: dict) -> list[str]:
    """Required fields that are still empty (shown to the reviewer)."""
    out = [k for k in SAP_REQUIRED_HEADER if not p[k]]
    if not p["Items"]:
        out.append("Items")
    for it in p["Items"]:
        out += [f"Item {it['PO_Item']}.{k}" for k in SAP_REQUIRED_ITEM if not it[k]]
    return out


# ======================= review gate =======================
def review(case: Case, po: PurchaseOrder, p: dict) -> dict:
    """What the reviewer must check for this payload, and why. needs_review if any critical/warning reason."""
    reasons: list[dict] = []

    def add(code: str, severity: str, message: str) -> None:
        reasons.append({"code": code, "severity": severity, "message": message})

    # 1. required fields
    if not p["SAP_Account_Number"]:
        add("sap_account_missing", "critical", "SAP account number is empty (it comes from the Salesforce account)")
    gaps = [g for g in missing(p) if g != "SAP_Account_Number"]
    if gaps:
        add("required_field_missing", "critical", "Required field(s) empty: " + ", ".join(gaps))

    # 2. extraction confidence
    limit = CFG.a4.review_min_confidence
    if po.overall_confidence < limit:
        add("low_confidence", "warning",
            f"Extraction confidence {po.overall_confidence:.2f} is below {limit:.2f} - check the flagged fields")

    # 3. maths on the PO
    if "math_mismatch" in po.flags:
        add("math_mismatch", "warning", "Quantity x price does not match a line total on the PO")
    if "total_mismatch" in po.flags:
        add("total_mismatch", "warning", "The line totals do not add up to the PO total")

    # 4. units SAP may not accept
    for it in p["Items"]:
        u = it["Unit_of_Measure"]
        if u and u not in CFG.a4.sap_uom_codes:
            add("unknown_uom", "warning", f"Item {it['PO_Item']}: unit {u!r} is not a confirmed SAP unit code")

    # 5. price unit vs quantity unit (e.g. price per CWT, quantity in LB -> Pricing would be read 100x wrong)
    kept, skipped = payload_lines(po)
    for it, line in zip(p["Items"], kept):
        per = price_unit(line.price_per.value)
        if per and per not in PER_ORDERED_UNIT and per != it["Unit_of_Measure"]:
            add("price_unit_mismatch", "warning",
                f"Item {it['PO_Item']}: price is per {per} but quantity is in {it['Unit_of_Measure'] or '?'} "
                f"- check Pricing before it goes to SAP")

    # 6. date sanity
    req, po_date = request_date(po), text(po.po_date.value)
    iso = re.compile(r"\d{4}-\d{2}-\d{2}")
    if CFG.a4.check_past_request_date and iso.fullmatch(req) and req < today_central().isoformat():
        add("request_date_past", "warning", f"Request date {us_date(req)} is already in the past")
    if iso.fullmatch(req) and iso.fullmatch(po_date) and req < po_date:
        add("request_before_po_date", "warning",
            f"Request date {us_date(req)} is before the PO date {us_date(po_date)}")

    # 7. duplicate and guard findings (from A2 / the guard)
    if "possible_duplicate" in case.flags or (case.classification and case.classification.category == "duplicate"):
        add("possible_duplicate", "warning",
            f"Possible duplicate of {case.duplicate_of}" if case.duplicate_of else "Possible duplicate order")
    if "guard_finding" in case.flags:
        add("guard_finding", "critical", "Instruction-like text found in the email or PO (prompt-injection check)")

    # 7b. revised PO found by content (po_compare) - the reviewer must see exactly what changed
    if "revised_by_content" in case.flags:
        changes = next((n.split("Changes: ", 1)[1] for n in reversed(case.notes)
                        if n.startswith("A3: PO ") and "Changes: " in n), "")
        add("revised_po", "critical", f"Revised PO - changed since {case.duplicate_of}: {changes}")

    # 8. quantity-0 lines
    for line in skipped:
        add("zero_qty_line_skipped", "info",
            f"PO line {text(line.line_no.value) or '?'} has quantity 0 and was left out of the payload")
    if CFG.a4.zero_qty_lines == "send":
        for it in p["Items"]:
            if is_zero(it["Quantity"]):
                add("zero_qty_line_sent", "warning", f"Item {it['PO_Item']}: quantity 0 is being sent to SAP")

    status = NEEDS_REVIEW if any(r["severity"] != "info" for r in reasons) else READY
    return {"po_number": p["PO_Number"], "review_status": status, "reasons": reasons,
            "overall_confidence": po.overall_confidence, "built_at": datetime.now(timezone.utc).isoformat(),
            "trace": step_trace()}


def run(out_dir: Path) -> list[Case]:
    """Build payloads/<PO>.json + <PO>.meta.json for every extracted case; set its status to payload_built."""
    cases = [Case.model_validate_json(f.read_text(encoding="utf-8")) for f in out_dir.glob("*/case.json")]
    todo = sorted((c for c in cases if c.status in ("extracted", "payload_built")), key=lambda c: c.case_id)
    for case in todo:
        t0 = time.perf_counter()
        folder = out_dir / case.case_id / "payloads"
        folder.mkdir(exist_ok=True)
        for old in folder.glob("*.json"):                       # payloads and their .meta.json
            old.unlink()
        statuses = []
        for n, po in enumerate(case.purchase_orders, 1):
            p = build_payload(case, po)
            meta = review(case, po, p)
            name = re.sub(r"[^A-Za-z0-9_.-]", "_", p["PO_Number"]) or f"po_{n}"
            (folder / f"{name}.json").write_text(json.dumps(p, indent=2), encoding="utf-8")
            (folder / f"{name}.meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
            statuses.append(meta["review_status"])
            serious = [r["code"] for r in meta["reasons"] if r["severity"] != "info"]
            print(f"{case.case_id}  {p['PO_Number'] or '-':<16} {len(p['Items'])} item(s)  {meta['review_status']}"
                  + (f"   {', '.join(serious)}" if serious else ""))
        case.status = "payload_built"
        seconds = round(time.perf_counter() - t0, 3)
        case.trace["a4"] = step_trace(review=statuses, usage={"seconds": seconds})   # config version = UoM map / thresholds
        (out_dir / case.case_id / "case.json").write_text(case.model_dump_json(indent=2), encoding="utf-8")
        with (out_dir / "audit.jsonl").open("a", encoding="utf-8") as f:
            f.write(json.dumps({"ts": datetime.now(timezone.utc).isoformat(), "case_id": case.case_id,
                                "step": "A4", "event": case.status,
                                "pos": [text(po.po_number.value) for po in case.purchase_orders],
                                "review": statuses, "seconds": seconds}) + "\n")
    return todo


def spec_issues(p: dict) -> list[str]:
    """Compare one payload with the client's spec: exact keys and order, which fields are filled, and formats."""
    out = []
    if list(p) != list(TEMPLATE):
        out.append("header keys/order differ from the RPA template")
    for i, it in enumerate(p.get("Items", [])):
        if list(it) != list(ITEM):
            out.append(f"Item {i + 1}: keys/order differ from the RPA template")
        if it.get("PO_Item") != str((i + 1) * 10):
            out.append(f"Item {i + 1}: PO_Item is {it.get('PO_Item')!r}, expected {(i + 1) * 10}")
        for k in ("Quantity", "Pricing"):
            if it.get(k) and not re.fullmatch(r"\d+(\.\d+)?", it[k]):
                out.append(f"Item {i + 1}: {k} {it[k]!r} is not a plain number")
        out += [f"Item {i + 1}: {k} should be empty" for k in ITEM if k not in SAP_SPEC_ITEM and it.get(k)]
        u = it.get("Unit_of_Measure", "")
        if u and (u != u.upper() or not re.fullmatch(r"[A-Z]{2,3}", u)):
            out.append(f"Item {i + 1}: Unit_of_Measure {u!r} is not a 2-3 letter SAP code in capitals (moderate - confirm SAP code)")
        out += [f"Item {i + 1}: {k} is empty (critical)" for k in SAP_CRITICAL if k in it and not it[k]]
        out += [f"Item {i + 1}: {k} is empty (moderate)" for k in SAP_MODERATE if not it.get(k)]
    if p.get("Created_By") != "NULL":
        out.append("Created_By should be 'NULL'")
    acct = p.get("SAP_Account_Number", "")
    if not acct:
        out.append("SAP_Account_Number is empty (critical)")
    elif not re.fullmatch(r"0000\d{6}", acct):
        out.append(f"SAP_Account_Number {acct!r} is not 0000 + 6 digits (critical)")
    if acct and p.get("Sold_To_Party") != sold_to(acct):
        out.append(f"Sold_To_Party should be {sold_to(acct)!r} (SAP account without the 4 zeros)")
    for k in SAP_STATIC:
        want = getattr(CFG.a4, k.lower())
        if want and p.get(k) != want:
            out.append(f"{k} should be {want!r}")
    out += [f"{k} should be empty" for k in TEMPLATE if k not in SAP_SPEC_HEADER + ("Created_By", "Items") and p.get(k)]
    if not re.fullmatch(r"\d{4}-\d{2}-\d{2} 00:00:00", p.get("Created_On", "")):
        out.append("Created_On is not 'yyyy-MM-dd 00:00:00'")
    if p.get("PO_Date") and not re.fullmatch(r"\d{2}/\d{2}/\d{2}", p["PO_Date"]):
        out.append(f"PO_Date {p['PO_Date']!r} is not MM/DD/YY")
    if p.get("Request_Date") and not re.fullmatch(r"\d{2}/\d{2}/\d{4}", p["Request_Date"]):
        out.append(f"Request_Date {p['Request_Date']!r} is not MM/DD/YYYY")
    return out


def spec_report(out_dir: Path) -> None:
    """Check every saved payload against the client's spec and show which spec fields we fill and how often."""
    files = [f for f in sorted(out_dir.glob("*/payloads/*.json")) if not f.name.endswith(".meta.json")]
    if not files:
        print("No payloads found."); return
    filled = {k: 0 for k in SAP_SPEC_HEADER + SAP_SPEC_ITEM}
    items = 0
    level = {"critical": 0, "moderate": 0, "format": 0, "clean": 0}
    for f in files:
        p = json.loads(f.read_text(encoding="utf-8"))
        for k in SAP_SPEC_HEADER:
            filled[k] += bool(p.get(k))
        for it in p["Items"]:
            items += 1
            for k in SAP_SPEC_ITEM:
                filled[k] += bool(it.get(k))
        issues = spec_issues(p)
        worst = ("critical" if any("(critical)" in m for m in issues) else
                 "moderate" if any("(moderate" in m for m in issues) else "format" if issues else "clean")
        level[worst] += 1
        for msg in issues:
            print(f"  {f.parent.parent.name} {p.get('PO_Number') or f.stem}: {msg}")
    print(f"\n{len(files)} payloads, {items} items: {level['clean']} clean, {level['critical']} with a critical gap "
          f"(SAP account / customer material), {level['moderate']} with a moderate gap (price / qty / UoM), "
          f"{level['format']} with other format issues\n")
    print(f"{'spec field':<22}{'filled':>10}   note")
    for k, n in filled.items():
        total = items if k in SAP_SPEC_ITEM else len(files)
        note = ("fixed value - client still to send it" if k in SAP_STATIC and not getattr(CFG.a4, k.lower()) else
                "fixed value from config.yaml" if k in SAP_STATIC else
                "from Salesforce account (critical)" if k in ("SAP_Account_Number", "Sold_To_Party") and n < total else "")
        print(f"{k:<22}{n:>4}/{total:<5}   {note}")


def check(out_dir: Path, keys_dir: Path) -> None:
    """Compare the payloads with the answer keys."""
    def key_date(s, year_digits=4):
        """Answer keys hold MM/DD/YYYY; turn it into the payload's format (MM/DD/YYYY or MM/DD/YY)."""
        m = re.match(r"(\d{2})/(\d{2})/(\d{4})", s or "")
        return f"{m[1]}/{m[2]}/{m[3][-year_digits:]}" if m else ""

    def same_num(a, b):
        """True if two values are the same number (ignoring tiny rounding)."""
        try:
            return abs(float(a) - float(b)) < 0.01
        except (TypeError, ValueError):
            return False

    right = total = 0
    for c in sorted((Case.model_validate_json(f.read_text(encoding="utf-8")) for f in out_dir.glob("*/case.json")),
                    key=lambda c: c.case_id):
        k = keys_dir / (Path(c.source_file).stem + ".json")
        if c.status != "payload_built" or not k.exists():
            continue
        for e in json.loads(k.read_text(encoding="utf-8")).get("purchase_orders") or []:
            f = out_dir / c.case_id / "payloads" / (re.sub(r"[^A-Za-z0-9_.-]", "_", e["customer_po_number"]) + ".json")
            p = json.loads(f.read_text(encoding="utf-8")) if f.exists() else None
            tests = [("payload", p is not None)]
            if p:
                tests += [("PO_Number", p["PO_Number"] == e["customer_po_number"]),
                          ("PO_Date", p["PO_Date"] == key_date(e.get("po_date"), 2)),
                          ("Request_Date", p["Request_Date"] == key_date(e.get("requested_delivery_date"))),
                          ("item_count", len(p["Items"]) == len(e.get("lines") or []))]
                for it, el in zip(p["Items"], e.get("lines") or []):
                    tests += [("CUST_MAT22", it["CUST_MAT22"] == el.get("customer_material")),
                              ("Quantity", same_num(it["Quantity"], el.get("qty"))),
                              ("Unit_of_Measure", it["Unit_of_Measure"] == sap_uom(el.get("uom"))),
                              ("Pricing", same_num(it["Pricing"], el.get("unit_price")))]
            for name, ok in tests:
                right += ok; total += 1
                if not ok:
                    print(f"  MISS {c.case_id} {e['customer_po_number']}: {name}")
    if total:
        print(f"\nA4 payload accuracy: {right}/{total} = {right / total:.0%}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="A4 - build the SAP payload and its review gate")
    ap.add_argument("--out", default=str(ROOT / CFG.paths.output))
    ap.add_argument("--check", help="answer_keys folder to compare against")
    ap.add_argument("--spec", action="store_true", help="only check the saved payloads against the client's spec")
    a = ap.parse_args()
    if a.spec:
        spec_report(Path(a.out))
        raise SystemExit
    done = run(Path(a.out))
    metas = [json.loads(m.read_text(encoding="utf-8")) for m in Path(a.out).glob("*/payloads/*.meta.json")]
    ready = sum(m["review_status"] == READY for m in metas)
    print(f"\n{len(done)} case(s), {len(metas)} payload(s): {ready} ready for SAP, {len(metas) - ready} need review")
    if a.check:
        check(Path(a.out), Path(a.check))