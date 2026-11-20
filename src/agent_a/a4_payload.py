"""
A4 - Build the SAP payload.

For every case A3 marked "extracted", one payload per PO, in the exact JSON the RPA sends to SAP via Azure:
  header: SAP_Account_Number  <- Salesforce (Case.SAP_Account_Number__c), 0000 + 6 digits (= ship-to)
          Sold_To_Party       <- SAP account without its 4 leading zeros
          Order_Reason 200, Order_Type TA, Sales_Org 1000, Distribution_Channel, Division <- fixed (config.yaml)
          PO_Number, PO_Date, Request_Date (ship date), contact_name  <- PO (A3)
          Created_On  <- today, US Central time, 'yyyy-MM-dd 00:00:00' (same as the RPA)
  Items:  one per PO line: PO_Item 10, 20, 30 ... + CUST_MAT22, Quantity, Pricing  <- PO (A3)
          Unit_of_Measure as SAP's 3-letter code in capitals (bags -> BAG, Sack of 2000 -> SAC)
  Everything else stays "" (Created_By = "NULL"), as in the template.
  Material stays "": SAP resolves CUST_MAT22 through CMIR.
Output: <case>/payloads/<PO_Number>.json, case status "payload_built".

Run from src/:
    python -m agent_a.a4_payload --out ..\\output --check ..\\data\\sample_answer_keys
    python -m agent_a.a4_payload --out ..\\output --spec        (check saved payloads against the client's spec)
"""
import argparse
import copy
import json
import re
from datetime import datetime, timedelta, timezone
from pathlib import Path

from .config import CFG, ROOT
from .constants import (SAP_CRITICAL, SAP_ITEM as ITEM, SAP_MODERATE, SAP_REQUIRED_HEADER, SAP_REQUIRED_ITEM,
                        SAP_SPEC_HEADER, SAP_SPEC_ITEM, SAP_STATIC, SAP_TEMPLATE as TEMPLATE, SAP_UOM, SAP_UOM_PREFIX)
from .models import Case, PurchaseOrder, SapPayload


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


def sold_to(account: str) -> str:
    """Sold_To_Party = SAP account number without its 4 leading zeros (0000104521 -> 104521)."""
    return account[4:] if account.startswith("0000") else account


def sap_uom(u) -> str:
    """Customer unit -> SAP unit code, ALL CAPS (bags -> BAG, Sack of 2000 -> SAC). Unknown units: just upper-case."""
    u = text(u).upper()
    if u in SAP_UOM:
        return SAP_UOM[u]
    return next((code for prefix, code in SAP_UOM_PREFIX.items() if u.startswith(prefix)), u)

def us_date(iso: str, year_digits: int = 4) -> str:
    """'2026-09-18' -> '09/18/2026' (Request_Date) or '09/18/26' (PO_Date), as in the RPA payloads."""
    m = re.fullmatch(r"(\d{4})-(\d{2})-(\d{2})", iso or "")
    return f"{m[2]}/{m[3]}/{m[1][-year_digits:]}" if m else text(iso)

def created_on() -> str:
    """Same as the RPA: today's date in US Central time, 'yyyy-MM-dd 00:00:00'."""
    try:
        from zoneinfo import ZoneInfo
        now = datetime.now(ZoneInfo(CFG.a4.timezone))
    except Exception:                                   # Windows without tzdata -> fixed UTC-6
        now = datetime.now(timezone.utc) - timedelta(hours=6)
    return now.strftime("%Y-%m-%d 00:00:00")


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
    req = po.requested_delivery_date.value or min(
        (l.delivery_date.value for l in po.lines if l.delivery_date.value), default=None)
    p["Request_Date"] = us_date(text(req))                           # MM/DD/YYYY, as the RPA sends it
    p["contact_name"] = text(po.buyer_name.value) or case.contact_name   # RPA takes it from the PO
    for i, line in enumerate(po.lines):
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


def run(out_dir: Path) -> list[Case]:
    """Build payloads/<PO>.json for every extracted case and set its status to payload_built."""
    cases = [Case.model_validate_json(f.read_text(encoding="utf-8")) for f in out_dir.glob("*/case.json")]
    todo = sorted((c for c in cases if c.status in ("extracted", "payload_built")), key=lambda c: c.case_id)
    for case in todo:
        folder = out_dir / case.case_id / "payloads"
        folder.mkdir(exist_ok=True)
        for old in folder.glob("*.json"):
            old.unlink()
        for n, po in enumerate(case.purchase_orders, 1):
            p = build_payload(case, po)
            name = re.sub(r"[^A-Za-z0-9_.-]", "_", p["PO_Number"]) or f"po_{n}"
            (folder / f"{name}.json").write_text(json.dumps(p, indent=2), encoding="utf-8")
            gaps = missing(p)
            print(f"{case.case_id}  {p['PO_Number'] or '-':<16} {len(p['Items'])} item(s)"
                  + (f"   missing: {', '.join(gaps)}" if gaps else ""))
        case.status = "payload_built"
        (out_dir / case.case_id / "case.json").write_text(case.model_dump_json(indent=2), encoding="utf-8")
        with (out_dir / "audit.jsonl").open("a", encoding="utf-8") as f:
            f.write(json.dumps({"ts": datetime.now(timezone.utc).isoformat(), "case_id": case.case_id,
                                "step": "A4", "event": case.status,
                                "pos": [text(po.po_number.value) for po in case.purchase_orders]}) + "\n")
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
    files = sorted(out_dir.glob("*/payloads/*.json"))
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
    ap = argparse.ArgumentParser(description="A4 - build the SAP payload")
    ap.add_argument("--out", default=str(ROOT / CFG.paths.output))
    ap.add_argument("--check", help="answer_keys folder to compare against")
    ap.add_argument("--spec", action="store_true", help="only check the saved payloads against the client's spec")
    a = ap.parse_args()
    if a.spec:
        spec_report(Path(a.out))
        raise SystemExit
    done = run(Path(a.out))
    print(f"\n{len(done)} case(s), {sum(len(c.purchase_orders) for c in done)} payload(s) built")
    if a.check:
        check(Path(a.out), Path(a.check))