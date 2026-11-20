"""
A4 - Build the SAP payload.

For every case A3 marked "extracted", one payload per PO, in the exact JSON the RPA sends to SAP via Azure:
  header: SAP_Account_Number, Sold_To_Party  <- Salesforce (Case.SAP_Account_Number__c)
          PO_Number, PO_Date, Request_Date, contact_name  <- PO (A3)
          Created_On  <- today, US Central time, 'yyyy-MM-dd 00:00:00' (same as the RPA)
  Items:  one per PO line: PO_Item 10, 20, 30 ... + CUST_MAT22, Quantity, Unit_of_Measure, Pricing  <- PO (A3)
  Everything else stays "" (Created_By = "NULL"), as in the template.
  Material stays "": SAP resolves CUST_MAT22 through CMIR.
Output: <case>/payloads/<PO_Number>.json, case status "payload_built".

Run from src/:
    python -m agent_a.a4_payload --out ..\\output --check ..\\data\\answer_keys
"""
import argparse
import copy
import json
import re
from datetime import datetime, timedelta, timezone
from pathlib import Path

from .config import CFG, ROOT
from .constants import SAP_ITEM as ITEM, SAP_REQUIRED_HEADER, SAP_REQUIRED_ITEM, SAP_TEMPLATE as TEMPLATE
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
    p["Sold_To_Party"] = text(case.sap_account_number)        # same as SAP account for now (to confirm)
    p["Created_On"] = created_on()
    p["PO_Number"] = text(po.po_number.value)
    p["PO_Date"] = text(po.po_date.value)                      # YYYY-MM-DD from A3
    req = po.requested_delivery_date.value or min(
        (l.delivery_date.value for l in po.lines if l.delivery_date.value), default=None)
    p["Request_Date"] = text(req)
    p["contact_name"] = text(po.buyer_name.value) or case.contact_name   # RPA takes it from the PO
    for i, line in enumerate(po.lines):
        it = copy.deepcopy(ITEM)
        it["PO_Item"] = str((i + 1) * 10)
        it["CUST_MAT22"] = text(line.customer_material.value)
        it["Quantity"] = num(line.quantity.value)
        it["Unit_of_Measure"] = text(line.uom.value)
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


def check(out_dir: Path, keys_dir: Path) -> None:
    """Compare the payloads with the synthetic answer keys."""
    def iso(s):
        m = re.match(r"(\d{2})/(\d{2})/(\d{4})", s or "")
        return f"{m[3]}-{m[1]}-{m[2]}" if m else ""

    def same_num(a, b):
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
                          ("PO_Date", p["PO_Date"] == iso(e.get("po_date"))),
                          ("Request_Date", p["Request_Date"] == iso(e.get("requested_delivery_date"))),
                          ("item_count", len(p["Items"]) == len(e.get("lines") or []))]
                for it, el in zip(p["Items"], e.get("lines") or []):
                    tests += [("CUST_MAT22", it["CUST_MAT22"] == el.get("customer_material")),
                              ("Quantity", same_num(it["Quantity"], el.get("qty"))),
                              ("Unit_of_Measure", it["Unit_of_Measure"].upper() == str(el.get("uom")).upper()),
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
    ap.add_argument("--check", help="answer_keys folder (synthetic emails only)")
    a = ap.parse_args()
    done = run(Path(a.out))
    print(f"\n{len(done)} case(s), {sum(len(c.purchase_orders) for c in done)} payload(s) built")
    if a.check:
        check(Path(a.out), Path(a.check))