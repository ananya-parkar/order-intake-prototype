"""
Duplicate or revised? Decided by the PO's content, after A3 has read it.

A2 only sees the email: "this PO number came in before" is all it can tell. Once A3 has the PO's lines,
we compare it with the earlier case that carried the same PO number:
    same lines (material, quantity, unit, price, dates)  -> duplicate   (resent PO; goes on, flagged for review)
    anything different                                   -> revised_po  (goes to the reviewer with the changes listed)
"""
from typing import Optional

from .a2_classify import norm_po
from .models import Case, FieldValue, LineItem, PurchaseOrder


def _v(f: FieldValue):
    """A comparable value: numbers rounded, text upper-case without spaces."""
    v = f.value
    if v is None or v == "":
        return None
    if isinstance(v, float):
        return round(v, 4)
    return "".join(str(v).upper().split())


def _line_key(ln: LineItem) -> tuple:
    """What identifies a line's product: customer material, else our material, else the description."""
    return _v(ln.customer_material) or _v(ln.supplier_material) or _v(ln.description) or ""


LINE_FIELDS = (("quantity", "quantity"), ("uom", "unit"), ("unit_price", "price"), ("price_per", "price per"),
               ("delivery_date", "delivery date"))


def differences(new: PurchaseOrder, old: PurchaseOrder) -> list[str]:
    """Plain-language list of what changed from the old PO to the new one (empty = same content)."""
    out = []
    a, b = _v(old.requested_delivery_date), _v(new.requested_delivery_date)
    if a != b:
        out.append(f"requested date {a or '-'} -> {b or '-'}")
    old_lines = {_line_key(l): l for l in old.lines}
    new_lines = {_line_key(l): l for l in new.lines}
    for k in old_lines.keys() - new_lines.keys():
        out.append(f"line removed: {k or '?'}")
    for k in new_lines.keys() - old_lines.keys():
        out.append(f"line added: {k or '?'}")
    for k in old_lines.keys() & new_lines.keys():
        for field, label in LINE_FIELDS:
            x, y = _v(getattr(old_lines[k], field)), _v(getattr(new_lines[k], field))
            if x != y:
                out.append(f"{k or 'line'} {label}: {x if x is not None else '-'} -> {y if y is not None else '-'}")
    return out


def _earlier_with_po(case: Case, po_no: str, cases: list[Case]) -> Optional[tuple[Case, PurchaseOrder]]:
    """The most recent earlier case (received before this one) whose extracted POs include this PO number."""
    best = None
    for c in cases:
        if c.case_id == case.case_id or not c.purchase_orders:
            continue
        if case.received_at and c.received_at and c.received_at >= case.received_at:
            continue
        for p in c.purchase_orders:
            if norm_po(str(p.po_number.value or "")) == po_no:
                if best is None or (c.received_at and best[0].received_at and c.received_at > best[0].received_at):
                    best = (c, p)
    return best


def check(case: Case, cases: list[Case]) -> None:
    """Compare each PO of this case with the earlier case that had the same PO number; reclassify if needed."""
    if not case.purchase_orders or not case.classification:
        return
    for po in case.purchase_orders:
        po_no = norm_po(str(po.po_number.value or ""))
        if not po_no:
            continue
        hit = _earlier_with_po(case, po_no, cases)
        if not hit:
            continue
        earlier, old = hit
        diffs = differences(po, old)
        cl = case.classification
        if not diffs:
            cl.category, case.duplicate_of = "duplicate", earlier.case_id
            if "possible_duplicate" not in case.flags:
                case.flags.append("possible_duplicate")
            case.notes.append(f"A3: PO {po_no} has the same content as in {earlier.case_id} -> duplicate (resent)")
        else:
            cl.category, cl.detail = "revised_po", "revised_by_content"
            case.duplicate_of = earlier.case_id                  # stays "extracted": A4 builds the payload and the
                                                                 # review gate shows the changes to the reviewer
            if "revised_by_content" not in case.flags:
                case.flags.append("revised_by_content")
            case.notes.append(f"A3: PO {po_no} changed since {earlier.case_id} -> revised PO. Changes: "
                              + "; ".join(diffs[:10]) + (" ..." if len(diffs) > 10 else ""))
            return                                               # one revised PO is enough to route the case