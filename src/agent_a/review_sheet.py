"""
Review sheet - check A2's classification by hand (works for real emails, no answer keys needed).

1. Make the sheet (one row per case, A2's answer + empty columns for you):
       python -m agent_a.review_sheet --out ..\\output_real
   -> output_real\\review.csv   (open in Excel)
   Or put the sheet anywhere else, e.g. outside the project:
       python -m agent_a.review_sheet --out ..\\output_real --sheet C:\\Users\\<you>\\Documents\\a2_review.csv

2. Fill in, for each row, ONE of:
       correct_category   new_order / revised_po / duplicate / not_an_order   (your judgement)
       sf_request_type    the case's Request Type in Salesforce, e.g. "Order Placement"
   correct_category wins if both are filled. Optional: comment.

3. Score:
       python -m agent_a.review_sheet --out ..\\output_real --score   (add the same --sheet if you used one)

Re-making the sheet keeps what you already filled in (matched by case_id).
The sheet stays inside the output folder, so for real emails it is git-ignored.
"""
import argparse
import csv
from pathlib import Path

from .models import Case

ROOT = Path(__file__).resolve().parents[2]
CATEGORIES = {"new_order", "revised_po", "duplicate", "not_an_order"}

# Ingredion Request_Type__c -> our A2 category (anything else = not_an_order)
REQUEST_TYPE_MAP = {
    "order placement": "new_order",
    "order change request": "revised_po",
}

A2_COLUMNS = ["case_id", "source_file", "subject", "from_address", "attachments",
              "a2_category", "a2_detail", "a2_po_numbers", "a2_confidence", "a2_reason", "flags", "status"]
YOUR_COLUMNS = ["correct_category", "sf_request_type", "comment"]


def load_cases(out_dir: Path) -> list[Case]:
    cases = [Case.model_validate_json(p.read_text(encoding="utf-8")) for p in out_dir.glob("*/case.json")]
    return sorted(cases, key=lambda c: c.case_id)


def read_sheet(path: Path) -> dict[str, dict]:
    if not path.exists():
        return {}
    with path.open(encoding="utf-8-sig", newline="") as f:
        return {row["case_id"]: row for row in csv.DictReader(f)}


def make(out_dir: Path, sheet: Path) -> Path:
    sheet.parent.mkdir(parents=True, exist_ok=True)
    old = read_sheet(sheet)                                  # keep what was already filled in
    rows = []
    for c in load_cases(out_dir):
        cl = c.classification
        files = [a.filename for a in c.attachments if not a.likely_signature]
        row = {
            "case_id": c.case_id, "source_file": c.source_file, "subject": c.subject,
            "from_address": c.from_address, "attachments": "; ".join(files),
            "a2_category": cl.category if cl else f"(none - {c.status})",
            "a2_detail": cl.detail if cl else "",
            "a2_po_numbers": ", ".join(cl.po_numbers) if cl else "",
            "a2_confidence": f"{cl.confidence:.2f}" if cl else "",
            "a2_reason": cl.reason if cl else "",
            "flags": " ".join(c.flags), "status": c.status,
        }
        for col in YOUR_COLUMNS:
            row[col] = old.get(c.case_id, {}).get(col, "")
        rows.append(row)

    try:
        with sheet.open("w", encoding="utf-8-sig", newline="") as f:     # utf-8-sig = Excel opens it cleanly
            w = csv.DictWriter(f, fieldnames=A2_COLUMNS + YOUR_COLUMNS)
            w.writeheader(); w.writerows(rows)
    except PermissionError:
        raise SystemExit(f"{sheet} is open in Excel - close it and run again.")
    kept = sum(any(r[c] for c in YOUR_COLUMNS) for r in rows)
    print(f"Wrote {sheet}  ({len(rows)} cases, {kept} already reviewed)")
    return sheet


def expected_of(row: dict) -> tuple[str, str]:
    """(expected category, where it came from) or ('', reason it can't be scored)."""
    mine = row.get("correct_category", "").strip().lower()
    if mine:
        return (mine, "you") if mine in CATEGORIES else ("", f"unknown correct_category '{mine}'")
    rt = row.get("sf_request_type", "").strip().lower()
    if rt:
        return REQUEST_TYPE_MAP.get(rt, "not_an_order"), "salesforce"
    return "", "not reviewed yet"


def score(sheet: Path) -> None:
    rows = read_sheet(sheet)
    if not rows:
        raise SystemExit(f"No {sheet} - make it first (run without --score).")

    right, misses, skipped, by_cat = 0, [], 0, {}
    for row in rows.values():
        expected, src = expected_of(row)
        got = row["a2_category"]
        if not expected or got.startswith("(none"):
            skipped += 1
            if expected == "" and src.startswith("unknown"):
                print(f"  SKIP {row['case_id']}: {src}")
            continue
        ok = got == expected
        right += ok
        t = by_cat.setdefault(expected, [0, 0]); t[0] += ok; t[1] += 1
        if not ok:
            misses.append((row["case_id"], got, expected, src))

    total = right + len(misses)
    for case_id, got, expected, src in misses:
        print(f"  MISS {case_id}: A2 said {got}, correct is {expected} (from {src})")
    if not total:
        print("Nothing reviewed yet - fill correct_category or sf_request_type in review.csv.")
        return
    print(f"\nA2 accuracy: {right}/{total} = {right / total:.0%}   ({skipped} not reviewed)")
    for cat, (ok, n) in sorted(by_cat.items()):
        print(f"  {cat:<13} {ok}/{n}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Make or score the A2 review sheet")
    ap.add_argument("--out", default=str(ROOT / "output"))
    ap.add_argument("--sheet", help="where the review sheet lives (default: <out>/review.csv)")
    ap.add_argument("--score", action="store_true", help="score the filled-in review sheet")
    a = ap.parse_args()
    sheet = Path(a.sheet) if a.sheet else Path(a.out) / "review.csv"
    score(sheet) if a.score else make(Path(a.out), sheet)