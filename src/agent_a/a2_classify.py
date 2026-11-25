"""
A2 - Classify the email.
Reads output/*/case.json (written by A1), asks the LLM what the email is, applies a few rules,
and writes the result back into case.json.

  new_order     -> status "classified"   (goes on to A3)
  duplicate     -> status "classified"   + flag possible_duplicate (reviewer decides)
  revised_po    -> status "manual_queue" (Phase 1: changes handled by people)
  not_an_order  -> status "manual_queue"

Run from the src folder:
    python -m agent_a.a2_classify --out ../output --check ../data/answer_keys
"""
import argparse
import json
import os
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

from pydantic import BaseModel

from . import llm
from .clean import clean_email_text
from .config import CFG, ROOT
from .constants import A2_SYSTEM_PROMPT as SYSTEM, ARIBA_KIND, ARIBA_SENDER
from .models import Case, Classification
from .trace import prompt_version, step_trace

SAMPLES = (ROOT / CFG.paths.samples).resolve()
BODY_LIMIT = CFG.a2.body_limit_chars          # characters of email text sent to the model
LOW_CONFIDENCE = CFG.a2.low_confidence


# What we ask the model to return (every field required -> works with Groq strict mode)
class ClassifyOut(BaseModel):
    """What the model must return for one email (strict JSON schema)."""
    category: Literal["new_order", "revised_po", "duplicate", "not_an_order"]
    detail: str
    po_numbers: list[str]
    po_count: int
    confidence: float
    reason: str

def build_prompt(case: Case) -> str:
    """Build the text the model sees: subject, sender, attachment list and the cleaned email body."""
    files = [a for a in case.attachments if not a.likely_signature]
    att_lines = "\n".join(
        f"- {a.filename} ({a.kind}{', inside forwarded email ' + a.from_nested_email if a.from_nested_email else ''})"
        for a in files) or "- none"
    subj = re.sub(r"^\s*\[EXTERNAL\]\s*", "", case.subject, flags=re.I)   # mail-gateway tag hides "RE:"
    text = clean_email_text(case.body_text)                  # drop banners / long links before the model
    body = text[:BODY_LIMIT] + ("\n[...truncated]" if len(text) > BODY_LIMIT else "")
    return (f"Subject: {case.subject}\n"
            f"From: {case.from_name} <{case.from_address}>\n"
            f"Is a reply/forward: {'yes' if case.in_reply_to or subj.lower().startswith(('re:', 'fw:', 'fwd:')) else 'no'}\n"
            f"Attachments:\n{att_lines}\n\n"
            f"Email body:\n<<<\n{body}\n>>>")


def norm_po(po: str) -> str:
    """'PO# cbb-450123 ' -> 'CBB-450123' so the same PO matches however it's written."""
    # Strip a "PO" label only when a space, ':' or '#' follows it (or "No."),
    # so a PO number that really starts with "PO-" (e.g. PO-CO6795) stays whole.
    po = re.sub(r"^\s*P\.?\s?O\.?(?:\s*(?:NO\.?|NUMBER))?(?:\s*[:#]\s*|\s+|(?<=NO\.))", "",
                po.strip(), flags=re.I)
    return po.upper().replace(" ", "")


def audit(out_dir: Path, case_id: str, event: str, **details):
    """Append one A2 event to audit.jsonl."""
    with (out_dir / "audit.jsonl").open("a", encoding="utf-8") as f:
        f.write(json.dumps({"ts": datetime.now(timezone.utc).isoformat(), "case_id": case_id,
                            "step": "A2", "event": event, **details}, default=str) + "\n")

def known_format(case: Case):
    """Return (ClassifyOut, rule name) for an email format we recognise, else None."""
    if CFG.a2.use_ariba_rule and ARIBA_SENDER.search(case.from_address.strip()):
        m = ARIBA_KIND.search(case.body_text)
        if m:
            kind = m.group(1).lower()
            po = case.subject.replace("[EXTERNAL]", "").strip().split(",")[0].strip()
            if not re.fullmatch(r"[A-Za-z0-9-]{4,}", po):          # fall back to the PDF name
                pdfs = [a.filename for a in case.attachments if a.kind == "pdf"]
                po = pdfs[0].rsplit(".", 1)[0] if pdfs else ""
            category = "new_order" if kind == "new" else "revised_po"
            detail = {"new": "procurement_system_po", "changed": "order_change"}.get(kind, "cancellation")
            return ClassifyOut(category=category, detail=detail, po_numbers=[po] if po else [],
                               po_count=1 if po else 0, confidence=1.0,
                               reason=f"Rule: SAP Ariba notification says 'There is a {kind} order'."), "ariba"
    return None


def classify_case(case: Case, seen_pos: dict[str, str]) -> None:
    """Classify one case (fixed rule or model), check duplicate POs and set the next status."""
    known = known_format(case)
    if known:
        result, rule = known
        findings = []
        case.notes.append(f"A2: decided by fixed rule '{rule}' (no model call)")
        case.trace["a2"] = step_trace(decided_by=f"rule:{rule}")
    else:
        result, findings = llm.chat_json(SYSTEM, build_prompt(case), ClassifyOut)
        case.trace["a2"] = step_trace(SYSTEM, decided_by="model")
    c = Classification(**result.model_dump())
    # Clean PO numbers; drop empty or all-zero values (a model sometimes returns "0" when it can't see one)
    c.po_numbers = [n for n in (norm_po(p) for p in c.po_numbers) if n and not re.fullmatch(r"0+", n)]

    if findings:
        case.flags.append("guard_finding")
        case.notes += findings

    # Rule: a "new" PO we already processed in an earlier email is a possible duplicate (not auto-blocked)
    if c.category in ("new_order", "duplicate"):
        earlier = [seen_pos[p] for p in c.po_numbers if p in seen_pos]
        if earlier:
            c.category = "duplicate"
            case.duplicate_of = earlier[0]
            case.flags.append("possible_duplicate")
        elif c.category == "new_order":
            for p in c.po_numbers:
                seen_pos[p] = case.case_id

    if c.confidence < LOW_CONFIDENCE:
        case.flags.append("low_confidence")

    case.classification = c
    if c.category in ("new_order", "duplicate"):
        case.status = "classified"
    else:
        case.status = "manual_queue"
        case.notes.append(f"A2: {c.category} ({c.detail}) - {c.reason}")


def a2_failed(c: Case) -> bool:
    """True if A2 failed on this case before (e.g. rate limit), so it can be retried."""
    return c.status == "manual_queue" and not c.classification and bool(c.notes) and c.notes[-1].startswith("A2 failed")


def run(out_dir: Path, retry_failed: bool = False) -> list[Case]:
    """A2 for every case in status 'read', oldest email first so duplicates are found in order."""
    cases = [Case.model_validate_json(p.read_text(encoding="utf-8")) for p in out_dir.glob("*/case.json")]
    todo = [c for c in cases if c.status == "read" or (retry_failed and a2_failed(c))]   # skip duplicates / failed reads
    for c in todo:
        if a2_failed(c):                                        # retry: clear the old failure
            c.notes.pop(); c.status = "read"
    todo.sort(key=lambda c: (c.received_at is None, c.received_at))   # oldest first -> duplicates found in order

    # SAFETY: real emails go to a cloud model only if .env explicitly allows it (manager-approved)
    allow_real = os.getenv("ALLOW_EXTERNAL_LLM_FOR_REAL_DATA", "false").strip().lower() == "true"
    if llm.is_external() and not allow_real:
        real = [c.source_file for c in todo if SAMPLES not in Path(c.source_path).resolve().parents]
        if real:
            raise SystemExit(f"Refusing to send non-synthetic emails to {llm.PROVIDER}: {real[:3]}... "
                             f"Set ALLOW_EXTERNAL_LLM_FOR_REAL_DATA=true in .env only if approved.")

    seen_pos: dict[str, str] = {}                               # POs from cases already classified earlier
    for c in cases:
        if c not in todo and c.classification and c.classification.category == "new_order":
            for po in c.classification.po_numbers:
                seen_pos.setdefault(po, c.case_id)
    run_total = {"llm_calls": 0, "cache_hits": 0, "tokens_in": 0, "tokens_out": 0, "seconds": 0.0}
    for case in todo:
        llm.usage_reset()
        t0 = time.perf_counter()
        try:
            classify_case(case, seen_pos)
        except Exception as e:                                  # model down, bad JSON twice, etc.
            case.status = "manual_queue"
            case.notes.append(f"A2 failed: {e}")
        usage = {**llm.usage_take(), "seconds": round(time.perf_counter() - t0, 2)}   # tokens, cost, time
        if "a2" in case.trace:
            case.trace["a2"]["usage"] = usage
        for k in run_total:
            run_total[k] += usage[k]
        (out_dir / case.case_id / "case.json").write_text(case.model_dump_json(indent=2), encoding="utf-8")
        cl = case.classification
        audit(out_dir, case.case_id, case.status, model=llm.model_name(), prompt_version=prompt_version(SYSTEM),
              category=cl.category if cl else None, pos=cl.po_numbers if cl else [], flags=case.flags, **usage)
        print(f"{case.case_id}  {case.source_file[:40]:<40} -> {cl.category if cl else 'ERROR':<13} "
              f"{(cl.detail if cl else ''):<24} {','.join(cl.po_numbers) if cl else ''} {' '.join(case.flags)}")
    if todo:
        print(usage_line("A2", run_total))
    return todo


def usage_line(step: str, t: dict) -> str:
    """One summary line for a run: model calls, tokens, cost and time."""
    cost = llm.cost_usd(t["tokens_in"], t["tokens_out"])
    return (f"\n{step} usage: {t['llm_calls']} model calls, {t['tokens_in']:,} tokens in, {t['tokens_out']:,} out"
            + (f", ~${cost:.4f}" if cost is not None else "") + f", {t['seconds']:.0f}s"
            + (f", {t['cache_hits']} answer(s) from cache" if t.get("cache_hits") else ""))


def check(cases: list[Case], keys_dir: Path) -> None:
    """Compare with the answer keys (synthetic emails only)."""
    right = total = 0
    for c in cases:
        k = keys_dir / (Path(c.source_file).stem + ".json")
        if not k.exists() or not c.classification:
            continue
        expected = json.loads(k.read_text())["expected"]["classification"]
        total += 1; right += (c.classification.category == expected)
        if c.classification.category != expected:
            print(f"  MISS {c.source_file}: got {c.classification.category}, expected {expected}")
    if total:
        print(f"\nA2 accuracy: {right}/{total} = {right / total:.0%}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="A2 - classify emails")
    ap.add_argument("--out", default=str(ROOT / CFG.paths.output))
    ap.add_argument("--check", help="answer_keys folder to compare against")
    ap.add_argument("--retry-failed", action="store_true",
                    help="also re-run cases where A2 failed before (e.g. rate limit)")
    a = ap.parse_args()
    print(f"Provider: {llm.PROVIDER}  model: {llm.model_name()}\n")
    done = run(Path(a.out), a.retry_failed)
    if a.check:
        check(done, Path(a.check))