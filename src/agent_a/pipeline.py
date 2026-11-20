"""
Agent A pipeline - runs every step in order with one command:

    A1 read -> A2 classify -> A3 extract PO -> A4 SAP payload -> save to Postgres

Run from src/:
    python -m agent_a.pipeline --out ..\\output --check ..\\data\\answer_keys

Continue after a stop (e.g. Groq daily limit in A3) without starting over:
    python -m agent_a.pipeline --out ..\\output --from a3 --retry-failed

Note: A1 starts every case clean, so a run from a1 redoes all steps (and uses LLM tokens again).
"""
import argparse
from pathlib import Path

from . import a1_read, a2_classify, a3_extract, a4_payload, llm
from .config import CFG, ROOT
from .models import Case

STEPS = ["a1", "a2", "a3", "a4", "db"]


def banner(title: str) -> None:
    """Print a clear header so each step's output is easy to find."""
    print(f"\n{'=' * 70}\n{title}\n{'=' * 70}")


def load_cases(out_dir: Path) -> list[Case]:
    """Every case.json in the output folder."""
    return [Case.model_validate_json(p.read_text(encoding="utf-8")) for p in out_dir.glob("*/case.json")]


def step_a1(a) -> None:
    """A1: email files -> Salesforce-shaped records + case.json."""
    banner("A1 - read emails")
    cases = a1_read.run(Path(a.inbox), a.out, Path(a.accounts))
    print(f"\n{len(cases)} files: {sum(c.status == 'read' for c in cases)} read, "
          f"{sum(c.status == 'duplicate_file' for c in cases)} duplicate, "
          f"{sum(c.status == 'manual_queue' for c in cases)} failed, "
          f"{sum('unknown_sender' in c.flags for c in cases)} unknown sender")


def step_a2(a) -> None:
    """A2: classify every case in status 'read'."""
    banner(f"A2 - classify   (model: {llm.model_name()})")
    done = a2_classify.run(a.out, a.retry_failed)
    if a.check:
        a2_classify.check(done, Path(a.check))


def step_a3(a) -> None:
    """A3: extract the PO from every classified case; tells you how to continue if the API stopped it."""
    banner(f"A3 - extract PO   (model: {llm.model_name()})")
    done = a3_extract.run(a.out, a.retry_failed)
    print(f"\nThis run: {len(done)} cases: {sum(c.status == 'extracted' for c in done)} extracted, "
          f"{sum(c.status == 'manual_queue' for c in done)} manual queue")
    if a.check:
        print()
        a3_extract.check([c for c in load_cases(a.out) if c.documents], Path(a.check))


def step_a4(a) -> None:
    """A4: one SAP payload per extracted PO."""
    banner("A4 - SAP payload")
    done = a4_payload.run(a.out)
    print(f"\n{len(done)} case(s), {sum(len(c.purchase_orders) for c in done)} payload(s) built")
    if a.check:
        a4_payload.check(a.out, Path(a.check))


def step_db(a) -> None:
    """Save everything in the output folder to Postgres (DATABASE_URL in .env)."""
    banner("DB - save to Postgres")
    from . import db                      # imported here so the pipeline still runs with --no-db and no psycopg
    db.run(a.out)


def final_summary(out_dir: Path) -> None:
    """Count cases by status and point out anything that still needs a retry."""
    banner("SUMMARY")
    cases = load_cases(out_dir)
    by_status: dict[str, int] = {}
    for c in cases:
        by_status[c.status] = by_status.get(c.status, 0) + 1
    for status, n in sorted(by_status.items(), key=lambda x: -x[1]):
        print(f"  {status:<15} {n}")
    a2_left = [c.case_id for c in cases if a2_classify.a2_failed(c)]
    a3_left = [c.case_id for c in cases if a3_extract.a3_failed(c)]
    if a2_left or a3_left:
        first = "a2" if a2_left else "a3"
        print(f"\n  {len(a2_left)} case(s) failed in A2, {len(a3_left)} in A3 (usually the API limit). Later run:\n"
              f"    python -m agent_a.pipeline --out {out_dir} --from {first} --retry-failed")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Agent A: run A1 -> A4 and save to Postgres")
    ap.add_argument("--inbox", default=str(ROOT / CFG.paths.samples))
    ap.add_argument("--out", default=str(ROOT / CFG.paths.output))
    ap.add_argument("--accounts", default=str(ROOT / CFG.paths.accounts))
    ap.add_argument("--check", help="answer_keys folder (synthetic emails only)")
    ap.add_argument("--from", dest="start", choices=STEPS, default="a1", help="step to start from (default a1)")
    ap.add_argument("--retry-failed", action="store_true", help="also redo cases that failed in A2/A3 before")
    ap.add_argument("--no-db", action="store_true", help="skip saving to Postgres")
    a = ap.parse_args()
    a.out = Path(a.out)
    a.out.mkdir(parents=True, exist_ok=True)

    run_step = {"a1": step_a1, "a2": step_a2, "a3": step_a3, "a4": step_a4, "db": step_db}
    for name in STEPS[STEPS.index(a.start):]:
        if name == "db" and a.no_db:
            continue
        run_step[name](a)
    final_summary(a.out)