"""
Demo UI - run the pipeline from a web page and watch each step work.

    python -m agent_a.demo_ui --inbox ..\\data\\inbox\\demo --out ..\\output_real\\demo --no-db
    -> opens http://localhost:8765 in your browser

On the page:
  - Run pipeline: A1 -> A2 -> A3 -> A4 (-> DB). Each step shows what it is doing and one line per
    case as it finishes (no raw log).
  - Results: summary tiles and a table of every case; click a case (or "View payload") for the email,
    A2's decision, every PO field with its confidence, and the SAP payload.

Plain Python (no extra installs). By default it listens on this computer only (127.0.0.1).

Hosting (e.g. Azure App Service): every option can also come from an environment variable
(DEMO_INBOX, DEMO_OUT, DEMO_ACCOUNTS, DEMO_NO_DB=1, HOST, PORT). Set DEMO_PASSWORD to require a
password (browser login box, any user name). Put the site behind Microsoft sign-in as well.
"""
import argparse
import base64
import hmac
import io
import json
import os
import re
import sys
import threading
import time
import traceback
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from . import a1_read, a2_classify, a3_extract

ROOT = Path(__file__).resolve().parents[2]

#        id     title               what it does (shown while running)                     one-line description
STEP_INFO = [
    ("a1", "Pick up the case", "Picking up new cases and their attachments", "Email → Salesforce-shaped case"),
    ("a2", "Classify", "Classifying each email", "New order, revised, duplicate or not an order"),
    ("a3", "Read the PO", "Reading the purchase orders", "PDF, scan, Excel, Word → fields + confidence"),
    ("a4", "SAP payload", "Building the SAP payloads", "PO values in the RPA's JSON format"),
    ("db", "Save", "Saving everything to the database", "Store cases, POs and payloads"),
]
CATEGORY = {"new_order": "New order", "revised_po": "Revised PO", "duplicate": "Possible duplicate",
            "not_an_order": "Not an order"}


# ======================= reading the output folder =======================
def cases_in(out: Path) -> list[dict]:
    cases = []
    for p in sorted(out.glob("*/case.json")):
        try:
            c = json.loads(p.read_text(encoding="utf-8"))
        except (OSError, ValueError):                         # a step is writing this file right now
            continue
        c["_dir"], c["_mtime"] = str(p.parent), p.stat().st_mtime
        cases.append(c)
    return cases


def payload_files(out: Path) -> list[Path]:
    """Payload files only (not the <PO>.meta.json review files next to them)."""
    return [p for p in sorted(out.glob("*/payloads/*.json")) if not p.name.endswith(".meta.json")]


def meta_for(payload: Path) -> dict:
    """The review gate written next to a payload (<PO>.meta.json), or {} if there is none."""
    try:
        return json.loads(payload.with_name(payload.stem + ".meta.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def fv(x):
    if isinstance(x, dict) and "value" in x:
        return x.get("value"), x.get("confidence")
    return x, None


def a3_failed(c: dict) -> bool:
    return c.get("status") == "manual_queue" and (c.get("notes") or [""])[-1].startswith("A3 failed")


def order_case(c: dict) -> bool:
    return (c.get("classification") or {}).get("category") in ("new_order", "duplicate")


# ======================= what each step did (plain-language lines) =======================
def step_items(step: str, out: Path, since: float) -> list[dict]:
    """One line per case this step has finished since `since` - shown under the step on the page."""
    cs = [c for c in cases_in(out) if c["_mtime"] >= since]
    cs.sort(key=lambda c: c["_mtime"])
    items = []
    if step == "a1":
        for c in cs:
            n = len([a for a in c.get("attachments") or [] if not a.get("likely_signature")])
            extra = []
            if c.get("status") == "duplicate_file":
                extra.append("same file as " + str(c.get("duplicate_of")))
            if "unknown_sender" in (c.get("flags") or []):
                extra.append("customer not matched")
            items.append({"case": c["case_id"], "text": (c.get("subject") or "(no subject)")[:90],
                          "tag": f"{n} attachment{'s' if n != 1 else ''}" + (" · " + " · ".join(extra) if extra else ""),
                          "kind": "warn" if c.get("status") == "duplicate_file" else "ok"})
    elif step == "a2":
        for c in cs:
            cl = c.get("classification")
            if cl:
                pos = ", ".join(cl.get("po_numbers") or [])
                items.append({"case": c["case_id"], "text": (c.get("subject") or "")[:90],
                              "tag": CATEGORY.get(cl.get("category"), cl.get("category")) + (f" · PO {pos}" if pos else ""),
                              "kind": "ok" if order_case(c) else "muted"})
            elif c.get("status") == "manual_queue" and (c.get("notes") or [""])[-1].startswith("A2 failed"):
                items.append({"case": c["case_id"], "text": (c.get("subject") or "")[:90],
                              "tag": "not finished - model limit, retry later", "kind": "err"})
    elif step == "a3":
        for c in cs:
            if not order_case(c) or c.get("status") == "classified":
                continue
            pos = c.get("purchase_orders") or []
            if pos:
                for p in pos:
                    conf = p.get("overall_confidence") or 0
                    items.append({"case": c["case_id"], "text": f"PO {fv(p.get('po_number'))[0] or '?'} · "
                                  f"{len(p.get('lines') or [])} line(s) · read from {p.get('source_file')}",
                                  "tag": f"confidence {conf:.2f}",
                                  "kind": "ok" if conf >= 0.85 else "warn" if conf >= 0.6 else "err"})
            elif a3_failed(c):
                items.append({"case": c["case_id"], "text": (c.get("subject") or "")[:90],
                              "tag": "not finished - model limit, retry later", "kind": "err"})
            elif c.get("documents"):
                items.append({"case": c["case_id"], "text": (c.get("subject") or "")[:90],
                              "tag": "no purchase order found - goes to a person", "kind": "warn"})
    elif step == "a4":
        for p in payload_files(out):
            try:
                if p.stat().st_mtime < since:
                    continue
                j = json.loads(p.read_text(encoding="utf-8"))
            except (OSError, ValueError):                     # A4 is rewriting this file right now
                continue
            m = meta_for(p)
            its = j.get("Items") if isinstance(j.get("Items"), list) else []
            needs = m.get("review_status") == "needs_review"
            n = sum(r.get("severity") != "info" for r in m.get("reasons", []))
            items.append({"case": p.parent.parent.name, "text": f"SAP payload for PO {j.get('PO_Number') or p.stem}",
                          "tag": f"{len(its)} item(s) · " + (f"needs review ({n})" if needs else "ready for SAP"),
                          "kind": "warn" if needs else "ok"})
    return items


def progress_for(step: str, out: Path, since: float) -> str:
    cs = cases_in(out)
    if step == "a2":
        todo = [c for c in cs if c.get("status") != "duplicate_file"]
        done = [c for c in todo if c["_mtime"] >= since and c.get("status") != "read"]
        return f"{len(done)} of {len(todo)} emails"
    if step == "a3":
        todo = [c for c in cs if order_case(c) and c.get("status") in ("classified", "extracted", "manual_queue")]
        done = [c for c in todo if c["_mtime"] >= since and c.get("status") != "classified"]
        return f"{len(done)} of {len(todo)} orders"
    return ""


def summary_for(step: str, out: Path) -> str:
    cs = cases_in(out)
    if step == "a1":
        return (f"{len(cs)} cases · {sum(c.get('status') == 'duplicate_file' for c in cs)} duplicate file"
                + ("s" if sum(c.get('status') == 'duplicate_file' for c in cs) != 1 else ""))
    if step == "a2":
        cats = [(c.get("classification") or {}).get("category") for c in cs]
        parts = [f"{cats.count('new_order')} new orders"]
        for k, label in (("revised_po", "revised"), ("duplicate", "duplicate"), ("not_an_order", "not orders")):
            if cats.count(k):
                parts.append(f"{cats.count(k)} {label}")
        return " · ".join(parts)
    if step == "a3":
        pos = sum(len(c.get("purchase_orders") or []) for c in cs)
        failed = sum(a3_failed(c) for c in cs)
        return f"{pos} POs extracted" + (f" · {failed} to retry" if failed else "")
    if step == "a4":
        metas = [meta_for(p) for p in payload_files(out)]
        ready = sum(m.get("review_status") == "ready_for_sap" for m in metas)
        return f"{len(metas)} SAP payloads · {ready} ready · {len(metas) - ready} need review"
    return "saved"


# ======================= running the pipeline =======================
class Quiet(io.TextIOBase):
    """Swallows the steps' terminal prints (the page shows plain-language progress instead)."""
    def __init__(self, state):
        self.state, self.buf = state, ""

    def write(self, s):
        self.buf = (self.buf + s)[-4000:]                      # keep the tail, for error messages
        return len(s)


class State:
    def __init__(self, cfg):
        self.cfg, self.lock = cfg, threading.Lock()
        self.running, self.error = False, ""
        self.steps = self.fresh_steps()

    def fresh_steps(self):
        return [{"id": i, "status": "pending", "summary": "", "secs": None, "started": 0.0, "items": []}
                for i, *_ in STEP_INFO if not (i == "db" and self.cfg.no_db)]

    def snapshot(self):
        with self.lock:
            steps = [dict(s) for s in self.steps]
            snap = {"running": self.running, "error": self.error, "steps": steps}
        for s in steps:
            if s["status"] == "running":                       # live lines + "3 of 6" while it works
                s["items"] = step_items(s["id"], self.cfg.out, s["started"])
                s["progress"] = progress_for(s["id"], self.cfg.out, s["started"])
            s.pop("started", None)
        return snap


def run_step(step: str, cfg) -> None:
    out = cfg.out
    if step == "a1":
        a1_read.run(cfg.inbox, out, cfg.accounts)
    elif step == "a2":
        done = a2_classify.run(out, cfg.retry_failed)
        if cfg.check:
            a2_classify.check(done, cfg.check)
    elif step == "a3":
        a3_extract.run(out, cfg.retry_failed)
    elif step == "a4":
        from . import a4_payload
        a4_payload.run(out)
    elif step == "db":
        from . import db
        db.run(out)


def pipeline_thread(state: State, start: str) -> None:
    quiet, old = Quiet(state), sys.stdout
    sys.stdout = quiet
    try:
        ids = [s["id"] for s in state.steps]
        for s in state.steps:
            if ids.index(s["id"]) < ids.index(start):
                s["status"] = "skipped"
        for s in state.steps:
            if s["status"] == "skipped":
                continue
            with state.lock:
                s["status"], s["started"] = "running", time.time() - 1
            try:
                run_step(s["id"], state.cfg)
            except BaseException as e:
                with state.lock:
                    s["status"], s["secs"] = "failed", round(time.time() - s["started"] - 1, 1)
                    s["items"] = step_items(s["id"], state.cfg.out, s["started"])
                    # hide user:password in connection strings before the message reaches the page
                    state.error = f"{s['id'].upper()} stopped: {type(e).__name__}: " + re.sub(r"://[^@\s]+@", "://****@", str(e))
                old.write(traceback.format_exc())
                return
            items = step_items(s["id"], state.cfg.out, s["started"])
            with state.lock:
                s["status"], s["secs"] = "done", round(time.time() - s["started"] - 1, 1)
                s["items"], s["summary"] = items, summary_for(s["id"], state.cfg.out)
    finally:
        sys.stdout = old
        with state.lock:
            state.running = False


# ======================= results for the page =======================
def case_row(c: dict) -> dict:
    cl = c.get("classification") or {}
    pos = c.get("purchase_orders") or []
    po_nums = [str(fv(p.get("po_number"))[0]) for p in pos if fv(p.get("po_number"))[0]] or cl.get("po_numbers") or []
    confs = [p.get("overall_confidence") for p in pos if p.get("overall_confidence") is not None]
    d = Path(c["_dir"]) / "payloads"
    files = [p for p in sorted(d.glob("*.json")) if not p.name.endswith(".meta.json")] if d.exists() else []
    metas = [meta_for(p) for p in files]
    review = ("needs_review" if any(m.get("review_status") == "needs_review" for m in metas)
              else "ready_for_sap" if metas else None)
    first = next((r["message"] for m in metas for r in m.get("reasons", []) if r.get("severity") != "info"), "")
    return {"case_id": c.get("case_id"), "received": (c.get("received_at") or "")[:16].replace("T", " "),
            "sender": c.get("from_name") or c.get("from_address") or "", "subject": c.get("subject") or "",
            "category": cl.get("category"), "po_numbers": po_nums, "confidence": min(confs) if confs else None,
            "status": c.get("status"), "payloads": len(files), "review": review, "review_reason": first,
            "reason": (c.get("notes") or [""])[-1] if c.get("status") == "manual_queue" else ""}


def results(out: Path) -> dict:
    cs = cases_in(out)
    rows = [case_row(c) for c in cs]
    cats = [r["category"] for r in rows]
    metas = [meta_for(p) for p in payload_files(out)]
    ready = sum(m.get("review_status") == "ready_for_sap" for m in metas)
    return {"rows": rows, "tiles": [
        ["Emails", len(rows)], ["New orders", cats.count("new_order")], ["Revised POs", cats.count("revised_po")],
        ["Not an order", cats.count("not_an_order")], ["POs extracted", sum(len(c.get("purchase_orders") or []) for c in cs)],
        ["SAP payloads", len(metas)], ["Ready for SAP", ready], ["Needs review", len(metas) - ready],
        ["Need a person", sum(r["status"] == "manual_queue" for r in rows)]]}


def case_detail(out: Path, case_id: str) -> dict:
    d = out / Path(case_id).name
    c = json.loads((d / "case.json").read_text(encoding="utf-8"))
    files = [p for p in sorted((d / "payloads").glob("*.json")) if not p.name.endswith(".meta.json")] \
        if (d / "payloads").exists() else []
    c["_payloads"] = [{"name": p.name, "json": json.loads(p.read_text(encoding="utf-8")), "meta": meta_for(p)}
                      for p in files]
    return c


# ======================= web server =======================
class Handler(BaseHTTPRequestHandler):
    state: State = None
    password: str = ""

    def log_message(self, *a):
        pass

    def authorised(self) -> bool:
        """Optional access password (DEMO_PASSWORD): browser login box, any user name."""
        if not self.password:
            return True
        header = self.headers.get("Authorization", "")
        if header.startswith("Basic "):
            try:
                given = base64.b64decode(header[6:]).decode("utf-8").split(":", 1)[1]
            except Exception:
                given = ""
            if hmac.compare_digest(given, self.password):
                return True
        self.send_response(401)
        self.send_header("WWW-Authenticate", 'Basic realm="Order Intake demo"')
        self.send_header("Content-Length", "0")
        self.end_headers()
        return False

    def send(self, body, ctype="application/json", code=200):
        data = body.encode("utf-8") if isinstance(body, str) else json.dumps(body, default=str).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", ctype + "; charset=utf-8")
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        if not self.authorised():
            return
        u, st = urlparse(self.path), self.state
        q = parse_qs(u.query)
        try:
            if u.path == "/":
                self.send(PAGE.replace("__CONFIG__", json.dumps({
                    "inbox": str(st.cfg.inbox), "out": str(st.cfg.out), "no_db": st.cfg.no_db,
                    "steps": [list(x) for x in STEP_INFO if not (x[0] == "db" and st.cfg.no_db)]})), "text/html")
            elif u.path == "/state":
                self.send(st.snapshot())
            elif u.path == "/results":
                self.send(results(st.cfg.out))
            elif u.path == "/case":
                self.send(case_detail(st.cfg.out, q["id"][0]))
            else:
                self.send({"error": "not found"}, code=404)
        except Exception as e:
            self.send({"error": f"{type(e).__name__}: {e}"}, code=500)

    def do_POST(self):
        if not self.authorised():
            return
        u, st = urlparse(self.path), self.state
        if u.path != "/run":
            return self.send({"error": "not found"}, code=404)
        q = parse_qs(u.query)
        with st.lock:
            if st.running:
                return self.send({"error": "A run is already in progress."}, code=409)
            st.cfg.retry_failed = q.get("retry", ["0"])[0] == "1"
            st.running, st.error, st.steps = True, "", st.fresh_steps()
        start = q.get("from", [st.steps[0]["id"]])[0]
        threading.Thread(target=pipeline_thread, args=(st, start), daemon=True).start()
        self.send({"ok": True})


# ======================= the page =======================
PAGE = r"""<!DOCTYPE html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Order Intake Agent</title>
<style>
:root{--g:#6db33e;--gd:#4f8a28;--gl:#eef7e7;--ink:#1f2933;--sub:#5f6b7a;--line:#e3e8ee;--bg:#f4f6f8}
*{box-sizing:border-box}body{margin:0;font-family:"Segoe UI",Arial,sans-serif;background:var(--bg);color:var(--ink)}
.topbar{height:4px;background:var(--g)}
header{background:#fff;border-bottom:1px solid var(--line);padding:16px 36px;display:flex;justify-content:space-between;align-items:center}
.brand{display:flex;align-items:center;gap:14px}.mark{width:38px;height:38px;border-radius:9px;background:var(--g);color:#fff;
 display:flex;align-items:center;justify-content:center;font-weight:700;font-size:15px;letter-spacing:.5px}
header h1{margin:0;font-size:19px;font-weight:600}header p{margin:2px 0 0;font-size:12.5px;color:var(--sub)}
.pill{font-size:12px;color:var(--gd);background:var(--gl);border:1px solid #d3eac4;padding:5px 11px;border-radius:20px}
main{max-width:1320px;margin:0 auto;padding:24px 36px 60px}
.card{background:#fff;border:1px solid var(--line);border-radius:12px;padding:22px 26px;margin-bottom:18px}
.card h2{margin:0 0 4px;font-size:15px;font-weight:600}.card .hint{margin:0 0 16px;font-size:12.5px;color:var(--sub)}
.controls{display:flex;gap:12px;align-items:center;flex-wrap:wrap}
button{font:inherit;border:0;border-radius:8px;padding:10px 18px;cursor:pointer;font-weight:600}
.primary{background:var(--g);color:#fff}.primary:hover{background:var(--gd)}.primary:disabled{background:#b9d9a4;cursor:default}
.ghost{background:#fff;color:var(--gd);border:1px solid #cfe5c0}.ghost:hover{background:var(--gl)}
select,label{font-size:13px;color:var(--sub)}select{padding:7px 8px;border-radius:8px;border:1px solid var(--line);color:var(--ink)}
.stepper{display:flex;align-items:flex-start;margin:28px 0 6px}
.step{flex:1;text-align:center;position:relative;padding:0 6px}
.step:not(:last-child)::after{content:"";position:absolute;top:27px;left:calc(50% + 34px);right:calc(-50% + 34px);height:3px;border-radius:2px;background:var(--line)}
.step.done:not(:last-child)::after{background:var(--g)}
.circle{width:54px;height:54px;border-radius:50%;margin:0 auto 10px;display:flex;align-items:center;justify-content:center;
 font-weight:700;font-size:15px;border:2px solid #cdd5df;color:#94a0b0;background:#fff;position:relative;transition:.3s}
.step.running .circle{border-color:var(--g);color:var(--gd);background:var(--gl)}
.step.running .circle::before{content:"";position:absolute;inset:-8px;border-radius:50%;border:3px solid transparent;
 border-top-color:var(--g);border-right-color:var(--g);animation:spin 1s linear infinite}
.step.done .circle{background:var(--g);border-color:var(--g);color:#fff;font-size:24px}
.step.failed .circle{background:#d64545;border-color:#d64545;color:#fff;font-size:22px}
.step.skipped .circle{border-style:dashed;opacity:.5}
@keyframes spin{to{transform:rotate(360deg)}}
.step h4{margin:0;font-size:13.5px;font-weight:600}.step .desc{font-size:12px;color:var(--sub);margin-top:3px}
.step .sum{font-size:12.5px;font-weight:600;color:var(--gd);margin-top:7px;min-height:17px}
.step.failed .sum{color:#b02a2a}.step .secs{font-size:11px;color:#94a0b0}
.banner{border-radius:8px;padding:10px 14px;font-size:13.5px;margin-top:16px;display:none}
.banner.ok{display:block;background:var(--gl);color:var(--gd);border:1px solid #d3eac4}.banner.err{display:block;background:#fdecec;color:#a12626;border:1px solid #f4cccc}
.act{border-left:3px solid var(--line);margin:0 0 18px 10px;padding:0 0 2px 20px;position:relative}
.act.running{border-left-color:var(--g)}.act.done{border-left-color:var(--g)}.act.failed{border-left-color:#d64545}
.act .head{display:flex;align-items:center;gap:10px;margin-bottom:8px}
.act .tag{background:var(--g);color:#fff;font-size:11px;font-weight:700;border-radius:5px;padding:2px 7px}
.act.failed .tag{background:#d64545}
.act .what{font-size:14px;font-weight:600}.act .state{font-size:12.5px;color:var(--sub)}
.act .more{margin-left:auto;font-size:12.5px;color:var(--gd);cursor:pointer;font-weight:600}.act .more:hover{text-decoration:underline}
.act.done:not(:last-child){margin-bottom:12px}
.dots::after{content:"";animation:dots 1.4s steps(4,end) infinite}@keyframes dots{0%{content:""}25%{content:"."}50%{content:".."}75%{content:"..."}}
.line{display:grid;grid-template-columns:130px 1fr auto;gap:14px;align-items:center;padding:7px 12px;border-radius:8px;font-size:13px;
 background:#fafbfc;margin-bottom:5px}.line.new{animation:fade .5s ease}
@keyframes fade{from{opacity:0;transform:translateY(-3px)}to{opacity:1}}
.line .cid{font-family:Consolas,monospace;font-size:12px;color:var(--sub)}
.k{font-size:12px;font-weight:600;padding:2px 9px;border-radius:12px;white-space:nowrap}
.k.ok{background:var(--gl);color:var(--gd)}.k.warn{background:#fff3dc;color:#8a5a00}.k.err{background:#fdecec;color:#a12626}.k.muted{background:#eef0f3;color:#5f6b7a}
.empty{font-size:13px;color:#94a0b0;padding:4px 0 8px}
.tiles{display:flex;gap:12px;flex-wrap:wrap;margin-bottom:16px}
.tile{flex:1;min-width:120px;border:1px solid var(--line);border-radius:10px;padding:12px 16px}
.tile b{display:block;font-size:24px;font-weight:600}.tile span{font-size:12px;color:var(--sub)}
.tile.hl{border-color:#cfe5c0;background:var(--gl)}.tile.hl b{color:var(--gd)}
table{width:100%;border-collapse:collapse;font-size:13px}
th{text-align:left;color:var(--sub);font-weight:600;padding:9px 10px;border-bottom:1px solid var(--line);font-size:12px;background:#fafbfc}
td{padding:10px;border-bottom:1px solid #eef1f4;vertical-align:top}
#rows td:nth-child(1),#rows td:nth-child(2){white-space:nowrap}
tbody tr{cursor:pointer}tbody tr:hover{background:#f7fbf3}tbody tr.sel{background:var(--gl)}
.badge{display:inline-block;padding:2px 9px;border-radius:12px;font-size:12px;font-weight:600;white-space:nowrap}
.new_order{background:var(--gl);color:var(--gd)}.revised_po{background:#fff3dc;color:#8a5a00}
.duplicate{background:#efeafb;color:#5a3ea8}.not_an_order{background:#eef0f3;color:#4a5263}.nocat{background:#fdecec;color:#a12626}
.conf{display:inline-block;min-width:42px;text-align:center;padding:1px 6px;border-radius:6px;font-weight:600;font-size:12px}
.hi{background:var(--gl);color:var(--gd)}.mid{background:#fff3dc;color:#8a5a00}.lo{background:#fdecec;color:#a12626}
.muted{color:#94a0b0}.flag{display:inline-block;background:#fff3dc;color:#8a5a00;border-radius:6px;padding:1px 7px;margin:1px 3px 1px 0;font-size:11px}
button.view{background:var(--g);color:#fff;padding:4px 11px;border-radius:7px;font-size:12px;white-space:nowrap}
.why{font-size:11px;color:#a12626;margin-top:3px;max-width:230px}.why.info{color:#94a0b0}
.legend{font-size:12px;color:var(--sub);margin:0 0 10px}
#shade{position:fixed;inset:0;background:#1f293340;display:none;z-index:9}
#detail{position:fixed;top:0;right:0;bottom:0;width:min(960px,92vw);background:#fff;overflow:auto;z-index:10;display:none;
 box-shadow:-10px 0 30px #1f293326;padding:0 30px 30px}
#detail .dhead{position:sticky;top:0;background:#fff;padding:20px 0 0;border-bottom:1px solid var(--line);z-index:2}
#detail h3{font-size:15px;margin:0 90px 12px 0}
.close{position:absolute;right:0;top:16px;background:#fff;border:1px solid var(--line);color:var(--ink);padding:6px 12px;font-size:13px}
.tabs{display:flex;gap:4px}
.tab{padding:9px 14px;font-size:13px;font-weight:600;color:var(--sub);cursor:pointer;border-bottom:3px solid transparent}
.tab.on{color:var(--gd);border-bottom-color:var(--g)}
.sec{font-size:13.5px;font-weight:600;margin:20px 0 10px;color:var(--ink)}
.kv td:first-child{color:var(--sub);width:200px}.quote{font-family:Consolas,monospace;font-size:11.5px;color:var(--sub)}
.email{background:#fafbfc;border:1px solid var(--line);border-radius:8px;padding:12px 14px;white-space:pre-wrap;font-size:12.5px;max-height:340px;overflow:auto}
pre.json{background:#1f2933;color:#e3ebf3;border-radius:8px;padding:12px 14px;font-size:12px;overflow:auto;max-height:440px}
.paybox{border:1px solid #cfe5c0;border-radius:10px;padding:16px 18px;margin:16px 0}
.paybox .ptitle{display:flex;justify-content:space-between;align-items:center;margin-bottom:10px}
.paybox .ptitle b{font-size:14px}
</style></head><body>
<div class="topbar"></div>
<header><div class="brand"><div class="mark">OI</div><div><h1>Order Intake Agent</h1><p>Agent A · Email order intake · Prototype</p></div></div>
<div class="pill" id="pill"></div></header>
<main>
<div class="card">
  <h2>Run the agent</h2><p class="hint" id="paths"></p>
  <div class="controls">
    <button class="primary" id="run">▶ Run pipeline</button>
    <label>Start from <select id="from"></select></label>
    <label><input type="checkbox" id="retry"> Retry cases that did not finish</label>
    <button class="ghost" id="show">Show results</button>
  </div>
  <div class="stepper" id="stepper"></div>
  <div class="banner" id="banner"></div>
</div>
<div class="card" id="actCard" style="display:none"><h2>What the agent is doing</h2><p class="hint">Each step, and every case as it finishes.</p><div id="activity"></div></div>
<div class="card" id="resultsCard" style="display:none">
  <h2>Results</h2><p class="hint">Click a case for the full detail, or <b>View payload</b> for the SAP order payload. Every case goes to a reviewer: <b>Ready for SAP</b> = nothing found, confirm only · <b>Needs review</b> = check the reasons.</p>
  <div class="tiles" id="tiles"></div>
  <p class="legend">Confidence <span class="conf hi">0.85+</span> trusted &nbsp;<span class="conf mid">0.60–0.85</span> check &nbsp;<span class="conf lo">&lt;0.60</span> doubtful</p>
  <table><thead><tr><th>Case</th><th>Received</th><th>From</th><th>Subject</th><th>Decision</th><th>PO number</th><th>Confidence</th><th>SAP payload</th><th>Status</th></tr></thead><tbody id="rows"></tbody></table>
</div>
</main>
<div id="shade"></div><div id="detail"></div>
<script>
const CFG = __CONFIG__;
const CAT = {new_order:"New order", revised_po:"Revised PO", duplicate:"Possible duplicate", not_an_order:"Not an order"};
const STATUS = {read:"Read", duplicate_file:"Duplicate file", classified:"Classified", extracted:"PO extracted", payload_built:"Payload ready", manual_queue:"Needs a person"};
const HEAD = [["po_number","PO number"],["po_date","PO date"],["requested_delivery_date","Requested delivery"],["customer_name","Customer (on PO)"],
 ["buyer_name","Buyer"],["buyer_phone","Buyer phone"],["buyer_email","Buyer email"],["vendor_name","Addressed to"],["ship_to_name","Ship-to name"],
 ["ship_to_address","Ship-to address"],["bill_to","Bill-to"],["currency","Currency"],["incoterms","Incoterms"],["payment_terms","Payment terms"],["notes","Notes"],["total","Total"]];
const LINE = [["line_no","Line"],["customer_material","Customer material"],["supplier_material","Our material"],["description","Description"],
 ["quantity","Qty"],["uom","UOM"],["unit_price","Price"],["price_per","Per"],["line_total","Line total"],["delivery_date","Delivery"]];
const $ = s => document.querySelector(s);
const esc = s => String(s ?? "").replace(/[&<>"']/g, c => ({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;","'":"&#39;"}[c]));
const conf = c => c == null ? "" : `<span class="conf ${c>=0.85?"hi":c>=0.6?"mid":"lo"}">${Number(c).toFixed(2)}</span>`;
const badge = c => c ? `<span class="badge ${c}">${CAT[c]||esc(c)}</span>` : `<span class="badge nocat">–</span>`;
const val = f => (f && typeof f === "object" && "value" in f) ? f : {value: f, confidence: null, source: {}};
const num = v => typeof v === "number" ? v.toLocaleString(undefined,{maximumFractionDigits:4}) : esc(v);
const reviewBadge = r => r === "needs_review" ? '<span class="k warn">Needs review</span>' : '<span class="k ok">Ready for SAP</span>';
const SEV = {critical: "err", warning: "warn", info: "muted"};

$("#pill").textContent = CFG.no_db ? "Offline run · database off" : "Offline run · saving to database";
$("#paths").textContent = `Emails from ${CFG.inbox}  ·  results in ${CFG.out}`;
$("#from").innerHTML = CFG.steps.map(s => `<option value="${s[0]}">${s[0].toUpperCase()} · ${esc(s[1])}</option>`).join("");

function drawSteps(steps){
  $("#stepper").innerHTML = CFG.steps.map(([id,title,doing,desc]) => {
    const s = (steps||[]).find(x => x.id === id) || {status:"pending"};
    const icon = s.status==="done" ? "✓" : s.status==="failed" ? "✕" : id.toUpperCase();
    const sum = s.status==="running" ? `Working${s.progress?" · "+esc(s.progress):""}` : s.status==="skipped" ? "Skipped" : esc(s.summary||"");
    return `<div class="step ${s.status}"><div class="circle">${icon}</div><h4>${id.toUpperCase()} · ${esc(title)}</h4>
      <div class="desc">${esc(desc)}</div><div class="sum">${sum}</div><div class="secs">${s.secs!=null?s.secs+" s":""}</div></div>`;
  }).join("");
}
function drawActivity(steps){
  const shown = (steps||[]).filter(s => ["running","done","failed"].includes(s.status));
  $("#actCard").style.display = shown.length ? "block" : "none";
  $("#activity").innerHTML = shown.map(s => {
    const info = CFG.steps.find(x => x[0] === s.id) || [s.id, s.id, s.id];
    const state = s.status==="running" ? `<span class="dots">${esc(info[2])}${s.progress?" · "+esc(s.progress):""}</span>`
                : s.status==="failed" ? "Stopped" : `Done${s.summary?" · "+esc(s.summary):""}${s.secs!=null?" · "+s.secs+" s":""}`;
    const lines = (s.items||[]).map(i => { const key = s.id+"|"+i.case+"|"+i.text, fresh = !SEEN.has(key); SEEN.add(key);
      return `<div class="line${fresh?" new":""}"><span class="cid">${esc(i.case)}</span><span>${esc(i.text)}</span><span class="k ${i.kind}">${esc(i.tag)}</span></div>`; }).join("");
    const body = lines || (s.status==="running" ? `<div class="empty">Starting…</div>` : `<div class="empty">Nothing to do in this step.</div>`);
    const n = (s.items||[]).length;
    const open = s.status !== "done" || OPEN.has(s.id);      // finished steps fold to one line
    const toggle = s.status === "done" && n ? `<span class="more" data-step="${s.id}">${open ? "Hide" : "Show"} ${n} case${n>1?"s":""}</span>` : "";
    return `<div class="act ${s.status}"><div class="head"><span class="tag">${s.id.toUpperCase()}</span><span class="what">${esc(info[1])}</span><span class="state">${state}</span>${toggle}</div>
      ${open ? body : ""}</div>`;
  }).join("");
  document.querySelectorAll("#activity .more").forEach(m => m.onclick = () => {
    OPEN.has(m.dataset.step) ? OPEN.delete(m.dataset.step) : OPEN.add(m.dataset.step); drawActivity(LAST);
  });
  LAST = steps;
}
const OPEN = new Set(), SEEN = new Set(); let LAST = [];
drawSteps([]);

let timer = null;
async function poll(){
  const r = await fetch("/state").then(r => r.json());
  drawSteps(r.steps); drawActivity(r.steps);
  if (!r.running){
    clearInterval(timer); timer = null; $("#run").disabled = false;
    const b = $("#banner");
    if (r.error){ b.className = "banner err"; b.textContent = r.error + "  ·  Tick “Retry cases that did not finish”, choose the step and run again."; }
    else if (r.steps.some(s => s.status === "done")){ b.className = "banner ok"; b.textContent = "Run finished. Results are below."; }
    if (r.steps.some(s => s.status === "done")) loadResults();
  }
}
// on run start, forget which finished steps were expanded

$("#run").onclick = async () => {
  $("#run").disabled = true; $("#banner").className = "banner"; OPEN.clear(); SEEN.clear();
  const res = await fetch(`/run?from=${$("#from").value}&retry=${$("#retry").checked?1:0}`, {method:"POST"});
  if (!res.ok){ $("#run").disabled = false; alert((await res.json()).error); return; }
  timer = setInterval(poll, 800); poll();
};
$("#show").onclick = () => loadResults().then(() => $("#resultsCard").scrollIntoView({behavior:"smooth"}));

async function loadResults(){
  const r = await fetch("/results").then(r => r.json());
  $("#resultsCard").style.display = "block";
  $("#tiles").innerHTML = r.tiles.map(([k,v]) => `<div class="tile ${k==="Ready for SAP"?"hl":""}"><b>${v}</b><span>${esc(k)}</span></div>`).join("");
  $("#rows").innerHTML = r.rows.map(x => `<tr data-id="${esc(x.case_id)}"><td><b>${esc(x.case_id)}</b></td><td>${esc(x.received)}</td>
    <td>${esc(x.sender).slice(0,40)}</td><td>${esc(x.subject).slice(0,100)}</td><td>${badge(x.category)}</td>
    <td>${esc(x.po_numbers.join(", "))||'<span class="muted">–</span>'}</td><td>${conf(x.confidence)||'<span class="muted">–</span>'}</td>
    <td>${x.payloads ? `<button class="view">View payload${x.payloads>1?" ("+x.payloads+")":""}</button>` : '<span class="muted">–</span>'}</td>
    <td>${x.review ? reviewBadge(x.review) : esc(STATUS[x.status]||x.status)}${x.review_reason?`<div class="why" title="${esc(x.review_reason)}">${esc(x.review_reason).slice(0,80)}</div>`:""}${x.reason?`<div class="why ${/^A\d failed/.test(x.reason)?"":"info"}" title="${esc(x.reason)}">${esc(x.reason.replace(/^A\d: /,"")).slice(0,80)}</div>`:""}</td></tr>`).join("")
    || `<tr><td colspan="9" class="muted">No cases yet. Run the pipeline first.</td></tr>`;
  document.querySelectorAll("#rows tr[data-id]").forEach(tr => tr.onclick = ev => {
    document.querySelectorAll("#rows tr").forEach(t => t.classList.remove("sel")); tr.classList.add("sel");
    showCase(tr.dataset.id, ev.target.closest("button.view") ? "payload" : "email");
  });
}

async function showCase(id, openTab){
  const c = await fetch(`/case?id=${encodeURIComponent(id)}`).then(r => r.json());
  const cl = c.classification || {}, pos = c.purchase_orders || [];
  const atts = (c.attachments||[]).filter(a => !a.likely_signature).map(a => `${esc(a.filename)} <span class="muted">(${esc(a.kind)})</span>`).join(", ") || "none";
  const flags = (c.flags||[]).map(f => `<span class="flag">${esc(f)}</span>`).join("");
  const tabs = [["payload","SAP payload"],["po",`Purchase order${pos.length>1?"s":""}`],["decision","Decision"],["email","Email"]];
  const panes = {
    email: `<table class="kv"><tr><td>From</td><td>${esc(c.from_name)} &lt;${esc(c.from_address)}&gt;</td></tr>
      <tr><td>Received</td><td>${esc((c.received_at||"").slice(0,16).replace("T"," "))}</td></tr><tr><td>Subject</td><td>${esc(c.subject)}</td></tr>
      <tr><td>Attachments</td><td>${atts}</td></tr><tr><td>Customer (Salesforce)</td><td>${c.account_name?esc(c.account_name)+(c.sap_account_number?" · SAP "+esc(c.sap_account_number):""):'<span class="muted">not matched yet</span>'}</td></tr>
      ${flags?`<tr><td>Flags</td><td>${flags}</td></tr>`:""}</table><div class="email" style="margin-top:14px">${esc((c.body_text||"").slice(0,4000))}</div>`,
    decision: !c.classification ? `<p class="muted">Not classified.</p>` :
      `<table class="kv"><tr><td>Decision</td><td>${badge(cl.category)} <span class="muted">${esc(cl.detail)}</span></td></tr>
      <tr><td>Confidence</td><td>${conf(cl.confidence)}</td></tr><tr><td>PO numbers seen</td><td>${esc((cl.po_numbers||[]).join(", "))||"–"}</td></tr>
      <tr><td>Why</td><td>${esc(cl.reason)}</td></tr></table>` + ((c.notes||[]).length?`<div class="sec">Notes</div><ul>${c.notes.map(n=>`<li>${esc(n)}</li>`).join("")}</ul>`:""),
    po: pos.length ? pos.map((p,i) => poHtml(p,i,pos.length)).join("") : `<p class="muted">No purchase order extracted for this case.</p>`,
    payload: (c._payloads||[]).length ? c._payloads.map(payloadHtml).join("")
      : `<p class="muted">No SAP payload for this case${c.status==="manual_queue"&&(c.notes||[]).length?": "+esc(c.notes[c.notes.length-1]):"."}</p>`
  };
  const first = openTab && panes[openTab] ? openTab : ((c._payloads||[]).length ? "payload" : "email");
  const d = $("#detail");
  d.innerHTML = `<div class="dhead"><button class="close" id="closeBtn">✕ Close</button><h3>${esc(c.case_id)} · ${esc(c.subject)}</h3>
    <div class="tabs">${tabs.map(([k,l]) => `<div class="tab ${k===first?"on":""}" data-k="${k}">${l}</div>`).join("")}</div></div><div id="pane">${panes[first]}</div>`;
  d.style.display = "block"; $("#shade").style.display = "block"; d.scrollTop = 0;
  d.querySelectorAll(".tab").forEach(t => t.onclick = () => { d.querySelectorAll(".tab").forEach(x => x.classList.remove("on")); t.classList.add("on"); $("#pane").innerHTML = panes[t.dataset.k]; });
  $("#closeBtn").onclick = closeCase;
}
function closeCase(){ $("#detail").style.display = "none"; $("#shade").style.display = "none"; document.querySelectorAll("#rows tr").forEach(t => t.classList.remove("sel")); }
$("#shade").onclick = closeCase;
document.addEventListener("keydown", e => { if (e.key === "Escape") closeCase(); });

function payloadHtml(p){
  const j = p.json || {}, items = Array.isArray(j.Items) ? j.Items : (j.Items ? [j.Items] : []);
  const m = p.meta || {}, rs = m.reasons || [];
  const reasons = rs.length ? `<div style="margin:0 0 12px">${rs.map(r => `<div style="margin:4px 0"><span class="k ${SEV[r.severity]||"muted"}">${esc(r.severity)}</span> ${esc(r.message)}</div>`).join("")}</div>`
    : (m.review_status ? `<p class="muted" style="font-size:12.5px;margin:0 0 12px">No problems found. The reviewer only needs to confirm.</p>` : "");
  const filled = Object.entries(j).filter(([k,v]) => k !== "Items" && v !== "" && v != null && v !== "NULL");
  const empty = Object.entries(j).filter(([k,v]) => k !== "Items" && (v === "" || v == null));
  const icols = items.length ? Object.keys(items[0]).filter(k => items.some(it => it[k] !== "" && it[k] != null)) : [];
  return `<div class="paybox"><div class="ptitle"><b>SAP order payload · PO ${esc(j.PO_Number || p.name)}</b>${m.review_status ? reviewBadge(m.review_status) : '<span class="k ok">ready for SAP simulation</span>'}</div>${reasons}
    <table class="kv"><tbody>${filled.map(([k,v]) => `<tr><td>${esc(k)}</td><td><b>${esc(v)}</b></td></tr>`).join("")}</tbody></table>
    ${items.length?`<div class="sec">Items</div><table><thead><tr>${icols.map(k=>`<th>${esc(k)}</th>`).join("")}</tr></thead><tbody>${items.map(it=>`<tr style="cursor:default">${icols.map(k=>`<td>${esc(it[k])}</td>`).join("")}</tr>`).join("")}</tbody></table>`:""}
    ${empty.length?`<p class="muted" style="font-size:12px;margin-top:12px">Left empty (filled later by Salesforce lookup or SAP): ${empty.map(([k])=>esc(k)).join(", ")}</p>`:""}
    <details style="margin-top:10px"><summary style="cursor:pointer;font-size:13px;color:var(--gd)">Show the JSON exactly as sent</summary><pre class="json">${esc(JSON.stringify(j,null,2))}</pre></details></div>`;
}
function poHtml(p, i, n){
  const rows = HEAD.map(([k,l]) => { const f = val(p[k]); if (f.value==null) return "";
    return `<tr><td>${l}</td><td>${num(f.value)}</td><td>${conf(f.confidence)}</td><td class="quote">${esc((f.source||{}).quote||"").slice(0,110)}</td></tr>`; }).join("");
  const lines = p.lines || [], cols = LINE.filter(([k]) => lines.some(l => val(l[k]).value != null));
  const ltable = lines.length ? `<table><thead><tr>${cols.map(([,l])=>`<th>${l}</th>`).join("")}</tr></thead><tbody>${lines.map(l =>
    `<tr style="cursor:default">${cols.map(([k]) => { const f = val(l[k]); return `<td>${f.value!=null?num(f.value)+"<br>"+conf(f.confidence):'<span class="muted">–</span>'}</td>`; }).join("")}</tr>`).join("")}</tbody></table>` : `<p class="muted">No lines found.</p>`;
  const flags = (p.flags||[]).map(f => `<span class="flag">${esc(f)}</span>`).join(""), miss = (p.missing_fields||[]).join(", ");
  return `<div class="sec">${n>1?`PO ${i+1} of ${n} · `:""}Overall confidence ${conf(p.overall_confidence)} <span class="muted" style="font-weight:400;font-size:12.5px">read from ${esc(p.source_file)} (${esc(p.read_method)})</span></div>
    ${flags?`<p>Flags: ${flags}</p>`:""}${miss?`<p>Missing: <b>${esc(miss)}</b></p>`:""}
    <table class="kv"><thead><tr><th>Field</th><th>Value</th><th>Confidence</th><th>Read from</th></tr></thead><tbody>${rows}</tbody></table>
    <div class="sec">Lines</div>${ltable}<div style="height:16px"></div>`;
}

async function openFromHash(){
  if (location.hash === "#results") return loadResults();
  const m = location.hash.match(/^#case=([^&]+)(?:&tab=(\w+))?$/);
  if (m){ await loadResults(); showCase(decodeURIComponent(m[1]), m[2]); }
}
fetch("/state").then(r => r.json()).then(r => {
  drawSteps(r.steps); drawActivity(r.steps);
  if (r.running){ $("#run").disabled = true; timer = setInterval(poll, 800); }
  else if (r.steps.some(s => s.status === "done")) loadResults().then(openFromHash);
  else openFromHash();
});
</script></body></html>"""


def main():
    env = os.environ.get
    ap = argparse.ArgumentParser(description="Order Intake Agent - demo UI")
    ap.add_argument("--inbox", default=env("DEMO_INBOX", str(ROOT / "data" / "samples")))
    ap.add_argument("--out", default=env("DEMO_OUT", str(ROOT / "output")))
    ap.add_argument("--accounts", default=env("DEMO_ACCOUNTS", str(ROOT / "data" / "mock_accounts.json")))
    ap.add_argument("--check", default=env("DEMO_CHECK"), help="answer_keys folder (synthetic emails only)")
    ap.add_argument("--no-db", action="store_true", default=env("DEMO_NO_DB", "") in ("1", "true", "yes"),
                    help="hide and skip the database step")
    ap.add_argument("--host", default=env("HOST", "127.0.0.1"), help="0.0.0.0 when hosted (App Service)")
    ap.add_argument("--port", type=int, default=int(env("PORT", env("WEBSITES_PORT", "8765"))))
    a = ap.parse_args()
    a.inbox, a.out, a.accounts = Path(a.inbox), Path(a.out), Path(a.accounts)
    a.check = Path(a.check) if a.check else None
    a.retry_failed = False
    a.out.mkdir(parents=True, exist_ok=True)

    Handler.state = State(a)
    Handler.password = env("DEMO_PASSWORD", "")
    server = ThreadingHTTPServer((a.host, a.port), Handler)
    url = f"http://localhost:{a.port}"
    hosted = bool(env("WEBSITE_SITE_NAME")) or a.host not in ("127.0.0.1", "localhost")
    print(f"Order Intake demo UI on {a.host}:{a.port}   inbox={a.inbox}   out={a.out}   "
          f"password={'on' if Handler.password else 'off'}   (Ctrl+C to stop)", flush=True)
    if not hosted:
        threading.Timer(1.0, lambda: webbrowser.open(url)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopped.")


if __name__ == "__main__":
    main()