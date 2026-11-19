"""
Postgres storage (Azure Database for PostgreSQL).

Copies everything A1-A4 wrote in the output folder into Postgres:
  Salesforce records as received -> sf_case, sf_email_message, sf_attachment
  our work on each case          -> agent_case (A1+A2), purchase_order + po_line (A3), sap_payload (A4)
  history                        -> audit_event
  where the next Salesforce poll starts -> poll_pointer
Safe to run again and again: each case is replaced, never duplicated.
Connection comes from DATABASE_URL in the project's .env.

Run from src/ (after any step):
    python -m agent_a.db --out ..\\output
"""
import argparse
import json
import os
import re
from pathlib import Path

import psycopg
from dotenv import load_dotenv
from psycopg.types.json import Jsonb

from .a4_payload import missing
from .models import Case, FieldValue

ROOT = Path(__file__).resolve().parents[2]
load_dotenv(ROOT / ".env")

SCHEMA = """
CREATE TABLE IF NOT EXISTS sf_case (
    id                 text PRIMARY KEY,          -- Case.Id (500...)
    case_number        text UNIQUE,
    subject            text,
    account_id         text,
    sap_account_number text,
    created_date       timestamptz,
    raw                jsonb NOT NULL,            -- the full record, exactly as received
    loaded_at          timestamptz DEFAULT now()
);
CREATE TABLE IF NOT EXISTS sf_email_message (
    id           text PRIMARY KEY,                -- EmailMessage.Id (02s...)
    sf_case_id   text REFERENCES sf_case(id) ON DELETE CASCADE,
    from_address text,
    subject      text,
    message_date timestamptz,
    raw          jsonb NOT NULL
);
CREATE TABLE IF NOT EXISTS sf_attachment (
    id             text PRIMARY KEY,              -- CombinedAttachments Id (069...)
    sf_case_id     text REFERENCES sf_case(id) ON DELETE CASCADE,
    title          text,
    file_extension text,
    content_size   bigint,
    raw            jsonb NOT NULL                 -- details only; the file itself stays in storage
);
CREATE TABLE IF NOT EXISTS agent_case (
    case_id            text PRIMARY KEY,          -- CASE-<CaseNumber>
    sf_case_id         text REFERENCES sf_case(id),
    case_number        text,
    status             text NOT NULL,             -- read / classified / extracted / payload_built / manual_queue ...
    source_file        text,
    source_sha256      text,
    received_at        timestamptz,
    from_address       text,
    sap_account_number text,
    account_name       text,
    contact_name       text,
    category           text,                      -- A2
    category_detail    text,
    class_confidence   real,
    class_reason       text,
    po_numbers         text[],
    flags              text[],
    duplicate_of       text,
    data               jsonb NOT NULL,            -- the whole case.json
    updated_at         timestamptz DEFAULT now()
);
CREATE INDEX IF NOT EXISTS agent_case_status ON agent_case (status);
CREATE INDEX IF NOT EXISTS agent_case_sha256 ON agent_case (source_sha256);
CREATE TABLE IF NOT EXISTS purchase_order (
    id                 bigserial PRIMARY KEY,
    case_id            text REFERENCES agent_case(case_id) ON DELETE CASCADE,
    source_file        text,
    read_method        text,
    po_number          text,
    po_date            date,
    request_date       date,
    buyer_name         text,
    overall_confidence real,
    missing_fields     text[],
    flags              text[],
    data               jsonb NOT NULL             -- every field with value + quote + confidence
);
CREATE INDEX IF NOT EXISTS purchase_order_po_number ON purchase_order (po_number);
CREATE TABLE IF NOT EXISTS po_line (
    id                bigserial PRIMARY KEY,
    po_id             bigint REFERENCES purchase_order(id) ON DELETE CASCADE,
    line_index        int,
    customer_material text,
    description       text,
    quantity          numeric,
    uom               text,
    unit_price        numeric,
    price_per         text,
    delivery_date     date,
    data              jsonb NOT NULL
);
CREATE TABLE IF NOT EXISTS sap_payload (
    case_id        text REFERENCES agent_case(case_id) ON DELETE CASCADE,
    po_number      text,
    payload        jsonb NOT NULL,                -- exactly what goes to SAP
    missing_fields text[],
    built_at       timestamptz DEFAULT now(),
    PRIMARY KEY (case_id, po_number)
);
CREATE TABLE IF NOT EXISTS audit_event (
    id      bigserial PRIMARY KEY,
    ts      timestamptz NOT NULL,
    case_id text,
    step    text,
    event   text,
    data    jsonb,
    UNIQUE (ts, case_id, step, event)
);
CREATE TABLE IF NOT EXISTS poll_pointer (
    source            text PRIMARY KEY,           -- 'salesforce_case'
    last_created_date timestamptz,                -- next poll asks Salesforce for cases after this
    last_case_number  text,
    updated_at        timestamptz DEFAULT now()
);
"""


def val(f: FieldValue):
    return f.value


def text(v):
    return None if v in (None, "") else str(v).strip()


def num(v):
    try:
        return float(str(v).replace(",", "")) if v not in (None, "") else None
    except ValueError:
        return None


def day(v):
    return v if isinstance(v, str) and re.fullmatch(r"\d{4}-\d{2}-\d{2}", v) else None


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def save_salesforce(conn, folder: Path):
    sf_case = read_json(folder / "salesforce" / "case.json")
    if not sf_case:
        return None
    conn.execute(
        """INSERT INTO sf_case (id, case_number, subject, account_id, sap_account_number, created_date, raw)
           VALUES (%s, %s, %s, %s, %s, %s::timestamptz, %s)
           ON CONFLICT (id) DO UPDATE SET case_number = EXCLUDED.case_number, subject = EXCLUDED.subject,
               account_id = EXCLUDED.account_id, sap_account_number = EXCLUDED.sap_account_number,
               created_date = EXCLUDED.created_date, raw = EXCLUDED.raw, loaded_at = now()""",
        (sf_case["Id"], sf_case.get("CaseNumber"), sf_case.get("Subject"), sf_case.get("AccountId"),
         sf_case.get("SAP_Account_Number__c"), sf_case.get("CreatedDate"), Jsonb(sf_case)))

    email = read_json(folder / "salesforce" / "email_message.json")
    if email:
        conn.execute(
            """INSERT INTO sf_email_message (id, sf_case_id, from_address, subject, message_date, raw)
               VALUES (%s, %s, %s, %s, %s::timestamptz, %s)
               ON CONFLICT (id) DO UPDATE SET from_address = EXCLUDED.from_address, subject = EXCLUDED.subject,
                   message_date = EXCLUDED.message_date, raw = EXCLUDED.raw""",
            (email["Id"], sf_case["Id"], email.get("FromAddress"), email.get("Subject"),
             email.get("MessageDate"), Jsonb(email)))

    for att in read_json(folder / "salesforce" / "attachments.json") or []:
        conn.execute(
            """INSERT INTO sf_attachment (id, sf_case_id, title, file_extension, content_size, raw)
               VALUES (%s, %s, %s, %s, %s, %s)
               ON CONFLICT (id) DO UPDATE SET title = EXCLUDED.title, file_extension = EXCLUDED.file_extension,
                   content_size = EXCLUDED.content_size, raw = EXCLUDED.raw""",
            (att["Id"], sf_case["Id"], att.get("Title"), att.get("FileExtension"), att.get("ContentSize"),
             Jsonb(att)))
    return sf_case["Id"]


def save_case(conn, case: Case, sf_case_id) -> None:
    c = case.classification
    conn.execute(
        """INSERT INTO agent_case (case_id, sf_case_id, case_number, status, source_file, source_sha256,
               received_at, from_address, sap_account_number, account_name, contact_name, category,
               category_detail, class_confidence, class_reason, po_numbers, flags, duplicate_of, data)
           VALUES (%s, %s, %s, %s, %s, %s, %s::timestamptz, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
           ON CONFLICT (case_id) DO UPDATE SET sf_case_id = EXCLUDED.sf_case_id,
               case_number = EXCLUDED.case_number, status = EXCLUDED.status,
               source_file = EXCLUDED.source_file, source_sha256 = EXCLUDED.source_sha256,
               received_at = EXCLUDED.received_at, from_address = EXCLUDED.from_address,
               sap_account_number = EXCLUDED.sap_account_number, account_name = EXCLUDED.account_name,
               contact_name = EXCLUDED.contact_name, category = EXCLUDED.category,
               category_detail = EXCLUDED.category_detail, class_confidence = EXCLUDED.class_confidence,
               class_reason = EXCLUDED.class_reason, po_numbers = EXCLUDED.po_numbers, flags = EXCLUDED.flags,
               duplicate_of = EXCLUDED.duplicate_of, data = EXCLUDED.data, updated_at = now()""",
        (case.case_id, sf_case_id, case.case_number or None, case.status, case.source_file, case.source_sha256,
         case.received_at.isoformat() if case.received_at else None, case.from_address,
         case.sap_account_number, case.account_name, case.contact_name or None,
         c.category if c else None, c.detail if c else None, c.confidence if c else None,
         c.reason if c else None, c.po_numbers if c else [], case.flags, case.duplicate_of,
         Jsonb(case.model_dump(mode="json"))))

    # A3: replace this case's POs (po_line rows go with them)
    conn.execute("DELETE FROM purchase_order WHERE case_id = %s", (case.case_id,))
    for po in case.purchase_orders:
        po_id = conn.execute(
            """INSERT INTO purchase_order (case_id, source_file, read_method, po_number, po_date, request_date,
                   buyer_name, overall_confidence, missing_fields, flags, data)
               VALUES (%s, %s, %s, %s, %s::date, %s::date, %s, %s, %s, %s, %s) RETURNING id""",
            (case.case_id, po.source_file, po.read_method, text(val(po.po_number)), day(val(po.po_date)),
             day(val(po.requested_delivery_date)), text(val(po.buyer_name)), po.overall_confidence,
             po.missing_fields, po.flags, Jsonb(po.model_dump(mode="json")))).fetchone()[0]
        for i, line in enumerate(po.lines, 1):
            conn.execute(
                """INSERT INTO po_line (po_id, line_index, customer_material, description, quantity, uom,
                       unit_price, price_per, delivery_date, data)
                   VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s::date, %s)""",
                (po_id, i, text(val(line.customer_material)), text(val(line.description)),
                 num(val(line.quantity)), text(val(line.uom)), num(val(line.unit_price)),
                 text(val(line.price_per)), day(val(line.delivery_date)), Jsonb(line.model_dump(mode="json"))))


def save_payloads(conn, case_id: str, folder: Path) -> int:
    conn.execute("DELETE FROM sap_payload WHERE case_id = %s", (case_id,))
    files = sorted((folder / "payloads").glob("*.json"))
    for f in files:
        p = json.loads(f.read_text(encoding="utf-8"))
        conn.execute("INSERT INTO sap_payload (case_id, po_number, payload, missing_fields) VALUES (%s, %s, %s, %s)",
                     (case_id, p.get("PO_Number") or f.stem, Jsonb(p), missing(p)))
    return len(files)


def save_audit(conn, out_dir: Path) -> None:
    path = out_dir / "audit.jsonl"
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        e = json.loads(line)
        extra = {k: v for k, v in e.items() if k not in ("ts", "case_id", "step", "event")}
        conn.execute(
            """INSERT INTO audit_event (ts, case_id, step, event, data) VALUES (%s::timestamptz, %s, %s, %s, %s)
               ON CONFLICT (ts, case_id, step, event) DO NOTHING""",
            (e["ts"], e.get("case_id"), e.get("step"), e.get("event"), Jsonb(extra)))


def move_pointer(conn) -> None:
    """Remember the newest Salesforce case we have. With real Salesforce, the next poll asks only for newer ones."""
    conn.execute(
        """INSERT INTO poll_pointer (source, last_created_date, last_case_number)
           SELECT 'salesforce_case', created_date, case_number FROM sf_case
           ORDER BY created_date DESC NULLS LAST, case_number DESC LIMIT 1
           ON CONFLICT (source) DO UPDATE SET last_created_date = EXCLUDED.last_created_date,
               last_case_number = EXCLUDED.last_case_number, updated_at = now()
           WHERE poll_pointer.last_created_date IS NULL
              OR EXCLUDED.last_created_date > poll_pointer.last_created_date""")


def run(out_dir: Path) -> None:
    with psycopg.connect(os.environ["DATABASE_URL"]) as conn:
        conn.execute(SCHEMA)
        folders = sorted(f.parent for f in out_dir.glob("*/case.json"))
        payloads = 0
        for folder in folders:
            case = Case.model_validate_json((folder / "case.json").read_text(encoding="utf-8"))
            with conn.transaction():
                sf_case_id = save_salesforce(conn, folder)
                save_case(conn, case, sf_case_id)
                payloads += save_payloads(conn, case.case_id, folder)
        with conn.transaction():
            save_audit(conn, out_dir)
            move_pointer(conn)
        print(f"Saved {len(folders)} cases, {payloads} payloads to Postgres.\n")
        summary(conn)


def summary(conn) -> None:
    for table in ("sf_case", "sf_email_message", "sf_attachment", "agent_case", "purchase_order", "po_line",
                  "sap_payload", "audit_event"):
        print(f"  {table:<17} {conn.execute(f'SELECT count(*) FROM {table}').fetchone()[0]}")
    print("\n  cases by status:")
    for status, n in conn.execute("SELECT status, count(*) FROM agent_case GROUP BY status ORDER BY 2 DESC").fetchall():
        print(f"    {status:<15} {n}")
    row = conn.execute("SELECT last_created_date, last_case_number FROM poll_pointer "
                       "WHERE source = 'salesforce_case'").fetchone()
    if row:
        print(f"\n  poll pointer: case {row[1]}, created {row[0]}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Save the A1-A4 output folder to Postgres")
    ap.add_argument("--out", default=str(ROOT / "output"))
    run(Path(ap.parse_args().out))