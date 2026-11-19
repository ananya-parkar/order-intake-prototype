"""Data shapes shared by all steps. A1 fills a Case; later steps add to it."""
from datetime import datetime
from typing import Literal, Optional, Union

from pydantic import BaseModel, Field


class Attachment(BaseModel):
    filename: str
    content_type: str
    kind: Literal["pdf", "excel", "word", "image", "email", "text", "other"]
    size_bytes: int
    saved_path: str                          # where A1 saved the file
    is_inline: bool = False                  # image embedded in the email body
    likely_signature: bool = False           # small inline image = logo, ignore later
    from_nested_email: Optional[str] = None  # set if it came from a forwarded email
    sf_id: Optional[str] = None              # Id of the CombinedAttachments row (069...)


class Classification(BaseModel):             # filled by A2
    category: Literal["new_order", "revised_po", "duplicate", "not_an_order"]
    detail: str = ""                         # e.g. order_status_inquiry, cancellation
    po_numbers: list[str] = Field(default_factory=list)
    po_count: int = 0
    confidence: float = 0.0
    reason: str = ""


# ---------------- A3: documents and extracted POs ----------------

class DocumentText(BaseModel):              # one attachment (or the email body) as A3 read it
    filename: str                           # attachment name, or "email_body"
    method: Literal["pdf_text", "ocr", "excel", "word", "text", "email_body", "none"]
    pages: int = 0
    chars: int = 0
    ocr_confidence: Optional[float] = None  # 0-1, average Tesseract word confidence (OCR only)
    text_path: str = ""                     # extracted/<file>.txt - what A3 actually read
    doc_type: str = ""                      # purchase_order, coa, invoice, bol, terms, other
    error: str = ""


class Source(BaseModel):                    # where a value was read from
    file: str = ""
    page: Optional[int] = None
    quote: str = ""                         # the exact text the value came from


class FieldValue(BaseModel):                # one extracted field
    value: Union[float, str, None] = None   # None = not on the document
    confidence: float = 0.0                 # 0-1, set by A3's checks (not just the model's opinion)
    source: Source = Field(default_factory=Source)
    checks: list[str] = Field(default_factory=list)   # why confidence was lowered, e.g. quote_not_found


def _f():
    return Field(default_factory=FieldValue)


class LineItem(BaseModel):
    line_no: FieldValue = _f()              # customer's line number -> PO_Item (10, 20...) in A4
    customer_material: FieldValue = _f()    # -> CUST_MAT22
    supplier_material: FieldValue = _f()    # Ingredion material, if printed -> Material
    description: FieldValue = _f()
    quantity: FieldValue = _f()             # -> Quantity
    uom: FieldValue = _f()                  # -> Unit_of_Measure
    unit_price: FieldValue = _f()           # -> Pricing
    price_per: FieldValue = _f()            # e.g. "100 LB", "CWT", "LB"
    line_total: FieldValue = _f()
    delivery_date: FieldValue = _f()        # YYYY-MM-DD


class PurchaseOrder(BaseModel):
    source_file: str                        # attachment name, or "email_body"
    read_method: str = ""                   # pdf_text / ocr / excel / word / email_body
    po_number: FieldValue = _f()            # -> PO_Number
    po_date: FieldValue = _f()              # -> PO_Date (YYYY-MM-DD)
    requested_delivery_date: FieldValue = _f()   # -> Request_Date
    customer_name: FieldValue = _f()        # buyer company as printed
    buyer_name: FieldValue = _f()           # -> contact_name (if Salesforce has none)
    buyer_phone: FieldValue = _f()          # -> telephone
    buyer_email: FieldValue = _f()
    vendor_name: FieldValue = _f()          # who the PO is addressed to
    ship_to_name: FieldValue = _f()
    ship_to_address: FieldValue = _f()
    bill_to: FieldValue = _f()
    currency: FieldValue = _f()
    incoterms: FieldValue = _f()
    payment_terms: FieldValue = _f()
    notes: FieldValue = _f()                # special instructions
    total: FieldValue = _f()
    lines: list[LineItem] = Field(default_factory=list)
    overall_confidence: float = 0.0         # lowest confidence among the key fields
    missing_fields: list[str] = Field(default_factory=list)
    flags: list[str] = Field(default_factory=list)   # e.g. math_mismatch, zero_price, low_ocr_confidence


class Case(BaseModel):
    """Our working copy of one Salesforce case. A1 creates it, later steps add to it.
    The untouched Salesforce records are kept next to it in salesforce/*.json."""
    case_id: str                          # "CASE-" + CaseNumber
    source_file: str
    source_path: str = ""                 # full path of the original email file
    source_sha256: str
    received_at: Optional[datetime] = None
    subject: str = ""
    from_address: str = ""
    from_name: str = ""
    to: str = ""
    cc: str = ""
    in_reply_to: str = ""
    body_text: str = ""
    attachments: list[Attachment] = Field(default_factory=list)

    # --- copied from the Salesforce records (see sf_adapter.from_salesforce) ---
    sf_case_id: str = ""                  # Case.Id (500...)
    case_number: str = ""                 # Case.CaseNumber
    sf_email_id: str = ""                 # EmailMessage.Id (02s...)
    owner_id: Optional[str] = None        # Case.OwnerId (queue not decided yet)
    account_id: Optional[str] = None      # Case.AccountId
    account_name: Optional[str] = None    # Case.Account_Name__c
    sap_account_number: Optional[str] = None  # Case.SAP_Account_Number__c -> payload
    bot_customer: bool = False            # Case.BOT_Customer__c
    contact_id: Optional[str] = None      # Case.ContactId
    contact_name: str = ""                # Contact name / SuppliedName -> payload
    contact_phone: str = ""               # Case.SuppliedPhone -> payload "telephone"

    status: str = "received"
    duplicate_of: Optional[str] = None
    classification: Optional[Classification] = None   # A2
    documents: list[DocumentText] = Field(default_factory=list)          # A3: what was read
    purchase_orders: list[PurchaseOrder] = Field(default_factory=list)   # A3: extracted POs
    flags: list[str] = Field(default_factory=list)    # e.g. possible_duplicate, low_confidence
    notes: list[str] = Field(default_factory=list)