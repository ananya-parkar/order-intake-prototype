"""Data shapes shared by all steps. A1 fills a Case; later steps add to it."""
from datetime import datetime
from typing import Literal, Optional

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


class Classification(BaseModel):             # filled by A2
    category: Literal["new_order", "revised_po", "duplicate", "not_an_order"]
    detail: str = ""                         # e.g. order_status_inquiry, cancellation
    po_numbers: list[str] = Field(default_factory=list)
    po_count: int = 0
    confidence: float = 0.0
    reason: str = ""


class Case(BaseModel):
    case_id: str
    source_file: str
    source_path: str = ""                    # full path of the original email file
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
    status: str = "received"
    duplicate_of: Optional[str] = None
    classification: Optional[Classification] = None   # A2
    flags: list[str] = Field(default_factory=list)    # e.g. possible_duplicate, low_confidence
    notes: list[str] = Field(default_factory=list)