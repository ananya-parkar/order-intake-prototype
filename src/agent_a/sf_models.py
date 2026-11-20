"""Salesforce record shapes, using Ingredion's real API field names.

These are the records A1 hands to the rest of the pipeline, exactly as a
Salesforce REST query would return them. Today A1 builds them from email
files; later the real Salesforce fetch fills the same models, and nothing
downstream has to change.

Rules:
- Field names match Salesforce exactly (including the __c custom fields).
- Fields that Salesforce fills on arrival are filled by A1.
- Fields an agent or a person fills later stay None.
"""
import hashlib
from datetime import datetime
from typing import Optional

from pydantic import BaseModel, ConfigDict
from .constants import SF_ID_PREFIX as PREFIX

# Salesforce's FileType values for common extensions
_FILE_TYPES = {
    "pdf": "PDF", "xlsx": "EXCEL_X", "xls": "EXCEL", "csv": "CSV",
    "docx": "WORD_X", "doc": "WORD", "jpg": "JPG", "jpeg": "JPG",
    "png": "PNG", "gif": "GIF", "txt": "TEXT", "html": "HTML",
    "htm": "HTML", "eml": "EML", "msg": "MSG", "zip": "ZIP",
}


class SFRecord(BaseModel):
    # Reject unknown fields, so a typo in a field name fails loudly
    model_config = ConfigDict(extra="forbid")


class SFAttachment(SFRecord):
    """One row of Case.CombinedAttachments (same fields the RPA query reads)."""
    Id: str
    ParentId: str                         # the Case Id
    Title: str                            # file name without extension
    FileExtension: Optional[str] = None
    FileType: Optional[str] = None
    ContentSize: int = 0                  # bytes
    ContentUrl: Optional[str] = None      # only for link-type files
    CreatedDate: Optional[datetime] = None
    IsDeleted: bool = False


class SFEmailMessage(SFRecord):
    """The email Email-to-Case stored on the Case."""
    Id: str
    ParentId: str                         # the Case Id
    Subject: Optional[str] = None
    FromAddress: Optional[str] = None
    FromName: Optional[str] = None
    ToAddress: Optional[str] = None
    CcAddress: Optional[str] = None
    BccAddress: Optional[str] = None
    TextBody: Optional[str] = None
    HtmlBody: Optional[str] = None
    Headers: Optional[str] = None
    MessageDate: Optional[datetime] = None
    MessageIdentifier: Optional[str] = None   # the email's Message-ID
    ThreadIdentifier: Optional[str] = None    # In-Reply-To / thread id
    Incoming: bool = True
    HasAttachment: bool = False
    Status: str = "0"                     # "0" = New
    CreatedDate: Optional[datetime] = None


class SFCase(SFRecord):
    """The Case Email-to-Case creates (fields from Ingredion's org)."""
    # --- Filled by Salesforce when the email arrives (A1 fills these) ---
    Id: str
    CaseNumber: str
    Subject: Optional[str] = None
    Description: Optional[str] = None
    Status: str = "New"
    Origin: str = "Email"
    Priority: Optional[str] = "Medium"
    Type: Optional[str] = None
    OwnerId: Optional[str] = None         # queue not decided yet -> None
    CreatedDate: Optional[datetime] = None
    SourceId: Optional[str] = None        # the EmailMessage Id
    SuppliedName: Optional[str] = None
    SuppliedEmail: Optional[str] = None
    SuppliedPhone: Optional[str] = None
    Email_Content_Truncated__c: bool = False
    Email_To_Case_Address__c: Optional[str] = None
    IsDeleted: bool = False
    ParentId: Optional[str] = None
    MasterRecordId: Optional[str] = None

    # --- From the matched Contact / Account ---
    AccountId: Optional[str] = None
    Account_Name__c: Optional[str] = None
    SAP_Account_Number__c: Optional[str] = None   # formula in Salesforce
    BOT_Customer__c: bool = False                 # = Account.Is_BOT_Customer__c
    ContactId: Optional[str] = None
    ContactEmail: Optional[str] = None
    ContactPhone: Optional[str] = None            # standard field, from the Contact

    # --- Filled later by the agent or a person (A1 leaves empty) ---
    Request_Type__c: Optional[str] = None
    Request_Sub_Type__c: Optional[str] = None
    PO_Number__c: Optional[str] = None
    SAP_Order_Number__c: Optional[str] = None
    Possible_duplicate_case__c: bool = False
    Order_Details_Validated__c: bool = False
    Customer_Service_Notes__c: Optional[str] = None
    Other__c: Optional[str] = None
    SSC_Team_Assigned__c: Optional[str] = None
    ClosedDate: Optional[datetime] = None


class SFContact(SFRecord):
    """A customer contact (used to match the sender's email)."""
    Id: str
    AccountId: str
    Name: str
    Email: str
    Phone: Optional[str] = None


class SFAccount(SFRecord):
    """A customer account (the source of SAP number and BOT flag)."""
    Id: str
    Name: str
    SAP_Account_Number__c: Optional[str] = None
    Is_BOT_Customer__c: bool = False
    Contacts: list[SFContact] = []


# ---------------- helpers ----------------

_B62 = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"


def _suffix(id15: str) -> str:
    """Salesforce's 3-character checksum that turns a 15-char Id into 18."""
    out = ""
    for i in range(3):
        chunk = id15[i * 5:(i + 1) * 5]
        bits = sum(1 << j for j, ch in enumerate(chunk) if ch.isupper())
        out += "ABCDEFGHIJKLMNOPQRSTUVWXYZ012345"[bits]
    return out


def fake_sf_id(sobject: str, seed: str) -> str:
    """A realistic 18-character Id. Same seed -> same Id on every run."""
    n = int(hashlib.sha256(f"{sobject}:{seed}".encode()).hexdigest(), 16)
    body = ""
    for _ in range(9):
        n, r = divmod(n, 62)
        body += _B62[r]
    id15 = PREFIX[sobject] + "000" + body          # 3 + 3 + 9 = 15
    return id15 + _suffix(id15)


def file_type_for(extension: str) -> str:
    """Salesforce FileType for an extension, e.g. 'xlsx' -> 'EXCEL_X'."""
    return _FILE_TYPES.get(extension.lower().lstrip("."), "UNKNOWN")