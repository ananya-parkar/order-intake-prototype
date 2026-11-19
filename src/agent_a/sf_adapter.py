"""The one bridge between Salesforce records and our pipeline Case.

    Salesforce shape (sf_models)  --from_salesforce()-->  pipeline Case (models)

Today A1 calls it with records it built from an email file.
Later the real Salesforce fetch calls it with records from the REST API.
A2, A3 and A4 only ever see the pipeline Case, so they never change.
"""
from typing import Optional

from .models import Attachment, Case
from .sf_models import SFAttachment, SFCase, SFEmailMessage


class AdapterError(ValueError):
    pass


def from_salesforce(
    sf_case: SFCase,
    sf_email: Optional[SFEmailMessage],
    sf_attachments: list[SFAttachment],
    local_files: list[Attachment],
    *,
    source_file: str = "",
    source_path: str = "",
    source_sha256: str = "",
    contact_name: str = "",
) -> Case:
    """Build the pipeline Case from the three Salesforce records.

    local_files: the downloaded attachment files (saved on disk), each with
                 sf_id set to the matching CombinedAttachments Id.
    contact_name: the matched Contact's name, if known (else SuppliedName is used).
    """
    # Every downloaded file must belong to an attachment row of this case
    known_ids = {a.Id for a in sf_attachments if not a.IsDeleted}
    for f in local_files:
        if f.sf_id not in known_ids:
            raise AdapterError(f"{f.filename}: sf_id {f.sf_id} is not an attachment of case {sf_case.CaseNumber}")
    if sf_email and sf_email.ParentId != sf_case.Id:
        raise AdapterError(f"EmailMessage {sf_email.Id} belongs to another case")

    e = sf_email or SFEmailMessage(Id="", ParentId=sf_case.Id)   # case without an email

    case = Case(
        case_id=f"CASE-{sf_case.CaseNumber}",
        source_file=source_file or sf_case.CaseNumber,
        source_path=source_path,
        source_sha256=source_sha256,
        received_at=e.MessageDate or sf_case.CreatedDate,
        subject=sf_case.Subject or e.Subject or "",
        from_address=e.FromAddress or sf_case.SuppliedEmail or "",
        from_name=e.FromName or sf_case.SuppliedName or "",
        to=e.ToAddress or "",
        cc=e.CcAddress or "",
        in_reply_to=e.ThreadIdentifier or "",
        # Full email text; Description may be cut at 32,000 characters
        body_text=e.TextBody or sf_case.Description or "",
        attachments=list(local_files),
        # Salesforce link fields
        sf_case_id=sf_case.Id,
        case_number=sf_case.CaseNumber,
        sf_email_id=e.Id,
        owner_id=sf_case.OwnerId,
        account_id=sf_case.AccountId,
        account_name=sf_case.Account_Name__c,
        sap_account_number=sf_case.SAP_Account_Number__c,
        bot_customer=sf_case.BOT_Customer__c,
        contact_id=sf_case.ContactId,
        contact_name=contact_name or sf_case.SuppliedName or e.FromName or "",
        contact_phone=sf_case.ContactPhone or sf_case.SuppliedPhone or "",
        status="read",
    )

    if not sf_case.AccountId:
        case.flags.append("unknown_sender")
        case.notes.append("A1: sender not matched to a Salesforce Contact/Account")
    if sf_case.Email_Content_Truncated__c:
        case.flags.append("description_truncated")
    return case