"""
Guard - offline stand-in for Model Armor.
Called before every LLM call (screen_input) and after it (screen_response).
This stub only LOOKS for obvious prompt-injection phrases and reports them; it never blocks.
Later the GCP version calls Model Armor (sanitizeUserPrompt / sanitizeModelResponse) instead.
"""
import re
from dataclasses import dataclass, field
from .constants import INJECTION_PATTERNS

@dataclass
class GuardResult:
    allowed: bool = True
    findings: list[str] = field(default_factory=list)


def _scan(text: str) -> GuardResult:
    hits = [p for p in INJECTION_PATTERNS if re.search(p, text, re.IGNORECASE)]
    return GuardResult(allowed=True, findings=[f"possible prompt injection: /{p}/" for p in hits])


def screen_input(text: str) -> GuardResult:
    """Check what we are about to send to the model (email text, PO text)."""
    return _scan(text)


def screen_response(text: str) -> GuardResult:
    """Check what the model sent back."""
    return _scan(text)