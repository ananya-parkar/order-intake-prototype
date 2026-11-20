"""Loads config.yaml once and checks it with Pydantic. Use: from .config import CFG, ROOT."""
from pathlib import Path

import yaml
from pydantic import BaseModel, ConfigDict

ROOT = Path(__file__).resolve().parents[2]          # project folder (config.yaml, .env, data/)


class Section(BaseModel):
    """Base for every config section: unknown keys are an error, so typos fail loudly."""
    model_config = ConfigDict(extra="forbid")


class LLMConfig(Section):
    """llm: which model to call and how (temperature, tokens, timeout, retries)."""
    provider: str
    model: str
    temperature: float
    reasoning_effort: str
    max_completion_tokens: int
    timeout_seconds: int
    max_attempts: int


class PathsConfig(Section):
    """paths: where the sample emails, output and account list are."""
    samples: str
    output: str
    accounts: str


class A1Config(Section):
    """a1: case numbering and attachment handling."""
    first_case_number: int
    signature_max_bytes: int
    max_nesting: int


class A2Config(Section):
    """a2: how much email text goes to the model and the low-confidence threshold."""
    body_limit_chars: int
    low_confidence: float
    use_ariba_rule: bool


class A3Config(Section):
    """a3: document size sent to the model, confidence caps and OCR settings."""
    supplier_names: str
    max_doc_chars: int
    ocr_low: float
    no_quote_cap: float
    pdf_dpi: int
    min_page_chars: int
    min_ocr_width: int

class OCRConfig(Section):
    """ocr: which engine reads scanned pages and photos, and how long to wait for it."""
    provider: str
    timeout_seconds: int
    poll_seconds: float
    max_attempts: int

class A4Config(Section):
    """a4: timezone for Created_On and the fixed SAP values the client gave us."""
    timezone: str
    order_reason: str
    order_type: str
    sales_org: str
    distribution_channel: str
    division: str


class Config(Section):
    """The whole config.yaml."""
    llm: LLMConfig
    paths: PathsConfig
    a1: A1Config
    a2: A2Config
    a3: A3Config
    a4: A4Config
    ocr: OCRConfig


def load(path: Path = ROOT / "config.yaml") -> Config:
    """Read config.yaml and validate it; stops with a clear error if a value is missing or wrong."""
    return Config.model_validate(yaml.safe_load(path.read_text(encoding="utf-8")))


CFG = load()