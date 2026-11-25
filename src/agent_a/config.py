"""Loads config.yaml once and checks it with Pydantic. Use: from .config import CFG, ROOT."""
from pathlib import Path
from typing import Literal

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
    price_input_per_mtok: float = 0.0       # USD per million input tokens (0 = don't show cost)
    price_output_per_mtok: float = 0.0      # USD per million output tokens


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
    price_per_1000_pages: float = 0.0       # USD per 1,000 pages (0 = free tier / don't show cost)


class A4Config(Section):
    """a4: timezone for Created_On, the fixed SAP values the client gave us, UoM mapping and the review gate."""
    timezone: str
    order_reason: str
    order_type: str
    sales_org: str
    distribution_channel: str
    division: str
    review_min_confidence: float            # PO confidence below this -> needs_review
    check_past_request_date: bool           # flag a Request_Date that is already in the past
    zero_qty_lines: Literal["skip", "send"] # quantity-0 lines: leave out of the payload, or send (flagged)
    sap_uom_codes: list[str]                # unit codes SAP accepts; anything else -> unknown_uom
    uom_map: dict[str, str]                 # customer spelling -> SAP code (exact match)
    uom_prefix: dict[str, str]              # starts with -> SAP code (BAG50LB -> BAG)


class CacheConfig(Section):
    """cache: reuse model answers and OCR results for the same input (see cache.py)."""
    enabled: bool = True
    dir: str = ".cache"                     # relative to the project folder; must stay git-ignored


class Config(Section):
    """The whole config.yaml."""
    llm: LLMConfig
    paths: PathsConfig
    a1: A1Config
    a2: A2Config
    a3: A3Config
    a4: A4Config
    ocr: OCRConfig
    cache: CacheConfig = CacheConfig()


def load(path: Path = ROOT / "config.yaml") -> Config:
    """Read config.yaml and validate it; stops with a clear error if a value is missing or wrong."""
    return Config.model_validate(yaml.safe_load(path.read_text(encoding="utf-8")))


CFG = load()