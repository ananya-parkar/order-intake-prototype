"""Loads config.yaml once and checks it with Pydantic. Use: from .config import CFG, ROOT."""
from pathlib import Path

import yaml
from pydantic import BaseModel, ConfigDict

ROOT = Path(__file__).resolve().parents[2]          # project folder (config.yaml, .env, data/)


class Section(BaseModel):
    """Base for every config section: unknown keys are an error, so typos fail loudly."""
    model_config = ConfigDict(extra="forbid")


class LLMConfig(Section):
    provider: str
    model: str
    temperature: float
    reasoning_effort: str
    max_completion_tokens: int
    timeout_seconds: int
    max_attempts: int


class PathsConfig(Section):
    samples: str
    output: str
    accounts: str


class A1Config(Section):
    first_case_number: int
    signature_max_bytes: int
    max_nesting: int


class A2Config(Section):
    body_limit_chars: int
    low_confidence: float
    use_ariba_rule: bool


class A3Config(Section):
    supplier_names: str
    max_doc_chars: int
    ocr_low: float
    no_quote_cap: float
    pdf_dpi: int
    min_page_chars: int
    min_ocr_width: int


class A4Config(Section):
    timezone: str


class Config(Section):
    llm: LLMConfig
    paths: PathsConfig
    a1: A1Config
    a2: A2Config
    a3: A3Config
    a4: A4Config


def load(path: Path = ROOT / "config.yaml") -> Config:
    """Read config.yaml and validate it; stops with a clear error if a value is missing or wrong."""
    return Config.model_validate(yaml.safe_load(path.read_text(encoding="utf-8")))


CFG = load()