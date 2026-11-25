"""
Traceability: which model, prompt, settings and code produced a result.

Every step that decides something stamps case.trace[<step>] in case.json, e.g.
    "trace": {"a2": {"provider": "anthropic", "model": "claude-sonnet-5-5", "prompt_version": "p-3f9c1e2a",
                     "config_version": "c-41aa02bd", "code_version": "7ed8d66", "at": "2026-11-25T07:10:03+00:00"}}
prompt_version / config_version are short hashes of the prompt text and config.yaml: they change by themselves
whenever the prompt or a setting changes, so results can always be tied to the exact prompt and settings.
code_version is the git commit (or the CODE_VERSION environment variable when there is no .git, e.g. on Azure).
"""
import hashlib
import os
import subprocess
from datetime import datetime, timezone
from functools import lru_cache

from .config import CFG, ROOT


def short_hash(text: str, prefix: str) -> str:
    """'p-3f9c1e2a': the first 8 characters of the SHA-256 of the text."""
    return f"{prefix}-{hashlib.sha256(text.encode('utf-8')).hexdigest()[:8]}"


def prompt_version(prompt: str) -> str:
    """Version id of a prompt: changes whenever one character of the prompt changes."""
    return short_hash(prompt, "p")


@lru_cache(maxsize=1)
def config_version() -> str:
    """Version id of config.yaml (read once per run)."""
    try:
        return short_hash((ROOT / "config.yaml").read_text(encoding="utf-8"), "c")
    except OSError:
        return ""


@lru_cache(maxsize=1)
def code_version() -> str:
    """Git commit of the code (CODE_VERSION env var if set, e.g. on Azure where there is no .git folder)."""
    if os.getenv("CODE_VERSION"):
        return os.environ["CODE_VERSION"]
    try:
        out = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, capture_output=True,
                             text=True, timeout=5)
        return out.stdout.strip() if out.returncode == 0 else ""
    except (OSError, subprocess.SubprocessError):
        return ""


def step_trace(prompt: str | None = None, **extra) -> dict:
    """The trace for one step. Pass the prompt if a model was called; extra keys are added as they are
    (e.g. decided_by="rule:ariba", ocr="azure_di")."""
    t = {}
    if prompt is not None:
        t.update({"provider": CFG.llm.provider, "model": CFG.llm.model, "prompt_version": prompt_version(prompt)})
    t.update({"config_version": config_version(), "code_version": code_version(),
              "at": datetime.now(timezone.utc).isoformat(timespec="seconds")})
    t.update(extra)
    return t