"""
LLM client - offline stand-in for Gemini on Vertex AI.

Provider and model come from config.yaml (llm section); the API key from .env (GROQ_API_KEY).

Every call:  guard.screen_input -> model -> guard.screen_response -> validate JSON with Pydantic
If the JSON doesn't validate, we ask the model once more to fix it.
"""
import os
import time
from typing import Type, TypeVar

import requests
from dotenv import load_dotenv
from pydantic import BaseModel, ValidationError

from . import guard
from .config import CFG, ROOT
from .constants import GROQ_URL

load_dotenv(ROOT / ".env")          # secrets only: GROQ_API_KEY, DATABASE_URL, TESSERACT_CMD

PROVIDER = CFG.llm.provider
T = TypeVar("T", bound=BaseModel)


class LLMError(Exception):
    """The model could not give a usable answer (no key, still busy after retries, bad JSON twice)."""


def is_external() -> bool:
    """True if the provider sends data outside this laptop."""
    return PROVIDER == "groq"


def model_name() -> str:
    """Model id from config.yaml, written into the audit log with every result."""
    return CFG.llm.model


def strict_schema(model: Type[BaseModel]) -> dict:
    """Pydantic schema -> Groq strict schema: every field required, no extra fields, no titles/defaults."""
    def fix(node):
        """Walk the schema tree and apply the strict-mode rules to every object."""
        if isinstance(node, list):
            return [fix(x) for x in node]
        if not isinstance(node, dict):
            return node
        out = {}
        for k, v in node.items():
            if k in ("title", "default"):
                continue
            out[k] = {name: fix(sub) for name, sub in v.items()} if k in ("properties", "$defs") else fix(v)
        if out.get("type") == "object" and "properties" in out:
            out["additionalProperties"] = False
            out["required"] = list(out["properties"])
        return out
    return fix(model.model_json_schema())


def _call(messages: list[dict], out_model: Type[BaseModel]) -> str:
    """Send one request to the model and return its raw JSON text; waits and retries when the API is busy."""
    if PROVIDER != "groq":
        raise LLMError(f"Unsupported llm.provider '{PROVIDER}' in config.yaml (only 'groq' for now)")
    key = os.getenv("GROQ_API_KEY")
    if not key:
        raise LLMError("GROQ_API_KEY is not set")
    body = {
        "model": CFG.llm.model,
        "messages": messages,
        "temperature": CFG.llm.temperature,
        "reasoning_effort": CFG.llm.reasoning_effort,
        "max_completion_tokens": CFG.llm.max_completion_tokens,
        "response_format": {"type": "json_schema", "json_schema": {
            "name": out_model.__name__, "schema": strict_schema(out_model), "strict": True}},
    }
    for attempt in range(CFG.llm.max_attempts):
        r = requests.post(GROQ_URL, json=body, timeout=CFG.llm.timeout_seconds,
                          headers={"Authorization": f"Bearer {key}"})
        if r.status_code == 200:
            return r.json()["choices"][0]["message"]["content"]
        if r.status_code in (429, 500, 502, 503):            # too busy / temporary error
            try:
                wait = float(r.headers.get("retry-after", 2 ** attempt))
            except ValueError:
                wait = 2 ** attempt
            wait = min(max(wait, 1), 60)
            print(f"   Groq busy ({r.status_code}), waiting {wait:.0f}s and retrying...")
            time.sleep(wait)
            continue
        raise LLMError(f"Groq {r.status_code}: {r.text[:300]}")
    raise LLMError(f"Groq still busy after {CFG.llm.max_attempts} tries")


def chat_json(system: str, user: str, out_model: Type[T]) -> tuple[T, list[str]]:
    """Ask the model and get back a validated Pydantic object + any guard findings."""
    findings = guard.screen_input(user).findings
    messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]

    for attempt in range(2):                                   # first try + one repair
        content = _call(messages, out_model)
        findings += guard.screen_response(content).findings
        try:
            return out_model.model_validate_json(content), findings
        except ValidationError as e:
            messages += [{"role": "assistant", "content": content},
                         {"role": "user", "content": f"That JSON did not match the schema: {e}. Return only corrected JSON."}]
    raise LLMError("Model did not return valid JSON after a retry")