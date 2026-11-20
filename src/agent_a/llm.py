"""
LLM client - offline stand-in for Gemini on Vertex AI.

Provider and model come from config.yaml (llm section): "anthropic" (Claude) or "groq".
API keys come from .env: ANTHROPIC_API_KEY or GROQ_API_KEY.

Every call:  guard.screen_input -> model -> guard.screen_response -> validate JSON with Pydantic
Both providers are asked for strict JSON that matches the Pydantic schema.
If the JSON still doesn't validate, we ask the model once more to fix it.
"""
import os
import time
from typing import Type, TypeVar

import requests
from dotenv import load_dotenv
from pydantic import BaseModel, ValidationError

from . import guard
from .config import CFG, ROOT
from .constants import ANTHROPIC_URL, ANTHROPIC_VERSION, GROQ_URL, RETRY_STATUS

load_dotenv(ROOT / ".env")          # secrets only: ANTHROPIC_API_KEY / GROQ_API_KEY, DATABASE_URL, TESSERACT_CMD

PROVIDER = CFG.llm.provider
T = TypeVar("T", bound=BaseModel)


class LLMError(Exception):
    """The model could not give a usable answer (no key, still busy after retries, bad JSON twice)."""


def is_external() -> bool:
    """True if the provider sends data outside this laptop (both Claude and Groq are cloud APIs)."""
    return PROVIDER in ("anthropic", "groq")


def model_name() -> str:
    """Model id from config.yaml, written into the audit log with every result."""
    return CFG.llm.model


def strict_schema(model: Type[BaseModel]) -> dict:
    """Pydantic schema -> strict schema: every field required, no extra fields, no titles/defaults."""
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


def _key(name: str) -> str:
    """Read an API key from .env; stop with a clear message if it is missing."""
    key = os.getenv(name)
    if not key:
        raise LLMError(f"{name} is not set in .env")
    return key


def _post(url: str, body: dict, headers: dict) -> dict:
    """POST to the model API; wait and retry when it is busy (429 / 5xx / 529 overloaded)."""
    for attempt in range(CFG.llm.max_attempts):
        r = requests.post(url, json=body, headers=headers, timeout=CFG.llm.timeout_seconds)
        if r.status_code == 200:
            return r.json()
        if r.status_code in RETRY_STATUS:
            try:
                wait = float(r.headers.get("retry-after", 2 ** attempt))
            except ValueError:
                wait = 2 ** attempt
            wait = min(max(wait, 1), 60)
            print(f"   {PROVIDER} busy ({r.status_code}), waiting {wait:.0f}s and retrying...")
            time.sleep(wait)
            continue
        raise LLMError(f"{PROVIDER} {r.status_code}: {r.text[:300]}")
    raise LLMError(f"{PROVIDER} still busy after {CFG.llm.max_attempts} tries")


def _call_anthropic(messages: list[dict], out_model: Type[BaseModel]) -> str:
    """Claude Messages API with structured output (output_config.format = our JSON schema)."""
    system = "\n\n".join(m["content"] for m in messages if m["role"] == "system")
    body = {
        "model": CFG.llm.model,
        "max_tokens": CFG.llm.max_completion_tokens,
        "system": system,
        "messages": [m for m in messages if m["role"] != "system"],
        "output_config": {"format": {"type": "json_schema", "schema": strict_schema(out_model)}},
    }
    data = _post(ANTHROPIC_URL, body, {"x-api-key": _key("ANTHROPIC_API_KEY"),
                                       "anthropic-version": ANTHROPIC_VERSION})
    return "".join(b.get("text", "") for b in data.get("content", []) if b.get("type") == "text")


def _call_groq(messages: list[dict], out_model: Type[BaseModel]) -> str:
    """Groq chat completions with strict json_schema response format."""
    body = {
        "model": CFG.llm.model,
        "messages": messages,
        "temperature": CFG.llm.temperature,
        "reasoning_effort": CFG.llm.reasoning_effort,
        "max_completion_tokens": CFG.llm.max_completion_tokens,
        "response_format": {"type": "json_schema", "json_schema": {
            "name": out_model.__name__, "schema": strict_schema(out_model), "strict": True}},
    }
    data = _post(GROQ_URL, body, {"Authorization": f"Bearer {_key('GROQ_API_KEY')}"})
    return data["choices"][0]["message"]["content"]


def _call(messages: list[dict], out_model: Type[BaseModel]) -> str:
    """Send one request to the provider set in config.yaml and return its raw JSON text."""
    if PROVIDER == "anthropic":
        return _call_anthropic(messages, out_model)
    if PROVIDER == "groq":
        return _call_groq(messages, out_model)
    raise LLMError(f"Unsupported llm.provider '{PROVIDER}' in config.yaml (use 'anthropic' or 'groq')")


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