"""
LLM client - offline stand-in for Gemini on Vertex AI.

Provider and model come from config.yaml (llm section): "anthropic" (Claude) or "groq".
API keys come from .env: ANTHROPIC_API_KEY or GROQ_API_KEY.

Every call:  guard.screen_input -> model -> guard.screen_response -> validate JSON with Pydantic
Both providers are asked for strict JSON that matches the Pydantic schema.
If the JSON still doesn't validate, we ask the model once more to fix it.
"""
import os
import threading
import time
from typing import Type, TypeVar

import requests
from dotenv import load_dotenv
from pydantic import BaseModel, ValidationError

from . import cache, guard
from .config import CFG, ROOT
from .constants import ANTHROPIC_URL, ANTHROPIC_VERSION, GROQ_URL, RETRY_STATUS

load_dotenv(ROOT / ".env")          # secrets only: ANTHROPIC_API_KEY / GROQ_API_KEY, DATABASE_URL, TESSERACT_CMD

PROVIDER = CFG.llm.provider
T = TypeVar("T", bound=BaseModel)

# ---------------- usage: tokens, time and cost per case ----------------
# Counted per thread, so cases running side by side (later) don't mix their numbers.
_local = threading.local()


def _usage() -> dict:
    """This thread's running totals since the last usage_reset()."""
    if getattr(_local, "u", None) is None:
        _local.u = {"llm_calls": 0, "cache_hits": 0, "tokens_in": 0, "tokens_out": 0, "llm_seconds": 0.0}
    return _local.u


def usage_reset() -> None:
    """Start counting from zero (call before each case)."""
    _local.u = None


def cost_usd(tokens_in: int, tokens_out: int):
    """Cost from the prices in config.yaml (USD per million tokens); None if no prices are set."""
    p_in, p_out = CFG.llm.price_input_per_mtok, CFG.llm.price_output_per_mtok
    if not (p_in or p_out):
        return None
    return round(tokens_in / 1e6 * p_in + tokens_out / 1e6 * p_out, 6)


def usage_take() -> dict:
    """Totals since the last reset - calls, tokens in/out, seconds waiting for the model, cost - then reset."""
    u = dict(_usage())
    usage_reset()
    u["llm_seconds"] = round(u["llm_seconds"], 2)
    u["llm_cost_usd"] = cost_usd(u["tokens_in"], u["tokens_out"])
    return u


def _count(tokens_in, tokens_out) -> None:
    """Add one model call's tokens to this thread's totals."""
    u = _usage()
    u["llm_calls"] += 1
    u["tokens_in"] += int(tokens_in or 0)
    u["tokens_out"] += int(tokens_out or 0)


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


def _wait(attempt: int, retry_after=None) -> float:
    """Seconds to wait before the next try: the server's retry-after if it sent one, else 1, 2, 4, 8 ... (max 60)."""
    try:
        wait = float(retry_after) if retry_after is not None else 2 ** attempt
    except ValueError:
        wait = 2 ** attempt
    return min(max(wait, 1), 60)


def _post(url: str, body: dict, headers: dict) -> dict:
    """POST to the model API; wait and retry when it is busy (429 / 5xx / 529 overloaded),
    when it does not answer in time (timeout) or when the connection drops."""
    last = ""
    ATTEMPTS = CFG.llm.max_attempts
    for attempt in range(ATTEMPTS):
        try:
            r = requests.post(url, json=body, headers=headers, timeout=CFG.llm.timeout_seconds)
        except (requests.Timeout, requests.ConnectionError) as e:   # no answer / network dropped -> try again
            last = type(e).__name__
            wait = _wait(attempt)
            if attempt < ATTEMPTS - 1:                      # last try failed -> give up below
                print(f"   {PROVIDER} {last}, trying again in {wait:.0f}s...")
                time.sleep(wait)
            continue
        if r.status_code == 200:
            return r.json()
        if r.status_code in RETRY_STATUS:
            last = f"busy ({r.status_code})"
            wait = _wait(attempt, r.headers.get("retry-after"))
            if attempt < ATTEMPTS - 1:                      # last try failed -> give up below
                print(f"   {PROVIDER} {last}, trying again in {wait:.0f}s...")
                time.sleep(wait)
            continue
        raise LLMError(f"{PROVIDER} {r.status_code}: {r.text[:300]}")   # 400 / 401 / 403 ... -> retrying won't help
    raise LLMError(f"{PROVIDER} still failing after {CFG.llm.max_attempts} tries ({last})")


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
    u = data.get("usage", {})
    _count(u.get("input_tokens", 0) + u.get("cache_read_input_tokens", 0) + u.get("cache_creation_input_tokens", 0),
           u.get("output_tokens", 0))
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
    u = data.get("usage", {})
    _count(u.get("prompt_tokens", 0), u.get("completion_tokens", 0))
    return data["choices"][0]["message"]["content"]


def _call(messages: list[dict], out_model: Type[BaseModel]) -> str:
    """Send one request to the provider set in config.yaml and return its raw JSON text."""
    t0 = time.perf_counter()
    try:
        if PROVIDER == "anthropic":
            return _call_anthropic(messages, out_model)
        if PROVIDER == "groq":
            return _call_groq(messages, out_model)
    finally:
        _usage()["llm_seconds"] += time.perf_counter() - t0          # time waiting for the model, retries included
    raise LLMError(f"Unsupported llm.provider '{PROVIDER}' in config.yaml (use 'anthropic' or 'groq')")


def chat_json(system: str, user: str, out_model: Type[T]) -> tuple[T, list[str]]:
    """Ask the model and get back a validated Pydantic object + any guard findings."""
    findings = guard.screen_input(user).findings
    messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]

    # Same provider + model + prompt + text + schema as before -> reuse that answer (no call, no cost)
    key = cache.make_key(PROVIDER, CFG.llm.model, system, user, strict_schema(out_model))
    hit = cache.get("llm", key)
    if hit is not None:
        try:
            result = out_model.model_validate_json(hit["content"])
            _usage()["cache_hits"] += 1
            return result, findings + guard.screen_response(hit["content"]).findings
        except (ValidationError, KeyError, TypeError):
            pass                                               # old / damaged entry -> ask the model

    for attempt in range(2):                                   # first try + one repair
        content = _call(messages, out_model)
        findings += guard.screen_response(content).findings
        try:
            result = out_model.model_validate_json(content)
            cache.put("llm", key, {"model": CFG.llm.model, "content": content})   # only answers that validated
            return result, findings
        except ValidationError as e:
            messages += [{"role": "assistant", "content": content},
                         {"role": "user", "content": f"That JSON did not match the schema: {e}. Return only corrected JSON."}]
    raise LLMError("Model did not return valid JSON after a retry")