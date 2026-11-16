"""
LLM client - offline stand-in for Gemini on Vertex AI.

Two providers, chosen with the LLM_PROVIDER environment variable:
  groq   (default) - cloud, fast, big model. SYNTHETIC EMAILS ONLY.
  ollama           - runs on your laptop. Use this for real emails.

Every call:  guard.screen_input -> model -> guard.screen_response -> validate JSON with Pydantic
If the JSON doesn't validate, we ask the model once more to fix it.
"""
import os
import time
from pathlib import Path
from typing import Type, TypeVar

import requests
from dotenv import load_dotenv
from pydantic import BaseModel, ValidationError

from . import guard

load_dotenv(Path(__file__).resolve().parents[2] / ".env")   # reads GROQ_API_KEY etc. from the project's .env

PROVIDER = os.getenv("LLM_PROVIDER", "groq")
GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"
GROQ_MODEL = os.getenv("GROQ_MODEL", "openai/gpt-oss-120b")
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen2.5:7b")

T = TypeVar("T", bound=BaseModel)


class LLMError(Exception):
    pass


def is_external() -> bool:
    """True if the provider sends data outside this laptop."""
    return PROVIDER == "groq"


def model_name() -> str:
    return GROQ_MODEL if PROVIDER == "groq" else OLLAMA_MODEL


def strict_schema(model: Type[BaseModel]) -> dict:
    """Pydantic schema -> Groq strict schema: every field required, no extra fields, no titles/defaults."""
    def fix(node):
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
      if PROVIDER == "groq":
        key = os.getenv("GROQ_API_KEY")
        if not key:
            raise LLMError("GROQ_API_KEY is not set")
        body = {
            "model": GROQ_MODEL,
            "messages": messages,
            "temperature": 0,
            "response_format": {"type": "json_schema", "json_schema": {
                "name": out_model.__name__, "schema": strict_schema(out_model), "strict": True}},
        }
        for attempt in range(6):                                 # up to 6 tries
            r = requests.post(GROQ_URL, json=body, timeout=120,
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
        raise LLMError("Groq still busy after 6 tries")

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