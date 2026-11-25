"""
Cache for model answers and OCR results, so re-running a step on the same input does not pay again.

Key = SHA-256 of everything that decides the answer:
    model answer: provider + model + system prompt + user text + JSON schema
    OCR:          OCR model + API version + the page image bytes
So a changed prompt, model, schema or document is a cache miss by itself - nothing to clear by hand.

Files: <cache.dir>/<kind>/<first 2 chars>/<key>.json   (cache.dir from config.yaml, default .cache,
       or the AGENT_CACHE_DIR environment variable - on Azure use /home/data/cache so it survives restarts)
The cache holds real email and PO text: it must stay git-ignored and on the machine running the agent.
Turn it off with  cache: enabled: false  in config.yaml, or AGENT_CACHE=off in the environment.
"""
import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any, Optional

from .config import CFG, ROOT


def enabled() -> bool:
    """True unless config.yaml or the AGENT_CACHE=off environment variable switches the cache off."""
    return CFG.cache.enabled and os.getenv("AGENT_CACHE", "on").lower() not in ("off", "0", "false", "no")


def make_key(*parts: Any) -> str:
    """SHA-256 over the parts (text, bytes or anything JSON can write)."""
    h = hashlib.sha256()
    for p in parts:
        h.update(p if isinstance(p, bytes) else json.dumps(p, sort_keys=True, default=str).encode("utf-8"))
        h.update(b"\x00")                                            # separator: ("ab","c") != ("a","bc")
    return h.hexdigest()


def _path(kind: str, key: str) -> Path:
    base = Path(os.getenv("AGENT_CACHE_DIR") or CFG.cache.dir)        # AGENT_CACHE_DIR e.g. /home/data/cache on Azure
    base = base if base.is_absolute() else ROOT / base
    return base / kind / key[:2] / f"{key}.json"


def get(kind: str, key: str) -> Optional[Any]:
    """The cached value, or None if there is none (or the cache is off / the file is damaged)."""
    if not enabled():
        return None
    try:
        return json.loads(_path(kind, key).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def put(kind: str, key: str, value: Any) -> None:
    """Save a value. Written to a temp file first, so a half-written file is never read."""
    if not enabled():
        return
    p = _path(kind, key)
    p.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=p.parent, suffix=".tmp")
    with os.fdopen(fd, "w", encoding="utf-8") as f:
        json.dump(value, f)
    os.replace(tmp, p)