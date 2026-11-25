"""
Azure Document Intelligence (prebuilt-read) - OCR for scanned pages and photos.

Used by text_extract.py when config.yaml has  ocr: provider: azure_di
Endpoint and key come from .env: DOCINTEL_ENDPOINT, DOCINTEL_KEY.
Returns the same shape as the Tesseract path: (text with line breaks, [(word, confidence 0-1)]).
"""
import io
import os
import threading
import time

import requests

from . import cache
from .config import CFG
from .constants import DOCINTEL_API_VERSION, DOCINTEL_MODEL

RETRY_STATUS = (429, 500, 502, 503, 504)        # busy / temporary errors -> wait and try again


# ---------------- usage: pages, time and cost per case (per thread) ----------------
_local = threading.local()


def _usage() -> dict:
    """This thread's running totals since the last usage_reset()."""
    if getattr(_local, "u", None) is None:
        _local.u = {"ocr_pages": 0, "ocr_cache_hits": 0, "ocr_seconds": 0.0}
    return _local.u


def usage_reset() -> None:
    """Start counting from zero (call before each case)."""
    _local.u = None


def usage_take() -> dict:
    """Pages read, seconds and cost (config.yaml ocr.price_per_1000_pages) since the last reset - then reset."""
    u = dict(_usage())
    usage_reset()
    u["ocr_seconds"] = round(u["ocr_seconds"], 2)
    price = CFG.ocr.price_per_1000_pages
    u["ocr_cost_usd"] = round(u["ocr_pages"] / 1000 * price, 6) if price else None
    return u


class DocIntelError(Exception):
    """Document Intelligence could not read the page (no key, still busy, or analysis failed)."""


def _settings() -> tuple[str, str]:
    """Endpoint and key from .env; stop with a clear message if either is missing."""
    endpoint, key = os.getenv("DOCINTEL_ENDPOINT", "").rstrip("/"), os.getenv("DOCINTEL_KEY", "")
    if not endpoint or not key:
        raise DocIntelError("DOCINTEL_ENDPOINT / DOCINTEL_KEY are not set in .env")
    return endpoint, key


def _wait(attempt: int, retry_after=None) -> float:
    """Seconds to wait before the next try: the server's retry-after if it sent one, else 1, 2, 4, 8 ... (max 60)."""
    try:
        wait = float(retry_after) if retry_after is not None else 2 ** attempt
    except ValueError:
        wait = 2 ** attempt
    return min(max(wait, 1), 60)


def _request(method: str, url: str, key: str, headers: dict | None = None, **kw) -> requests.Response:
    """One HTTP call; waits and retries when the service is busy (429 / 5xx), does not answer in time,
    or the connection drops."""
    hdrs = {"Ocp-Apim-Subscription-Key": key, **(headers or {})}   # built once: every retry sends the same headers
    last = ""
    ATTEMPTS = CFG.ocr.max_attempts
    for attempt in range(ATTEMPTS):
        try:
            r = requests.request(method, url, headers=hdrs, timeout=CFG.ocr.timeout_seconds, **kw)
        except (requests.Timeout, requests.ConnectionError) as e:   # no answer / network dropped -> try again
            last = type(e).__name__
            wait = _wait(attempt)
            if attempt < ATTEMPTS - 1:                      # last try failed -> give up below
                print(f"   Document Intelligence {last}, trying again in {wait:.0f}s...")
                time.sleep(wait)
            continue
        if r.status_code in RETRY_STATUS:
            last = f"busy ({r.status_code})"
            wait = _wait(attempt, r.headers.get("retry-after"))
            if attempt < ATTEMPTS - 1:                      # last try failed -> give up below
                print(f"   Document Intelligence {last}, trying again in {wait:.0f}s...")
                time.sleep(wait)
            continue
        return r
    raise DocIntelError(f"Document Intelligence still failing after {CFG.ocr.max_attempts} tries ({last})")


def analyze_image(img) -> tuple[str, list[tuple[str, float]]]:
    """OCR one page image with Document Intelligence -> (text with line breaks, [(word, confidence 0-1)]).
    The same page image read before (same OCR model) comes from the cache: no call, no page billed."""
    buf = io.BytesIO()
    img.convert("RGB").save(buf, format="JPEG", quality=90)        # small upload (free tier max 4 MB)
    data = buf.getvalue()
    ckey = cache.make_key(DOCINTEL_MODEL, DOCINTEL_API_VERSION, data)
    hit = cache.get("ocr", ckey)
    if hit is not None:
        _usage()["ocr_cache_hits"] += 1
        return hit["text"], [tuple(w) for w in hit["words"]]
    t0 = time.perf_counter()
    try:
        text, words = _analyze(data)
    finally:
        u = _usage()
        u["ocr_pages"] += 1                                          # one image = one billed page
        u["ocr_seconds"] += time.perf_counter() - t0
    cache.put("ocr", ckey, {"text": text, "words": words})
    return text, words


def _analyze(data: bytes) -> tuple[str, list[tuple[str, float]]]:
    """The Document Intelligence calls for one page: upload, wait for the result, read lines and words."""
    endpoint, key = _settings()
    url = (f"{endpoint}/documentintelligence/documentModels/{DOCINTEL_MODEL}:analyze"
           f"?api-version={DOCINTEL_API_VERSION}")
    r = _request("POST", url, key, data=data, headers={"Content-Type": "image/jpeg"})
    if r.status_code != 202:
        raise DocIntelError(f"analyze {r.status_code}: {r.text[:300]}")
    result_url = r.headers["Operation-Location"]

    deadline = time.time() + CFG.ocr.timeout_seconds                # the result is ready a few seconds later
    while True:
        time.sleep(CFG.ocr.poll_seconds)
        res = _request("GET", result_url, key).json()
        status = res.get("status")
        if status == "succeeded":
            break
        if status == "failed" or time.time() > deadline:
            raise DocIntelError(f"analysis {status or 'timed out'}: {str(res.get('error', ''))[:300]}")

    lines, words = [], []
    for page in res.get("analyzeResult", {}).get("pages", []):
        lines += [ln["content"] for ln in page.get("lines", [])]
        words += [(w["content"], float(w.get("confidence", 0.0))) for w in page.get("words", [])]
    return "\n".join(lines), words