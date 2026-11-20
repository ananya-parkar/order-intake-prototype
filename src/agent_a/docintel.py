"""
Azure Document Intelligence (prebuilt-read) - OCR for scanned pages and photos.

Used by text_extract.py when config.yaml has  ocr: provider: azure_di
Endpoint and key come from .env: DOCINTEL_ENDPOINT, DOCINTEL_KEY.
Returns the same shape as the Tesseract path: (text with line breaks, [(word, confidence 0-1)]).
"""
import io
import os
import time

import requests

from .config import CFG
from .constants import DOCINTEL_API_VERSION, DOCINTEL_MODEL


class DocIntelError(Exception):
    """Document Intelligence could not read the page (no key, still busy, or analysis failed)."""


def _settings() -> tuple[str, str]:
    """Endpoint and key from .env; stop with a clear message if either is missing."""
    endpoint, key = os.getenv("DOCINTEL_ENDPOINT", "").rstrip("/"), os.getenv("DOCINTEL_KEY", "")
    if not endpoint or not key:
        raise DocIntelError("DOCINTEL_ENDPOINT / DOCINTEL_KEY are not set in .env")
    return endpoint, key


def _request(method: str, url: str, key: str, **kw) -> requests.Response:
    """One HTTP call; waits and retries when the service is busy (429 / 5xx)."""
    for attempt in range(CFG.ocr.max_attempts):
        r = requests.request(method, url, headers={"Ocp-Apim-Subscription-Key": key, **kw.pop("headers", {})},
                             timeout=CFG.ocr.timeout_seconds, **kw)
        if r.status_code in (429, 500, 502, 503):
            try:
                wait = float(r.headers.get("retry-after", 2 ** attempt))
            except ValueError:
                wait = 2 ** attempt
            wait = min(max(wait, 1), 60)
            print(f"   Document Intelligence busy ({r.status_code}), waiting {wait:.0f}s and retrying...")
            time.sleep(wait)
            continue
        return r
    raise DocIntelError(f"Document Intelligence still busy after {CFG.ocr.max_attempts} tries")


def analyze_image(img) -> tuple[str, list[tuple[str, float]]]:
    """OCR one page image with Document Intelligence -> (text with line breaks, [(word, confidence 0-1)])."""
    endpoint, key = _settings()
    buf = io.BytesIO()
    img.convert("RGB").save(buf, format="JPEG", quality=90)        # small upload (free tier max 4 MB)
    url = (f"{endpoint}/documentintelligence/documentModels/{DOCINTEL_MODEL}:analyze"
           f"?api-version={DOCINTEL_API_VERSION}")
    r = _request("POST", url, key, data=buf.getvalue(), headers={"Content-Type": "image/jpeg"})
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