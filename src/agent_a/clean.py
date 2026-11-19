"""Clean email text before it goes to the model.

The Salesforce records keep the email exactly as received. This only tidies
the copy we send to the LLM, so the real message isn't buried in noise:
  - removes the "External Sender / Report phish" security banner (Proofpoint)
  - unwraps urldefense.com links back to the real URL, then shortens long URLs
  - shortens long system codes (e.g. SAP Ariba AN-ORD-EID#prod#... strings)
  - squeezes blank lines and spaces
"""
import re

_BANNER = re.compile(r"ZjQcmQRYFpfptBannerStart.*?ZjQcmQRYFpfptBannerEnd", re.S)
_URLDEFENSE = re.compile(r"https?://urldefense\.com/v3/__(.+?)__;[^\s<>\"]*")
_URL = re.compile(r"https?://[^\s<>\"\]]+")
_LONG_CODE = re.compile(r"\S{60,}")             # base64 blobs, tracking ids
_MAX_URL = 60


def _short_url(m: re.Match) -> str:
    url = m.group(0)
    if len(url) <= _MAX_URL:
        return url
    host = re.match(r"https?://[^/]+", url)
    return (host.group(0) if host else url[:_MAX_URL]) + "/..."


def _short_code(m: re.Match) -> str:
    s = m.group(0)
    return s if s.startswith("http") else s[:25] + "...[code]"


def clean_email_text(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n").replace("‍", "").replace("​", "")
    text = _BANNER.sub("", text)
    text = _URLDEFENSE.sub(lambda m: m.group(1), text)
    text = _URL.sub(_short_url, text)
    text = _LONG_CODE.sub(_short_code, text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r" *\n *", "\n", text)
    return re.sub(r"\n{3,}", "\n\n", text).strip()