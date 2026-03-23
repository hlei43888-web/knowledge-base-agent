"""Document parsing service: PDF, Word, and URL."""

import logging
from io import BytesIO

import httpx
import requests
from bs4 import BeautifulSoup
from PyPDF2 import PdfReader
from docx import Document

from app.config import REQUEST_TIMEOUT, MAX_RETRIES

logger = logging.getLogger(__name__)

_MIN_CONTENT_LENGTH = 50

_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                  "(KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9,zh-CN;q=0.8,zh;q=0.7",
}


def parse_pdf(file_bytes: bytes, filename: str = "unknown.pdf") -> str:
    """Extract text from PDF bytes."""
    reader = PdfReader(BytesIO(file_bytes))
    pages: list[str] = []
    for page in reader.pages:
        text = page.extract_text()
        if text:
            pages.append(text.strip())
    full_text = "\n\n".join(pages)
    logger.info(f"Parsed PDF '{filename}': {len(reader.pages)} pages, {len(full_text)} chars")
    return full_text


def parse_docx(file_bytes: bytes, filename: str = "unknown.docx") -> str:
    """Extract text from Word (.docx) bytes."""
    doc = Document(BytesIO(file_bytes))
    paragraphs = [p.text.strip() for p in doc.paragraphs if p.text.strip()]
    full_text = "\n\n".join(paragraphs)
    logger.info(f"Parsed DOCX '{filename}': {len(paragraphs)} paragraphs, {len(full_text)} chars")
    return full_text


def _extract_text_from_html(html: str) -> str:
    """Extract clean text from raw HTML."""
    soup = BeautifulSoup(html, "html.parser")

    for tag in soup(["script", "style", "nav", "footer", "header", "aside",
                      "noscript", "svg", "link", "meta"]):
        tag.decompose()

    main = soup.find("main") or soup.find("article")
    if main is None:
        main = soup.find("body") or soup

    text = main.get_text(separator="\n", strip=True)
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    clean_lines = [l for l in lines if _is_meaningful_line(l)]
    return "\n\n".join(clean_lines)


def _is_meaningful_line(line: str) -> bool:
    """Filter out lines that are CSS/JS code, keep all natural language content."""
    # If line contains any CJK character, it's content — always keep
    if any('\u4e00' <= ch <= '\u9fff' for ch in line):
        return True
    # Only filter lines that look like pure CSS/JS code
    line_lower = line.lower().strip()
    code_patterns = ["function(", "function (", "var ", "const ", "let ",
                     "display:", "margin:", "padding:", "background:",
                     "border:", "position:", "document.", "window.",
                     "=>", "===", "!==", "&&", "||"]
    # Line must start with or be dominated by code patterns
    if any(line_lower.startswith(p) for p in code_patterns):
        return False
    # Pure punctuation / single char — skip
    stripped = line.strip(".,;:!?|/\\-_=+()[]{}#@&* \t\"'")
    if len(stripped) == 0:
        return False
    return True


async def fetch_url(url: str) -> str:
    """Fetch URL content. Tries async httpx first, falls back to sync requests."""
    # Step 1: Try async httpx
    try:
        clean_text = await _fetch_httpx(url)
        if len(clean_text) >= _MIN_CONTENT_LENGTH:
            logger.info(f"Fetched URL (httpx) '{url}': {len(clean_text)} chars")
            return clean_text
        reason = f"only {len(clean_text)} chars"
    except Exception as e:
        reason = str(e)

    # Step 2: Fallback to sync requests (different session, handles some 403 cases)
    logger.info(f"httpx failed ({reason}), falling back to requests for '{url}'")
    clean_text = _fetch_requests(url)
    logger.info(f"Fetched URL (requests) '{url}': {len(clean_text)} chars")
    return clean_text


async def _fetch_httpx(url: str) -> str:
    """Fetch with httpx async."""
    async with httpx.AsyncClient(timeout=REQUEST_TIMEOUT, follow_redirects=True) as client:
        response = await client.get(url, headers=_HEADERS)
        response.raise_for_status()
    return _extract_text_from_html(response.text)


def _fetch_requests(url: str) -> str:
    """Fetch with requests (sync fallback). Uses session with retries."""
    session = requests.Session()
    adapter = requests.adapters.HTTPAdapter(max_retries=MAX_RETRIES)
    session.mount("http://", adapter)
    session.mount("https://", adapter)

    response = session.get(url, headers=_HEADERS, timeout=REQUEST_TIMEOUT, allow_redirects=True)
    response.raise_for_status()
    response.encoding = response.apparent_encoding
    return _extract_text_from_html(response.text)


def parse_file(file_bytes: bytes, filename: str) -> str:
    """Parse file based on extension."""
    lower = filename.lower()
    if lower.endswith(".pdf"):
        return parse_pdf(file_bytes, filename)
    elif lower.endswith(".docx"):
        return parse_docx(file_bytes, filename)
    else:
        raise ValueError(f"Unsupported file type: {filename}. Supported: .pdf, .docx")
