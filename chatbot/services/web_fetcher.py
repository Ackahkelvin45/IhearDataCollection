"""
Shared web fetching utilities — reusable across chatbot tools, management commands,
and the data_insights agent workflow.

Handles: HTTP fetching, HTML → text extraction, truncation, and error recovery.
"""

import logging
import re
from typing import Optional, Tuple
from urllib.parse import urljoin, urlparse

import requests
import urllib3
from bs4 import BeautifulSoup

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

logger = logging.getLogger(__name__)

DEFAULT_TIMEOUT = 15
DEFAULT_MAX_CHARS = 8000
DEFAULT_USER_AGENT = "Mozilla/5.0 (compatible; IhearBot/1.0)"

HEADERS = {"User-Agent": DEFAULT_USER_AGENT}

# Tags to strip before extracting text
STRIP_TAGS = ["script", "style", "nav", "footer", "header", "noscript", "iframe"]


def clean_text(html: str) -> str:
    """Extract readable text from HTML, stripping noise tags. Public alias for use outside this module."""
    soup = BeautifulSoup(html, "html.parser")

    for tag in soup(STRIP_TAGS):
        tag.decompose()

    text = soup.get_text(separator="\n")
    # Collapse 3+ newlines to 2
    text = re.sub(r"\n{3,}", "\n\n", text)
    # Collapse 3+ spaces
    text = re.sub(r"[ \t]{3,}", "  ", text)
    return text.strip()


# Backwards-compatible name used internally
_clean_text = clean_text


def fetch_page_html(
    url: str,
    timeout: int = DEFAULT_TIMEOUT,
) -> str | None:
    """
    Fetch a URL and return the raw HTML content. For crawlers that need to
    extract links before cleaning text. Returns None on failure.
    """
    try:
        r = requests.get(url, headers=HEADERS, timeout=timeout, verify=False)
        r.raise_for_status()
        return r.text
    except Exception as e:
        logger.warning("Failed to fetch HTML from %s: %s", url, e)
        return None


def fetch_page_text(
    url: str,
    timeout: int = DEFAULT_TIMEOUT,
    max_chars: int = DEFAULT_MAX_CHARS,
) -> Tuple[Optional[str], Optional[str]]:
    """
    Fetch a URL and return extracted text.

    Returns:
        (text, error) — exactly one is None.
        text is truncated to max_chars if longer.
    """
    try:
        r = requests.get(
            url,
            headers=HEADERS,
            timeout=timeout,
            verify=False,
        )
        r.raise_for_status()

        content_type = r.headers.get("Content-Type", "").lower()
        if "text/html" not in content_type:
            return None, f"URL returned non-HTML content type: {content_type}"

        text = _clean_text(r.text)
        if not text or len(text) < 50:
            return None, "Page contains too little readable text"

        if len(text) > max_chars:
            text = text[:max_chars] + "\n\n[Content truncated]"

        return text, None

    except requests.exceptions.Timeout:
        return None, f"Request timed out after {timeout}s"
    except requests.exceptions.ConnectionError:
        return None, f"Could not connect to {url}"
    except requests.exceptions.HTTPError as e:
        return None, f"HTTP error: {e.response.status_code}"
    except Exception as e:
        logger.warning(f"Failed to fetch {url}: {e}")
        return None, f"Failed to fetch page: {e}"


def extract_links(
    html: str, current_url: str, same_domain_only: bool = True
) -> set[str]:
    """Extract all internal links from an HTML page."""
    soup = BeautifulSoup(html, "html.parser")
    links = set()

    base_domain = urlparse(current_url).netloc

    for a in soup.find_all("a", href=True):
        href = a["href"].strip()
        if (
            href.startswith("#")
            or href.startswith("mailto:")
            or href.startswith("javascript:")
        ):
            continue

        full_url = urljoin(current_url, href)
        full_url = full_url.split("#")[0]

        if same_domain_only and urlparse(full_url).netloc != base_domain:
            continue

        links.add(full_url)

    return links


def is_internal_url(url: str, base_url: str) -> bool:
    """Check if url is on the same domain as base_url."""
    return urlparse(url).netloc == urlparse(base_url).netloc
