from __future__ import annotations
import httpx
from bs4 import BeautifulSoup, Tag

BOILERPLATE_TAGS = {"nav", "footer", "header", "script", "style", "aside"}
FETCH_TIMEOUT = 10.0


def fetch_content(input_type: str, input_value: str) -> str:
    """Return raw HTML string. Raises ValueError on fetch failure."""
    if input_type == "url":
        try:
            headers = {"User-Agent": "Mozilla/5.0 (compatible; AEGIS-bot/1.0)"}
            resp = httpx.get(input_value, timeout=FETCH_TIMEOUT, follow_redirects=True, headers=headers)
            resp.raise_for_status()
            return resp.text
        except httpx.TimeoutException:
            raise ValueError(f"Connection timeout after {int(FETCH_TIMEOUT)}s")
        except httpx.HTTPStatusError as e:
            raise ValueError(f"HTTP {e.response.status_code} from {input_value}")
        except httpx.RequestError as e:
            raise ValueError(str(e))
    return input_value  # input_type == "text"


def strip_boilerplate(soup: BeautifulSoup) -> BeautifulSoup:
    """Remove nav/footer/header/script/style tags in-place and return soup."""
    for tag in soup.find_all(BOILERPLATE_TAGS):
        tag.decompose()
    return soup


def get_plain_text(html: str) -> str:
    """Return plain text with boilerplate removed."""
    soup = BeautifulSoup(html, "html.parser")
    strip_boilerplate(soup)
    return soup.get_text(separator=" ", strip=True)


def extract_first_paragraph(html: str) -> str:
    """
    Return the text of the first meaningful <p> tag.
    Falls back to the first double-newline-delimited paragraph in plain text.
    """
    soup = BeautifulSoup(html, "html.parser")
    strip_boilerplate(soup)

    for p in soup.find_all("p"):
        text = p.get_text(strip=True)
        if len(text.split()) >= 5:  # skip trivially short tags
            return text

    # Fallback: plain-text paragraph split
    plain = soup.get_text(separator="\n", strip=True)
    for block in plain.split("\n\n"):
        block = block.strip()
        if len(block.split()) >= 5:
            return block

    return ""
