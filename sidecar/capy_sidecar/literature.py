"""Shared curated bibliography; unresolved references remain labelled searches."""
from functools import lru_cache
import json
import re
import unicodedata
from urllib.parse import quote, urlsplit
from .settings import DOCS_DIR


def normal(text: str) -> str:
    return ''.join(c for c in unicodedata.normalize('NFD', text) if not unicodedata.combining(c)).lower()


@lru_cache(maxsize=1)
def records() -> list[dict]:
    return json.loads((DOCS_DIR / 'explain' / 'literature.json').read_text(encoding='utf-8'))


def safe_url(url: str) -> bool:
    try:
        parsed = urlsplit(url)
        return parsed.scheme in ('https', 'http') and bool(parsed.netloc) and not parsed.username and not parsed.password
    except ValueError:
        return False


def resolve(reference: str) -> dict:
    text = normal(reference)
    matches = [r for r in records() if
               (r.get('match') and all(part in text for part in r['match']))
               or any(normal(alias) == text for alias in r.get('aliases', []))]
    if len(matches) == 1:
        row = matches[0]
        locator = re.search(r'(?:ch(?:apter)?\.?\s*\d+|part\s+[IVX]+)', reference, re.I)
        citation = row['citation'] + (' ' + locator[0] + '.' if locator else '')
        return {'id': row['id'], 'citation': citation, 'url': row['url'], 'resolved': True}
    explicit = re.search(r'https?://[^\s<>]+', reference)
    if explicit and safe_url(explicit[0].rstrip('.,;')):
        return {'id': reference, 'citation': reference, 'url': explicit[0].rstrip('.,;'), 'resolved': True}
    return {'id': reference, 'citation': reference,
            'url': 'https://scholar.google.com/scholar?q=' + quote(reference), 'resolved': False}
