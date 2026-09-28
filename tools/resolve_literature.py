"""Resolve long catalogue references against Crossref's publisher metadata.

Explicit maintenance command; never invoked by the app. Only accepts matching
author, publication year and near-exact titles. Ambiguous matches stay unresolved.
"""
import concurrent.futures
import difflib
import json
import re
import time
import unicodedata
import urllib.parse
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEST = ROOT / 'docs/explain/literature.json'


def normal(text):
    text = ''.join(c for c in unicodedata.normalize('NFKD', text) if not unicodedata.combining(c))
    return re.sub(r'[^a-z0-9 ]', '', text.lower()).strip()


def resolve(ref):
    match = re.match(r'(.+?)\s*\((\d{4})\),\s*(.+)', ref)
    if not match:
        return None
    authors, year, title = match.groups()
    if len(title.split()) < 3 or re.search(r'\b(ch\.|chapter|part|ed\.)', title):
        return None
    query = urllib.parse.urlencode({'query.bibliographic': ref, 'rows': 3})
    time.sleep(0.5)
    try:
        req = urllib.request.Request('https://api.crossref.org/works?' + query,
                                     headers={'User-Agent': 'CausalCapybaraLiteratureAudit/1.0'})
        with urllib.request.urlopen(req, timeout=25) as response:
            items = json.load(response)['message']['items']
    except Exception as error:
        print(f'Metadata unavailable: {type(error).__name__}: {error}', flush=True)
        return None
    candidates = []
    first_author = normal(re.split(r',| & | and | et al', authors)[0])
    for item in items:
        years = [str(item.get(field, {}).get('date-parts', [[None]])[0][0])
                 for field in ['published', 'published-print', 'published-online', 'issued']]
        item_title = item.get('title', [''])[0]
        ratio = difflib.SequenceMatcher(None, normal(title), normal(item_title)).ratio()
        surnames = [normal(a.get('family', '')) for a in item.get('author', [])]
        prefix = len(normal(title)) >= 35 and normal(item_title).startswith(normal(title))
        if year in years and first_author in surnames and (ratio >= .9 or prefix) and item.get('DOI'):
            candidates.append(item)
    if len(candidates) != 1:
        return None
    item = candidates[0]
    names = []
    for a in item.get('author', []):
        initials = ' '.join(word[0] + '.' for word in a.get('given', '').split() if word)
        names.append(f"{a.get('family', '')}, {initials}".strip(', '))
    citation = '; '.join(names) + f" ({year}). {item['title'][0]}."
    journal = item.get('container-title', [''])[0]
    if journal:
        citation += ' ' + journal
        if item.get('volume'): citation += ', ' + item['volume']
        if item.get('issue'): citation += '(' + item['issue'] + ')'
        if item.get('page'): citation += ', ' + item['page']
        citation += '.'
    return {'id': item['DOI'], 'match': [], 'aliases': [ref, citation], 'citation': citation,
            'url': 'https://doi.org/' + item['DOI'], 'topic': 'Additional methodology',
            'metadata_source': 'https://api.crossref.org/works/' + item['DOI']}


def main():
    records = json.loads(DEST.read_text(encoding='utf-8'))
    catalogue = json.loads((ROOT / 'app/src/generated/catalogue.json').read_text(encoding='utf-8'))
    refs = sorted({ref for article in catalogue['articles'].values() for ref in article.get('references', [])})
    def known(ref):
        return any((row.get('match') and all(normal(part) in normal(ref) for part in row['match']))
                   or ref in row.get('aliases', []) for row in records)
    unknown = [ref for ref in refs if not known(ref)]
    by_url = {row['url']: row for row in records}
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
        for index, row in enumerate(pool.map(resolve, unknown), 1):
            if row:
                if row['url'] in by_url:
                    existing = by_url[row['url']]
                    existing['aliases'] = sorted(set(existing.get('aliases', []) + row['aliases']))
                else:
                    records.append(row); by_url[row['url']] = row
            if index % 25 == 0: print(f'Checked {index}/{len(unknown)}; {len(records)} resolved sources', flush=True)
    DEST.write_text(json.dumps(records, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(f'Saved {len(records)} sources to {DEST}', flush=True)


if __name__ == '__main__':
    main()
